# =============================================================================
# text_features.py — 深度語意特徵擴充模組（網頁內容 165 關鍵詞 + BERT Ensemble 掛鉤）
# =============================================================================
# 目的：
#   features.py 產生的 46 維特徵全部來自「URL 字串本身」，速度快、成本低，
#   但無法理解「網頁實際內容在講什麼」。本模組在使用者要求 deep_scan 時：
#
#     1. 用 httpx 非同步抓取網頁 HTML（有 SSRF 防護，見下方）。
#     2. 從 HTML 中取出可讀文字（去除 script/style）。
#     3. 偵測頁面文字中的 165 公告高風險詞（博弈／假投資／假交易所／一般誘因，
#        詞庫與 rules_config.py 共用），回傳 content_signals 供 main.py 小幅加分。
#     4.（選用）把文字丟進 BERT 取得語意向量，並提供與 46 維傳統特徵合併的
#        build_ensemble_vector()，供未來重新訓練 Ensemble 模型使用。
#
# 修正紀錄（v7.0）：
#   - 【SSRF 防護】deep_scan 會替使用者去抓任意網址，舊版可被拿來打內網
#     （http://127.0.0.1:5500/、雲端 metadata http://169.254.169.254/）。現在：
#       * 只允許 http / https；拒絕 localhost、*.localhost、*.local、*.internal 等名稱；
#       * DNS 解析後只要任一位址是 loopback／私有／link-local／保留／多播／
#         CGNAT 等非公開位址就拒絕（含 IPv4-mapped IPv6）；
#       * 連線時把請求「釘選」到剛驗證過的 IP（Host 標頭與 TLS SNI 仍用原網域），
#         避免 DNS rebinding（驗證時是公開 IP、連線時變成內網 IP）；
#       * 不自動跟隨重新導向，改為手動逐跳檢查，最多 MAX_REDIRECTS 次；
#       * 只接受 80／443 與 1024 以上的連接埠；
#       * 回應大小上限 MAX_RESPONSE_BYTES（約 1MB），超過即停止讀取；
#       * 只處理 text/html（application/xhtml+xml）；不保存任何 cookie。
#   - bs4 改為延遲匯入（未安裝時退回簡易的標籤移除），模組載入不再強制依賴 bs4。
#   - 新增 detect_content_signals() 與 analyze_page_semantics() 回傳的 content_signals。
#   - build_ensemble_vector() 改用 len(FEATURE_NAMES) 檢查維度（不再寫死 31）。
#   - print 改為 logging。
# =============================================================================

from __future__ import annotations

import asyncio
import ipaddress
import logging
import os
import re
import socket
from dataclasses import dataclass, field
from http.cookiejar import CookieJar, DefaultCookiePolicy
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urljoin, urlsplit

import httpx

from features import FEATURE_NAMES
from rules_config import CRYPTO_TERMS, GAMBLING_TERMS, INVESTMENT_TERMS, SCAM_WORD_ZH

logger = logging.getLogger("truthmark.text_features")

# -----------------------------------------------------------------------
# 設定
# -----------------------------------------------------------------------

# 是否啟用 BERT 語意推論。預設關閉（torch/transformers 為重量級選用依賴）。
ENABLE_BERT = os.environ.get("TRUTHMARK_ENABLE_BERT", "0") == "1"
MODEL_NAME = os.environ.get("TRUTHMARK_BERT_MODEL", "distilbert-base-multilingual-cased")

MAX_TEXT_CHARS = 4000            # 送進 BERT 的文字長度上限
MAX_SIGNAL_TEXT_CHARS = 20000    # 關鍵詞偵測使用的文字長度上限
MAX_RESPONSE_BYTES = 1_000_000   # 回應大小上限（約 1MB）
MAX_REDIRECTS = 3                # 最多跟隨幾次重新導向（每一跳都重新做 SSRF 檢查）
FETCH_TIMEOUT_SECONDS = 6.0      # 單次請求逾時
DNS_TIMEOUT_SECONDS = 3.0        # DNS 解析逾時
ALLOWED_SCHEMES = ("http", "https")
USER_AGENT = (
    "Mozilla/5.0 (compatible; TruthMarkBot/1.0; "
    "+https://example.com/truthmark-bot)"
)

_BLOCKED_HOSTNAMES = frozenset({"localhost", "localhost.localdomain", "ip6-localhost", "ip6-loopback"})
_BLOCKED_HOST_SUFFIXES = (".localhost", ".local", ".internal", ".intranet", ".lan", ".home.arpa", ".corp")
_HTML_TYPES = ("text/html", "application/xhtml+xml")
_REDIRECT_CODES = frozenset({301, 302, 303, 307, 308})


@dataclass
class PageTextResult:
    """非同步抓取並清理後的網頁文字結果。"""

    ok: bool
    text: str = ""
    status_code: Optional[int] = None
    error: str = ""
    blocked: bool = False          # 被 SSRF 防護拒絕
    final_url: str = ""
    redirects: int = 0
    truncated: bool = False
    content_type: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


class UnsafeURLError(Exception):
    """目標網址不允許抓取（SSRF 防護）。"""


def new_http_client(timeout: float = FETCH_TIMEOUT_SECONDS) -> httpx.AsyncClient:
    """
    建立供 deep_scan 使用的 AsyncClient：不自動跟隨重新導向（由本模組逐跳檢查）、
    不保存 cookie、限制連線數。main.py 的 lifespan 也用這個函式建立共用 client。
    """
    no_cookies = CookieJar(policy=DefaultCookiePolicy(allowed_domains=[]))
    return httpx.AsyncClient(
        follow_redirects=False,
        timeout=timeout,
        headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.1"},
        cookies=no_cookies,
        limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        trust_env=False,
    )


# -----------------------------------------------------------------------
# 1. SSRF 防護
# -----------------------------------------------------------------------

def _is_public_ip(ip: ipaddress._BaseAddress) -> bool:  # type: ignore[name-defined]
    """只有全域可路由（公開）的單播位址才允許連線。"""
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:
            return _is_public_ip(ip.ipv4_mapped)
        if ip.sixtofour is not None and not _is_public_ip(ip.sixtofour):
            return False
        if ip.teredo is not None:
            return False
    if (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast
            or ip.is_reserved or ip.is_unspecified):
        return False
    if isinstance(ip, ipaddress.IPv4Address) and ip in ipaddress.ip_network("100.64.0.0/10"):
        return False  # CGNAT
    return bool(ip.is_global)


def _parse_target(url: str) -> Tuple[str, str, int]:
    """回傳 (scheme, hostname, port)；不合法時丟 UnsafeURLError。"""
    try:
        parts = urlsplit(url)
        scheme = (parts.scheme or "").lower()
        host = (parts.hostname or "").lower().rstrip(".")
        port = parts.port
    except ValueError as exc:
        raise UnsafeURLError(f"網址格式錯誤：{exc}") from exc
    if scheme not in ALLOWED_SCHEMES:
        raise UnsafeURLError("只允許抓取 http／https 網址")
    if not host:
        raise UnsafeURLError("網址缺少主機名稱")
    if parts.username is not None or parts.password is not None:
        raise UnsafeURLError("網址含帳號密碼欄位（@ 偽裝），不予抓取")
    if port is None:
        port = 443 if scheme == "https" else 80
    if port not in (80, 443) and port < 1024:
        raise UnsafeURLError(f"不允許抓取連接埠 {port}")
    if host in _BLOCKED_HOSTNAMES or host.endswith(_BLOCKED_HOST_SUFFIXES):
        raise UnsafeURLError("不允許抓取本機或內部網路主機")
    return scheme, host, port


async def _resolve_public_addresses(host: str, port: int) -> List[str]:
    """
    解析 hostname；只要任何一個解析結果不是公開位址就拒絕（避免一筆公開、一筆內網的混合解析）。
    回傳驗證通過的 IP 字串清單。
    """
    try:
        literal = ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        literal = None
    if literal is not None:
        if not _is_public_ip(literal):
            raise UnsafeURLError("不允許抓取私有、本機或保留 IP 位址")
        return [str(literal)]

    loop = asyncio.get_running_loop()
    try:
        infos = await asyncio.wait_for(
            loop.getaddrinfo(host, port, type=socket.SOCK_STREAM), timeout=DNS_TIMEOUT_SECONDS
        )
    except asyncio.TimeoutError as exc:
        raise UnsafeURLError("DNS 解析逾時") from exc
    except (socket.gaierror, OSError, UnicodeError) as exc:
        raise UnsafeURLError(f"DNS 解析失敗：{exc}") from exc

    addresses: List[str] = []
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0].split("%", 1)[0])
        except (ValueError, IndexError, TypeError):
            raise UnsafeURLError("DNS 解析結果無法辨識") from None
        if not _is_public_ip(ip):
            raise UnsafeURLError("網域解析到私有、本機或保留 IP 位址，不予抓取")
        if str(ip) not in addresses:
            addresses.append(str(ip))
    if not addresses:
        raise UnsafeURLError("DNS 解析沒有可用位址")
    return addresses


async def check_url_safe(url: str) -> Tuple[str, str, int, List[str]]:
    """對外的 SSRF 檢查：回傳 (scheme, host, port, 已驗證 IP 清單)；不安全時丟 UnsafeURLError。"""
    scheme, host, port = _parse_target(url)
    addresses = await _resolve_public_addresses(host, port)
    return scheme, host, port, addresses


def _pinned_request(url: str, host: str, port: int, ip: str) -> Tuple[str, Dict[str, str], Dict[str, Any]]:
    """把請求網址的主機換成已驗證的 IP；Host 標頭與 TLS SNI 維持原網域。"""
    parts = urlsplit(url)
    default_port = 443 if parts.scheme == "https" else 80
    ip_text = f"[{ip}]" if ":" in ip else ip
    netloc = ip_text if port == default_port else f"{ip_text}:{port}"
    pinned = parts._replace(netloc=netloc).geturl()
    host_header = host if port == default_port else f"{host}:{port}"
    extensions: Dict[str, Any] = {"sni_hostname": host} if parts.scheme == "https" else {}
    return pinned, {"Host": host_header}, extensions


# -----------------------------------------------------------------------
# 2. 非同步網頁爬取（逐跳 SSRF 檢查、大小上限、只處理 HTML）
# -----------------------------------------------------------------------

def _charset_of(content_type: str) -> Optional[str]:
    m = re.search(r"charset=([\w.:-]+)", content_type or "", re.I)
    return m.group(1).strip("'\"") if m else None


def _decode_body(raw: bytes, content_type: str) -> str:
    candidates = [c for c in (_charset_of(content_type),) if c]
    head = raw[:2048].decode("ascii", errors="ignore")
    meta = re.search(r"<meta[^>]+charset=[\"']?([\w.:-]+)", head, re.I)
    if meta:
        candidates.append(meta.group(1))
    candidates += ["utf-8", "cp950", "gb18030"]
    for enc in candidates:
        try:
            return raw.decode(enc)
        except (LookupError, UnicodeDecodeError):
            continue
    return raw.decode("utf-8", errors="replace")


def _html_to_text(html: str) -> str:
    """HTML → 可讀文字；bs4 延遲匯入，未安裝時退回簡易標籤移除。"""
    try:
        from bs4 import BeautifulSoup  # 延遲匯入：只有 deep_scan 才需要

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "svg", "iframe", "template"]):
            tag.decompose()
        title = soup.title.get_text(" ", strip=True) if soup.title else ""
        metas = " ".join(
            m.get("content", "") for m in soup.find_all("meta")
            if (m.get("name") or m.get("property") or "").lower() in ("description", "keywords", "og:title", "og:description")
        )
        body = soup.get_text(separator=" ", strip=True)
        raw_text = " ".join(t for t in (title, metas, body) if t)
    except Exception:  # noqa: BLE001 — bs4 未安裝或解析失敗
        stripped = re.sub(r"(?is)<(script|style|noscript|svg|iframe|template)[^>]*>.*?</\1>", " ", html)
        raw_text = re.sub(r"(?s)<[^>]+>", " ", stripped)
        raw_text = re.sub(r"&nbsp;|&#160;", " ", raw_text)
    return re.sub(r"\s+", " ", raw_text).strip()


async def _fetch_once(
    client: httpx.AsyncClient, url: str, host: str, port: int, ip: str
) -> Tuple[int, Dict[str, str], bytes, bool]:
    """送出單一請求（已釘選 IP），回傳 (status, headers, body, truncated)；不跟隨重新導向。"""
    pinned_url, headers, extensions = _pinned_request(url, host, port, ip)
    request = client.build_request("GET", pinned_url, headers=headers, extensions=extensions)
    response = await client.send(request, stream=True, follow_redirects=False)
    try:
        status = response.status_code
        resp_headers = {k.lower(): v for k, v in response.headers.items()}
        if status in _REDIRECT_CODES or status >= 400:
            return status, resp_headers, b"", False
        content_type = resp_headers.get("content-type", "")
        if content_type and not any(t in content_type.lower() for t in _HTML_TYPES):
            return status, resp_headers, b"", False
        # 串流讀取，最多 MAX_RESPONSE_BYTES；超過即截斷並停止讀取（不信任 Content-Length）
        chunks: List[bytes] = []
        size = 0
        truncated = False
        async for chunk in response.aiter_bytes():
            remaining = MAX_RESPONSE_BYTES - size
            if len(chunk) > remaining:
                chunks.append(chunk[:remaining])
                truncated = True
                break
            chunks.append(chunk)
            size += len(chunk)
        return status, resp_headers, b"".join(chunks), truncated
    finally:
        await response.aclose()


async def fetch_page_text(
    url: str,
    client: Optional[httpx.AsyncClient] = None,
) -> PageTextResult:
    """
    非同步抓取網頁並回傳清理過的可讀文字（含 SSRF 防護）。

    任何網路錯誤、逾時、非 2xx 狀態碼、被 SSRF 防護拒絕，都不會拋例外中斷呼叫端，
    而是回傳 PageTextResult(ok=False, error=...)。
    """
    owns_client = client is None
    active_client = client or new_http_client()
    current = url
    redirects = 0

    try:
        while True:
            try:
                scheme, host, port, addresses = await check_url_safe(current)
            except UnsafeURLError as exc:
                return PageTextResult(ok=False, error=str(exc), blocked=True, final_url=current, redirects=redirects)

            status, headers, body, truncated = await asyncio.wait_for(
                _fetch_once(active_client, current, host, port, addresses[0]),
                timeout=FETCH_TIMEOUT_SECONDS + 2,
            )

            if status in _REDIRECT_CODES:
                location = headers.get("location", "")
                if not location:
                    return PageTextResult(ok=False, status_code=status, error="重新導向缺少 Location", final_url=current)
                redirects += 1
                if redirects > MAX_REDIRECTS:
                    return PageTextResult(ok=False, status_code=status, error=f"重新導向次數超過上限（{MAX_REDIRECTS} 次）",
                                          final_url=current, redirects=redirects)
                current = urljoin(current, location)
                continue

            if status >= 400:
                return PageTextResult(ok=False, status_code=status, error=f"HTTP {status}",
                                      final_url=current, redirects=redirects)

            content_type = headers.get("content-type", "")
            if not body:
                if content_type and not any(t in content_type.lower() for t in _HTML_TYPES):
                    return PageTextResult(ok=False, status_code=status, error=f"非 HTML 內容（{content_type[:60]}）",
                                          final_url=current, redirects=redirects, content_type=content_type)
                if truncated:
                    return PageTextResult(ok=False, status_code=status, error="回應內容超過大小上限",
                                          final_url=current, redirects=redirects, truncated=True)
                return PageTextResult(ok=True, text="", status_code=status, final_url=current,
                                      redirects=redirects, content_type=content_type)

            if not content_type:
                sniff = body[:1024].lstrip().lower()
                if not (sniff.startswith(b"<!doctype html") or b"<html" in sniff):
                    return PageTextResult(ok=False, status_code=status, error="無法確認為 HTML 內容",
                                          final_url=current, redirects=redirects)

            text = _html_to_text(_decode_body(body, content_type))[:MAX_SIGNAL_TEXT_CHARS]
            return PageTextResult(ok=True, text=text, status_code=status, final_url=current,
                                  redirects=redirects, truncated=truncated, content_type=content_type)

    except (asyncio.TimeoutError, httpx.TimeoutException):
        return PageTextResult(ok=False, error="抓取網頁逾時", final_url=current, redirects=redirects)
    except httpx.HTTPError as exc:
        return PageTextResult(ok=False, error=f"抓取網頁失敗：{type(exc).__name__}", final_url=current, redirects=redirects)
    except Exception as exc:  # noqa: BLE001 — 任何解析錯誤都不應讓 /predict 掛掉
        logger.warning("deep_scan 解析失敗：%s", exc)
        return PageTextResult(ok=False, error=f"解析網頁內容失敗：{type(exc).__name__}", final_url=current, redirects=redirects)
    finally:
        if owns_client:
            await active_client.aclose()


# -----------------------------------------------------------------------
# 3. 頁面文字的 165 高風險詞偵測（詞庫來自 rules_config，與網址特徵共用）
# -----------------------------------------------------------------------

def _english_terms(tokens: Sequence[str]) -> frozenset:
    # 頁面文字的英文詞只取 ≥4 字的專一詞（ex、dex、otc、ipo 這類短詞在一般文章太常見）
    return frozenset(t for t in tokens if len(t) >= 4 and not t.isdigit())


_CONTENT_LEXICON: Dict[str, Dict[str, Any]] = {
    "gambling": {
        "zh": tuple(GAMBLING_TERMS.get("zh", ())),
        "en": _english_terms(GAMBLING_TERMS.get("strong_tokens", ())),
        "label": "博弈",
    },
    "investment": {
        "zh": tuple(INVESTMENT_TERMS.get("zh", ())),
        "en": _english_terms(INVESTMENT_TERMS.get("strong_tokens", ())),
        "label": "假投資",
    },
    "crypto": {
        "zh": tuple(CRYPTO_TERMS.get("zh", ())),
        "en": _english_terms(CRYPTO_TERMS.get("strong_tokens", ())),
        "label": "假交易所／虛擬貨幣",
    },
    "scam_generic": {
        "zh": tuple(SCAM_WORD_ZH),
        "en": frozenset(),
        "label": "一般詐騙誘因",
    },
}

# 同一類別至少出現幾個「不同」的詞才算有訊號（避免新聞頁偶爾提到一次就加分）
CONTENT_SIGNAL_MIN_TERMS = 2
_EN_TOKEN_RE = re.compile(r"[a-z0-9]+")


def detect_content_signals(text: str) -> Dict[str, Any]:
    """
    偵測頁面文字中的 165 高風險詞。回傳：
      {"categories": [達門檻的類別], "total_terms": n,
       "<類別>": {"label", "count"(不同詞數), "terms"(最多 5 個), "flagged"(bool)}}
    """
    result: Dict[str, Any] = {"categories": [], "total_terms": 0, "min_terms": CONTENT_SIGNAL_MIN_TERMS}
    if not text:
        for name, lex in _CONTENT_LEXICON.items():
            result[name] = {"label": lex["label"], "count": 0, "terms": [], "flagged": False}
        return result

    sample = text[:MAX_SIGNAL_TEXT_CHARS]
    lowered = sample.lower()
    tokens = set(_EN_TOKEN_RE.findall(lowered))
    for name, lex in _CONTENT_LEXICON.items():
        found: List[str] = []
        for term in lex["zh"]:
            if term and term in sample and term not in found:
                found.append(term)
        for term in sorted(lex["en"] & tokens):
            found.append(term)
        flagged = len(found) >= CONTENT_SIGNAL_MIN_TERMS
        result[name] = {"label": lex["label"], "count": len(found), "terms": found[:5], "flagged": flagged}
        result["total_terms"] += len(found)
        if flagged and name != "scam_generic":
            result["categories"].append(name)
    if result["scam_generic"]["flagged"] and not result["categories"]:
        result["categories"].append("scam_generic")
    return result


# -----------------------------------------------------------------------
# 4. BERT 語意向量擷取（延遲載入，未安裝依賴時安全降級）
# -----------------------------------------------------------------------

_tokenizer = None
_bert_model = None
_bert_failed = False


def _lazy_load_bert() -> bool:
    """延遲載入 tokenizer / 模型；失敗一次後不再重試（避免每個請求都重新嘗試載入）。"""
    global _tokenizer, _bert_model, _bert_failed

    if _tokenizer is not None and _bert_model is not None:
        return True
    if _bert_failed:
        return False

    try:
        import torch  # noqa: F401
        from transformers import AutoModel, AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _bert_model = AutoModel.from_pretrained(MODEL_NAME)
        _bert_model.eval()
        return True
    except Exception as exc:  # noqa: BLE001
        _bert_failed = True
        logger.warning("BERT 模型載入失敗，將略過語意向量：%s", exc)
        return False


def get_bert_embedding(text: str) -> Optional[List[float]]:
    """
    將文字轉換成 BERT [CLS] token 的語意向量。
    回傳 None：ENABLE_BERT 未開啟、依賴未安裝／載入失敗、或輸入文字為空。
    """
    if not ENABLE_BERT or not text:
        return None

    if not _lazy_load_bert():
        return None

    try:
        import torch

        with torch.no_grad():
            inputs = _tokenizer(
                text[:MAX_TEXT_CHARS],
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding=True,
            )
            outputs = _bert_model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0)
            return cls_embedding.tolist()
    except Exception as exc:  # noqa: BLE001
        logger.warning("BERT 推論失敗：%s", exc)
        return None


# -----------------------------------------------------------------------
# 5. Ensemble：傳統特徵向量 ⊕ BERT 語意向量
# -----------------------------------------------------------------------

def build_ensemble_vector(
    traditional_features: List[float],
    bert_embedding: Optional[List[float]],
) -> List[float]:
    """
    把「len(FEATURE_NAMES) 維傳統 URL 特徵」與「BERT 語意向量」融合成單一輸入向量。

      - traditional_features 維度必須等於 len(FEATURE_NAMES)（目前 45），否則丟 ValueError，
        避免舊版 31 維向量被默默拼進新模型。
      - bert_embedding 為 None 時直接回傳傳統特徵，與現有 scam_model.pkl 完全相容。
      - 拼接後的加長向量必須搭配「用相同維度重新訓練」的模型使用；建議先用 PCA
        把 BERT 向量降到 16~32 維再拼接。
    """
    expected = len(FEATURE_NAMES)
    if len(traditional_features) != expected:
        raise ValueError(f"傳統特徵維度應為 {expected}，實際為 {len(traditional_features)}")
    if not bert_embedding:
        return list(traditional_features)
    return list(traditional_features) + list(bert_embedding)


async def analyze_page_semantics(
    url: str,
    client: Optional[httpx.AsyncClient] = None,
) -> dict:
    """
    對外的主要入口：抓取網頁文字 → 偵測 165 高風險詞 →（若啟用）BERT 向量 →
    回傳可放進 API 回應的摘要資訊。刻意不回傳完整頁面文字或完整向量。
    """
    page_result = await fetch_page_text(url, client=client)

    if not page_result.ok:
        return {
            "fetched": False,
            "error": page_result.error,
            "blocked": page_result.blocked,
            "status_code": page_result.status_code,
            "redirects": page_result.redirects,
            "bert_enabled": ENABLE_BERT,
            "content_signals": detect_content_signals(""),
        }

    embedding = get_bert_embedding(page_result.text)

    final_host = ""
    try:
        final_host = (urlsplit(page_result.final_url).hostname or "").lower()
    except ValueError:
        final_host = ""

    return {
        "fetched": True,
        "status_code": page_result.status_code,
        "text_length": len(page_result.text),
        "truncated": page_result.truncated,
        "redirects": page_result.redirects,
        "final_host": final_host,
        "bert_enabled": ENABLE_BERT,
        "bert_embedding_dim": len(embedding) if embedding else 0,
        "content_signals": detect_content_signals(page_result.text),
    }
