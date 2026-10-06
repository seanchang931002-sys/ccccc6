# -*- coding: utf-8 -*-
# =============================================================================
# sync_165.py — 165 反詐騙諮詢專線「遭停止解析涉詐網站」自動匯入黑名單
# =============================================================================
# 資料來源（政府資料開放平臺，警政署，政府資料開放授權條款-第1版）：
#   「165反詐騙諮詢專線_遭停止解析涉詐網站」 https://data.gov.tw/dataset/176455
#   欄位：民國年月、網域、網站性質、法律依據、聲請單位（CSV，不定期更新）
#   （舊的「假投資(博弈)網站」資料集已由上述資料集接手；欄位為 網站名稱／網址，
#    本模組也能解析，可用逗號分隔多個網址一併匯入。）
#
# 流程：下載 CSV（https、大小上限、ETag/Last-Modified 條件式請求）→ 解碼（UTF-8／Big5）
#       → 解析欄位 → 安全過濾 → BlocklistStore.import_domains（原子寫入 dynamic_blocklist.json
#       並即時併入記憶體黑名單）。
#
# 安全設計（寧可少匯，也不誤封）：
#   1. 只新增、不刪除：165 清單移除網域不代表已安全，移除請走 DELETE /blocklist/{domain}。
#   2. 可信任網域（白名單、.gov.tw 等）由 BlocklistStore.is_protected 略過。
#   3. 共用主機網域（github.io、pages.dev…）與公共後綴本身一律略過；
#      因為 main.is_blocked_domain 會比對「所有父網域」，一旦收進去整個平台都會被封。
#      （yourname.github.io 這種「子網域」不受影響，照常匯入。）
#   4. 筆數過少（預設 < 50）或無法解析的比例過高（> 50%）視為來源異常，整批中止、不寫入。
#   5. 同時只允許一個同步作業；失敗時不動現有黑名單。
#
# 設定（環境變數，見 .env.example）：
#   TRUTHMARK_165_CSV_URL        資料集 CSV 下載網址（可逗號分隔多個）。未設定＝不啟用自動同步。
#   TRUTHMARK_165_SYNC_HOURS     自動同步間隔（小時，預設 24；0＝只在啟動後同步一次）
#   TRUTHMARK_165_MIN_ROWS       來源最少筆數（預設 50）
#   TRUTHMARK_165_STATE_PATH     同步狀態檔路徑（預設與 dynamic_blocklist.json 同目錄）
#
# 命令列（排程／離線匯入）：
#   python sync_165.py                      # 依環境變數下載並匯入
#   python sync_165.py --file 165.csv       # 匯入本機已下載的 CSV
#   python sync_165.py --url https://... --dry-run
#   注意：命令列是另一個行程，只會更新 dynamic_blocklist.json；執行中的伺服器要重啟，
#   或呼叫 POST /blocklist/sync-165（伺服器內建同步會直接更新記憶體）才會生效。
# =============================================================================

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os
import re
import sys
import threading
import time
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import httpx
from dotenv import load_dotenv
load_dotenv()
from blocklist_store import BlocklistStore, _atomic_write_json, _iter_domain_cells, normalize_domain
from features import PUBLIC_SUFFIX_FALLBACKS

logger = logging.getLogger("truthmark.sync165")

SOURCE_NAME = "165_stop_resolve"
ENV_URL = "TRUTHMARK_165_CSV_URL"
ENV_HOURS = "TRUTHMARK_165_SYNC_HOURS"
ENV_MIN_ROWS = "TRUTHMARK_165_MIN_ROWS"
ENV_STATE = "TRUTHMARK_165_STATE_PATH"

DEFAULT_MIN_ROWS = 50
DEFAULT_INTERVAL_HOURS = 24.0
MAX_DOWNLOAD_BYTES = 64 * 1024 * 1024   # 目前約 1.3 萬列、不到 2 MB；64 MB 只是防呆上限
MAX_INVALID_RATIO = 0.5
USER_AGENT = "TruthMark-165Sync/1.0 (+anti-scam research; contact: project team)"

# 共用主機／可由任何人註冊子網域的平台：整個平台網域不得進黑名單（個別子網域可以）。
SHARED_HOSTS = frozenset({
    "github.io", "githubusercontent.com", "gitlab.io", "pages.dev", "workers.dev", "vercel.app",
    "netlify.app", "herokuapp.com", "onrender.com", "fly.dev", "glitch.me", "repl.co", "replit.app",
    "blogspot.com", "blogspot.tw", "wordpress.com", "wixsite.com", "weebly.com", "carrd.co",
    "notion.site", "myshopify.com", "web.app", "firebaseapp.com", "appspot.com", "azurewebsites.net",
    "cloudfront.net", "amazonaws.com", "s3.amazonaws.com", "ngrok.io", "ngrok-free.app",
    "trycloudflare.com", "duckdns.org", "sites.google.com", "docs.google.com", "line.me",
    "t.me", "bit.ly", "tinyurl.com", "reurl.cc", "lin.ee",
})

_sync_lock = threading.Lock()


# -----------------------------------------------------------------------------
# 設定
# -----------------------------------------------------------------------------

def configured_urls() -> List[str]:
    """TRUTHMARK_165_CSV_URL（逗號／空白／換行分隔，僅接受 http(s)）。"""
    raw = os.environ.get(ENV_URL, "")
    urls = [u.strip() for u in re.split(r"[,\s]+", raw) if u.strip()]
    return [u for u in urls if u.lower().startswith(("https://", "http://"))]


def configured_interval_hours() -> float:
    try:
        return max(0.0, float(os.environ.get(ENV_HOURS, DEFAULT_INTERVAL_HOURS)))
    except ValueError:
        return DEFAULT_INTERVAL_HOURS


def configured_min_rows() -> int:
    try:
        return max(1, int(os.environ.get(ENV_MIN_ROWS, DEFAULT_MIN_ROWS)))
    except ValueError:
        return DEFAULT_MIN_ROWS


def state_path_for(store: BlocklistStore) -> str:
    return os.environ.get(ENV_STATE) or os.path.join(
        os.path.dirname(os.path.abspath(store.dynamic_blocklist_path)), "sync_165_state.json")


def load_state(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


# -----------------------------------------------------------------------------
# 下載
# -----------------------------------------------------------------------------

class SyncError(Exception):
    """可預期的同步失敗（來源不可用、格式異常…）；訊息可直接顯示給管理員。"""


def fetch_csv(url: str, etag: Optional[str] = None, last_modified: Optional[str] = None, *,
              max_bytes: int = MAX_DOWNLOAD_BYTES, timeout: float = 30.0,
              transport: Optional[httpx.BaseTransport] = None) -> Dict[str, Any]:
    """
    下載 CSV。回傳 {status, text, etag, last_modified, bytes}；
    伺服器回 304（內容未變）時 text=None。只允許 http(s)；超過 max_bytes 即中止。
    """
    if not url.lower().startswith(("https://", "http://")):
        raise SyncError(f"不支援的網址（僅限 http/https）：{url[:80]}")
    headers = {"User-Agent": USER_AGENT, "Accept": "text/csv,*/*;q=0.5"}
    if etag:
        headers["If-None-Match"] = etag
    if last_modified:
        headers["If-Modified-Since"] = last_modified
    try:
        with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(timeout), transport=transport) as client:
            with client.stream("GET", url, headers=headers) as resp:
                if resp.status_code == 304:
                    return {"status": 304, "text": None, "etag": etag, "last_modified": last_modified, "bytes": 0}
                if resp.status_code != 200:
                    raise SyncError(f"下載失敗：HTTP {resp.status_code}（{url[:80]}）")
                declared = resp.headers.get("content-length")
                if declared and declared.isdigit() and int(declared) > max_bytes:
                    raise SyncError(f"檔案過大（{int(declared)} bytes > 上限 {max_bytes}），已中止")
                chunks: List[bytes] = []
                total = 0
                for chunk in resp.iter_bytes():
                    total += len(chunk)
                    if total > max_bytes:
                        raise SyncError(f"下載內容超過上限 {max_bytes} bytes，已中止")
                    chunks.append(chunk)
                return {"status": 200, "text": decode_bytes(b"".join(chunks)),
                        "etag": resp.headers.get("etag"), "last_modified": resp.headers.get("last-modified"),
                        "bytes": total}
    except httpx.HTTPError as exc:
        raise SyncError(f"連線失敗：{type(exc).__name__}: {exc}") from exc


def decode_bytes(raw: bytes) -> str:
    """政府 CSV 編碼不一：優先 UTF-8（含 BOM），失敗改 Big5/CP950，最後才容錯解碼。"""
    for enc in ("utf-8-sig", "cp950"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


# -----------------------------------------------------------------------------
# 解析
# -----------------------------------------------------------------------------

_DOMAIN_HEADERS_EXACT = ("網域", "domain", "weburl", "url", "網址")
_DOMAIN_HEADERS_CONTAINS = ("網域", "domain", "url", "網址")


def _clean_cell(value: Any) -> str:
    """去除空白、BOM、萬用字元前綴，並還原常見的「防點擊」寫法（hxxp、[.]）。"""
    text = str(value or "").strip().strip('"\'').lstrip("\ufeff").strip()
    text = text.replace("[.]", ".").replace("(.)", ".")
    text = re.sub(r"^hxxp", "http", text, flags=re.IGNORECASE)
    text = re.sub(r"^\*\.", "", text)
    text = text.lstrip("?")   # 官方資料偶有開頭殘留 "?"（例如 ?shopkings.top）
    return text


def _find_col(header: List[str], exact: Iterable[str], contains: Iterable[str],
              exclude: Iterable[str] = ()) -> Optional[int]:
    """先找「完全相符」的表頭，再找「包含關鍵字」且不含 exclude 字樣的表頭。"""
    for hint in exact:
        for i, h in enumerate(header):
            if h == hint:
                return i
    for hint in contains:
        for i, h in enumerate(header):
            if hint in h and not any(x in h for x in exclude):
                return i
    return None


def extract_records(text: str) -> List[Dict[str, str]]:
    """
    把 CSV 文字轉成 [{domain, category, period, agency}]。
    有表頭（網域／網址／domain／url）時依欄位取值；否則退回掃描所有儲存格。
    """
    head = text[:4096]
    delimiter = max((",", "\t", ";"), key=lambda d: head.count(d)) if head else ","
    rows = list(csv.reader(io.StringIO(text), delimiter=delimiter))
    rows = [r for r in rows if any((c or "").strip() for c in r)]
    if not rows:
        return []

    header = [(h or "").strip().lstrip("\ufeff").strip().lower() for h in rows[0]]
    # 網域欄不可誤選「網站名稱」「網站性質」這類含「網站」字樣的欄位
    dom_i = _find_col(header, _DOMAIN_HEADERS_EXACT, _DOMAIN_HEADERS_CONTAINS, exclude=("名稱", "性質"))
    if dom_i is None:
        return [{"domain": _clean_cell(c), "category": "", "period": "", "agency": ""}
                for c in _iter_domain_cells(text)]

    cat_i = _find_col(header, (), ("性質", "category", "類別"))
    per_i = _find_col(header, ("民國年月",), ("年月", "統計起始", "date", "日期"))
    ag_i = _find_col(header, ("聲請單位",), ("單位",))

    def cell(row: List[str], idx: Optional[int]) -> str:
        return (row[idx].strip() if idx is not None and idx < len(row) else "")

    records = []
    for row in rows[1:]:
        domain = _clean_cell(cell(row, dom_i))
        if domain:
            records.append({"domain": domain, "category": cell(row, cat_i),
                            "period": cell(row, per_i), "agency": cell(row, ag_i)})
    return records


# -----------------------------------------------------------------------------
# 安全過濾
# -----------------------------------------------------------------------------

try:  # 離線 PSL 快照，與 features.py 相同作法；缺套件時只靠 SHARED_HOSTS
    import tldextract as _tldextract
    _EXTRACT = _tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)
except Exception:  # noqa: BLE001
    _EXTRACT = None


def is_shared_or_suffix(domain: str) -> bool:
    """網域本身是共用主機平台（github.io…）或公共後綴（com.tw、co.uk…）時為 True。"""
    if domain in SHARED_HOSTS:
        return True
    if _EXTRACT is not None:
        try:
            ext = _EXTRACT(domain)
            if not ext.domain:      # 整串都是後綴，例如 "com.tw"
                return True
        except Exception:  # noqa: BLE001
            pass
    # 缺少 tldextract 時仍需保護本專題常見的公共後綴，避免將整個 ccTLD／平台網域封鎖。
    return domain in PUBLIC_SUFFIX_FALLBACKS or domain in {
        "co.uk", "com.tw", "co.jp", "co.kr", "com.hk", "com.cn", "com.sg", "com.my",
    }


def _note(rec: Dict[str, str]) -> str:
    # 聲請單位在整份資料幾乎都相同，不寫入備註（86k 筆可省數 MB 檔案大小）
    parts = [rec.get("category", ""), rec.get("period", "")]
    return "｜".join(p for p in parts if p)[:200]


# -----------------------------------------------------------------------------
# 主流程
# -----------------------------------------------------------------------------

def _empty_report(dry_run: bool) -> Dict[str, Any]:
    return {"ok": False, "dry_run": dry_run, "source": SOURCE_NAME, "sources": [], "rows": 0,
            "candidates": 0, "invalid": 0, "skipped_shared": 0, "not_modified": 0,
            "added": 0, "already": 0, "protected": 0, "added_sample": [], "error": None,
            "synced_at": None, "duration_s": 0.0}


def sync_165(
    store: BlocklistStore,
    urls: Optional[List[str]] = None,
    *,
    local_file: Optional[str] = None,
    dry_run: bool = False,
    min_rows: Optional[int] = None,
    state_path: Optional[str] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """
    下載（或讀本機檔）→ 解析 → 過濾 → 匯入 store。永遠回傳 report dict（ok=False 時帶 error），
    不對呼叫端丟 SyncError。force=True 忽略 ETag（強制重新下載）。
    """
    report = _empty_report(dry_run)
    started = time.monotonic()
    if not _sync_lock.acquire(blocking=False):
        report["error"] = "已有另一個 165 同步作業進行中，請稍後再試"
        return report
    try:
        min_rows = min_rows if min_rows is not None else configured_min_rows()
        state_path = state_path or state_path_for(store)
        state = load_state(state_path)
        url_state: Dict[str, Any] = state.get("urls") if isinstance(state.get("urls"), dict) else {}

        sources: List[Tuple[str, Optional[str]]] = []   # (label, text)
        if local_file:
            with open(local_file, "rb") as f:
                sources.append((os.path.basename(local_file), decode_bytes(f.read())))
        else:
            urls = urls if urls is not None else configured_urls()
            if not urls:
                raise SyncError(f"尚未設定 {ENV_URL}（資料集 CSV 下載網址），也未指定 --url / --file")
            for url in urls:
                prev = url_state.get(url, {})
                res = fetch_csv(url, None if force else prev.get("etag"), None if force else prev.get("last_modified"))
                if res["status"] == 304:
                    report["not_modified"] += 1
                    logger.info("165 來源未變更（304）：%s", url)
                    continue
                sources.append((url, res["text"]))
                url_state[url] = {"etag": res.get("etag"), "last_modified": res.get("last_modified")}

        domains: List[str] = []
        notes: Dict[str, str] = {}
        seen = set()
        for label, text in sources:
            records = extract_records(text or "")
            report["rows"] += len(records)
            report["sources"].append({"source": label, "rows": len(records)})
            if len(records) < min_rows:
                raise SyncError(f"來源「{label}」只有 {len(records)} 筆（< {min_rows}），疑似下載不完整或格式改變，"
                                f"已中止、未寫入任何資料")
            invalid = 0
            for rec in records:
                d = normalize_domain(rec["domain"])
                if not d:
                    invalid += 1
                    continue
                if d in seen:
                    continue
                seen.add(d)
                if is_shared_or_suffix(d):
                    report["skipped_shared"] += 1
                    continue
                domains.append(d)
                note = _note(rec)
                if note:
                    notes[d] = note
            if records and invalid / len(records) > MAX_INVALID_RATIO:
                raise SyncError(f"來源「{label}」有 {invalid}/{len(records)} 筆無法解析為網域，疑似欄位改變，已中止")
            report["invalid"] += invalid

        report["candidates"] = len(domains)
        if dry_run:
            fresh = [d for d in domains if d not in store.blocked_domains and not store._protected(d)]
            report["added"] = len(fresh)
            report["already"] = sum(1 for d in domains if d in store.blocked_domains)
            report["protected"] = sum(1 for d in domains if d not in store.blocked_domains and store._protected(d))
            report["added_sample"] = fresh[:5]
        elif domains:
            before = set(store.blocked_domains)
            result = store.import_domains(domains, source=SOURCE_NAME, notes=notes)
            report["added"], report["already"], report["protected"] = result["added"], result["already"], result["protected"]
            report["added_sample"] = [d for d in domains if d not in before and d in store.blocked_domains][:5]

        report["ok"] = True
        report["synced_at"] = datetime.now().isoformat(timespec="seconds")
        if not dry_run:
            state.update({
                "urls": url_state,
                "last_success_at": report["synced_at"],
                "last_error": None,
                "last_result": {k: report[k] for k in ("rows", "candidates", "added", "already", "protected",
                                                       "skipped_shared", "invalid")},
            })
            _save_state(state_path, state)
        logger.info("165 同步完成%s：來源 %d 列、候選 %d、新增 %d、已存在 %d、可信任略過 %d、共用主機略過 %d、無法解析 %d",
                    "（dry-run）" if dry_run else "", report["rows"], report["candidates"], report["added"],
                    report["already"], report["protected"], report["skipped_shared"], report["invalid"])
    except SyncError as exc:
        report["error"] = str(exc)
        logger.warning("165 同步失敗：%s", exc)
        _record_failure(store, state_path, str(exc), dry_run)
    except Exception as exc:  # noqa: BLE001 — 同步失敗絕不能拖垮服務
        report["error"] = f"{type(exc).__name__}: {exc}"
        logger.exception("165 同步發生未預期錯誤")
        _record_failure(store, state_path, report["error"], dry_run)
    finally:
        report["duration_s"] = round(time.monotonic() - started, 2)
        _sync_lock.release()
    return report


def _save_state(path: str, state: Dict[str, Any]) -> None:
    try:
        _atomic_write_json(path, state)
    except Exception as exc:  # noqa: BLE001
        logger.warning("無法寫入同步狀態檔 %s：%s", path, exc)


def _record_failure(store: BlocklistStore, state_path: Optional[str], error: str, dry_run: bool) -> None:
    if dry_run:
        return
    try:
        path = state_path or state_path_for(store)
        state = load_state(path)
        state["last_error"] = {"at": datetime.now().isoformat(timespec="seconds"), "message": error}
        _save_state(path, state)
    except Exception:  # noqa: BLE001
        pass


def status(store: BlocklistStore) -> Dict[str, Any]:
    """供 GET /blocklist/sync-165 使用：設定與上次同步結果。"""
    state = load_state(state_path_for(store))
    return {
        "configured": bool(configured_urls()),
        "urls": configured_urls(),
        "interval_hours": configured_interval_hours(),
        "min_rows": configured_min_rows(),
        "last_success_at": state.get("last_success_at"),
        "last_result": state.get("last_result"),
        "last_error": state.get("last_error"),
    }


# -----------------------------------------------------------------------------
# 命令列
# -----------------------------------------------------------------------------

def _main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="匯入 165 遭停止解析涉詐網站清單到 TruthMark 動態黑名單")
    ap.add_argument("--url", action="append", help=f"CSV 下載網址（可重複；未給則讀環境變數 {ENV_URL}）")
    ap.add_argument("--file", help="改為匯入本機已下載的 CSV")
    ap.add_argument("--dry-run", action="store_true", help="只試算、不寫入")
    ap.add_argument("--force", action="store_true", help="忽略 ETag，強制重新下載")
    ap.add_argument("--min-rows", type=int, help=f"來源最少筆數（預設 {DEFAULT_MIN_ROWS}）")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s：%(message)s")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    store = BlocklistStore(base_dir, set(), threshold=3)
    store.load_dynamic_blocklist()
    report = sync_165(store, args.url, local_file=args.file, dry_run=args.dry_run,
                      min_rows=args.min_rows, force=args.force)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(_main())
