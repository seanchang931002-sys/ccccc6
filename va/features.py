# =============================================================================
# features.py — VeriAd / TruthMark URL 特徵擷取模組 v7.1
# =============================================================================
# 修正紀錄（v7.1，依 CONTRACT §5.1；特徵 45 → 46 維，必須重新執行 train_model.py）
# -----------------------------------------------------------------------------
#   - 新增第 46 個特徵 sld_randomness（0～1）：SLD 的字元 trigram 平均驚奇度（英文＋漢語拼音＋日文
#     羅馬拼音統計，char_ngram_table.json 由 tools/build_ngram_table.py 以離線語料產生），輔以最長
#     連續子音與數字字母交錯；只看 SLD、官方網域也照算。統計表讀不到時退回子音／母音啟發式。
#   - newly_registered_like 改為「sld_randomness ≥ 0.70」且不含品牌、有意義字詞、不在白名單
#     （舊版子音規則誤判 threads／flickr／tumblr／kktix／mcdonalds…）。
#   - dot_count／digit_ratio／special_chars 改為「只算主機名稱」（去 www.，不含 userinfo、埠、路徑、
#     參數）：group-CV 實驗（10 次）真實樣本 PR-AUC 0.9376 → 0.9370（差異在雜訊內），同一網域在
#     原始／形狀翻轉／廣告落地頁三種寫法的機率漂移平均 0.091 → 0.062、P90 0.244 → 0.175。
#     url_length、hyphen_count 仍以整個網址（去 scheme 與 www.）計算。
#   - 黏字切分：博弈／假投資／假交易所詞庫詞與「數字／常見後綴／另一個詞庫詞」黏在同一個 token
#     （biaoguvip、188bettw、kucoinexchangevip、wakuangvip、twlicaiplus）也能命中；品牌後面黏多個
#     後綴（kucoin+exchange+vip）或前面黏後綴詞（netbank+scsb）也算品牌冒用。整個 token 必須能被
#     完整切開，alphabet、betterhelp 這類一般字不受影響。
# -----------------------------------------------------------------------------
# 修正紀錄（v4.1）：brand_in_sld 加入可疑 TLD 條件。
# 新增（v5.0）：levenshtein_brand_dist、newly_registered_like（29 → 31 維）。
# 重構（v6）：URLComponents 集中解析、型別提示、lru_cache、explain 可重用 feature_dict。
#
# 修正紀錄（v7.0，依 CONTRACT §0～2；特徵 31 → 45 維，必須重新執行 train_model.py）
# -----------------------------------------------------------------------------
# 共通修正（根因 §0）：
#   - canonicalize_url()：NFKC、移除全形空白／零寬字元／所有空白；無 scheme 時補
#     「https://」（訓練與推論共用，消除「有 https = 正常」捷徑）；scheme 與 host 小寫、
#     路徑大小寫保留；IDN 中文網域轉 punycode；host 去尾端點。
#   - hostname 統一：小寫、去尾端點、去開頭 www.、不含埠號與 userinfo。
#   - registered domain / SLD / TLD 改用 tldextract 離線 PSL 快照（含 private 後綴，
#     例如 github.io、web.app 會被當成公開後綴），失敗或 TLD 不在 PSL 時退回簡易解析。
#   - 語意／中文比對一律用 percent-decode 後的字串；英文詞一律 token（以非英數切分、
#     去除前後數字）或 token 前後綴比對，不再使用裸子字串（learn ≠ earn、alphabet ≠ bet）。
#   - URL 解析自行實作，非法 IPv6（http://[abc）、奇怪 unicode、超長字串都不丟例外。
# 前 31 個特徵（名稱與順序不變，語意以下列新定義為準）：
#    1 url_length            去掉 scheme 與 www. 後的網址長度（路徑只有 "/" 時不計）
#    2 domain_length         hostname（去 www.）長度
#    3 path_length           path 長度（"/" 視為 0；不含 query 與 #fragment）
#    4 query_length          query 字串長度
#    5 digit_ratio           v7.1：主機名稱（去 www.）中數字字元數 / 主機名稱長度
#    6 hyphen_count          去 scheme 與 www. 後的整個網址中的「-」數量
#    7 dot_count             v7.1：主機名稱（去 www.）中的「.」數量
#    8 special_chars         v7.1：主機名稱（去 www.）中非英數字元數量（點號、連字號等）
#    9 subdomain_depth       PSL 切分後子網域層數（www 不算；sub.momo.com.tw = 1）
#   10 path_depth            path 非空段數
#   11 query_params          query 參數種類數（含空值參數）
#   12 is_https              明確 https 或「沒寫 scheme」= 1；明確 http（或其他協定）= 0
#   13 is_ip_address         hostname 為 IPv4 / IPv6 / 十進位整數 IP
#   14 suspicious_tld        TLD 風險等級 ≥ 1（rules_config.TLD_RISK_TIERS）
#   15 has_scam_word         解碼後網址含一般詐騙誘因詞（英文 token / bigram / 中文）；
#                            新聞、查核、政府、教育與白名單網域的路徑不採計
#   16 brand_in_sld          SLD 的 token 含已知品牌、TLD 風險 ≥ 1、且非該品牌官方網域
#   17 has_utm               query/fragment 含 utm_ 參數
#   18 has_gclid             含 gclid / gbraid / wbraid / msclkid / dclid
#   19 double_http           解碼後網址出現 2 次以上 http(s)://
#   20 long_domain           hostname 長度 > 30
#   21 domain_entropy        SLD 的 Shannon entropy
#   22 has_shortener         registered domain 屬於短網址服務
#   23 gambling_number_pattern  hostname 數字段為 168/888/666/999/777/88/99… 幸運數字，
#                            或 win888／888bet 類組合（純數字 SLD 如 1688.com、8591 不算）
#   24 brand_typo_like       hostname 含品牌「拼字變形」：明確變形詞（faebook、tikmall…）、
#                            同形字（sh0pee）、截斷字（walmar…）、或有效編輯距離 1～2；
#                            且非該品牌官方網域（正牌 poyabuy/cosmed/watsons 不再命中）
#   25 cloud_hosting         hostname 以雲端/CDN/PaaS 基礎設施後綴結尾（後綴比對，不再是子字串）
#   26 mobile_lure_path      hostname 首段為 m/h5/wap/app/sj/mobile，或路徑段含 /h5、/wap、首段 /m
#   27 suspicious_keyword_in_domain  hostname token 含詐騙/博弈/假投資/加密貨幣強詞
#   28 many_subdomains       subdomain_depth ≥ 3
#   29 contains_percent_encoding  網址含 %XX 編碼
#   30 levenshtein_brand_dist  與「非官方」品牌的有效編輯距離：0 = 與品牌同名但非官方網域、
#                            1～2 = 拼字近似、99 = 無相似品牌或本身就是官方網域。
#                            ≤4 字品牌只接受距離 1 且需搭配其他風險訊號（moma/nine/pola 不算）
#   31 newly_registered_like v7.1：sld_randomness ≥ 0.70 且不含品牌、有意義字詞，也不在白名單
#                            （統計表讀不到時退回舊版「母音比例低／罕見子音連接」規則）
# v7 新增 14 個（名稱、順序固定，詞庫與網域清單見 rules_config.py）：
#   32 tld_risk_level  33 gambling_keyword  34 investment_lure_keyword
#   35 crypto_exchange_lure  36 brand_impersonation  37 free_hosting_platform
#   38 tunnel_or_ephemeral_host  39 social_invite_link  40 punycode_domain
#   41 url_has_at_symbol  42 non_standard_port  43 sld_digit_count  44 sld_length
#   45 path_scam_route
# v7.1 新增 1 個：
#   46 sld_randomness（0～1，網域名稱亂碼度；≥0.70 warn、≥0.50 neutral）
# 介面：FEATURE_VERSION、FEATURE_SCHEMA_ID、FEATURE_SPECS、assess_feature()／assess_all()
#   （門檻唯一來源；alert=高風險、warn=需注意、safe=正常、neutral=中性資訊）、
#   explain_features()（訊息由 assess 結果產生）、get_rule_targets()（給硬規則比對）。
# =============================================================================

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Set, Tuple
from urllib.parse import ParseResult, parse_qsl, unquote, urlparse

from rules_config import (
    BRAND_AFFIX_TOKENS,
    BRAND_GENERIC_WORDS,
    BRAND_NAME_LIST,
    BRAND_OFFICIAL_DOMAINS,
    BRAND_PREFIX_TOKENS,
    BRAND_TRUNCATION_TARGETS,
    BRAND_TYPO_TERMS,
    LEVENSHTEIN_COMMON_WORDS,
    brand_hyphen_variants,
    CLOUD_HOSTING_SUFFIXES,
    CONTENT_PLATFORM_DOMAINS,
    CRYPTO_TERMS,
    FREE_HOSTING_HOSTS,
    FREE_HOSTING_REGEXES,
    FREE_HOSTING_SUFFIXES,
    GAMBLING_TERMS,
    INVESTMENT_TERMS,
    IPFS_PATH_PATTERN,
    MEANINGFUL_WORDS,
    MULTI_CC_BRANDS,
    SEGMENT_AFFIX_TOKENS,
    SCAM_WORD_BIGRAMS,
    SCAM_WORD_TOKENS,
    SCAM_WORD_ZH,
    SHORTENER_DOMAINS,
    SOCIAL_INVITE_PATTERNS,
    SUSPICIOUS_TLDS,
    TLD_RISK_TIERS,
    TRUSTED_DOMAINS,
    TRUSTED_SUFFIXES,
    TUNNEL_HOSTS,
    TUNNEL_SUFFIXES,
)

try:  # tldextract 為正式環境建議依賴；不可用時退回有限、可預期的離線 suffix fallback
    import tldextract as _tldextract
except Exception:  # noqa: BLE001
    _tldextract = None

# -----------------------------------------------------------------------
# v7.2：環境依賴驗證 — tldextract 缺失時不再悄悄退回，而是在啟動時大聲警告。
# -----------------------------------------------------------------------
# 背景：requirements.txt 明確要求 tldextract>=5.1，但若執行環境忘了安裝，
# 程式先前只會「靜默」改用簡化的 suffix 解析，語意與正式版不同（例如
# binance-tw-pro.github.io 會被簡易解析誤判 SLD=github、.comn 這類打字錯誤
# 尾碼也可能被簡易解析誤判為合法尾碼），導致特徵值、測試結果與正式環境不一致，
# 卻完全沒有任何跡象可供排查。現在改成：匯入失敗時立即印出醒目的 WARNING，
# 並提供 tldextract_status() 讓 /health 與啟動日誌可以回報「降級」狀態，
# 而不是讓使用者誤以為系統一切正常。
# -----------------------------------------------------------------------
# 正式環境優先使用 tldextract + 內建 PSL；離線／缺少依賴時使用此有限 fallback。
# 這不是完整 Public Suffix List，而是涵蓋本專題常見國別後綴、共用主機與
# 常見多段 suffix，目的在於讓「缺依賴」時的行為可預期，而不是猜測最後一段字串。
PUBLIC_SUFFIX_FALLBACKS = frozenset({
    # Taiwan / Asia 常見
    "com.tw", "net.tw", "org.tw", "gov.tw", "edu.tw", "idv.tw",
    "co.jp", "ne.jp", "or.jp", "co.kr", "co.nz", "com.hk", "com.cn",
    "com.sg", "com.my", "com.au", "co.id", "co.th", "co.in", "co.uk",
    # 其他常見多段 suffix
    "com.br", "com.mx", "com.tr", "com.ua", "co.za", "co.il", "com.ar",
    "com.co", "com.pe", "com.pk", "com.ph", "com.vn", "com.tw",
    # 常見 private / shared hosting suffix
    "github.io", "githubusercontent.com", "gitlab.io", "pages.dev", "workers.dev",
    "vercel.app", "netlify.app", "web.app", "firebaseapp.com", "appspot.com",
    "blogspot.com", "blogspot.tw", "wordpress.com", "wixsite.com", "weebly.com",
    "myshopify.com", "notion.site", "onrender.com", "herokuapp.com", "fly.dev",
    "cloudfront.net", "azurewebsites.net", "s3.amazonaws.com", "amazonaws.com",
})

_logger = logging.getLogger("truthmark.features")

if _tldextract is None:
    _logger.warning(
        "=" * 70 + "\n"
        "[降級警告] 找不到 tldextract 套件（requirements.txt 已列為正式依賴："
        "tldextract>=5.1）。\n"
        "目前將使用有限的離線 suffix fallback，registered domain／SLD 判斷不等同完整 PSL。\n"
        "特徵值、模型分數與測試結果都可能與正式環境不一致。\n"
        "請立即執行：pip install -r requirements.txt\n" + "=" * 70
    )


def tldextract_status() -> Dict[str, Any]:
    """供 /health 與啟動檢查使用：回報 tldextract 是否可用（degraded=True 代表使用有限 fallback）。"""
    return {
        "available": _tldextract is not None,
        "degraded": _tldextract is None,
        "message": (
            "tldextract 已安裝，使用離線 PSL 快照解析網域"
            if _tldextract is not None
            else "tldextract 未安裝，目前使用有限 suffix fallback（非完整 PSL，請安裝 tldextract>=5.1）"
        ),
    }


# =============================================================================
# 特徵名稱（共 46 個；前 31 個名稱與順序與 v5/v6 完全相同，32～45 為 v7.0，46 為 v7.1）
# =============================================================================

FEATURE_NAMES: List[str] = [
    # 長度類
    "url_length",
    "domain_length",
    "path_length",
    "query_length",

    # 字元比例類
    "digit_ratio",
    "hyphen_count",
    "dot_count",
    "special_chars",

    # 結構類
    "subdomain_depth",
    "path_depth",
    "query_params",

    # 協定 / 網域類
    "is_https",
    "is_ip_address",
    "suspicious_tld",

    # 語意類
    "has_scam_word",
    "brand_in_sld",

    # 廣告追蹤參數類
    "has_utm",
    "has_gclid",

    # 混淆手法類
    "double_http",
    "long_domain",

    # 隨機性
    "domain_entropy",

    # 強化特徵
    "has_shortener",
    "gambling_number_pattern",
    "brand_typo_like",
    "cloud_hosting",
    "mobile_lure_path",
    "suspicious_keyword_in_domain",
    "many_subdomains",
    "contains_percent_encoding",

    # v5.0：品牌相似度 & 域名新鮮度
    "levenshtein_brand_dist",
    "newly_registered_like",

    # v7.0：165 新型態詐騙特徵
    "tld_risk_level",
    "gambling_keyword",
    "investment_lure_keyword",
    "crypto_exchange_lure",
    "brand_impersonation",
    "free_hosting_platform",
    "tunnel_or_ephemeral_host",
    "social_invite_link",
    "punycode_domain",
    "url_has_at_symbol",
    "non_standard_port",
    "sld_digit_count",
    "sld_length",
    "path_scam_route",

    # v7.1：網域名稱可讀性（字元 n-gram）
    "sld_randomness",
]

FEATURE_VERSION = "7.1.0"
FEATURE_SCHEMA_ID = hashlib.sha1("|".join(FEATURE_NAMES).encode("utf-8")).hexdigest()[:12]

# 無法解析時的安全預設值（不觸發任何風險）
_NO_BRAND_DISTANCE = 99
FEATURE_DEFAULTS: Dict[str, float] = {name: 0 for name in FEATURE_NAMES}
FEATURE_DEFAULTS["is_https"] = 1
FEATURE_DEFAULTS["levenshtein_brand_dist"] = _NO_BRAND_DISTANCE

# 單一輸入的最大處理長度（防止惡意超長字串拖慢 regex；main 另有 2048 上限）
_MAX_INPUT_LENGTH = 16384


# =============================================================================
# 舊版相容常數（仍可被外部 import；實際資料來自 rules_config）
# =============================================================================

IP_ADDRESS_PATTERN = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")

_GOV_SUFFIX_LABELS = frozenset({"gov", "edu", "mil", "ac", "go", "gob", "gouv", "govt"})
_CCSLD_SECOND_LEVEL = frozenset({
    "com", "net", "org", "gov", "edu", "co", "ac", "mil", "idv", "or", "ne", "go", "gr",
    "game", "ebiz", "club", "biz", "info", "ltd", "plc", "nic", "gob", "gouv", "nom",
})

_DIGITS = "0123456789"
_TOKEN_SPLIT_RE = re.compile(r"[^a-z0-9]+")
_ZERO_WIDTH_RE = re.compile("[­᠎​-‏‪-‮⁠-⁤⁦-⁩﻿]")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_WS_RE = re.compile(r"\s+")
_SCHEME_RE = re.compile(r"^([a-zA-Z][a-zA-Z0-9+.\-]{0,31}):(//)?")
_OPAQUE_SCHEMES = frozenset({
    "javascript", "data", "mailto", "tel", "sms", "about", "blob", "vbscript", "file",
    "intent", "market",
})
_HIER_RE = re.compile(r"^([a-z][a-z0-9+.\-]*)://([^/?#]*)(.*)$", re.S)
_PERCENT_RE = re.compile(r"%[0-9a-fA-F]{2}")
_EMBEDDED_URL_RE = re.compile(r"https?://[^\s\"'<>]+", re.I)

_TIER2 = TLD_RISK_TIERS.get(2, frozenset())
_TIER1 = TLD_RISK_TIERS.get(1, frozenset())

_GAMBLING_COMBO_RE = re.compile(GAMBLING_TERMS["combo_regex"])
_CRYPTO_EX_RE = re.compile(CRYPTO_TERMS["ex_suffix_regex"])
_SOCIAL_INVITE_RES = tuple(re.compile(p, re.I) for p in SOCIAL_INVITE_PATTERNS)
_FREE_HOSTING_RES = tuple(re.compile(p, re.I) for p in FREE_HOSTING_REGEXES)
_IPFS_PATH_RE = re.compile(IPFS_PATH_PATTERN, re.I)
_LUCKY_RUN_RE = re.compile(r"168|888|666|999|777|8899|9988|6688|8866|5566|1314|^(?:88|99|66|77)$")
_HAS_GCLID_RE = re.compile(r"(?:^|[?&#])(?:gclid|gbraid|wbraid|msclkid|dclid)=")

_SCAM_ROUTE_RES = tuple(re.compile(p, re.I) for p in (
    r"#/pages/[a-z]",
    r"/(?:h5|wap)/#/",
    r"#/(?:register|reg|signup|sign-up)(?:[/?&]|$)",
    r"[?&](?:invite_?code|agent_?code|recommend_?code|share_?code|inviter_?code|tjm|yqm)=[a-z0-9]",
    r"^/(?:[^?#]*/)?(?:register|reg|signup)(?:\.html?|\.php)?\?(?:[^#]*&)?(?:code|ic|agent|inv|pid)=[a-z0-9]",
    r"^/?\?(?:[^#]*&)?(?:agent|ic|tjm|yqm)=[a-z0-9]",
    r"^/(?:[^?#]*/)?(?:download|down|appdown|app-download|app)\.(?:html?|php)(?:[?#]|$)",
    r"^[^?#]*\.(?:apk|ipa|mobileconfig)(?:[?#]|$)",
))

_MOBILE_HOST_PREFIXES = frozenset({"m", "h5", "wap", "app", "sj", "mobile"})

# 所有品牌 token（長的優先，避免 line 先吃掉 linebank）
_BRANDS: Tuple[str, ...] = tuple(sorted(BRAND_OFFICIAL_DOMAINS, key=lambda s: (-len(s), s)))
_BRAND_SET: FrozenSet[str] = frozenset(_BRANDS)
_CRYPTO_BRANDS: FrozenSet[str] = frozenset(
    b for b, domains in BRAND_OFFICIAL_DOMAINS.items()
    if any(d in {"binance.com", "okx.com"} for d in domains) or b in {
        "bybit", "bitget", "kucoin", "coinbase", "kraken", "mexc", "htx", "huobi", "bitfinex",
        "bitflyer", "upbit", "bingx", "maicoin", "bitopro", "hoyabit", "xrex", "metamask",
        "trustwallet", "tronlink", "imtoken", "tokenpocket", "safepal", "trezor", "uniswap",
        "pancakeswap", "opensea", "walletconnect", "coinw", "bitmart", "phemex",
    }
)
_GOV_BRANDS: FrozenSet[str] = frozenset({"165", "npa", "mof", "moi", "nhi", "fsc", "cib", "etax"})
_GOV_CO_TOKENS: FrozenSet[str] = frozenset({
    "npa", "gov", "police", "fraud", "antifraud", "refund", "tax", "tw", "taiwan", "165",
})
# 截斷型拼字變形（walmar、faceboo…）只對 ≥7 字且非一般英文字的品牌啟用
_TRUNCATION_BRANDS: Tuple[str, ...] = tuple(BRAND_TRUNCATION_TARGETS)
# 品牌中間插入連字號（binanc-e）：(左段, 右段) → 品牌
_HYPHEN_SPLITS: Dict[Tuple[str, str], str] = {
    tuple(v.split("-", 1)): v.replace("-", "") for v in brand_hyphen_variants()  # type: ignore[misc]
}
# 隨機字串判斷用：英文／拼音常見的「子音＋子音」組合（其餘子音連接視為罕見）
_COMMON_CC_BIGRAMS: FrozenSet[str] = frozenset("""
bl br ch ck cl cr ct dr fl fr ft gh gl gn gr kn lb lc ld lf lk ll lm ln lp ls lt lv mb mm mn mp
ms nc nd nf ng nk nn ns nt nv nz ph pl pp pr ps pt rb rc rd rf rg rk rl rm rn rp rr rs rt rv sc
sh sk sl sm sn sp ss st sw th tl tr ts tt tw wh wl wn wr ws xt zh dd ff gg bb cc nh rh nb nl nw
nm nj nq nx ny
""".split())
_HOMOGLYPH_MAPS: Tuple[Dict[str, str], ...] = (
    {"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b", "@": "a", "$": "s"},
    {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b", "@": "a", "$": "s"},
)

_ALL_OFFICIAL_DOMAINS: FrozenSet[str] = frozenset(
    d for domains in BRAND_OFFICIAL_DOMAINS.values() for d in domains
)

# 舊版名稱相容：SHORTENER_DOMAINS / SUSPICIOUS_TLDS 仍可自本模組 import（實際定義在 rules_config）
__all_legacy__ = ("SHORTENER_DOMAINS", "SUSPICIOUS_TLDS", "IP_ADDRESS_PATTERN")


# =============================================================================
# URL 正規化與解析（永不丟例外）
# =============================================================================

def normalize_url(url: Any) -> str:
    """
    URL 基本清理（不補 scheme）：NFKC（全形轉半形）、移除零寬字元、控制字元與
    所有空白（含全形空白 \\u3000）。None 或無法轉成字串時回傳 ""。
    """
    if url is None:
        return ""
    try:
        s = str(url)
    except Exception:  # noqa: BLE001
        return ""
    if len(s) > _MAX_INPUT_LENGTH:
        s = s[:_MAX_INPUT_LENGTH]
    try:
        s = unicodedata.normalize("NFKC", s)
    except Exception:  # noqa: BLE001
        pass
    s = _ZERO_WIDTH_RE.sub("", s)
    s = _WS_RE.sub("", s)
    s = _CONTROL_RE.sub("", s)
    return s


def _detect_scheme(s: str) -> Optional[str]:
    """回傳明確寫出的 scheme（小寫）；host:port 形式（example.com:8080）不算 scheme。"""
    m = _SCHEME_RE.match(s)
    if not m:
        return None
    scheme = m.group(1).lower()
    if m.group(2) or scheme in _OPAQUE_SCHEMES:
        return scheme
    if scheme in ("http", "https") and s[len(scheme) + 1:len(scheme) + 2] in ("/", "\\"):
        return scheme
    return None


def url_has_explicit_scheme(url: Any) -> bool:
    """輸入是否明確寫了 scheme（http://、https://、javascript: 等）。"""
    try:
        return _detect_scheme(normalize_url(url)) is not None
    except Exception:  # noqa: BLE001
        return False


def _idna_ascii(host: str) -> str:
    """把含非 ASCII 字元的 label 轉成 punycode（xn--）；失敗則保留原字串。"""
    if host.isascii():
        return host
    labels = []
    for label in host.split("."):
        if label.isascii() or not label:
            labels.append(label)
            continue
        try:
            labels.append("xn--" + label.encode("punycode").decode("ascii"))
        except Exception:  # noqa: BLE001
            labels.append(label)
    return ".".join(labels)


def _idna_unicode(host: str) -> str:
    """把 xn-- label 解回 unicode（用於中文詞比對）；失敗則保留原 label。"""
    if "xn--" not in host:
        return host
    labels = []
    for label in host.split("."):
        if label.startswith("xn--"):
            try:
                labels.append(label[4:].encode("ascii").decode("punycode"))
                continue
            except Exception:  # noqa: BLE001
                pass
        labels.append(label)
    return ".".join(labels)


def _ascii_fold(text: str) -> str:
    """去除重音符號並只保留 ASCII 英數（esunbänk → esunbank）。"""
    try:
        decomposed = unicodedata.normalize("NFKD", text)
    except Exception:  # noqa: BLE001
        decomposed = text
    return "".join(ch for ch in decomposed if ch.isascii() and ch.isalnum()).lower()


def _split_hostport(hostport: str) -> Tuple[str, str]:
    if hostport.startswith("["):
        end = hostport.find("]")
        if end == -1:
            return hostport[1:], ""
        rest = hostport[end + 1:]
        return hostport[1:end], (rest[1:] if rest.startswith(":") else "")
    host, _, port = hostport.partition(":")
    return host, port


def _canonical_authority(authority: str) -> str:
    authority = authority.lower()
    userinfo, at, hostport = authority.rpartition("@")
    host, port = _split_hostport(hostport)
    is_v6 = hostport.startswith("[")
    host = host.rstrip(".")
    if not is_v6:
        host = _idna_ascii(host)
    host_text = f"[{host}]" if is_v6 else host
    port_text = f":{port}" if port else ""
    return f"{userinfo}{at}{host_text}{port_text}"


@lru_cache(maxsize=8192)
def _canonicalize_cached(s: str) -> str:
    if not s:
        return ""
    scheme = _detect_scheme(s)
    if scheme is None:
        s = ("https:" + s) if s.startswith("//") else ("https://" + s)
        scheme = "https"
    elif scheme in _OPAQUE_SCHEMES:
        return scheme + s[len(scheme):]
    rest = s[len(scheme) + 1:]
    rest = rest.lstrip("/\\")
    m = re.match(r"([^/?#\\]*)(.*)", rest, re.S)
    authority, tail = (m.group(1), m.group(2)) if m else (rest, "")
    if scheme in ("http", "https"):
        tail = tail.replace("\\", "/")
    return f"{scheme}://{_canonical_authority(authority)}{tail}"


def canonicalize_url(url: Any) -> str:
    """
    訓練與推論共用的 URL 正規化（v7）：
      - normalize_url()（NFKC、移除零寬字元與空白）
      - 沒有 scheme 時補「https://」（不再補 http://，消除 scheme 捷徑）
      - scheme 與 host 轉小寫、host 去尾端點、中文網域轉 punycode；路徑大小寫保留
    永不丟例外，無法處理時回傳 normalize_url() 的結果。
    """
    s = normalize_url(url)
    try:
        return _canonicalize_cached(s)
    except Exception:  # noqa: BLE001
        return s


def parse_url(url: Any) -> ParseResult:
    """安全解析 URL（先 canonicalize）；任何錯誤都回傳空的 ParseResult。"""
    try:
        return urlparse(canonicalize_url(url))
    except Exception:  # noqa: BLE001
        return urlparse("")


def _percent_decode(text: str) -> str:
    """percent-decode（最多兩輪，處理雙重編碼）＋ NFKC；失敗時回傳原字串。"""
    try:
        out = text
        for _ in range(2):
            if "%" not in out:
                break
            decoded = unquote(out, errors="replace")
            if decoded == out:
                break
            out = decoded
        return unicodedata.normalize("NFKC", out)
    except Exception:  # noqa: BLE001
        return text


# =============================================================================
# PSL（tldextract 離線快照）與有限 fallback 解析
# =============================================================================

_EXTRACTOR: Any = None
_EXTRACTOR_FAILED = False


def _get_extractor() -> Any:
    global _EXTRACTOR, _EXTRACTOR_FAILED
    if _EXTRACTOR is not None or _EXTRACTOR_FAILED or _tldextract is None:
        return _EXTRACTOR
    try:
        try:
            _EXTRACTOR = _tldextract.TLDExtract(
                suffix_list_urls=(), include_psl_private_domains=True, cache_dir=None
            )
        except TypeError:
            _EXTRACTOR = _tldextract.TLDExtract(
                suffix_list_urls=(), include_psl_private_domains=True
            )
    except Exception:  # noqa: BLE001
        _EXTRACTOR = None
        _EXTRACTOR_FAILED = True
    return _EXTRACTOR


def _fallback_public_suffix(host: str) -> str:
    """以最長匹配方式從專題常見 suffix 集合取得公開後綴。"""
    host = host.strip(".").lower()
    if not host:
        return ""
    labels = host.split(".")
    for i in range(len(labels) - 1):
        candidate = ".".join(labels[i:])
        if candidate in PUBLIC_SUFFIX_FALLBACKS:
            return candidate
    # 單段常見 gTLD：只接受明確且常見的值，不再把任意字母尾段當成 TLD。
    if labels[-1] in {
        "com", "net", "org", "gov", "edu", "mil", "biz", "info", "name", "pro",
        "xyz", "top", "vip", "shop", "site", "online", "club", "me", "io", "ai", "app",
        "dev", "tech", "store", "live", "cc", "tv", "ly", "to", "gg", "co", "uk", "de", "fr",
        "jp", "kr", "cn", "au", "nz", "sg", "my", "hk", "tw",
    }:
        return labels[-1]
    return ""


def _simple_split(host: str) -> Tuple[str, str, str]:
    """離線 fallback：優先最長 suffix 匹配，再退回常見 gTLD。"""
    parts = [p for p in host.split(".") if p]
    if not parts:
        return "", "", ""
    if len(parts) == 1:
        return "", parts[0], ""
    suffix = _fallback_public_suffix(host)
    if suffix:
        suffix_labels = suffix.split(".")
        if len(parts) >= len(suffix_labels) + 1:
            sld_index = len(parts) - len(suffix_labels) - 1
            return ".".join(parts[:sld_index]), parts[sld_index], suffix
        # host 本身就是 suffix，例如 com.tw；保留與 tldextract 相近的退化結果
        if len(parts) == len(suffix_labels):
            return "", parts[0], ".".join(parts[1:])
    # 最後一道保守 fallback：兩段 host 的最後一段作 suffix。
    return ".".join(parts[:-2]), parts[-2], parts[-1]


@lru_cache(maxsize=8192)
def _split_host(host: str) -> Tuple[str, str, str]:
    """回傳 (subdomain, sld, public_suffix)；host 已去 www. 且非 IP。"""
    if not host:
        return "", "", ""
    extractor = _get_extractor()
    if extractor is not None:
        try:
            result = extractor(host)
            suffix = result.suffix or ""
            if suffix:
                if result.domain:
                    return result.subdomain or "", result.domain, suffix
                # host 本身就是公開後綴（例如 github.io）：把第一段當 SLD
                head, _, tail = suffix.partition(".")
                return result.subdomain or "", head, tail
        except Exception:  # noqa: BLE001
            pass
    return _simple_split(host)


# =============================================================================
# 公用小工具（保留 v6 介面，全部不丟例外）
# =============================================================================

def is_ip_address(hostname: Any) -> int:
    """判斷 hostname 是否為 IP（IPv4、IPv6、十進位整數形式）。"""
    try:
        host = str(hostname or "").strip().strip("[]").lower()
        if not host:
            return 0
        if IP_ADDRESS_PATTERN.match(host):
            return int(all(0 <= int(part) <= 255 for part in host.split(".")))
        if ":" in host and re.fullmatch(r"[0-9a-f:.]+", host) and host.count(":") >= 2:
            return 1
        if host.isdigit() and 8 <= len(host) <= 10 and int(host) <= 0xFFFFFFFF:
            return 1
        if re.fullmatch(r"0x[0-9a-f]{8}", host):
            return 1
    except Exception:  # noqa: BLE001
        return 0
    return 0


def levenshtein_distance(s1: str, s2: str) -> int:
    """計算兩個字串的 Levenshtein 編輯距離（不依賴外部套件）。"""
    s1 = s1 or ""
    s2 = s2 or ""
    m, n = len(s1), len(s2)
    if m == 0:
        return n
    if n == 0:
        return m
    prev = list(range(n + 1))
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


@lru_cache(maxsize=4096)
def _min_brand_levenshtein_cached(sld_lower: str) -> int:
    if len(sld_lower) < 3 or len(sld_lower) > 64:
        return _NO_BRAND_DISTANCE
    return min(levenshtein_distance(sld_lower, brand) for brand in BRAND_NAME_LIST)


def min_brand_levenshtein(sld: Any) -> int:
    """
    SLD 與 BRAND_NAME_LIST 的「原始」最小編輯距離（舊介面保留，不考慮官方網域）。
    特徵 levenshtein_brand_dist 改用 _effective_brand_distance()（會排除官方網域與短品牌誤判）。
    """
    try:
        return _min_brand_levenshtein_cached(re.sub(r"[^a-z0-9]", "", str(sld or "").lower()))
    except Exception:  # noqa: BLE001
        return _NO_BRAND_DISTANCE


@lru_cache(maxsize=4096)
def _calc_entropy_cached(s: str) -> float:
    if not s:
        return 0.0
    freq: Dict[str, int] = {}
    for char in s:
        freq[char] = freq.get(char, 0) + 1
    length = len(s)
    return -sum((count / length) * math.log2(count / length) for count in freq.values())


def calc_entropy(s: Any) -> float:
    """字串的 Shannon entropy；隨機字元組成的網域（如 xhgkqpz）entropy 較高。"""
    try:
        return _calc_entropy_cached(str(s or ""))
    except Exception:  # noqa: BLE001
        return 0.0


# =============================================================================
# URL 解析結果容器
# =============================================================================

@dataclass(frozen=True)
class URLComponents:
    """集中保存單一 URL 解析一次後的所有片段（frozen，可安全被快取共用）。"""

    original_url: str          # normalize_url 後的輸入（未補 scheme）
    canonical_url: str         # canonicalize_url 結果
    lower_url: str             # canonical 小寫
    decoded_url: str           # canonical percent-decode 後（保留大小寫）
    has_scheme: bool
    scheme: str
    userinfo: str
    has_userinfo: bool
    host_full: str             # 含 www.（小寫、無埠、無 userinfo、IPv6 無括號）
    hostname: str              # 去 www.
    host_unicode: str          # punycode 解回 unicode
    port_text: str
    port: Optional[int]
    path: str
    query: str
    fragment: str
    rest: str                  # path + ?query + #fragment（原始）
    decoded_rest: str          # 同上解碼後
    body: str                  # 去 scheme 與 www. 的網址（長度/字元特徵用）
    is_ip: bool
    subdomain: str
    sld: str
    suffix: str
    tld: str
    registered_domain: str
    host_tokens: Tuple[str, ...]   # 公開後綴以外的 hostname token（punycode 已折疊）
    sld_tokens: Tuple[str, ...]
    sld_folded: str                # SLD（punycode 折疊為 ASCII）
    path_tokens: Tuple[str, ...]   # 解碼後 path/query/fragment 的英數 token（小寫）
    is_trusted: bool
    is_content_platform: bool
    tld_risk: int


def _tokens_of_label(label: str) -> List[str]:
    if label.startswith("xn--"):
        folded = _ascii_fold(_idna_unicode(label))
        return [folded] if folded else []
    return [t for t in _TOKEN_SPLIT_RE.split(label) if t]


def _host_matches_any(host: str, registered: str, domains: Iterable[str]) -> bool:
    return registered in domains or host in domains


def _endswith_suffix(host: str, suffixes: Iterable[str]) -> bool:
    dotted = "." + host
    return any(dotted.endswith(s if s.startswith(".") else "." + s) for s in suffixes)


def _empty_components(original: str = "", canonical: str = "") -> URLComponents:
    return URLComponents(
        original_url=original, canonical_url=canonical, lower_url=canonical.lower(),
        decoded_url=canonical, has_scheme=False, scheme="", userinfo="", has_userinfo=False,
        host_full="", hostname="", host_unicode="", port_text="", port=None, path="", query="",
        fragment="", rest="", decoded_rest="", body="", is_ip=False, subdomain="", sld="",
        suffix="", tld="", registered_domain="", host_tokens=(), sld_tokens=(), sld_folded="",
        path_tokens=(), is_trusted=False, is_content_platform=False, tld_risk=0,
    )


def _build_url_components(url: Any) -> URLComponents:
    """將輸入 URL 解析一次，組成 URLComponents 供後續所有特徵函式共用。"""
    original = normalize_url(url)
    if not original:
        return _empty_components()
    canonical = canonicalize_url(original)
    has_scheme = _detect_scheme(original) is not None

    m = _HIER_RE.match(canonical)
    if not m:
        # javascript:、mailto: 等不具 host 的網址
        scheme = canonical.split(":", 1)[0].lower() if ":" in canonical else ""
        comp = _empty_components(original, canonical)
        body = canonical[len(scheme) + 1:] if scheme else canonical
        return URLComponents(**{**comp.__dict__, "has_scheme": has_scheme, "scheme": scheme,
                                "body": body, "rest": body, "decoded_rest": _percent_decode(body),
                                "decoded_url": _percent_decode(canonical)})

    scheme, authority, rest = m.group(1), m.group(2), m.group(3)
    userinfo, at, hostport = authority.rpartition("@")
    host_full, port_text = _split_hostport(hostport)
    host_full = host_full.strip(".")
    is_v6 = hostport.startswith("[")
    hostname = host_full[4:] if host_full.startswith("www.") and "." in host_full[4:] else host_full

    port: Optional[int] = None
    if port_text.isdigit() and len(port_text) <= 5 and int(port_text) <= 65535:
        port = int(port_text)

    frag_idx = rest.find("#")
    fragment = rest[frag_idx + 1:] if frag_idx >= 0 else ""
    before_frag = rest[:frag_idx] if frag_idx >= 0 else rest
    q_idx = before_frag.find("?")
    query = before_frag[q_idx + 1:] if q_idx >= 0 else ""
    path = before_frag[:q_idx] if q_idx >= 0 else before_frag

    host_display = f"[{host_full}]" if is_v6 else hostname
    tail = "" if rest == "/" else rest
    body = (f"{userinfo}@" if at else "") + host_display + (f":{port_text}" if port_text else "") + tail

    ip = bool(is_ip_address(host_full))
    if ip or not hostname:
        subdomain, sld, suffix = "", "", ""
        registered = hostname
    else:
        subdomain, sld, suffix = _split_host(hostname)
        registered = f"{sld}.{suffix}" if sld and suffix else (sld or hostname)
    tld = suffix.rsplit(".", 1)[-1] if suffix else ""

    labels = [lab for lab in subdomain.split(".") if lab] + ([sld] if sld else [])
    host_tokens: List[str] = []
    for label in labels:
        host_tokens.extend(_tokens_of_label(label))
    sld_tokens = tuple(_tokens_of_label(sld)) if sld else ()
    sld_folded = _ascii_fold(_idna_unicode(sld)) if sld.startswith("xn--") else sld

    decoded_rest = _percent_decode(rest)
    path_tokens = tuple(t for t in _TOKEN_SPLIT_RE.split(decoded_rest.lower()) if t)

    is_trusted = bool(hostname) and (
        _host_matches_any(hostname, registered, TRUSTED_DOMAINS)
        or host_full in TRUSTED_DOMAINS
        or _endswith_suffix(hostname, TRUSTED_SUFFIXES)
    )
    is_content = bool(hostname) and (
        _host_matches_any(hostname, registered, CONTENT_PLATFORM_DOMAINS)
        or _endswith_suffix(hostname, (".gov.tw", ".edu.tw"))
    )
    tld_risk = 2 if tld in _TIER2 else (1 if tld in _TIER1 else 0)

    return URLComponents(
        original_url=original,
        canonical_url=canonical,
        lower_url=canonical.lower(),
        decoded_url=_percent_decode(canonical),
        has_scheme=has_scheme,
        scheme=scheme,
        userinfo=userinfo,
        has_userinfo=bool(at),
        host_full=host_full,
        hostname=hostname,
        host_unicode=_idna_unicode(hostname),
        port_text=port_text,
        port=port,
        path=path,
        query=query,
        fragment=fragment,
        rest=rest,
        decoded_rest=decoded_rest,
        body=body,
        is_ip=ip,
        subdomain=subdomain,
        sld=sld,
        suffix=suffix,
        tld=tld,
        registered_domain=registered,
        host_tokens=tuple(host_tokens),
        sld_tokens=sld_tokens,
        sld_folded=sld_folded,
        path_tokens=path_tokens,
        is_trusted=is_trusted,
        is_content_platform=is_content,
        tld_risk=tld_risk,
    )


@lru_cache(maxsize=8192)
def _components_cached(url: str) -> URLComponents:
    try:
        return _build_url_components(url)
    except Exception:  # noqa: BLE001
        return _empty_components(normalize_url(url))


def _components(url: Any) -> URLComponents:
    try:
        key = url if isinstance(url, str) else ("" if url is None else str(url))
    except Exception:  # noqa: BLE001
        key = ""
    return _components_cached(key[:_MAX_INPUT_LENGTH])


def get_hostname(url: Any) -> str:
    """取得小寫 hostname（去 www.、去尾端點、無埠號與 userinfo；IDN 為 punycode）。"""
    return _components(url).hostname


def get_registered_domain(url: Any) -> str:
    """
    取得 registered domain（tldextract 離線 PSL；含 private 後綴）。
      sub.momo.com.tw → momo.com.tw；binance-tw-pro.github.io → binance-tw-pro.github.io
    IP 位址回傳 IP 本身。
    """
    return _components(url).registered_domain


def get_sld(url: Any) -> str:
    """取得 SLD（registered domain 去掉公開後綴）：sub.momo.com.tw → momo。"""
    return _components(url).sld


def get_tld(url: Any) -> str:
    """取得公開後綴的最後一段（小寫）：momo.com.tw → tw；IP 回傳 ""。"""
    return _components(url).tld


def get_rule_targets(url: Any) -> Dict[str, str]:
    """
    產生硬規則比對字串（rules_config.HardRule.target）：
      url = canonical、decoded = 解碼後 canonical、host = hostname、path = 解碼後 path+query+fragment
    """
    c = _components(url)
    return {
        "url": c.canonical_url,
        "decoded": c.decoded_url,
        "host": c.hostname,
        "path": c.decoded_rest,
    }


def extract_redirect_targets(url: Any, limit: int = 3) -> List[str]:
    """
    取出網址 query/path 中內嵌的跳轉目標（例如 google.com/url?q=https://x.top/、
    doubleclick 的 adurl=）。建議 main.py 對這些內層網址另外評估並取較高分。
    """
    try:
        c = _components(url)
        found: List[str] = []
        for raw in _EMBEDDED_URL_RE.findall(c.decoded_rest):
            target = canonicalize_url(raw.split("&", 1)[0])
            if target and target not in found:
                found.append(target)
            if len(found) >= limit:
                break
        return found
    except Exception:  # noqa: BLE001
        return []


# =============================================================================
# 品牌比對
# =============================================================================

def _strip_digits(token: str) -> str:
    return token.strip(_DIGITS)


def _is_official_for(brand: str, c: URLComponents) -> bool:
    """registered domain 是否屬於該品牌官方（或白名單／政府學術網域）。"""
    if c.is_trusted:
        return True
    domains = BRAND_OFFICIAL_DOMAINS.get(brand, set())
    if c.registered_domain in domains or c.hostname in domains:
        return True
    if c.registered_domain in _ALL_OFFICIAL_DOMAINS and brand not in _GOV_BRANDS:
        # 例：rakuten-bank.com.tw 屬 rakuten 官方，不算 bank 類品牌冒用
        return True
    first_label = c.suffix.split(".", 1)[0] if c.suffix else ""
    if first_label in _GOV_SUFFIX_LABELS or c.suffix in ("gov", "edu", "mil"):
        return True
    if (brand in MULTI_CC_BRANDS and c.sld == brand and len(c.tld) == 2 and c.tld_risk == 0):
        return True
    return False


@lru_cache(maxsize=16384)
def _brand_token_match(token: str) -> Tuple[str, str]:
    """
    單一 token 的品牌比對，回傳 (brand, kind)；未命中回傳 ("", "")。
    kind：exact（完全相等）/ affix（品牌+詐騙後綴，可接數字）/ digits（品牌+數字，品牌≥5字）/
          prefix（tw/vip…+品牌）/ tail（≥6 字品牌 + 1～3 個亂碼字母）
    """
    if not token or len(token) > 40:
        return "", ""
    if token in _BRAND_SET:
        return token, "exact"
    for brand in _BRANDS:
        if token.startswith(brand) and len(token) > len(brand):
            rest = token[len(brand):]
            if rest.isdigit():
                if len(brand) >= 5:
                    return brand, "digits"
                continue
            rest_alpha = rest.rstrip(_DIGITS)
            if rest_alpha in BRAND_AFFIX_TOKENS:
                return brand, "affix"
            # v7.1：品牌後面黏多個後綴（kucoinexchangevip = kucoin + exchange + vip）
            if len(rest) >= 4 and _affix_chain(rest):
                return brand, "affix"
            if (len(brand) >= 6 and brand not in BRAND_GENERIC_WORDS and rest_alpha.isalpha()
                    and 1 <= len(rest_alpha) <= 3):
                return brand, "tail"
    for prefix in BRAND_PREFIX_TOKENS:
        if token.startswith(prefix) and len(token) > len(prefix):
            rest = token[len(prefix):].rstrip(_DIGITS)
            if rest in _BRAND_SET:
                return rest, "prefix"
    # v7.1：「填充詞串＋品牌」（netbankscsb = netbank + scsb）；品牌須 ≥4 字且不是一般英文字
    core = token.rstrip(_DIGITS)
    for brand in _BRANDS:
        if (len(brand) >= 4 and brand not in BRAND_GENERIC_WORDS and core.endswith(brand)
                and len(core) >= len(brand) + 2 and _affix_chain(core[:-len(brand)])):
            return brand, "prefix"
    return "", ""


def _brand_hits(c: URLComponents, tokens: Tuple[str, ...]) -> List[str]:
    """回傳 tokens 中「冒用」的品牌（已排除官方網域、一般英文字的單獨出現）。"""
    hits: List[str] = []
    token_set = set(tokens)
    # 政府字樣冒用：gov + tw 相鄰（npa-gov-tw、post-gov-tw），但不是 .gov.tw 官方網域
    if not c.is_trusted and not _endswith_suffix(c.hostname, (".gov.tw",)):
        pairs = set(zip(tokens, tokens[1:]))
        if ("gov", "tw") in pairs or ("gov", "taiwan") in pairs or "govtw" in token_set:
            hits.append("gov")
    for idx, token in enumerate(tokens):
        brand, kind = _brand_token_match(token)
        if not brand:
            continue
        if kind == "exact":
            neighbours = set(tokens[max(0, idx - 1):idx]) | set(tokens[idx + 1:idx + 2])
            if brand in _GOV_BRANDS:
                # 機關縮寫（165、npa、mof、fsc…）單獨出現常是別的意思（fsc.org），需搭配 gov/tax/refund 等字
                if not (token_set & (_GOV_CO_TOKENS - {brand})):
                    continue
            elif brand in BRAND_GENERIC_WORDS:
                if not (neighbours & (BRAND_AFFIX_TOKENS | BRAND_PREFIX_TOKENS) or c.tld_risk >= 1):
                    continue
        if _is_official_for(brand, c):
            continue
        hits.append(brand)
    return hits


def _homoglyph_variants(token: str) -> Set[str]:
    if not (any(ch.isdigit() for ch in token) and any(ch.isalpha() for ch in token)):
        return set()
    out: Set[str] = set()
    for mapping in _HOMOGLYPH_MAPS:
        out.add("".join(mapping.get(ch, ch) for ch in token))
    out.discard(token)
    return out


def _is_char_doubling(candidate: str, brand: str) -> bool:
    """candidate 是否為 brand 某個字元重複一次（lazadaa、shopeee、googgle）。"""
    if len(candidate) != len(brand) + 1:
        return False
    for i in range(1, len(candidate)):
        if candidate[i] == candidate[i - 1] and candidate[:i] + candidate[i + 1:] == brand:
            return True
    return False


def _effective_brand_distance(c: URLComponents, other_signal: bool) -> int:
    """
    levenshtein_brand_dist 的實作：與「非官方」品牌的有效最小編輯距離。
      0 = 與品牌同名（或同形字還原後同名）但不是官方網域
      1～2 = 拼字近似（≤4 字品牌只接受 1 且需其他風險訊號；5～6 字品牌距離 1 需其他訊號、
             重複字元或同形字；≥7 字品牌接受 1～2）
      99 = 無相似品牌／本身就是官方網域／SLD 太短
    """
    if c.is_ip or not c.sld or c.is_trusted:
        return _NO_BRAND_DISTANCE
    base = re.sub(r"[^a-z0-9]", "", c.sld_folded.lower())
    raw_candidates: Set[str] = set()
    if base:
        raw_candidates.add(base)
    raw_candidates.update(t for t in c.sld_tokens if len(t) >= 4)
    candidates: Dict[str, bool] = {cand: False for cand in raw_candidates}
    for cand in raw_candidates:
        for variant in _homoglyph_variants(cand):
            candidates.setdefault(variant, True)

    best = _NO_BRAND_DISTANCE
    for cand, from_homoglyph in candidates.items():
        n = len(cand)
        if n < 3 or n > 30:
            continue
        # 候選字本身就是「品牌＋後綴/數字」（esunbanktw、kucoinx）→ 視為含品牌原名（距離 0）
        matched, kind = _brand_token_match(cand)
        if matched and kind != "exact":
            if not _is_official_for(matched, c) and (matched not in BRAND_GENERIC_WORDS or other_signal):
                best = 0
            continue
        common_word = cand in LEVENSHTEIN_COMMON_WORDS
        for brand in BRAND_NAME_LIST:
            lb = len(brand)
            if abs(n - lb) > 2:
                continue
            d = levenshtein_distance(cand, brand)
            if d > 2 or d >= best:
                continue
            if _is_official_for(brand, c):
                continue
            if d == 0:
                ok = from_homoglyph or brand not in BRAND_GENERIC_WORDS or other_signal
            elif common_word:
                ok = False
            elif lb <= 4:
                ok = d == 1 and other_signal
            elif lb <= 6:
                ok = d == 1 and (other_signal or from_homoglyph or _is_char_doubling(cand, brand))
            else:
                ok = d == 1 or (d == 2 and n >= 6)
            if ok:
                best = d
    return best


def _brand_typo_like(c: URLComponents, lev_dist: int) -> int:
    if c.is_ip or c.is_trusted or not c.hostname:
        return 0
    candidates = list(c.host_tokens)
    base = re.sub(r"[^a-z0-9]", "", c.sld_folded.lower())
    if base and base not in candidates:
        candidates.append(base)
    for token in candidates:
        # (a) 明確的拼字變形詞
        for term, brand in BRAND_TYPO_TERMS.items():
            if not token.startswith(term) or token.startswith(brand):
                continue
            rest = token[len(term):]
            if (not rest or rest.isdigit() or rest.rstrip(_DIGITS) in BRAND_AFFIX_TOKENS
                    or len(term) >= 7):
                if not _is_official_for(brand, c):
                    return 1
        # (b) 同形字（0→o、1→l/i、3→e…）還原後是品牌
        if not _brand_token_match(token)[0]:
            for variant in _homoglyph_variants(token):
                brand, _ = _brand_token_match(variant)
                if brand and not _is_official_for(brand, c):
                    return 1
        # (c) 截斷字：品牌少最後一個字母後接其他字（walmarvahu、faceboo）
        for brand in _TRUNCATION_BRANDS:
            prefix = brand[:-1]
            if token.startswith(prefix) and not token.startswith(brand) and len(token) >= len(prefix):
                if not _is_official_for(brand, c):
                    return 1
    # (d) 品牌中間插入連字號（binanc-e、c-oinbase）
    tokens = c.sld_tokens
    for left, right in zip(tokens, tokens[1:]):
        brand = _HYPHEN_SPLITS.get((left, right))
        if brand and not _is_official_for(brand, c):
            return 1
    # (e) 有效編輯距離 1～2
    return int(1 <= lev_dist <= 2)


# =============================================================================
# v7.1 黏字切分：在 token 中找詞庫詞，且「剩餘部分」必須是數字／常見後綴／另一個詞庫詞
# （biaoguvip = biaogu+vip、188bettw = 188+bet+tw、kucoinexchangevip = kucoin+exchange+vip、
#   twlicaiplus = tw+licai+plus）。整個 token 必須能被完整切開，避免 alphabet → alpha+bet 這類誤判。
# =============================================================================

_SEG_FILLERS: FrozenSet[str] = frozenset(t for t in SEGMENT_AFFIX_TOKENS if t.isalnum() and len(t) >= 2)
_CRYPTO_SHORT_STRONG: FrozenSet[str] = frozenset(t for t in CRYPTO_TERMS["strong_tokens"] if len(t) <= 3)
_CRYPTO_SPECIFIC_WEAK: FrozenSet[str] = frozenset(CRYPTO_TERMS["weak_tokens"]) & frozenset({
    "exchange", "swap", "coin", "coins", "crypto", "token", "tokens", "btc", "bitcoin", "eth", "ether",
    "bnb", "xrp", "doge", "nft", "stake", "mining", "miner", "futures", "perp", "perpetual", "leverage",
})


def _build_seg_vocab(groups: Dict[str, Iterable[str]]) -> Dict[str, FrozenSet[str]]:
    vocab: Dict[str, Set[str]] = {w: set() for w in _SEG_FILLERS}
    for kind, words in groups.items():
        for word in words:
            if word and word.isalnum() and len(word) >= 2:
                vocab.setdefault(word, set()).add(kind)
    return {w: frozenset(k) for w, k in vocab.items()}


_SEG_VOCAB: Dict[str, Dict[str, FrozenSet[str]]] = {
    "gambling": _build_seg_vocab({
        "strong": GAMBLING_TERMS["strong_tokens"],
        "bet": {"bet", "bets"},
    }),
    "investment": _build_seg_vocab({
        "strong": INVESTMENT_TERMS["strong_tokens"],
        "core": INVESTMENT_TERMS["core_tokens"],
        "clure": INVESTMENT_TERMS["compound_lure_tokens"],
    }),
    "crypto": _build_seg_vocab({
        "strong": {t for t in CRYPTO_TERMS["strong_tokens"] if len(t) >= 4},
        "short": _CRYPTO_SHORT_STRONG,
        "weak": _CRYPTO_SPECIFIC_WEAK,
        "brand": _CRYPTO_BRANDS,
    }),
    "affix": _build_seg_vocab({}),
}
_SEG_MAXLEN: Dict[str, int] = {cat: max(len(w) for w in vocab) for cat, vocab in _SEG_VOCAB.items()}
_SEG_MAX_SIGNATURES = 64


def _merge_signature(tail: Tuple[Tuple[str, int], ...], kinds: Iterable[str]) -> Tuple[Tuple[str, int], ...]:
    counts = dict(tail)
    for kind in list(kinds) + ["_pieces"]:
        counts[kind] = min(counts.get(kind, 0) + 1, 2)
    return tuple(sorted(counts.items()))


@lru_cache(maxsize=16384)
def _segment_signatures(category: str, token: str) -> FrozenSet[Tuple[Tuple[str, int], ...]]:
    """
    token 能被「完整」切成 詞庫詞／填充詞（SEGMENT_AFFIX_TOKENS）／數字段 時，回傳每種切法的
    kinds 計數簽章（例如 (("_pieces", 2), ("strong", 1))，計數上限 2）；不能完整切開時回傳空集合。
    只保留切成 2 段以上的切法（單一詞已由一般 token 比對處理）。
    """
    vocab = _SEG_VOCAB.get(category)
    n = len(token)
    if not vocab or n < 4 or n > 40:
        return frozenset()
    maxlen = _SEG_MAXLEN[category]
    sigs: List[Set[Tuple[Tuple[str, int], ...]]] = [set() for _ in range(n + 1)]
    sigs[n].add(())
    for i in range(n - 1, -1, -1):
        options: List[Tuple[int, Iterable[str]]] = []
        if token[i].isdigit():
            j = i
            while j < n and token[j].isdigit():
                j += 1
            run = token[i:j]
            options.append((j, ("digits", "lucky") if _LUCKY_RUN_RE.search(run) else ("digits",)))
        for length in range(2, min(maxlen, n - i) + 1):
            kinds = vocab.get(token[i:i + length])
            if kinds is not None:
                options.append((i + length, kinds))
        for j, kinds in options:
            for tail in sigs[j]:
                sigs[i].add(_merge_signature(tail, kinds))
                if len(sigs[i]) >= _SEG_MAX_SIGNATURES:
                    break
    return frozenset(s for s in sigs[0] if dict(s).get("_pieces", 0) >= 2)


def _segment_hit(category: str, token: str, strong_only: bool = False) -> bool:
    """依類別判斷 token 的黏字切分是否構成關鍵詞訊號（strong_only：只認含強詞的切法）。"""
    if len(token) < 5:
        return False
    for sig in _segment_signatures(category, token):
        d = dict(sig)
        if strong_only:
            if d.get("strong"):
                return True
            continue
        if category == "gambling":
            if d.get("strong") or d.get("bet"):
                return True
        elif category == "investment":
            if d.get("strong") or (d.get("core") and d.get("clure")):
                return True
        elif category == "crypto":
            if (d.get("strong")
                    or (d.get("brand") and (d.get("weak") or d.get("short")))
                    or d.get("weak", 0) >= 2
                    or (d.get("weak") and d.get("short"))):
                return True
    return False


@lru_cache(maxsize=8192)
def _affix_chain(text: str) -> bool:
    """text 能否完整切成 1 個以上填充詞／數字段（品牌後面黏多個後綴：exchangevip、netbank）。"""
    if not text:
        return False
    if text in _SEG_FILLERS or text.isdigit():
        return True
    return bool(_segment_signatures("affix", text))


# =============================================================================
# 詞庫比對
# =============================================================================

def _semantic_text(c: URLComponents) -> str:
    """中文詞比對對象：IDN 解碼後的 host；路徑只在非內容平台、非白名單時納入。"""
    if c.is_content_platform or c.is_trusted:
        return c.host_unicode
    return c.host_unicode + " " + c.decoded_rest


def _zh_hit(text: str, terms: Iterable[str]) -> bool:
    return any(term in text for term in terms)


def _path_tokens_usable(c: URLComponents) -> bool:
    return not (c.is_content_platform or c.is_trusted)


def _gambling_keyword(c: URLComponents, zh_text: str) -> int:
    g = GAMBLING_TERMS
    for token in c.host_tokens:
        core = _strip_digits(token)
        if token in g["strong_tokens"] or core in g["strong_tokens"]:
            return 1
        if any(term in token for term in g["substring_terms"]):
            return 1
        if _GAMBLING_COMBO_RE.match(token):
            return 1
    if not c.is_trusted and any(_segment_hit("gambling", t) for t in c.host_tokens):
        return 1
    if c.tld in g["tlds"]:
        return 1
    if _path_tokens_usable(c):
        if any(_strip_digits(t) in g["path_tokens"] for t in c.path_tokens):
            return 1
    return int(_zh_hit(zh_text, g["zh"]))


def _investment_keyword(c: URLComponents, zh_text: str) -> int:
    inv = INVESTMENT_TERMS
    core_set, lure_set, clure_set = inv["core_tokens"], inv["lure_tokens"], inv["compound_lure_tokens"]
    tokens = c.host_tokens
    for token in tokens:
        core = _strip_digits(token)
        if token in inv["strong_tokens"] or core in inv["strong_tokens"]:
            return 1
    if not c.is_trusted:
        cores = [i for i, t in enumerate(tokens) if _strip_digits(t) in core_set]
        lures = [i for i, t in enumerate(tokens) if _strip_digits(t) in lure_set]
        if cores and lures and (set(cores) != set(lures) or len(cores) > 1):
            return 1
        for token in tokens:
            stripped = token.rstrip(_DIGITS)
            trailing_digits = len(token) - len(stripped)
            for word in core_set:
                if len(word) >= 4 and stripped.startswith(word) and len(stripped) > len(word):
                    rest = stripped[len(word):]
                    if rest in clure_set or (rest in core_set and trailing_digits >= 2):
                        return 1
                if len(word) >= 4 and stripped == word and trailing_digits >= 2:
                    return 1
            for word in clure_set:
                if stripped.startswith(word) and stripped[len(word):] in core_set:
                    return 1
        if any(_segment_hit("investment", t) for t in tokens):
            return 1
    return int(_zh_hit(zh_text, inv["zh"]))


def _crypto_keyword(c: URLComponents, zh_text: str, brand_hits: List[str]) -> int:
    cr = CRYPTO_TERMS
    if c.is_trusted:
        return 0
    weak_seen: Set[str] = set()
    for token in c.host_tokens:
        core = _strip_digits(token)
        if token in cr["strong_tokens"] or core in cr["strong_tokens"]:
            return 1
        if core in cr["weak_tokens"]:
            weak_seen.add(core)
            continue
        for sub in cr["substring_terms"]:
            if sub in token and len(token) > len(sub):
                weak_seen.add(sub)
    crypto_brand = any(b in _CRYPTO_BRANDS for b in brand_hits)
    sld_plain = re.sub(r"[^a-z0-9]", "", c.sld_folded.lower())
    if (sld_plain and _CRYPTO_EX_RE.match(sld_plain) and sld_plain not in cr["ex_exceptions"]
            and (c.tld_risk >= 1 or weak_seen or crypto_brand)):
        return 1
    if len(weak_seen) >= 2:
        return 1
    if weak_seen and (c.tld_risk >= 1 or crypto_brand):
        return 1
    # v7.1 黏字切分（kucoinexchangevip、wakuangvip）；官方網域上的子網域不採計品牌組合
    if c.registered_domain not in _ALL_OFFICIAL_DOMAINS and any(_segment_hit("crypto", t) for t in c.host_tokens):
        return 1
    if _path_tokens_usable(c):
        strong_path = cr["strong_tokens"] - {"ex", "otc", "dex", "cex", "p2p"}
        if any(_strip_digits(t) in strong_path for t in c.path_tokens):
            return 1
    return int(_zh_hit(zh_text, cr["zh"]))


def _has_scam_word(c: URLComponents, zh_text: str) -> int:
    tokens: Tuple[str, ...] = c.host_tokens
    if _path_tokens_usable(c):
        tokens = tokens + c.path_tokens
    for token in tokens:
        if token in SCAM_WORD_TOKENS or _strip_digits(token) in SCAM_WORD_TOKENS:
            return 1
    # hostname 以 vip/vvip/svip 開頭的複合字（vvipstoreonline、vipstock）
    if any(re.match(r"^v?s?vip[a-z]", t) for t in c.host_tokens):
        return 1
    for seq in (c.host_tokens, c.path_tokens if _path_tokens_usable(c) else ()):
        pairs = set(zip(seq, seq[1:]))
        if any(bigram in pairs for bigram in SCAM_WORD_BIGRAMS):
            return 1
    return int(_zh_hit(zh_text, SCAM_WORD_ZH))


_DOMAIN_KEYWORDS: FrozenSet[str] = frozenset(
    set(SCAM_WORD_TOKENS)
    | set(GAMBLING_TERMS["strong_tokens"])
    | set(INVESTMENT_TERMS["strong_tokens"])
    | (set(CRYPTO_TERMS["strong_tokens"]) - {"ex"})
)


def _suspicious_keyword_in_domain(c: URLComponents) -> int:
    for token in c.host_tokens:
        if token in _DOMAIN_KEYWORDS or _strip_digits(token) in _DOMAIN_KEYWORDS:
            return 1
        if any(term in token for term in GAMBLING_TERMS["substring_terms"]):
            return 1
    if not c.is_trusted and any(_segment_hit(cat, t, strong_only=True)
                                for t in c.host_tokens for cat in ("gambling", "investment", "crypto")):
        return 1
    host_u = c.host_unicode
    if host_u != c.hostname:
        for terms in (GAMBLING_TERMS["zh"], INVESTMENT_TERMS["zh"], CRYPTO_TERMS["zh"], SCAM_WORD_ZH):
            if _zh_hit(host_u, terms):
                return 1
    return 0


def _random_like_sld(sld: str) -> bool:
    """
    SLD 是否像隨機字串：無母音；或「罕見子音連接」比例 ≥ 0.3；
    或罕見比例 ≥ 0.15 且母音比例 < 0.3 且（連續子音 ≥ 3 或 entropy ≥ 2.75）。
    英文品牌（kingston、longchamp、porsche）的子音連接多為常見組合，不會命中。
    """
    letters = re.sub(r"[^a-z]", "", sld.lower())
    n = len(letters)
    if n < 5:
        return False
    vowels = sum(ch in "aeiou" for ch in letters)
    if vowels == 0:
        return True
    ratio = vowels / n
    run = longest = 0
    for ch in letters:
        if ch in "aeiouy":
            run = 0
        else:
            run += 1
            longest = max(longest, run)
    rare = sum(
        1 for a, b in zip(letters, letters[1:])
        if a not in "aeiouy" and b not in "aeiouy" and (a + b) not in _COMMON_CC_BIGRAMS
    )
    rare_ratio = rare / (n - 1)
    if rare_ratio >= 0.3:
        return True
    if rare_ratio >= 0.15 and ratio < 0.3 and (longest >= 3 or calc_entropy(letters) >= 2.75):
        return True
    return False


def _has_meaningful_word(sld: str) -> bool:
    plain = re.sub(r"[^a-z]", "", sld.lower())
    tokens = set(_TOKEN_SPLIT_RE.split(sld.lower()))
    for word in MEANINGFUL_WORDS:
        if len(word) >= 4 and word in plain:
            return True
        if len(word) <= 3 and word in tokens:
            return True
    return False


# =============================================================================
# v7.1：sld_randomness（網域名稱可讀性；字元 trigram 平均驚奇度）
# =============================================================================
# 統計表 char_ngram_table.json 由 tools/build_ngram_table.py 以離線語料產生（Python 標準庫註解／
# docstring 英文字、程式化產生的漢語拼音音節、少量品牌 token；不含任何訓練／驗收網址）。
# 計算（只看 SLD，官方網域也照算）：
#   1. 以非英文字母切成字母段，長度 1 的字母段不計（kyoto-u 的 u、tw1 的數字旁單字母）。
#   2. 每段以「最小描述長度」斷詞（複合字 kfc|club、nbc|news 的接縫不算罕見），每多一段加 2 bits；
#      另試「去母音命名」變體（tumblr → tumbler、scribd → scribed），取平均驚奇度較低者。
#   3. 單一字元驚奇度上限 14 bits（一兩個罕見接縫不主導平均）；平均驚奇度以 logistic 轉成 0～1
#      （5.0 bits/字元 → 0.5；英文／拼音約 3 bits、均勻亂數約 10 bits）。
#   4. 綜合：0.85 × n-gram + 0.10 × 最長連續子音（≥4 起算）+ 0.05 × 數字字母交錯（≥2 次起算），
#      再乘上長度信心（字母數 (n-2)/3.5，上限 1；4 字母縮寫如 hncb 最多 0.57）。
# 統計表讀不到時退回「罕見子音連接比例＋母音不足」啟發式（不丟例外）。

_NGRAM_TABLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "char_ngram_table.json")
_NGRAM_TABLE_FORMAT = "truthmark-char-ngram/1"
_SURPRISAL_CAP = 14.0
_SURPRISAL_MID = 5.0
_SURPRISAL_SCALE = 0.6
_SPLIT_PENALTY_BITS = 2.0
_MAX_SPLIT_RUN = 30
_RANDOMNESS_WEIGHTS = (0.85, 0.10, 0.05)   # n-gram、最長連續子音、數字字母交錯
# newly_registered_like 的 sld_randomness 門檻：自建合法網域清單誤判約 1%（舊版 8.5%）、
# 真實詐騙（data_clean）觸發約 26%（舊版 31%），數據見 model_report.feature_experiments
NEWLY_REGISTERED_THRESHOLD = 0.70


def _load_ngram_table(path: str = _NGRAM_TABLE_PATH
                      ) -> Optional[Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]]:
    """讀取字元 n-gram 統計表；任何錯誤都回傳 None（改用啟發式）。"""
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict) or data.get("format") != _NGRAM_TABLE_FORMAT:
            return None
        bi = {str(k): float(v) for k, v in data.get("bi", {}).items()}
        tri = {str(k): float(v) for k, v in data.get("tri", {}).items()}
        bo = {str(k): float(v) for k, v in data.get("bo", {}).items()}
        if len(bi) < 26 * 27:
            return None
        return bi, tri, bo
    except Exception:  # noqa: BLE001
        return None


_NGRAM = _load_ngram_table()
NGRAM_TABLE_LOADED = _NGRAM is not None


@lru_cache(maxsize=32768)
def _segment_surprisal(seg: str) -> float:
    """「^^seg$」的總驚奇度（bits，每字元上限 _SURPRISAL_CAP）。"""
    bi, tri, bo = _NGRAM  # type: ignore[misc]
    padded = "^^" + seg + "$"
    total = 0.0
    for i in range(2, len(padded)):
        a, b, ch = padded[i - 2], padded[i - 1], padded[i]
        lp = tri.get(a + b + ch)
        if lp is None:
            lp = bo.get(a + b, 0.0) + bi.get(b + ch, -_SURPRISAL_CAP)
        total += min(-lp, _SURPRISAL_CAP)
    return total


def _best_split_surprisal(run: str) -> Tuple[float, int]:
    """把字母段切成 ≥2 字的片段使「總驚奇度 + 斷點懲罰」最小，回傳 (總 bits, 片段數)。"""
    n = len(run)
    if n < 4 or n > _MAX_SPLIT_RUN:
        return _segment_surprisal(run), 1
    best: List[Tuple[float, int]] = [(math.inf, 0)] * (n + 1)
    best[0] = (0.0, 0)
    for i in range(2, n + 1):
        for j in range(0, i - 1):
            prev, k = best[j]
            if prev == math.inf:
                continue
            cost = prev + _segment_surprisal(run[j:i]) + (_SPLIT_PENALTY_BITS if j else 0.0)
            if cost < best[i][0]:
                best[i] = (cost, k + 1)
    return best[n]


def _naming_variants(run: str) -> Tuple[str, ...]:
    """常見「去母音」品牌命名（tumblr、flickr、sprinklr、scribd）補回 e 的變體。"""
    if len(run) >= 5 and re.search(r"[bcdfghjklmnpqrstvwxz][rd]$", run):
        return run, run[:-1] + "e" + run[-1]
    return (run,)


def _fallback_lexical(letters: str) -> float:
    """沒有統計表時的可讀性啟發式：罕見子音連接比例＋母音不足（0～1）。"""
    n = len(letters)
    if n < 2:
        return 0.0
    vowels = sum(ch in "aeiouy" for ch in letters)
    rare = sum(
        1 for a, b in zip(letters, letters[1:])
        if a not in "aeiouy" and b not in "aeiouy" and (a + b) not in _COMMON_CC_BIGRAMS
    )
    rare_ratio = rare / (n - 1)
    vowel_deficit = max(0.0, 0.38 - vowels / n) / 0.38
    return min(1.0, max(0.0, (rare_ratio - 0.1) / 0.35) * 0.7 + 0.3 * vowel_deficit)


@lru_cache(maxsize=8192)
def _sld_randomness_cached(sld: str) -> float:
    s = sld.lower()
    runs = re.findall(r"[a-z]+", s)
    letters = sum(len(r) for r in runs)
    if letters == 0:
        return 0.0
    if _NGRAM is not None:
        scored = [r for r in runs if len(r) >= 2] or runs
        total_bits = 0.0
        total_symbols = 0
        for run in scored:
            best_h, best_syms = math.inf, len(run) + 1
            for variant in _naming_variants(run):
                bits, pieces = _best_split_surprisal(variant)
                h = bits / (len(variant) + pieces)
                if h < best_h:
                    best_h, best_syms = h, len(run) + pieces
            total_bits += best_h * best_syms
            total_symbols += best_syms
        mean_bits = total_bits / max(total_symbols, 1)
        lexical = 1.0 / (1.0 + math.exp(-(mean_bits - _SURPRISAL_MID) / _SURPRISAL_SCALE))
    else:
        lexical = _fallback_lexical("".join(runs))
    longest = run_len = 0
    for ch in s:
        if "a" <= ch <= "z" and ch not in "aeiouy":
            run_len += 1
            longest = max(longest, run_len)
        else:
            run_len = 0
    alnum = re.sub(r"[^a-z0-9]", "", s)
    transitions = sum(1 for a, b in zip(alnum, alnum[1:]) if a.isdigit() != b.isdigit())
    w_lex, w_cons, w_alt = _RANDOMNESS_WEIGHTS
    value = (w_lex * lexical
             + w_cons * min(max((longest - 3) / 4.0, 0.0), 1.0)
             + w_alt * min(max((transitions - 1) / 2.0, 0.0), 1.0))
    confidence = min(1.0, max(letters - 2, 0) / 3.5)
    return round(min(1.0, max(0.0, value)) * confidence, 4)


def sld_randomness(sld: Any) -> float:
    """
    網域名稱（SLD）像「隨機字串」的程度（0～1；越接近 1 越像 xkqplbwz 這種無法發音的亂碼）。
    只看 SLD 本身，不做白名單判斷。永不丟例外。
    """
    try:
        text = str(sld or "")
        if len(text) > 63:
            text = text[:63]
        return _sld_randomness_cached(text)
    except Exception:  # noqa: BLE001
        return 0.0


# =============================================================================
# 主要特徵擷取
# =============================================================================

def _compute(c: URLComponents) -> Dict[str, float]:
    body = c.body
    body_len = len(body)
    # v7.1：dot_count／digit_ratio／special_chars 只算主機名稱（去 www.；IP 用原字串、IPv6 不含括號），
    # 消除「路徑寫法」造成的機率漂移（實驗見 model_report.feature_experiments.char_scope）
    host_text = c.host_full if c.is_ip else c.hostname
    host_digits = sum(ch.isdigit() for ch in host_text)
    subdomain_depth = len([lab for lab in c.subdomain.split(".") if lab]) if c.subdomain else 0
    path_for_len = "" if c.path == "/" else c.path
    path_depth = len([seg for seg in c.path.split("/") if seg])
    try:
        query_params = len({k for k, _ in parse_qsl(c.query, keep_blank_values=True)})
    except Exception:  # noqa: BLE001
        query_params = 0

    zh_text = _semantic_text(c)
    host = c.hostname
    tld_risk = c.tld_risk

    # 短網址 / 免費架站 / 通道 / 雲端
    has_shortener = int(c.registered_domain in SHORTENER_DOMAINS or host in SHORTENER_DOMAINS)
    free_hosting = 0
    if host and not c.is_ip:
        if host in FREE_HOSTING_HOSTS:
            free_hosting = 1
        elif any(host.endswith(s) and len(host) > len(s) for s in FREE_HOSTING_SUFFIXES):
            free_hosting = 1
        elif any(rx.search(host) for rx in _FREE_HOSTING_RES):
            free_hosting = 1
    tunnel = 0
    if host and not c.is_ip:
        if any(host.endswith(s) and len(host) > len(s) for s in TUNNEL_SUFFIXES):
            tunnel = 1
        elif host in TUNNEL_HOSTS and re.search(r"/ip[fn]s/", c.rest, re.I):
            tunnel = 1
    if _IPFS_PATH_RE.search(c.rest):
        tunnel = 1
    cloud_hosting = int(bool(host) and _endswith_suffix(host, CLOUD_HOSTING_SUFFIXES))

    # 社群邀請、路由
    invite_target = host + c.path
    social_invite = int(any(rx.search(invite_target) for rx in _SOCIAL_INVITE_RES))
    rest_decoded = c.decoded_rest
    scam_route = int(
        c.scheme == "itms-services"
        or any(rx.search(rest_decoded) for rx in _SCAM_ROUTE_RES)
    )

    # 關鍵詞
    gambling_kw = _gambling_keyword(c, zh_text)
    investment_kw = _investment_keyword(c, zh_text)
    brand_hits = _brand_hits(c, c.host_tokens)
    crypto_kw = _crypto_keyword(c, zh_text, brand_hits)
    brand_impersonation = int(bool(brand_hits))
    sld_brand_hits = _brand_hits(c, c.sld_tokens) if c.sld_tokens else []
    brand_in_sld = int(bool(sld_brand_hits) and tld_risk >= 1)

    punycode = int("xn--" in c.host_full or not c.host_full.isascii())
    sld_digits = sum(ch.isdigit() for ch in c.sld)

    # 短品牌近似（line≈ligne）需搭配的「其他風險訊號」；SLD 含數字不算（避免 line6.com）
    other_signal = bool(
        tld_risk >= 1 or free_hosting or tunnel or gambling_kw or investment_kw or crypto_kw
        or scam_route or punycode
    )
    lev_dist = _effective_brand_distance(c, other_signal)

    # 博弈幸運數字（純數字 SLD 如 1688、8591 不算）
    gambling_number = 0
    if host and not c.is_ip:
        sld_plain = re.sub(r"[^a-z0-9]", "", c.sld)
        labels = [lab for lab in c.subdomain.split(".") if lab]
        if c.sld and not sld_plain.isdigit():
            labels.append(c.sld)
        for label in labels:
            if any(_LUCKY_RUN_RE.search(run) for run in re.findall(r"\d+", label)):
                gambling_number = 1
                break
        if not gambling_number and any(_GAMBLING_COMBO_RE.match(t) for t in c.host_tokens):
            gambling_number = 1

    # 行動版誘導
    first_label = c.subdomain.split(".", 1)[0] if c.subdomain else ""
    segments = [seg.lower() for seg in c.path.split("/") if seg]
    mobile_lure = int(
        first_label in _MOBILE_HOST_PREFIXES
        or any(seg in ("h5", "wap") for seg in segments)
        or (bool(segments) and segments[0] == "m")
    )

    # v7.1：隨機度改用字元 n-gram 可讀性（sld_randomness）；統計表讀不到時退回舊版子音啟發式
    sld_rand = sld_randomness(c.sld_folded) if (c.sld and not c.is_ip) else 0.0
    random_like = (sld_rand >= NEWLY_REGISTERED_THRESHOLD) if _NGRAM is not None else _random_like_sld(c.sld_folded)
    sld_letters_ok = bool(c.sld) and not c.is_ip and not c.is_trusted
    newly_registered = int(
        sld_letters_ok
        and random_like
        and not brand_hits
        and not any(_brand_token_match(t)[0] for t in c.sld_tokens)
        and not _has_meaningful_word(c.sld_folded)
    )

    query_frag = (c.query + "&" + c.fragment).lower()

    features: Dict[str, float] = {
        "url_length": body_len,
        "domain_length": len(host),
        "path_length": len(path_for_len),
        "query_length": len(c.query),
        "digit_ratio": host_digits / max(len(host_text), 1),
        "hyphen_count": body.count("-"),
        "dot_count": host_text.count("."),
        "special_chars": sum(1 for ch in host_text if not (ch.isascii() and ch.isalnum())),
        "subdomain_depth": subdomain_depth,
        "path_depth": path_depth,
        "query_params": query_params,
        "is_https": int(c.scheme == "https" or not c.has_scheme),
        "is_ip_address": int(c.is_ip),
        "suspicious_tld": int(tld_risk >= 1),
        "has_scam_word": _has_scam_word(c, zh_text),
        "brand_in_sld": brand_in_sld,
        "has_utm": int("utm_" in query_frag),
        "has_gclid": int(bool(_HAS_GCLID_RE.search("&" + query_frag))),
        "double_http": int(len(re.findall(r"https?://", c.decoded_url, re.I)) >= 2),
        "long_domain": int(len(host) > 30),
        "domain_entropy": calc_entropy(c.sld),
        "has_shortener": has_shortener,
        "gambling_number_pattern": gambling_number,
        "brand_typo_like": _brand_typo_like(c, lev_dist),
        "cloud_hosting": cloud_hosting,
        "mobile_lure_path": mobile_lure,
        "suspicious_keyword_in_domain": _suspicious_keyword_in_domain(c),
        "many_subdomains": int(subdomain_depth >= 3),
        "contains_percent_encoding": int(bool(_PERCENT_RE.search(c.canonical_url))),
        "levenshtein_brand_dist": lev_dist,
        "newly_registered_like": newly_registered,
        "tld_risk_level": tld_risk,
        "gambling_keyword": gambling_kw,
        "investment_lure_keyword": investment_kw,
        "crypto_exchange_lure": crypto_kw,
        "brand_impersonation": brand_impersonation,
        "free_hosting_platform": free_hosting,
        "tunnel_or_ephemeral_host": tunnel,
        "social_invite_link": social_invite,
        "punycode_domain": punycode,
        "url_has_at_symbol": int(c.has_userinfo),
        "non_standard_port": int(bool(c.port_text) and c.port not in (80, 443)),
        "sld_digit_count": sld_digits,
        "sld_length": len(c.sld),
        "path_scam_route": scam_route,
        "sld_randomness": sld_rand,
    }
    return features


@lru_cache(maxsize=4096)
def _compute_feature_dict_cached(url: str) -> Tuple[Tuple[str, float], ...]:
    """
    回傳 tuple-of-tuples（不可變）供 lru_cache 共用；呼叫端每次都組成新的 dict，
    避免任何呼叫端修改回傳值而汙染快取。
    """
    try:
        merged = _compute(_components_cached(url))
        return tuple((name, merged.get(name, FEATURE_DEFAULTS[name])) for name in FEATURE_NAMES)
    except Exception:  # noqa: BLE001
        return tuple((name, FEATURE_DEFAULTS[name]) for name in FEATURE_NAMES)


def _compute_feature_dict(url: Any) -> Dict[str, float]:
    try:
        key = url if isinstance(url, str) else ("" if url is None else str(url))
        return dict(_compute_feature_dict_cached(key[:_MAX_INPUT_LENGTH]))
    except Exception:  # noqa: BLE001
        return dict(FEATURE_DEFAULTS)


def extract_feature_dict(url: Any) -> Dict[str, float]:
    """回傳全部 len(FEATURE_NAMES)（v7.1 為 46）個特徵的 dict（永不丟例外；每次回傳新的 dict）。"""
    return _compute_feature_dict(url)


def extract_features(url: Any) -> List[float]:
    """將 URL 轉成 len(FEATURE_NAMES) 維（v7.1 為 46）特徵向量，順序與 FEATURE_NAMES 完全一致。"""
    feature_dict = _compute_feature_dict(url)
    return [feature_dict[name] for name in FEATURE_NAMES]


# =============================================================================
# 特徵規格與門檻（門檻唯一來源：前端與 explain 都以 assess_feature() 為準）
# =============================================================================

FEATURE_SPECS: Dict[str, Dict[str, str]] = {
    "url_length": {"zh": "網址長度", "unit": "字元", "kind": "count",
                   "desc": "去掉 http(s):// 與 www. 後的網址總長度。過長的網址可能夾帶混淆參數。"},
    "domain_length": {"zh": "網域長度", "unit": "字元", "kind": "count",
                      "desc": "主機名稱（去 www.）的字元數，超過 30 字元常見於混淆型網域。"},
    "path_length": {"zh": "路徑長度", "unit": "字元", "kind": "count",
                    "desc": "網址路徑（不含參數與 # 片段）的長度。"},
    "query_length": {"zh": "參數長度", "unit": "字元", "kind": "count",
                     "desc": "? 之後查詢參數的長度，過長可能夾帶跳轉或追蹤資料。"},
    "digit_ratio": {"zh": "數字比例", "unit": "", "kind": "ratio",
                    "desc": "網域名稱（主機名稱，不含路徑與參數）中數字字元所占比例，隨機數字常見於大量註冊的詐騙網域。"},
    "hyphen_count": {"zh": "連字號數量", "unit": "個", "kind": "count",
                     "desc": "網址中「-」的數量，常見於「品牌-tw-login」這類拼接仿冒。"},
    "dot_count": {"zh": "網域點號數", "unit": "個", "kind": "count",
                  "desc": "網域名稱（主機名稱，不含 www. 與路徑）中「.」的數量，過多可能是多層子網域混淆。"},
    "special_chars": {"zh": "網域符號數", "unit": "個", "kind": "count",
                      "desc": "網域名稱（主機名稱，不含 www. 與路徑）中非英數字元（點號、連字號等）的數量。"},
    "subdomain_depth": {"zh": "子網域層數", "unit": "層", "kind": "count",
                        "desc": "依公開後綴清單切分後的子網域層數（www 不計）。"},
    "path_depth": {"zh": "路徑層數", "unit": "層", "kind": "count",
                   "desc": "網址路徑以 / 分隔的層數。"},
    "query_params": {"zh": "參數個數", "unit": "個", "kind": "count",
                     "desc": "查詢參數的種類數量。"},
    "is_https": {"zh": "HTTPS 加密", "unit": "", "kind": "bool",
                 "desc": "明確使用 https 或未指定協定為 1；明確使用 http 為 0。"},
    "is_ip_address": {"zh": "IP 位址網域", "unit": "", "kind": "bool",
                      "desc": "網址直接使用 IP 位址而非網域名稱。"},
    "suspicious_tld": {"zh": "可疑頂級域名", "unit": "", "kind": "bool",
                       "desc": "頂級域名屬於常被濫用的清單（TLD 風險等級 ≥ 1）。"},
    "has_scam_word": {"zh": "詐騙誘因詞", "unit": "", "kind": "bool",
                      "desc": "解碼後網址含投資、獎勵、加密貨幣等誘因詞（新聞與政府網站路徑不計）。"},
    "brand_in_sld": {"zh": "品牌＋可疑 TLD", "unit": "", "kind": "bool",
                     "desc": "網域名稱含知名品牌字樣、使用可疑頂級域名，且不是該品牌官方網域。"},
    "has_utm": {"zh": "UTM 追蹤參數", "unit": "", "kind": "bool",
                "desc": "網址帶有 utm_ 行銷追蹤參數（中性資訊）。"},
    "has_gclid": {"zh": "廣告點擊 ID", "unit": "", "kind": "bool",
                  "desc": "網址帶有 gclid／msclkid 等搜尋廣告點擊 ID（中性資訊）。"},
    "double_http": {"zh": "內嵌網址", "unit": "", "kind": "bool",
                    "desc": "網址中出現兩次以上 http(s)://，可能是跳轉或偽裝。"},
    "long_domain": {"zh": "超長網域", "unit": "", "kind": "bool",
                    "desc": "主機名稱超過 30 字元。"},
    "domain_entropy": {"zh": "網域隨機度", "unit": "", "kind": "score",
                       "desc": "網域名稱的資訊熵；越高代表字元越隨機。"},
    "has_shortener": {"zh": "短網址", "unit": "", "kind": "bool",
                      "desc": "使用 bit.ly、reurl.cc 等短網址服務，可能隱藏真實目的地。"},
    "gambling_number_pattern": {"zh": "博弈幸運數字", "unit": "", "kind": "bool",
                                "desc": "網域含 168、888、666 等博弈常見數字組合。"},
    "brand_typo_like": {"zh": "品牌拼字變形", "unit": "", "kind": "bool",
                        "desc": "網域疑似品牌名稱的拼字變形（sh0pee、faebook、walmar…）且非官方。"},
    "cloud_hosting": {"zh": "雲端／CDN 主機", "unit": "", "kind": "bool",
                      "desc": "網址架在雲端或 CDN 基礎設施主機（CloudFront、Azure、pages.dev…）。"},
    "mobile_lure_path": {"zh": "行動版落地頁", "unit": "", "kind": "bool",
                         "desc": "主機名稱首段為 m／h5／wap／app／sj／mobile，或路徑含 /h5、/wap 段（或以 /m 開頭）。"},
    "suspicious_keyword_in_domain": {"zh": "網域含詐騙詞", "unit": "", "kind": "bool",
                                     "desc": "詐騙、博弈、假投資或加密貨幣強詞直接出現在網域名稱中。"},
    "many_subdomains": {"zh": "多層子網域", "unit": "", "kind": "bool",
                        "desc": "子網域層數達 3 層以上。"},
    "contains_percent_encoding": {"zh": "百分比編碼", "unit": "", "kind": "bool",
                                  "desc": "網址含 %XX 編碼（中文網址常見，屬中性資訊）。"},
    "levenshtein_brand_dist": {"zh": "品牌相似距離", "unit": "", "kind": "score",
                               "desc": "與非官方品牌名稱的編輯距離：0 = 同名但非官方、1～2 = 拼字近似、99 = 無相似。"},
    "newly_registered_like": {"zh": "疑似隨機新網域", "unit": "", "kind": "bool",
                              "desc": "網域名稱為無意義的隨機字元組合，符合即用即棄型詐騙網域。"},
    "tld_risk_level": {"zh": "TLD 風險等級", "unit": "級", "kind": "level",
                       "desc": "頂級域名風險：0 = 一般、1 = 常被濫用、2 = 高度濫用／即用即棄。"},
    "gambling_keyword": {"zh": "博弈詞", "unit": "", "kind": "bool",
                         "desc": "含博弈英文、中文或拼音（casino、百家樂、yule、bocai、pujing…）。"},
    "investment_lure_keyword": {"zh": "假投資詞", "unit": "", "kind": "bool",
                                "desc": "含飆股、投顧、老師帶單、IPO 抽籤、AI 選股、stock-vip 等假投資字詞。"},
    "crypto_exchange_lure": {"zh": "假交易所／錢包詞", "unit": "", "kind": "bool",
                             "desc": "含 USDT、OTC、staking、airdrop、-ex 等假交易所或錢包盜取字詞。"},
    "brand_impersonation": {"zh": "品牌冒用", "unit": "", "kind": "bool",
                            "desc": "網域使用知名品牌、銀行、交易所或政府名稱，但不屬於該品牌官方網域。"},
    "free_hosting_platform": {"zh": "免費架站平台", "unit": "", "kind": "bool",
                              "desc": "架在 web.app、pages.dev、github.io、wixsite 等免費架站或暫存平台。"},
    "tunnel_or_ephemeral_host": {"zh": "臨時通道主機", "unit": "", "kind": "bool",
                                 "desc": "使用 ngrok、trycloudflare、loca.lt 或 IPFS 閘道等臨時網址。"},
    "social_invite_link": {"zh": "社群群組邀請", "unit": "", "kind": "bool",
                           "desc": "LINE／Telegram／WhatsApp／Discord 群組或官方帳號邀請連結。"},
    "punycode_domain": {"zh": "國際化網域", "unit": "", "kind": "bool",
                        "desc": "網域含 xn--（punycode）或非 ASCII 字元，可能是同形字仿冒。"},
    "url_has_at_symbol": {"zh": "@ 帳號偽裝", "unit": "", "kind": "bool",
                          "desc": "網址在網域前放了 @ 帳號欄位，實際連往 @ 後的網域。"},
    "non_standard_port": {"zh": "非標準埠", "unit": "", "kind": "bool",
                          "desc": "網址明確指定 80／443 以外的連接埠。"},
    "sld_digit_count": {"zh": "網域數字個數", "unit": "個", "kind": "count",
                        "desc": "網域主要名稱（SLD）中的數字個數。"},
    "sld_length": {"zh": "網域名稱長度", "unit": "字元", "kind": "count",
                   "desc": "網域主要名稱（SLD）的字元數。"},
    "path_scam_route": {"zh": "詐騙 App 路由", "unit": "", "kind": "bool",
                        "desc": "含 #/pages/、/h5/#/、邀請碼參數、download.html 等假投資 App 網頁路由。"},
    "sld_randomness": {"zh": "網域亂碼度", "unit": "", "kind": "score",
                       "desc": "以英文與漢語拼音的字母組合統計，衡量網域名稱是否像無法發音的亂碼"
                               "（0 = 像一般字詞、1 = 像 xkqplbwz）；≥ 0.7 視為疑似亂碼。只看網域名稱本身。"},
}

# 數值型門檻（value ≥ warn → warn；value ≥ alert → alert）
FEATURE_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "url_length": {"warn": 120},
    "domain_length": {"warn": 31},
    "path_length": {"warn": 100},
    "query_length": {"warn": 200},
    "digit_ratio": {"warn": 0.3},
    "hyphen_count": {"warn": 4},
    "dot_count": {"warn": 5},        # v7.1 起只算主機名稱（≥6 段）
    "special_chars": {"warn": 7},    # v7.1 起只算主機名稱（點號＋連字號）
    "subdomain_depth": {"warn": 3},
    "path_depth": {"warn": 6},
    "query_params": {"warn": 8},
    "domain_entropy": {"warn": 3.5, "neutral": 2.5},
    "sld_digit_count": {"warn": 5},
    "sld_length": {"warn": 20},
    "sld_randomness": {"warn": NEWLY_REGISTERED_THRESHOLD, "neutral": 0.5},
}

# (超過門檻的狀態, 文字, 未超過的狀態, 文字)
_NUMERIC_LABELS: Dict[str, Tuple[str, str, str, str]] = {
    "url_length": ("warn", "偏長", "safe", "正常"),
    "domain_length": ("warn", "網域偏長", "safe", "正常"),
    "path_length": ("warn", "路徑很長", "neutral", "一般"),
    "query_length": ("warn", "參數很長", "neutral", "一般"),
    "digit_ratio": ("warn", "數字偏多", "safe", "正常"),
    "hyphen_count": ("warn", "連字號偏多", "safe", "正常"),
    "dot_count": ("warn", "點號偏多", "safe", "正常"),
    "special_chars": ("warn", "符號偏多", "safe", "正常"),
    "subdomain_depth": ("warn", "子網域過多", "safe", "正常"),
    "path_depth": ("warn", "路徑較深", "neutral", "一般"),
    "query_params": ("warn", "參數較多", "neutral", "一般"),
    "sld_digit_count": ("warn", "數字偏多", "safe", "正常"),
    "sld_length": ("warn", "名稱偏長", "neutral", "一般"),
}

# bool 特徵：(值為 1 的狀態, 文字, 值為 0 的狀態, 文字)
_BOOL_LABELS: Dict[str, Tuple[str, str, str, str]] = {
    "is_https": ("safe", "HTTPS／未指定", "warn", "未加密 HTTP"),
    "is_ip_address": ("alert", "IP 位址", "safe", "一般網域"),
    "suspicious_tld": ("warn", "常被濫用 TLD", "safe", "一般 TLD"),
    "has_scam_word": ("warn", "含誘因詞", "safe", "未偵測"),
    "brand_in_sld": ("alert", "品牌＋可疑 TLD", "safe", "未偵測"),
    "has_utm": ("neutral", "含行銷追蹤", "neutral", "無"),
    "has_gclid": ("neutral", "含廣告點擊 ID", "neutral", "無"),
    "double_http": ("warn", "內嵌網址", "safe", "無"),
    "long_domain": ("warn", "網域過長", "safe", "正常"),
    "has_shortener": ("warn", "短網址", "safe", "非短網址"),
    "gambling_number_pattern": ("warn", "博弈數字", "safe", "未偵測"),
    "brand_typo_like": ("alert", "疑似拼字仿冒", "safe", "未偵測"),
    "cloud_hosting": ("warn", "雲端／CDN 主機", "safe", "一般主機"),
    "mobile_lure_path": ("warn", "行動版落地頁", "safe", "未偵測"),
    "suspicious_keyword_in_domain": ("alert", "網域含詐騙詞", "safe", "未偵測"),
    "many_subdomains": ("warn", "子網域過多", "safe", "正常"),
    "contains_percent_encoding": ("neutral", "含編碼", "neutral", "無"),
    "newly_registered_like": ("warn", "疑似隨機新網域", "safe", "正常"),
    "gambling_keyword": ("alert", "博弈詞", "safe", "未偵測"),
    "investment_lure_keyword": ("alert", "假投資詞", "safe", "未偵測"),
    "crypto_exchange_lure": ("alert", "假交易所詞", "safe", "未偵測"),
    "brand_impersonation": ("alert", "品牌冒用", "safe", "未偵測"),
    "free_hosting_platform": ("warn", "免費架站平台", "safe", "未偵測"),
    "tunnel_or_ephemeral_host": ("alert", "臨時通道", "safe", "未偵測"),
    "social_invite_link": ("warn", "群組邀請連結", "safe", "未偵測"),
    "punycode_domain": ("warn", "國際化網域", "safe", "未偵測"),
    "url_has_at_symbol": ("alert", "@ 偽裝", "safe", "未偵測"),
    "non_standard_port": ("warn", "非標準埠", "safe", "標準埠"),
    "path_scam_route": ("warn", "疑似詐騙 App 路由", "safe", "未偵測"),
}


def _to_number(value: Any) -> float:
    try:
        number = float(value)
        if math.isnan(number):
            return 0.0
        return number
    except Exception:  # noqa: BLE001
        return 0.0


def assess_feature(name: str, value: Any) -> Dict[str, str]:
    """
    單一特徵的狀態評估（前端顯示與 explain_features 共用的唯一門檻來源）。
    status：alert = 高風險、warn = 需注意、safe = 正常、neutral = 中性資訊。
    """
    v = _to_number(value)
    if name in _BOOL_LABELS:
        on_status, on_text, off_status, off_text = _BOOL_LABELS[name]
        return {"status": on_status, "text": on_text} if v >= 1 else {"status": off_status, "text": off_text}
    if name == "levenshtein_brand_dist":
        if v == 0:
            return {"status": "alert", "text": "與品牌同名但非官方"}
        if v == 1:
            return {"status": "alert", "text": "高度相似品牌"}
        if v == 2:
            return {"status": "warn", "text": "相似品牌"}
        return {"status": "safe", "text": "無相似品牌"}
    if name == "tld_risk_level":
        if v >= 2:
            return {"status": "alert", "text": "高度濫用 TLD"}
        if v >= 1:
            return {"status": "warn", "text": "常被濫用 TLD"}
        return {"status": "safe", "text": "一般 TLD"}
    if name == "domain_entropy":
        th = FEATURE_THRESHOLDS["domain_entropy"]
        if v >= th["warn"]:
            return {"status": "warn", "text": "隨機性高"}
        if v >= th["neutral"]:
            return {"status": "neutral", "text": "中等"}
        return {"status": "safe", "text": "規律"}
    if name == "sld_randomness":
        th = FEATURE_THRESHOLDS["sld_randomness"]
        if v >= th["warn"]:
            return {"status": "warn", "text": "疑似亂碼"}
        if v >= th["neutral"]:
            return {"status": "neutral", "text": "略不規則"}
        return {"status": "safe", "text": "可讀"}
    if name in _NUMERIC_LABELS:
        th = FEATURE_THRESHOLDS[name]
        over_status, over_text, under_status, under_text = _NUMERIC_LABELS[name]
        if "alert" in th and v >= th["alert"]:
            return {"status": "alert", "text": over_text}
        if v >= th["warn"]:
            return {"status": over_status, "text": over_text}
        return {"status": under_status, "text": under_text}
    return {"status": "neutral", "text": "資訊"}


def assess_all(feature_dict: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
    """對 feature_dict 內每個特徵呼叫 assess_feature()；未知特徵一律 neutral。"""
    result: Dict[str, Dict[str, str]] = {}
    for name, value in (feature_dict or {}).items():
        try:
            result[name] = assess_feature(name, value)
        except Exception:  # noqa: BLE001
            result[name] = {"status": "neutral", "text": "資訊"}
    return result


# =============================================================================
# 可讀說明
# =============================================================================

_EXPLAIN_MESSAGES: Dict[str, str] = {
    "url_length": "網址較長（{v:.0f} 字元）且沒有常見廣告追蹤參數（utm_、gclid），可能存在混淆或隱藏資訊。",
    "path_length": "網址路徑很長（{v:.0f} 字元），可能夾帶混淆或追蹤資訊。",
    "query_length": "網址參數很長（{v:.0f} 字元），可能夾帶編碼後的跳轉或追蹤資料。",
    "digit_ratio": "網域名稱中數字比例偏高（約 {pct}），隨機數字常見於大量註冊的詐騙網域。",
    "hyphen_count": "網址含較多連字號（{v:.0f} 個），常見於「品牌-tw-login」這類拼接仿冒。",
    "dot_count": "網域名稱中的點號偏多（{v:.0f} 個），可能以多層子網域混淆真實網域。",
    "special_chars": "網域名稱含大量符號（{v:.0f} 個點號／連字號），常見於以長串拼接混淆的仿冒網域。",
    "is_https": "此網址明確使用未加密的 http:// 連線，資料傳輸不安全，詐騙網站常見此手法。",
    "is_ip_address": "網址使用 IP 位址作為網域，合法廣告幾乎不會這樣做，通常代表刻意隱藏真實身份。",
    "has_scam_word": "網址（解碼後）含投資、獎勵、加密貨幣或高誘因相關字詞。",
    "brand_in_sld": "網域名稱含知名品牌字樣並使用常被濫用的頂級域名，但不是該品牌官方網域，疑似品牌仿冒。",
    "double_http": "網址中內嵌另一個 http/https 網址（轉址或追蹤參數），請確認最終導向的網站是否可信。",
    "long_domain": "網域名稱過長（超過 30 字元），常見於以長網域混淆視覺的詐騙網站。",
    "domain_entropy": "網域名稱字元隨機性偏高（entropy = {v:.2f}），可能為大量自動產生的臨時詐騙網域。",
    "has_shortener": "此網址使用短網址服務（如 bit.ly、reurl.cc），可能隱藏真實目的地。",
    "gambling_number_pattern": "網域中出現 168、888、666 等博弈常見幸運數字組合。",
    "brand_typo_like": "網域疑似知名品牌的拼字變形（如 sh0pee、faebook、walmar），且不是該品牌官方網域。",
    "cloud_hosting": "此網址架在雲端／CDN 基礎設施主機（如 CloudFront、Azure、pages.dev），需留意是否為臨時詐騙頁面。",
    "mobile_lure_path": "網址使用 m／h5／wap／app／sj／mobile 行動版主機名稱，或 /h5、/wap、/m 行動版路徑，符合行動版詐騙落地頁常見格式。",
    "suspicious_keyword_in_domain": "詐騙相關關鍵字（如 vip、casino、usdt、tougu）直接出現在網域名稱中，風險極高。",
    "many_subdomains": "子網域層數過多（3 層以上），常見於詐騙網站以子網域混淆視覺。",
    "newly_registered_like": "網域名稱由無意義的隨機字元組成，不像已建立的品牌網站，符合詐騙網站快速申請新域名後即用即棄的模式。",
    "gambling_keyword": "網址含博弈相關字詞（如 casino、百家樂、娛樂城拼音 yule、葡京 pujing），疑似非法博弈網站。",
    "investment_lure_keyword": "網址含假投資常見字詞（如 飆股、投顧 tougu、老師帶單、IPO 抽籤、AI 選股、stock-vip）。",
    "crypto_exchange_lure": "網址含假交易所／錢包常見字詞（如 USDT、OTC、staking、airdrop、-ex 交易所後綴），且不是官方網域。",
    "brand_impersonation": "網域使用知名品牌、銀行、交易所或政府機關名稱（可能加上 tw、vip、login 等字），但實際網域不屬於該品牌官方，疑似冒用。",
    "free_hosting_platform": "網址架在免費架站／雲端暫存平台（如 web.app、pages.dev、github.io、wixsite），任何人都能免費建立，詐騙網站常用來快速上線。",
    "tunnel_or_ephemeral_host": "網址使用臨時通道服務（如 ngrok、trycloudflare）或 IPFS 閘道，網址隨時可換，常見於即用即棄的詐騙或釣魚頁面。",
    "social_invite_link": "網址為 LINE／Telegram／WhatsApp／Discord 群組或官方帳號邀請連結；假投資詐騙常以廣告導流加入群組，請先查證來源。",
    "punycode_domain": "網域為國際化網域（punycode，xn--），實際顯示文字可能與正牌網址極為相似，請仔細確認。",
    "url_has_at_symbol": "網址在網域前加入「@」帳號欄位，瀏覽器實際連往 @ 後面的網域，屬典型偽裝手法。",
    "non_standard_port": "網址指定了非標準連接埠（80／443 以外），合法網站極少這樣做。",
    "sld_digit_count": "網域名稱含 {v:.0f} 個數字，品牌＋長數字常見於大量註冊的詐騙網域。",
    "path_scam_route": "網址含假投資／假交易所 App 網頁常見的路由（如 #/pages/login、/h5/#/、邀請碼參數、download.html 下載誘導），請確認來源。",
    "sld_randomness": "網域名稱不像一般英文或拼音字詞（亂碼度 {v:.2f}），大量註冊、即用即棄的詐騙網域常見這種隨機字母組合。",
}
# 這些特徵的狀態只供前端顯示，不另外產生說明（避免與其他說明重複）
_EXPLAIN_SKIP = frozenset({
    "domain_length", "subdomain_depth", "path_depth", "query_params", "suspicious_tld",
    "has_utm", "has_gclid", "contains_percent_encoding", "sld_length",
})


def explain_features(
    url: Any,
    feature_dict: Optional[Dict[str, float]] = None,
) -> List[Dict[str, str]]:
    """
    依特徵產生可讀的風險說明清單：[{key, level("high"|"medium"), message}]。
    level 由 assess_feature() 決定（alert → high、warn → medium），與前端門檻一致。
    feature_dict：呼叫端已算過 extract_feature_dict(url) 時可直接傳入。永不丟例外。
    """
    try:
        features = dict(FEATURE_DEFAULTS)
        features.update(feature_dict if feature_dict is not None else extract_feature_dict(url))
        assessment = assess_all(features)
        high: List[Dict[str, str]] = []
        medium: List[Dict[str, str]] = []

        def add(key: str, level: str, message: str) -> None:
            (high if level == "high" else medium).append({"key": key, "level": level, "message": message})

        for name in FEATURE_NAMES:
            status = assessment.get(name, {}).get("status")
            if name in _EXPLAIN_SKIP or status not in ("alert", "warn"):
                continue
            # is_https 的 safe/warn 語意反向：值為 0（明確 http）時 status = warn
            value = _to_number(features.get(name))
            level = "high" if status == "alert" else "medium"

            if name == "url_length" and (features.get("has_utm") or features.get("has_gclid")):
                continue
            if name == "levenshtein_brand_dist":
                if value == 0:
                    if features.get("brand_impersonation"):
                        continue
                    add(name, level, "網域名稱（去除連字號、還原同形字後）與知名品牌同名，但不是該品牌的官方網域，疑似冒用品牌名稱。")
                else:
                    add(name, level, f"網域名稱與知名品牌高度相似（編輯距離 = {value:.0f}），疑似輕微拼字仿冒（如 sh0pee、lazadaa）。")
                continue
            if name == "tld_risk_level":
                tld = get_tld(url) if isinstance(url, str) else ""
                tld_text = f"（.{tld}）" if tld else ""
                if value >= 2:
                    add(name, level, f"此網址使用高度濫用的頂級域名{tld_text}，這類低成本域名大量被用於即用即棄型詐騙網站。")
                else:
                    add(name, level, f"此網址使用常被濫用的頂級域名{tld_text}，需搭配其他特徵判斷。")
                continue
            if name == "brand_in_sld" and features.get("brand_impersonation"):
                continue
            if name == "sld_randomness" and features.get("newly_registered_like"):
                continue   # 已由「疑似隨機新網域」說明
            if name == "has_scam_word" and any(features.get(k) for k in (
                    "gambling_keyword", "investment_lure_keyword", "crypto_exchange_lure",
                    "suspicious_keyword_in_domain")):
                continue
            template = _EXPLAIN_MESSAGES.get(name)
            if not template:
                continue
            add(name, level, template.format(v=value, pct=f"{value:.0%}"))
        return high + medium
    except Exception:  # noqa: BLE001
        return []
