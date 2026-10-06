# =============================================================================
# main.py — TruthMark（自動化廣告與網站可信度審查平台）FastAPI 後端 v7.2
# =============================================================================
# 功能：
# 1. 提供首頁 index.html（以及同層的 script.js / style.css）
# 2. 提供 /predict URL 風險預測、/predict/batch 批次預測（≤50 筆）
# 3. 提供 /features 特徵清單與規格（FEATURE_SPECS、FEATURE_SCHEMA_ID）
# 4. 提供 /health 系統狀態（含模型降級資訊）
# 5. 提供 /feedback 使用者回饋（背景任務處理，立即回應前端）
# 6. 提供 /reload-model 重新載入模型（原子替換）
# 7. 提供 /cache/stats 快取命中率觀測、/blocklist 黑名單管理
#
# 使用方式：
#   uvicorn main:app --reload     或     python run_server.py
#
# v6.0 重點（保留）：全面非同步化、共用 httpx.AsyncClient、TTLCache 快取、
#   /feedback 背景任務、compact 精簡回應、deep_scan 語意分析掛鉤。
#
# 修正紀錄（v7.1，對齊論文表 3-1 安全性與濫用防範機制；細節見 security.py）：
#   1. CORS 由 ["*"] 改為白名單（TRUTHMARK_ALLOWED_ORIGINS 可覆寫）＋ chrome-extension origin。
#   2. /feedback 20 次／分、/blocklist/add 10 次／分的速率限制，超過回 HTTP 429＋Retry-After。
#   3. /blocklist/add 以「回報者指紋（IP 加鹽雜湊）」計算不同回報者數，同一人重複回報不加次數。
#   4. /reload-model 與新增的 DELETE /blocklist/{domain}（誤封復原）需 X-Admin-Key（TRUTHMARK_ADMIN_KEY）；
#      未設定金鑰時僅允許本機呼叫。
#   5. feedback.csv 的讀改寫另以 filelock 跨行程加鎖。
#
# 修正紀錄（v7.0，依 CONTRACT §3）：
#   1. 【硬規則依 target 比對】改用 rules_config.match_hard_rules(
#      features.get_rule_targets(url))；舊版對原始網址 re.search，host 錨定的
#      規則（ip_address_url、tunnel、punycode、博弈/假投資 host 規則）永遠不會觸發。
#   2. 【白名單】改用 rules_config 的 WHITELIST_OVERRIDE_SIGNALS（自本檔移出）、
#      WHITELIST_OVERRIDE_PATTERNS（對 canonical 與 decoded 網址各比對一次）與
#      TRUSTED_SUFFIXES；line.me 群組邀請、google.com/url?q=高風險網址、
#      sites.google.com、Google 表單等改走完整流程，不再直接給 5 分。
#   3. 【黑名單】registered domain、hostname 以及 hostname 的每一層父網域都比對。
#   4. 【內嵌跳轉】extract_redirect_targets() 取出的內層網址另外評分，取較高分
#      （Google／doubleclick／FB 跳轉不再被 double_http 吃掉）。
#   5. 【融合】max(AI, 最高硬規則) ＋「多條彼此獨立的中高風險硬規則」有上限加成；
#      灰色地帶補強改用 v7 新特徵（0≤lev≤2、gambling/investment/crypto 關鍵詞、
#      brand_impersonation、tunnel、punycode、@ 偽裝…），已由硬規則涵蓋的同一訊號
#      不重複加分；可信任後綴（.gov.tw…）與官方品牌網域不加分，且未命中硬規則時
#      AI 分數上限 35（避免純網址特徵的模型把官方網域判成中高風險）。
#      模型不可用時改用特徵加權的 heuristic_score()（取代舊版幾個 if），標 degraded。
#   6. 【模型】支援 joblib bundle 與舊格式純 estimator；比對 bundle/meta 的
#      feature_names 與 FEATURE_NAMES，不符就不用模型（rules_only + degraded + 說明）；
#      推論失敗 try/except 降級而非 500；重新載入以快照物件原子替換。
#   7. 【API v7】新增 api_version / feature_version / degraded / degrade_reason；
#      debug 加 feature_assessment；/predict/batch；/health 與 /features 擴充；
#      錯誤一律 {ok:false, error, message}；輸入驗證（≤2048 字元、非法 IPv6、
#      奇怪 unicode）；全域例外處理器回 JSON（不回 500）；print 改 logging。
#   8. 【快取】key 用 canonicalize_url，網域索引用 get_registered_domain；
#      /blocklist/add 清該網域、/feedback 清該 URL、/reload-model 清全部；
#      錯誤回應與推論失敗的降級結果不快取。
#   9. 【deep_scan】text_features 已加 SSRF 防護；頁面文字命中 165 高風險詞時
#      依 content_signals 小幅加分並寫入 reasons。
#  10. 【整合驗證】內容平台（rules_config.CONTENT_PLATFORM_DOMAINS：新聞／論壇／百科）
#      未命中硬規則時比照官方網域：AI 分數上限 35、不做灰色地帶補強（文章網址的
#      字面特徵會被純網址模型誤判，例如 chinatimes 即時新聞頁）。
# =============================================================================

from __future__ import annotations

import asyncio
import csv
import ipaddress
import json
import logging
import math
import os
import re
import shutil
import tempfile
import threading
from contextlib import contextmanager
import time
import warnings
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncIterator, Dict, Iterable, List, Optional, Set, Tuple

import httpx
import joblib
import numpy as np
import sklearn
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

import sync_165
from blocklist_store import BlocklistStore, normalize_domain
from cache import TTLCache
from security import (
    AuditLog,
    RateLimiter,
    admin_key,
    check_admin,
    client_ip as _resolve_client_ip,
    cors_settings,
    reporter_fingerprint,
)
from features import (
    FEATURE_DEFAULTS,
    FEATURE_NAMES,
    FEATURE_SCHEMA_ID,
    FEATURE_SPECS,
    FEATURE_THRESHOLDS,
    FEATURE_VERSION,
    assess_all,
    canonicalize_url,
    explain_features,
    extract_feature_dict,
    extract_redirect_targets,
    get_hostname,
    get_registered_domain,
    get_rule_targets,
    is_ip_address,
    normalize_url,
    tldextract_status,
)
from rules_config import (
    BRAND_OFFICIAL_DOMAINS,
    CONTENT_PLATFORM_DOMAINS,
    HARD_RULES,
    INITIAL_BLOCKED_DOMAINS,
    RULES_VERSION,
    TRUSTED_DOMAINS,
    TRUSTED_SUFFIXES,
    WHITELIST_OVERRIDE_PATTERNS,
    WHITELIST_OVERRIDE_SIGNALS,
    HardRule,
    match_hard_rules,
)
from text_features import analyze_page_semantics, new_http_client


# =============================================================================
# 版本、路徑與日誌
# =============================================================================

API_VERSION = "7.2.0"
MODEL_BUNDLE_FORMAT = "truthmark-model-bundle/1"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "scam_model.pkl")
META_PATH = os.path.join(BASE_DIR, "model_meta.json")
REPORT_PATH = os.path.join(BASE_DIR, "model_report.json")
# 執行期狀態可透過 TRUTHMARK_STATE_DIR 移到持久化 volume；未設定時沿用專案目錄。
STATE_DIR = os.path.abspath(os.environ.get("TRUTHMARK_STATE_DIR", "") or BASE_DIR)
try:
    os.makedirs(STATE_DIR, exist_ok=True)
except OSError:
    STATE_DIR = BASE_DIR
FEEDBACK_PATH = os.path.join(STATE_DIR, "feedback.csv")

# 第一次使用獨立 STATE_DIR（例如 Docker volume）時，把專案內附的 165 動態黑名單快照
# 複製到持久化目錄；之後所有新增、回報與同步都直接寫 STATE_DIR。
STATE_DYNAMIC_BLOCKLIST_PATH = os.path.join(STATE_DIR, "dynamic_blocklist.json")
STATE_DOMAIN_REPORTS_PATH = os.path.join(STATE_DIR, "domain_reports.json")
INITIAL_DYNAMIC_BLOCKLIST_PATH = os.path.join(BASE_DIR, "dynamic_blocklist.json")
if STATE_DIR != BASE_DIR and os.path.exists(INITIAL_DYNAMIC_BLOCKLIST_PATH) and not os.path.exists(STATE_DYNAMIC_BLOCKLIST_PATH):
    try:
        shutil.copy2(INITIAL_DYNAMIC_BLOCKLIST_PATH, STATE_DYNAMIC_BLOCKLIST_PATH)
    except OSError as exc:
        logging.getLogger("truthmark").warning("無法建立動態黑名單初始快照：%s", exc)

STATIC_DIR = os.path.join(BASE_DIR, "static")
INDEX_PATH = os.path.join(BASE_DIR, "index.html")
SCRIPT_JS_PATH = os.path.join(BASE_DIR, "script.js")
STYLE_CSS_PATH = os.path.join(BASE_DIR, "style.css")
FAVICON_PATH = os.path.join(BASE_DIR, "外掛", "icons", "icon48.png")

MAX_URL_LENGTH = 2048          # 單一網址長度上限（字元）
MAX_BATCH_SIZE = 50            # /predict/batch 單批上限
BATCH_CONCURRENCY = 8          # 批次內同時處理的網址數
MAX_REDIRECT_TARGETS = 3       # 內嵌跳轉目標最多評估幾個
DEEP_SCAN_TIMEOUT = 12.0       # deep_scan 整體時間上限（秒；含 DNS、逐跳重新導向）

logger = logging.getLogger("truthmark")


def _configure_logging() -> None:
    """未設定任何 handler 時（例如直接 uvicorn main:app）補一個簡單的 stderr handler。"""
    if logger.handlers or logging.getLogger().handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s：%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(os.environ.get("TRUTHMARK_LOG_LEVEL", "INFO").upper())
    logger.propagate = False


_configure_logging()


# =============================================================================
# 非同步 HTTP 用戶端（連線池）與應用程式生命週期
# =============================================================================
# deep_scan 共用同一個 AsyncClient（由 text_features.new_http_client() 建立：
# 不自動跟隨重新導向、不存 cookie、不讀系統 proxy），由本模組逐跳做 SSRF 檢查。

http_client: Optional[httpx.AsyncClient] = None


SYNC_165_STARTUP_DELAY = 5.0   # 讓服務先完成啟動、開始接請求，再背景下載


async def run_165_sync(actor: str, *, dry_run: bool = False, force: bool = False) -> Dict[str, Any]:
    """
    執行一次 165 同步（下載與寫檔在執行緒池）。有新增網域時清空預測快取（舊快取可能還記著
    「未封鎖」的結果）；非 dry-run 一律寫入稽核紀錄。
    """
    report = await run_in_threadpool(lambda: sync_165.sync_165(blocklist_store, dry_run=dry_run, force=force))
    report["cache_cleared"] = prediction_cache.clear() if (report["ok"] and not dry_run and report["added"]) else 0
    if not dry_run:
        audit_log.record(
            "blocklist.sync_165",
            actor=actor,
            target=sync_165.SOURCE_NAME,
            success=report["ok"],
            detail={k: report.get(k) for k in ("rows", "added", "already", "protected", "skipped_shared",
                                                "not_modified", "cache_cleared", "error")},
        )
    return report


async def _sync_165_loop() -> None:
    """背景排程：啟動後同步一次，之後每 TRUTHMARK_165_SYNC_HOURS 小時一次（失敗時最多 1 小時後重試）。"""
    await asyncio.sleep(SYNC_165_STARTUP_DELAY)
    first_run = True
    while True:
        interval = sync_165.configured_interval_hours() * 3600
        ok = False
        try:
            # 行程啟動後的第一次一律忽略 ETag：狀態檔可能比 dynamic_blocklist.json 更久存（例如只掛載
            # 了 data volume），若沿用 ETag 會得到 304、什麼都沒匯入，黑名單就是空的。
            ok = (await run_165_sync("system", force=first_run))["ok"]
            first_run = False
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — 排程不能因單次例外而終止
            logger.exception("165 自動同步發生未預期錯誤")
        if interval <= 0:
            return
        await asyncio.sleep(interval if ok else min(interval, 3600.0))


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    global http_client
    http_client = new_http_client()
    await run_in_threadpool(model_state.load)
    try:
        blocklist_store.load_dynamic_blocklist()
    except Exception as exc:  # noqa: BLE001 — 黑名單檔損毀不應讓服務起不來
        logger.warning("動態黑名單載入失敗：%s", exc)
    if not admin_key():
        logger.warning("未設定 TRUTHMARK_ADMIN_KEY：/reload-model 與 DELETE /blocklist/{domain} 僅允許本機（loopback）呼叫；"
                       "對外部署前請設定管理員金鑰")
    _tld_status = tldextract_status()
    if _tld_status["degraded"]:
        logger.warning("啟動檢查：%s（執行 pip install -r requirements.txt 可修復）", _tld_status["message"])
    logger.info(
        "TruthMark API %s 啟動：特徵 %s（%d 維，schema=%s）、硬規則 %d 條、黑名單 %d 筆、模型%s",
        API_VERSION, FEATURE_VERSION, len(FEATURE_NAMES), FEATURE_SCHEMA_ID, len(HARD_RULES),
        len(BLOCKED_DOMAINS), "可用" if model_state.usable else f"不可用（{model_state.load_error}）",
    )
    sync_task: Optional["asyncio.Task[None]"] = None
    if sync_165.configured_urls():
        sync_task = asyncio.create_task(_sync_165_loop(), name="sync-165")
        logger.info("165 涉詐網域自動同步已啟用：啟動後 %d 秒首次同步，之後%s",
                    int(SYNC_165_STARTUP_DELAY),
                    f"每 {sync_165.configured_interval_hours():g} 小時一次" if sync_165.configured_interval_hours() > 0
                    else "不再自動重複（TRUTHMARK_165_SYNC_HOURS=0）")
    else:
        logger.info("未設定 %s：165 自動同步未啟用（可用 POST /blocklist/sync-165 或 sync_165.py 手動匯入）",
                    sync_165.ENV_URL)
    try:
        yield
    finally:
        if sync_task is not None:
            sync_task.cancel()
            try:
                await sync_task
            except asyncio.CancelledError:
                pass
        client, http_client = http_client, None
        if client is not None:
            await client.aclose()


# =============================================================================
# FastAPI 初始化
# =============================================================================

app = FastAPI(
    title="TruthMark API",
    description="自動化廣告與網站可信度審查平台：白名單 → 黑名單 → 硬規則 → AI 模型（不可用時特徵啟發式）＋快取與非同步爬取",
    version=API_VERSION,
    lifespan=lifespan,
)

_CORS_ORIGINS, _CORS_EXTENSION_REGEX = cors_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_origin_regex=_CORS_EXTENSION_REGEX,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "X-Admin-Key"],
)

# 濫用防範（security.py）：速率限制、回報者指紋、管理員金鑰
rate_limiter = RateLimiter()

# 稽核紀錄（v7.2 新增）：記錄管理端點（模型重載、黑名單移除）與黑名單新增回報，
# 路徑可用 TRUTHMARK_AUDIT_LOG_PATH 覆寫，預設寫在服務工作目錄下的 audit.log。
audit_log = AuditLog()


def get_client_ip(request: Request) -> str:
    """用戶端 IP；僅在 TRUTHMARK_TRUST_PROXY=1 時採用 X-Forwarded-For 第一段。"""
    return _resolve_client_ip(request.client.host if request.client else None,
                              request.headers.get("x-forwarded-for"))


def enforce_rate_limit(request: Request, route: str) -> None:
    """超過上限時丟 HTTP 429（由 _handle_http_error 轉成統一的 ok:false JSON，並帶 Retry-After）。"""
    allowed, retry_after = rate_limiter.check(route, get_client_ip(request))
    if not allowed:
        raise StarletteHTTPException(
            status_code=429,
            detail=f"請求過於頻繁，請 {retry_after} 秒後再試",
            headers={"Retry-After": str(retry_after)},
        )


def require_admin(request: Request) -> None:
    """管理端點守門：X-Admin-Key 不符（或未設金鑰且非本機）回 401／403。"""
    ok, reason = check_admin(request.headers.get("x-admin-key"),
                             request.client.host if request.client else None)
    if not ok:
        status = 401 if reason == "admin_key_invalid" else 403
        text = ("管理員金鑰不正確" if status == 401
                else "伺服器尚未設定 TRUTHMARK_ADMIN_KEY，管理端點僅限本機呼叫")
        raise StarletteHTTPException(status_code=status, detail=text)

# /predict 結果快取：key = canonical URL + 回應變體；以 registered domain 建索引。
prediction_cache = TTLCache(max_size=4096, default_ttl_seconds=600.0)
TRUSTED_CACHE_TTL = 600.0
BLOCKED_CACHE_TTL = 600.0
SCORED_CACHE_TTL = 60.0

if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# =============================================================================
# 錯誤格式與全域例外處理
# =============================================================================

def error_payload(message: str, code: str = "bad_request", **extra: Any) -> Dict[str, Any]:
    """統一錯誤格式：error 與 message 同內容（前端有的讀 error、有的讀 message）。"""
    text = str(message or "發生未知錯誤")
    payload: Dict[str, Any] = {"ok": False, "error": text, "message": text, "error_code": code}
    payload.update(extra)
    return payload


def _validation_detail(errors: Iterable[Any]) -> List[Dict[str, Any]]:
    detail: List[Dict[str, Any]] = []
    for err in list(errors)[:10]:
        try:
            detail.append({
                "loc": [str(x) for x in err.get("loc", ())],
                "msg": str(err.get("msg", "")),
                "type": str(err.get("type", "")),
            })
        except Exception:  # noqa: BLE001
            continue
    return detail


@app.exception_handler(RequestValidationError)
async def _handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    detail = _validation_detail(exc.errors())
    first = detail[0] if detail else {}
    loc = ".".join(x for x in first.get("loc", []) if x != "body")
    if first.get("type") == "json_invalid":
        text = "請求內容不是有效的 JSON"
    elif first.get("type") == "missing" and loc:
        text = f"缺少必要欄位：{loc}"
    elif loc:
        text = f"欄位 {loc} 格式錯誤：{first.get('msg', '')}"
    else:
        text = "請求格式錯誤：請以 JSON 物件傳送，例如 {\"url\": \"https://example.com\"}"
    return JSONResponse(status_code=422, content=error_payload(text, "validation_error", detail=detail))


@app.exception_handler(StarletteHTTPException)
async def _handle_http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
    messages = {404: "找不到此 API 路徑", 405: "此路徑不支援這個 HTTP 方法"}
    text = messages.get(exc.status_code) or str(exc.detail or f"HTTP {exc.status_code}")
    return JSONResponse(status_code=exc.status_code, content=error_payload(text, f"http_{exc.status_code}"),
                        headers=getattr(exc, "headers", None))


@app.exception_handler(Exception)
async def _handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    # 最後防線：任何未預期例外都回 JSON（HTTP 200 + ok:false），不讓前端收到 500。
    logger.exception("未預期的伺服器錯誤（%s %s）", request.method, request.url.path)
    return JSONResponse(
        status_code=200,
        content=error_payload("伺服器處理時發生未預期錯誤，請稍後再試", "internal_error"),
    )


# -----------------------------------------------------------------------
# 首頁同層靜態檔（index.html 以 <script src="script.js"> 引用）
# -----------------------------------------------------------------------

@app.get("/script.js")
async def serve_script_js() -> Any:
    if not os.path.exists(SCRIPT_JS_PATH):
        return HTMLResponse("// script.js not found", status_code=404)
    return FileResponse(SCRIPT_JS_PATH, media_type="application/javascript")


@app.get("/style.css")
async def serve_style_css() -> Any:
    if not os.path.exists(STYLE_CSS_PATH):
        return HTMLResponse("/* style.css not found */", status_code=404)
    return FileResponse(STYLE_CSS_PATH, media_type="text/css")


@app.get("/favicon.ico", include_in_schema=False)
async def serve_favicon() -> Any:
    # 瀏覽器會自動請求 /favicon.ico：沿用外掛圖示；找不到時回 204（避免 console 出現 404 錯誤）
    if os.path.exists(FAVICON_PATH):
        return FileResponse(FAVICON_PATH, media_type="image/png")
    return Response(status_code=204)


# =============================================================================
# 白名單 / 黑名單（實際內容見 rules_config.py）
# =============================================================================

# 官方網域集合（白名單 ∪ 各品牌官方網域）：用於「不要對官方網域誤加分」
OFFICIAL_DOMAINS: Set[str] = set(TRUSTED_DOMAINS)
for _domains in BRAND_OFFICIAL_DOMAINS.values():
    OFFICIAL_DOMAINS |= set(_domains)

_WHITELIST_OVERRIDE_RES = tuple(re.compile(p, re.IGNORECASE) for p in WHITELIST_OVERRIDE_PATTERNS)
_OVERRIDE_SIGNALS_LOWER = tuple(s.lower() for s in WHITELIST_OVERRIDE_SIGNALS)

# 執行期間可變動的黑名單集合：以 rules_config 的種子清單（正規化後）為初始內容，
# 之後由 BlocklistStore 依使用者回報或 165 開放資料匯入動態新增。
BLOCKED_DOMAINS: Set[str] = {normalize_domain(d) or str(d).strip().lower() for d in INITIAL_BLOCKED_DOMAINS}

# 回報門檻機制：同一網域累積回報達 REPORT_THRESHOLD 次才正式列入黑名單（預設 3）。
REPORT_THRESHOLD = int(os.environ.get("BLOCKLIST_REPORT_THRESHOLD", "3"))


def parent_domains(hostname: str) -> List[str]:
    """hostname 本身與每一層父網域（不含單獨的 TLD）：a.b.evil.com → [a.b.evil.com, b.evil.com, evil.com]。"""
    labels = [lab for lab in (hostname or "").strip(".").split(".") if lab]
    return [".".join(labels[i:]) for i in range(max(len(labels) - 1, 1))] if labels else []


def _has_trusted_suffix(hostname: str) -> bool:
    dotted = "." + hostname
    return any(dotted.endswith(s) or hostname == s.lstrip(".") for s in TRUSTED_SUFFIXES)


def is_protected_domain(domain: str) -> bool:
    """可信任網域（白名單或其子網域、TRUSTED_SUFFIXES）：不得被使用者回報自動列入黑名單。"""
    host = (domain or "").strip().lower().rstrip(".")
    if not host:
        return False
    return any(p in TRUSTED_DOMAINS for p in parent_domains(host)) or _has_trusted_suffix(host)


blocklist_store = BlocklistStore(
    base_dir=STATE_DIR,
    blocked_domains=BLOCKED_DOMAINS,
    threshold=REPORT_THRESHOLD,
    is_protected=is_protected_domain,
)


def _host_is_trusted(hostname: str, registered: str) -> bool:
    if not hostname or is_ip_address(hostname):
        return False
    if registered in TRUSTED_DOMAINS:
        return True
    return any(p in TRUSTED_DOMAINS for p in parent_domains(hostname)) or _has_trusted_suffix(hostname)


def is_official_host(hostname: str, registered: str) -> bool:
    """官方網域（白名單 ∪ 品牌官方網域 ∪ 可信任後綴）：灰色地帶補強與啟發式評分不加分。"""
    if not hostname or is_ip_address(hostname):
        return False
    if registered in OFFICIAL_DOMAINS or _has_trusted_suffix(hostname):
        return True
    return any(p in OFFICIAL_DOMAINS for p in parent_domains(hostname))


def is_content_platform_host(hostname: str, registered: str) -> bool:
    """新聞／查核／論壇／百科等內容平台（rules_config.CONTENT_PLATFORM_DOMAINS）：路徑是文章代號，
    純網址字面的 AI 分數意義不大（未命中硬規則時套用 AI 上限）。"""
    if not hostname or is_ip_address(hostname):
        return False
    if registered in CONTENT_PLATFORM_DOMAINS:
        return True
    return any(p in CONTENT_PLATFORM_DOMAINS for p in parent_domains(hostname))


def whitelist_override_reason(url: str) -> str:
    """白名單網域上的高風險內容：回傳命中的訊號說明（空字串 = 未命中）。"""
    targets = get_rule_targets(url)
    canonical = targets.get("url", "") or ""
    decoded = targets.get("decoded", "") or ""
    lowered = (canonical + "\n" + decoded).lower()
    for signal in _OVERRIDE_SIGNALS_LOWER:
        if signal and signal in lowered:
            return f"signal:{signal}"
    for regex in _WHITELIST_OVERRIDE_RES:
        if regex.search(canonical) or regex.search(decoded):
            return f"pattern:{regex.pattern[:60]}"
    return ""


def is_trusted_domain(url: str) -> bool:
    """
    URL 是否落在白名單（TRUSTED_DOMAINS 或其子網域、TRUSTED_SUFFIXES），且不含足以
    覆蓋白名單的強烈詐騙信號（WHITELIST_OVERRIDE_SIGNALS / WHITELIST_OVERRIDE_PATTERNS）。
    """
    hostname = get_hostname(url)
    if not _host_is_trusted(hostname, get_registered_domain(url)):
        return False
    return not whitelist_override_reason(url)


def is_blocked_domain(url: str) -> bool:
    """registered domain、hostname 或 hostname 的任一層父網域在黑名單中即命中。"""
    hostname = get_hostname(url)
    if not hostname:
        return False
    if get_registered_domain(url) in BLOCKED_DOMAINS:
        return True
    return any(p in BLOCKED_DOMAINS for p in parent_domains(hostname))


# =============================================================================
# Pydantic Schema
# =============================================================================

class PredictRequest(BaseModel):
    url: str
    debug: bool = False
    # 精簡回應：只回傳擴充套件渲染徽章/清單所需的最小欄位（見 to_compact_dict）。
    compact: bool = False
    # 深度語意掃描：非同步抓取網頁內容（含 SSRF 防護）＋ 165 高風險詞偵測（見 text_features.py）。
    deep_scan: bool = False


class BatchPredictRequest(BaseModel):
    urls: List[Any]
    debug: bool = False
    compact: bool = True


class FeedbackRequest(BaseModel):
    url: str
    label: int
    ai_score: float = 0.0
    note: str = ""


class BlocklistAddRequest(BaseModel):
    domain: str
    note: str = ""


class Sync165Request(BaseModel):
    # 不接受呼叫端指定下載網址（避免 SSRF）：來源只來自環境變數 TRUTHMARK_165_CSV_URL。
    dry_run: bool = False   # 只試算、不寫入
    force: bool = False     # 忽略 ETag，強制重新下載


# =============================================================================
# AI 模型狀態管理（支援 bundle / 舊格式 estimator；特徵不符即降級）
# =============================================================================

class ModelUnavailableError(RuntimeError):
    """模型不可用（未載入或特徵不相容）。"""


@dataclass(frozen=True)
class ModelSnapshot:
    """一次載入的完整結果；重新載入時整個物件原子替換，推論端只讀取一次參照。"""

    estimator: Any = None
    file_loaded: bool = False          # 模型檔是否成功讀取
    usable: bool = False               # 是否可用於推論（檔案可讀 + 特徵相容 + 試算成功）
    load_error: str = ""
    degrade_reason: str = ""
    version_ok: bool = False
    feature_names_ok: bool = False
    feature_check: str = ""            # names / count / mismatch / unknown
    model_format: str = ""             # bundle / legacy
    model_feature_count: Optional[int] = None
    info: Dict[str, Any] = field(default_factory=dict)
    loaded_at: str = ""


class ModelState:
    """
    集中管理 AI 模型的載入狀態。
    對外屬性（model / loaded / load_error / version_ok）維持 v6 語意；
    v7 新增 usable / feature_names_ok / degraded / degrade_reason。
    """

    def __init__(self, model_path: str, meta_path: str) -> None:
        self.model_path = model_path
        self.meta_path = meta_path
        self._snapshot = ModelSnapshot(
            load_error="模型尚未載入",
            degrade_reason="AI 模型尚未載入，改用硬規則＋特徵啟發式評分",
        )
        self._load_lock = threading.Lock()

    # ---- 唯讀屬性（每次讀取同一個快照） ----
    @property
    def snapshot(self) -> ModelSnapshot:
        return self._snapshot

    @property
    def model(self) -> Any:
        return self._snapshot.estimator if self._snapshot.usable else None

    @property
    def usable(self) -> bool:
        return self._snapshot.usable

    @property
    def loaded(self) -> bool:
        # v7：model_loaded 代表「模型已載入且可用於推論」（特徵不符時為 False）
        return self._snapshot.usable

    @property
    def load_error(self) -> str:
        return self._snapshot.load_error

    @property
    def version_ok(self) -> bool:
        return self._snapshot.version_ok

    @property
    def feature_names_ok(self) -> bool:
        return self._snapshot.feature_names_ok

    @property
    def degraded(self) -> bool:
        return not self._snapshot.usable

    @property
    def degrade_reason(self) -> str:
        return "" if self._snapshot.usable else self._snapshot.degrade_reason

    def load_meta(self) -> Dict[str, Any]:
        """讀取 model_meta.json；不存在或格式錯誤時回傳空 dict。"""
        if not os.path.exists(self.meta_path):
            return {}
        try:
            with open(self.meta_path, "r", encoding="utf-8-sig") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:  # noqa: BLE001
            return {}

    # ---- 載入 ----
    @staticmethod
    def _failed(error: str, reason: str, **kw: Any) -> ModelSnapshot:
        return ModelSnapshot(load_error=error, degrade_reason=reason, loaded_at=datetime.now().isoformat(), **kw)

    def _build_snapshot(self) -> ModelSnapshot:
        model_path = self.model_path
        if not os.path.exists(model_path):
            return self._failed(
                "找不到 scam_model.pkl，請先執行 train_model.py",
                "找不到 AI 模型檔，改用硬規則＋特徵啟發式評分",
            )

        meta = self.load_meta()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                obj = joblib.load(model_path)
        except Exception as exc:  # noqa: BLE001
            return self._failed(
                f"模型檔載入失敗：{type(exc).__name__}: {str(exc)[:200]}",
                "AI 模型檔無法讀取，改用硬規則＋特徵啟發式評分",
            )

        info: Dict[str, Any] = {}
        if isinstance(obj, dict):
            model_format = "bundle"
            estimator = obj.get("estimator")
            names = obj.get("feature_names")
            if names is None:
                names = meta.get("feature_names")
            trained_sklearn = obj.get("sklearn_version") or meta.get("sklearn_version")
            for key in ("format", "feature_version", "feature_schema_id", "thresholds", "trained_at", "best_model"):
                if key in obj:
                    info[key] = obj.get(key)
            if obj.get("format") not in (None, MODEL_BUNDLE_FORMAT):
                logger.warning("模型 bundle 格式為 %r（預期 %s），仍嘗試使用", obj.get("format"), MODEL_BUNDLE_FORMAT)
        else:
            model_format = "legacy"
            estimator = obj
            names = meta.get("feature_names")
            trained_sklearn = meta.get("sklearn_version")
        info["sklearn_version"] = trained_sklearn

        if estimator is None or not (hasattr(estimator, "predict_proba") or hasattr(estimator, "predict")):
            return self._failed("模型檔內容不是可用的分類器（缺少 estimator 或 predict_proba）",
                                "AI 模型檔格式錯誤，改用硬規則＋特徵啟發式評分",
                                file_loaded=True, model_format=model_format, info=info)

        expected = list(FEATURE_NAMES)
        n_in = getattr(estimator, "n_features_in_", None)
        est_names = getattr(estimator, "feature_names_in_", None)
        model_count: Optional[int] = None
        problem = ""
        check = "unknown"

        if names is not None:
            try:
                names = [str(n) for n in names]
            except TypeError:
                names = None
        if names is not None:
            model_count = len(names)
            check = "names"
            if names != expected:
                diff = "名稱或順序不同" if len(names) == len(expected) else f"{len(names)} 維／目前 {len(expected)} 維"
                problem = f"模型特徵與目前 v{FEATURE_VERSION} 特徵不一致（{diff}）"
        if not problem and n_in is not None:
            model_count = model_count or int(n_in)
            if int(n_in) != len(expected):
                problem = f"模型輸入維度 {int(n_in)} 與目前特徵數 {len(expected)} 不一致"
            elif check == "unknown":
                check = "count"
        if not problem and est_names is not None:
            if [str(n) for n in est_names] != expected:
                problem = "模型訓練時的欄位名稱與目前 FEATURE_NAMES 不一致"
        if not problem and check == "unknown":
            meta_count = meta.get("feature_count")
            if isinstance(meta_count, int):
                model_count = meta_count
                if meta_count != len(expected):
                    problem = f"model_meta.json 記載的特徵數 {meta_count} 與目前特徵數 {len(expected)} 不一致"
                else:
                    check = "count"

        if problem:
            return self._failed(
                f"{problem}，已停用模型；請重新執行 train_model.py",
                f"AI 模型與目前特徵版本不符（{problem}），改用硬規則＋特徵啟發式評分",
                file_loaded=True, feature_check="mismatch", model_format=model_format,
                model_feature_count=model_count, info=info,
            )

        # 試算一次，確保維度與型別真的可用（未知格式的模型也能在這裡被擋下）
        try:
            self._raw_predict(estimator, dict(FEATURE_DEFAULTS))
        except Exception as exc:  # noqa: BLE001
            return self._failed(
                f"模型試算失敗：{type(exc).__name__}: {str(exc)[:200]}",
                "AI 模型無法以目前特徵推論，改用硬規則＋特徵啟發式評分",
                file_loaded=True, feature_check="mismatch", model_format=model_format,
                model_feature_count=model_count, info=info,
            )

        version_ok = not trained_sklearn or trained_sklearn == sklearn.__version__
        if not version_ok:
            logger.warning("sklearn 版本不同：訓練版本=%s，目前版本=%s（仍使用模型）",
                           trained_sklearn, sklearn.__version__)
        if check == "count":
            logger.warning("模型未記錄 feature_names，只能以特徵數（%d）驗證相容性", len(expected))
        return ModelSnapshot(
            estimator=estimator, file_loaded=True, usable=True, version_ok=version_ok,
            feature_names_ok=True, feature_check=check, model_format=model_format,
            model_feature_count=model_count or len(expected), info=info,
            loaded_at=datetime.now().isoformat(),
        )

    def load(self) -> bool:
        """載入（或重新載入）scam_model.pkl；建好完整快照後才原子替換。回傳是否可用。"""
        with self._load_lock:
            try:
                snapshot = self._build_snapshot()
            except Exception as exc:  # noqa: BLE001
                logger.exception("模型載入發生未預期錯誤")
                snapshot = self._failed(f"模型載入發生未預期錯誤：{exc}",
                                        "AI 模型載入失敗，改用硬規則＋特徵啟發式評分")
            self._snapshot = snapshot
        if snapshot.usable:
            logger.info("Truth 模型載入成功（%s，特徵檢查=%s）", snapshot.model_format, snapshot.feature_check)
        else:
            logger.warning("模型不可用，改用規則評分：%s", snapshot.load_error)
        return snapshot.usable

    # ---- 推論 ----
    @staticmethod
    def _raw_predict(estimator: Any, feature_dict: Dict[str, float]) -> float:
        row = []
        for name in FEATURE_NAMES:
            try:
                value = float(feature_dict.get(name, FEATURE_DEFAULTS[name]))
            except (TypeError, ValueError):
                value = float(FEATURE_DEFAULTS[name])
            row.append(value if math.isfinite(value) else 0.0)
        X: Any = np.array([row], dtype=float)
        if getattr(estimator, "feature_names_in_", None) is not None:
            import pandas as pd  # 延遲匯入：只有以 DataFrame 訓練的模型需要

            X = pd.DataFrame(X, columns=list(FEATURE_NAMES))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if hasattr(estimator, "predict_proba"):
                proba = estimator.predict_proba(X)[0]
                classes = list(getattr(estimator, "classes_", []))
                index = classes.index(1) if 1 in classes else len(proba) - 1
                probability = float(proba[index])
            else:
                probability = float(estimator.predict(X)[0])
        if not math.isfinite(probability):
            raise ValueError("模型輸出不是有效數值")
        return min(100.0, max(0.0, probability * 100.0))

    def predict_score(self, feature_dict: Dict[str, float]) -> float:
        """
        使用 AI 模型取得詐騙機率，回傳 0~100。模型不可用時丟 ModelUnavailableError，
        由呼叫端降級為規則評分（不會回 500）。
        """
        snapshot = self._snapshot
        if not snapshot.usable or snapshot.estimator is None:
            raise ModelUnavailableError(snapshot.degrade_reason or "AI 模型不可用")
        return self._raw_predict(snapshot.estimator, feature_dict)

    async def predict_score_async(self, feature_dict: Dict[str, float]) -> float:
        """predict_score() 的非同步版本（sklearn 推論丟執行緒池，不阻塞事件迴圈）。"""
        return await run_in_threadpool(self.predict_score, feature_dict)

    def describe(self) -> Dict[str, Any]:
        snapshot = self._snapshot
        return {
            "model_loaded": snapshot.usable,
            "model_file_loaded": snapshot.file_loaded,
            "model_load_error": snapshot.load_error,
            "model_format": snapshot.model_format,
            "model_feature_check": snapshot.feature_check,
            "model_feature_count": snapshot.model_feature_count,
            "model_feature_names_ok": snapshot.feature_names_ok,
            "model_loaded_at": snapshot.loaded_at,
            "model_info": snapshot.info,
            "version_ok": snapshot.version_ok,
            "degraded": not snapshot.usable,
            "degrade_reason": "" if snapshot.usable else snapshot.degrade_reason,
        }


model_state = ModelState(MODEL_PATH, META_PATH)


# =============================================================================
# 輸入驗證
# =============================================================================

_UNSUPPORTED_SCHEMES = frozenset({
    "javascript", "vbscript", "data", "about", "blob", "file", "mailto", "tel", "sms",
    "chrome", "chrome-extension", "edge", "view-source", "ws", "wss",
})
_OPAQUE_ALLOWED_SCHEMES = frozenset({"itms-services"})   # iOS 企業簽安裝：交給硬規則判斷
_SCHEME_PREFIX_RE = re.compile(r"^([a-z][a-z0-9+.\-]*):", re.I)
_RAW_SCHEME_RE = re.compile(r"^[a-z][a-z0-9+.\-]*:[/\\]{2}", re.I)
_AUTHORITY_RE = re.compile(r"^[a-z][a-z0-9+.\-]*://([^/?#]*)", re.I)
_HOST_CHARS_RE = re.compile(r"^[a-z0-9_-]+(?:\.[a-z0-9_-]+)*$")


def normalize_input_url(url: Optional[str]) -> str:
    """去頭尾空白＋features.normalize_url（NFKC、移除全形空白／零寬字元／控制字元）。"""
    if url is None:
        return ""
    return normalize_url(str(url).strip())


def url_structure_error(url: str) -> str:
    """回傳網址結構錯誤的中文說明；可評估時回傳空字串。永不丟例外。"""
    try:
        # 先看原始輸入的 authority：canonicalize 會自動補上 IPv6 的右中括號，這裡要擋下
        raw_scheme = _RAW_SCHEME_RE.match(url)
        raw_rest = url[raw_scheme.end():] if raw_scheme else url.lstrip("/")
        raw_hostport = re.split(r"[/?#\\]", raw_rest, maxsplit=1)[0].rpartition("@")[2]
        if raw_hostport.startswith("[") and "]" not in raw_hostport:
            return "IPv6 位址格式錯誤（缺少右中括號 ]）"

        canonical = canonicalize_url(url)
        m = _SCHEME_PREFIX_RE.match(canonical)
        scheme = m.group(1).lower() if m else ""
        if scheme in _UNSUPPORTED_SCHEMES:
            return f"不支援的網址類型（{scheme}:），請輸入 http／https 網址"
        if scheme in _OPAQUE_ALLOWED_SCHEMES:
            return ""
        am = _AUTHORITY_RE.match(canonical)
        if not am:
            return "無法解析網址格式，請輸入完整網址（例如 https://example.com）"
        hostport = am.group(1).rpartition("@")[2]
        port = ""
        if hostport.startswith("["):
            end = hostport.find("]")
            if end < 0:
                return "IPv6 位址格式錯誤（缺少右中括號 ]）"
            try:
                ipaddress.IPv6Address(hostport[1:end].split("%", 1)[0])
            except ValueError:
                return "IPv6 位址格式錯誤"
            rest = hostport[end + 1:]
            if rest:
                if not rest.startswith(":"):
                    return "網址主機格式錯誤"
                port = rest[1:]
        else:
            host, _, port = hostport.partition(":")
            host = host.rstrip(".")
            if not host:
                return "網址缺少網域名稱"
            if len(host) > 253:
                return "網域名稱過長（超過 253 字元）"
            if not _HOST_CHARS_RE.match(host):
                return "網域名稱含有不合法的字元"
            if any(len(label) > 63 for label in host.split(".")):
                return "網域名稱格式錯誤（單一段落超過 63 字元）"
            if "." not in host and not is_ip_address(host):
                return "網址缺少有效的網域名稱（例如 example.com）"
        if port and (not port.isdigit() or len(port) > 5 or not 0 < int(port) <= 65535):
            return "連接埠格式錯誤（必須是 1～65535 的數字）"
        return ""
    except Exception:  # noqa: BLE001
        return "無法解析網址格式"


def validate_url_input(raw: Any) -> Tuple[str, Optional[Dict[str, Any]]]:
    """回傳 (正規化後網址, 錯誤回應或 None)。"""
    if raw is None:
        return "", error_payload("URL 不可為空", "empty_url")
    if not isinstance(raw, str):
        return "", error_payload("url 必須是字串", "invalid_url")
    stripped = raw.strip()
    if len(stripped) > MAX_URL_LENGTH:
        return "", error_payload(f"URL 過長（{len(stripped)} 字元），上限為 {MAX_URL_LENGTH} 字元", "url_too_long")
    url = normalize_input_url(stripped)
    if not url:
        return "", error_payload("URL 不可為空", "empty_url")
    if len(url) > MAX_URL_LENGTH:
        return "", error_payload(f"URL 過長，上限為 {MAX_URL_LENGTH} 字元", "url_too_long")
    problem = url_structure_error(url)
    if problem:
        return "", error_payload(problem, "invalid_url")
    return url, None


# =============================================================================
# 硬規則與分數融合
# =============================================================================

def check_hard_rules(url: str) -> List[HardRule]:
    """依 rule.target 對 原始(canonical)/解碼/host/path 字串比對，回傳命中的規則清單。"""
    try:
        return match_hard_rules(get_rule_targets(url))
    except Exception:  # noqa: BLE001
        logger.exception("硬規則比對失敗：%s", url[:200])
        return []


# 規則家族：同一家族的多條規則視為同一個證據（不疊加），也用來判斷灰色地帶
# 補強時哪些特徵已被硬規則涵蓋。依名稱前綴對應（長前綴優先）。
_RULE_FAMILY_PREFIXES: Tuple[Tuple[str, str], ...] = (
    ("ip_address_url", "ip"),
    ("url_userinfo_at_spoof", "at_spoof"),
    ("official_domain_in_subdomain", "brand"),
    ("gov_agency_spoof", "brand"),
    ("brand_", "brand"),
    ("tunnel_or_ephemeral_host", "ephemeral"),
    ("ipfs_", "ephemeral"),
    ("punycode_", "punycode"),
    ("double_http", "redirect"),
    ("redirect_", "redirect"),
    ("lottery_", "gambling"),
    ("gambling_", "gambling"),
    ("investment_", "investment"),
    ("finance_term", "investment"),
    ("crypto_", "crypto"),
    ("vip_token", "vip"),
    ("free_hosting", "hosting"),
    ("uniapp_", "app_route"),
    ("register_invite", "app_route"),
    ("invite_code", "app_route"),
    ("mobile_", "mobile"),
    ("app_download", "app_install"),
    ("ios_profile", "app_install"),
    ("apk_ipa", "app_install"),
    ("non_standard_port", "port"),
    ("suspicious_tld_random", "random_domain"),
    ("software_piracy", "piracy"),
    ("social_group", "social"),
    ("line_official", "social"),
    ("shortener_", "shortener"),
)

# 家族 → 已被涵蓋、灰色地帶補強不再重複加分的特徵
_FAMILY_FEATURES: Dict[str, Tuple[str, ...]] = {
    "ip": ("is_ip_address",),
    "at_spoof": ("url_has_at_symbol",),
    "brand": ("brand_impersonation", "brand_typo_like", "levenshtein_brand_dist", "brand_in_sld"),
    "ephemeral": ("tunnel_or_ephemeral_host",),
    "punycode": ("punycode_domain",),
    "redirect": ("double_http",),
    "gambling": ("gambling_keyword", "gambling_number_pattern"),
    "investment": ("investment_lure_keyword",),
    "crypto": ("crypto_exchange_lure",),
    "vip": ("gambling_number_pattern",),
    "hosting": ("free_hosting_platform",),
    "app_route": ("path_scam_route",),
    "app_install": ("path_scam_route",),
    "mobile": ("mobile_lure_path",),
    "port": ("non_standard_port",),
    "random_domain": ("newly_registered_like",),
    "social": ("social_invite_link",),
    "shortener": ("has_shortener",),
}

OFFICIAL_AI_CAP = 35           # 官方網域（未命中硬規則）時 AI 分數上限（低於中風險門檻）
MULTI_RULE_MIN_SCORE = 60      # 參與加成的規則最低分（中高風險）
MULTI_RULE_STEP = 5            # 每多一個獨立家族 +5
MULTI_RULE_CAP = 10            # 加成上限


def rule_family(rule_name: str) -> str:
    for prefix, family in _RULE_FAMILY_PREFIXES:
        if rule_name.startswith(prefix):
            return family
    return rule_name


def combine_hard_rules(rules: List[HardRule]) -> Tuple[int, int, int]:
    """回傳 (最高分, 多條獨立規則加成, 加成後分數)。同一家族只算一次。"""
    if not rules:
        return 0, 0, 0
    top = max(rule.score for rule in rules)
    best_by_family: Dict[str, int] = {}
    for rule in rules:
        if rule.score >= MULTI_RULE_MIN_SCORE:
            fam = rule_family(rule.name)
            best_by_family[fam] = max(best_by_family.get(fam, 0), rule.score)
    independent = max(len(best_by_family) - 1, 0)
    bonus = min(MULTI_RULE_CAP, MULTI_RULE_STEP * independent) if top >= MULTI_RULE_MIN_SCORE else 0
    return top, bonus, min(100, top + bonus)


def covered_features(rules: List[HardRule]) -> Set[str]:
    covered: Set[str] = set()
    for rule in rules:
        covered.update(_FAMILY_FEATURES.get(rule_family(rule.name), ()))
    return covered


def _f(feature_dict: Dict[str, float], name: str, default: float = 0.0) -> float:
    try:
        value = float(feature_dict.get(name, default))
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def _lev_hit(feature_dict: Dict[str, float]) -> bool:
    """0 ≤ levenshtein_brand_dist ≤ 2（0 = 同名但非官方；99 = 無相似品牌）。"""
    return 0 <= _f(feature_dict, "levenshtein_brand_dist", 99) <= 2


# ---- 模型不可用時的特徵加權啟發式評分（rules_only） ----
HEURISTIC_BASE = 8.0
HEURISTIC_CAP = 95.0
HEURISTIC_WEIGHTS: Dict[str, float] = {
    "url_has_at_symbol": 30, "tunnel_or_ephemeral_host": 30, "is_ip_address": 24,
    "brand_impersonation": 18, "brand_typo_like": 15,
    "gambling_keyword": 15, "investment_lure_keyword": 15, "crypto_exchange_lure": 15,
    "path_scam_route": 10, "newly_registered_like": 10, "suspicious_keyword_in_domain": 8,
    "punycode_domain": 8, "free_hosting_platform": 8, "has_scam_word": 6, "brand_in_sld": 6,
    "gambling_number_pattern": 6, "non_standard_port": 6, "has_shortener": 6,
    "social_invite_link": 5, "mobile_lure_path": 4, "double_http": 3, "many_subdomains": 3,
    "long_domain": 2,
}
_LURE_KEYWORDS = ("gambling_keyword", "investment_lure_keyword", "crypto_exchange_lure")


def heuristic_score(feature_dict: Dict[str, float], benign_context: bool = False) -> Tuple[float, Dict[str, float]]:
    """
    以特徵加權估計風險分數（取代 v6 rules_only 的幾個 if）。回傳 (分數, 各項貢獻)。
    單靠弱訊號（http、TLD、行動版路徑、隨機度）最多約 30 分，不會把正常網站推到中風險；
    官方網域（benign_context）上限 30 分。
    """
    parts: Dict[str, float] = {"base": HEURISTIC_BASE}
    for name, weight in HEURISTIC_WEIGHTS.items():
        if _f(feature_dict, name) >= 1:
            parts[name] = weight
    tld_risk = _f(feature_dict, "tld_risk_level")
    if tld_risk >= 2:
        parts["tld_risk_level"] = 8
    elif tld_risk >= 1:
        parts["tld_risk_level"] = 4
    if _f(feature_dict, "is_https", 1) == 0:
        parts["is_https"] = 5          # v7：只代表「明確寫了 http://」
    if _f(feature_dict, "domain_entropy") >= FEATURE_THRESHOLDS.get("domain_entropy", {}).get("warn", 3.5):
        parts["domain_entropy"] = 4
    if _lev_hit(feature_dict) and not (_f(feature_dict, "brand_impersonation") or _f(feature_dict, "brand_typo_like")):
        parts["levenshtein_brand_dist"] = 10
    if tld_risk >= 1 and any(_f(feature_dict, k) >= 1 for k in _LURE_KEYWORDS):
        parts["lure_keyword_risky_tld"] = 6
    score = min(HEURISTIC_CAP, sum(parts.values()))
    if benign_context:
        score = min(score, 30.0)
    return score, parts


# ---- 灰色地帶補強（35～70 分） ----
GREY_ZONE_LOW = 35
GREY_ZONE_HIGH = 70
GREY_BOOST_CAP = 30
GREY_WEAK_CAP = 8              # 弱訊號（http、TLD、行動版、隨機度）合計上限

# (特徵, 加分, 是否強訊號, level, 說明)
_GREY_SIGNALS: Tuple[Tuple[str, int, bool, str, str], ...] = (
    ("url_has_at_symbol", 20, True, "high", "網址含 @ 帳號欄位偽裝，實際連往 @ 後面的網域。"),
    ("tunnel_or_ephemeral_host", 20, True, "high", "使用 ngrok、trycloudflare 等臨時通道或 IPFS 閘道。"),
    ("brand_impersonation", 18, True, "high", "網域使用知名品牌／銀行／交易所／政府名稱但非官方網域。"),
    ("brand_typo_like", 15, True, "high", "偵測到疑似品牌仿冒或拼字變形。"),
    ("gambling_keyword", 15, True, "high", "網址含博弈相關字詞（英文、中文或拼音）。"),
    ("investment_lure_keyword", 15, True, "high", "網址含假投資常見字詞（飆股、投顧、老師帶單、IPO 抽籤…）。"),
    ("crypto_exchange_lure", 15, True, "high", "網址含假交易所／錢包常見字詞（USDT、OTC、staking、airdrop…）。"),
    ("path_scam_route", 10, True, "medium", "網址含假投資／假交易所 App 常見路由或邀請碼參數。"),
    ("newly_registered_like", 10, True, "medium", "網域名稱為無意義隨機字元，符合即用即棄型新域名特徵。"),
    ("punycode_domain", 8, True, "medium", "網域為國際化網域（punycode），可能是同形字仿冒。"),
    ("free_hosting_platform", 8, True, "medium", "網址架在免費架站／暫存平台。"),
    ("gambling_number_pattern", 8, True, "medium", "偵測到 VIP、168、888 等博弈常見數字模式。"),
    ("suspicious_keyword_in_domain", 8, True, "medium", "網域名稱包含投資、加密貨幣或博弈相關字詞。"),
    ("non_standard_port", 6, True, "medium", "網址指定了非標準連接埠。"),
    ("social_invite_link", 5, True, "medium", "網址為社群群組或官方帳號邀請連結。"),
    ("mobile_lure_path", 4, False, "medium", "網址使用 h5、wap 等行動版落地頁格式。"),
)


def apply_grey_zone_boost(
    url: str,
    base_score: float,
    feature_dict: Dict[str, float],
    covered: Optional[Set[str]] = None,
    benign_context: bool = False,
) -> Tuple[float, List[Dict[str, str]]]:
    """
    灰色地帶補強：分數落在 35～70 時，依「尚未被硬規則涵蓋」的高風險特徵加分（上限 +30）。
    官方網域／可信任後綴（benign_context）不加分。url 參數保留舊介面相容。
    """
    reasons: List[Dict[str, str]] = []
    if benign_context or not (GREY_ZONE_LOW <= base_score <= GREY_ZONE_HIGH):
        return base_score, reasons
    covered = covered or set()
    strong = 0
    weak = 0

    def add(key: str, points: int, is_strong: bool, level: str, message: str) -> None:
        nonlocal strong, weak
        if is_strong:
            strong += points
        else:
            weak += points
        reasons.append({"key": f"grey_{key}", "level": level, "message": f"灰色地帶補強：{message}"})

    for name, points, is_strong, level, message in _GREY_SIGNALS:
        if name in covered or _f(feature_dict, name) < 1:
            continue
        add(name, points, is_strong, level, message)

    if "levenshtein_brand_dist" not in covered and _lev_hit(feature_dict) \
            and not (_f(feature_dict, "brand_impersonation") or _f(feature_dict, "brand_typo_like")):
        dist = int(_f(feature_dict, "levenshtein_brand_dist", 99))
        add("levenshtein_brand_dist", 15, True, "high",
            f"網域名稱與知名品牌同名或編輯距離極近（dist={dist}），但不是官方網域。")

    tld_risk = _f(feature_dict, "tld_risk_level")
    if tld_risk >= 2:
        add("tld_risk_level", 8, False, "medium", "使用高度濫用的頂級域名。")
    elif tld_risk >= 1:
        add("tld_risk_level", 4, False, "medium", "使用常被濫用的頂級域名。")
    if _f(feature_dict, "is_https", 1) == 0:
        add("no_https", 5, False, "medium", "網址明確使用未加密的 http://。")
    if _f(feature_dict, "domain_entropy") >= 3.5:
        add("high_entropy", 4, False, "medium", "網域隨機性偏高。")

    boost = min(GREY_BOOST_CAP, strong + min(weak, GREY_WEAK_CAP))
    if boost <= 0:
        return base_score, []
    return min(100.0, base_score + boost), reasons


def get_risk_level(score: float) -> Tuple[str, str]:
    """根據分數回傳風險等級（≥70 high、≥40 medium、其餘 low）。"""
    if score >= 70:
        return "high", "高風險"
    if score >= 40:
        return "medium", "中風險"
    return "low", "低風險"


def get_verdict(level: str) -> str:
    return {"high": "高風險廣告網址", "medium": "中風險，建議查證"}.get(level, "低風險網址")


def get_suggestion(level: str) -> str:
    if level == "high":
        return "此網址符合多項高風險特徵，建議不要輸入個資、信用卡資料或加入陌生社群群組，並可至 165 或官方防詐平台查詢。"
    if level == "medium":
        return "此網址具有部分可疑特徵，建議先確認是否為官方來源，避免輸入個資、驗證碼或付款資訊。"
    return "目前未發現明顯高風險特徵，但仍建議確認網址來源是否可信。"


BLOCKLIST_SUGGESTION = ("此網址已確認為詐騙網站，請勿點擊、輸入任何資料或轉傳連結。"
                        "若有親友受害，請撥打 165 反詐騙專線。")


# ---- deep_scan 頁面文字加分 ----
CONTENT_SIGNAL_POINTS = {"gambling": 6, "investment": 6, "crypto": 6, "scam_generic": 3}
CONTENT_SIGNAL_CAP = 12


def content_signal_boost(semantic: Dict[str, Any]) -> Tuple[int, List[Dict[str, str]]]:
    signals = (semantic or {}).get("content_signals") or {}
    if not semantic or not semantic.get("fetched") or not isinstance(signals, dict):
        return 0, []
    boost = 0
    reasons: List[Dict[str, str]] = []
    for category in signals.get("categories") or []:
        info = signals.get(category) or {}
        points = CONTENT_SIGNAL_POINTS.get(category, 0)
        if not points:
            continue
        boost += points
        terms = "、".join(str(t) for t in (info.get("terms") or [])[:3])
        reasons.append({
            "key": f"content_{category}",
            "level": "medium",
            "message": f"深度掃描：網頁內容出現多個{info.get('label', category)}相關詞（{terms}）。",
        })
    return min(CONTENT_SIGNAL_CAP, boost), reasons


# =============================================================================
# 核心評估
# =============================================================================

@dataclass
class Assessment:
    """單一網址的內部評估結果（build_response 再組成 API 回應）。"""

    url: str
    canonical: str
    hostname: str
    registered: str
    kind: str                       # trusted / blocked / scored
    score: float
    source: str
    feature_dict: Dict[str, float]
    ai_score: float = 0.0
    hard_score: int = 0
    triggered_rules: List[str] = field(default_factory=list)
    reasons: List[Dict[str, str]] = field(default_factory=list)
    degraded: bool = False
    degrade_reason: str = ""
    cacheable: bool = True
    breakdown: Dict[str, Any] = field(default_factory=dict)


def _safe_features(url: str) -> Tuple[Dict[str, float], str]:
    try:
        feature_dict = extract_feature_dict(url)
        if not isinstance(feature_dict, dict):
            raise TypeError("extract_feature_dict 回傳格式錯誤")
        merged = dict(FEATURE_DEFAULTS)
        merged.update(feature_dict)
        return merged, ""
    except Exception as exc:  # noqa: BLE001
        logger.exception("特徵擷取失敗：%s", url[:200])
        return dict(FEATURE_DEFAULTS), f"特徵擷取失敗（{type(exc).__name__}），本次只依硬規則判斷"


def _dedupe_reasons(reasons: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    seen: Set[str] = set()
    unique: List[Dict[str, str]] = []
    for reason in reasons:
        key = str(reason.get("key", ""))
        if key in seen:
            continue
        seen.add(key)
        unique.append(reason)
    return unique


def _trusted_assessment(url: str, canonical: str, hostname: str, registered: str,
                        feature_dict: Dict[str, float]) -> Assessment:
    return Assessment(
        url=url, canonical=canonical, hostname=hostname, registered=registered, kind="trusted",
        score=5, source="trusted_domain", feature_dict=feature_dict,
        reasons=[{"key": "trusted_domain", "level": "safe", "message": "此網址網域位於系統可信任清單。"}],
        # 白名單／黑名單判定不經過模型：degraded 描述「這筆結果」是否受降級影響（系統狀態見 /health）
        degraded=False,
    )


def _blocked_assessment(url: str, canonical: str, hostname: str, registered: str,
                        feature_dict: Dict[str, float]) -> Assessment:
    return Assessment(
        url=url, canonical=canonical, hostname=hostname, registered=registered, kind="blocked",
        score=100, source="blocklist", feature_dict=feature_dict, hard_score=100,
        triggered_rules=["blocklist_match"],
        reasons=[{
            "key": "blocklist_match",
            "level": "high",
            "message": "此網址的網域已被台灣刑事局詐欺犯罪防制中心或 165 反詐騙平台列為封鎖名單，"
                       "確認為詐騙或高風險網站。",
        }],
        degraded=False,
    )


async def assess_url(url: str, depth: int = 0) -> Assessment:
    """
    單一網址評估：白名單 → 黑名單 → 硬規則 → 模型（或啟發式）→ 融合 → 內嵌跳轉目標。
    url 必須是 validate_url_input() 處理過的字串。depth=1 用於內嵌跳轉目標（不再遞迴）。
    """
    canonical = canonicalize_url(url)
    hostname = get_hostname(url)
    registered = get_registered_domain(url)
    feature_dict, feature_error = _safe_features(url)

    redirect_targets: List[str] = []
    if depth == 0:
        try:
            redirect_targets = [t for t in extract_redirect_targets(url, limit=MAX_REDIRECT_TARGETS)
                                if t and t != canonical and not url_structure_error(t)]
        except Exception:  # noqa: BLE001
            redirect_targets = []

    trusted_host = _host_is_trusted(hostname, registered)
    override = whitelist_override_reason(url) if trusted_host else ""

    if trusted_host and not override:
        result = _trusted_assessment(url, canonical, hostname, registered, feature_dict)
        # 可信任平台的跳轉連結（google.com/url?q=…）：目的地不可信時仍要評估目的地
        untrusted_targets = [t for t in redirect_targets if not is_trusted_domain(t)]
        if untrusted_targets:
            await _adopt_redirect_targets(result, untrusted_targets)
        return result

    if is_blocked_domain(url):
        return _blocked_assessment(url, canonical, hostname, registered, feature_dict)

    benign_context = is_official_host(hostname, registered)

    # ---- 硬規則 ----
    hard_matches = check_hard_rules(url)
    hard_top, hard_bonus, hard_effective = combine_hard_rules(hard_matches)
    # 內容平台（新聞／論壇／百科）在未命中硬規則時，比照官方網域：AI 上限、不做灰色地帶補強
    content_capped = (not benign_context and not hard_matches
                      and is_content_platform_host(hostname, registered))
    low_prior = benign_context or content_capped
    covered = covered_features(hard_matches)
    hard_reasons = [
        {"key": rule.name, "level": "high" if rule.score >= 70 else "medium", "message": rule.message}
        for rule in sorted(hard_matches, key=lambda r: -r.score)
    ]
    if hard_bonus:
        hard_reasons.append({
            "key": "multi_rule_bonus",
            "level": "high",
            "message": f"同時命中多條彼此獨立的高風險規則，風險分數額外加 {hard_bonus} 分。",
        })

    # ---- AI 模型（不可用或推論失敗 → 特徵啟發式） ----
    degraded = model_state.degraded
    degrade_reason = model_state.degrade_reason
    cacheable = True
    ai_score = 0.0
    used_model = False
    if feature_error:
        degraded, degrade_reason = True, feature_error
        cacheable = False
    elif model_state.usable:
        try:
            ai_score = await model_state.predict_score_async(feature_dict)
            used_model = True
        except Exception as exc:  # noqa: BLE001 — 推論失敗降級，不回 500
            logger.warning("AI 模型推論失敗，改用規則評分：%s", exc)
            degraded = True
            degrade_reason = f"AI 模型推論失敗（{type(exc).__name__}），本次改用硬規則＋特徵啟發式評分"
            cacheable = False

    breakdown: Dict[str, Any] = {"hard_max": hard_top, "hard_bonus": hard_bonus}
    grey_reasons: List[Dict[str, str]] = []
    if used_model:
        breakdown["ai_raw"] = round(ai_score, 2)
        if low_prior and not hard_matches and ai_score > OFFICIAL_AI_CAP:
            # 已驗證的官方品牌網域（或內容平台）且未命中任何硬規則：純網址特徵的 AI 分數不足以判為中高風險
            ai_score = float(OFFICIAL_AI_CAP)
            if benign_context:
                breakdown["ai_capped_official"] = True
                hard_reasons.append({
                    "key": "official_domain",
                    "level": "safe",
                    "message": "此網址屬於已知品牌或機構的官方網域，且未命中任何高風險規則。",
                })
            else:
                breakdown["ai_capped_content_platform"] = True
                hard_reasons.append({
                    "key": "content_platform",
                    "level": "safe",
                    "message": "此網址屬於新聞、論壇或百科等內容平台，且未命中任何高風險規則；"
                               "文章網址的字面特徵不足以判為詐騙。",
                })
        base = max(ai_score, float(hard_effective))
        source = "hybrid_ai_hard_rule" if hard_top > 0 else "ai_model"
        breakdown["ai"] = round(ai_score, 2)
        score, grey_reasons = apply_grey_zone_boost(url, base, feature_dict, covered, low_prior)
    else:
        source = "rules_only"
        heur, parts = (0.0, {}) if feature_error else heuristic_score(feature_dict, low_prior)
        breakdown["heuristic"] = round(heur, 2)
        breakdown["heuristic_parts"] = parts
        base = max(heur, float(hard_effective))
        if hard_effective >= heur:
            # 硬規則是單一證據：未被涵蓋的其他特徵在灰色地帶仍可補強（啟發式本身已含特徵，不再補強）
            score, grey_reasons = apply_grey_zone_boost(url, base, feature_dict, covered, low_prior)
        else:
            score = base
    breakdown["grey_boost"] = round(score - base, 2)

    if override:
        hard_reasons.append({
            "key": "whitelist_override",
            "level": "medium",
            "message": "此網址位於可信任平台，但內容型態（例如群組邀請、表單、App 側載、跳轉或高誘因字詞）"
                       "常被詐騙利用，已改走完整檢查。",
        })

    feature_reasons = explain_features(url, feature_dict=feature_dict) if not feature_error else []
    result = Assessment(
        url=url, canonical=canonical, hostname=hostname, registered=registered, kind="scored",
        score=min(100.0, max(0.0, score)), source=source, feature_dict=feature_dict,
        ai_score=round(ai_score, 2), hard_score=hard_top,
        triggered_rules=[rule.name for rule in hard_matches],
        reasons=hard_reasons + grey_reasons + feature_reasons,
        degraded=degraded or not used_model, degrade_reason=degrade_reason if (degraded or not used_model) else "",
        cacheable=cacheable, breakdown=breakdown,
    )
    if redirect_targets:
        await _adopt_redirect_targets(result, redirect_targets)
    return result


async def _adopt_redirect_targets(result: Assessment, targets: List[str]) -> None:
    """評估內嵌跳轉目標；目標風險較高時以目標分數為準（取較高分）並附上原因。"""
    best: Optional[Assessment] = None
    for target in targets[:MAX_REDIRECT_TARGETS]:
        try:
            inner = await assess_url(target, depth=1)
        except Exception:  # noqa: BLE001
            logger.exception("內嵌跳轉目標評估失敗：%s", target[:200])
            continue
        if inner.kind == "trusted":
            continue
        if best is None or inner.score > best.score:
            best = inner
    if best is None:
        return
    result.breakdown["redirect_target_score"] = round(best.score, 2)
    if best.score <= result.score:
        return
    level, label = get_risk_level(best.score)
    result.reasons = [{
        "key": "redirect_target",
        "level": "high" if level == "high" else "medium",
        "message": f"此網址內嵌跳轉到 {best.hostname or best.canonical[:80]}，"
                   f"該目的地的風險評分為 {int(round(best.score))} 分（{label}），以較高者為準。",
    }] + [
        {**r, "key": f"redirect_{r.get('key', '')}", "message": f"跳轉目的地：{r.get('message', '')}"}
        for r in best.reasons if r.get("level") in ("high", "medium")
    ] + result.reasons
    result.triggered_rules = result.triggered_rules + [
        f"redirect:{name}" for name in best.triggered_rules if f"redirect:{name}" not in result.triggered_rules
    ]
    result.score = best.score
    result.source = best.source
    result.kind = "blocked" if best.kind == "blocked" else "scored"
    result.ai_score = max(result.ai_score, best.ai_score)
    result.hard_score = max(result.hard_score, best.hard_score)
    result.degraded = result.degraded or best.degraded
    if best.degraded and not result.degrade_reason:
        result.degrade_reason = best.degrade_reason
    result.cacheable = result.cacheable and best.cacheable


def to_compact_dict(response: Dict[str, Any]) -> Dict[str, Any]:
    """
    封包輕量化：只挑出前端擴充套件渲染徽章／清單實際需要的欄位
    （用於 compact=True，例如批次連結掃描）。欄位名稱與型別與完整回應相同。
    """
    if not response.get("ok", False):
        return response
    return {
        "ok": True,
        "url": response.get("url", ""),
        "risk_score": response.get("risk_score", 0),
        "risk_level": response.get("risk_level", "low"),
        "confidence": response.get("confidence", 0.0),
        "source": response.get("source", ""),
        "degraded": bool(response.get("degraded", False)),
        "cache_hit": bool(response.get("cache_hit", False)),
    }


def _response_from_assessment(result: Assessment, debug: bool) -> Dict[str, Any]:
    final_score = int(round(min(100.0, max(0.0, result.score))))
    level, label = get_risk_level(final_score)
    if result.kind == "blocked":
        verdict, suggestion = "高風險廣告網址", BLOCKLIST_SUGGESTION
    else:
        verdict, suggestion = get_verdict(level), get_suggestion(level)

    response: Dict[str, Any] = {
        "ok": True,
        "url": result.url,
        "canonical_url": result.canonical,
        "risk_score": final_score,
        "risk_level": level,
        "risk_label": label,
        "verdict": verdict,
        "source": result.source,
        "confidence": 1.0 if result.kind == "blocked" else round(final_score / 100, 4),
        "ai_score": round(float(result.ai_score), 2),
        "hard_rule_score": int(result.hard_score),
        "triggered_rules": list(result.triggered_rules),
        "reasons": _dedupe_reasons(result.reasons),
        "suggestion": suggestion,
        "model_loaded": model_state.loaded,
        "version_ok": model_state.version_ok,
        "api_version": API_VERSION,
        "feature_version": FEATURE_VERSION,
        "degraded": bool(result.degraded),
        "cache_hit": False,
    }
    if result.degraded:
        response["degrade_reason"] = result.degrade_reason or model_state.degrade_reason or "AI 模型暫不可用"
    if debug:
        response["features"] = dict(result.feature_dict)
        response["feature_names"] = list(FEATURE_NAMES)
        response["feature_assessment"] = assess_all(result.feature_dict)
        response["feature_schema_id"] = FEATURE_SCHEMA_ID
        response["score_breakdown"] = result.breakdown
    return response


def _cache_key(canonical: str, debug: bool, compact: bool, deep_scan: bool) -> str:
    return f"{canonical}|d{int(debug)}c{int(compact)}s{int(deep_scan)}"


async def build_response(
    url: Any,
    debug: bool = False,
    compact: bool = False,
    deep_scan: bool = False,
) -> Dict[str, Any]:
    """
    核心預測流程（v7）：
    1. 輸入驗證（空字串、>2048 字元、非法 IPv6／埠號、不支援的協定 → ok:false）
    2. 快取（key = canonical URL；以 registered domain 建索引）
    3. 白名單（含 TRUSTED_SUFFIXES、WHITELIST_OVERRIDE_*；可信任平台的跳轉仍評估目的地）
    4. 黑名單（registered domain、hostname 與每一層父網域）
    5. 硬規則（依 target）→ AI 模型（不可用時特徵啟發式）→ 融合與灰色地帶補強
    6. 內嵌跳轉目標另外評分取較高分；（可選）deep_scan 頁面文字加分
    7. 回傳可解釋結果（或依 compact 精簡）；錯誤與推論失敗的降級結果不快取
    """
    normalized, error = validate_url_input(url)
    if error is not None:
        return error

    canonical = canonicalize_url(normalized)
    cache_key = _cache_key(canonical, debug, compact, deep_scan)
    cached = prediction_cache.get(cache_key)
    if cached is not None:
        cached_copy = dict(cached)
        cached_copy["url"] = normalized
        cached_copy["cache_hit"] = True
        return cached_copy

    result = await assess_url(normalized)
    response = _response_from_assessment(result, debug)

    if deep_scan and result.kind == "scored":
        try:
            semantic = await asyncio.wait_for(analyze_page_semantics(canonical, client=http_client),
                                              timeout=DEEP_SCAN_TIMEOUT)
        except asyncio.TimeoutError:
            semantic = {"fetched": False, "error": f"深度掃描逾時（{DEEP_SCAN_TIMEOUT:.0f} 秒）"}
        except Exception as exc:  # noqa: BLE001 — 語意分析失敗不影響主要判斷
            logger.warning("deep_scan 失敗：%s", exc)
            semantic = {"fetched": False, "error": f"語意分析失敗：{type(exc).__name__}"}
        # 新聞／查核平台與官方網域的頁面本來就常報導詐騙詞，不依頁面文字加分
        content_exempt = (is_official_host(result.hostname, result.registered)
                          or result.registered in CONTENT_PLATFORM_DOMAINS
                          or any(p in CONTENT_PLATFORM_DOMAINS for p in parent_domains(result.hostname)))
        boost, content_reasons = (0, []) if content_exempt else content_signal_boost(semantic)
        if boost and response["risk_score"] < 100:
            new_score = min(100, response["risk_score"] + boost)
            level, label = get_risk_level(new_score)
            response.update(risk_score=new_score, risk_level=level, risk_label=label,
                            verdict=get_verdict(level), suggestion=get_suggestion(level),
                            confidence=round(new_score / 100, 4))
            response["reasons"] = _dedupe_reasons(content_reasons + response["reasons"])
            if debug:
                response["score_breakdown"]["content_boost"] = boost
        response["semantic_analysis"] = semantic

    final = to_compact_dict(response) if compact else response
    if result.cacheable:
        ttl = {"trusted": TRUSTED_CACHE_TTL, "blocked": BLOCKED_CACHE_TTL}.get(result.kind, SCORED_CACHE_TTL)
        prediction_cache.set(cache_key, final, ttl_seconds=ttl,
                             domain=result.registered or result.hostname, group=canonical)
    return final


async def safe_build_response(url: Any, debug: bool = False, compact: bool = False,
                              deep_scan: bool = False) -> Dict[str, Any]:
    """build_response 外再包一層：任何未預期例外都轉成 ok:false（不快取、不回 500）。"""
    try:
        return await build_response(url, debug=debug, compact=compact, deep_scan=deep_scan)
    except Exception:  # noqa: BLE001
        logger.exception("預測流程發生未預期錯誤：%r", str(url)[:200])
        return error_payload("分析時發生內部錯誤，請稍後再試", "internal_error")


# =============================================================================
# Routes
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def serve_index() -> Any:
    """回傳首頁；若沒有 index.html，回傳簡易 API 狀態頁。"""
    if os.path.exists(INDEX_PATH):
        return FileResponse(INDEX_PATH)
    return HTMLResponse(
        f"""
        <html>
            <head>
                <title>Truth API</title>
                <meta charset="utf-8">
                <style>
                    body {{ font-family: Arial, "Microsoft JhengHei", sans-serif; padding: 40px;
                           background: #f5fbff; color: #1f3b57; }}
                    code {{ background: #e8f4ff; padding: 4px 8px; border-radius: 6px; }}
                </style>
            </head>
            <body>
                <h1>Truth API v{API_VERSION}</h1>
                <p>後端服務已啟動。</p>
                <p>預測 API：<code>POST /predict</code>、<code>POST /predict/batch</code></p>
                <p>健康檢查：<code>GET /health</code></p>
                <p>特徵清單：<code>GET /features</code></p>
            </body>
        </html>
        """
    )


@app.get("/health")
async def health() -> Dict[str, Any]:
    state = model_state.describe()
    return {
        "status": "ok",
        "project": "Truth",
        "api_version": API_VERSION,
        "feature_version": FEATURE_VERSION,
        "feature_schema_id": FEATURE_SCHEMA_ID,
        "feature_names": list(FEATURE_NAMES),
        "feature_count": len(FEATURE_NAMES),
        "rules_version": RULES_VERSION,
        "hard_rules_count": len(HARD_RULES),
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        **state,
        "model_meta": model_state.load_meta(),
        "blocklist_count": len(BLOCKED_DOMAINS),
        "cache": prediction_cache.stats(),
        # v7.2：tldextract 若未安裝，registered domain／SLD 解析語意會不同於正式環境，
        # 不應只在日誌裡悄悄降級——讓 /health 直接回報，維運與測試環境都能一眼看到。
        "tldextract": tldextract_status(),
    }


@app.get("/features")
async def features() -> Dict[str, Any]:
    return {
        "count": len(FEATURE_NAMES),
        "features": list(FEATURE_NAMES),
        "version": FEATURE_VERSION,
        "schema_id": FEATURE_SCHEMA_ID,
        "specs": {name: dict(FEATURE_SPECS.get(name, {})) for name in FEATURE_NAMES},
        "thresholds": FEATURE_THRESHOLDS,
    }


@app.get("/model-report")
async def model_report() -> Dict[str, Any]:
    if not os.path.exists(REPORT_PATH):
        return error_payload("找不到 model_report.json，請先執行 train_model.py", "not_found")
    try:
        with open(REPORT_PATH, "r", encoding="utf-8-sig") as f:
            report = json.load(f)
        return {"ok": True, "report": report}
    except Exception as exc:  # noqa: BLE001
        return error_payload(f"讀取 model_report.json 失敗：{exc}", "read_error")


@app.post("/predict")
async def predict(data: PredictRequest, request: Request) -> Dict[str, Any]:
    # v7.2：/predict 補上限流（見 security.DEFAULT_LIMITS 的 "predict" 門檻）。
    # 先前只限流 /feedback、/blocklist/add，但 /predict 才是系統中最常被呼叫、
    # 也最耗運算資源（特徵擷取＋模型推論）的路由，公開部署時仍需防護。
    enforce_rate_limit(request, "predict")
    return await safe_build_response(data.url, debug=data.debug, compact=data.compact, deep_scan=data.deep_scan)


@app.post("/predict/batch")
async def predict_batch(data: BatchPredictRequest, request: Request) -> Dict[str, Any]:
    """
    批次預測：{urls:[…≤50], debug=false, compact=true} →
    {ok, count, results:{原樣輸入的 url 字串: 單筆回應}}；單筆失敗不影響其他筆。
    """
    # v7.2：batch 門檻比單筆 /predict 更嚴格（見 security.DEFAULT_LIMITS 的
    # "predict_batch"），因為單次請求最多可帶 50 筆網址，運算量遠高於單筆查詢。
    enforce_rate_limit(request, "predict_batch")
    urls = list(data.urls or [])
    if len(urls) > MAX_BATCH_SIZE:
        return error_payload(f"一次最多 {MAX_BATCH_SIZE} 筆網址（收到 {len(urls)} 筆）", "batch_too_large")

    keys: List[str] = []
    raw_by_key: Dict[str, Any] = {}
    for raw in urls:
        key = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False, default=str)
        if key not in raw_by_key:
            raw_by_key[key] = raw
            keys.append(key)

    semaphore = asyncio.Semaphore(BATCH_CONCURRENCY)

    async def run_one(key: str) -> Tuple[str, Dict[str, Any]]:
        async with semaphore:
            return key, await safe_build_response(raw_by_key[key], debug=data.debug, compact=data.compact)

    pairs = await asyncio.gather(*(run_one(k) for k in keys))
    results = {key: value for key, value in pairs}
    return {
        "ok": True,
        "count": len(results),
        "api_version": API_VERSION,
        "degraded": model_state.degraded,
        "results": results,
    }


# =============================================================================
# 使用者回饋（背景寫入 feedback.csv）
# =============================================================================

FEEDBACK_FIELDS = ["url", "label", "ai_score", "note", "time", "source"]
_feedback_lock = threading.Lock()

try:  # 跨行程檔案鎖；未安裝時退回只有執行緒鎖
    from filelock import FileLock as _FileLock
except Exception:  # noqa: BLE001
    _FileLock = None
_feedback_xlocks: Dict[str, Any] = {}


@contextmanager
def _feedback_guard():
    """執行緒鎖 + 跨行程檔案鎖（feedback.csv.lock）；包住 feedback.csv 的讀改寫。"""
    with _feedback_lock:
        if _FileLock is None:
            yield
            return
        lock_path = FEEDBACK_PATH + ".lock"
        xlock = _feedback_xlocks.get(lock_path)
        if xlock is None:
            xlock = _feedback_xlocks[lock_path] = _FileLock(lock_path, timeout=10)
        with xlock:
            yield


def _csv_safe(text: str) -> str:
    """避免試算表公式注入（備註欄以 = + - @ 開頭時前置單引號）。"""
    text = str(text or "")
    return "'" + text if text[:1] in ("=", "+", "-", "@") else text


def _read_feedback_header(path: str) -> List[str]:
    try:
        with open(path, "r", newline="", encoding="utf-8-sig") as f:
            header = next(csv.reader(f), [])
        return [h for h in header if h] or list(FEEDBACK_FIELDS)
    except (OSError, StopIteration, csv.Error):
        return list(FEEDBACK_FIELDS)


def _append_feedback_row(row: Dict[str, Any]) -> None:
    """將單筆回報寫入 feedback.csv（同步 I/O，設計上只從背景任務呼叫）。"""
    path = FEEDBACK_PATH
    with _feedback_guard():
        exists = os.path.exists(path) and os.path.getsize(path) > 0
        fieldnames = _read_feedback_header(path) if exists else list(FEEDBACK_FIELDS)
        with open(path, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if not exists:
                writer.writeheader()
            writer.writerow(row)


def _clean_feedback_file() -> Dict[str, int]:
    """
    feedback.csv 基本清理（為主動學習鋪路）：
      - 依 (url, label) 去除完全重複的回報列，只保留最新一筆；
      - 去除 url 欄位為空的髒資料列。
    以暫存檔＋os.replace 原子寫回。回傳清理前後的列數。
    """
    path = FEEDBACK_PATH
    with _feedback_guard():
        if not os.path.exists(path):
            return {"before": 0, "after": 0}
        with open(path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            fieldnames = [h for h in (reader.fieldnames or []) if h] or list(FEEDBACK_FIELDS)
            rows = list(reader)

        deduped: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for row in rows:
            url_value = (row.get("url") or "").strip()
            if not url_value:
                continue
            deduped.pop((url_value, str(row.get("label", ""))), None)
            deduped[(url_value, str(row.get("label", "")))] = row
        cleaned = list(deduped.values())

        if len(cleaned) != len(rows):
            directory = os.path.dirname(os.path.abspath(path)) or "."
            fd, tmp_path = tempfile.mkstemp(prefix=".tmp-feedback-", suffix=".csv", dir=directory)
            try:
                with os.fdopen(fd, "w", newline="", encoding="utf-8-sig") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                    writer.writeheader()
                    writer.writerows(cleaned)
                for attempt in range(5):   # Windows 上目標檔可能暫時被防毒或試算表鎖住
                    try:
                        os.replace(tmp_path, path)
                        break
                    except PermissionError:
                        if attempt == 4:
                            raise
                        time.sleep(0.05 * (attempt + 1))
            except Exception:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
                raise
        return {"before": len(rows), "after": len(cleaned)}


def process_feedback_in_background(row: Dict[str, Any]) -> None:
    """背景任務：寫入 feedback.csv 後立即清理（回應已先送出，不阻塞使用者）。"""
    try:
        _append_feedback_row(row)
        stats = _clean_feedback_file()
        logger.info("[背景任務] feedback.csv 已更新並清理完成：%d → %d 筆（去重後）",
                    stats["before"], stats["after"])
    except Exception as exc:  # noqa: BLE001 — 背景任務失敗不應影響已回應的請求
        logger.warning("[背景任務] feedback.csv 寫入或清理失敗：%s", exc)


@app.post("/feedback")
async def feedback(data: FeedbackRequest, background_tasks: BackgroundTasks, request: Request) -> Dict[str, Any]:
    """
    使用者回報：label = 1 表示使用者認為是詐騙、0 表示認為正常。
    檔案寫入與清理丟到背景執行；該網址的快取（所有回應變體）立即失效。
    """
    enforce_rate_limit(request, "feedback")
    raw = (data.url or "").strip()
    if len(raw) > MAX_URL_LENGTH:
        return error_payload(f"URL 過長，上限為 {MAX_URL_LENGTH} 字元", "url_too_long")
    url = normalize_input_url(raw)
    if not url:
        return error_payload("URL 不可為空", "empty_url")
    if data.label not in (0, 1):
        return error_payload("label 必須是 0 或 1", "invalid_label")

    ai_score = float(data.ai_score) if math.isfinite(float(data.ai_score)) else 0.0
    row = {
        "url": url,
        "label": int(data.label),
        "ai_score": ai_score,
        "note": _csv_safe(str(data.note or "")[:500]),
        "time": datetime.now().isoformat(),
        "source": "user_feedback",
    }
    background_tasks.add_task(process_feedback_in_background, row)

    removed = prediction_cache.delete_group(canonicalize_url(url))
    return {
        "ok": True,
        "message": "回饋已收到，正在背景寫入並整理，將於下次重新訓練時納入資料。",
        "cache_invalidated": removed,
    }


@app.post("/reload-model")
async def reload_model(request: Request) -> Dict[str, Any]:
    """重新載入模型（管理端點：需 X-Admin-Key；未設金鑰時僅限本機）。"""
    require_admin(request)
    # joblib.load 是同步磁碟 I/O，丟執行緒池；完成後清除全部快取（新模型分數可能不同）
    success = await run_in_threadpool(model_state.load)
    cleared = prediction_cache.clear()
    state = model_state.describe()
    audit_log.record(
        "model.reload",
        actor=reporter_fingerprint(get_client_ip(request)),
        target="scam_model.pkl",
        success=success,
        detail={"cache_cleared": cleared, "error": None if success else model_state.load_error},
    )
    return {
        "ok": success,
        "message": "模型已重新載入" if success else model_state.load_error,
        **({} if success else {"error": model_state.load_error}),
        "model_loaded": state["model_loaded"],
        "version_ok": state["version_ok"],
        "degraded": state["degraded"],
        "degrade_reason": state["degrade_reason"],
        "model_feature_names_ok": state["model_feature_names_ok"],
        "cache_cleared": cleared,
    }


@app.get("/cache/stats")
async def cache_stats() -> Dict[str, Any]:
    """快取觀測端點：容量、命中次數、命中率、淘汰數。"""
    return {"ok": True, "prediction_cache": prediction_cache.stats()}


@app.get("/audit-log")
async def get_audit_log(request: Request, limit: int = 50) -> Dict[str, Any]:
    """
    稽核紀錄查詢（v7.2 新增，管理端點：需 X-Admin-Key；未設金鑰時僅限本機）。
    回傳最近的模型重載、黑名單新增／移除紀錄（新到舊），對應論文第三章
    NIST CSF Govern／Zero Trust 的可問責性（accountability）落地證據。
    """
    require_admin(request)
    limit = max(1, min(int(limit or 50), 500))
    return {"ok": True, "count": limit, "entries": audit_log.tail(limit)}


# =============================================================================
# 黑名單管理 API
# =============================================================================

@app.get("/blocklist")
async def get_blocklist() -> Dict[str, Any]:
    """查詢目前封鎖黑名單域名清單。"""
    return {"ok": True, "count": len(BLOCKED_DOMAINS), "domains": sorted(BLOCKED_DOMAINS)}


@app.get("/blocklist/reports")
async def get_domain_reports() -> Dict[str, Any]:
    """查詢各網域累積的回報數（含尚未達門檻、還在觀察中的網域）。"""
    reports = await run_in_threadpool(blocklist_store.load_domain_reports)
    return {"ok": True, "threshold": REPORT_THRESHOLD, "count": len(reports), "reports": reports}


@app.post("/blocklist/add")
async def add_to_blocklist(data: BlocklistAddRequest, background_tasks: BackgroundTasks, request: Request) -> Dict[str, Any]:
    """
    使用者回報網域為詐騙。同一網域累積「不同回報者」達 REPORT_THRESHOLD 個才正式列入黑名單
    （回報者以 IP 加鹽雜湊的指紋區分，同一人重複回報不加次數；另有速率限制），
    並寫入 dynamic_blocklist.json（重啟後仍保留）。可信任網域只記錄回報、不自動封鎖。
    回報後立即清除該網域（registered domain 索引）的所有快取。
    """
    enforce_rate_limit(request, "blocklist_add")
    raw = (data.domain or "").strip()
    domain = normalize_domain(raw) if raw and len(raw) <= MAX_URL_LENGTH else ""
    if not domain or ("." not in domain and not is_ip_address(domain)):
        return error_payload("無效的域名格式", "invalid_domain")

    fingerprint = reporter_fingerprint(get_client_ip(request))
    result = await run_in_threadpool(blocklist_store.register_domain_report, domain, data.note, fingerprint)

    removed = prediction_cache.delete_domain(get_registered_domain(domain) or domain)
    if get_registered_domain(domain) != domain:
        removed += prediction_cache.delete_domain(domain)

    if not result.get("duplicate"):
        row = {
            "url": domain,
            "label": 1,
            "ai_score": 100.0,
            "note": _csv_safe(f"[blocklist_report {result['count']}/{result['threshold']}] {str(data.note or '')[:300]}"),
            "time": datetime.now().isoformat(),
            "source": "manual_blocklist" if result["blocked"] else "pending_report",
        }
        background_tasks.add_task(process_feedback_in_background, row)

    audit_log.record(
        "blocklist.add",
        actor=fingerprint,
        target=domain,
        success=True,
        detail={
            "report_count": result["count"],
            "threshold": result["threshold"],
            "newly_blocked": result["newly_blocked"],
            "duplicate": bool(result.get("duplicate")),
            "protected": bool(result.get("protected")),
        },
    )

    if result.get("duplicate"):
        message = (f"您已回報過 {domain}，同一回報者重複回報不會增加累積次數"
                   f"（目前 {result['count']}/{result['threshold']}）。")
    elif result.get("protected"):
        message = (f"{domain} 屬於系統可信任網域，已記錄回報（累積 {result['count']} 次），"
                   f"將交由人工審核，不會自動列入黑名單。")
    elif result["newly_blocked"]:
        message = (f"{domain} 累積回報已達 {result['threshold']} 次門檻，"
                   f"已正式加入黑名單，並寫入 dynamic_blocklist.json（重啟伺服器後仍會保留）。")
    elif result["blocked"]:
        message = f"{domain} 先前已達門檻並列入黑名單（目前累積回報 {result['count']} 次）。"
    else:
        remaining = max(result["threshold"] - result["count"], 0)
        message = (f"已記錄回報，{domain} 目前累積 {result['count']}/{result['threshold']} 次回報，"
                   f"還需 {remaining} 次不同回報才會正式列入黑名單。")

    return {
        "ok": True,
        "message": message,
        "domain": domain,
        "report_count": result["count"],
        "threshold": result["threshold"],
        "blocked": result["blocked"],
        "newly_blocked": result["newly_blocked"],
        "protected": bool(result.get("protected")),
        "duplicate": bool(result.get("duplicate")),
        "total_blocked": len(BLOCKED_DOMAINS),
        "cache_invalidated": removed,
    }


@app.delete("/blocklist/{domain:path}")
async def remove_from_blocklist(domain: str, request: Request) -> Dict[str, Any]:
    """
    管理員復原誤封網域（需 X-Admin-Key；未設金鑰時僅限本機）。只能移除動態黑名單
    （dynamic_blocklist.json）內的網域；內建／政府黑名單不可由此移除。
    移除後該網域（registered domain 索引）的快取立即失效，回報累積歸零。
    """
    require_admin(request)
    raw = (domain or "").strip()
    domain = normalize_domain(raw) if raw and len(raw) <= MAX_URL_LENGTH else ""
    if not domain or ("." not in domain and not is_ip_address(domain)):
        return error_payload("無效的域名格式", "invalid_domain")
    result = await run_in_threadpool(blocklist_store.remove_domain, domain)
    if not result["removed"]:
        text = {"builtin": f"{domain} 屬於內建黑名單（程式碼或政府清單），無法透過 API 移除"}.get(
            result["reason"], f"{domain} 不在黑名單中")
        audit_log.record(
            "blocklist.remove",
            actor=reporter_fingerprint(get_client_ip(request)),
            target=domain,
            success=False,
            detail={"reason": result["reason"] or "not_found"},
        )
        return error_payload(text, result["reason"] or "not_found")
    removed_cache = prediction_cache.delete_domain(get_registered_domain(domain) or domain)
    if get_registered_domain(domain) != domain:
        removed_cache += prediction_cache.delete_domain(domain)
    audit_log.record(
        "blocklist.remove",
        actor=reporter_fingerprint(get_client_ip(request)),
        target=domain,
        success=True,
        detail={"cache_invalidated": removed_cache},
    )
    return {
        "ok": True,
        "message": f"已將 {domain} 從動態黑名單移除，回報累積已歸零",
        "domain": domain,
        "total_blocked": len(BLOCKED_DOMAINS),
        "cache_invalidated": removed_cache,
    }


@app.get("/blocklist/sync-165")
async def sync_165_status(request: Request) -> Dict[str, Any]:
    """165 自動同步的設定與上次結果（管理端點：需 X-Admin-Key；未設金鑰時僅限本機）。"""
    require_admin(request)
    return {"ok": True, **sync_165.status(blocklist_store), "total_blocked": len(BLOCKED_DOMAINS)}


@app.post("/blocklist/sync-165")
async def sync_165_now(request: Request, data: Optional[Sync165Request] = None) -> Dict[str, Any]:
    """
    立即從 165 資料集同步涉詐網域到黑名單（管理端點：需 X-Admin-Key；未設金鑰時僅限本機）。
    下載網址只讀環境變數 TRUTHMARK_165_CSV_URL。dry_run=true 只試算。
    """
    require_admin(request)
    data = data or Sync165Request()
    report = await run_165_sync(reporter_fingerprint(get_client_ip(request)),
                                dry_run=data.dry_run, force=data.force)
    if not report["ok"]:
        return error_payload(report["error"] or "165 同步失敗", "sync_165_failed", report=report)
    verb = "試算完成（未寫入）" if data.dry_run else "同步完成"
    return {
        "ok": True,
        "message": f"165 {verb}：新增 {report['added']}、已存在 {report['already']}",
        "report": report,
        "total_blocked": len(BLOCKED_DOMAINS),
    }


# =============================================================================
# 本機執行
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    # 埠號與 run_server.py／Chrome 擴充功能預期的 127.0.0.1:5500 一致。
    # 預設只綁本機；需要讓區域網路其他裝置連線時設定環境變數 HOST=0.0.0.0。
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", 5500))
    logger.info("TruthMark 後端服務啟動（v%s）：http://%s:%d", API_VERSION, host, port)
    uvicorn.run("main:app", host=host, port=port, reload=os.environ.get("TRUTHMARK_RELOAD", "0") == "1")
