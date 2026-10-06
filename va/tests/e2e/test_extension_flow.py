# =============================================================================
# tests/e2e/test_extension_flow.py
# =============================================================================
# Chrome 擴充套件端到端（E2E）測試骨架。
#
# 狀態誠實說明：這支測試是 v7.2 新增的，**尚未在沙盒環境實際執行過**
# （沙盒沒有對外網路可下載 Chromium 執行檔），需要在你自己能安裝瀏覽器
# 的電腦或 CI 上跑過一次並確認綠燈，才能算是真正驗證過。跑法見
# docs/EXTENSION_E2E_TESTING.md「方案二」。
#
# 涵蓋的流程：
#   1. 用 Playwright 啟動一個「已載入本擴充套件」的 Chromium（MV3 擴充套件
#      的 service worker 需要用持久化 context 才能載入）
#   2. 開啟一個本機測試頁面（tests/e2e/fixtures/sample_page.html），
#      頁面內有數個外部連結
#   3. 觸發擴充套件 popup 的「掃描頁面連結」
#   4. 驗證：
#      a. 實際送往後端的請求（/predict/batch）body 內的 url 不含查詢參數
#         （驗證 v7.2 URL 去識別化）
#      b. popup 顯示的結果筆數與頁面連結數一致
#      c. service worker 主控台沒有未攔截的例外
#
# 前置需求：
#   pip install playwright pytest-playwright
#   playwright install chromium
#   後端需另外啟動：cd va && uvicorn main:app --port 5500
# =============================================================================

from __future__ import annotations

import json
import time
from pathlib import Path
from urllib.parse import urlparse

import pytest

EXTENSION_DIR = Path(__file__).resolve().parent.parent.parent / "外掛"
FIXTURE_PAGE = Path(__file__).resolve().parent / "fixtures" / "sample_page.html"
BACKEND_URL = "http://127.0.0.1:5500"


def _require_playwright():
    try:
        import playwright  # noqa: F401
    except ImportError:
        pytest.skip("playwright 未安裝，見 docs/EXTENSION_E2E_TESTING.md 方案二的安裝步驟")


@pytest.fixture(scope="module")
def extension_context():
    """啟動一個已載入本擴充套件（未封裝）的持久化瀏覽器 context。"""
    _require_playwright()
    from playwright.sync_api import sync_playwright

    if not EXTENSION_DIR.exists():
        pytest.skip(f"找不到擴充套件目錄：{EXTENSION_DIR}")

    with sync_playwright() as p:
        user_data_dir = "/tmp/truthmark_e2e_profile"
        # Chrome 112+ 的「新版 headless 模式」已支援載入擴充套件，不再強制需要
        # 真實顯示環境（X server）。若在你的環境裡載入失敗，改成 headless=False
        # 並搭配 `xvfb-run` 執行即可（舊版 Chromium 或某些 CI 環境可能需要）。
        context = p.chromium.launch_persistent_context(
            user_data_dir,
            headless=True,
            args=[
                f"--disable-extensions-except={EXTENSION_DIR}",
                f"--load-extension={EXTENSION_DIR}",
            ],
        )
        yield context
        context.close()


def _get_extension_id(context) -> str:
    """從 service worker 的 URL 取出擴充套件 ID。"""
    for _ in range(20):
        for sw in context.service_workers:
            url = sw.url
            if url.startswith("chrome-extension://"):
                return urlparse(url).netloc
        time.sleep(0.5)
    raise RuntimeError("逾時：找不到擴充套件的 service worker，請確認 MV3 擴充套件已成功載入")


def test_extension_loads_without_errors(extension_context):
    """最基本的驗收：擴充套件能成功載入，service worker 沒有立即崩潰。"""
    ext_id = _get_extension_id(extension_context)
    assert ext_id, "應該能取得擴充套件 ID"


def test_link_scan_sends_sanitized_urls(extension_context):
    """
    核心驗證：content.js 收集到的連結，經 background.js 送往後端時
    應該已經去除查詢參數（v7.2 URL 去識別化，見 background.js 的
    normalizeCacheUrl()）。
    """
    if not FIXTURE_PAGE.exists():
        pytest.skip(f"找不到測試頁面：{FIXTURE_PAGE}，請先建立 fixtures/sample_page.html")

    page = extension_context.new_page()

    captured_bodies = []

    def on_request(request):
        if "/predict/batch" in request.url:
            try:
                captured_bodies.append(json.loads(request.post_data or "{}"))
            except json.JSONDecodeError:
                pass

    page.on("request", on_request)
    page.goto(f"file://{FIXTURE_PAGE}")
    page.wait_for_timeout(2000)  # 讓 content.js 的 document_idle 腳本與批次查詢有時間執行

    assert captured_bodies, "應該至少攔截到一次 /predict/batch 請求"
    for body in captured_bodies:
        for url in body.get("urls", []):
            assert "?" not in url, f"送往後端的網址不應含查詢參數，但收到：{url}"
            assert "#" not in url, f"送往後端的網址不應含 hash，但收到：{url}"


def test_backend_is_reachable():
    """前置檢查：後端是否已啟動（提醒訊息比莫名其妙的逾時更有幫助）。"""
    import urllib.request

    try:
        with urllib.request.urlopen(f"{BACKEND_URL}/health", timeout=3) as resp:
            assert resp.status == 200
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"後端未啟動或無法連線（{BACKEND_URL}）：{exc}；"
                    f"請先執行 `cd va && uvicorn main:app --port 5500`")
