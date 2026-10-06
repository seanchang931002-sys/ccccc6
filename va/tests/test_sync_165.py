# -*- coding: utf-8 -*-
"""
165 涉詐網域自動同步（sync_165.py、/blocklist/sync-165）測試。
全部離線：下載以 httpx.MockTransport 或 monkeypatch 取代，不連任何網站。
"""
import json
import os
import shutil
import tempfile
import unittest
import uuid
from unittest import mock

import httpx

import _support as S  # noqa: F401  # 把 va/ 加進 sys.path
import sync_165 as SY
from blocklist_store import BlocklistStore

HEADER = "民國年月,網域,網站性質,法律依據,聲請單位"


def make_csv(domains, header=HEADER, category="電子商務"):
    lines = [header]
    lines += [f"115年08月,{d},{category},詐欺犯罪危害防制條例第42條,內政部警政署刑事警察局" for d in domains]
    return "\n".join(lines)


def many(n, prefix="scam"):
    return [f"{prefix}{i:04d}-{uuid.uuid4().hex[:6]}.top" for i in range(n)]


class ParsingTests(unittest.TestCase):
    def test_decode_utf8_bom_and_big5(self):
        text = "民國年月,網域\n115年08月,a.top\n"
        self.assertEqual(SY.decode_bytes(text.encode("utf-8-sig")), text)
        self.assertEqual(SY.decode_bytes(text.encode("big5")), text)

    def test_stop_resolve_layout(self):
        rec = SY.extract_records(make_csv(["a-b.top"], category="釣魚網站"))
        self.assertEqual(rec, [{"domain": "a-b.top", "category": "釣魚網站", "period": "115年08月",
                                "agency": "內政部警政署刑事警察局"}])

    def test_legacy_gambling_layout_picks_url_not_name(self):
        text = "網站名稱,網址,件數,統計起始日期,統計結束日期\n某平台,https://x-y.top/a,12,109/03/19,109/03/25\n"
        rec = SY.extract_records(text)
        self.assertEqual(rec[0]["domain"], "https://x-y.top/a")
        self.assertEqual(rec[0]["period"], "109/03/19")

    def test_headerless_falls_back_to_cell_scan(self):
        rec = SY.extract_records("evil-one.top\nevil-two.xyz\n")
        self.assertEqual([r["domain"] for r in rec], ["evil-one.top", "evil-two.xyz"])

    def test_defanged_and_wildcard(self):
        self.assertEqual(SY._clean_cell("hxxps://evil[.]example.cc/p"), "https://evil.example.cc/p")
        self.assertEqual(SY._clean_cell("*.wild.top"), "wild.top")
        self.assertEqual(SY._clean_cell("?shopkings.top"), "shopkings.top")   # 官方資料的殘留字元

    def test_empty_and_blank_rows(self):
        self.assertEqual(SY.extract_records(""), [])
        self.assertEqual(SY.extract_records(f"{HEADER}\n\n,,,,\n"), [])

    def test_shared_host_and_suffix_detection(self):
        for d in ("github.io", "pages.dev", "com.tw", "co.uk"):
            self.assertTrue(SY.is_shared_or_suffix(d), d)
        for d in ("yourname.github.io", "evil-shop.top", "abc.com.tw"):
            self.assertFalse(SY.is_shared_or_suffix(d), d)


class SyncFlowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="truthmark-sync165-")
        self.blocked = set()
        self.store = BlocklistStore(self.tmp, self.blocked, threshold=3,
                                    is_protected=lambda d: d.endswith(".gov.tw") or d == "google.com")
        self.csv_path = os.path.join(self.tmp, "165.csv")
        self.state = os.path.join(self.tmp, "state.json")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write(self, text, enc="utf-8-sig"):
        with open(self.csv_path, "wb") as f:
            f.write(text.encode(enc))

    def _run(self, **kw):
        kw.setdefault("min_rows", 3)
        return SY.sync_165(self.store, local_file=self.csv_path, state_path=self.state, **kw)

    def _dynamic(self):
        with open(os.path.join(self.tmp, "dynamic_blocklist.json"), encoding="utf-8") as f:
            return json.load(f)

    def test_imports_and_records_metadata(self):
        ds = many(5)
        self._write(make_csv(ds, category="假冒電商"))
        r = self._run()
        self.assertTrue(r["ok"], r["error"])
        self.assertEqual((r["rows"], r["added"], r["already"]), (5, 5, 0))
        self.assertTrue(set(ds) <= self.blocked)
        src = self._dynamic()["sources"][ds[0]]
        self.assertEqual(src["source"], SY.SOURCE_NAME)
        self.assertIn("假冒電商", src["note"])
        self.assertIn("115年08月", src["note"])

    def test_big5_file(self):
        ds = many(4)
        self._write(make_csv(ds), enc="big5")
        self.assertEqual(self._run()["added"], 4)

    def test_idempotent_second_run(self):
        self._write(make_csv(many(5)))
        self._run()
        r2 = self._run()
        self.assertEqual((r2["added"], r2["already"]), (0, 5))
        self.assertEqual(len(self._dynamic()["domains"]), 5)

    def test_incremental_adds_only_new(self):
        first = many(5)
        self._write(make_csv(first))
        self._run()
        extra = many(2, "new")
        self._write(make_csv(first + extra))
        r = self._run()
        self.assertEqual((r["added"], r["already"]), (2, 5))

    def test_filters_shared_hosts_protected_invalid_and_duplicates(self):
        good = many(3)
        rows = good + good[:1] + ["github.io", "com.tw", "yourname.github.io", "www.moi.gov.tw",
                                   "google.com", "not a domain"]
        self._write(make_csv(rows))
        r = self._run()
        self.assertTrue(r["ok"], r["error"])
        self.assertEqual(r["skipped_shared"], 2)          # github.io、com.tw
        self.assertEqual(r["protected"], 2)               # moi.gov.tw、google.com
        self.assertEqual(r["invalid"], 1)
        self.assertNotIn("github.io", self.blocked)
        self.assertNotIn("com.tw", self.blocked)
        self.assertIn("yourname.github.io", self.blocked)  # 共用主機的「子網域」照常匯入
        self.assertNotIn("moi.gov.tw", self.blocked)
        self.assertNotIn("google.com", self.blocked)

    def test_too_few_rows_aborts_without_writing(self):
        self._write(make_csv(many(2)))
        r = self._run(min_rows=50)
        self.assertFalse(r["ok"])
        self.assertIn("已中止", r["error"])
        self.assertFalse(self.blocked)
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "dynamic_blocklist.json")))
        self.assertIn("last_error", SY.load_state(self.state))

    def test_mostly_unparsable_aborts(self):
        self._write(make_csv(["bad one", "bad two", "bad three", "ok-one.top"]))
        r = self._run()
        self.assertFalse(r["ok"])
        self.assertFalse(self.blocked)

    def test_dry_run_writes_nothing(self):
        self._write(make_csv(many(5)))
        r = self._run(dry_run=True)
        self.assertTrue(r["ok"] and r["dry_run"])
        self.assertEqual(r["added"], 5)
        self.assertEqual(len(r["added_sample"]), 5)
        self.assertFalse(self.blocked)
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "dynamic_blocklist.json")))
        self.assertFalse(os.path.exists(self.state))

    def test_never_removes_existing_entries(self):
        old = many(3, "old")
        self._write(make_csv(old))
        self._run()
        self._write(make_csv(many(3, "fresh")))          # 新清單完全不含舊網域
        self._run()
        self.assertTrue(set(old) <= self.blocked)

    def test_missing_config_reports_error_instead_of_raising(self):
        with mock.patch.dict(os.environ, {SY.ENV_URL: ""}):
            r = SY.sync_165(self.store, state_path=self.state)
        self.assertFalse(r["ok"])
        self.assertIn(SY.ENV_URL, r["error"])

    def test_concurrent_run_is_refused(self):
        self._write(make_csv(many(5)))
        self.assertTrue(SY._sync_lock.acquire(blocking=False))
        try:
            r = self._run()
        finally:
            SY._sync_lock.release()
        self.assertFalse(r["ok"])
        self.assertIn("進行中", r["error"])
        self.assertFalse(self.blocked)

    def test_etag_roundtrip_and_not_modified(self):
        text = make_csv(many(5))
        calls = []

        def fake_fetch(url, etag=None, last_modified=None, **kw):
            calls.append((etag, last_modified))
            if etag == "v1":
                return {"status": 304, "text": None, "etag": etag, "last_modified": last_modified, "bytes": 0}
            return {"status": 200, "text": text, "etag": "v1", "last_modified": "Sat, 03 Oct 2026 00:00:00 GMT",
                    "bytes": len(text)}

        with mock.patch.object(SY, "fetch_csv", fake_fetch):
            r1 = SY.sync_165(self.store, ["https://example.test/165.csv"], state_path=self.state, min_rows=3)
            r2 = SY.sync_165(self.store, ["https://example.test/165.csv"], state_path=self.state, min_rows=3)
            r3 = SY.sync_165(self.store, ["https://example.test/165.csv"], state_path=self.state, min_rows=3,
                             force=True)
        self.assertEqual((r1["ok"], r1["added"]), (True, 5))
        self.assertEqual((r2["ok"], r2["not_modified"], r2["added"]), (True, 1, 0))
        self.assertEqual(calls[0], (None, None))
        self.assertEqual(calls[1][0], "v1")
        self.assertEqual(calls[2], (None, None))           # force 忽略 ETag
        self.assertEqual(r3["added"], 0)

    def test_configured_urls_parsing(self):
        with mock.patch.dict(os.environ, {SY.ENV_URL: "https://a.test/x.csv, ftp://bad ,http://b.test/y.csv\nnonsense"}):
            self.assertEqual(SY.configured_urls(), ["https://a.test/x.csv", "http://b.test/y.csv"])
        with mock.patch.dict(os.environ, {SY.ENV_HOURS: "abc"}):
            self.assertEqual(SY.configured_interval_hours(), SY.DEFAULT_INTERVAL_HOURS)


class FetchTests(unittest.TestCase):
    @staticmethod
    def _t(handler):
        return httpx.MockTransport(handler)

    def test_200_decodes_and_returns_validators(self):
        body = "民國年月,網域\n115年08月,a.top\n".encode("big5")
        t = self._t(lambda req: httpx.Response(200, content=body, headers={"etag": "abc", "last-modified": "x"}))
        r = SY.fetch_csv("https://example.test/a.csv", transport=t)
        self.assertEqual((r["status"], r["etag"]), (200, "abc"))
        self.assertIn("a.top", r["text"])

    def test_sends_conditional_headers_and_handles_304(self):
        seen = {}

        def handler(req):
            seen.update(req.headers)
            return httpx.Response(304)

        r = SY.fetch_csv("https://example.test/a.csv", etag="abc", last_modified="lm", transport=self._t(handler))
        self.assertEqual(r["status"], 304)
        self.assertEqual(seen.get("if-none-match"), "abc")
        self.assertEqual(seen.get("if-modified-since"), "lm")
        self.assertIn("TruthMark", seen.get("user-agent", ""))

    def test_http_error_raises_sync_error(self):
        with self.assertRaises(SY.SyncError):
            SY.fetch_csv("https://example.test/a.csv", transport=self._t(lambda r: httpx.Response(503)))

    def test_size_limit_enforced(self):
        t = self._t(lambda r: httpx.Response(200, content=b"x" * 5000))
        with self.assertRaises(SY.SyncError) as cm:
            SY.fetch_csv("https://example.test/a.csv", max_bytes=1000, transport=t)
        self.assertIn("上限", str(cm.exception))

    def test_non_http_scheme_rejected(self):
        for bad in ("file:///etc/passwd", "ftp://example.test/a.csv", "gopher://x"):
            with self.assertRaises(SY.SyncError):
                SY.fetch_csv(bad)

    def test_network_failure_wrapped(self):
        def boom(req):
            raise httpx.ConnectError("dns fail")

        with self.assertRaises(SY.SyncError):
            SY.fetch_csv("https://example.test/a.csv", transport=self._t(boom))


# -----------------------------------------------------------------------------
# 端點與整合：POST /blocklist/sync-165、GET /blocklist/sync-165
# -----------------------------------------------------------------------------
try:
    from fastapi.testclient import TestClient
    import main as M
    _HAS_APP = True
except Exception:  # pragma: no cover
    _HAS_APP = False


@unittest.skipUnless(_HAS_APP, "無法載入 fastapi / main")
class Sync165EndpointTests(unittest.TestCase):
    ADMIN = "sync165-test-key"
    ENV_KEYS = ("TRUTHMARK_ADMIN_KEY", "TRUTHMARK_TRUST_PROXY", SY.ENV_URL, SY.ENV_MIN_ROWS, SY.ENV_STATE)

    @classmethod
    def setUpClass(cls):
        import test_api as T
        cls.T = T
        cls._protected = S.ProtectedFiles()
        cls._protected.snapshot()
        cls.tmp = tempfile.mkdtemp(prefix="truthmark-sync165-api-")
        cls.patched = T.redirect_paths(T.REDIRECT_FILES, cls.tmp)
        cls.env_backup = {k: os.environ.get(k) for k in cls.ENV_KEYS}
        os.environ["TRUTHMARK_ADMIN_KEY"] = cls.ADMIN
        os.environ["TRUTHMARK_TRUST_PROXY"] = "1"
        os.environ[SY.ENV_MIN_ROWS] = "3"
        os.environ[SY.ENV_STATE] = os.path.join(cls.tmp, "sync_state.json")
        os.environ.pop(SY.ENV_URL, None)      # 啟動時未設定 → 不會啟動背景排程
        M.rate_limiter.enabled = False
        cls.client = TestClient(M.app, raise_server_exceptions=False)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)
        for k, v in cls.env_backup.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        cls.T.undo_patches(cls.patched)
        cls._protected.restore()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        os.environ[SY.ENV_URL] = "https://example.test/165.csv"
        self.hdr = {"X-Admin-Key": self.ADMIN}

    def tearDown(self):
        os.environ.pop(SY.ENV_URL, None)

    def _fake(self, domains):
        text = make_csv(domains)
        return lambda url, etag=None, last_modified=None, **kw: {
            "status": 200, "text": text, "etag": None, "last_modified": None, "bytes": len(text)}

    def test_requires_admin_key(self):
        self.assertEqual(self.client.post("/blocklist/sync-165").status_code, 401)
        self.assertEqual(self.client.post("/blocklist/sync-165", headers={"X-Admin-Key": "wrong"}).status_code, 401)
        self.assertEqual(self.client.get("/blocklist/sync-165").status_code, 401)

    def test_sync_then_domain_is_blocked_and_stale_cache_cleared(self):
        ds = many(4, "e2e")
        victim = ds[0]
        before = self.client.post("/predict", json={"url": f"https://{victim}/login"}).json()
        self.assertNotEqual(before.get("source"), "blocklist")        # 同步前：可能進快取

        with mock.patch.object(SY, "fetch_csv", self._fake(ds)):
            r = self.client.post("/blocklist/sync-165", headers=self.hdr)
        body = r.json()
        self.assertEqual(r.status_code, 200)
        self.assertTrue(body["ok"], body)
        self.assertEqual(body["report"]["added"], 4)
        self.assertGreaterEqual(body["report"]["cache_cleared"], 1)

        after = self.client.post("/predict", json={"url": f"https://{victim}/login"}).json()
        self.assertEqual(after["source"], "blocklist")                 # 快取已失效、立即命中黑名單
        self.assertEqual(after["risk_level"], "high")
        self.assertIn(victim, self.client.get("/blocklist").json()["domains"])

        audit = self.client.get("/audit-log?limit=20", headers=self.hdr).json()["entries"]
        entry = next(e for e in audit if e["action"] == "blocklist.sync_165")
        self.assertTrue(entry["success"])
        self.assertEqual(entry["detail"]["added"], 4)

    def test_dry_run_does_not_change_blocklist(self):
        ds = many(4, "dry")
        with mock.patch.object(SY, "fetch_csv", self._fake(ds)):
            body = self.client.post("/blocklist/sync-165", headers=self.hdr, json={"dry_run": True}).json()
        self.assertTrue(body["ok"] and body["report"]["dry_run"])
        self.assertEqual(body["report"]["added"], 4)
        domains = self.client.get("/blocklist").json()["domains"]
        self.assertFalse(any(d in domains for d in ds))

    def test_unconfigured_returns_error_payload(self):
        os.environ.pop(SY.ENV_URL, None)
        body = self.client.post("/blocklist/sync-165", headers=self.hdr).json()
        self.assertIs(body["ok"], False)
        self.assertEqual(body["error_code"], "sync_165_failed")
        self.assertIn(SY.ENV_URL, body["error"])

    def test_source_failure_leaves_blocklist_untouched(self):
        count_before = self.client.get("/blocklist").json()["count"]

        def boom(*a, **k):
            raise SY.SyncError("連線失敗：模擬")

        with mock.patch.object(SY, "fetch_csv", boom):
            body = self.client.post("/blocklist/sync-165", headers=self.hdr).json()
        self.assertIs(body["ok"], False)
        self.assertEqual(self.client.get("/blocklist").json()["count"], count_before)

    def test_status_endpoint(self):
        with mock.patch.object(SY, "fetch_csv", self._fake(many(4, "stat"))):
            self.client.post("/blocklist/sync-165", headers=self.hdr)
        body = self.client.get("/blocklist/sync-165", headers=self.hdr).json()
        self.assertTrue(body["ok"] and body["configured"])
        self.assertIsNotNone(body["last_success_at"])
        self.assertEqual(body["last_result"]["added"], 4)


@unittest.skipUnless(_HAS_APP, "無法載入 fastapi / main")
class Sync165SchedulerTests(unittest.TestCase):
    """伺服器啟動後，背景排程要在不呼叫任何端點的情況下自動同步。"""
    ENV_KEYS = ("TRUTHMARK_ADMIN_KEY", SY.ENV_URL, SY.ENV_MIN_ROWS, SY.ENV_STATE, SY.ENV_HOURS)

    def setUp(self):
        import test_api as T
        self.T = T
        self._protected = S.ProtectedFiles()
        self._protected.snapshot()
        self.tmp = tempfile.mkdtemp(prefix="truthmark-sync165-sched-")
        self.patched = T.redirect_paths(T.REDIRECT_FILES, self.tmp)
        self.env_backup = {k: os.environ.get(k) for k in self.ENV_KEYS}
        self.delay_backup = M.SYNC_165_STARTUP_DELAY

    def tearDown(self):
        M.SYNC_165_STARTUP_DELAY = self.delay_backup
        for k, v in self.env_backup.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self.T.undo_patches(self.patched)
        self._protected.restore()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _wait_for(self, predicate, timeout=5.0):
        import time
        end = time.time() + timeout
        while time.time() < end:
            if predicate():
                return True
            time.sleep(0.05)
        return False

    def test_background_sync_runs_after_startup(self):
        ds = many(4, "sched")
        text = make_csv(ds)
        fake = lambda url, etag=None, last_modified=None, **kw: {  # noqa: E731
            "status": 200, "text": text, "etag": None, "last_modified": None, "bytes": len(text)}
        os.environ.update({SY.ENV_URL: "https://example.test/165.csv", SY.ENV_MIN_ROWS: "3",
                           SY.ENV_HOURS: "0", SY.ENV_STATE: os.path.join(self.tmp, "st.json"),
                           "TRUTHMARK_ADMIN_KEY": "k"})
        M.SYNC_165_STARTUP_DELAY = 0.05
        with mock.patch.object(SY, "fetch_csv", fake):
            with TestClient(M.app, raise_server_exceptions=False) as client:
                ok = self._wait_for(lambda: all(d in M.BLOCKED_DOMAINS for d in ds))
                self.assertTrue(ok, "啟動後背景同步沒有把 165 網域加入黑名單")
                self.assertEqual(client.post("/predict", json={"url": f"https://{ds[0]}/"}).json()["source"],
                                 "blocklist")
        # 重新啟動（新行程模擬）：即使不再下載，也要從 dynamic_blocklist.json 還原
        for d in ds:
            M.BLOCKED_DOMAINS.discard(d)
        os.environ.pop(SY.ENV_URL)
        with TestClient(M.app, raise_server_exceptions=False):
            self.assertTrue(all(d in M.BLOCKED_DOMAINS for d in ds), "重啟後未從檔案還原 165 網域")

    def test_startup_sync_ignores_stale_etag(self):
        """狀態檔有 ETag、但黑名單檔不在（例如只掛載 data volume）：啟動同步仍須完整匯入。"""
        ds = many(4, "etag")
        text = make_csv(ds)
        state_file = os.path.join(self.tmp, "st.json")
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump({"urls": {"https://example.test/165.csv": {"etag": "stale", "last_modified": None}}}, f)

        def fake(url, etag=None, last_modified=None, **kw):
            if etag == "stale":     # 真實伺服器會對舊 ETag 回 304
                return {"status": 304, "text": None, "etag": etag, "last_modified": None, "bytes": 0}
            return {"status": 200, "text": text, "etag": "fresh", "last_modified": None, "bytes": len(text)}

        os.environ.update({SY.ENV_URL: "https://example.test/165.csv", SY.ENV_MIN_ROWS: "3",
                           SY.ENV_HOURS: "0", SY.ENV_STATE: state_file, "TRUTHMARK_ADMIN_KEY": "k"})
        M.SYNC_165_STARTUP_DELAY = 0.05
        with mock.patch.object(SY, "fetch_csv", fake):
            with TestClient(M.app, raise_server_exceptions=False):
                self.assertTrue(self._wait_for(lambda: all(d in M.BLOCKED_DOMAINS for d in ds)),
                                "啟動同步沿用了過期 ETag，導致黑名單沒有補回來")

    def test_not_started_when_url_unset(self):
        os.environ.pop(SY.ENV_URL, None)
        M.SYNC_165_STARTUP_DELAY = 0.05
        called = []
        with mock.patch.object(SY, "fetch_csv", lambda *a, **k: called.append(1)):
            with TestClient(M.app, raise_server_exceptions=False):
                import time
                time.sleep(0.3)
        self.assertEqual(called, [])


if __name__ == "__main__":
    unittest.main()
