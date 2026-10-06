# -*- coding: utf-8 -*-
"""
security.py 與 main.py 濫用防範機制測試（論文表 3-1）。

涵蓋：速率限制（HTTP 429 ＋ Retry-After）、回報者指紋去重、管理員金鑰、CORS 白名單、
誤封網域復原（DELETE /blocklist/{domain}）、BlocklistStore 的檔案鎖與回報者邏輯。
全部使用暫存目錄，不會改動真實的 feedback.csv / dynamic_blocklist.json / domain_reports.json。
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import unittest
import uuid

import _support as S  # noqa: F401  （同時把 va/ 加進 sys.path）

import security as SEC
from blocklist_store import BlocklistStore


class RateLimiterTests(unittest.TestCase):
    def test_blocks_after_limit_and_recovers_after_window(self):
        rl = SEC.RateLimiter({"r": (3, 60)})
        for i in range(3):
            self.assertEqual(rl.check("r", "1.1.1.1", now=100.0 + i), (True, 0))
        allowed, retry = rl.check("r", "1.1.1.1", now=103.0)
        self.assertFalse(allowed)
        self.assertGreaterEqual(retry, 1)
        self.assertTrue(rl.check("r", "1.1.1.1", now=161.0)[0], "視窗過後應恢復")

    def test_clients_and_routes_are_independent(self):
        rl = SEC.RateLimiter({"a": (1, 60), "b": (1, 60)})
        self.assertTrue(rl.check("a", "x", now=1.0)[0])
        self.assertFalse(rl.check("a", "x", now=2.0)[0])
        self.assertTrue(rl.check("a", "y", now=2.0)[0])
        self.assertTrue(rl.check("b", "x", now=2.0)[0])

    def test_rejected_requests_do_not_extend_window(self):
        rl = SEC.RateLimiter({"r": (1, 10)})
        self.assertTrue(rl.check("r", "x", now=0.0)[0])
        for t in (1.0, 5.0, 9.0):
            self.assertFalse(rl.check("r", "x", now=t)[0])
        self.assertTrue(rl.check("r", "x", now=10.5)[0])

    def test_disabled_and_unknown_route_always_allowed(self):
        self.assertTrue(SEC.RateLimiter({"r": (1, 60)}, enabled=False).check("r", "x")[0])
        self.assertTrue(SEC.RateLimiter({"r": (1, 60)}).check("other", "x")[0])

    def test_memory_is_bounded(self):
        rl = SEC.RateLimiter({"r": (5, 60)}, max_keys=100)
        for i in range(1000):
            rl.check("r", f"ip-{i}", now=float(i) * 0.001)
        self.assertLessEqual(len(rl._hits), 100)

    def test_default_limits_match_thesis(self):
        self.assertEqual(SEC.DEFAULT_LIMITS["feedback"], (20, 60))
        self.assertEqual(SEC.DEFAULT_LIMITS["blocklist_add"], (10, 60))
        # v7.2：/predict、/predict/batch 補上限流，門檻比 feedback／blocklist_add 寬鬆。
        self.assertEqual(SEC.DEFAULT_LIMITS["predict"], (120, 60))
        self.assertEqual(SEC.DEFAULT_LIMITS["predict_batch"], (30, 60))


class FingerprintAndAdminTests(unittest.TestCase):
    def test_fingerprint_is_salted_stable_and_hides_ip(self):
        a = SEC.reporter_fingerprint("203.0.113.5", salt="s1")
        self.assertEqual(a, SEC.reporter_fingerprint("203.0.113.5", salt="s1"))
        self.assertNotEqual(a, SEC.reporter_fingerprint("203.0.113.5", salt="s2"))
        self.assertNotEqual(a, SEC.reporter_fingerprint("203.0.113.6", salt="s1"))
        self.assertNotIn("203.0.113.5", a)
        self.assertEqual(len(a), 16)

    def test_client_ip_ignores_xff_unless_trusted(self):
        self.assertEqual(SEC.client_ip("9.9.9.9", "1.2.3.4", trust_forwarded=False), "9.9.9.9")
        self.assertEqual(SEC.client_ip("9.9.9.9", "1.2.3.4, 5.6.7.8", trust_forwarded=True), "1.2.3.4")
        self.assertEqual(SEC.client_ip(None, None, trust_forwarded=True), "unknown")

    def test_admin_with_key(self):
        self.assertEqual(SEC.check_admin("k", "8.8.8.8", configured_key="k"), (True, ""))
        self.assertEqual(SEC.check_admin("bad", "127.0.0.1", configured_key="k"), (False, "admin_key_invalid"))
        self.assertEqual(SEC.check_admin(None, "127.0.0.1", configured_key="k"), (False, "admin_key_invalid"))

    def test_admin_without_key_is_loopback_only(self):
        self.assertTrue(SEC.check_admin(None, "127.0.0.1", configured_key="")[0])
        self.assertTrue(SEC.check_admin(None, "::1", configured_key="")[0])
        self.assertEqual(SEC.check_admin(None, "8.8.8.8", configured_key=""), (False, "admin_key_not_configured"))

    def test_cors_defaults_are_a_whitelist(self):
        origins, regex = SEC.cors_settings()
        self.assertNotIn("*", origins)
        self.assertIn("http://127.0.0.1:5500", origins)
        self.assertRegex("chrome-extension://" + "a" * 32, regex)
        self.assertNotRegex("https://evil.example", regex)


class TrustedProxyCidrTests(unittest.TestCase):
    """v7.2 新增：TRUTHMARK_TRUST_PROXY_CIDRS 的網段比對，取代／補充全域布林開關。"""

    def setUp(self):
        self._orig = os.environ.get("TRUTHMARK_TRUST_PROXY_CIDRS")

    def tearDown(self):
        if self._orig is None:
            os.environ.pop("TRUTHMARK_TRUST_PROXY_CIDRS", None)
        else:
            os.environ["TRUTHMARK_TRUST_PROXY_CIDRS"] = self._orig

    def test_peer_inside_cidr_is_trusted(self):
        os.environ["TRUTHMARK_TRUST_PROXY_CIDRS"] = "10.0.0.0/8,192.168.1.0/24"
        self.assertTrue(SEC.is_trusted_proxy_peer("10.1.2.3"))
        self.assertTrue(SEC.is_trusted_proxy_peer("192.168.1.50"))
        self.assertFalse(SEC.is_trusted_proxy_peer("203.0.113.9"))

    def test_no_cidr_configured_trusts_nothing(self):
        os.environ.pop("TRUTHMARK_TRUST_PROXY_CIDRS", None)
        self.assertFalse(SEC.is_trusted_proxy_peer("10.1.2.3"))

    def test_invalid_peer_ip_is_never_trusted(self):
        os.environ["TRUTHMARK_TRUST_PROXY_CIDRS"] = "10.0.0.0/8"
        self.assertFalse(SEC.is_trusted_proxy_peer("not-an-ip"))
        self.assertFalse(SEC.is_trusted_proxy_peer(None))

    def test_client_ip_honours_cidr_without_global_flag(self):
        os.environ["TRUTHMARK_TRUST_PROXY_CIDRS"] = "10.0.0.0/8"
        os.environ["TRUTHMARK_TRUST_PROXY"] = "0"
        # 連線來源在受信任網段內：即使沒開全域 TRUST_PROXY，也採用 XFF 第一段。
        self.assertEqual(SEC.client_ip("10.0.0.5", "203.0.113.9, 10.0.0.5"), "203.0.113.9")
        # 連線來源不在受信任網段：忽略 XFF，直接用 TCP 層的真實來源。
        self.assertEqual(SEC.client_ip("203.0.113.77", "1.2.3.4"), "203.0.113.77")


class AuditLogTests(unittest.TestCase):
    """v7.2 新增：管理動作（模型重載、黑名單新增／移除）的稽核紀錄。"""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="truthmark_audit_")
        self.path = os.path.join(self.tmp_dir, "audit.log")
        self.log = SEC.AuditLog(self.path)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_record_writes_jsonl_and_tail_reads_back_newest_first(self):
        self.log.record("blocklist.add", actor="fp1", target="a.example.com", success=True, detail={"count": 1})
        self.log.record("blocklist.add", actor="fp2", target="b.example.com", success=True, detail={"count": 2})
        self.log.record("model.reload", actor="fp3", target="scam_model.pkl", success=False)

        entries = self.log.tail(10)
        self.assertEqual(len(entries), 3)
        # tail() 回傳新到舊：最後寫入的 model.reload 應排在第一筆。
        self.assertEqual(entries[0]["action"], "model.reload")
        self.assertFalse(entries[0]["success"])
        self.assertEqual(entries[-1]["target"], "a.example.com")

        # 每一行必須是獨立、可被解析的 JSON（JSON Lines 格式）。
        with open(self.path, "r", encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if line.strip()]
        self.assertEqual(len(lines), 3)
        for line in lines:
            parsed = json.loads(line)
            self.assertIn("time", parsed)
            self.assertIn("actor", parsed)

    def test_tail_limit_is_respected(self):
        for i in range(5):
            self.log.record("blocklist.add", actor="fp", target=f"d{i}.example.com")
        self.assertEqual(len(self.log.tail(2)), 2)

    def test_does_not_leak_raw_ip_only_fingerprint(self):
        # 呼叫端應傳入 reporter_fingerprint() 的結果，而非原始 IP；
        # 這裡驗證 AuditLog 本身不會自作主張把任意字串當成 IP 處理或還原。
        self.log.record("blocklist.add", actor=SEC.reporter_fingerprint("203.0.113.5"), target="x.example.com")
        entries = self.log.tail(1)
        self.assertNotIn("203.0.113.5", json.dumps(entries))


class BlocklistStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="truthmark-sec-")
        self.blocked = set()
        self.store = BlocklistStore(self.tmp, self.blocked, threshold=3, is_protected=lambda d: d == "google.com")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_same_reporter_counts_once(self):
        r1 = self.store.register_domain_report("evil-a.top", "x", reporter="fp1")
        r2 = self.store.register_domain_report("evil-a.top", "x", reporter="fp1")
        self.assertEqual((r1["count"], r1["duplicate"]), (1, False))
        self.assertEqual((r2["count"], r2["duplicate"]), (1, True))
        self.assertFalse(r2["blocked"])

    def test_blocks_only_after_threshold_distinct_reporters(self):
        for i in range(2):
            self.assertFalse(self.store.register_domain_report("evil-b.top", reporter=f"fp{i}")["blocked"])
        last = self.store.register_domain_report("evil-b.top", reporter="fp2")
        self.assertTrue(last["blocked"] and last["newly_blocked"])
        self.assertIn("evil-b.top", self.blocked)

    def test_reporter_hash_only_is_persisted(self):
        self.store.register_domain_report("evil-c.top", reporter="abcdef0123456789")
        data = json.load(open(os.path.join(self.tmp, "domain_reports.json"), encoding="utf-8"))
        self.assertEqual(data["evil-c.top"]["reporters"], ["abcdef0123456789"])

    def test_legacy_call_without_reporter_still_increments(self):
        self.assertEqual(self.store.register_domain_report("evil-d.top")["count"], 1)
        self.assertEqual(self.store.register_domain_report("evil-d.top")["count"], 2)

    def test_protected_domain_never_auto_blocked(self):
        for i in range(5):
            res = self.store.register_domain_report("google.com", reporter=f"fp{i}")
        self.assertTrue(res["protected"])
        self.assertFalse(res["blocked"])

    def test_remove_domain_resets_reports(self):
        for i in range(3):
            self.store.register_domain_report("evil-e.top", reporter=f"fp{i}")
        self.assertIn("evil-e.top", self.blocked)
        res = self.store.remove_domain("evil-e.top")
        self.assertTrue(res["removed"])
        self.assertNotIn("evil-e.top", self.blocked)
        dyn = json.load(open(os.path.join(self.tmp, "dynamic_blocklist.json"), encoding="utf-8"))
        self.assertNotIn("evil-e.top", dyn["domains"])
        rep = json.load(open(os.path.join(self.tmp, "domain_reports.json"), encoding="utf-8"))
        self.assertEqual(rep["evil-e.top"]["count"], 0)
        # 復原後同一批回報者不會立刻再把它封回去（指紋已清空，需重新累積）
        again = self.store.register_domain_report("evil-e.top", reporter="fp0")
        self.assertEqual(again["count"], 1)

    def test_remove_unknown_and_builtin(self):
        self.blocked.add("builtin-bad.example")
        self.assertEqual(self.store.remove_domain("builtin-bad.example")["reason"], "builtin")
        self.assertIn("builtin-bad.example", self.blocked)
        self.assertEqual(self.store.remove_domain("nope.example")["reason"], "not_found")

    def test_concurrent_reports_are_not_lost(self):
        errors = []

        def work(i):
            try:
                self.store.register_domain_report("evil-f.top", reporter=f"fp{i}")
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=work, args=(i,)) for i in range(12)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        self.assertEqual(errors, [])
        data = json.load(open(os.path.join(self.tmp, "domain_reports.json"), encoding="utf-8"))
        self.assertEqual(data["evil-f.top"]["count"], 12)


try:
    from fastapi.testclient import TestClient
    import main as M
    _HAS_APP = True
except Exception:  # pragma: no cover
    _HAS_APP = False


@unittest.skipUnless(_HAS_APP, "無法載入 fastapi / main")
class EndpointSecurityTests(unittest.TestCase):
    ADMIN = "endpoint-test-key"

    @classmethod
    def setUpClass(cls):
        import test_api as T  # 重用路徑重導向，確保不寫到真實檔案
        cls.T = T
        cls._protected = S.ProtectedFiles()
        cls._protected.snapshot()
        cls.tmp = tempfile.mkdtemp(prefix="truthmark-sec-api-")
        cls.patched = T.redirect_paths(T.REDIRECT_FILES, cls.tmp)
        cls.env_backup = {k: os.environ.get(k) for k in ("TRUTHMARK_ADMIN_KEY", "TRUTHMARK_TRUST_PROXY")}
        os.environ["TRUTHMARK_ADMIN_KEY"] = cls.ADMIN
        os.environ["TRUTHMARK_TRUST_PROXY"] = "1"
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
        M.rate_limiter.enabled = True
        M.rate_limiter.reset()

    def tearDown(self):
        M.rate_limiter.reset()
        M.rate_limiter.enabled = False

    def _post(self, path, payload, ip="198.51.100.1", **headers):
        h = {"X-Forwarded-For": ip, **headers}
        return self.client.post(path, json=payload, headers=h)

    def _delete(self, domain, ip="198.51.100.1", **headers):
        return self.client.delete(f"/blocklist/{domain}", headers={"X-Forwarded-For": ip, **headers})

    def test_feedback_rate_limited_with_429(self):
        url = f"https://rl-{uuid.uuid4().hex[:6]}.example.com/"
        codes = [self._post("/feedback", {"url": url, "label": 0, "ai_score": 10.0}).status_code for _ in range(21)]
        self.assertEqual(codes[:20], [200] * 20)
        self.assertEqual(codes[20], 429)
        r = self._post("/feedback", {"url": url, "label": 0, "ai_score": 10.0})
        self.assertEqual(r.status_code, 429)
        self.assertIn("Retry-After", r.headers)
        body = r.json()
        self.assertIs(body["ok"], False)
        self.assertTrue(body["error"])
        # 換一個 IP 不受影響
        self.assertEqual(self._post("/feedback", {"url": url, "label": 0, "ai_score": 10.0}, ip="198.51.100.2").status_code, 200)

    def test_blocklist_add_rate_limited_after_10(self):
        codes = [self._post("/blocklist/add", {"domain": f"rl-{i}-{uuid.uuid4().hex[:5]}.top"}).status_code for i in range(11)]
        self.assertEqual(codes[:10], [200] * 10)
        self.assertEqual(codes[10], 429)

    def test_duplicate_reporter_does_not_add_count(self):
        d = f"dup-{uuid.uuid4().hex[:8]}.top"
        r1 = self._post("/blocklist/add", {"domain": d}, ip="203.0.113.10").json()
        r2 = self._post("/blocklist/add", {"domain": d}, ip="203.0.113.10").json()
        r3 = self._post("/blocklist/add", {"domain": d}, ip="203.0.113.11").json()
        self.assertEqual((r1["report_count"], r1["duplicate"]), (1, False))
        self.assertEqual((r2["report_count"], r2["duplicate"]), (1, True))
        self.assertEqual(r3["report_count"], 2)
        self.assertFalse(r3["blocked"])

    def test_untrusted_xff_is_ignored_by_default(self):
        os.environ["TRUTHMARK_TRUST_PROXY"] = "0"
        try:
            d = f"xff-{uuid.uuid4().hex[:8]}.top"
            # 偽造不同 XFF 也被視為同一個 IP（TestClient 的 client.host）→ 第二次是 duplicate
            self._post("/blocklist/add", {"domain": d}, ip="1.1.1.1")
            r = self._post("/blocklist/add", {"domain": d}, ip="2.2.2.2").json()
            self.assertTrue(r["duplicate"])
        finally:
            os.environ["TRUTHMARK_TRUST_PROXY"] = "1"

    def test_reload_model_requires_admin_key(self):
        self.assertEqual(self._post("/reload-model", None).status_code, 401)
        self.assertEqual(self._post("/reload-model", None, **{"X-Admin-Key": "wrong"}).status_code, 401)
        ok = self.client.post("/reload-model", headers={"X-Admin-Key": self.ADMIN})
        self.assertEqual(ok.status_code, 200)
        self.assertIn("model_loaded", ok.json())

    def test_admin_endpoints_denied_remote_when_key_unset(self):
        os.environ["TRUTHMARK_ADMIN_KEY"] = ""
        try:
            r = self.client.post("/reload-model")  # TestClient 的 client.host = "testclient"（非 loopback）
            self.assertEqual(r.status_code, 403)
        finally:
            os.environ["TRUTHMARK_ADMIN_KEY"] = self.ADMIN

    def test_blocklist_remove_flow(self):
        d = f"mis-{uuid.uuid4().hex[:8]}.top"
        for i in range(3):
            body = self._post("/blocklist/add", {"domain": d}, ip=f"203.0.113.{20 + i}").json()
        self.assertTrue(body["blocked"])
        self.assertEqual(self._delete(d).status_code, 401)
        r = self._delete(d, **{"X-Admin-Key": self.ADMIN})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["ok"], r.text)
        self.assertNotIn(d, self.client.get("/blocklist").json()["domains"])
        again = self._delete(d, **{"X-Admin-Key": self.ADMIN}).json()
        self.assertIs(again["ok"], False)

    def test_blocklist_remove_rejects_invalid_domain(self):
        r = self._delete("nodot", **{"X-Admin-Key": self.ADMIN}).json()
        self.assertIs(r["ok"], False)

    def test_cors_allows_whitelist_and_blocks_others(self):
        ok = self.client.options("/predict", headers={"Origin": "http://127.0.0.1:5500",
                                                      "Access-Control-Request-Method": "POST"})
        self.assertEqual(ok.headers.get("access-control-allow-origin"), "http://127.0.0.1:5500")
        ext = self.client.options("/predict", headers={"Origin": "chrome-extension://" + "a" * 32,
                                                       "Access-Control-Request-Method": "POST"})
        self.assertEqual(ext.headers.get("access-control-allow-origin"), "chrome-extension://" + "a" * 32)
        dele = self.client.options("/blocklist/x.com", headers={"Origin": "http://127.0.0.1:5500",
                                                                "Access-Control-Request-Method": "DELETE"})
        self.assertEqual(dele.status_code, 200)
        bad = self.client.options("/predict", headers={"Origin": "https://evil.example",
                                                       "Access-Control-Request-Method": "POST"})
        self.assertIsNone(bad.headers.get("access-control-allow-origin"))


if __name__ == "__main__":
    unittest.main()
