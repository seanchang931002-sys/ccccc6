# -*- coding: utf-8 -*-
"""
後端 v7 高可用與融合邏輯測試（main.py / cache.py / blocklist_store.py / text_features.py）。

補 test_api.py 沒涵蓋的部分：
  - 內嵌跳轉目標另外評分取較高分；可信任平台跳轉到不可信目的地仍會評估
  - 白名單覆寫（LINE 群組邀請、sites.google.com、內嵌高風險跳轉）走完整流程
  - 黑名單比對 hostname 的每一層父網域；可信任網域不會被回報自動封鎖
  - 多條獨立硬規則加成有上限、同家族不疊加；灰色地帶補強不重複計算硬規則已涵蓋的訊號
  - 官方網域與內容平台（未命中硬規則）AI 分數上限；heuristic 不會把正常網站推到中風險
  - TTLCache：O(1) LRU、TTL、delete_domain／delete_group／clear
  - BlocklistStore：原子寫入、網域正規化、import_domains、執行緒安全
  - deep_scan 經 /predict 對 127.0.0.1:5500 與 169.254.169.254 被 SSRF 防護拒絕（不發出請求）

隔離：所有寫入都導到暫存目錄；模型只讀（需要模型情境時用暫存目錄的假模型）。
執行（在 va 目錄下）：python -m unittest discover -s tests -p "test_backend_v7.py" -v
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
import uuid
from typing import Any, Dict, List, Tuple

import _support as S  # noqa: E402  (同時把 va/ 加進 sys.path)

import features as F  # noqa: E402
import rules_config as R  # noqa: E402

try:
    from fastapi.testclient import TestClient
except Exception as _exc:  # pragma: no cover
    raise unittest.SkipTest(f"無法載入 fastapi.testclient：{_exc}")

import main as M  # noqa: E402
from blocklist_store import BlocklistStore, normalize_domain  # noqa: E402
from cache import TTLCache  # noqa: E402

_client: TestClient = None  # type: ignore[assignment]
_tmpdir = ""
_undo: List[Tuple[Any, str, Any]] = []
_protected = S.ProtectedFiles()


def _patch(obj: Any, attr: str, value: Any) -> None:
    _undo.append((obj, attr, getattr(obj, attr)))
    setattr(obj, attr, value)


_ADMIN_KEY = "unittest-admin-key"
_ENV: Dict[str, Any] = {}


def setUpModule():  # noqa: N802
    global _client, _tmpdir
    _protected.snapshot()
    _tmpdir = tempfile.mkdtemp(prefix="truthmark-backend-")
    _patch(M, "FEEDBACK_PATH", os.path.join(_tmpdir, "feedback.csv"))
    _patch(M.blocklist_store, "dynamic_blocklist_path", os.path.join(_tmpdir, "dynamic_blocklist.json"))
    _patch(M.blocklist_store, "domain_reports_path", os.path.join(_tmpdir, "domain_reports.json"))
    # /reload-model 為管理端點（需 X-Admin-Key）；本檔以環境變數設定金鑰，並在 post 時帶入
    _ENV["old"] = os.environ.get("TRUTHMARK_ADMIN_KEY")
    os.environ["TRUTHMARK_ADMIN_KEY"] = _ADMIN_KEY
    _client = TestClient(M.app, raise_server_exceptions=False, headers={"X-Admin-Key": _ADMIN_KEY})
    _client.__enter__()


def tearDownModule():  # noqa: N802
    global _client
    try:
        if _client is not None:
            _client.__exit__(None, None, None)
    finally:
        _client = None  # type: ignore[assignment]
        if _ENV.get("old") is None:
            os.environ.pop("TRUTHMARK_ADMIN_KEY", None)
        else:
            os.environ["TRUTHMARK_ADMIN_KEY"] = _ENV["old"]
        for obj, attr, value in reversed(_undo):
            setattr(obj, attr, value)
        touched = _protected.restore()
        if touched:
            sys.stderr.write(f"\n[test_backend_v7] 警告：真實檔案被改動並已還原：{touched}\n")
        shutil.rmtree(_tmpdir, ignore_errors=True)


def predict(url: str, **flags) -> Dict[str, Any]:
    r = _client.post("/predict", json={"url": url, **flags})
    assert r.status_code < 500, r.text[:300]
    return r.json()


def uniq(base: str) -> str:
    """加上隨機參數避開快取。"""
    sep = "&" if "?" in base else "?"
    return f"{base}{sep}tm={uuid.uuid4().hex[:6]}"


class _FakeModel:
    """暫時把 main 的模型路徑指到假模型（恆回固定機率），離開時改回並重新載入。"""

    def __init__(self, probability_one: bool = True):
        self.dir = os.path.join(_tmpdir, f"model-{uuid.uuid4().hex[:6]}")
        self.constant = 1 if probability_one else 0

    def __enter__(self):
        import joblib
        import numpy as np
        import sklearn
        from sklearn.dummy import DummyClassifier

        os.makedirs(self.dir, exist_ok=True)
        clf = DummyClassifier(strategy="constant", constant=self.constant)
        clf.fit(np.zeros((4, len(F.FEATURE_NAMES))), np.array([0, 1, 0, 1]))
        joblib.dump({"format": "truthmark-model-bundle/1", "estimator": clf,
                     "feature_names": list(F.FEATURE_NAMES), "feature_version": F.FEATURE_VERSION,
                     "feature_schema_id": F.FEATURE_SCHEMA_ID, "sklearn_version": sklearn.__version__},
                    os.path.join(self.dir, "scam_model.pkl"))
        self.saved = (M.model_state.model_path, M.model_state.meta_path)
        M.model_state.model_path = os.path.join(self.dir, "scam_model.pkl")
        M.model_state.meta_path = os.path.join(self.dir, "model_meta.json")
        assert _client.post("/reload-model").json().get("ok"), "假模型載入失敗"
        return self

    def __exit__(self, *exc):
        M.model_state.model_path, M.model_state.meta_path = self.saved
        _client.post("/reload-model")
        M.prediction_cache.clear()
        return False


# =============================================================================
# 白名單 / 跳轉 / 黑名單
# =============================================================================

class WhitelistAndRedirectTests(unittest.TestCase):
    def test_trusted_suffix_short_circuit(self):
        for url in ("https://www.ntu.edu.tw/", "https://www.mof.gov.tw/", "165.npa.gov.tw"):
            body = predict(url)
            self.assertEqual(body["source"], "trusted_domain", url)
            self.assertEqual(body["risk_score"], 5, url)

    def test_override_patterns_go_full_pipeline(self):
        for url in ("https://line.me/ti/g2/AbCdEfGhIjKlMnOp", "https://line.me/R/ti/g/AbCdEf123",
                    "https://sites.google.com/view/tw-stock-vip/home",
                    "https://docs.google.com/forms/d/e/1FAIpQLSf/viewform"):
            body = predict(uniq(url))
            self.assertNotEqual(body["source"], "trusted_domain", url)

    def test_line_group_invite_is_medium(self):
        body = predict(uniq("https://line.me/ti/g2/StockVipTeacher888"))
        self.assertIn("social_group_invite", body["triggered_rules"])
        self.assertGreaterEqual(body["risk_score"], 40)

    def test_redirect_target_scored_higher(self):
        body = predict(uniq("https://www.google.com/url?q=https://binance-tw-pro.top/&sa=D"))
        self.assertGreaterEqual(body["risk_score"], 70, body)
        self.assertTrue(any(r.startswith("redirect:") for r in body["triggered_rules"]), body["triggered_rules"])
        self.assertTrue(any(x["key"] == "redirect_target" for x in body["reasons"]))

    def test_trusted_redirector_to_untrusted_target_is_evaluated(self):
        # google.com 跳轉到 .com 網域（不在覆寫樣式內）：外層可信，但仍要評估目的地
        body = predict(uniq("https://www.google.com/url?q=https://binance-tw-pro.com/login&sa=D"))
        self.assertGreaterEqual(body["risk_score"], 70, body)
        # 目的地也可信時維持白名單
        safe = predict(uniq("https://www.google.com/url?q=https://www.books.com.tw/&sa=D"))
        self.assertEqual(safe["source"], "trusted_domain", safe)
        self.assertLess(safe["risk_score"], 40)

    def test_doubleclick_adurl_to_official_not_flagged(self):
        url = ("https://adclick.g.doubleclick.net/pcs/click?xai=AKAOjsu&sig=Cg0ArKJ&urlfix=1"
               "&adurl=https://www.momoshop.com.tw/goods/GoodsDetail.jsp?i_code=12477016")
        self.assertLess(predict(uniq(url))["risk_score"], 40)

    def test_blocklist_parent_domains(self):
        domain = f"tm-parent-{uuid.uuid4().hex[:6]}.xyz"
        M.BLOCKED_DOMAINS.add(domain)
        try:
            for url in (f"https://{domain}/", f"https://a.b.{domain}/x", f"http://www.{domain}:8080/"):
                body = predict(uniq(url))
                self.assertEqual(body["source"], "blocklist", url)
                self.assertEqual(body["risk_score"], 100, url)
            self.assertNotEqual(predict(uniq(f"https://{domain}.example.org/"))["source"], "blocklist",
                                "只有父網域才算命中，不能做字串前綴比對")
        finally:
            M.BLOCKED_DOMAINS.discard(domain)
            M.prediction_cache.clear()

    def test_trusted_domain_report_not_blocked(self):
        for _ in range(M.REPORT_THRESHOLD + 1):
            body = _client.post("/blocklist/add", json={"domain": "www.books.com.tw", "note": "unittest"}).json()
        self.assertTrue(body["ok"], body)
        self.assertTrue(body.get("protected"), body)
        self.assertFalse(body["blocked"], body)
        self.assertNotIn("books.com.tw", M.BLOCKED_DOMAINS)
        self.assertEqual(predict("https://www.books.com.tw/")["source"], "trusted_domain")


# =============================================================================
# 融合邏輯
# =============================================================================

class FusionTests(unittest.TestCase):
    def test_rule_family_and_bonus_cap(self):
        rules = {r.name: r for r in R.HARD_RULES}
        top, bonus, eff = M.combine_hard_rules([rules["gambling_strong_term_host"], rules["gambling_zh_term"]])
        self.assertEqual(bonus, 0, "同家族規則不得疊加")
        self.assertEqual(eff, top)
        many = [rules[n] for n in ("ip_address_url", "url_userinfo_at_spoof", "brand_spoof_finance",
                                   "uniapp_route_with_invite", "gambling_strong_term_host")]
        top, bonus, eff = M.combine_hard_rules(many)
        self.assertEqual(bonus, M.MULTI_RULE_CAP)
        self.assertLessEqual(eff, 100)
        weak = [rules["social_group_invite"], rules["line_official_shortlink"], rules["invite_code_param"]]
        self.assertEqual(M.combine_hard_rules(weak)[1], 0, "低於 60 分的弱規則不加成")

    def test_grey_boost_skips_covered_and_benign(self):
        fd = F.extract_feature_dict("https://tougu-vip888.com/#/pages/login/login?invitecode=AB12CD")
        boosted, reasons = M.apply_grey_zone_boost("x", 50, fd)
        self.assertGreater(boosted, 50)
        covered = {"gambling_keyword", "investment_lure_keyword", "path_scam_route", "gambling_number_pattern",
                   "suspicious_keyword_in_domain"}
        _, reasons2 = M.apply_grey_zone_boost("x", 50, fd, covered=covered)
        keys = {r["key"] for r in reasons2}
        self.assertFalse(keys & {f"grey_{k}" for k in covered}, keys)
        self.assertEqual(M.apply_grey_zone_boost("x", 50, fd, benign_context=True), (50, []))
        self.assertEqual(M.apply_grey_zone_boost("x", 80, fd)[0], 80, "灰色地帶外不補強")
        self.assertLessEqual(boosted - 50, M.GREY_BOOST_CAP)

    def test_lev_zero_counts(self):
        fd = dict(F.FEATURE_DEFAULTS)
        fd["levenshtein_brand_dist"] = 0
        self.assertGreater(M.apply_grey_zone_boost("x", 50, fd)[0], 50, "lev=0（同名非官方）應補強")
        fd["levenshtein_brand_dist"] = 3
        self.assertEqual(M.apply_grey_zone_boost("x", 50, fd)[0], 50, "lev=3 不再補強")
        fd["levenshtein_brand_dist"] = 99
        self.assertEqual(M.apply_grey_zone_boost("x", 50, fd)[0], 50)

    def test_heuristic_weak_signals_stay_low(self):
        for url in ("http://www.cathaylife.com.tw/", "https://www.uniqlo.com/tw/zh_TW/special-feature/h5/",
                    "https://www.threads.net/@zuck", "https://abc.xyz/", "https://www.ptt.cc/bbs/Stock/index.html"):
            score, parts = M.heuristic_score(F.extract_feature_dict(url))
            self.assertLess(score, 40, f"{url} {parts}")

    def test_heuristic_strong_signals(self):
        for url in ("https://www.binance.com@binance-tw-pro.top/login", "https://gentle-river-abc.trycloudflare.com/",
                    "https://stock-vip-tw.top/h5/#/register"):
            score, parts = M.heuristic_score(F.extract_feature_dict(url))
            self.assertGreaterEqual(score, 40, f"{url} {parts}")

    def test_official_domain_ai_cap(self):
        with _FakeModel(probability_one=True):
            official = predict(uniq("https://www.tsmc.com/chinese"), debug=True)
            self.assertEqual(official["source"], "ai_model")
            self.assertLess(official["risk_score"], 40, official.get("score_breakdown"))
            self.assertTrue(official["score_breakdown"].get("ai_capped_official"))
            unknown = predict(uniq("https://example-bakery-taipei.com.tw/menu"))
            self.assertGreaterEqual(unknown["risk_score"], 70, "非官方網域不受上限影響")
            self.assertFalse(unknown["degraded"])

    def test_content_platform_ai_cap(self):
        self.assertTrue(M.is_content_platform_host("news.ltn.com.tw", "ltn.com.tw"))
        self.assertFalse(M.is_content_platform_host("ltn.com.tw.evil.top", "evil.top"))
        self.assertTrue(M.is_official_host("www.threads.net", "threads.net"), "Threads 為 Meta 官方網域")
        with _FakeModel(probability_one=True):
            news = predict(uniq("https://www.ettoday.net/news/20250101/2871234.htm"), debug=True)
            self.assertEqual(news["source"], "ai_model")
            self.assertLess(news["risk_score"], 40, news.get("score_breakdown"))
            self.assertTrue(news["score_breakdown"].get("ai_capped_content_platform"))
            # 命中硬規則時不套用上限（例如內容平台網域被塞進詐騙網域的子網域不算內容平台）
            spoof = predict(uniq("https://ettoday.net.yule8899.vip/register"), debug=True)
            self.assertGreaterEqual(spoof["risk_score"], 70, spoof.get("score_breakdown"))

    def test_ai_mode_hybrid_takes_max(self):
        with _FakeModel(probability_one=False):
            body = predict(uniq("https://stock-vip-tw.top/h5/#/register?invitecode=AB12CD"), debug=True)
            self.assertEqual(body["source"], "hybrid_ai_hard_rule")
            self.assertGreaterEqual(body["risk_score"], body["hard_rule_score"])
            self.assertEqual(body["ai_score"], 0.0)


# =============================================================================
# 輸入驗證與錯誤格式
# =============================================================================

class InputValidationTests(unittest.TestCase):
    def test_structure_errors(self):
        bad = ["http://[abc", "https://[::1", "http://[::1]:99999/", "https://example.com:abc/", "@@@",
               "http://%zz/", "javascript:alert(1)", "file:///etc/passwd", "https://" + "a." * 300 + "com",
               "https://localhost-only/"]
        for url in bad:
            body = predict(url)
            self.assertIs(body["ok"], False, url)
            self.assertEqual(body["error"], body["message"], url)

    def test_accepted_variants(self):
        for url in ("example.com", "ｈｔｔｐｓ：／／ｅｘａｍｐｌｅ．ｃｏｍ", "https://例子.測試/", "HTTPS://WWW.BOOKS.COM.TW./",
                    "http://45.12.34.56:8888/login", "https://[2001:db8::1]:8443/", " https://example.com/​ "):
            self.assertIs(predict(url)["ok"], True, url)

    def test_global_error_shape_for_unknown_route(self):
        r = _client.get("/no-such-route")
        self.assertEqual(r.status_code, 404)
        body = r.json()
        self.assertIs(body["ok"], False)
        self.assertEqual(body["error"], body["message"])


# =============================================================================
# deep_scan SSRF（經 /predict）
# =============================================================================

class DeepScanTests(unittest.TestCase):
    class _NoNetworkClient:
        def __init__(self):
            self.calls = 0

        def __getattr__(self, name):
            def _rec(*_a, **_k):
                self.calls += 1
                raise RuntimeError("測試：不應發出任何請求")
            return _rec

        async def aclose(self):
            return None

    def test_internal_targets_rejected(self):
        fake = self._NoNetworkClient()
        saved = M.http_client
        M.http_client = fake  # type: ignore[assignment]
        try:
            for url in ("http://127.0.0.1:5500/", "http://169.254.169.254/latest/meta-data/",
                        "http://localhost.localdomain:5500/", "http://10.1.2.3/"):
                body = predict(uniq(url), deep_scan=True)
                if body.get("ok") is False:
                    continue
                sem = body.get("semantic_analysis") or {}
                self.assertFalse(sem.get("fetched"), url)
                self.assertTrue(sem.get("blocked"), f"{url} 應被 SSRF 防護拒絕：{sem}")
            self.assertEqual(fake.calls, 0, "SSRF 防護應在發出請求前就拒絕")
        finally:
            M.http_client = saved

    def test_content_signal_boost(self):
        import text_features as T
        signals = T.detect_content_signals("百家樂 娛樂城 真人視訊 首儲送 USDT 出金 老師帶單 飆股")
        self.assertIn("gambling", signals["categories"])
        boost, reasons = M.content_signal_boost({"fetched": True, "content_signals": signals})
        self.assertTrue(0 < boost <= M.CONTENT_SIGNAL_CAP)
        self.assertTrue(reasons)
        self.assertEqual(M.content_signal_boost({"fetched": False, "content_signals": signals}), (0, []))


# =============================================================================
# cache.py
# =============================================================================

class CacheTests(unittest.TestCase):
    def test_lru_eviction_and_move_to_end(self):
        c = TTLCache(max_size=3, default_ttl_seconds=60)
        for k in "abc":
            c.set(k, k.upper(), domain=f"{k}.com")
        self.assertEqual(c.get("a"), "A")         # a 變成最近使用
        c.set("d", "D", domain="d.com")           # 淘汰最久未用的 b
        self.assertIsNone(c.get("b"))
        self.assertEqual(c.get("a"), "A")
        self.assertEqual(c.stats()["evictions"], 1)
        self.assertEqual(c.delete_domain("b.com"), 0, "被淘汰的項目索引也要移除")

    def test_ttl_expiry(self):
        c = TTLCache(max_size=10, default_ttl_seconds=0.05)
        c.set("k", 1)
        self.assertEqual(c.get("k"), 1)
        time.sleep(0.08)
        self.assertIsNone(c.get("k"))
        self.assertEqual(c.stats()["expirations"], 1)
        c.set("z", 1, ttl_seconds=0)
        self.assertIsNone(c.get("z"), "ttl<=0 不寫入")

    def test_domain_and_group_invalidation(self):
        c = TTLCache(max_size=100)
        c.set("https://evil.com/a|d0", 1, domain="evil.com", group="https://evil.com/a")
        c.set("https://evil.com/a|d1", 2, domain="www.evil.com", group="https://evil.com/a")
        c.set("https://m.evil.com/b|d0", 3, domain="EVIL.COM.", group="https://m.evil.com/b")
        c.set("https://good.com/|d0", 4, domain="good.com", group="https://good.com/")
        self.assertEqual(c.delete_group("https://evil.com/a"), 2)
        self.assertEqual(c.delete_domain("evil.com"), 1)
        self.assertEqual(c.get("https://good.com/|d0"), 4)
        self.assertEqual(c.clear(), 1)
        self.assertEqual(len(c), 0)

    def test_thread_safety(self):
        c = TTLCache(max_size=50)

        def worker(n):
            for i in range(500):
                c.set(f"{n}-{i}", i, domain=f"d{i % 7}.com")
                c.get(f"{n}-{i - 1}")
                if i % 50 == 0:
                    c.delete_domain(f"d{i % 7}.com")

        threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertLessEqual(len(c), 50)


# =============================================================================
# blocklist_store.py
# =============================================================================

class BlocklistStoreTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="tm-bl-", dir=_tmpdir)
        self.blocked = set()
        self.store = BlocklistStore(self.dir, self.blocked, threshold=2)

    def test_normalize_domain(self):
        cases = {
            "WWW.Evil-Shop.COM.": "evil-shop.com",
            "https://user:pw@www.evil.com:8443/path?q=1": "evil.com",
            "evil.com/login": "evil.com",
            "娛樂城.vip": F.get_hostname("娛樂城.vip"),
            "": "", "nodot": "", "   ": "", None: "",
        }
        for raw, expected in cases.items():
            self.assertEqual(normalize_domain(raw), expected, repr(raw))

    def test_threshold_and_atomic_files(self):
        r1 = self.store.register_domain_report("WWW.Scam-Site.xyz", "n1")
        self.assertFalse(r1["blocked"])
        r2 = self.store.register_domain_report("scam-site.xyz.", "n2")
        self.assertTrue(r2["newly_blocked"])
        self.assertIn("scam-site.xyz", self.blocked)
        with open(os.path.join(self.dir, "dynamic_blocklist.json"), encoding="utf-8") as fh:
            data = json.load(fh)
        self.assertIn("scam-site.xyz", data["domains"])
        self.assertEqual(data["sources"]["scam-site.xyz"]["source"], "user_report")
        leftovers = [n for n in os.listdir(self.dir) if n.startswith(".tmp-")]
        self.assertEqual(leftovers, [], "原子寫入不應留下暫存檔")
        fresh = set()
        BlocklistStore(self.dir, fresh, threshold=2).load_dynamic_blocklist()
        self.assertIn("scam-site.xyz", fresh)

    def test_import_domains(self):
        stats = self.store.import_domains(["https://a-scam.top/x", "www.a-scam.top", "b-scam.vip", "bad",
                                           "google.com", "x.gov.tw"], source="165_opendata")
        self.assertEqual(stats["added"], 2, stats)
        self.assertEqual(stats["already"], 1, stats)
        self.assertEqual(stats["invalid"], 1, stats)
        self.assertEqual(stats["protected"], 2, stats)
        self.assertEqual(self.blocked, {"a-scam.top", "b-scam.vip"})
        again = self.store.import_domains(["a-scam.top"])
        self.assertEqual(again["added"], 0)

    def test_import_file_csv(self):
        path = os.path.join(self.dir, "165_opendata_test.csv")
        with open(path, "w", encoding="utf-8-sig") as fh:
            fh.write("WEBNAME,WEBURL,CNT\n假投資,https://c-scam.cyou/h5/#/,3\n假博弈,d-scam.icu,1\n")
        stats = self.store.import_file(path)
        self.assertEqual(stats["added"], 2, stats)
        self.assertIn("c-scam.cyou", self.blocked)

    def test_concurrent_reports(self):
        store = BlocklistStore(self.dir, set(), threshold=1000)
        threads = [threading.Thread(target=lambda: [store.register_domain_report("race.xyz") for _ in range(10)])
                   for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(store.load_domain_reports()["race.xyz"]["count"], 50, "併發回報不可少算")


if __name__ == "__main__":
    unittest.main()
