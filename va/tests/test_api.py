# -*- coding: utf-8 -*-
"""
main.py API 測試（FastAPI TestClient）＋以 tests/fixtures/url_cases.json 做端到端回歸。

隔離原則（絕不改壞真實資料）：
  - feedback.csv / dynamic_blocklist.json / domain_reports.json：測試開始前把 main 模組（以及其內
    物件，例如 blocklist_store）中指向這些檔案的路徑全部改指到暫存目錄；另外在模組開始前先快照
    真實檔案位元組，結束時比對並還原（雙重保險）。
  - scam_model.pkl / model_meta.json：只讀。需要「模型不可用 / 特徵不一致 / 新格式 bundle」情境時，
    把模型路徑暫時指到暫存目錄內的假模型，再呼叫 /reload-model，測試後改回並重新載入。

回歸門檻（CONTRACT §6，不得為了通過而放寬）：
  - benign 誤判率（最終分數 ≥ 40）≤ 3%
  - scam 召回率（最終分數 ≥ 40）≥ 90%
  - min_score 達成率 ≥ 85%

執行（在 va 目錄下）：
    python -m unittest discover -s tests -v
環境變數 TRUTHMARK_REGRESSION_REPORT=<路徑> 可把逐筆回歸結果另存成 JSON。
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
import types
import unittest
import uuid
import warnings
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import _support as S  # noqa: E402  (同時把 va/ 加進 sys.path)

import features as F  # noqa: E402

try:
    from fastapi.testclient import TestClient
except Exception as _exc:  # pragma: no cover
    raise unittest.SkipTest(f"無法載入 fastapi.testclient：{_exc}")

import main as M  # noqa: E402

BENIGN_FP_MAX = 0.03
SCAM_RECALL_MIN = 0.90
MIN_SCORE_RATE_MIN = 0.85

REDIRECT_FILES = {"feedback.csv", "dynamic_blocklist.json", "domain_reports.json"}
MODEL_FILES = {"scam_model.pkl", "model_meta.json"}

FULL_FIELDS = (
    "ok", "url", "risk_score", "risk_level", "risk_label", "verdict", "source", "confidence",
    "ai_score", "hard_rule_score", "triggered_rules", "reasons", "suggestion", "model_loaded",
    "version_ok", "api_version", "feature_version", "degraded",
)
COMPACT_FIELDS = ("ok", "url", "risk_score", "risk_level", "confidence", "source")
LEVEL_LABEL = {"high": "高風險", "medium": "中風險", "low": "低風險"}

BENIGN_UNKNOWN_URL = "https://example-bakery-taipei.com.tw/menu"
SCAMMY_URL = "https://stock-vip-tw.top/h5/#/register?invitecode=AB12CD"
TRUSTED_URL = "https://www.google.com/"

_client: Optional[TestClient] = None
_tmpdir: str = ""
_protected = S.ProtectedFiles()
_patched: List[Tuple[Any, str, Any]] = []


# =============================================================================
# 路徑重導向工具
# =============================================================================

def _is_va_path(value: Any, names) -> bool:
    if not isinstance(value, str) or os.path.basename(value) not in names:
        return False
    try:
        return os.path.normcase(os.path.abspath(os.path.dirname(value))) == os.path.normcase(S.VA_DIR)
    except Exception:
        return False


_HOLDER_MODULES = {"main", "blocklist_store", "cache", "model_store"}


def _candidate_holders(module: types.ModuleType):
    """main 模組本身，以及 main 內「可能持有檔案路徑」的物件實例（例如 model_state、blocklist_store）。"""
    yield module
    for value in list(vars(module).values()):
        if isinstance(value, (types.ModuleType, type, types.FunctionType, types.BuiltinFunctionType)):
            continue
        if not hasattr(value, "__dict__"):
            continue
        # 一般資料物件（不可呼叫），或來自專案模組的實例（即使可呼叫）
        if not callable(value) or type(value).__module__ in _HOLDER_MODULES:
            yield value


def redirect_paths(names, target_dir: str) -> List[Tuple[Any, str, Any]]:
    """把 main 模組與其內物件中所有指向 va/<names> 的字串屬性改指到 target_dir，回傳還原清單。"""
    undo = []
    for holder in _candidate_holders(M):
        attrs = vars(holder)
        for attr, value in list(attrs.items()):
            if _is_va_path(value, names):
                setattr(holder, attr, os.path.join(target_dir, os.path.basename(value)))
                undo.append((holder, attr, value))
    return undo


def undo_patches(undo: List[Tuple[Any, str, Any]]) -> None:
    for holder, attr, value in reversed(undo):
        setattr(holder, attr, value)


_TEST_ADMIN_KEY = "unittest-admin-key"
_reporter_counter = 0
_ENV_BACKUP: Dict[str, Optional[str]] = {}


def post(path: str, payload: Any = None, **kw):
    """測試用 POST：管理端點自動帶 X-Admin-Key；/blocklist/add 預設每次換一個「回報者 IP」。"""
    global _reporter_counter
    assert _client is not None
    headers = dict(kw.pop("headers", None) or {})
    if path == "/reload-model":
        headers.setdefault("X-Admin-Key", _TEST_ADMIN_KEY)
    if path == "/blocklist/add" and "X-Forwarded-For" not in headers:
        _reporter_counter += 1
        headers["X-Forwarded-For"] = f"10.9.{_reporter_counter // 250}.{_reporter_counter % 250 + 1}"
    kw["headers"] = headers
    if payload is None and "content" not in kw:
        return _client.post(path, **kw)
    if "content" in kw:
        return _client.post(path, **kw)
    return _client.post(path, json=payload, **kw)


def predict(url: Any, **flags) -> Dict[str, Any]:
    r = post("/predict", {"url": url, **flags})
    assert r.status_code < 500, f"/predict 對 {url!r} 回 {r.status_code}：{r.text[:300]}"
    return r.json()


def setUpModule():  # noqa: N802
    global _client, _tmpdir, _patched
    _protected.snapshot()
    _tmpdir = tempfile.mkdtemp(prefix="truthmark-tests-")
    _patched = redirect_paths(REDIRECT_FILES, _tmpdir)
    # 安全機制（security.py）：本檔測的是功能面，限流關閉、管理金鑰固定、信任 X-Forwarded-For
    # 以便模擬多位回報者；限流、指紋、管理員驗證本身由 test_security.py 專門測試。
    for key, value in (("TRUTHMARK_ADMIN_KEY", _TEST_ADMIN_KEY), ("TRUTHMARK_TRUST_PROXY", "1")):
        _ENV_BACKUP[key] = os.environ.get(key)
        os.environ[key] = value
    M.rate_limiter.enabled = False
    _client = TestClient(M.app, raise_server_exceptions=False)
    _client.__enter__()


def tearDownModule():  # noqa: N802
    global _client
    try:
        if _client is not None:
            _client.__exit__(None, None, None)
    finally:
        _client = None
        M.rate_limiter.enabled = True
        for key, old in _ENV_BACKUP.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old
        _ENV_BACKUP.clear()
        undo_patches(_patched)
        touched = _protected.restore()
        if touched:
            sys.stderr.write(f"\n[test_api] 警告：測試期間真實檔案被改動並已還原：{touched}\n")
        shutil.rmtree(_tmpdir, ignore_errors=True)


def assert_error_shape(tc: unittest.TestCase, body: Any, label: str = "") -> None:
    tc.assertIsInstance(body, dict, label)
    tc.assertIs(body.get("ok"), False, f"{label} 應回 ok:false：{str(body)[:200]}")
    tc.assertIsInstance(body.get("error"), str, f"{label} 缺 error 欄位：{str(body)[:200]}")
    tc.assertIsInstance(body.get("message"), str, f"{label} 缺 message 欄位：{str(body)[:200]}")
    tc.assertTrue(body["error"], label)
    tc.assertEqual(body["error"], body["message"], f"{label} error 與 message 應同內容")


def assert_level_consistent(tc: unittest.TestCase, body: Dict[str, Any], label: str = "") -> None:
    score = body["risk_score"]
    expected = "high" if score >= 70 else "medium" if score >= 40 else "low"
    tc.assertEqual(body["risk_level"], expected, f"{label} score={score}")
    if "risk_label" in body:
        tc.assertEqual(body["risk_label"], LEVEL_LABEL[expected], label)


# =============================================================================
# 假模型（暫存目錄），用於降級／bundle 測試
# =============================================================================

def _fit_dummy(n_features: int, constant: int = 1):
    import numpy as np
    from sklearn.dummy import DummyClassifier
    X = np.zeros((4, n_features))
    y = np.array([0, 1, 0, 1])
    clf = DummyClassifier(strategy="constant", constant=constant)
    clf.fit(X, y)
    return clf


def write_fake_model(directory: str, kind: str) -> None:
    """kind: bundle_ok / bundle_mismatch / legacy_31 / legacy_45 / missing / corrupt"""
    import joblib
    import sklearn
    os.makedirs(directory, exist_ok=True)
    model_path = os.path.join(directory, "scam_model.pkl")
    meta_path = os.path.join(directory, "model_meta.json")
    for p in (model_path, meta_path):
        if os.path.exists(p):
            os.remove(p)
    names45 = list(F.FEATURE_NAMES)
    names31 = list(F.FEATURE_NAMES[:31])
    meta: Dict[str, Any] = {"sklearn_version": sklearn.__version__, "project": "Truth"}
    if kind in ("bundle_ok", "bundle_mismatch"):
        names = names45 if kind == "bundle_ok" else names31
        bundle = {
            "format": "truthmark-model-bundle/1",
            "estimator": _fit_dummy(len(names)),
            "feature_names": names,
            "feature_version": F.FEATURE_VERSION if kind == "bundle_ok" else "6.0.0",
            "feature_schema_id": F.FEATURE_SCHEMA_ID if kind == "bundle_ok" else "legacy",
            "sklearn_version": sklearn.__version__,
            "thresholds": {"medium": 0.40, "high": 0.70},
            "trained_at": "2026-01-01T00:00:00",
        }
        joblib.dump(bundle, model_path)
        meta.update(feature_names=names, feature_count=len(names), feature_version=bundle["feature_version"],
                    feature_schema_id=bundle["feature_schema_id"], thresholds=bundle["thresholds"])
    elif kind in ("legacy_31", "legacy_45"):
        # legacy_45：舊格式（純 estimator）但維度＝目前 FEATURE_NAMES（v7.0 為 45、v7.1 為 46），應可使用
        n = 31 if kind == "legacy_31" else len(names45)
        joblib.dump(_fit_dummy(n), model_path)
        names = names31 if kind == "legacy_31" else names45
        meta.update(feature_names=names, feature_count=n)
        if kind == "legacy_45":
            meta.update(feature_version=F.FEATURE_VERSION, feature_schema_id=F.FEATURE_SCHEMA_ID)
    elif kind == "corrupt":
        with open(model_path, "wb") as fh:
            fh.write(b"not a pickle at all")
        meta.update(feature_names=names45, feature_count=len(names45), feature_version=F.FEATURE_VERSION,
                    feature_schema_id=F.FEATURE_SCHEMA_ID)
    elif kind == "missing":
        pass
    else:  # pragma: no cover
        raise ValueError(kind)
    if kind != "missing":
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False)


class ModelSwap:
    """暫時把 main 的模型路徑改指到假模型並 /reload-model；離開時改回並重新載入真實模型。"""

    def __init__(self, tc: unittest.TestCase, kind: str) -> None:
        self.tc = tc
        self.kind = kind
        self.dir = os.path.join(_tmpdir, f"model-{kind}-{uuid.uuid4().hex[:6]}")
        self.undo: List[Tuple[Any, str, Any]] = []

    def __enter__(self):
        write_fake_model(self.dir, self.kind)
        self.undo = redirect_paths(MODEL_FILES, self.dir)
        if not any(os.path.basename(v) == "scam_model.pkl" for _, _, v in self.undo):
            undo_patches(self.undo)
            self.tc.skipTest("找不到 main 內指向 scam_model.pkl 的路徑屬性，無法模擬模型狀態")
        _clear_cache()
        r = post("/reload-model")
        self.tc.assertLess(r.status_code, 500, r.text[:300])
        self.reload_body = r.json()
        return self

    def __exit__(self, *exc):
        undo_patches(self.undo)
        post("/reload-model")
        _clear_cache()
        return False


def _clear_cache() -> None:
    cache = getattr(M, "prediction_cache", None)
    for meth in ("clear", "flushall", "flush"):
        fn = getattr(cache, meth, None)
        if callable(fn):
            try:
                fn()
                return
            except Exception:
                pass


# =============================================================================
# 基本端點
# =============================================================================

class HealthAndFeaturesTests(unittest.TestCase):
    def test_health(self):
        r = _client.get("/health")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body.get("status"), "ok")
        self.assertEqual(body.get("api_version"), "7.2.0")
        self.assertEqual(body.get("feature_version"), F.FEATURE_VERSION)
        self.assertEqual(body.get("feature_schema_id"), F.FEATURE_SCHEMA_ID)
        self.assertEqual(body.get("feature_names"), F.FEATURE_NAMES)
        self.assertEqual(body.get("feature_count"), len(F.FEATURE_NAMES))
        self.assertIsInstance(body.get("model_feature_names_ok"), bool)
        self.assertIsInstance(body.get("degraded"), bool)
        self.assertIsInstance(body.get("model_loaded"), bool)
        if body["model_loaded"] and body["model_feature_names_ok"]:
            self.assertFalse(body["degraded"])

    def test_features_endpoint(self):
        r = _client.get("/features")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body.get("count"), len(F.FEATURE_NAMES))
        self.assertEqual(body.get("features"), F.FEATURE_NAMES)
        self.assertEqual(body.get("version"), F.FEATURE_VERSION)
        self.assertEqual(body.get("schema_id"), F.FEATURE_SCHEMA_ID)
        specs = body.get("specs") or {}
        self.assertEqual(set(specs), set(F.FEATURE_NAMES))

    def test_static_routes(self):
        for path in ("/", "/script.js", "/style.css"):
            r = _client.get(path)
            self.assertEqual(r.status_code, 200, path)
        # 瀏覽器自動請求的 favicon 不可 404（否則網頁 console 會出現錯誤）
        self.assertIn(_client.get("/favicon.ico").status_code, (200, 204))

    def test_cache_stats(self):
        r = _client.get("/cache/stats")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json().get("ok"))


class PredictShapeTests(unittest.TestCase):
    def _check_full(self, url: str) -> Dict[str, Any]:
        body = predict(url)
        self.assertTrue(body.get("ok"), body)
        missing = [k for k in FULL_FIELDS if k not in body]
        self.assertEqual(missing, [], f"{url} 缺欄位")
        self.assertEqual(body["api_version"], "7.2.0")
        self.assertEqual(body["feature_version"], F.FEATURE_VERSION)
        self.assertIn(body["source"], S.VALID_SOURCES)
        self.assertIsInstance(body["risk_score"], int)
        self.assertTrue(0 <= body["risk_score"] <= 100)
        self.assertTrue(0.0 <= float(body["confidence"]) <= 1.0)
        self.assertIsInstance(body["degraded"], bool)
        self.assertIsInstance(body["triggered_rules"], list)
        self.assertIsInstance(body["reasons"], list)
        for reason in body["reasons"]:
            self.assertTrue({"key", "level", "message"} <= set(reason), reason)
        assert_level_consistent(self, body, url)
        if body["degraded"]:
            self.assertIn(body["source"], {"rules_only", "trusted_domain", "blocklist"})
        return body

    def test_full_fields_all_sources(self):
        self.assertEqual(self._check_full(TRUSTED_URL)["source"], "trusted_domain")
        import rules_config as R
        blocked = sorted(R.INITIAL_BLOCKED_DOMAINS)[0]
        self.assertEqual(self._check_full(f"https://{blocked}/")["source"], "blocklist")
        self._check_full(SCAMMY_URL)
        self._check_full(BENIGN_UNKNOWN_URL)

    def test_debug_fields_aligned_with_features_py(self):
        for url in (SCAMMY_URL, BENIGN_UNKNOWN_URL, TRUSTED_URL):
            body = predict(url, debug=True)
            self.assertTrue(body.get("ok"), body)
            feats = body.get("features")
            self.assertIsInstance(feats, dict, f"{url} debug 應回 features")
            self.assertEqual(set(feats), set(F.FEATURE_NAMES), url)
            expected = F.extract_feature_dict(url)
            diff = {k: (feats[k], expected[k]) for k in F.FEATURE_NAMES if abs(float(feats[k]) - float(expected[k])) > 1e-9}
            self.assertEqual(diff, {}, f"{url} API 特徵值與 features.extract_feature_dict 不一致")
            self.assertEqual(body.get("feature_names"), F.FEATURE_NAMES, url)
            fa = body.get("feature_assessment")
            self.assertIsInstance(fa, dict, f"{url} debug 應回 feature_assessment")
            self.assertEqual(set(fa), set(F.FEATURE_NAMES), url)
            self.assertEqual(fa, F.assess_all(expected), f"{url} feature_assessment 應等於 features.assess_all")

    def test_non_debug_has_no_feature_payload(self):
        body = predict(BENIGN_UNKNOWN_URL)
        self.assertFalse(body.get("features"), "非 debug 不應夾帶 features")

    def test_compact_fields(self):
        for url in (SCAMMY_URL, TRUSTED_URL, BENIGN_UNKNOWN_URL):
            full = predict(url)
            body = predict(url, compact=True)
            self.assertTrue(body.get("ok"), body)
            self.assertEqual([k for k in COMPACT_FIELDS if k not in body], [], url)
            for k in ("features", "feature_assessment", "feature_names"):
                self.assertNotIn(k, body, f"compact 不應含 {k}")
            self.assertEqual(body["risk_score"], full["risk_score"], url)
            self.assertEqual(body["source"], full["source"], url)
            assert_level_consistent(self, body, url)

    def test_scheme_variants_same_score(self):
        bare = "stock-vip-tw.top/h5/#/register"
        a = predict(bare)
        b = predict("https://" + bare)
        self.assertEqual(a["risk_score"], b["risk_score"], "無 scheme 與 https:// 應得到相同分數")


class PredictErrorTests(unittest.TestCase):
    def test_empty_and_blank(self):
        for url in ("", "   ", "　", "​"):
            assert_error_shape(self, predict(url), repr(url))

    def test_too_long(self):
        url = "https://example.com/" + "a" * (2049 - len("https://example.com/"))
        self.assertEqual(len(url), 2049)
        assert_error_shape(self, predict(url), "2049 字元")
        ok_url = url[:2048]
        body = predict(ok_url)
        self.assertIn("ok", body)

    def test_weird_inputs_no_500(self):
        for url in ("http://[abc", "https://[::1", "http://[::1]:99999/", "https://example.com:abc/",
                    "javascript:alert(1)", "ｈｔｔｐｓ：／／ｅｘａｍｐｌｅ．ｃｏｍ", "https://例子.測試/",
                    "http://%zz/", "@@@", "\x00http://x.com", "https://" + "a." * 300 + "com"):
            with self.subTest(url=url[:40]):
                body = predict(url)
                self.assertIn("ok", body)
                if body["ok"] is False:
                    assert_error_shape(self, body, url[:40])
                else:
                    assert_level_consistent(self, body, url[:40])

    def test_invalid_payloads_json_error(self):
        cases = [
            ("missing url", {"json": {}}),
            ("url int", {"json": {"url": 123}}),
            ("url null", {"json": {"url": None}}),
            ("not json", {"content": b"not json", "headers": {"Content-Type": "application/json"}}),
            ("array", {"json": ["https://x.com"]}),
        ]
        for label, kw in cases:
            with self.subTest(label=label):
                r = _client.post("/predict", **kw)
                self.assertLess(r.status_code, 500, f"{label}: {r.text[:200]}")
                assert_error_shape(self, r.json(), label)

    def test_error_responses_not_cached(self):
        predict("")
        body = predict("")
        self.assertFalse(body.get("cache_hit"), "錯誤回應不應被快取")

    def test_deep_scan_internal_target_no_500(self):
        t0 = time.time()
        body = predict("http://127.0.0.1:9/", deep_scan=True)
        self.assertIn("ok", body)
        if body.get("ok"):
            sem = body.get("semantic_analysis")
            if sem is not None:
                self.assertFalse(sem.get("fetched"), "deep_scan 不應抓取內網位址")
        self.assertLess(time.time() - t0, 15)


class PredictBatchTests(unittest.TestCase):
    def test_batch_mixed(self):
        urls = [TRUSTED_URL, "http://[abc", "", SCAMMY_URL, "https://example.com/" + "b" * 3000,
                BENIGN_UNKNOWN_URL]
        r = post("/predict/batch", {"urls": urls})
        self.assertLess(r.status_code, 500, r.text[:300])
        self.assertNotEqual(r.status_code, 404, "/predict/batch 尚未實作")
        body = r.json()
        self.assertTrue(body.get("ok"), body)
        results = body.get("results")
        self.assertIsInstance(results, dict)
        self.assertEqual(set(results), set(urls), "results 的 key 必須是原樣輸入的 url 字串")
        self.assertEqual(body.get("count"), len(set(urls)))
        self.assertTrue(results[TRUSTED_URL].get("ok"))
        self.assertTrue(results[SCAMMY_URL].get("ok"))
        self.assertTrue(results[BENIGN_UNKNOWN_URL].get("ok"))
        assert_error_shape(self, results[""], "batch 空字串")
        assert_error_shape(self, results["https://example.com/" + "b" * 3000], "batch 過長")
        self.assertIn("ok", results["http://[abc"])
        # 預設 compact=true
        self.assertNotIn("features", results[SCAMMY_URL])
        # 與單筆 /predict 一致
        for u in (TRUSTED_URL, SCAMMY_URL, BENIGN_UNKNOWN_URL):
            self.assertEqual(results[u]["risk_score"], predict(u)["risk_score"], u)

    def test_batch_limit(self):
        base = [f"https://site{i}-example.com/" for i in range(51)]
        r50 = post("/predict/batch", {"urls": base[:50]})
        self.assertLess(r50.status_code, 500)
        self.assertNotEqual(r50.status_code, 404, "/predict/batch 尚未實作")
        b50 = r50.json()
        self.assertTrue(b50.get("ok"), str(b50)[:300])
        self.assertEqual(b50.get("count"), 50)
        r51 = post("/predict/batch", {"urls": base})
        self.assertLess(r51.status_code, 500)
        assert_error_shape(self, r51.json(), "batch 51 筆")

    def test_batch_debug(self):
        r = post("/predict/batch", {"urls": [SCAMMY_URL], "debug": True, "compact": False})
        self.assertLess(r.status_code, 500)
        self.assertNotEqual(r.status_code, 404, "/predict/batch 尚未實作")
        res = r.json()["results"][SCAMMY_URL]
        self.assertEqual(set(res.get("features", {})), set(F.FEATURE_NAMES))

    def test_batch_bad_payload(self):
        for payload in ({}, {"urls": "https://x.com"}, {"urls": None}):
            r = post("/predict/batch", payload)
            self.assertLess(r.status_code, 500)
            self.assertNotEqual(r.status_code, 404, "/predict/batch 尚未實作")
            assert_error_shape(self, r.json(), str(payload))


# =============================================================================
# 寫入型端點（暫存目錄）＋快取失效
# =============================================================================

class WriteEndpointTests(unittest.TestCase):
    def _assert_real_files_untouched(self):
        time.sleep(0.3)  # /blocklist/add 的背景寫入以 create_task 排程，稍等完成
        changed = _protected.changed()
        if changed:
            _protected.restore()
            self.fail(f"測試寫到了真實檔案 {changed}（main 未使用可替換的路徑常數，已還原）")

    def test_feedback_invalidates_url_cache(self):
        url = f"https://tm-feedback-{uuid.uuid4().hex[:8]}.xyz/promo"
        predict(url)
        second = predict(url)
        if "cache_hit" not in second:
            self.skipTest("回應沒有 cache_hit 欄位，無法觀察快取")
        self.assertTrue(second["cache_hit"], "同一網址第二次應命中快取")
        r = post("/feedback", {"url": url, "label": 1, "ai_score": 50.0, "note": "unittest"})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json().get("ok"))
        self.assertFalse(predict(url).get("cache_hit"), "/feedback 後該 URL 快取應失效")
        fb = os.path.join(_tmpdir, "feedback.csv")
        self.assertTrue(os.path.exists(fb), "feedback 應寫入暫存目錄的 feedback.csv")
        with open(fb, encoding="utf-8-sig") as fh:
            self.assertIn(url, fh.read())
        self._assert_real_files_untouched()

    def test_feedback_invalid(self):
        for payload in ({"url": "", "label": 1}, {"url": "https://x.com", "label": 5}):
            r = post("/feedback", payload)
            self.assertLess(r.status_code, 500)
            assert_error_shape(self, r.json(), str(payload))

    def test_blocklist_add_invalidates_domain_cache(self):
        domain = f"tm-block-{uuid.uuid4().hex[:8]}.xyz"
        urls = [f"https://{domain}/promo", f"https://www.{domain}/a/b?x=1", f"https://m.{domain}/"]
        before = {u: predict(u) for u in urls}
        for u in urls:
            self.assertNotEqual(before[u].get("source"), "blocklist")
            predict(u)  # 確保已進快取
        blocked = False
        body: Dict[str, Any] = {}
        try:
            for _ in range(10):
                r = post("/blocklist/add", {"domain": domain, "note": "unittest"})
                self.assertEqual(r.status_code, 200, r.text[:300])
                body = r.json()
                self.assertTrue(body.get("ok"), body)
                if body.get("blocked"):
                    blocked = True
                    break
            self.assertTrue(blocked, f"回報 10 次仍未列入黑名單：{body}")
            for u in urls:
                after = predict(u)
                self.assertFalse(after.get("cache_hit"), f"{u} 快取未失效")
                self.assertEqual(after.get("source"), "blocklist", f"{u} 應改判 blocklist（快取未以網域失效）")
                self.assertEqual(after.get("risk_score"), 100)
            self.assertTrue(os.path.exists(os.path.join(_tmpdir, "domain_reports.json")),
                            "domain_reports.json 應寫入暫存目錄")
            self._assert_real_files_untouched()
        finally:
            for holder in (getattr(M, "BLOCKED_DOMAINS", None),
                           getattr(getattr(M, "blocklist_store", None), "blocked_domains", None)):
                if isinstance(holder, set):
                    holder.discard(domain)
            _clear_cache()

    def test_blocklist_add_invalid(self):
        for d in ("", "nodot", "   "):
            r = post("/blocklist/add", {"domain": d})
            self.assertLess(r.status_code, 500)
            assert_error_shape(self, r.json(), repr(d))


# =============================================================================
# 模型降級 / bundle 支援
# =============================================================================

class ModelDegradeTests(unittest.TestCase):
    def _predict_unknown(self) -> Dict[str, Any]:
        body = predict(BENIGN_UNKNOWN_URL + f"?t={uuid.uuid4().hex[:6]}")
        self.assertTrue(body.get("ok"), body)
        return body

    def _assert_degraded(self, label: str):
        body = self._predict_unknown()
        self.assertIs(body.get("degraded"), True, f"{label}：應 degraded=true")
        self.assertEqual(body.get("source"), "rules_only", label)
        assert_level_consistent(self, body, label)
        h = _client.get("/health").json()
        self.assertIs(h.get("degraded"), True, f"{label}：/health 應 degraded=true")
        # 降級時仍要能處理詐騙網址
        scam = predict(SCAMMY_URL + f"&t={uuid.uuid4().hex[:6]}")
        self.assertTrue(scam.get("ok"), scam)
        self.assertEqual(scam.get("source"), "rules_only", label)
        self.assertGreaterEqual(scam.get("risk_score", 0), 40, f"{label}：規則降級仍應攔下明顯詐騙")

    def test_model_missing_degraded(self):
        with ModelSwap(self, "missing"):
            self._assert_degraded("模型檔不存在")

    def test_model_corrupt_degraded(self):
        with ModelSwap(self, "corrupt"):
            self._assert_degraded("模型檔損毀")

    def test_bundle_feature_mismatch_degraded(self):
        with ModelSwap(self, "bundle_mismatch"):
            self._assert_degraded("bundle feature_names 不一致")
            self.assertIs(_client.get("/health").json().get("model_feature_names_ok"), False)

    def test_legacy_31_estimator_degraded(self):
        with ModelSwap(self, "legacy_31"):
            self._assert_degraded("舊格式 31 維 estimator")

    def test_bundle_ok_uses_model(self):
        with ModelSwap(self, "bundle_ok") as swap:
            self.assertTrue(swap.reload_body.get("ok"), swap.reload_body)
            body = self._predict_unknown()
            self.assertIs(body.get("degraded"), False, body)
            self.assertIn(body.get("source"), {"ai_model", "hybrid_ai_hard_rule"})
            self.assertTrue(body.get("model_loaded"))
            self.assertGreater(float(body.get("ai_score", 0)), 50, "假模型恆回 1，ai_score 應接近 100")
            h = _client.get("/health").json()
            self.assertIs(h.get("model_feature_names_ok"), True)
            self.assertIs(h.get("degraded"), False)

    def test_legacy_45_estimator_supported(self):
        with ModelSwap(self, "legacy_45"):
            body = self._predict_unknown()
            self.assertIs(body.get("degraded"), False, body)
            self.assertIn(body.get("source"), {"ai_model", "hybrid_ai_hard_rule"})

    def test_model_exception_falls_back(self):
        with ModelSwap(self, "bundle_ok"):
            state = getattr(M, "model_state", None)
            if state is None or not hasattr(state, "predict_score"):
                self.skipTest("main.model_state.predict_score 不存在，無法注入推論例外")

            def boom(*_a, **_k):
                raise RuntimeError("unittest：模擬推論失敗")

            state.predict_score = boom
            try:
                body = self._predict_unknown()
                self.assertTrue(body.get("ok"), body)
                self.assertIs(body.get("degraded"), True, "推論失敗應降級為規則評分")
                self.assertEqual(body.get("source"), "rules_only")
            finally:
                del state.predict_score

    def test_feature_extraction_exception_no_500(self):
        original = getattr(M, "extract_feature_dict", None)
        if original is None:
            self.skipTest("main 未直接引用 extract_feature_dict")

        def boom(*_a, **_k):
            raise RuntimeError("unittest：模擬特徵擷取失敗")

        M.extract_feature_dict = boom
        try:
            r = post("/predict", {"url": f"https://tm-boom-{uuid.uuid4().hex[:6]}.com/"})
            # 內部非預期例外：理想是攔下並降級（<500）；至少全域例外處理器要回 JSON 錯誤格式（CONTRACT §3）
            try:
                body = r.json()
            except Exception:
                self.fail(f"內部例外時回應不是 JSON（status={r.status_code}）：{r.text[:200]}")
            self.assertIn("ok", body, r.text[:300])
            if r.status_code >= 500 or body["ok"] is False:
                assert_error_shape(self, body, "特徵擷取例外")
        finally:
            M.extract_feature_dict = original
            _clear_cache()


# =============================================================================
# 端到端回歸（url_cases.json）
# =============================================================================

class RegressionTests(unittest.TestCase):
    results: List[Dict[str, Any]] = []

    @classmethod
    def setUpClass(cls):
        _clear_cache()
        out = []
        for case in S.load_cases():
            r = post("/predict", {"url": case["url"]})
            entry = dict(case)
            entry["status"] = r.status_code
            try:
                body = r.json()
            except Exception:
                body = {}
            entry["ok"] = bool(body.get("ok"))
            entry["score"] = body.get("risk_score")
            entry["source"] = body.get("source")
            entry["rules"] = body.get("triggered_rules")
            entry["ai_score"] = body.get("ai_score")
            entry["degraded"] = body.get("degraded")
            out.append(entry)
        cls.results = out
        cls._summary()

    @classmethod
    def _summary(cls):
        res = cls.results
        benign = [e for e in res if e["expect"] == "benign"]
        scam = [e for e in res if e["expect"] == "scam"]
        fp = [e for e in benign if not e["ok"] or (e["score"] or 0) >= 40]
        fn = [e for e in scam if not e["ok"] or (e["score"] or 0) < 40]
        ms = [e for e in res if "min_score" in e]
        ms_fail = [e for e in ms if not e["ok"] or (e["score"] or 0) < e["min_score"]]
        cls.fp, cls.fn, cls.ms, cls.ms_fail, cls.benign, cls.scam = fp, fn, ms, ms_fail, benign, scam
        lines = [
            "",
            "[回歸摘要] benign 誤判 %d/%d (%.1f%%)；scam 召回 %d/%d (%.1f%%)；min_score 達成 %d/%d (%.1f%%)" % (
                len(fp), len(benign), 100 * len(fp) / max(1, len(benign)),
                len(scam) - len(fn), len(scam), 100 * (len(scam) - len(fn)) / max(1, len(scam)),
                len(ms) - len(ms_fail), len(ms), 100 * (len(ms) - len(ms_fail)) / max(1, len(ms))),
            "  source 分布：%s" % dict(Counter(e["source"] for e in res)),
        ]
        by_cat = defaultdict(lambda: [0, 0])
        for e in res:
            bad = e in fp or e in fn
            by_cat[e["category"]][0] += 1
            by_cat[e["category"]][1] += int(bad)
        lines.append("  各類別失敗數：" + "、".join(f"{c} {b}/{n}" for c, (n, b) in sorted(by_cat.items())))
        sys.stderr.write("\n".join(lines) + "\n")
        report = os.environ.get("TRUTHMARK_REGRESSION_REPORT")
        if report:
            with open(report, "w", encoding="utf-8") as fh:
                json.dump(res, fh, ensure_ascii=False, indent=1)

    @staticmethod
    def _fmt(items, key="score"):
        return "\n".join(
            f"  [{e['category']}] {e['url']} → score={e.get('score')} src={e.get('source')} "
            f"min={e.get('min_score', '-')} rules={e.get('rules')}"
            for e in items)

    def test_no_server_errors(self):
        bad = [e for e in self.results if e["status"] >= 500 or not e["ok"]]
        self.assertEqual(bad, [], "回歸清單中有網址回 500 或 ok:false：\n" + self._fmt(bad))

    def test_benign_false_positive_rate(self):
        rate = len(self.fp) / max(1, len(self.benign))
        self.assertLessEqual(rate, BENIGN_FP_MAX,
                             f"benign 誤判率 {rate:.1%} > {BENIGN_FP_MAX:.0%}（{len(self.fp)}/{len(self.benign)}）：\n"
                             + self._fmt(self.fp))

    def test_scam_recall(self):
        recall = 1 - len(self.fn) / max(1, len(self.scam))
        self.assertGreaterEqual(recall, SCAM_RECALL_MIN,
                                f"scam 召回率 {recall:.1%} < {SCAM_RECALL_MIN:.0%}（漏報 {len(self.fn)}/{len(self.scam)}）：\n"
                                + self._fmt(self.fn))

    def test_min_score_attainment(self):
        rate = 1 - len(self.ms_fail) / max(1, len(self.ms))
        self.assertGreaterEqual(rate, MIN_SCORE_RATE_MIN,
                                f"min_score 達成率 {rate:.1%} < {MIN_SCORE_RATE_MIN:.0%}"
                                f"（未達 {len(self.ms_fail)}/{len(self.ms)}）：\n" + self._fmt(self.ms_fail))

    def test_levels_consistent(self):
        for e in self.results:
            if e["ok"] and e["score"] is not None:
                self.assertTrue(0 <= e["score"] <= 100, e["url"])


if __name__ == "__main__":
    unittest.main()
