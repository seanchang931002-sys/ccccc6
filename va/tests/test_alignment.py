# -*- coding: utf-8 -*-
"""
跨模組特徵對齊測試（消除 Feature Drift）。

檢查 features.py ⇄ model_meta.json / scam_model.pkl ⇄ rules_config.py ⇄ text_features.py
⇄ script.js / index.html / 外掛 icons.js 的特徵名稱、維度、圖示與規則格式是否 100% 一致。

執行（在 va 目錄下）：
    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
import sys
import unittest
import warnings

import _support as S  # noqa: E402  (同時把 va/ 加進 sys.path)

import features as F  # noqa: E402
import rules_config as R  # noqa: E402

SCRIPT_JS = os.path.join(S.VA_DIR, "script.js")
INDEX_HTML = os.path.join(S.VA_DIR, "index.html")
ICONS_JS = os.path.join(S.EXT_DIR, "icons.js")
META_PATH = os.path.join(S.VA_DIR, "model_meta.json")
MODEL_PATH = os.path.join(S.VA_DIR, "scam_model.pkl")
VALID_TARGETS = {"url", "decoded", "host", "path", "any"}


# -----------------------------------------------------------------------------
# 極簡 JS 掃描器：跳過字串／註解，找出物件字面值的頂層 key 與其值的原始碼片段
# -----------------------------------------------------------------------------

def _skip_string(src: str, i: int) -> int:
    quote = src[i]
    i += 1
    while i < len(src):
        ch = src[i]
        if ch == "\\":
            i += 2
            continue
        if ch == quote:
            return i + 1
        i += 1
    return i


def _object_span(src: str, open_idx: int) -> int:
    """src[open_idx] == '{'，回傳對應 '}' 的索引。"""
    depth = 0
    i = open_idx
    while i < len(src):
        ch = src[i]
        if ch in "'\"`":
            i = _skip_string(src, i)
            continue
        if src.startswith("//", i):
            j = src.find("\n", i)
            i = len(src) if j < 0 else j
            continue
        if src.startswith("/*", i):
            j = src.find("*/", i + 2)
            i = len(src) if j < 0 else j + 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise ValueError("物件字面值沒有閉合")


_KEY_RE = re.compile(r"""\s*(?:([A-Za-z_$][\w$]*)|'([^']+)'|"([^"]+)")\s*:""")


def parse_js_object(src: str, var_name: str) -> dict:
    """回傳 {頂層 key: 該 key 值的原始碼片段}（依出現順序）。"""
    m = re.search(r"(?:const|let|var)\s+" + re.escape(var_name) + r"\s*=\s*\{", src)
    if not m:
        raise AssertionError(f"找不到 {var_name} 物件定義")
    start = m.end() - 1
    end = _object_span(src, start)
    body = src[start + 1:end]

    entries = {}
    depth = 0
    i = 0
    cur_key = None
    cur_start = 0
    expect_key = True
    while i < len(body):
        ch = body[i]
        if depth == 0 and expect_key:
            km = _KEY_RE.match(body, i)
            if km and not body[i:km.end()].strip().startswith(("//", "/*")):
                cur_key = km.group(1) or km.group(2) or km.group(3)
                cur_start = km.end()
                i = km.end()
                expect_key = False
                continue
        if ch in "'\"`":
            i = _skip_string(body, i)
            continue
        if body.startswith("//", i):
            j = body.find("\n", i)
            i = len(body) if j < 0 else j
            continue
        if body.startswith("/*", i):
            j = body.find("*/", i + 2)
            i = len(body) if j < 0 else j + 2
            continue
        if ch in "{[(":
            depth += 1
        elif ch in "}])":
            depth -= 1
        elif ch == "," and depth == 0:
            if cur_key is not None:
                entries[cur_key] = body[cur_start:i]
            cur_key = None
            expect_key = True
        i += 1
    if cur_key is not None:
        entries[cur_key] = body[cur_start:]
    return entries


def icons_js_names(src: str) -> set:
    m = re.search(r"var\s+ICONS\s*=\s*\{", src)
    if not m:
        raise AssertionError("icons.js 找不到 ICONS 物件")
    end = _object_span(src, m.end() - 1)
    return set(re.findall(r"""^\s*'([a-z0-9-]+)'\s*:""", src[m.end():end], re.M))


def _norm_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip("\n")


# =============================================================================
# features.py 本身
# =============================================================================

class FeatureNameTests(unittest.TestCase):
    def test_count_and_unique(self):
        expected = len(S.LEGACY_31) + len(S.NEW_14) + len(S.NEW_V71)   # v7.1：46
        self.assertEqual(len(F.FEATURE_NAMES), expected)
        self.assertEqual(len(set(F.FEATURE_NAMES)), expected, "FEATURE_NAMES 有重複名稱")

    def test_first_31_unchanged_vs_legacy_list(self):
        self.assertEqual(F.FEATURE_NAMES[:31], S.LEGACY_31)

    def test_first_31_unchanged_vs_backup(self):
        legacy = S.backup_feature_names()
        if legacy is None:
            self.skipTest(f"找不到備份 features.py（{S.BACKUP_DIR}），略過與備份比對")
        self.assertEqual(len(legacy), 31, "備份的 FEATURE_NAMES 應為 31 個")
        self.assertEqual(F.FEATURE_NAMES[:31], legacy)

    def test_appended_14_order(self):
        self.assertEqual(F.FEATURE_NAMES[31:45], S.NEW_14)

    def test_appended_v71_order(self):
        self.assertEqual(F.FEATURE_NAMES[45:], S.NEW_V71)

    def test_schema_id(self):
        expected = hashlib.sha1("|".join(F.FEATURE_NAMES).encode("utf-8")).hexdigest()[:12]
        self.assertEqual(F.FEATURE_SCHEMA_ID, expected)
        self.assertEqual(F.FEATURE_VERSION, "7.1.0")

    def test_specs_cover_all_names(self):
        self.assertEqual(set(F.FEATURE_SPECS), set(F.FEATURE_NAMES),
                         "FEATURE_SPECS 與 FEATURE_NAMES 不一致")
        for name in F.FEATURE_NAMES:
            spec = F.FEATURE_SPECS[name]
            self.assertTrue(spec.get("zh"), f"{name} 缺中文名稱")
            self.assertIn("unit", spec, name)
            self.assertIn(spec.get("kind"), {"count", "ratio", "bool", "level", "score"}, name)

    def test_assess_feature_covers_all_names(self):
        fd = F.extract_feature_dict("https://stock-vip-tw.top/h5/#/register?invitecode=AB12")
        for name in F.FEATURE_NAMES:
            for value in (fd[name], 0, 1):
                res = F.assess_feature(name, value)
                self.assertIn(res.get("status"), S.VALID_STATUSES, f"{name}={value}")
                self.assertIsInstance(res.get("text"), str)
                self.assertTrue(res["text"], f"{name}={value} 沒有標籤文字")
        allres = F.assess_all(fd)
        self.assertEqual(set(allres), set(F.FEATURE_NAMES))

    def test_extract_features_order(self):
        url = "https://binance-tw-pro.web.app/#/pages/login?inviteCode=X1"
        fd = F.extract_feature_dict(url)
        self.assertEqual(list(fd.keys()), F.FEATURE_NAMES, "extract_feature_dict 的 key 順序應與 FEATURE_NAMES 相同")
        self.assertEqual(F.extract_features(url), [fd[n] for n in F.FEATURE_NAMES])

    def test_feature_dict_is_fresh_copy(self):
        url = "https://example.com/"
        a = F.extract_feature_dict(url)
        a["url_length"] = -12345
        self.assertNotEqual(F.extract_feature_dict(url)["url_length"], -12345,
                            "extract_feature_dict 回傳了共享的快取 dict")


# =============================================================================
# 模型檔與 meta
# =============================================================================

class ModelArtifactTests(unittest.TestCase):
    def test_model_meta_feature_names(self):
        if not os.path.exists(META_PATH):
            self.skipTest("model_meta.json 不存在（模型尚未訓練）")
        with open(META_PATH, encoding="utf-8") as fh:
            meta = json.load(fh)
        names = meta.get("feature_names")
        if names is None or meta.get("feature_version") != F.FEATURE_VERSION:
            self.skipTest(
                "model_meta.json 仍為舊版（無 feature_names 或 feature_version="
                f"{meta.get('feature_version')!r}），模型尚未以 v7 特徵重訓"
            )
        self.assertEqual(names, F.FEATURE_NAMES)
        if "feature_count" in meta:
            self.assertEqual(meta["feature_count"], len(F.FEATURE_NAMES))
        if "feature_schema_id" in meta:
            self.assertEqual(meta["feature_schema_id"], F.FEATURE_SCHEMA_ID)

    def test_model_bundle_feature_names(self):
        if not os.path.exists(MODEL_PATH):
            self.skipTest("scam_model.pkl 不存在（模型尚未訓練）")
        import joblib
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            obj = joblib.load(MODEL_PATH)  # 只讀，不寫回
        if not isinstance(obj, dict):
            n_in = getattr(obj, "n_features_in_", None)
            self.skipTest(f"scam_model.pkl 為舊格式純 estimator（n_features_in_={n_in}），模型尚未以 v7 bundle 重訓")
        self.assertEqual(obj.get("format"), "truthmark-model-bundle/1")
        self.assertEqual(obj.get("feature_names"), F.FEATURE_NAMES)
        self.assertEqual(obj.get("feature_schema_id"), F.FEATURE_SCHEMA_ID)
        est = obj.get("estimator")
        self.assertIsNotNone(est)
        self.assertTrue(hasattr(est, "predict_proba") or hasattr(est, "predict"))
        n_in = getattr(est, "n_features_in_", None)
        if n_in is not None:
            self.assertEqual(n_in, len(F.FEATURE_NAMES))
        # 用 v7 特徵實際跑一次推論，確保維度對得上
        import numpy as np
        X = np.array([F.extract_features("https://stock-vip-tw.top/")], dtype=float)
        if hasattr(est, "predict_proba"):
            p = est.predict_proba(X)[0][1]
            self.assertTrue(0.0 <= float(p) <= 1.0)


# =============================================================================
# rules_config.py
# =============================================================================

class HardRuleFormatTests(unittest.TestCase):
    def test_rules_compile_and_targets(self):
        self.assertTrue(R.HARD_RULES, "HARD_RULES 不可為空")
        names = []
        for rule in R.HARD_RULES:
            self.assertIsInstance(rule, R.HardRule)
            try:
                re.compile(rule.pattern, re.I)
            except re.error as exc:  # pragma: no cover - 失敗時才會進來
                self.fail(f"{rule.name} 無法編譯：{exc}")
            self.assertIn(rule.target, VALID_TARGETS, rule.name)
            self.assertTrue(0 < rule.score <= 100, f"{rule.name} score={rule.score}")
            self.assertTrue(rule.message, f"{rule.name} 缺 message")
            names.append(rule.name)
        dup = sorted({n for n in names if names.count(n) > 1})
        self.assertEqual(dup, [], "硬規則名稱重複")

    def test_hard_rule_is_frozen(self):
        rule = R.HARD_RULES[0]
        with self.assertRaises(Exception):
            rule.score = 1  # type: ignore[misc]

    def test_contract_constants_exist(self):
        for attr in ("TRUSTED_DOMAINS", "TRUSTED_SUFFIXES", "WHITELIST_OVERRIDE_SIGNALS",
                     "WHITELIST_OVERRIDE_PATTERNS", "INITIAL_BLOCKED_DOMAINS", "BRAND_OFFICIAL_DOMAINS",
                     "GAMBLING_TERMS", "INVESTMENT_TERMS", "CRYPTO_TERMS", "TLD_RISK_TIERS",
                     "FREE_HOSTING_SUFFIXES", "TUNNEL_SUFFIXES", "SOCIAL_INVITE_PATTERNS"):
            self.assertTrue(hasattr(R, attr), f"rules_config 缺 {attr}")
        for p in R.WHITELIST_OVERRIDE_PATTERNS:
            re.compile(p, re.I)
        for p in R.SOCIAL_INVITE_PATTERNS:
            re.compile(p, re.I)
        self.assertEqual(set(R.TLD_RISK_TIERS), {1, 2})
        for d in ("poyabuy.com.tw", "cosmed.com.tw", "watsons.com.tw"):
            self.assertIn(d, R.TRUSTED_DOMAINS)

    def test_initial_blocklist_preserved(self):
        legacy_path = os.path.join(S.BACKUP_DIR, "rules_config.py")
        if not os.path.exists(legacy_path):
            self.skipTest("找不到備份 rules_config.py")
        tree = ast.parse(S.read_text(legacy_path))
        legacy = None
        for node in tree.body:
            if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == "INITIAL_BLOCKED_DOMAINS":
                legacy = ast.literal_eval(node.value)
            elif isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "INITIAL_BLOCKED_DOMAINS"
                                                      for t in node.targets):
                legacy = ast.literal_eval(node.value)
        if legacy is None:
            self.skipTest("備份 INITIAL_BLOCKED_DOMAINS 不是字面值，無法比對")
        missing = sorted(set(legacy) - set(R.INITIAL_BLOCKED_DOMAINS))
        self.assertEqual(missing, [], "INITIAL_BLOCKED_DOMAINS 遺失了原有網域")


# =============================================================================
# text_features.py
# =============================================================================

class TextFeaturesAlignmentTests(unittest.TestCase):
    def test_ensemble_vector_not_hardcoded_31(self):
        import text_features as T
        n = len(F.FEATURE_NAMES)
        base = [0.5] * n
        self.assertEqual(T.build_ensemble_vector(base, None), base)
        self.assertEqual(len(T.build_ensemble_vector(base, [1.0, 2.0, 3.0])), n + 3)

    def test_ensemble_source_has_no_literal_31(self):
        src = S.read_text(os.path.join(S.VA_DIR, "text_features.py"))
        tree = ast.parse(src)
        func = next((n for n in ast.walk(tree)
                     if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "build_ensemble_vector"),
                    None)
        self.assertIsNotNone(func, "text_features 缺 build_ensemble_vector")
        consts = [c.value for c in ast.walk(func) if isinstance(c, ast.Constant) and c.value == 31
                  and not isinstance(c.value, bool)]
        self.assertEqual(consts, [], "build_ensemble_vector 程式碼內仍寫死 31")

    def test_bs4_lazy_import(self):
        code = ("import sys; sys.path.insert(0, %r); import text_features; "
                "print('bs4' in sys.modules)") % S.VA_DIR
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip().splitlines()[-1], "False", "text_features 在模組載入時就 import 了 bs4")


# =============================================================================
# 前端 script.js / index.html / 外掛 icons.js
# =============================================================================

class FrontendAlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = S.read_text(SCRIPT_JS)
        cls.index = S.read_text(INDEX_HTML)
        cls.icons_src = S.read_text(ICONS_JS)
        cls.icon_names = icons_js_names(cls.icons_src)
        cls.feature_map = parse_js_object(cls.script, "FEATURE_MAP")

    def test_parser_sanity(self):
        self.assertGreater(len(self.icon_names), 30)
        self.assertIn("ruler", self.icon_names)
        self.assertIn("url_length", self.feature_map)

    def test_feature_map_keys_equal_feature_names(self):
        keys = list(self.feature_map)
        missing = [n for n in F.FEATURE_NAMES if n not in self.feature_map]
        extra = [k for k in keys if k not in F.FEATURE_NAMES]
        self.assertEqual((missing, extra), ([], []),
                         f"script.js FEATURE_MAP 與 FEATURE_NAMES 不一致：缺 {missing}，多 {extra}")

    def test_feature_map_entries_complete(self):
        problems = []
        for key, body in self.feature_map.items():
            for field in ("name", "unit", "desc", "icon"):
                if not re.search(r"\b" + field + r"\s*:", body):
                    problems.append(f"{key} 缺 {field}")
        self.assertEqual(problems, [])

    def test_feature_map_icons_exist(self):
        bad = []
        for key, body in self.feature_map.items():
            m = re.search(r"""\bicon\s*:\s*['"]([^'"]+)['"]""", body)
            if not m:
                bad.append(f"{key}: 無 icon")
            elif m.group(1) not in self.icon_names:
                bad.append(f"{key}: {m.group(1)}")
        self.assertEqual(bad, [], "FEATURE_MAP 引用了 icons.js 不存在的圖示")

    def test_all_script_icon_literals_exist(self):
        used = set(re.findall(r"""\bicon\s*:\s*['"]([a-z0-9-]+)['"]""", self.script))
        used |= set(re.findall(r"""\bico\(\s*['"]([a-z0-9-]+)['"]""", self.script))
        self.assertEqual(sorted(used - self.icon_names), [], "script.js 引用了不存在的圖示")

    def test_index_icon_block_matches_icons_js(self):
        m = re.search(r"<!--\s*truth-icons:start\s*-->\s*<script>(.*?)</script>\s*<!--\s*truth-icons:end\s*-->",
                      self.index, re.S)
        self.assertIsNotNone(m, "index.html 找不到 truth-icons:start/end 內嵌區塊")
        self.assertEqual(_norm_newlines(m.group(1)), _norm_newlines(self.icons_src),
                         "index.html 內嵌圖示區塊與 外掛/icons.js 不一致（請同步貼回）")

    def test_index_icon_names_exist(self):
        page = re.sub(r"<!--\s*truth-icons:start\s*-->.*?<!--\s*truth-icons:end\s*-->", "", self.index, flags=re.S)
        page = re.sub(r"<!--.*?-->", "", page, flags=re.S)
        used = set(re.findall(r"""data-icon=["']([a-z0-9-]+)["']""", page))
        self.assertEqual(sorted(used - self.icon_names), [], "index.html data-icon 引用了不存在的圖示")

    def test_index_no_hardcoded_31_features(self):
        visible = re.sub(r"<!--.*?-->", "", self.index, flags=re.S)
        visible = re.sub(r"<script>.*?</script>", "", visible, flags=re.S)
        hits = re.findall(r"31\s*(?:個|項|維|種)\s*(?:特徵|指標)?", visible)
        self.assertEqual(hits, [], "index.html 仍寫死 31 個特徵，應依 API 動態顯示")

    def test_script_uses_backend_assessment(self):
        self.assertIn("feature_assessment", self.script,
                      "script.js 應優先使用後端回傳的 feature_assessment（門檻唯一來源）")

    def test_error_fields_read_both(self):
        """讀 API 回應錯誤時須同時相容 error 與 message（CONTRACT §3/§4）。"""
        files = {"script.js": self.script}
        for name in ("background.js", "popup.js", "content.js"):
            files[name] = S.read_text(os.path.join(S.EXT_DIR, name))
        bad = []
        for fname, src in files.items():
            for lineno, line in enumerate(src.splitlines(), 1):
                for m in re.finditer(r"\bdata\.(error|message)\s*\|\|", line):
                    other = "message" if m.group(1) == "error" else "error"
                    if f"data.{other}" not in line:
                        bad.append(f"{fname}:{lineno}: {line.strip()[:90]}")
        self.assertEqual(bad, [], "只讀 error 或只讀 message 其中之一")


class ExtensionContractTests(unittest.TestCase):
    def test_background_uses_batch_endpoint(self):
        src = S.read_text(os.path.join(S.EXT_DIR, "background.js"))
        self.assertIn("/predict/batch", src, "background.js 的 CHECK_LINKS 應改用 /predict/batch")
        self.assertRegex(src, r"compact\s*:\s*true", "批次請求應帶 compact:true")

    def test_manifest_loads_icons(self):
        with open(os.path.join(S.EXT_DIR, "manifest.json"), encoding="utf-8") as fh:
            manifest = json.load(fh)
        self.assertEqual(manifest.get("manifest_version"), 3)
        scripts = [s for cs in manifest.get("content_scripts", []) for s in cs.get("js", [])]
        if scripts:
            self.assertIn("icons.js", scripts)
            self.assertLess(scripts.index("icons.js"), scripts.index("content.js") if "content.js" in scripts else 99)
        for s in scripts:
            self.assertTrue(os.path.exists(os.path.join(S.EXT_DIR, s)), f"manifest 引用不存在的 {s}")


if __name__ == "__main__":
    unittest.main()
