# -*- coding: utf-8 -*-
"""
訓練資料流程回歸測試（data_sources.py、clean_data.py、train_model.py 的資料／擴增／驗證工具）。

不做完整訓練（python train_model.py 需數分鐘），只檢查：
  - canonical 去重、衝突排除、feedback 以最新為準、165 開放資料自動偵測與略過
  - 擴增：真實樣本形狀翻轉、模板雙胞胎、url_cases 洩漏防護、group 指派
  - group 切分：測試索引只含真實樣本、同 group 不跨折
  - verify_environment() 與 model_meta.json／scam_model.pkl 一致（存在時）
所有暫存檔都寫在 tempfile 目錄，不動專案內任何資料檔。

執行（在 va 目錄下）：python -m unittest tests.test_training_pipeline -v
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
import warnings

import _support as S  # noqa: F401  (把 va/ 加進 sys.path)

import numpy as np
import pandas as pd

import data_sources as DS
import features as F
import train_model as T

warnings.filterwarnings("ignore", category=DeprecationWarning)


class DedupeAndCleanTests(unittest.TestCase):
    def test_dedupe_key_equivalence(self):
        same = ["www.adidas.com.tw", "https://www.adidas.com.tw", "https://www.adidas.com.tw/",
                "HTTPS://WWW.ADIDAS.COM.TW/", " www.adidas.com.tw　"]
        self.assertEqual(len({DS.dedupe_key(u) for u in same}), 1)
        self.assertNotEqual(DS.dedupe_key("http://a.com/"), DS.dedupe_key("https://a.com/"),
                            "明確 http 與 https 是不同寫法（is_https 不同）")
        self.assertNotEqual(DS.dedupe_key("https://a.com/x"), DS.dedupe_key("https://a.com/"))

    def test_clean_labeled_frame(self):
        df = pd.DataFrame({
            DS.URL_COL: ["www.a.com", "https://www.a.com/", "b.top", "https://b.top", "", None,
                         "SM-wholesale shop", "c.vip", "d.cc"],
            DS.LABEL_COL: [0, 0, 1, 0, 1, 0, 1, 1, "x"],
        })
        clean, conflicts, stats = DS.clean_labeled_frame(df, verbose=False)
        keys = set(clean["_key"])
        self.assertIn("https://www.a.com", keys)
        self.assertNotIn("https://b.top", keys, "同網址標籤衝突應排除")
        self.assertEqual(stats["conflict_urls"], 1)
        self.assertEqual(stats["duplicates"], 1)
        self.assertEqual(stats["invalid_host"], 1, "非網址（無點號主機）應移除")
        self.assertEqual(stats["invalid_label"], 1)
        self.assertEqual(len(clean), 2)
        self.assertEqual(set(conflicts["_key"]), {"https://b.top"})


class FeedbackTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="tm-fb-")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_latest_wins_and_overrides_dataset(self):
        path = os.path.join(self.tmp, "feedback.csv")
        pd.DataFrame({
            "url": ["x-shop.top", "https://x-shop.top/", "y.com", "y.com"],
            "label": [0, 1, 1, 1],
            "ai_score": [0.1, 0.9, 0.5, 0.5],
            "note": ["", "", "", ""],
            "time": ["2026-07-01T10:00:00", "2026-07-02T10:00:00", "", ""],
            "source": ["user_feedback"] * 4,
        }).to_csv(path, index=False, encoding="utf-8-sig")
        fb, info = DS.load_feedback(path, weight=2.0, verbose=False)
        self.assertEqual(info["unique"], 2)
        self.assertEqual(info["conflict_urls"], 1)
        row = fb[fb["url"] == "https://x-shop.top"].iloc[0]
        self.assertEqual(int(row["label"]), 1, "同網址應以最新 time 為準")
        self.assertEqual(float(row["weight"]), 2.0)

        main = pd.DataFrame({"url": ["https://x-shop.top", "https://z.com"], "raw_url": ["x-shop.top", "z.com"],
                             "label": [0, 0], "source": "dataset", "weight": 1.0, "time": "", "note": ""})
        merged, rep = DS.merge_sources(main, fb, None)
        self.assertEqual(int(merged.loc[merged["url"] == "https://x-shop.top", "label"].iloc[0]), 1)
        self.assertEqual(len(rep["feedback_override"]), 1)
        self.assertEqual(rep["feedback_new"], 1)

    def test_missing_feedback(self):
        fb, info = DS.load_feedback(os.path.join(self.tmp, "nope.csv"), verbose=False)
        self.assertTrue(fb.empty)


class OpenDataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="tm-165-")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_missing_dir_is_skipped(self):
        df, info = DS.load_165_opendata(os.path.join(self.tmp, "no-such-dir"), verbose=False)
        self.assertTrue(df.empty)
        self.assertTrue(info["skipped"])

    def test_autodetect_columns_encodings_and_typo_tld(self):
        # 165 假投資(博弈)網站：cp950、網址欄名「網址」、一格兩個網址、打錯的 .comn
        pd.DataFrame({
            "網站名稱": ["OKX-S", "星鏈數位理財", "測試", "空白"],
            "網址": ["m.okxsweb3.com", "www.nhkutdp.comn", "a1.example-scam.top、b2.example-scam.vip", "-"],
            "件數": [3, 2, 1, 0],
        }).to_csv(os.path.join(self.tmp, "165_invest.csv"), index=False, encoding="cp950")
        # 停止解析清單：utf-8-sig、網址欄名「網域」、另有民國年月
        pd.DataFrame({
            "民國年月": ["11507", "11508"],
            "網域": ["qzxkvbn.shop", "not a url"],
            "網站性質": ["電子商務", "釣魚網站"],
        }).to_csv(os.path.join(self.tmp, "stop_opendata_2026.csv"), index=False, encoding="utf-8-sig")
        # 不符合檔名樣式的檔案應被忽略
        pd.DataFrame({"url": ["ignored.top"]}).to_csv(os.path.join(self.tmp, "other.csv"), index=False)

        df, info = DS.load_165_opendata(self.tmp, verbose=False)
        urls = set(df["url"])
        self.assertIn("https://m.okxsweb3.com", urls)
        self.assertIn("https://www.nhkutdp.com", urls, ".comn 應修正為 .com")
        self.assertIn("https://a1.example-scam.top", urls)
        self.assertIn("https://b2.example-scam.vip", urls)
        self.assertIn("https://qzxkvbn.shop", urls)
        self.assertNotIn("https://ignored.top", urls)
        self.assertTrue((df["label"] == 1).all())
        cols = {f["file"]: f.get("url_column") for f in info["files"]}
        self.assertEqual(cols["165_invest.csv"], "網址")
        self.assertEqual(cols["stop_opendata_2026.csv"], "網域")
        self.assertEqual(info["fixed_tld"], 1)

        # 與主資料衝突（主資料標為正常）→ 保留主資料、列入衝突
        main = pd.DataFrame({"url": ["https://qzxkvbn.shop"], "raw_url": ["qzxkvbn.shop"], "label": [0],
                             "source": "dataset", "weight": 1.0, "time": "", "note": ""})
        merged, rep = DS.merge_sources(main, DS._empty_std(), df)
        self.assertEqual(int(merged.loc[merged["url"] == "https://qzxkvbn.shop", "label"].iloc[0]), 0)
        self.assertEqual(rep["opendata_conflicts"], ["https://qzxkvbn.shop"])


class AugmentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = T.load_url_cases()
        real, _ = T.load_dataset(verbose=False)
        cls.real = real
        cls.frame, cls.stats = T.build_training_frame(real, cls.cases, verbose=False)

    def test_columns_and_flags(self):
        for col in ("url", "key", "label", "weight", "is_synthetic", "group", "category", "template"):
            self.assertIn(col, self.frame.columns)
        self.assertTrue(set(self.frame["is_synthetic"].unique()) <= {0, 1})
        self.assertEqual(int((self.frame["is_synthetic"] == 0).sum()), len(self.real))
        self.assertFalse(self.frame["group"].isna().any())
        self.assertTrue((self.frame["weight"] > 0).all())

    def test_no_url_cases_leakage(self):
        case_keys = {DS.dedupe_key(c["url"]) for c in self.cases}
        synth = self.frame[self.frame["is_synthetic"] == 1]
        self.assertEqual(set(synth["key"]) & case_keys, set(), "擴增列不得與驗收清單同網址")
        scam_hosts = {F.get_hostname(c["url"]) for c in self.cases if c["expect"] == "scam"}
        scam_hosts -= {"line.me", "t.me", "lin.ee", "chat.whatsapp.com", "discord.gg", "google.com", "bit.ly",
                       "reurl.cc", "tinyurl.com", "ipfs.io"}
        tpl_scam = synth[(synth["label"] == 1) & ~synth["category"].str.startswith("real")]
        self.assertEqual(set(tpl_scam["url"].map(F.get_hostname)) & scam_hosts, set())

    def test_synthetic_does_not_overwhelm_real(self):
        tpl_w = self.stats["weight_sum_templates"]
        real_w = self.stats["weight_sum_real_family"]
        self.assertLess(tpl_w, real_w * 0.7, "模板擴增權重不應壓倒真實資料")
        for cat in ("scam_gambling", "scam_investment", "scam_fake_exchange", "scam_ephemeral_host",
                    "benign_substring_trap", "benign_tw_media", "benign_finance", "benign_crypto_exchange",
                    "benign_cloud_docs", "benign_mobile_official", "benign_retail_tracking"):
            self.assertIn(cat, self.stats["by_category"], cat)

    def test_shape_balance(self):
        """加權後兩類樣本「有路徑／有參數／https」比例應接近（消除捷徑）。"""
        X = T.build_feature_matrix(self.frame["url"].tolist())
        D = pd.DataFrame(X, columns=F.FEATURE_NAMES)
        y, w = self.frame["label"].to_numpy(), self.frame["weight"].to_numpy()
        for name, col in (("has_path", D["path_length"] > 0), ("has_query", D["query_length"] > 0),
                          ("is_https", D["is_https"] > 0), ("has_utm", D["has_utm"] > 0)):
            rates = [np.average(col[y == k], weights=w[y == k]) for k in (0, 1)]
            self.assertLess(abs(rates[0] - rates[1]), 0.08, f"{name} 加權比例差距過大：{rates}")

    def test_real_families_have_both_shapes(self):
        fam = self.frame[self.frame["category"].str.startswith("real_variant")]
        parents = set(fam["parent"])
        sample = self.real[self.real["url"].isin(parents)].head(200)
        for url in sample["url"]:
            rows = pd.concat([self.frame[self.frame["key"] == url], fam[fam["parent"] == url]])
            if len(rows) < 3:
                continue  # 變體與驗收清單或其他真實列重複而被洩漏防護剔除（例如 165.npa.gov.tw）
            shapes = {T._split_canonical(F.canonicalize_url(u))[1] in ("", "/") for u in rows["url"]}
            self.assertEqual(shapes, {True, False}, f"{url} 的家族應同時含裸網域與有路徑寫法")
            self.assertEqual(rows["group"].nunique(), 1, "同家族必須同 group")

    def test_group_splits_eval_real_only(self):
        y = self.frame["label"].to_numpy()
        groups = self.frame["group"].to_numpy()
        is_real = (self.frame["is_synthetic"] == 0).to_numpy()
        splits = T.make_group_splits(y, groups, is_real)
        seen = set()
        for train_idx, test_idx in splits:
            self.assertTrue(is_real[test_idx].all(), "評估索引只能是真實樣本")
            self.assertEqual(set(groups[train_idx]) & set(groups[test_idx]), set(), "同 group 不可跨訓練／測試")
            seen |= set(test_idx.tolist())
        self.assertEqual(seen, set(np.where(is_real)[0].tolist()), "每筆真實樣本都要被評估一次")

    def test_shape_twin(self):
        g = T._Gen(1)
        pool = ["/zh-tw/products/1"]
        self.assertIn(T._split_canonical(F.canonicalize_url(T.shape_twin(g, "https://abc-shop.top/login", pool)))[1],
                      ("", "/"))
        twin = T.shape_twin(g, "abc-shop.top", pool)
        self.assertNotIn(T._split_canonical(F.canonicalize_url(twin))[1], ("", "/"))


class ArtifactTests(unittest.TestCase):
    def test_verify_environment_matches_meta(self):
        if not os.path.exists(T.META_PATH):
            self.skipTest("model_meta.json 不存在")
        with open(T.META_PATH, encoding="utf-8") as fh:
            meta = json.load(fh)
        if meta.get("feature_names") != F.FEATURE_NAMES:
            self.skipTest("model_meta.json 尚未以 v7 特徵重訓")
        self.assertTrue(T.verify_environment())
        for key in ("feature_names", "feature_version", "feature_schema_id", "sklearn_version", "numpy_version",
                    "best_model", "params", "thresholds", "metrics", "data_count", "label_counts", "synthetic_count"):
            self.assertIn(key, meta, key)
        self.assertEqual(meta["thresholds"], {"medium": 0.40, "high": 0.70})

    def test_report_contents(self):
        if not os.path.exists(T.REPORT_PATH):
            self.skipTest("model_report.json 不存在")
        with open(T.REPORT_PATH, encoding="utf-8") as fh:
            rep = json.load(fh)
        if "cv_design" not in rep:
            self.skipTest("model_report.json 為舊版")
        self.assertEqual(len(rep["permutation_importance_top15"]), 15)
        self.assertIn("false_positives", rep["url_cases"])
        self.assertIn("false_negatives", rep["url_cases"])
        self.assertIn("shortcut_check", rep)
        self.assertLess(rep["shortcut_check"]["max_neutral_spread"], 0.25,
                        "同網域裸網域／https／一般路徑的模型機率差距應很小")

    def test_bundle_scheme_invariance(self):
        if not os.path.exists(T.MODEL_PATH):
            self.skipTest("scam_model.pkl 不存在")
        import joblib
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            obj = joblib.load(T.MODEL_PATH)
        if not isinstance(obj, dict) or obj.get("feature_names") != F.FEATURE_NAMES:
            self.skipTest("scam_model.pkl 尚未以 v7 bundle 重訓")
        est = obj["estimator"]
        for host in ("az6rxm.com", "tw-fxmart.top", "www.momoshop.com.tw"):
            p = est.predict_proba(T.build_feature_matrix([host, f"https://{host}/"]))[:, 1]
            self.assertAlmostEqual(float(p[0]), float(p[1]), places=6, msg=f"{host}：裸網域與 https:// 應同分")


if __name__ == "__main__":
    unittest.main()
