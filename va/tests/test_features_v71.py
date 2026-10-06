# -*- coding: utf-8 -*-
"""
features.py v7.1 回歸測試（標準庫 unittest）：
  - sld_randomness（第 46 個特徵）：範圍、可讀網域低、隨機字串高、永不丟例外、統計表讀不到時的退回
  - newly_registered_like 改用 sld_randomness：交接列出的合法品牌不再命中，隨機字母網域仍命中
  - dot_count／digit_ratio／special_chars 只算主機名稱（路徑寫法不影響）
  - 黏字切分：biaoguvip、188bettw、kucoinexchangevip、wakuangvip、netbankscsb 能命中，一般英文字不受影響
  - char_ngram_table.json 格式與語料說明（不得使用訓練／驗收網址）

執行（在 va 目錄下）：python -m unittest discover -s tests -p "test_features_v71.py" -v
"""
from __future__ import annotations

import json
import os
import unittest

import _support as S  # noqa: F401  (把 va/ 加進 sys.path)

import features as F

TABLE_PATH = os.path.join(S.VA_DIR, "char_ngram_table.json")

READABLE = ["threads", "flickr", "tumblr", "scribd", "sprinklr", "google", "stackoverflow", "chinatimes",
            "yulecheng", "tougu", "momoshop", "kingstone"]
RANDOM_LIKE = ["xkqplbwz", "zucfjdi", "flnvf", "fxmnvg", "lbqmzpd", "hhdshhfh"]
KNOWN_LEGIT = ["https://www.threads.net/@zuck", "https://www.flickr.com/", "https://www.tumblr.com/",
               "https://www.scribd.com/", "https://www.zdnet.com/", "https://www.mcdonalds.com/",
               "https://kktix.com/", "https://www.kfcclub.com.tw/", "https://www.hrblock.com/",
               "https://www.sprinklr.com/", "https://www.zscaler.com/"]


def _clear_caches():
    for fn in (F._sld_randomness_cached, F._compute_feature_dict_cached):
        fn.cache_clear()


class SldRandomnessTests(unittest.TestCase):
    def test_schema_and_spec(self):
        self.assertEqual(F.FEATURE_NAMES[-1], "sld_randomness")
        self.assertEqual(F.FEATURE_SPECS["sld_randomness"]["kind"], "score")
        self.assertEqual(F.assess_feature("sld_randomness", 0.9)["status"], "warn")
        self.assertEqual(F.assess_feature("sld_randomness", 0.6)["status"], "neutral")
        self.assertEqual(F.assess_feature("sld_randomness", 0.1)["status"], "safe")

    def test_table_loaded(self):
        self.assertTrue(F.NGRAM_TABLE_LOADED, "char_ngram_table.json 應存在且格式正確")

    def test_range_and_never_raises(self):
        for value in ("", None, 12345, "a", "x" * 500, "xn--fiq228c", "8591", "-", "娛樂城", b"bytes", ["x"]):
            r = F.sld_randomness(value)
            self.assertTrue(0.0 <= r <= 1.0, f"{value!r} → {r}")
        self.assertEqual(F.sld_randomness("8591"), 0.0, "純數字 SLD 不算隨機字母")

    def test_readable_low(self):
        for sld in READABLE:
            self.assertLess(F.sld_randomness(sld), 0.35, sld)
        for sld in ("kktix", "mcdonalds", "zdnet", "hrblock", "zscaler"):
            self.assertLess(F.sld_randomness(sld), F.NEWLY_REGISTERED_THRESHOLD, sld)

    def test_random_high(self):
        for sld in RANDOM_LIKE:
            self.assertGreaterEqual(F.sld_randomness(sld), F.NEWLY_REGISTERED_THRESHOLD, sld)
        self.assertGreater(F.sld_randomness("xkqplbwz"), F.sld_randomness("threads") + 0.5)

    def test_feature_value_uses_sld_only(self):
        a = F.extract_feature_dict("https://zucfjdi.cn/")["sld_randomness"]
        b = F.extract_feature_dict("http://m.zucfjdi.cn/h5/#/register?utm_source=fb")["sld_randomness"]
        self.assertEqual(a, b)
        self.assertEqual(F.extract_feature_dict("http://45.12.34.56:8888/login")["sld_randomness"], 0.0)

    def test_fallback_without_table(self):
        saved = F._NGRAM
        try:
            F._NGRAM = None
            _clear_caches()
            for sld in READABLE + RANDOM_LIKE:
                self.assertTrue(0.0 <= F.sld_randomness(sld) <= 1.0)
            fd = F.extract_feature_dict("https://xkqplbwz.top/")
            self.assertEqual(len(fd), len(F.FEATURE_NAMES))
            self.assertEqual(fd["newly_registered_like"], 1, "退回舊版子音規則時隨機網域仍應命中")
            self.assertGreater(F.sld_randomness("xkqplbwz"), F.sld_randomness("threads"))
        finally:
            F._NGRAM = saved
            _clear_caches()
        self.assertIsNone(F._load_ngram_table(os.path.join(S.TESTS_DIR, "no-such-table.json")))


class NewlyRegisteredTests(unittest.TestCase):
    def test_known_legit_not_flagged(self):
        for url in KNOWN_LEGIT:
            self.assertEqual(F.extract_feature_dict(url)["newly_registered_like"], 0, url)

    def test_random_domains_flagged(self):
        for url in ("https://zucfjdi.cn/", "https://www.flnvf.cc/", "https://fxmnvg.com/", "https://xkqplbwz.top/"):
            self.assertEqual(F.extract_feature_dict(url)["newly_registered_like"], 1, url)

    def test_brand_and_whitelist_excluded(self):
        # 銀行縮寫可能偏高，但在白名單／品牌清單內不得命中
        for url in ("https://www.hncb.com.tw/", "https://www.scsb.com.tw/", "https://www.twse.com.tw/"):
            self.assertEqual(F.extract_feature_dict(url)["newly_registered_like"], 0, url)


class HostScopeCharFeatureTests(unittest.TestCase):
    KEYS = ("dot_count", "digit_ratio", "special_chars")

    def test_path_does_not_change_char_features(self):
        variants = ["az6rxm.com", "https://az6rxm.com/", "https://www.az6rxm.com/zh-tw/index.html?utm_source=fb&id=12345",
                    "https://az6rxm.com/news/2025/10/03/1234567.html", "http://az6rxm.com/#/pages/login/login"]
        base = {k: F.extract_feature_dict(variants[0])[k] for k in self.KEYS}
        for v in variants[1:]:
            self.assertEqual({k: F.extract_feature_dict(v)[k] for k in self.KEYS}, base, v)
        self.assertAlmostEqual(base["digit_ratio"], 1 / len("az6rxm.com"))
        self.assertEqual(base["dot_count"], 1)

    def test_host_values(self):
        fd = F.extract_feature_dict("https://secure-login.a.b.example-pay.xyz/x.y.z?a=1.2.3")
        self.assertEqual(fd["dot_count"], 4)
        self.assertEqual(fd["special_chars"], 6)
        self.assertEqual(F.extract_feature_dict("http://45.12.34.56:8888/login")["dot_count"], 3)
        self.assertEqual(F.extract_feature_dict("javascript:alert(1)")["dot_count"], 0)


class StickyKeywordTests(unittest.TestCase):
    def test_sticky_hits(self):
        cases = {
            "https://188bettw.com/": "gambling_keyword",
            "https://biaoguvip.com/": "investment_lure_keyword",
            "https://twlicaiplus.com/": "investment_lure_keyword",
            "https://kucoinexchangevip.com/": "crypto_exchange_lure",
            "https://wakuangvip.com/": "crypto_exchange_lure",
            "https://netbankscsb.com/": "brand_impersonation",
        }
        for url, key in cases.items():
            self.assertEqual(F.extract_feature_dict(url)[key], 1, f"{url} {key}")
        self.assertEqual(F.extract_feature_dict("https://kucoinexchangevip.com/")["brand_impersonation"], 1)

    def test_no_false_split(self):
        keys = ("gambling_keyword", "investment_lure_keyword", "crypto_exchange_lure", "brand_impersonation",
                "suspicious_keyword_in_domain")
        for url in ("https://www.alphabet.com/", "https://www.betterhelp.com/", "https://shopline.tw/",
                    "https://www.tibet.net/", "https://www.kucoin.com/", "https://www.coinbase.com/",
                    "https://www.betaseries.com/", "https://www.investopedia.com/", "https://www.tradingview.com/",
                    "https://www.netbank.com.au/", "https://www.goldmansachs.com/"):
            fd = F.extract_feature_dict(url)
            self.assertEqual([k for k in keys if fd[k]], [], url)


class NgramTableTests(unittest.TestCase):
    def test_table_format(self):
        with open(TABLE_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
        self.assertEqual(data["format"], "truthmark-char-ngram/1")
        self.assertEqual(len(data["bi"]), 27 * 27)
        self.assertGreater(len(data["tri"]), 1000)
        self.assertIn("excluded", data["corpus"])
        self.assertLess(os.path.getsize(TABLE_PATH), 200 * 1024, "統計表應保持精簡")
        self.assertTrue(os.path.exists(os.path.join(S.VA_DIR, "tools", "build_ngram_table.py")))


if __name__ == "__main__":
    unittest.main()
