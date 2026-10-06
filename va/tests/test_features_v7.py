# -*- coding: utf-8 -*-
"""
features.py / rules_config.py v7 回歸測試（標準庫 unittest）。

執行：在 va 目錄下
    python -m unittest discover -s tests -p "test_*.py" -v
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import unittest

VA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if VA_DIR not in sys.path:
    sys.path.insert(0, VA_DIR)

import features as F  # noqa: E402
import rules_config as R  # noqa: E402

FIXTURE_PATH = os.path.join(VA_DIR, "tests", "fixtures", "url_cases.json")

LEGACY_31 = [
    "url_length", "domain_length", "path_length", "query_length", "digit_ratio",
    "hyphen_count", "dot_count", "special_chars", "subdomain_depth", "path_depth",
    "query_params", "is_https", "is_ip_address", "suspicious_tld", "has_scam_word",
    "brand_in_sld", "has_utm", "has_gclid", "double_http", "long_domain", "domain_entropy",
    "has_shortener", "gambling_number_pattern", "brand_typo_like", "cloud_hosting",
    "mobile_lure_path", "suspicious_keyword_in_domain", "many_subdomains",
    "contains_percent_encoding", "levenshtein_brand_dist", "newly_registered_like",
]
NEW_14 = [
    "tld_risk_level", "gambling_keyword", "investment_lure_keyword", "crypto_exchange_lure",
    "brand_impersonation", "free_hosting_platform", "tunnel_or_ephemeral_host",
    "social_invite_link", "punycode_domain", "url_has_at_symbol", "non_standard_port",
    "sld_digit_count", "sld_length", "path_scam_route",
]
NEW_V71 = ["sld_randomness"]
NEW_FLAGS = [n for n in NEW_14 if n not in ("sld_digit_count", "sld_length")]


def hard_rule_hits(url):
    return R.match_hard_rules(F.get_rule_targets(url))


def load_cases():
    with open(FIXTURE_PATH, encoding="utf-8") as fh:
        return json.load(fh)


class SchemaTests(unittest.TestCase):
    def test_feature_names_order(self):
        self.assertEqual(F.FEATURE_NAMES[:31], LEGACY_31)
        self.assertEqual(F.FEATURE_NAMES[31:45], NEW_14)
        self.assertEqual(F.FEATURE_NAMES[45:], NEW_V71)
        expected = len(LEGACY_31) + len(NEW_14) + len(NEW_V71)   # v7.1：46
        self.assertEqual(len(F.FEATURE_NAMES), expected)
        self.assertEqual(len(set(F.FEATURE_NAMES)), expected)

    def test_schema_id(self):
        expected = hashlib.sha1("|".join(F.FEATURE_NAMES).encode("utf-8")).hexdigest()[:12]
        self.assertEqual(F.FEATURE_SCHEMA_ID, expected)
        self.assertEqual(F.FEATURE_VERSION, "7.1.0")

    def test_specs_cover_all(self):
        self.assertEqual(set(F.FEATURE_SPECS), set(F.FEATURE_NAMES))
        for name, spec in F.FEATURE_SPECS.items():
            self.assertIn(spec["kind"], ("count", "ratio", "bool", "level", "score"), name)
            self.assertTrue(spec["zh"], name)
            self.assertIn("unit", spec)

    def test_assess_statuses(self):
        for name in F.FEATURE_NAMES:
            for value in (0, 1, 2, 3.7, 99, 500, None, "x"):
                result = F.assess_feature(name, value)
                self.assertIn(result["status"], ("alert", "warn", "safe", "neutral"))
                self.assertTrue(result["text"])
        self.assertEqual(F.assess_feature("levenshtein_brand_dist", 0)["status"], "alert")
        self.assertEqual(F.assess_feature("levenshtein_brand_dist", 99)["status"], "safe")
        self.assertEqual(F.assess_feature("is_https", 0)["status"], "warn")

    def test_extract_order(self):
        url = "https://stock-vip-tw.top/h5/#/register"
        self.assertEqual(F.extract_features(url), [F.extract_feature_dict(url)[n] for n in F.FEATURE_NAMES])


class CanonicalizeTests(unittest.TestCase):
    def test_scheme_added(self):
        self.assertEqual(F.canonicalize_url("books.com.tw"), "https://books.com.tw")
        self.assertEqual(F.canonicalize_url("http://Example.COM/A/B"), "http://example.com/A/B")
        self.assertEqual(F.canonicalize_url("ｈｔｔｐｓ：／／ｗｗｗ．ｇｏｏｇｌｅ．ｃｏｍ／"), "https://www.google.com/")
        self.assertEqual(F.canonicalize_url("goo​gle.com"), "https://google.com")
        self.assertFalse(F.url_has_explicit_scheme("example.com:8080/x"))
        self.assertTrue(F.url_has_explicit_scheme("http://example.com"))

    def test_scheme_and_www_do_not_change_length_features(self):
        variants = ["books.com.tw", "https://books.com.tw", "https://www.books.com.tw/",
                    "www.books.com.tw", "https://WWW.BOOKS.COM.TW/", "https://www.books.com.tw./"]
        keys = ("url_length", "domain_length", "dot_count", "special_chars", "subdomain_depth",
                "digit_ratio", "path_length", "is_https")
        baseline = {k: F.extract_feature_dict(variants[0])[k] for k in keys}
        for v in variants[1:]:
            self.assertEqual({k: F.extract_feature_dict(v)[k] for k in keys}, baseline, v)
        self.assertEqual(F.extract_feature_dict("http://www.books.com.tw/")["is_https"], 0)

    def test_hostname_and_psl(self):
        self.assertEqual(F.get_hostname("https://www.cw.com.tw/article/1"), "cw.com.tw")
        self.assertEqual(F.get_registered_domain("https://sub.momoshop.com.tw/x"), "momoshop.com.tw")
        self.assertEqual(F.get_registered_domain("https://shopee.com.my/"), "shopee.com.my")
        self.assertEqual(F.get_sld("https://binance-tw-pro.github.io/"), "binance-tw-pro")
        self.assertEqual(F.get_tld("https://www.pola.co.jp/"), "jp")
        self.assertEqual(F.get_registered_domain("http://1.2.3.4:8080/"), "1.2.3.4")

    def test_never_raises(self):
        samples = ["", None, "http://[abc", "a" * 5000, "https://x.com:abc/", "javascript:alert(1)",
                   "https://娛樂城.vip/", "http://%zz/", 12345, ["x"], b"bytes", "\x00http://x", "@@@"]
        for s in samples:
            fd = F.extract_feature_dict(s)
            self.assertEqual(len(fd), len(F.FEATURE_NAMES))
            F.explain_features(s if isinstance(s, str) else "", fd)
            F.get_rule_targets(s)
            F.get_hostname(s)
            F.min_brand_levenshtein(s)
        self.assertEqual(F.extract_feature_dict("http://[abc")["is_ip_address"], 0)


class SemanticTests(unittest.TestCase):
    def test_substring_traps(self):
        for url in ("https://www.duolingo.com/learn", "https://www.alphabet.com/", "https://www.method.com/",
                    "https://www.1688.com/", "https://onlinelibrary.wiley.com/", "https://www.airlines.org/",
                    "https://www.betterhelp.com/"):
            fd = F.extract_feature_dict(url)
            for key in ("has_scam_word", "gambling_number_pattern", "suspicious_keyword_in_domain",
                        "gambling_keyword", "brand_typo_like", "brand_impersonation"):
                self.assertEqual(fd[key], 0, f"{url} {key}")

    def test_official_brands_not_spoof(self):
        for url in ("https://www.poyabuy.com.tw/", "https://www.cosmed.com.tw/", "https://www.watsons.com.tw/",
                    "https://shopee.sg/", "https://www.linebank.com.tw/", "https://www.binance.com/zh-TC"):
            fd = F.extract_feature_dict(url)
            self.assertEqual(fd["brand_typo_like"], 0, url)
            self.assertEqual(fd["brand_impersonation"], 0, url)
            self.assertEqual(fd["levenshtein_brand_dist"], 99, url)

    def test_short_brand_lookalikes(self):
        for url in ("https://www.moma.org/", "https://www.nine.com.au/", "https://www.pola.co.jp/"):
            self.assertEqual(F.extract_feature_dict(url)["levenshtein_brand_dist"], 99, url)

    def test_percent_decoded_chinese(self):
        fd = F.extract_feature_dict("https://example-tougu.top/%E9%A3%86%E8%82%A1%E7%BE%A4")
        self.assertEqual(fd["investment_lure_keyword"], 1)
        news = F.extract_feature_dict("https://tw.news.yahoo.com/%E7%B7%9A%E4%B8%8A%E8%B3%AD%E5%A0%B4-1.html")
        self.assertEqual(news["gambling_keyword"], 0)

    def test_new_flags(self):
        cases = {
            "https://a1b2.ngrok-free.app/login": "tunnel_or_ephemeral_host",
            "https://stock-vip-tw.web.app/": "free_hosting_platform",
            "https://line.me/R/ti/g/AbCdEf123": "social_invite_link",
            "https://xn--binnce-wsa.com/": "punycode_domain",
            "https://www.esunbank.com.tw@evil-login.xyz/": "url_has_at_symbol",
            "https://tw-stock-vip.top:8443/": "non_standard_port",
            "https://tougu-vip888.com/#/pages/login/login?invitecode=AB12CD": "path_scam_route",
            "https://binance-tw-pro.com/": "brand_impersonation",
            "https://bocai365.cc/": "gambling_keyword",
            "https://usdt-otc-tw.shop/": "crypto_exchange_lure",
        }
        for url, key in cases.items():
            self.assertEqual(F.extract_feature_dict(url)[key], 1, f"{url} {key}")
        self.assertEqual(F.extract_feature_dict("https://line.me/R/ti/p/@linebank")["social_invite_link"], 0)
        self.assertEqual(F.extract_feature_dict("https://www.threads.net/@zuck")["url_has_at_symbol"], 0)


class HardRuleTests(unittest.TestCase):
    def test_compile_and_targets(self):
        self.assertEqual(R.validate_hard_rules(), [])
        for rule in R.HARD_RULES:
            self.assertIn(rule.target, R.VALID_RULE_TARGETS)
            self.assertTrue(0 < rule.score <= 100)

    def test_benign_fixtures_no_medium_rule(self):
        failures = []
        for case in load_cases():
            if case["expect"] != "benign":
                continue
            bad = [f"{r.name}:{r.score}" for r in hard_rule_hits(case["url"]) if r.score >= 40]
            if bad:
                failures.append((case["url"], bad))
        self.assertEqual(failures, [])

    def test_scam_fixtures_have_signal(self):
        failures = []
        for case in load_cases():
            if case["expect"] != "scam":
                continue
            fd = F.extract_feature_dict(case["url"])
            if not hard_rule_hits(case["url"]) and not any(fd[k] for k in NEW_FLAGS):
                failures.append(case["url"])
        self.assertEqual(failures, [])

    def test_known_regressions(self):
        self.assertEqual([r.name for r in hard_rule_hits("https://www.cw.com.tw/") if r.score >= 40], [])
        self.assertEqual([r.name for r in hard_rule_hits("https://www.poyabuy.com.tw/") if r.score >= 40], [])
        names = {r.name for r in hard_rule_hits("https://www.binance.com@binance-tw-pro.top/login")}
        self.assertIn("url_userinfo_at_spoof", names)


if __name__ == "__main__":
    unittest.main()
