# -*- coding: utf-8 -*-
"""
features.py 行為測試：健壯性、scheme 不變性、token 化誤判回歸、v7 新特徵正例，
以及 text_features 深度掃描的 SSRF 防護。

執行（在 va 目錄下）：
    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import asyncio
import math
import unittest

import _support as S  # noqa: E402  (同時把 va/ 加進 sys.path)

import features as F  # noqa: E402

# 長度／字元類特徵：必須以「去掉 scheme 與 www.」後的字串計算，與 scheme 無關。
LENGTH_CHAR_FEATURES = (
    "url_length", "domain_length", "path_length", "query_length", "digit_ratio",
    "hyphen_count", "dot_count", "special_chars", "subdomain_depth", "path_depth",
    "query_params", "long_domain", "domain_entropy", "many_subdomains",
    "contains_percent_encoding", "sld_digit_count", "sld_length",
)

# 「詐騙語意」類旗標：hard negative 不得觸發。
SCAM_SEMANTIC_FLAGS = (
    "has_scam_word", "gambling_number_pattern", "suspicious_keyword_in_domain",
    "gambling_keyword", "investment_lure_keyword", "crypto_exchange_lure",
    "brand_typo_like", "brand_impersonation", "social_invite_link", "path_scam_route",
)


def _strip_scheme(url: str) -> str:
    for p in ("https://", "http://"):
        if url.lower().startswith(p):
            return url[len(p):]
    return url


def _assert_valid_feature_dict(tc: unittest.TestCase, fd, label: str) -> None:
    tc.assertIsInstance(fd, dict, label)
    tc.assertEqual(list(fd.keys()), F.FEATURE_NAMES, f"{label}：key 不完整或順序不同")
    for k, v in fd.items():
        tc.assertIsInstance(v, (int, float), f"{label}：{k} 不是數值（{type(v).__name__}）")
        tc.assertFalse(isinstance(v, float) and (math.isnan(v) or math.isinf(v)), f"{label}：{k}={v}")


class FixtureExtractionTests(unittest.TestCase):
    def test_all_fixture_urls_extract(self):
        errors = []
        for case in S.load_cases():
            url = case["url"]
            try:
                fd = F.extract_feature_dict(url)
                _assert_valid_feature_dict(self, fd, url)
                vec = F.extract_features(url)
                self.assertEqual(len(vec), len(F.FEATURE_NAMES), url)
                F.explain_features(url, feature_dict=fd)
                F.assess_all(fd)
            except AssertionError:
                raise
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{url}: {type(exc).__name__}: {exc}")
        self.assertEqual(errors, [], "特徵擷取不得丟例外")

    def test_explain_shape(self):
        for case in S.load_cases()[:120]:
            for r in F.explain_features(case["url"]):
                self.assertEqual(set(r) >= {"key", "level", "message"}, True, r)
                self.assertIn(r["level"], ("high", "medium"), r)
                self.assertTrue(r["message"])


class MaliciousInputTests(unittest.TestCase):
    SAMPLES = [
        "", "   ", None, "http://[abc", "https://[::1", "http://[::1]:8080/x",
        "a" * 5000, "https://" + "a" * 5000 + ".com/", "https://example.com/" + "%" * 3000,
        "goo​gle.com", "﻿https://example.com", "ｈｔｔｐｓ：／／ｅｘａｍｐｌｅ．ｃｏｍ／",
        "https://exam　ple.com", "https://例子.測試/路徑", "https://xn--/",
        "http://192.168.0.1/login", "http://999.999.999.999/", "http://[2001:db8::1]/",
        "https://example.com:99999/", "https://example.com:abc/", "https://example.com:/",
        "https://user:pass@evil.com/", "https://@evil.com", "@@@", "http://", "https://",
        "javascript:alert(1)", "data:text/html,<script>alert(1)</script>", "mailto:a@b.com",
        "\x00\x01http://x.com", "http://x.com/\n\r\t/path", "https://.....", "https://-.-/",
        "http://%zz%zz/", "https://example.com/%E4%B8", "//example.com/path", "\\\\server\\share",
    ]

    def test_never_raises_and_keys_complete(self):
        for s in self.SAMPLES:
            with self.subTest(sample=repr(s)[:60]):
                fd = F.extract_feature_dict(s)
                _assert_valid_feature_dict(self, fd, repr(s)[:60])
                self.assertEqual(len(F.extract_features(s)), len(F.FEATURE_NAMES))
                F.explain_features(s if isinstance(s, str) else "", fd)
                F.canonicalize_url(s)
                F.url_has_explicit_scheme(s)
                for fn in (F.get_hostname, F.get_registered_domain, F.get_sld, F.get_tld):
                    self.assertIsInstance(fn(s), str)

    def test_helpers_never_raise(self):
        for s in self.SAMPLES:
            F.parse_url(s)
            F.normalize_url(s)
            F.is_ip_address(s)
            F.calc_entropy(s)
            F.min_brand_levenshtein(s)
        self.assertEqual(F.levenshtein_distance("", ""), 0)
        self.assertEqual(F.levenshtein_distance("kitten", "sitting"), 3)

    def test_specific_values(self):
        self.assertEqual(F.extract_feature_dict("http://[abc")["is_ip_address"], 0)
        self.assertEqual(F.extract_feature_dict("http://192.168.0.1/login")["is_ip_address"], 1)
        self.assertEqual(F.extract_feature_dict("https://example.com:8443/")["non_standard_port"], 1)
        self.assertEqual(F.extract_feature_dict("https://example.com:443/")["non_standard_port"], 0)
        self.assertEqual(F.extract_feature_dict("http://example.com:80/")["non_standard_port"], 0)
        self.assertEqual(F.extract_feature_dict("https://user:pass@evil.com/")["url_has_at_symbol"], 1)

    def test_zero_width_and_fullwidth_equivalent(self):
        plain = F.extract_feature_dict("https://google.com/")
        self.assertEqual(F.extract_feature_dict("https://goo​gle.com/"), plain)
        self.assertEqual(F.extract_feature_dict("ｈｔｔｐｓ：／／ｇｏｏｇｌｅ．ｃｏｍ／"), plain)
        self.assertEqual(F.extract_feature_dict("  https://google.com/  "), plain)


class SchemeInvarianceTests(unittest.TestCase):
    """CONTRACT §0-1 / §1：scheme 不得影響長度／字元類特徵（消除訓練捷徑）。"""

    def test_fixture_bare_vs_https(self):
        diffs = []
        for case in S.load_cases():
            bare = _strip_scheme(case["url"])
            a = F.extract_feature_dict(bare)
            b = F.extract_feature_dict("https://" + bare)
            d = {k: (a[k], b[k]) for k in F.FEATURE_NAMES if a[k] != b[k]}
            if d:
                diffs.append((bare, d))
        self.assertEqual(diffs, [], "裸網域與 https:// 前綴的特徵應完全相同（無 scheme 視為 https）")

    def test_fixture_http_vs_https_only_is_https(self):
        diffs = []
        for case in S.load_cases():
            bare = _strip_scheme(case["url"])
            a = F.extract_feature_dict("http://" + bare)
            b = F.extract_feature_dict("https://" + bare)
            d = {k: (a[k], b[k]) for k in LENGTH_CHAR_FEATURES if a[k] != b[k]}
            if d:
                diffs.append((bare, d))
            if a["is_https"] != 0 or b["is_https"] != 1:
                diffs.append((bare, {"is_https": (a["is_https"], b["is_https"])}))
        self.assertEqual(diffs, [], "http 與 https 只能在 is_https 上不同")

    def test_www_does_not_change_length(self):
        for bare in ("books.com.tw/", "stock-vip-tw.top/h5/#/register", "example.com/a/b?c=1"):
            a = F.extract_feature_dict("https://" + bare)
            b = F.extract_feature_dict("https://www." + bare)
            self.assertEqual({k: a[k] for k in LENGTH_CHAR_FEATURES}, {k: b[k] for k in LENGTH_CHAR_FEATURES}, bare)

    def test_canonicalize(self):
        self.assertEqual(F.canonicalize_url("books.com.tw"), "https://books.com.tw")
        self.assertTrue(F.canonicalize_url("HTTP://EXAMPLE.com/Path").startswith("http://example.com/"))
        self.assertTrue(F.canonicalize_url("HTTP://EXAMPLE.com/Path").endswith("/Path"), "路徑大小寫應保留")
        self.assertFalse(F.url_has_explicit_scheme("books.com.tw/x"))
        self.assertTrue(F.url_has_explicit_scheme("https://books.com.tw/x"))


class TokenizationRegressionTests(unittest.TestCase):
    """CONTRACT §0-3：英文詞一律 token 比對，不得裸子字串命中。"""

    HARD_NEGATIVES = [
        "https://www.duolingo.com/learn",            # earn ⊂ learn
        "https://learn.microsoft.com/zh-tw/",         # earn ⊂ learn
        "https://www.alphabet.com/",                  # bet ⊂ alphabet
        "https://www.betterhelp.com/",                # bet ⊂ better
        "https://www.method.com/together",            # eth ⊂ method/together
        "https://www.online.com/",                    # line ⊂ online
        "https://onlinelibrary.wiley.com/",           # line ⊂ online
        "https://www.airline.com/",                   # line ⊂ airline
        "https://www.1688.com/",                      # 168 ⊂ 1688
        "https://shopee.sg/",                         # shope ⊂ shopee（正牌）
        "https://shopee.tw/",
        "https://www.poyabuy.com.tw/",                # 正牌寶雅
        "https://www.cosmed.com.tw/",                 # 正牌康是美
        "https://www.watsons.com.tw/",                # 正牌屈臣氏
    ]

    def test_hard_negatives_no_scam_flags(self):
        bad = []
        for url in self.HARD_NEGATIVES:
            fd = F.extract_feature_dict(url)
            hit = [k for k in SCAM_SEMANTIC_FLAGS if fd[k]]
            if hit:
                bad.append((url, hit))
        self.assertEqual(bad, [])

    def test_official_brand_distance_not_zero(self):
        # 正牌網域不得顯示為「與品牌距離 0」（前端曾把 0 誤標為正版品牌）
        for url in ("https://shopee.sg/", "https://www.poyabuy.com.tw/", "https://www.cosmed.com.tw/",
                    "https://www.1688.com/"):
            self.assertEqual(F.extract_feature_dict(url)["levenshtein_brand_dist"], 99, url)

    def test_token_positive_counterparts(self):
        # 對照組：真正以 token 出現時仍要命中
        self.assertEqual(F.extract_feature_dict("https://bet-888.top/")["gambling_number_pattern"]
                         or F.extract_feature_dict("https://bet-888.top/")["gambling_keyword"], 1)
        self.assertEqual(F.extract_feature_dict("https://fubon168.top/")["brand_impersonation"], 1)
        self.assertEqual(F.extract_feature_dict("https://shopeemall-tw.shop/")["brand_impersonation"], 1)


class NewFeaturePositiveTests(unittest.TestCase):
    CASES = [
        ("https://xn--shpee-vqa.com/", "punycode_domain"),
        ("https://www.esunbank.com.tw@evil-login.xyz/", "url_has_at_symbol"),
        ("https://binance.com@binance-tw.top/login", "url_has_at_symbol"),
        ("https://tw-stock-vip.top:8443/", "non_standard_port"),
        ("http://45.32.10.1:8888/h5/", "non_standard_port"),
        ("https://a1b2c3.ngrok-free.app/login", "tunnel_or_ephemeral_host"),
        ("https://abc-def-ghi.trycloudflare.com/", "tunnel_or_ephemeral_host"),
        ("https://line.me/ti/g/AbCdEf123", "social_invite_link"),
        ("https://line.me/R/ti/g/AbCdEf123", "social_invite_link"),
        ("https://t.me/+AbCdEfGh123", "social_invite_link"),
        ("https://t.me/joinchat/AbCdEfGh", "social_invite_link"),
        ("https://chat.whatsapp.com/AbCdEf123", "social_invite_link"),
        ("https://discord.gg/abcdef", "social_invite_link"),
        ("https://tougu-vip888.com/#/pages/login/login?invitecode=AB12CD", "path_scam_route"),
        ("https://abc-ex.com/h5/#/register", "path_scam_route"),
        ("https://app-exchange.cc/register?inviteCode=X9Y8", "path_scam_route"),
        ("https://stock-vip-tw.web.app/", "free_hosting_platform"),
        ("https://binance-tw-pro.github.io/", "free_hosting_platform"),
        ("https://usdt-otc-tw.shop/", "crypto_exchange_lure"),
        ("https://bocai365.cc/", "gambling_keyword"),
        ("https://example-tougu.top/%E9%A3%86%E8%82%A1%E7%BE%A4", "investment_lure_keyword"),
        ("https://binance-tw-pro.com/", "brand_impersonation"),
    ]

    def test_positive_flags(self):
        bad = [(u, k, F.extract_feature_dict(u)[k]) for u, k in self.CASES if F.extract_feature_dict(u)[k] != 1]
        self.assertEqual(bad, [])

    def test_negative_counterparts(self):
        neg = [
            ("https://www.google.com:443/", "non_standard_port"),
            ("https://medium.com/@someone", "url_has_at_symbol"),
            ("https://www.threads.net/@zuck", "url_has_at_symbol"),
            ("https://line.me/R/ti/p/@linebank", "social_invite_link"),
            ("https://t.me/s/somechannel", "social_invite_link"),
            ("https://www.google.com/", "punycode_domain"),
            ("https://www.google.com/", "free_hosting_platform"),
            ("https://www.google.com/", "tunnel_or_ephemeral_host"),
            ("https://docs.github.com/", "free_hosting_platform"),
            ("https://www.gov.tw/", "path_scam_route"),
        ]
        bad = [(u, k) for u, k in neg if F.extract_feature_dict(u)[k] != 0]
        self.assertEqual(bad, [])

    def test_numeric_features(self):
        fd = F.extract_feature_dict("https://fubon168.top/")
        self.assertEqual(fd["sld_length"], 8)
        self.assertEqual(fd["sld_digit_count"], 3)
        self.assertEqual(fd["tld_risk_level"], 2)
        self.assertEqual(F.extract_feature_dict("https://www.google.com/")["tld_risk_level"], 0)
        self.assertIn(F.extract_feature_dict("https://example.xyz/")["tld_risk_level"], (1, 2))
        for url in ("https://a.b.c.example.co.uk/", "https://x.y/"):
            fd = F.extract_feature_dict(url)
            self.assertGreaterEqual(fd["sld_length"], 0)
            self.assertIn(fd["tld_risk_level"], (0, 1, 2))


class DeepScanSSRFTests(unittest.TestCase):
    """CONTRACT §0-11：deep_scan 不得對內網／metadata 位址發出請求。"""

    class _RecordingClient:
        def __init__(self):
            self.calls = []

        def __getattr__(self, name):
            if name in ("get", "post", "stream", "request", "send", "head"):
                def _rec(*args, **kwargs):
                    self.calls.append((name, args[:1]))
                    raise RuntimeError("測試用假 client：不應發出任何請求")
                return _rec
            raise AttributeError(name)

        async def aclose(self):
            return None

    def test_private_targets_not_fetched(self):
        import text_features as T
        targets = [
            "http://127.0.0.1/", "http://localhost:8000/admin", "http://169.254.169.254/latest/meta-data/",
            "http://10.0.0.5/", "http://192.168.1.1/", "http://[::1]/", "http://0.0.0.0/",
            "file:///etc/passwd", "ftp://example.com/",
        ]
        for url in targets:
            client = self._RecordingClient()
            res = asyncio.run(T.analyze_page_semantics(url, client=client))
            self.assertEqual(client.calls, [], f"{url} 不應被抓取")
            self.assertFalse(res.get("fetched"), url)


if __name__ == "__main__":
    unittest.main()
