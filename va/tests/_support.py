# -*- coding: utf-8 -*-
"""
測試共用工具（非測試模組，檔名以底線開頭，unittest discover 不會把它當成測試收集）。

- 統一把專案根目錄（va/）加進 sys.path，讓測試不論從哪個目錄啟動都能 import 產品模組。
- 讀取共用回歸清單 tests/fixtures/url_cases.json。
- 以 ast 解析備份版 features.py 的 FEATURE_NAMES（只讀，不 import 備份模組）。
- 真實資料檔快照／還原：保證測試不會改壞 scam_model.pkl、dynamic_blocklist.json、
  feedback.csv、domain_reports.json。
"""
from __future__ import annotations

import ast
import json
import os
import sys
from typing import Dict, List, Optional

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
VA_DIR = os.path.dirname(TESTS_DIR)
EXT_DIR = os.path.join(VA_DIR, "外掛")
FIXTURE_PATH = os.path.join(TESTS_DIR, "fixtures", "url_cases.json")

# 原始完整備份（審查前快照）。預設不設定（空字串），與備份比對的測試會 skip；
BACKUP_DIR = os.environ.get("TRUTHMARK_BACKUP_DIR", "")

for _p in (VA_DIR, TESTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# 前 31 個特徵（v6 以前的模型輸入），名稱與順序不得變動。
LEGACY_31: List[str] = [
    "url_length", "domain_length", "path_length", "query_length", "digit_ratio",
    "hyphen_count", "dot_count", "special_chars", "subdomain_depth", "path_depth",
    "query_params", "is_https", "is_ip_address", "suspicious_tld", "has_scam_word",
    "brand_in_sld", "has_utm", "has_gclid", "double_http", "long_domain", "domain_entropy",
    "has_shortener", "gambling_number_pattern", "brand_typo_like", "cloud_hosting",
    "mobile_lure_path", "suspicious_keyword_in_domain", "many_subdomains",
    "contains_percent_encoding", "levenshtein_brand_dist", "newly_registered_like",
]

# v7 依序追加的 14 個特徵（CONTRACT.md §1）。
NEW_14: List[str] = [
    "tld_risk_level", "gambling_keyword", "investment_lure_keyword", "crypto_exchange_lure",
    "brand_impersonation", "free_hosting_platform", "tunnel_or_ephemeral_host",
    "social_invite_link", "punycode_domain", "url_has_at_symbol", "non_standard_port",
    "sld_digit_count", "sld_length", "path_scam_route",
]

# v7.1 追加的特徵（CONTRACT.md §5.1）。
NEW_V71: List[str] = ["sld_randomness"]

VALID_SOURCES = {"trusted_domain", "blocklist", "hybrid_ai_hard_rule", "ai_model", "rules_only"}
VALID_STATUSES = {"alert", "warn", "safe", "neutral"}

# 測試期間可能被 API 寫入、必須快照還原的真實資料檔。
# 注意：scam_model.pkl / model_meta.json 不在此列——測試只讀不寫，且訓練腳本可能同時在重訓，
# 若在這裡「還原」反而會蓋掉剛訓練好的模型。
PROTECTED_FILES = ("dynamic_blocklist.json", "feedback.csv", "domain_reports.json")


def load_cases() -> List[dict]:
    with open(FIXTURE_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def read_text(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def backup_feature_names() -> Optional[List[str]]:
    """以 ast 解析備份 features.py 的 FEATURE_NAMES 字面清單；備份不存在時回傳 None。"""
    if not BACKUP_DIR:  # 未指定備份目錄：不可退回目前工作目錄（會讀到現行檔案）
        return None
    path = os.path.join(BACKUP_DIR, "features.py")
    if not os.path.exists(path):
        return None
    tree = ast.parse(read_text(path))
    for node in tree.body:
        target = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target, value = node.target.id, node.value
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target, value = node.targets[0].id, node.value
        if target == "FEATURE_NAMES":
            return list(ast.literal_eval(value))
    return None


class ProtectedFiles:
    """快照真實資料檔（位元組＋是否存在），測試結束後原樣還原。"""

    def __init__(self, base_dir: str = VA_DIR, names=PROTECTED_FILES) -> None:
        self.paths = [os.path.join(base_dir, n) for n in names]
        self._snap: Dict[str, Optional[bytes]] = {}

    def snapshot(self) -> None:
        for p in self.paths:
            if os.path.exists(p):
                with open(p, "rb") as fh:
                    self._snap[p] = fh.read()
            else:
                self._snap[p] = None

    def changed(self) -> List[str]:
        out = []
        for p, data in self._snap.items():
            now = None
            if os.path.exists(p):
                with open(p, "rb") as fh:
                    now = fh.read()
            if now != data:
                out.append(os.path.basename(p))
        return out

    def restore(self) -> List[str]:
        """還原並回傳被動過的檔名（理想狀況應為空清單）。"""
        touched = self.changed()
        for p, data in self._snap.items():
            try:
                if data is None:
                    if os.path.exists(p):
                        os.remove(p)
                else:
                    with open(p, "rb") as fh:
                        if fh.read() == data:
                            continue
                    with open(p, "wb") as fh:
                        fh.write(data)
            except OSError as exc:  # pragma: no cover - 只在檔案被鎖住時發生
                print(f"[tests] 還原 {p} 失敗：{exc}")
        return touched
