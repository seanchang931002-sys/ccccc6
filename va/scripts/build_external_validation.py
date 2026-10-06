#!/usr/bin/env python3
# =============================================================================
# scripts/build_external_validation.py
# =============================================================================
# 用途：協助建置「獨立外部驗證集」（見 docs/EXTERNAL_VALIDATION_PLAN.md）。
#
# 本腳本**不會**自動產生任何網址或標籤——候選名單必須由人工查核後
# 準備成 CSV（欄位：url,label,source,checked_by,checked_at），這支腳本
# 只負責：
#   1. 把候選名單與 data.csv／feedback.csv 的網址／網域做查重比對，
#      標示出「已存在於訓練資料」的項目（這些不該進外部驗證集，否則
#      失去「模型從未看過」的驗證意義）；
#   2. 做基本的格式檢查（label 必須是 0/1、url 不可為空）；
#   3. 輸出去重後的 external_validation.csv。
#
# 用法：
#   python scripts/build_external_validation.py \
#       --candidates candidates.csv \
#       --output external_validation.csv
# =============================================================================

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Set

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features import get_registered_domain, normalize_url  # noqa: E402


def load_known_urls_and_domains(va_dir: Path) -> tuple[Set[str], Set[str]]:
    """讀入 data.csv 與 feedback.csv，回傳 (已知網址集合, 已知 registered domain 集合)。"""
    known_urls: Set[str] = set()
    known_domains: Set[str] = set()

    for fname in ["data.csv", "data_clean.csv", "feedback.csv"]:
        path = va_dir / fname
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if not row or not row[0].strip():
                    continue
                url = row[0].strip()
                try:
                    norm = normalize_url(url)
                    known_urls.add(norm)
                    domain = get_registered_domain(norm)
                    if domain:
                        known_domains.add(domain)
                except Exception:
                    known_urls.add(url)

    return known_urls, known_domains


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, help="人工查核後的候選名單 CSV")
    parser.add_argument("--output", default="external_validation.csv", help="輸出檔名")
    parser.add_argument("--va-dir", default=str(Path(__file__).resolve().parent.parent),
                         help="va/ 目錄路徑（預設為本腳本的上一層）")
    args = parser.parse_args()

    va_dir = Path(args.va_dir)
    candidates_path = Path(args.candidates)

    if not candidates_path.exists():
        print(f"[錯誤] 找不到候選名單檔案：{candidates_path}", file=sys.stderr)
        return 1

    known_urls, known_domains = load_known_urls_and_domains(va_dir)
    print(f"[資訊] 已知網址數：{len(known_urls)}，已知 registered domain 數：{len(known_domains)}")

    kept: List[Dict[str, str]] = []
    dropped_duplicate: List[Dict[str, str]] = []
    dropped_invalid: List[Dict[str, str]] = []

    with open(candidates_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required_cols = {"url", "label", "source", "checked_by", "checked_at"}
        missing = required_cols - set(reader.fieldnames or [])
        if missing:
            print(f"[錯誤] 候選名單缺少必要欄位：{missing}", file=sys.stderr)
            print(f"       必要欄位：{sorted(required_cols)}", file=sys.stderr)
            return 1

        for row in reader:
            url = (row.get("url") or "").strip()
            label = (row.get("label") or "").strip()

            if not url or label not in ("0", "1"):
                dropped_invalid.append(row)
                continue

            try:
                norm = normalize_url(url)
                domain = get_registered_domain(norm)
            except Exception:
                dropped_invalid.append(row)
                continue

            # normalize_url() 設計上對寬鬆輸入較寬容（因為要處理真實世界五花八門的
            # URL 寫法），但這裡是驗證集查核流程的最後一道防線，必須更嚴格：
            # 沒有 registered domain（代表不是一個像樣的網域）就直接剔除，
            # 避免「not a url」這類明顯不是網址的髒資料混進驗證集。
            if not domain or "." not in domain:
                dropped_invalid.append(row)
                continue

            if norm in known_urls or (domain and domain in known_domains):
                dropped_duplicate.append(row)
                continue

            kept.append({**row, "url": norm})

    # 一律寫出檔案（即使 0 筆也只寫表頭），避免舊的輸出檔殘留造成「以為是
    # 最新結果」的誤解——曾經在開發這支腳本時自己踩過這個坑。
    with open(args.output, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["url", "label", "source", "checked_by", "checked_at"])
        writer.writeheader()
        writer.writerows(kept)

    print(f"[完成] 保留 {len(kept)} 筆 → {args.output}")
    print(f"[警告] 與既有訓練資料重複，已排除 {len(dropped_duplicate)} 筆")
    print(f"[警告] 格式不正確，已排除 {len(dropped_invalid)} 筆")

    if kept:
        n_benign = sum(1 for r in kept if r["label"] == "0")
        n_scam = sum(1 for r in kept if r["label"] == "1")
        print(f"[統計] Benign={n_benign}，Scam={n_scam}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
