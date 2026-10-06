# =============================================================================
# clean_data.py — VeriAd 資料清理腳本
# =============================================================================
# 功能：
# 1. 清除 data.csv 中的空白列
# 2. 清除無效 Label
# 3. 移除重複 URL
# 4. 檢查是否有同一 URL 但標籤衝突
# 5. 輸出 data_clean.csv
#
# 修正紀錄（v7.0）：
#   - 正規化與衝突處理改用 data_sources.clean_labeled_frame()（與 train_model.py 共用）：
#     URL 先經 features.normalize_url（NFKC、移除零寬字元與所有空白），
#     重複與衝突以 features.canonicalize_url 的結果判斷（沒有 scheme 時視同 https://，
#     「www.a.com.tw」與「https://www.a.com.tw/」是同一筆）。
#   - 輸出檔保留清理後的原始寫法（不強制補 scheme），訓練時再統一 canonicalize。
#   - 無法解析出主機的非網址資料（例如「SM-wholesale shop」「imtoken」）一併移除。
#   - 同 URL 標籤衝突：排除並輸出 conflict_urls.csv 供人工檢查（主資料沒有時間戳，
#     無法判斷哪一筆較新；feedback.csv 的衝突則由 train_model.py 以最新為準）。
#   - 可用 --input / --output 指定檔案（預設 data.csv → data_clean.csv）。
#
# 使用方式：
# python clean_data.py
# =============================================================================

import argparse
import os

import pandas as pd

from data_sources import LABEL_COL, URL_COL, clean_labeled_frame, dedupe_key
from features import normalize_url as _features_normalize_url

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_PATH = os.path.join(BASE_DIR, "data.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data_clean.csv")
CONFLICT_PATH = os.path.join(BASE_DIR, "conflict_urls.csv")


def normalize_url(url) -> str:
    """
    （保留舊介面）基本 URL 正規化：改用 features.normalize_url
    （NFKC、移除零寬字元、全形空白與所有空白）。去重／衝突判斷請用 canonical_key()。
    """
    if url is None or (isinstance(url, float) and pd.isna(url)):
        return ""
    return _features_normalize_url(url)


def canonical_key(url) -> str:
    """去重與衝突判斷用的 key（features.canonicalize_url；與 train_model.py 相同）。"""
    return dedupe_key(url)


def load_raw_data(path: str = INPUT_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到資料檔案：{path}")

    df = pd.read_csv(path, encoding="utf-8-sig")

    if URL_COL not in df.columns:
        raise ValueError(f"找不到欄位：{URL_COL}")

    if LABEL_COL not in df.columns:
        raise ValueError(f"找不到欄位：{LABEL_COL}")

    return df


def clean_data(df: pd.DataFrame, conflict_path: str = CONFLICT_PATH) -> pd.DataFrame:
    print("=" * 70)
    print("VeriAd 資料清理開始（v7：canonicalize_url 正規化）")
    print("=" * 70)

    print(f"原始資料筆數：{len(df)}")

    clean_df, conflict_df, stats = clean_labeled_frame(df, URL_COL, LABEL_COL, verbose=False)

    print(f"移除空 URL：{stats['empty_url']} 筆")
    print(f"移除無效 Label：{stats['invalid_label']} 筆")
    print(f"移除無法解析主機的非網址資料：{stats['invalid_host']} 筆")

    if not conflict_df.empty:
        print("\n警告：發現同一 URL（canonical 後）有不同標籤，請人工檢查：")
        for url in conflict_df[URL_COL].drop_duplicates().head(20):
            print(" -", url)

        conflict_df[[URL_COL, LABEL_COL]].to_csv(
            conflict_path,
            index=False,
            encoding="utf-8-sig"
        )
        print(f"標籤衝突資料已輸出：{conflict_path}")
        print(f"為避免污染訓練資料，已排除 {stats['conflict_urls']} 個衝突 URL（{stats['conflict_rows']} 列）")
    else:
        print("未發現同 URL 不同標籤衝突")

    print(f"移除重複資料（canonical 相同）：{stats['duplicates']} 筆")

    # 重新排序
    clean_df = clean_df[[URL_COL, LABEL_COL]].sort_values(by=[LABEL_COL, URL_COL]).reset_index(drop=True)

    print("\n清理後資料統計：")
    print(clean_df[LABEL_COL].value_counts().rename(index={0: "正常", 1: "詐騙"}))

    print(f"\n清理後總筆數：{len(clean_df)}")

    return clean_df


def main():
    parser = argparse.ArgumentParser(description="VeriAd 資料清理（data.csv → data_clean.csv）")
    parser.add_argument("--input", default=INPUT_PATH, help="輸入 CSV（預設 data.csv）")
    parser.add_argument("--output", default=OUTPUT_PATH, help="輸出 CSV（預設 data_clean.csv）")
    parser.add_argument("--conflicts", default=CONFLICT_PATH, help="衝突清單輸出路徑")
    args = parser.parse_args()

    df = load_raw_data(args.input)
    clean_df = clean_data(df, conflict_path=args.conflicts)

    clean_df.to_csv(
        args.output,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n資料清理完成")
    print(f"輸出檔案：{args.output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
