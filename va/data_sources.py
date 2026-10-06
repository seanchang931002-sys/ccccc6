# =============================================================================
# data_sources.py — TruthMark / VeriAd 訓練資料來源與正規化（v7.0 新增）
# =============================================================================
# train_model.py 與 clean_data.py 共用：
#   - 網址正規化一律使用 features.canonicalize_url（訓練／清理／推論同一函式，
#     沒有 scheme 時補 https://，消除「有 https = 正常」捷徑）
#   - dedupe_key()：canonical 網址；「https://host」與「https://host/」視為同一筆
#   - 衝突處理：
#       * 主資料（data_clean.csv / data.csv）同一網址標籤不一致 → 沒有時間戳無法判斷，
#         排除並列出供人工檢查
#       * feedback.csv 同一網址多次回報 → 以最新一筆（time 欄位；缺少時以列順序）為準，
#         並覆蓋主資料的標籤（使用者回報視為較新的人工判斷）
#       * 165 開放資料（label=1）與主資料標籤衝突 → 保留主資料標籤並列出供人工檢查
#   - load_165_opendata()：可選的 165 開放資料匯入（掃描 data/ 下 165*.csv 或
#     *opendata*.csv，自動偵測網址欄位與編碼；不存在時印出略過；**不連網下載**）
# =============================================================================

from __future__ import annotations

import fnmatch
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from features import (PUBLIC_SUFFIX_FALLBACKS, canonicalize_url, get_hostname,
                      get_registered_domain, is_ip_address, normalize_url)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(BASE_DIR, "data")

URL_COL = "URL"
LABEL_COL = "Label（1 代表詐騙，0 代表正常）"

# 統一後的欄位
STD_COLUMNS = ["url", "raw_url", "label", "source", "weight", "time", "note"]

# 165 開放資料：檔名樣式（不分大小寫）
OPENDATA_PATTERNS = ("165*.csv", "*opendata*.csv")
# 網址欄位名稱關鍵字（越前面越優先；實際仍以內容判斷）
URL_HEADER_HINTS = (
    "網址", "偽冒網址", "網域名稱", "網域", "域名", "網站網址", "url", "domain", "website", "link", "連結",
)
NAME_HEADER_HINTS = ("網站名稱", "名稱", "name", "網站性質", "類別")
DATE_HEADER_HINTS = ("統計結束日期", "停止解析日期", "通報受理日期", "民國年月", "日期", "date")
# 165 原始資料常見打錯的 TLD（研究報告 §2.3）；無法修正者剔除
TYPO_TLD_FIXES = {
    "comn": "com", "comm": "com", "con": "com", "cmo": "com", "ocm": "com", "coom": "com",
    "nett": "net", "nte": "net", "ogr": "org", "vipp": "vip", "topp": "top",
}
_ENCODINGS = ("utf-8-sig", "utf-8", "cp950", "big5", "utf-16")
_SPLIT_CELL_RE = re.compile(r"[\s、，,;；|]+")
_LOOKS_LIKE_HOST_RE = re.compile(r"^(?:[a-z][a-z0-9+.\-]*://)?[^\s/?#@]*\.[^\s/?#]+", re.I)


# =============================================================================
# 共用：正規化 key
# =============================================================================

def dedupe_key(url: Any) -> str:
    """去重用 key：canonical 網址；只有根路徑的「/」不計（https://a.com 與 https://a.com/ 相同）。"""
    c = canonicalize_url(url)
    m = re.match(r"^([a-z][a-z0-9+.\-]*://[^/?#]*)/?$", c)
    return m.group(1) if m else c


def has_valid_host(url: Any) -> bool:
    """網址能解析出「含點號的主機名稱」或 IP（排除 'SM-wholesale shop'、'imtoken' 這類非網址）。"""
    host = get_hostname(url)
    return bool(host) and ("." in host or bool(is_ip_address(host)))


def _empty_std() -> pd.DataFrame:
    return pd.DataFrame(columns=STD_COLUMNS)


def _to_label(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


# =============================================================================
# 主資料（data_clean.csv / data.csv）
# =============================================================================

def clean_labeled_frame(
    df: pd.DataFrame,
    url_col: str = URL_COL,
    label_col: str = LABEL_COL,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """
    清理「URL + Label」表格（clean_data.py 與 train_model.py 共用）：
      1. normalize_url（NFKC、移除零寬字元與所有空白）
      2. 移除空 URL、無效 Label
      3. 以 dedupe_key（canonical）偵測同網址標籤衝突 → 排除並回傳衝突列
      4. 以 dedupe_key 去重（保留第一筆）
    回傳 (clean_df[url_col, label_col, "_key"], conflict_df, stats)。
    """
    if url_col not in df.columns:
        raise ValueError(f"找不到欄位：{url_col}")
    if label_col not in df.columns:
        raise ValueError(f"找不到欄位：{label_col}")

    stats: Dict[str, int] = {"raw": int(len(df))}
    out = df[[url_col, label_col]].copy()
    out[url_col] = out[url_col].map(lambda v: "" if pd.isna(v) else normalize_url(v))
    out[label_col] = _to_label(out[label_col])

    before = len(out)
    out = out[(out[url_col] != "") & (out[url_col].str.lower() != "nan")].copy()
    stats["empty_url"] = int(before - len(out))

    before = len(out)
    out = out[out[label_col].isin([0, 1])].copy()
    stats["invalid_label"] = int(before - len(out))
    out[label_col] = out[label_col].astype(int)

    out["_key"] = out[url_col].map(dedupe_key)
    before = len(out)
    out = out[out["_key"].map(has_valid_host)].copy()
    stats["invalid_host"] = int(before - len(out))

    n_labels = out.groupby("_key")[label_col].nunique()
    conflict_keys = set(n_labels[n_labels > 1].index)
    conflict_df = out[out["_key"].isin(conflict_keys)].copy()
    out = out[~out["_key"].isin(conflict_keys)].copy()
    stats["conflict_rows"] = int(len(conflict_df))
    stats["conflict_urls"] = int(len(conflict_keys))

    before = len(out)
    out = out.drop_duplicates(subset=["_key"], keep="first").copy()
    stats["duplicates"] = int(before - len(out))
    stats["clean"] = int(len(out))

    if verbose:
        print(f"  原始 {stats['raw']} 筆；空 URL {stats['empty_url']}、無效 Label {stats['invalid_label']}、"
              f"非網址 {stats['invalid_host']}、標籤衝突 {stats['conflict_urls']} 個網址（{stats['conflict_rows']} 列，已排除）、"
              f"canonical 重複 {stats['duplicates']} → 保留 {stats['clean']} 筆")
    return out.reset_index(drop=True), conflict_df.reset_index(drop=True), stats


def load_main_dataset(clean_path: str, raw_path: str, verbose: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """讀主資料（優先 data_clean.csv，其次 data.csv），回傳標準欄位 DataFrame 與統計。"""
    if os.path.exists(clean_path):
        path = clean_path
    elif os.path.exists(raw_path):
        path = raw_path
    else:
        raise FileNotFoundError("找不到 data_clean.csv 或 data.csv")
    if verbose:
        print(f"主資料：{path}")
    raw = pd.read_csv(path, encoding="utf-8-sig")
    clean, conflicts, stats = clean_labeled_frame(raw, verbose=verbose)
    std = pd.DataFrame({
        "url": clean["_key"],
        "raw_url": clean[URL_COL],
        "label": clean[LABEL_COL].astype(int),
        "source": "dataset",
        "weight": 1.0,
        "time": "",
        "note": "",
    })
    info: Dict[str, Any] = {"path": os.path.basename(path), **stats,
                            "conflict_examples": conflicts[URL_COL].head(20).tolist()}
    return std.reset_index(drop=True), info


# =============================================================================
# feedback.csv（同 URL 以最新為準）
# =============================================================================

def load_feedback(path: str, weight: float = 2.0, verbose: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    讀 feedback.csv（url,label,ai_score,note,time,source）。同一網址（dedupe_key）多次回報時
    以 time 最新者為準（time 缺漏或無法解析時以列順序，越後面越新）。
    """
    info: Dict[str, Any] = {"path": os.path.basename(path), "rows": 0, "unique": 0, "conflict_urls": 0}
    if not os.path.exists(path):
        if verbose:
            print("  找不到 feedback.csv，略過")
        return _empty_std(), info
    try:
        fb = pd.read_csv(path, encoding="utf-8-sig")
    except Exception as exc:  # noqa: BLE001
        if verbose:
            print(f"  讀取 feedback.csv 失敗，略過：{exc}")
        info["error"] = str(exc)
        return _empty_std(), info
    if "url" not in fb.columns or "label" not in fb.columns:
        if verbose:
            print("  feedback.csv 欄位不完整（需要 url、label），略過")
        return _empty_std(), info

    fb = fb.copy()
    fb["_order"] = range(len(fb))
    fb["raw_url"] = fb["url"].map(lambda v: "" if pd.isna(v) else normalize_url(v))
    fb["label"] = _to_label(fb["label"])
    fb = fb[(fb["raw_url"] != "") & (fb["raw_url"].str.lower() != "nan") & fb["label"].isin([0, 1])].copy()
    fb["label"] = fb["label"].astype(int)
    fb["_key"] = fb["raw_url"].map(dedupe_key)
    fb = fb[fb["_key"] != ""].copy()
    info["rows"] = int(len(fb))
    if fb.empty:
        return _empty_std(), info

    time_col = fb["time"] if "time" in fb.columns else pd.Series([""] * len(fb), index=fb.index)
    fb["_ts"] = pd.to_datetime(time_col, errors="coerce")
    n_labels = fb.groupby("_key")["label"].nunique()
    info["conflict_urls"] = int((n_labels > 1).sum())
    # 最新優先：時間（NaT 視為最舊）→ 列順序
    fb["_ts_sort"] = fb["_ts"].fillna(pd.Timestamp.min)
    latest = fb.sort_values(["_ts_sort", "_order"]).drop_duplicates(subset=["_key"], keep="last")
    info["unique"] = int(len(latest))

    std = pd.DataFrame({
        "url": latest["_key"],
        "raw_url": latest["raw_url"],
        "label": latest["label"].astype(int),
        "source": "feedback",
        "weight": float(weight),
        "time": latest["_ts"].map(lambda t: "" if pd.isna(t) else t.isoformat()),
        "note": latest["note"].fillna("").astype(str) if "note" in latest.columns else "",
    })
    if verbose:
        print(f"  feedback.csv：{info['rows']} 筆 → {info['unique']} 個網址"
              f"（同網址標籤不一致 {info['conflict_urls']} 個，已取最新）")
    return std.reset_index(drop=True), info


# =============================================================================
# 165 開放資料（可選，不連網）
# =============================================================================

_PSL_EXTRACTOR: Any = None


def _psl_suffix(host: str) -> str:
    """回傳 host 的公開後綴（離線 PSL）；無法判斷時回傳 ""。"""
    global _PSL_EXTRACTOR
    if _PSL_EXTRACTOR is None:
        try:
            import tldextract
            try:
                _PSL_EXTRACTOR = tldextract.TLDExtract(
                    suffix_list_urls=(), include_psl_private_domains=True, cache_dir=None
                )
            except TypeError:
                _PSL_EXTRACTOR = tldextract.TLDExtract(suffix_list_urls=(), include_psl_private_domains=True)
        except Exception:  # noqa: BLE001
            _PSL_EXTRACTOR = False
    if not _PSL_EXTRACTOR:
        # 缺少 tldextract 時採用 features.py 共用的有限 suffix fallback。
        labels = host.strip(".").lower().split(".") if host.strip(".") else []
        for i in range(len(labels) - 1):
            candidate = ".".join(labels[i:])
            if candidate in PUBLIC_SUFFIX_FALLBACKS:
                return candidate
        if labels and labels[-1] in {
            "com", "net", "org", "gov", "edu", "mil", "biz", "info", "name", "pro",
            "xyz", "top", "vip", "shop", "site", "online", "club", "me", "io", "ai", "app",
            "dev", "tech", "store", "live", "cc", "tv", "ly", "to", "gg", "co", "uk", "de",
            "fr", "jp", "kr", "cn", "au", "nz", "sg", "my", "hk", "tw",
        }:
            return labels[-1]
        return ""
    try:
        return _PSL_EXTRACTOR(host).suffix or ""
    except Exception:  # noqa: BLE001
        return ""


def _host_of(canonical: str) -> str:
    m = re.match(r"^[a-z][a-z0-9+.\-]*://(?:[^/?#@]*@)?([^/?#:]*)", canonical)
    return (m.group(1) if m else "").strip(".")


def sanitize_opendata_url(value: Any) -> Tuple[str, str]:
    """
    把 165 開放資料的一格內容轉成 canonical 網址。
    回傳 (canonical, 狀態)：狀態為 ok / fixed_tld / invalid。
    """
    s = normalize_url(value)
    if not s or s.lower() in ("nan", "none", "-"):
        return "", "invalid"
    c = canonicalize_url(s)
    host = _host_of(c)
    if not host or "." not in host:
        return "", "invalid"
    if re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", host):
        return dedupe_key(c), "ok"
    if _psl_suffix(host):
        return dedupe_key(c), "ok"
    head, _, tld = host.rpartition(".")
    fixed_tld = TYPO_TLD_FIXES.get(tld)
    if head and fixed_tld:
        fixed_host = f"{head}.{fixed_tld}"
        if _psl_suffix(fixed_host):
            return dedupe_key(c.replace(host, fixed_host, 1)), "fixed_tld"
    return "", "invalid"


def find_opendata_files(data_dir: str = DEFAULT_DATA_DIR) -> List[str]:
    """列出 data_dir 下符合 165*.csv 或 *opendata*.csv 的檔案（不遞迴、不分大小寫）。"""
    if not os.path.isdir(data_dir):
        return []
    found: List[str] = []
    for name in sorted(os.listdir(data_dir)):
        lower = name.lower()
        if any(fnmatch.fnmatch(lower, pat) for pat in OPENDATA_PATTERNS):
            path = os.path.join(data_dir, name)
            if os.path.isfile(path):
                found.append(path)
    return found


def _read_csv_any(path: str) -> Tuple[Optional[pd.DataFrame], str]:
    last_error = ""
    for enc in _ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc, dtype=str, keep_default_na=False), enc
        except Exception as exc:  # noqa: BLE001
            last_error = f"{enc}: {exc}"
    return None, last_error


def _valid_ratio(values: Iterable[Any], sample: int = 200) -> float:
    vals = [v for v in values if isinstance(v, str) and v.strip()][:sample]
    if not vals:
        return 0.0
    ok = 0
    for v in vals:
        first = _SPLIT_CELL_RE.split(v.strip())[0]
        if _LOOKS_LIKE_HOST_RE.match(first) and sanitize_opendata_url(first)[0]:
            ok += 1
    return ok / len(vals)


def detect_url_column(df: pd.DataFrame) -> Optional[str]:
    """自動偵測網址欄位：先看欄名關鍵字，再以內容（可解析為網域的比例 ≥ 0.5）確認。"""
    columns = [str(c) for c in df.columns]

    def hint_rank(col: str) -> int:
        lower = col.strip().lower()
        for i, hint in enumerate(URL_HEADER_HINTS):
            if hint in lower:
                return i
        return len(URL_HEADER_HINTS)

    scored = []
    for col in columns:
        ratio = _valid_ratio(df[col].tolist())
        scored.append((col, hint_rank(col), ratio))
    hinted = [s for s in scored if s[1] < len(URL_HEADER_HINTS) and s[2] >= 0.5]
    if hinted:
        hinted.sort(key=lambda s: (-round(s[2], 1), s[1]))
        return hinted[0][0]
    sniffed = [s for s in scored if s[2] >= 0.5]
    if sniffed:
        sniffed.sort(key=lambda s: -s[2])
        return sniffed[0][0]
    return None


def _pick_column(df: pd.DataFrame, hints: Iterable[str], exclude: Optional[str] = None) -> Optional[str]:
    for hint in hints:
        for col in df.columns:
            if col != exclude and hint in str(col).strip().lower():
                return col
    return None


def load_165_opendata(
    data_dir: str = DEFAULT_DATA_DIR,
    max_rows: Optional[int] = 4000,
    weight: float = 1.0,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    可選的 165 開放資料匯入（label 一律為 1）。只讀本機檔案，不連網下載。
      - 掃描 data_dir 下 165*.csv / *opendata*.csv；不存在時印出略過並回傳空表
      - 自動偵測編碼（utf-8-sig / utf-8 / cp950 / big5 / utf-16）與網址欄位
      - 一格多個網址（、，; 換行分隔）會拆開；打錯的 TLD（.comn/.comm…）修正，無法修正者剔除
      - 「網站名稱」欄位（常為被冒用品牌）存入 note
      - max_rows：避免數萬筆官方清單壓倒其他資料（以 key 排序後等距取樣，結果可重現）
    """
    info: Dict[str, Any] = {"data_dir": data_dir, "files": [], "rows": 0, "kept": 0,
                            "fixed_tld": 0, "invalid": 0, "skipped": ""}
    files = find_opendata_files(data_dir)
    if not files:
        reason = "資料夾不存在" if not os.path.isdir(data_dir) else "沒有符合 165*.csv / *opendata*.csv 的檔案"
        info["skipped"] = reason
        if verbose:
            print(f"  165 開放資料：{reason}（{data_dir}），略過")
        return _empty_std(), info

    rows: List[Dict[str, Any]] = []
    for path in files:
        name = os.path.basename(path)
        df, enc = _read_csv_any(path)
        if df is None or df.empty:
            if verbose:
                print(f"  165 開放資料 {name}：無法讀取或為空（{enc}），略過")
            info["files"].append({"file": name, "error": enc})
            continue
        url_col = detect_url_column(df)
        if url_col is None:
            if verbose:
                print(f"  165 開放資料 {name}：找不到網址欄位（欄位：{list(df.columns)[:8]}），略過")
            info["files"].append({"file": name, "error": "no_url_column"})
            continue
        name_col = _pick_column(df, NAME_HEADER_HINTS, exclude=url_col)
        date_col = _pick_column(df, DATE_HEADER_HINTS, exclude=url_col)
        kept = fixed = invalid = 0
        for _, rec in df.iterrows():
            cell = str(rec.get(url_col, "") or "")
            for part in _SPLIT_CELL_RE.split(cell.strip()):
                if not part:
                    continue
                key, status = sanitize_opendata_url(part)
                if not key:
                    invalid += 1
                    continue
                fixed += status == "fixed_tld"
                kept += 1
                rows.append({
                    "url": key,
                    "raw_url": normalize_url(part),
                    "label": 1,
                    "source": f"165_opendata:{name}",
                    "weight": float(weight),
                    "time": str(rec.get(date_col, "") or "") if date_col else "",
                    "note": str(rec.get(name_col, "") or "") if name_col else "",
                })
        info["files"].append({"file": name, "encoding": enc, "url_column": url_col,
                              "kept": kept, "fixed_tld": fixed, "invalid": invalid})
        info["fixed_tld"] += fixed
        info["invalid"] += invalid
        if verbose:
            print(f"  165 開放資料 {name}：編碼 {enc}、網址欄位「{url_col}」→ 有效 {kept} 筆"
                  f"（修正 TLD {fixed}、剔除 {invalid}）")

    out = pd.DataFrame(rows, columns=STD_COLUMNS)
    info["rows"] = int(len(out))
    if out.empty:
        return _empty_std(), info
    out = out.drop_duplicates(subset=["url"], keep="first")
    if max_rows and len(out) > max_rows:
        out = out.sort_values("url")
        step = len(out) / float(max_rows)
        out = out.iloc[[int(i * step) for i in range(max_rows)]]
        if verbose:
            print(f"  165 開放資料共 {info['rows']} 筆，依 max_rows 等距取樣 {max_rows} 筆")
    info["kept"] = int(len(out))
    return out.reset_index(drop=True), info


# =============================================================================
# 合併
# =============================================================================

def merge_sources(
    main_df: pd.DataFrame,
    feedback_df: pd.DataFrame,
    opendata_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    合併三種來源（url 已是 dedupe_key）：
      feedback（最新人工判斷）> 主資料 > 165 開放資料。
    回傳 (合併後 DataFrame, 衝突統計)。
    """
    report: Dict[str, Any] = {"feedback_override": [], "feedback_new": 0,
                              "opendata_new": 0, "opendata_conflicts": []}
    merged = main_df.copy()
    if feedback_df is not None and not feedback_df.empty:
        idx = {k: i for i, k in enumerate(merged["url"].tolist())}
        new_rows = []
        for _, fb in feedback_df.iterrows():
            i = idx.get(fb["url"])
            if i is None:
                new_rows.append(fb.to_dict())
                continue
            old_label = int(merged.at[i, "label"])
            if old_label != int(fb["label"]):
                report["feedback_override"].append({"url": fb["url"], "dataset": old_label,
                                                    "feedback": int(fb["label"])})
            for col in ("label", "source", "weight", "time", "note"):
                merged.at[i, col] = fb[col]
        report["feedback_new"] = len(new_rows)
        if new_rows:
            merged = pd.concat([merged, pd.DataFrame(new_rows, columns=STD_COLUMNS)], ignore_index=True)

    if opendata_df is not None and not opendata_df.empty:
        known = dict(zip(merged["url"], merged["label"]))
        fresh = []
        for _, od in opendata_df.iterrows():
            if od["url"] in known:
                if int(known[od["url"]]) != 1:
                    report["opendata_conflicts"].append(od["url"])
                continue
            fresh.append(od.to_dict())
        report["opendata_new"] = len(fresh)
        if fresh:
            merged = pd.concat([merged, pd.DataFrame(fresh, columns=STD_COLUMNS)], ignore_index=True)

    merged["label"] = merged["label"].astype(int)
    merged["weight"] = merged["weight"].astype(float)
    merged["registered_domain"] = merged["url"].map(get_registered_domain)
    return merged.reset_index(drop=True), report


if __name__ == "__main__":  # 簡易自我檢查：列出偵測到的 165 開放資料
    frame, meta = load_165_opendata()
    print(meta)
    print(frame.head())
