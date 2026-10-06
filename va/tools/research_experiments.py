# -*- coding: utf-8 -*-
# =============================================================================
# tools/research_experiments.py — 論文用補充實驗（v7.2）
# =============================================================================
# 目的：補齊論文第四、五章需要、但 train_model.py 主流程沒有輸出的研究證據。
# 不會改動 scam_model.pkl／model_meta.json／model_report.json（只讀、只輸出到 docs/experiments/）。
#
#   threshold  門檻分析：以「巢狀 group-CV 的 sigmoid 校準 OOF 機率」掃描 0.10～0.90，
#              輸出 precision／recall／F1／F2／FPR（真實樣本，目標詐騙比例 20% 校準）
#   ablation   特徵群消融：同一組 GradientBoosting 超參數、同一個 StratifiedGroupKFold 切分，
#              每次移除一個特徵群後重新做 OOF，比較 PR-AUC／ROC-AUC／F1@0.40／FPR@0.40／hard-negative AUC
#   models     候選模型比較：直接讀 model_report.json 的 OOF 結果，另外實測部署模型的推論時間
#   errors     誤差分析與標註審查候選：讀 model_report.json 的 url_cases／label_review，
#              輸出 error_analysis.md 與 label_audit_candidates.csv（需人工逐筆確認標籤）
#
# 執行（在 va 目錄下）：
#   python tools/research_experiments.py                       # 全部
#   python tools/research_experiments.py --only threshold,ablation
#   python tools/research_experiments.py --out ../docs/experiments
# 結果受 sklearn 版本與資料影響；單次 OOF 的差異小於約 0.005（對照 model_report.json 的
# feature_experiments 多種子標準差 0.002～0.003）不應解讀為特徵有／無貢獻。
# =============================================================================

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import warnings
from typing import Any, Dict, List, Sequence

import numpy as np

VA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if VA_DIR not in sys.path:
    sys.path.insert(0, VA_DIR)

import train_model as T  # noqa: E402
from features import FEATURE_NAMES, extract_features  # noqa: E402

REPORT_PATH = os.path.join(VA_DIR, "model_report.json")
DEFAULT_OUT = os.path.join(os.path.dirname(VA_DIR), "docs", "experiments")

THRESHOLDS = [round(t, 2) for t in np.arange(0.10, 0.91, 0.05)]

FEATURE_GROUPS: Dict[str, List[str]] = {
    "品牌相關（品牌字詞／仿冒／編輯距離）": [
        "brand_in_sld", "brand_typo_like", "levenshtein_brand_dist", "brand_impersonation"],
    "TLD 風險": ["suspicious_tld", "tld_risk_level"],
    "詐騙話術關鍵字（博弈／投資／加密）": [
        "has_scam_word", "gambling_number_pattern", "gambling_keyword", "investment_lure_keyword",
        "crypto_exchange_lure", "suspicious_keyword_in_domain"],
    "網域隨機性與新註冊代理": ["sld_randomness", "newly_registered_like", "domain_entropy"],
    "架站平台／暫時性主機": [
        "cloud_hosting", "free_hosting_platform", "tunnel_or_ephemeral_host", "social_invite_link"],
    "網址混淆（punycode／@／埠號／編碼／IP）": [
        "punycode_domain", "url_has_at_symbol", "non_standard_port", "contains_percent_encoding",
        "is_ip_address", "double_http", "has_shortener"],
    "長度與結構統計": [
        "url_length", "domain_length", "path_length", "query_length", "digit_ratio", "hyphen_count",
        "dot_count", "special_chars", "subdomain_depth", "path_depth", "query_params", "long_domain",
        "many_subdomains", "sld_digit_count", "sld_length"],
    "廣告追蹤與行動誘導路徑": ["has_utm", "has_gclid", "mobile_lure_path", "path_scam_route", "is_https"],
}


def _log(msg: str) -> None:
    print(msg, flush=True)


def _write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def load_report() -> Dict[str, Any]:
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare():
    """與 train_model.main() 相同的資料準備流程。"""
    warnings.filterwarnings("ignore")
    url_cases = T.load_url_cases()
    real, _info = T.load_dataset(T.OPENDATA_DIR, 4000)
    frame, _stats = T.build_training_frame(real, url_cases)
    X = T.build_feature_matrix(frame["url"].tolist())
    y = frame["label"].to_numpy(dtype=int)
    w = frame["weight"].to_numpy(dtype=float)
    groups = frame["group"].to_numpy()
    is_real = (frame["is_synthetic"] == 0).to_numpy()
    is_template = ((frame["is_synthetic"] == 1) & ~frame["category"].str.startswith("real")).to_numpy()
    splits = T.make_group_splits(y, groups, is_real, real_only_test=False)
    return frame, X, y, w, groups, is_real, is_template, splits


# ---------------------------------------------------------------------------
# 1. 門檻分析
# ---------------------------------------------------------------------------

def run_threshold(ctx, params, out_dir: str) -> Dict[str, Any]:
    frame, X, y, w, groups, is_real, is_template, splits = ctx
    t0 = time.time()
    calib = T.nested_calibration("GradientBoosting", params, X, y, w, groups, is_real, splits, is_real,
                                 is_template, T.TARGET_PRIOR, methods=("sigmoid",), verbose=False)
    oof = calib["sigmoid"]["oof_proba"]
    yr, pr = y[is_real], oof[is_real]
    a = T.TARGET_PRIOR / float(np.mean(yr))
    b = (1 - T.TARGET_PRIOR) / (1 - float(np.mean(yr)))
    rows = []
    for thr in THRESHOLDS:
        m = T.threshold_metrics(yr, pr, thr)
        (tn, fp), (fn, tp) = m["confusion"]
        denom = tp * a + fp * b
        m["precision_target_prior"] = round(tp * a / denom, 4) if denom else 0.0
        rows.append(m)
    result = {"method": "巢狀 StratifiedGroupKFold(5) + sigmoid 校準，target_prior=0.20，只計真實樣本",
              "n_real": int(len(yr)), "real_scam": int(yr.sum()), "rows": rows,
              "seconds": round(time.time() - t0, 1)}
    _write_json(os.path.join(out_dir, "threshold_analysis.json"), result)
    return result


# ---------------------------------------------------------------------------
# 2. 特徵群消融
# ---------------------------------------------------------------------------

def _eval_columns(ctx, params, keep: List[int]) -> Dict[str, Any]:
    frame, X, y, w, groups, is_real, is_template, splits = ctx
    oof, _folds = T.cross_val_oof("GradientBoosting", params, X[:, keep], y, w, splits, is_real)
    yr, pr = y[is_real], oof[is_real]
    m = T.prob_metrics(yr, pr)
    tm = T.template_metrics(y, oof, is_template, is_real)
    return {"pr_auc": m["pr_auc"], "roc_auc": m["roc_auc"], "f1_0.40": m["at_0.40"]["f1"],
            "recall_0.40": m["at_0.40"]["recall"], "fpr_0.40": m["at_0.40"]["fpr"],
            "hard_negative_auc": tm.get("hard_negative_auc")}


def run_ablation(ctx, params, out_dir: str) -> Dict[str, Any]:
    t0 = time.time()
    index = {n: i for i, n in enumerate(FEATURE_NAMES)}
    covered = {n for names in FEATURE_GROUPS.values() for n in names}
    unknown = covered - set(index)
    if unknown:
        raise RuntimeError(f"FEATURE_GROUPS 含不存在的特徵：{sorted(unknown)}")
    leftovers = [n for n in FEATURE_NAMES if n not in covered]
    groups = dict(FEATURE_GROUPS)
    if leftovers:
        groups["其他"] = leftovers
    all_cols = list(range(len(FEATURE_NAMES)))

    rows = [{"setting": f"完整 {len(FEATURE_NAMES)} 維", "removed": [], **_eval_columns(ctx, params, all_cols)}]
    _log(f"  [ablation] 完整：{rows[0]}")
    experiments = [(f"移除：{name}", names) for name, names in groups.items()]
    experiments.append(("只移除 sld_randomness（v7.1 新增）", ["sld_randomness"]))
    for label, names in experiments:
        drop = {index[n] for n in names}
        keep = [i for i in all_cols if i not in drop]
        res = _eval_columns(ctx, params, keep)
        rows.append({"setting": label, "removed": names, **res})
        _log(f"  [ablation] {label}：PR-AUC {res['pr_auc']}  ROC-AUC {res['roc_auc']}")
    base = rows[0]
    for r in rows[1:]:
        r["delta_pr_auc"] = round(r["pr_auc"] - base["pr_auc"], 4)
        r["delta_roc_auc"] = round(r["roc_auc"] - base["roc_auc"], 4)
    result = {"method": "GradientBoosting（部署超參數，未校準）× 同一 StratifiedGroupKFold 切分；只計真實樣本 OOF",
              "note": "單次 OOF；差異小於約 0.005 屬雜訊範圍（model_report.json 多種子標準差 0.002～0.003）",
              "rows": rows, "seconds": round(time.time() - t0, 1)}
    _write_json(os.path.join(out_dir, "ablation.json"), result)
    return result


# ---------------------------------------------------------------------------
# 3. 候選模型比較
# ---------------------------------------------------------------------------

def run_models(report: Dict[str, Any], out_dir: str) -> Dict[str, Any]:
    import joblib

    best = report["cross_validation"]["best_per_model"]
    rows = []
    for name, entry in best.items():
        oof = entry["oof"]
        rows.append({"model": name, "params": entry["params"], "pr_auc": oof["pr_auc"],
                     "roc_auc": oof["roc_auc"], "brier": oof["brier"],
                     "f1_0.40": oof["at_0.40"]["f1"], "recall_0.40": oof["at_0.40"]["recall"],
                     "fpr_0.40": oof["at_0.40"]["fpr"]})
    # 部署模型實測推論時間（特徵擷取＋predict_proba，單筆平均）
    bundle = joblib.load(os.path.join(VA_DIR, "scam_model.pkl"))
    estimator = bundle["estimator"] if isinstance(bundle, dict) else bundle
    import pandas as pd
    urls = pd.read_csv(os.path.join(VA_DIR, "data_clean.csv"))["URL"].astype(str).tolist()[:500]
    t0 = time.perf_counter()
    X = np.array([extract_features(u) for u in urls], dtype=float)
    t1 = time.perf_counter()
    estimator.predict_proba(X)
    t2 = time.perf_counter()
    timing = {"n_urls": len(urls),
              "feature_extraction_ms_per_url": round((t1 - t0) * 1000 / len(urls), 3),
              "predict_proba_ms_per_url_batch": round((t2 - t1) * 1000 / len(urls), 3)}
    result = {"rows": rows, "selected": report["cross_validation"]["selection"]["chosen"],
              "deployed_inference_timing": timing}
    _write_json(os.path.join(out_dir, "model_comparison.json"), result)
    return result


# ---------------------------------------------------------------------------
# 4. 誤差分析與標註審查
# ---------------------------------------------------------------------------

def _short(url: str, n: int = 70) -> str:
    return url if len(url) <= n else url[: n - 1] + "…"


def run_errors(report: Dict[str, Any], out_dir: str) -> Dict[str, Any]:
    cases = report.get("url_cases", {})
    review = report.get("label_review", {})
    fps = cases.get("false_positives", [])
    fns = cases.get("false_negatives", [])
    hcm = cases.get("high_confidence_misses", [])
    benign_hi = review.get("benign_with_high_proba", [])
    scam_lo = review.get("scam_with_low_proba", [])

    lines = ["# 誤差分析（Error Analysis）", "",
             "資料來源：`va/model_report.json`（OOF 機率；硬規則與白名單未計入，最終分數以 API 為準）。", "",
             "## 回歸清單（tests/fixtures/url_cases.json）模型層級誤判", ""]
    lines += [f"- 誤判（合法被判詐騙）@0.40：{len(fps)} 筆；漏判（詐騙被判正常）@0.40：{len(fns)} 筆；"
              f"高信心漏判 @0.70：{len(hcm)} 筆。", ""]

    def case_rows(items):
        return [[it.get("category", ""), f"`{_short(str(it.get('url', '')))}`",
                 it.get("model_proba", it.get("proba", it.get("oof_proba", "")))] for it in items]

    lines += ["### False Positive（合法被判詐騙）", "", _md_table(["類別", "網址", "機率"], case_rows(fps)), "",
              "### False Negative（詐騙被判正常）", "", _md_table(["類別", "網址", "機率"], case_rows(fns)), "",
              "### 高信心漏判（@0.70）", "", _md_table(["類別", "網址", "機率"], case_rows(hcm)), "",
              "## 標註審查候選（需人工確認標籤是否正確）", "",
              "### 標為正常、但 OOF 機率偏高", "",
              _md_table(["網址", "OOF 機率"], [[f"`{_short(str(i['url']))}`", i.get('oof_proba', '')] for i in benign_hi]), "",
              "### 標為詐騙、但 OOF 機率偏低", "",
              _md_table(["網址", "OOF 機率"], [[f"`{_short(str(i['url']))}`", i.get('oof_proba', '')] for i in scam_lo]), ""]
    with open(os.path.join(out_dir, "error_analysis.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    path = os.path.join(out_dir, "label_audit_candidates.csv")
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["url", "current_label", "oof_proba", "reviewer_label", "reviewer_note"])
        for i in benign_hi:
            writer.writerow([i["url"], 0, i.get("oof_proba", ""), "", ""])
        for i in scam_lo:
            writer.writerow([i["url"], 1, i.get("oof_proba", ""), "", ""])
    result = {"false_positives": len(fps), "false_negatives": len(fns), "high_confidence_misses": len(hcm),
              "label_audit_candidates": len(benign_hi) + len(scam_lo)}
    _write_json(os.path.join(out_dir, "error_summary.json"), result)
    return result


# ---------------------------------------------------------------------------
# 報告彙整
# ---------------------------------------------------------------------------

def write_markdown(out_dir: str, results: Dict[str, Any]) -> None:
    parts = ["# 補充實驗結果（tools/research_experiments.py）", "",
             "所有數字來自本專案資料與切分（StratifiedGroupKFold 5 折，只計真實樣本 OOF），"
             "可用 `python tools/research_experiments.py` 重現。", ""]
    if "threshold" in results:
        rows = [[r["threshold"], r["precision"], r["recall"], r["f1"], r["f2"], r["fpr"],
                 r["precision_target_prior"]] for r in results["threshold"]["rows"]]
        parts += ["## 門檻分析", "", results["threshold"]["method"], "",
                  _md_table(["門檻", "Precision", "Recall", "F1", "F2", "FPR", "Precision（20% 先驗）"], rows), ""]
    if "ablation" in results:
        rows = [[r["setting"], r["pr_auc"], r["roc_auc"], r["f1_0.40"], r["fpr_0.40"], r["hard_negative_auc"],
                 r.get("delta_pr_auc", "—")] for r in results["ablation"]["rows"]]
        parts += ["## 特徵群消融", "", results["ablation"]["note"], "",
                  _md_table(["設定", "PR-AUC", "ROC-AUC", "F1@0.40", "FPR@0.40", "hard-neg AUC", "ΔPR-AUC"], rows), ""]
    if "models" in results:
        rows = [[r["model"], r["pr_auc"], r["roc_auc"], r["brier"], r["f1_0.40"], r["recall_0.40"], r["fpr_0.40"]]
                for r in results["models"]["rows"]]
        t = results["models"]["deployed_inference_timing"]
        parts += ["## 候選模型比較", "", _md_table(["模型", "PR-AUC", "ROC-AUC", "Brier", "F1@0.40", "Recall@0.40",
                                                 "FPR@0.40"], rows), "",
                  f"部署模型推論時間：特徵擷取約 {t['feature_extraction_ms_per_url']} ms／筆，"
                  f"predict_proba 約 {t['predict_proba_ms_per_url_batch']} ms／筆（批次 {t['n_urls']} 筆）。", ""]
    with open(os.path.join(out_dir, "experiments.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TruthMark 論文補充實驗")
    parser.add_argument("--only", default="threshold,ablation,models,errors")
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    wanted = {s.strip() for s in args.only.split(",") if s.strip()}
    os.makedirs(args.out, exist_ok=True)

    report = load_report()
    params = report["best"]["params"]
    results: Dict[str, Any] = {}

    if wanted & {"threshold", "ablation"}:
        _log("準備資料與特徵矩陣…")
        ctx = prepare()
        if "threshold" in wanted:
            _log("門檻分析（巢狀校準 OOF）…")
            results["threshold"] = run_threshold(ctx, params, args.out)
        if "ablation" in wanted:
            _log("特徵群消融…")
            results["ablation"] = run_ablation(ctx, params, args.out)
    if "models" in wanted:
        results["models"] = run_models(report, args.out)
    if "errors" in wanted:
        results["errors"] = run_errors(report, args.out)

    write_markdown(args.out, results)
    _log(f"完成，輸出於 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
