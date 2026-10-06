# TruthMark v7.2 — Markdown 數值更新紀錄

本包中的數值型 Markdown 文件已依目前 `model_report.json` / `model_meta.json` 更新。

## 目前模型主要指標

- 模型：GradientBoosting
- 特徵數：46
- Schema：3563ef6db762
- 真實樣本：1,050（正常 552、詐騙 498）
- 合成樣本：4,019
- 總訓練資料：5,069
- PR-AUC：0.9363
- ROC-AUC：0.9363
- Brier：0.1263
- Log Loss：0.3954
- 0.40：Precision 0.9284、Recall 0.7028、F1 0.8000、F2 0.7387、FPR 0.0489、Accuracy 0.8333
- 0.70：Precision 0.9928、Recall 0.5502、F1 0.7080、F2 0.6041、FPR 0.0036、Accuracy 0.7848
- 0.05（最佳 F2，非產品門檻）：F2 0.8892、Precision 0.6671、Recall 0.9699

## 驗證

- Python unittest：233 tests，233 passed，1 skipped
- JavaScript `node --check`：5 個檔案全部通過
- 模型特徵／schema 一致性：通過

## 包含檔案

- `README.md`
- `va/README.md`
- `va/docs/ERROR_ANALYSIS.md`
- `va/docs/THRESHOLD_RATIONALE.md`
- `va/docs/LABEL_REVIEW.md`
- `va/docs/PERFORMANCE.md`
- `va/docs/FINAL_TEST_REPORT.md`
- `va/docs/MD_UPDATE_NOTES.md`
