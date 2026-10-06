# TruthMark Architecture

## 系統流程

Chrome MV3 擴充套件 → FastAPI → URL 正規化 → 白名單／黑名單 → 硬規則 → 46 維特徵 → Gradient Boosting → Risk Fusion → 使用者回報／動態黑名單。

深度掃描為選用路徑：僅在使用者要求時抓取頁面文字，並執行 SSRF 防護、大小限制、重新導向逐跳檢查與內容訊號分析。

## 四層主要判斷

1. Trusted / Blocklist：優先處理可信任與已知惡意網域。
2. Hard Rules：針對 URL 結構、網域、跳轉、品牌冒用等明確訊號。
3. ML：以 46 維特徵輸入 Gradient Boosting。
4. Gray-zone / Fusion：整合多來源分數並避免同一訊號重複加分。

## 可靠性設計

- 模型 feature schema mismatch → rules-only degraded mode
- 推論例外 → 不回 500，改走降級策略
- 動態檔案寫入 → file lock + atomic write
- 預測快取 → URL canonical key + TTL
- 管理操作 → Admin Key + Audit Log
