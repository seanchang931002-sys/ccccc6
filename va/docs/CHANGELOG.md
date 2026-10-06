# TruthMark 版本變更摘要

## v7.2 Final

- 保留 v7.1 的 46 維特徵、模型 schema 與詐騙風險融合策略。
- 強化 API 安全：CORS 白名單、Rate Limit、Admin Key、回報者 fingerprint、Audit Log、Trusted Proxy CIDR。
- Chrome MV3：CSP、batch API、URL 去識別化與前後端合約對齊。
- 深度掃描：SSRF 防護、DNS/IP 驗證、逐跳 redirect 檢查、response size limit。
- 新增 165 涉詐網域 CSV 同步：ETag、最小列數、格式異常中止、共用主機／公共後綴保護、只新增不刪除。
- 執行期狀態可透過 `TRUTHMARK_STATE_DIR` 轉到持久化 volume。
- 交付包移除 `.env`、cache、`.pyc`、audit log 等本機暫存物。
- Public Suffix 缺少 `tldextract` 時改用有限、明確的離線 fallback；`/health` 會標示 degraded。

## v7.1 Feature Baseline

- 46 維特徵：31 個既有特徵 + 14 個 v7.0 特徵 + `sld_randomness`。
- 重新以 group-CV / OOF 驗證模型，並固定 feature schema `3563ef6db762`。
