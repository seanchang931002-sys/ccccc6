# Threat Model

| 威脅 | 防護 | 殘餘風險 |
|---|---|---|
| API 大量請求 | Rate Limit | 多來源分散流量仍可能增加成本 |
| 偽造管理操作 | Admin Key + loopback fallback | 金鑰洩漏仍需輪替 |
| 偽造 Proxy IP | Trusted Proxy CIDR | Proxy 設定錯誤仍有風險 |
| SSRF | DNS/IP 檢查、禁止私有位址、逐跳 redirect 驗證、IP pinning | DNS／雲端環境仍需持續測試 |
| 重複回報灌水 | salted reporter fingerprint + threshold | 更換來源 IP 可繞過單一來源限制 |
| 檔案競爭 | filelock + atomic write | 高併發產品環境仍應考慮 DB |
| 模型不一致 | feature schema 驗證 + degraded fallback | 需重新訓練模型才能恢復 AI |
| 瀏覽網址隱私 | 前端移除 query/hash 後送出 | Path 本身仍可能包含識別資訊 |
