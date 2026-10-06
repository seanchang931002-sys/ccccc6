# TruthMark v7.2 Final Release

## 版本定位

本版本是畢業專題程式交付版，以 v7.2 安全／驗證能力與 165 涉詐資料同步整合為主，並清理本機執行殘留檔。

## 本版收尾重點

1. 46 維特徵與模型 artifact 對齊，啟動時驗證 feature schema。
2. `/predict`、`/predict/batch`、`/feedback`、`/blocklist`、`/reload-model`、`/audit-log` 與 165 同步端點完成安全控管。
3. Chrome MV3 擴充套件、CSP、URL 去識別化與 batch API 對齊。
4. 165 CSV 同步採最少筆數、無法解析比例、共用主機／公共後綴與只新增不刪除策略。
5. Public Suffix 缺失時使用「有限且明確標示為降級」的離線 fallback；正式環境仍以 tldextract + 離線 PSL 為準。
6. 交付包不含 `.env`、`.pyc`、cache、audit log 與其他本機暫存檔。

7. `TRUTHMARK_STATE_DIR` 會集中保存 dynamic blocklist、domain reports、feedback 與 165 同步狀態；第一次使用獨立狀態目錄時會自動帶入專案附帶的初始黑名單快照。
