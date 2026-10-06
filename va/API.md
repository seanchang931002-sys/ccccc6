# TruthMark API 參考（v7.2）

Base URL（開發預設）：`http://127.0.0.1:5500`

所有回應皆為 JSON，成功時至少含 `ok: true`；失敗時 `ok: false` 並附
`error`／`message`。本文件按實際路由定義（`main.py`）整理，數字與欄位
以程式碼為準，若與論文敘述有出入請以此文件和程式碼為準。

## 公開端點

### `GET /health`
系統狀態檢查：模型是否載入、特徵版本、規則版本、黑名單筆數、快取統計、
`tldextract` 是否降級（v7.2 新增）。

### `GET /features`
回傳目前使用的 46 維特徵名稱與版本資訊。

### `GET /model-report`
回傳 `model_report.json` 全文（訓練與評估結果，含本文件其他地方引用的
OOF 指標、label_review 等）。

### `POST /predict`
單一網址風險判斷。

```json
// Request
{ "url": "https://example.com/path", "debug": false, "compact": false, "deep_scan": false }

// Response（compact=false，預設）
{
  "ok": true,
  "url": "https://example.com/path",
  "risk_score": 12,
  "risk_level": "low",
  "risk_label": "低風險",
  "verdict": "...",
  "source": "ai_model",
  "triggered_rules": [],
  "reasons": [...],
  "degraded": false
}

// Response（compact=true，供批次掃描使用，欄位精簡）
{ "ok": true, "risk_score": 12, "risk_level": "low", "confidence": 0.88 }
```

v7.2 起有限流：120 次／分鐘（依來源 IP 或已信任的 X-Forwarded-For，
見 `docs/PERFORMANCE.md` 與 `.env.example` 的 Trusted Proxy 設定）。

### `POST /predict/batch`
批次網址風險判斷，單次最多 50 筆。

```json
// Request
{ "urls": ["https://a.com", "https://b.com"], "debug": false, "compact": true }

// Response
{ "ok": true, "count": 2, "results": { "https://a.com": {...}, "https://b.com": {...} } }
```

單筆失敗不影響其他筆（個別結果內會是 `ok:false`）。v7.2 起限流：
30 次／分鐘（比單筆 `/predict` 更嚴格，因為單次請求運算量更大）。

### `POST /feedback`
使用者回報（詐騙／正常），立即觸發背景寫入與資料清理。

```json
{ "url": "https://example.com", "label": 1, "ai_score": 0.82, "note": "" }
```

`label`：`1`＝詐騙，`0`＝正常。限流：20 次／分鐘。

### `GET /blocklist`
回傳目前完整黑名單（內建＋動態新增）。

### `GET /blocklist/reports`
回傳尚未達到自動封鎖門檻的網域回報統計。

### `POST /blocklist/add`
手動將網域加入黑名單候選回報；達到門檻後立即生效並清除相關快取。限流：
10 次／分鐘。

```json
{ "domain": "evil.example.com", "note": "使用者回報" }
```

### `GET /cache/stats`
`prediction_cache` 的容量、命中率統計。

## 管理端點（需 `X-Admin-Key` 標頭，未設定 `TRUTHMARK_ADMIN_KEY` 時僅限本機）

### `POST /reload-model`
重新載入 `scam_model.pkl`，並清除全部預測快取。會寫入稽核紀錄
（`model.reload`）。

### `DELETE /blocklist/{domain}`
從動態黑名單移除指定網域（內建／政府清單網域無法移除）。會寫入稽核
紀錄（`blocklist.remove`）。

### `GET /blocklist/sync-165`（管理端點）
165 自動匯入的設定與上次結果：`configured`、`urls`、`interval_hours`、`min_rows`、
`last_success_at`、`last_result`（rows／candidates／added／already／protected／
skipped_shared／invalid）、`last_error`、`total_blocked`。

### `POST /blocklist/sync-165`（管理端點）
立即從 `TRUTHMARK_165_CSV_URL` 同步涉詐網域。下載網址只讀環境變數，不接受請求指定。
有新增網域時清空預測快取；寫入稽核紀錄（`blocklist.sync_165`）。

```json
// Request（可省略 body）
{ "dry_run": false, "force": false }   // dry_run：只試算；force：忽略 ETag 強制重新下載

// Response
{ "ok": true, "message": "165 同步完成：新增 120、已存在 13207",
  "report": { "rows": 13327, "candidates": 13325, "added": 120, "already": 13207,
              "protected": 0, "skipped_shared": 2, "invalid": 0, "not_modified": 0,
              "cache_cleared": 37, "added_sample": ["..."], "duration_s": 1.1 },
  "total_blocked": 13480 }
```

失敗（未設定網址、連線失敗、來源筆數過少／格式改變）回 `ok:false`、
`error_code: "sync_165_failed"`，並附完整 `report`；失敗不會改動現有黑名單。

### `GET /audit-log?limit=50`
查詢最近的管理動作稽核紀錄（v7.2 新增），回傳新到舊排序的 JSON 陣列，
每筆含 `time`／`actor`（回報者指紋，非原始 IP）／`action`／`target`／
`success`／`detail`。

## 錯誤回應格式

```json
{ "ok": false, "error": "給使用者看的說明", "message": "同 error（相容舊前端）",
  "error_code": "http_429" }
```

`error` 與 `message` 內容相同（前端有的讀 error、有的讀 message）；機器可讀
的代碼在 `error_code`，例如 `validation_error`（HTTP 422）、`http_404`、
`http_429`（限流，另帶 `Retry-After` 標頭）、`admin_key_invalid`、
`internal_error`（未預期例外：HTTP 200 + `ok:false`，不讓前端收到 500）。
