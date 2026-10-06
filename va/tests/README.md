# TruthMark / VeriAd 自動化測試

目前共有 233 項自動化測試，使用 Python 標準庫 `unittest`（不需安裝 pytest），不需要啟動伺服器、不連網。

## 執行

在 `va` 目錄下：

```powershell
cd va
python -m unittest discover -s tests -v
```

只跑單一檔案或單一類別：

```powershell
python -m unittest discover -s tests -p "test_alignment.py" -v
python -m unittest discover -s tests -p "test_api.py" -k RegressionTests -v
```

Windows 主控台若出現中文亂碼，先設定 `$env:PYTHONIOENCODING="utf-8"`。

## 檔案說明

| 檔案 | 內容 |
| --- | --- |
| `_support.py` | 共用工具：把 `va/` 加進 `sys.path`、讀回歸清單、以 ast 解析備份 `features.py` 的 FEATURE_NAMES、真實資料檔快照／還原。檔名以底線開頭，不會被當成測試收集。 |
| `test_alignment.py` | 跨模組對齊：FEATURE_NAMES 46 個（31 舊＋14 v7.0＋1 v7.1）且前 31 個與備份相同、FEATURE_SCHEMA_ID、FEATURE_SPECS / assess_feature 涵蓋全部特徵、model_meta.json 與 scam_model.pkl（bundle）的 feature_names、script.js FEATURE_MAP 的 key 與圖示、index.html 內嵌圖示區塊與 `外掛/icons.js` 逐字一致、HARD_RULES 可編譯且 target 合法、`build_ensemble_vector` 不寫死 31、外掛改用 `/predict/batch`。 |
| `test_features.py` | 特徵擷取：回歸清單全部網址不丟例外、惡意輸入（空字串、`http://[abc`、5000 字元、零寬／全形、IP、port、@）、scheme 不變性（裸網域 = https://；http 只差 is_https）、token 化誤判回歸（learn / alphabet / better / online / 1688 / shopee.sg / poyabuy / cosmed）、v7 新特徵正例與反例、deep_scan SSRF 防護。 |
| `test_features_v7.py` | features.py / rules_config.py v7 階段的回歸測試（canonicalize、PSL、語意詞庫、硬規則 hard negative）。 |
| `test_features_v71.py` | v7.1：`sld_randomness`（範圍、可讀網域低／隨機字串高、統計表讀不到時退回啟發式）、`newly_registered_like` 不再誤判 threads／flickr／kktix 等合法品牌、dot_count／digit_ratio／special_chars 只算主機名稱、黏字切分（biaoguvip、188bettw、kucoinexchangevip）與反例、`char_ngram_table.json` 格式。 |
| `test_api.py` | 以 FastAPI `TestClient` 測 `/health`、`/features`、`/predict`（一般 / debug / compact / 錯誤格式 / 2049 字元 / 非法 IPv6）、`/predict/batch`、`/feedback` 與 `/blocklist/add` 的快取失效、模型缺檔／損毀／特徵不一致時的降級，以及 `fixtures/url_cases.json` 端到端回歸。 |
| `test_backend_v7.py` | 後端 v7 融合與高可用：內嵌跳轉目標評分、白名單覆寫（LINE 群組、sites.google、Google 表單）、黑名單父網域、可信任網域不被回報自動封鎖、多規則加成上限、灰色地帶補強、官方網域與內容平台 AI 上限、輸入驗證、deep_scan SSRF、`cache.py`、`blocklist_store.py`。 |
| `test_training_pipeline.py` | 訓練流程：canonical 去重、標籤衝突排除、feedback 取最新、165 開放資料自動偵測、擴增形狀平衡、驗收清單防洩漏、group 不跨折、模型產物（bundle / model_meta / model_report）內容。 |
| `test_security.py` | v7.1 濫用防範（論文表 3-1）：RateLimiter 滑動視窗（超過回 HTTP 429＋Retry-After、換 IP 不受影響）、回報者指紋（加鹽雜湊、同人重複回報不加次數、只存雜湊）、X-Admin-Key 保護 `/reload-model` 與 `DELETE /blocklist/{domain}`（未設金鑰時僅限本機）、CORS 白名單、BlocklistStore 的 remove／併發寫入不遺失。 |
| `test_sync_165.py` | 165 涉詐網域自動匯入：CSV 解析（UTF-8／Big5、兩種資料集欄位、防點擊寫法）、共用主機／公共後綴／可信任網域過濾、筆數過少與格式異常中止且不寫入、dry-run、只增不刪、冪等、ETag／304、下載大小上限與僅限 http(s)、同步互斥、`POST/GET /blocklist/sync-165`（管理金鑰、快取失效、稽核）、啟動後背景排程與重啟還原。全部離線。 |
| `fixtures/url_cases.json` | 共用回歸清單（由資料負責人維護，測試只讀）。 |

## 端到端回歸門檻（不得為了通過而放寬）

`test_api.RegressionTests` 對 `fixtures/url_cases.json` 每筆網址呼叫 `/predict`：

- benign 誤判率（最終分數 ≥ 40 或 ok:false）≤ **3%**
- scam 召回率（最終分數 ≥ 40）≥ **90%**
- 有 `min_score` 欄位的案例，分數達到 `min_score` 的比例 ≥ **85%**

失敗訊息會列出每一筆誤判／漏報（類別、網址、分數、來源、命中規則）。執行時 stderr 也會印出
「回歸摘要」與各類別失敗數。設定環境變數 `TRUTHMARK_REGRESSION_REPORT=<路徑>` 可把逐筆結果另存成 JSON：

```powershell
$env:TRUTHMARK_REGRESSION_REPORT="$env:TEMP/truthmark_regression.json"
python -m unittest discover -s tests -p "test_api.py" -k RegressionTests -v
```

## 不會改動真實資料

- `feedback.csv`、`dynamic_blocklist.json`、`domain_reports.json`：`test_api` 在模組開始時把 `main`
  模組與其內物件（例如 `blocklist_store`、`model_state`）中指向這些檔案的路徑改指到暫存目錄；
  另外先快照真實檔案位元組，結束時比對並還原（雙重保險，若真的被寫到會讓測試失敗並提示）。
- `scam_model.pkl`、`model_meta.json`：只讀。降級／bundle 測試在暫存目錄產生假模型（sklearn
  `DummyClassifier`），暫時把 `main` 的模型路徑指過去再呼叫 `/reload-model`，結束後改回並重新載入真實模型。

## 預期的 skip

- 模型尚未以 v7 特徵重訓（`model_meta.json` 沒有 v7 `feature_names`、或 `scam_model.pkl` 仍是舊格式
  純 estimator）時，`test_alignment.ModelArtifactTests` 會 skip 並註明原因。
- 未設定 `TRUTHMARK_BACKUP_DIR`（或找不到備份目錄）時，與備份比對的測試會 skip；需要時用該環境變數指定備份位置。
