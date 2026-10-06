# TruthMark — 自動化廣告與網站可信度審查平台（v7.2）

以白名單／黑名單（含政府 165 開放資料可選匯入）／硬規則／AI 模型（Gradient
Boosting）四層防禦為核心的詐騙廣告與釣魚網址即時偵測系統，包含 FastAPI
後端與 Chrome MV3 擴充套件。本 README 涵蓋安裝、啟動、測試、部署與重新
訓練的完整步驟；架構與研究方法請見 `TruthMark.docx`，延伸技術文件見
`docs/` 目錄。

## 目錄

- [一、系統需求](#一系統需求)
- [二、安裝](#二安裝)
- [三、啟動後端](#三啟動後端)
- [四、安裝 Chrome 擴充套件](#四安裝-chrome-擴充套件)
- [五、環境變數](#五環境變數)
- [六、執行測試](#六執行測試)
- [七、重新訓練模型](#七重新訓練模型)
- [八、部署](#八部署)
- [九、延伸文件](#九延伸文件)
- [十、165 涉詐網域自動匯入](#十165-涉詐網域自動匯入)

## 一、系統需求

- Python 3.12（開發與測試皆在此版本驗證；3.10+ 應可運作但未逐一驗證）
- Node.js（僅用於擴充套件 JS 語法檢查，非執行期必要）
- 約 200MB 可用記憶體（含 scikit-learn／numpy 等執行環境）

## 二、安裝

```bash
cd va
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

`requirements.txt` 內的 `tldextract>=5.1` 是**必要依賴**，不是可省略的
選用套件——若未安裝，系統會退回語意不同的簡易網域解析，影響特徵值與
測試結果（詳見 `docs/PERFORMANCE.md` 與 `/health` 回應中的 `tldextract`
欄位）。安裝完成後可執行下列指令確認：

```bash
python3 -c "import tldextract; print('OK:', tldextract.__version__)"
```

## 三、啟動後端

```bash
cp .env.example .env   # 依需求填入，至少建議設定 TRUTHMARK_FINGERPRINT_SALT
# 將 .env 內容匯出為環境變數，或使用 python-dotenv／部署平台的環境變數設定
python3 run_server.py
```

預設啟動於 `http://127.0.0.1:5500`（僅限本機存取；要開放區域網路存取
請設定 `HOST=0.0.0.0`，但正式對外部署請見下方「八、部署」的安全性建議，
不要直接把開發伺服器暴露在公開網路）。

啟動後可用下列指令確認服務正常：

```bash
curl http://127.0.0.1:5500/health
```

## 四、安裝 Chrome 擴充套件

1. 開啟 `chrome://extensions`
2. 開啟右上角「開發人員模式」
3. 點選「載入未封裝項目」，選擇本專案的 `外掛/` 資料夾
4. 確認擴充套件圖示出現、無錯誤訊息

擴充套件預設會依序嘗試連線 `manifest.json → host_permissions` 內列出的
後端位址（本機開發埠與正式雲端網域）。若你的後端跑在其他埠號／網域，
需要同步更新 `manifest.json` 的 `host_permissions` 與
`content_security_policy.extension_pages` 的 `connect-src`，否則瀏覽器
會因 CSP 阻擋對外連線。

## 五、環境變數

完整清單與說明見 [`.env.example`](.env.example)。對外部署前**至少**
應該設定：

| 變數 | 用途 | 未設定時的行為 |
|---|---|---|
| `TRUTHMARK_ADMIN_KEY` | 保護 `/reload-model`、`DELETE /blocklist/{domain}`、`/audit-log` | 僅允許本機（loopback）呼叫 |
| `TRUTHMARK_FINGERPRINT_SALT` | 回報者指紋雜湊鹽值 | 使用內建預設鹽值（安全性較低） |
| `TRUTHMARK_TRUST_PROXY_CIDRS` 或 `TRUTHMARK_TRUST_PROXY` | 是否信任反向代理的 `X-Forwarded-For` | 一律使用 TCP 層真實來源 IP，不信任標頭 |

## 六、執行測試

```bash
# 完整測試（233 項，約需 10～15 秒）
python3 -m unittest discover -s tests

# 或用 pytest（可同時取得覆蓋率報告）
pip install pytest pytest-cov
python3 -m pytest tests/ --cov=. --cov-report=term-missing
```

本版本測試套件共 233 項；測試不需啟動伺服器，也不需外網。覆蓋率報告請依目前環境重新產生，不把舊環境的覆蓋率數字當成最終保證。

Chrome 擴充套件的瀏覽器端到端測試見 `docs/EXTENSION_E2E_TESTING.md`
（含手動驗收清單與 Playwright 自動化測試骨架）。

## 七、重新訓練模型

```bash
python3 train_model.py                    # 完整訓練（所有候選模型 × 完整超參數搜尋）
python3 train_model.py --quick            # 開發用：每個模型只試 2 組超參數，較快
python3 train_model.py --no-save          # 只評估、不覆寫 scam_model.pkl／model_report.json
python3 train_model.py --benign-validation path/to/external_validation.json
                                           # 使用獨立驗證集選擇 target prior（見 docs/EXTERNAL_VALIDATION_PLAN.md）
```

訓練完成後會更新 `scam_model.pkl`、`model_meta.json`、`model_report.json`。
**若有任何一個檔案更新，記得同步檢查 `docs/ERROR_ANALYSIS.md`、
`docs/THRESHOLD_RATIONALE.md` 內引用的數字是否需要一併更新**（這兩份
文件目前引用的是本次 v7.2 的 `model_report.json` 快照）。

正式環境（已啟動的服務）若要套用新模型，呼叫（需 Admin Key）：

```bash
curl -X POST http://127.0.0.1:5500/reload-model -H "X-Admin-Key: <你的金鑰>"
```

## 八、部署

提供 `Dockerfile` 與 `docker-compose.yml` 作為容器化部署的起點：

```bash
docker compose up --build
```

雲端部署（例如 Azure App Service）的安全檢查清單：

- [ ] 已設定 `TRUTHMARK_ADMIN_KEY`（高強度亂數字串，不要用預設值）
- [ ] 已設定 `TRUTHMARK_FINGERPRINT_SALT`
- [ ] 若在反向代理後方，已設定 `TRUTHMARK_TRUST_PROXY_CIDRS`（優先於
      全域的 `TRUTHMARK_TRUST_PROXY=1`）
- [ ] `TRUTHMARK_ALLOWED_ORIGINS` 僅包含實際需要的網域，不使用 `*`
- [ ] `TRUTHMARK_RELOAD=0`（正式環境不需要自動重載）
- [ ] 已確認 `/health` 回應的 `tldextract.degraded` 為 `false`
- [ ] 擴充套件 `manifest.json` 的 `host_permissions`／CSP 已指向正式網域

## 九、延伸文件

| 文件 | 內容 |
|---|---|
| [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) | 實際 API 延遲量測（P50/P95/P99）、啟動時間、記憶體、測試覆蓋率 |
| [`docs/ERROR_ANALYSIS.md`](docs/ERROR_ANALYSIS.md) | 由 `model_report.json` 實際資料產生的錯誤分析 |
| [`docs/LABEL_REVIEW.md`](docs/LABEL_REVIEW.md) | 疑似標註錯誤個案的查證結果（含網路查證佐證） |
| [`docs/THRESHOLD_RATIONALE.md`](docs/THRESHOLD_RATIONALE.md) | 0.05／0.40／0.70 三個風險門檻的選擇理由 |
| [`docs/EXTERNAL_VALIDATION_PLAN.md`](docs/EXTERNAL_VALIDATION_PLAN.md) | 獨立外部驗證集建置方法論與腳本 |
| [`docs/EXTENSION_E2E_TESTING.md`](docs/EXTENSION_E2E_TESTING.md) | 瀏覽器端到端測試（手動清單＋ Playwright 骨架） |
| [`API.md`](API.md) | API 端點參考 |
| [`docs/CHANGELOG.md`](docs/CHANGELOG.md) | v7.1 → v7.2 版本變更摘要 |

## 十、165 涉詐網域自動匯入

自動把警政署「**165反詐騙諮詢專線_遭停止解析涉詐網站**」（資料量會隨官方更新而變動，
[data.gov.tw/dataset/176455](https://data.gov.tw/dataset/176455)，政府資料開放授權條款-第1版）
匯入動態黑名單（`dynamic_blocklist.json`，重啟後保留）。命中者 `/predict` 直接回
`source: "blocklist"`、高風險。

**啟用（一次性設定）**

1. 開啟上述資料集頁面，在「資料資源下載網址」的 CSV 按鈕上按右鍵 → 複製連結網址。
2. 寫進 `.env`：`TRUTHMARK_165_CSV_URL=<剛複製的網址>`
3. 重新啟動後端。啟動後約 5 秒首次同步，之後每 24 小時一次（`TRUTHMARK_165_SYNC_HOURS`，
   `0` ＝只在啟動時同步）。看到日誌 `165 同步完成：…新增 N` 即成功。

**手動操作**

```bash
# 立即同步（伺服器內建，直接更新記憶體；需 X-Admin-Key，未設金鑰時僅限本機）
curl -X POST http://127.0.0.1:5500/blocklist/sync-165 -H "X-Admin-Key: $TRUTHMARK_ADMIN_KEY"
curl -X POST http://127.0.0.1:5500/blocklist/sync-165 -H "X-Admin-Key: ..." -d '{"dry_run": true}' -H "Content-Type: application/json"
curl http://127.0.0.1:5500/blocklist/sync-165 -H "X-Admin-Key: ..."     # 上次結果

# 離線／排程：匯入已下載的 CSV（另一個行程，需重啟伺服器才生效）
python sync_165.py --file 165.csv --dry-run
python sync_165.py --file 165.csv
```

**安全設計**：只新增不刪除（誤封請用 `DELETE /blocklist/{domain}`）；白名單／`.gov.tw`
等可信任網域略過；`github.io`、`pages.dev` 等共用主機網域與公共後綴略過
（個別子網域如 `xxx.github.io` 照常匯入）；來源少於 `TRUTHMARK_165_MIN_ROWS`（預設 50）筆或
欄位無法解析（> 50%）整批中止、不寫入；下載有大小上限、僅限 http(s)、ETag 條件式請求；
下載網址只能由環境變數設定（API 不接受呼叫端指定，避免 SSRF）。每筆網域的來源
（`165_stop_resolve`）、網站性質、民國年月會記在 `dynamic_blocklist.json` 的 `sources`。

> 多 worker 部署：每個 worker 各自排程同步（檔案寫入有跨行程鎖，結果一致，只是各下載一次）。
> Docker：`dynamic_blocklist.json` 預設在容器內；每次行程啟動的第一次同步會忽略 ETag 強制重新下載，
> 所以重建容器後黑名單會自動補回來。
