# Chrome 擴充套件 E2E 測試

## 誠實說明：這次實際做到哪裡

這次比原先預期更進一步——本沙盒環境**意外已經預裝了 Playwright 與
Chromium**，所以沒有停在「只寫測試骨架」，而是實際嘗試執行了
`tests/e2e/test_extension_flow.py`：

- 瀏覽器能成功啟動（Chromium 141.0.7390.37，新版 headless 模式，
  不再需要 X server，排除了「沒有顯示環境」這個原本預期的障礙）；
- 但**擴充套件的 service worker 始終沒有註冊成功**，進一步排查發現
  連 `chrome://extensions/` 這類瀏覽器內部頁面都會回傳
  `net::ERR_INVALID_URL`，判斷是這個沙盒內建的 Chromium 屬於針對
  網頁爬取最佳化的精簡版本（`chromium_headless_shell`），本來就不是
  設計給「載入未封裝擴充套件、操作 chrome:// 管理頁面」這種情境使用，
  不是 `background.js`／`content.js`／`manifest.json` 本身的問題。

**結論**：測試骨架（含查詢參數去識別化的斷言邏輯）已經寫好且語法／
邏輯驗證過，`pytest tests/e2e/test_extension_flow.py` 在沒有 playwright
或後端未啟動時會正確地 `skip`（不會誤報成功），但「擴充套件真的能在
瀏覽器裡跑起來」這件事仍需要在一般桌機（有完整 Chrome/Chromium、而非
爬蟲用精簡版）上實際執行一次才能確認，見下方「方案二」。

目前 191＋ 項自動化測試（`tests/test_alignment.py` 等）驗證的是**靜態
結構對齊**：

- `manifest.json` 格式正確、icon 檔案存在
- `background.js`／`content.js`／`popup.js` 之間的訊息協定欄位一致
- 擴充套件呼叫的 API 路徑／參數與 `main.py` 實際路由定義一致
- `FEATURE_MAP`（特徵名稱）在前後端兩邊一致

這些測試能抓到「API 改了但擴充套件沒跟著改」這類整合性錯誤，但
**不能**驗證瀏覽器執行期行為（DOM 操作、訊息傳遞的 race condition、
實際網頁上 content.js 的選取器是否真的選到廣告元素等）。這正是審查
意見指出的落差，以下提供兩個可行方案。

## 方案一（推薦，成本最低）：手動驗收清單

不需要自動化框架，由人工在本機 Chrome 操作一次，花費約 15～20 分鐘，
建議在口試前執行一次並截圖存證：

```
□ 1. chrome://extensions 載入未封裝項目，確認無錯誤訊息
□ 2. 開啟 va/index.html（或任何一般網頁），確認右下角/工具列徽章正常顯示
□ 3. 開啟一個含多個外部連結的頁面（例如新聞首頁），點擊擴充套件圖示，
      觸發「掃描頁面連結」，確認：
      □ popup 內出現掃描結果（不是一直轉圈或報錯）
      □ 開發者工具 Network 分頁可見 /predict/batch 的請求，
        request body 的 url 欄位**不含查詢參數**（驗證 v7.2 去識別化）
□ 4. 對一個已知會被硬規則攔截的測試網址（例如含 wallet-drainer 字樣的
      測試網址）執行分析，確認風險分數與來源（source）欄位符合預期
□ 5. 點擊「回報為詐騙」，確認：
      □ popup 顯示已送出
      □ 之後再次查詢同一網址，source 變成 blocklist 相關來源
□ 6. 切換到另一個分頁，確認徽章會依據 autoCheckTab 設定正確更新或保持關閉
□ 7. chrome://extensions 檢視服務工作者（service worker）主控台，
      確認沒有未攔截的例外（uncaught exception）
```

把這份清單的勾選結果（可附截圖）放進論文附錄，就是「我真的測過瀏覽器
端完整流程」的具體證據，不需要重型框架也能交代過去。

## 方案二：Playwright 自動化（程式碼已備妥，需要能裝瀏覽器的環境執行）

`tests/e2e/test_extension_flow.py`（本次新增）已經寫好一支可執行的
Playwright 測試骨架，驗證「開啟頁面 → content.js 找到連結 → background.js
批次查詢 → 收到結果 → 頁面出現標記」的完整流程，其中最關鍵的斷言是
「送往 `/predict/batch` 的請求 body 不得含查詢參數」（驗證 v7.2 的
URL 去識別化）。**這支測試本次在沙盒環境跑過，但因沙盒內建的是爬蟲用
精簡版 Chromium（見上方說明），擴充套件無法在此載入成功**，需要在你
自己有完整 Chrome/Chromium 的電腦（或一般 CI runner）上重跑確認：

```bash
pip install playwright pytest-playwright
playwright install chromium

# 啟動後端（另開一個終端機）
cd va && uvicorn main:app --port 5500

# 執行 E2E 測試
pytest tests/e2e/test_extension_flow.py -v
```

請在自己的電腦上實際跑過一次，確認綠燈，再把結果（終端機輸出截圖即可）
放進論文附錄。如果紅燈，那正是這次補強最有價值的發現——代表真的抓到
一個只有真實瀏覽器環境才會暴露的問題。
