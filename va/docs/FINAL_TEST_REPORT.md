# TruthMark v7.2 Final — 最終程式驗證紀錄

## 驗證日期
2026-10-04

## 自動化測試

- Python unittest：**233 tests，233 passed，1 skipped**。
- Skip 原因：備份目錄比較測試只有在提供 `TRUTHMARK_BACKUP_DIR` 時才會執行。
- 測試命令：`python -m unittest discover -s tests -v`

## 前端 JavaScript

以下 5 個檔案均通過 `node --check`：

- `外掛/background.js`
- `外掛/content.js`
- `外掛/icons.js`
- `外掛/popup.js`
- `script.js`

## API 冒煙測試

在獨立 `TRUTHMARK_STATE_DIR` 下重新啟動 FastAPI：

- `GET /health`：HTTP 200
- 模型：成功載入，46 維特徵與 schema `3563ef6db762` 對齊
- `POST /predict`（`https://www.google.com/`）：HTTP 200，結果為 `low / trusted_domain`
- 動態黑名單：啟動時成功從持久化 state directory 載入 86,076 筆快照資料

## Docker

- `docker-compose.yml` 已通過 YAML 解析與關鍵環境／volume 結構檢查。
- 本次工作環境沒有 Docker CLI，因此**未執行實際 `docker build` / `docker compose up`**；不可將此項視為已完成的容器實機驗證。

## 重要環境備註

本次工作環境沒有安裝 `tldextract`，因此啟動時會標示 `degraded`，並使用有限、明確列出的離線 suffix fallback。正式環境請依 `requirements.txt` 安裝 `tldextract>=5.1`，以使用完整的離線 Public Suffix List 解析。
