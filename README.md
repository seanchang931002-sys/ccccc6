# TruthMark Final — 大學畢業專題程式

本套件以 TruthMark v7.2 為主體，整合：

- 46 維 URL 特徵與 Gradient Boosting 模型
- 白名單／黑名單／硬規則／AI 四層判斷
- Chrome MV3 擴充套件
- 使用者回報與動態黑名單
- 政府 165 涉詐網域資料同步
- Rate Limit、Admin Key、Audit Log、Trusted Proxy、SSRF 防護
- FastAPI、Docker、完整 unittest 與 E2E 測試骨架

## 啟動

```bash
cd va
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env   # Windows
# cp .env.example .env   # Linux/macOS
python run_server.py
```

服務預設：`http://127.0.0.1:5500`

## 測試

```bash
cd va
python -m unittest discover -s tests -v
```

完整測試不需要啟動伺服器，也不需要外網。

## Docker

```bash
docker compose up --build
```

## Chrome 擴充套件

開啟 `chrome://extensions` → 開啟開發人員模式 →「載入未封裝項目」→ 選擇 `va/外掛/`。

> 正式部署前請從 `va/.env.example` 設定管理金鑰、指紋鹽值及實際允許的 CORS／Proxy 設定。
