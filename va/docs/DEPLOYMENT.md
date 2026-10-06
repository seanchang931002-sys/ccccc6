# Deployment Guide

## 本機

```bash
pip install -r requirements.txt
python run_server.py
```

## Docker

```bash
docker compose up --build
```

正式環境建議：

- `TRUTHMARK_RELOAD=0`
- 設定高強度 `TRUTHMARK_ADMIN_KEY`
- 設定高強度 `TRUTHMARK_FINGERPRINT_SALT`
- 使用 `TRUTHMARK_TRUST_PROXY_CIDRS`，避免全域信任 `X-Forwarded-For`
- `TRUTHMARK_ALLOWED_ORIGINS` 僅列必要來源
- 確認 `/health` 的 `tldextract.degraded` 為 `false`
- 使用持久化 volume 保存 runtime state

## Chrome

正式 API 位址變更時，同步更新：

- `va/外掛/manifest.json` 的 `host_permissions`
- `content_security_policy.extension_pages.connect-src`

然後重新載入未封裝擴充套件。
