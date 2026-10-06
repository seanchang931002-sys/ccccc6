# TruthMark 後端容器映像（v7.2）
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=5500 \
    TRUTHMARK_RELOAD=0

WORKDIR /app

# 先裝依賴（利用 Docker layer cache）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 只複製後端執行所需檔案（不含 tests、docs、外掛）
COPY *.py *.json *.csv index.html script.js style.css scam_model.pkl ./
# favicon 路由讀取 外掛/icons/icon48.png
COPY 外掛/icons ./外掛/icons

# 以非 root 使用者執行。正式容器會將 dynamic_blocklist.json、domain_reports.json、
# feedback.csv、165 同步狀態與 audit log 集中寫入 /app/data；docker-compose 以 volume 持久化。
RUN useradd -m truthmark && mkdir -p /app/data && chown -R truthmark /app
USER truthmark
ENV TRUTHMARK_AUDIT_LOG_PATH=/app/data/audit.log \
    TRUTHMARK_STATE_DIR=/app/data

EXPOSE 5500
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:5500/health',timeout=4).status==200 else 1)"

CMD ["python", "run_server.py"]
