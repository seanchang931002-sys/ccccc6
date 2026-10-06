# =============================================================================
# run_server.py — 本機啟動 TruthMark 後端（v7.2）
# =============================================================================
# 預設 127.0.0.1:5500（Chrome 外掛呼叫的位址，僅限本機存取）。可用環境變數覆寫：
#   HOST=0.0.0.0（開放區域網路其他裝置連線）、PORT=5500、
#   TRUTHMARK_RELOAD=0（關閉自動重新載入）、TRUTHMARK_LOG_LEVEL=DEBUG
# 修正紀錄（v7.0）：print 改 logging；host/port/reload 可由環境變數設定。
# =============================================================================

import logging
import os

import uvicorn

if __name__ == "__main__":
    level = os.environ.get("TRUTHMARK_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s：%(message)s")
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5500"))
    reload = os.environ.get("TRUTHMARK_RELOAD", "0") == "1"
    logging.getLogger("truthmark").info("Truth 後端服務正在啟動：http://%s:%d（reload=%s）", host, port, reload)
    uvicorn.run("main:app", host=host, port=port, reload=reload, log_level=level.lower())
