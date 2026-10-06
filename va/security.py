# -*- coding: utf-8 -*-
# =============================================================================
# security.py — TruthMark 濫用防範機制（v7.2）
# =============================================================================
# 論文表 3-1 所述的四項機制集中在此，main.py 只負責呼叫：
#
#   1. CORS 白名單          cors_settings()：環境變數 TRUTHMARK_ALLOWED_ORIGINS（逗號分隔）覆寫預設白名單。
#   2. 速率限制（HTTP 429） RateLimiter：以「用戶端 IP」為單位的滑動視窗，行程內狀態。
#   3. 回報者指紋           reporter_fingerprint()：IP ＋ 鹽 的 SHA-256；原始 IP 不落地。
#   4. 管理員金鑰           check_admin()：X-Admin-Key 以 hmac.compare_digest 常數時間比對；
#                           未設定金鑰時，只允許本機（loopback）呼叫，避免「忘了設定」變成「全開」。
#
# 已知邊界（論文 6.2 如實揭露）：
#   - 指紋與限流都以 IP 為基礎，換 IP 或使用代理即可繞過，只能拉高濫用成本，不是身分驗證。
#   - 狀態存在單一行程記憶體；多 worker 部署時各自獨立（需改用 Redis 等共用儲存）。
#   - 位於反向代理後方時，須設定 TRUTHMARK_TRUST_PROXY=1 才會採用 X-Forwarded-For 的第一段；
#     預設不信任該標頭，避免用戶端自行偽造 IP 來繞過限流。
# =============================================================================
from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import os
import secrets
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------

DEFAULT_CORS_ORIGINS: Tuple[str, ...] = (
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    "http://127.0.0.1:8000",
    "http://localhost:8000",
    # 展示用 Azure 部署（與 外掛/manifest.json 的 host_permissions 一致）
    "https://confirmtm-dvccd9d5hydrcjfy.japanwest-01.azurewebsites.net",
    "https://mycgu-aabjg2dgatbkb9gw.malaysiawest-01.azurewebsites.net",
)
# Chrome 擴充功能的 origin 為 chrome-extension://<32 碼小寫英文字母>
CORS_EXTENSION_REGEX = r"^chrome-extension://[a-z]{32}$"

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})

# 路由名稱 → (視窗內最多次數, 視窗秒數)
# v7.2：補上 /predict、/predict/batch 的限流。這兩條是系統中最常被呼叫、也最耗
# CPU（特徵擷取＋模型推論）的路由；先前只限流 /feedback、/blocklist/add，若服務
# 對外開放，仍可能被大量 predict／batch 請求形成運算資源濫用（API DoS）。
# 門檻刻意比 /feedback、/blocklist/add 寬鬆許多，避免影響 Chrome 擴充套件正常
# 的頁面連結／廣告掃描（單頁最多 50 筆，使用者快速切換分頁時可能短時間內觸發
# 多次批次請求）。
DEFAULT_LIMITS: Dict[str, Tuple[int, int]] = {
    "feedback": (20, 60),         # /feedback         20 次／分
    "blocklist_add": (10, 60),    # /blocklist/add    10 次／分
    "predict": (120, 60),         # /predict          120 次／分（單一網址查詢，含徽章輪詢）
    "predict_batch": (30, 60),    # /predict/batch    30 次／分（單次最多 50 筆，已有運算量）
}


def cors_settings() -> Tuple[List[str], str]:
    """回傳 (allow_origins, allow_origin_regex)。TRUTHMARK_ALLOWED_ORIGINS 非空時取代預設清單。"""
    raw = os.environ.get("TRUTHMARK_ALLOWED_ORIGINS", "").strip()
    if raw:
        origins = [o.strip().rstrip("/") for o in raw.split(",") if o.strip()]
    else:
        origins = list(DEFAULT_CORS_ORIGINS)
    return origins, CORS_EXTENSION_REGEX


# ---------------------------------------------------------------------------
# 用戶端 IP 與回報者指紋
# ---------------------------------------------------------------------------

def trust_proxy() -> bool:
    """全域信任開關：TRUTHMARK_TRUST_PROXY=1 時，不論連線來源是誰都採用 X-Forwarded-For。"""
    return os.environ.get("TRUTHMARK_TRUST_PROXY", "0") == "1"


def _trusted_proxy_networks() -> Tuple["ipaddress._BaseNetwork", ...]:
    """
    v7.2 新增：TRUTHMARK_TRUST_PROXY_CIDRS（逗號分隔的 CIDR 清單，例如
    "10.0.0.0/8,172.16.0.0/12"）。比起全域布林開關 TRUST_PROXY，這裡可以把
    「信任 X-Forwarded-For」收斂到「只有來自指定反向代理網段的連線」才生效，
    降低設定錯誤（或代理前方再被插入一層不受信任的轉發）時 IP 被偽造的風險。
    """
    raw = os.environ.get("TRUTHMARK_TRUST_PROXY_CIDRS", "").strip()
    nets: List["ipaddress._BaseNetwork"] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            nets.append(ipaddress.ip_network(part, strict=False))
        except ValueError:
            continue
    return tuple(nets)


def is_trusted_proxy_peer(peer_ip: Optional[str]) -> bool:
    """實際 TCP 連線來源 IP 是否落在 TRUTHMARK_TRUST_PROXY_CIDRS 設定的網段內。"""
    if not peer_ip:
        return False
    try:
        ip_obj = ipaddress.ip_address(peer_ip)
    except ValueError:
        return False
    return any(ip_obj in net for net in _trusted_proxy_networks())


def client_ip(client_host: Optional[str], forwarded_for: Optional[str] = None,
              *, trust_forwarded: Optional[bool] = None) -> str:
    """
    取得用戶端 IP。只有在下列任一條件成立時才採用 X-Forwarded-For 的第一段：
      1. trust_forwarded 明確傳入 True（供測試或呼叫端覆寫）；
      2. TRUTHMARK_TRUST_PROXY=1（全域信任，沿用既有行為）；
      3. 實際連線來源（client_host，即 TCP 層的真正對端）落在
         TRUTHMARK_TRUST_PROXY_CIDRS 設定的網段內（v7.2 新增，較精確）。
    三者皆不成立時一律採用 TCP 層的 client_host，不理會用戶端可自行偽造的標頭。
    """
    if trust_forwarded is None:
        use_xff = trust_proxy() or is_trusted_proxy_peer(client_host)
    else:
        use_xff = trust_forwarded
    if use_xff and forwarded_for:
        first = forwarded_for.split(",")[0].strip()
        if first:
            return first[:64]
    return (client_host or "unknown")[:64]


def _load_salt() -> str:
    env = os.environ.get("TRUTHMARK_FINGERPRINT_SALT", "").strip()
    return env or secrets.token_hex(16)


# 未設定環境變數時，每次啟動隨機產生：重啟後同一 IP 會得到不同指紋（隱私優先，代價是
# 重啟後舊回報無法與新回報去重）。正式部署請固定 TRUTHMARK_FINGERPRINT_SALT。
_SALT = _load_salt()


def reporter_fingerprint(ip: str, salt: Optional[str] = None) -> str:
    """IP 加鹽雜湊（取前 16 個十六進位字元）；只存雜湊，不存原始 IP。"""
    material = f"{salt if salt is not None else _SALT}|{ip}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()[:16]


# ---------------------------------------------------------------------------
# 速率限制（滑動視窗）
# ---------------------------------------------------------------------------

class RateLimiter:
    """執行緒安全的滑動視窗限流器。key = (路由名稱, 用戶端識別)。"""

    def __init__(self, limits: Optional[Dict[str, Tuple[int, int]]] = None,
                 *, enabled: bool = True, max_keys: int = 10000) -> None:
        self.limits: Dict[str, Tuple[int, int]] = dict(limits or DEFAULT_LIMITS)
        self.enabled = enabled
        self._max_keys = max_keys
        self._hits: Dict[Tuple[str, str], Deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, route: str, client: str, now: Optional[float] = None) -> Tuple[bool, int]:
        """登記一次請求。回傳 (允許, 建議 Retry-After 秒數)；被拒絕的請求不計入視窗。"""
        if not self.enabled or route not in self.limits:
            return True, 0
        limit, window = self.limits[route]
        t = time.monotonic() if now is None else now
        key = (route, client)
        with self._lock:
            q = self._hits.get(key)
            if q is None:
                if len(self._hits) >= self._max_keys:
                    self._evict(t)
                q = self._hits[key] = deque()
            cutoff = t - window
            while q and q[0] <= cutoff:
                q.popleft()
            if len(q) >= limit:
                retry = max(1, int(q[0] + window - t) + 1)
                return False, retry
            q.append(t)
            return True, 0

    def _evict(self, now: float) -> None:
        """記憶體保護：清掉視窗已過期的 key；仍超量則丟掉最舊的一半。"""
        for key in list(self._hits):
            _, window = self.limits.get(key[0], (0, 60))
            q = self._hits[key]
            if not q or q[-1] <= now - window:
                del self._hits[key]
        if len(self._hits) >= self._max_keys:
            for key in list(self._hits)[: len(self._hits) // 2]:
                del self._hits[key]

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


# ---------------------------------------------------------------------------
# 管理員金鑰
# ---------------------------------------------------------------------------

def admin_key() -> str:
    return os.environ.get("TRUTHMARK_ADMIN_KEY", "").strip()


# ---------------------------------------------------------------------------
# 稽核紀錄（Audit Log）— v7.2 新增
# ---------------------------------------------------------------------------
# 目的：讓「誰、何時、對哪個網域、做了什麼管理動作、成不成功」有可追溯紀錄，
# 對應論文第三章提到的 NIST CSF Govern／Zero Trust「可問責性」精神——光有
# Admin Key 驗證只回答「這個人有沒有權限」，稽核紀錄才能回答「這個人實際做了
# 什麼」。採 JSON Lines（每行一筆 JSON）格式，方便後續用任何工具逐行解析，
# 不需要載入整個檔案、也不會因為單行格式錯誤讓整份記錄無法解析。
#
# 寫入的事件只記錄「指紋」（reporter_fingerprint 的雜湊），不記錄原始 IP，
# 與系統其他地方（feedback.csv、blocklist 回報）的隱私原則一致。
# ---------------------------------------------------------------------------

class AuditLog:
    """執行緒安全、附加寫入（append-only）的管理動作稽核紀錄。"""

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = Path(path or os.environ.get("TRUTHMARK_AUDIT_LOG_PATH", "audit.log"))
        self._lock = threading.Lock()

    def record(
        self,
        action: str,
        *,
        actor: str = "admin",
        target: str = "",
        success: bool = True,
        detail: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        寫入一筆稽核紀錄並回傳該筆紀錄（方便呼叫端順便記錄或測試）。
        action 建議用固定代碼，例如：
          "blocklist.add" / "blocklist.remove" / "model.reload"
        """
        entry: Dict[str, Any] = {
            "time": datetime.now(timezone.utc).isoformat(),
            "actor": actor,
            "action": action,
            "target": target,
            "success": bool(success),
            "detail": detail or {},
        }
        line = json.dumps(entry, ensure_ascii=False)
        with self._lock:
            try:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except OSError:
                # 稽核紀錄寫入失敗不應讓管理操作本身失敗；僅盡力而為。
                pass
        return entry

    def tail(self, limit: int = 50) -> List[Dict[str, Any]]:
        """讀回最近 N 筆稽核紀錄（新到舊），供 /audit-log 管理端點使用。"""
        if not self.path.exists():
            return []
        with self._lock:
            try:
                lines = self.path.read_text(encoding="utf-8").splitlines()
            except OSError:
                return []
        out: List[Dict[str, Any]] = []
        for line in reversed(lines[-max(limit, 0) * 2 or len(lines):]):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if len(out) >= limit:
                break
        return out


def check_admin(provided: Optional[str], client_host: Optional[str],
                *, configured_key: Optional[str] = None) -> Tuple[bool, str]:
    """
    驗證管理員權限，回傳 (通過, 失敗原因代碼)。
      - 已設定金鑰：X-Admin-Key 必須相符（常數時間比對）。
      - 未設定金鑰：只有 loopback 用戶端可通過（本機開發方便，遠端一律拒絕）。
    """
    key = admin_key() if configured_key is None else configured_key
    if key:
        if provided and hmac.compare_digest(provided.encode("utf-8"), key.encode("utf-8")):
            return True, ""
        return False, "admin_key_invalid"
    if (client_host or "") in LOOPBACK_HOSTS:
        return True, ""
    return False, "admin_key_not_configured"
