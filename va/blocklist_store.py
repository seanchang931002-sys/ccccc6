# =============================================================================
# blocklist_store.py — 動態黑名單與網域回報門檻的持久化邏輯
# =============================================================================
# 重構紀錄（v6）：把原本寫在 main.py 的 load_domain_reports / save_domain_reports /
# register_domain_report / load_dynamic_blocklist / save_domain_to_dynamic_blocklist
# 包成 BlocklistStore 類別；路徑、門檻、黑名單集合都由建構子明確傳入。
#
# 修正紀錄（v7.0）：
#   - 寫檔改為原子寫入（同目錄暫存檔 → fsync → os.replace），程式中途當掉或
#     多個請求同時寫入時，不會留下寫到一半、無法解析的 JSON。
#   - 所有「讀取 → 修改 → 寫回」流程加上執行緒鎖（RLock）；FastAPI 會把這些
#     同步 I/O 丟到執行緒池，多個回報同時進來時舊版可能互相覆蓋、少算次數。
#   - 網域正規化 normalize_domain()：小寫、去 www.、去尾端點、去埠號與 userinfo，
#     也接受使用者貼上的完整網址；與 features.get_hostname() 採同一套規則，
#     main.py 比對黑名單時才不會因為大小寫、www 或尾端點而漏判。
#   - 新增 import_domains(iterable, source)：供 165 開放資料（詐騙網站清單）
#     批次匯入；import_file(path, source) 可直接讀 CSV / TXT（自動偵測欄位）。
#   - print 改為 logging。
#   - 可信任網域保護（is_protected）：白名單網域（含子網域）與 .gov.tw 等可信任後綴
#     即使累積回報達門檻、或出現在匯入檔中，也不會自動列入黑名單（回報仍會記錄，
#     回傳 protected=True 供人工審核），避免惡意回報或資料錯誤封鎖官方網站。
#
# 修正紀錄（v7.1，對齊論文表 3-1）：
#   - 跨行程檔案鎖：filelock 包住所有「讀取 → 修改 → 寫回」；多個 worker／程序同時寫入
#     同一組 JSON 時不會互相覆蓋（未安裝 filelock 時退回只有執行緒鎖，並在啟動時記錄警告）。
#   - 回報者指紋：register_domain_report(domain, note, reporter=…) 以「不同回報者指紋數」
#     計算累積次數；同一指紋重複回報不重複計數（reporters 只存雜湊，最多保留 50 個）。
#     未帶 reporter 時維持 v7.0 行為（每次回報 +1），供匯入與舊測試使用。
#   - remove_domain()：管理員移除誤封的動態黑名單網域，並清空其回報累積；
#     內建（程式碼／政府）黑名單不在 dynamic_blocklist.json 中，不可由此移除。
#   對外行為、JSON 檔案格式（domains / last_updated）、回傳 dict 內容與 v6 相容；
#   dynamic_blocklist.json 另外新增 sources 欄位記錄每個網域的來源。
# =============================================================================

from __future__ import annotations

import csv
import io
import json
import logging
import os
import re
import tempfile
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional, Set

try:  # 與特徵擷取採用同一套 hostname 規則（失敗時退回內建簡易版）
    from features import get_hostname as _features_get_hostname
except Exception:  # noqa: BLE001
    _features_get_hostname = None

try:  # 跨行程檔案鎖（requirements.txt：filelock）
    from filelock import FileLock as _FileLock
except Exception:  # noqa: BLE001
    _FileLock = None

logger = logging.getLogger("truthmark.blocklist")
if _FileLock is None:
    logger.warning("未安裝 filelock：僅有執行緒鎖，多 worker／多程序同時寫檔可能互相覆蓋（pip install filelock）")

MAX_REPORTERS_KEPT = 50   # 每個網域最多保留的回報者指紋數

_DOMAIN_RE = re.compile(r"^(?=.{3,253}$)(?:[a-z0-9_](?:[a-z0-9_-]{0,61}[a-z0-9_])?\.)+[a-z0-9-]{2,63}$")
_IPV4_RE = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")


def normalize_domain(value: Any) -> str:
    """
    將使用者輸入（網域或完整網址）正規化為黑名單比對用的 hostname：
    小寫、去 www.、去尾端點、去埠號與 userinfo、中文網域轉 punycode。
    無法解析或格式不合法時回傳 ""。
    """
    if value is None:
        return ""
    try:
        text = str(value).strip()
    except Exception:  # noqa: BLE001
        return ""
    if not text or len(text) > 2048:
        return ""

    host = ""
    if _features_get_hostname is not None:
        try:
            host = _features_get_hostname(text)
        except Exception:  # noqa: BLE001
            host = ""
    if not host:
        # 退回簡易解析：去 scheme、路徑、userinfo、埠號
        t = text.lower()
        t = re.sub(r"^[a-z][a-z0-9+.-]*://", "", t)
        t = re.split(r"[/?#\\]", t, maxsplit=1)[0]
        t = t.rpartition("@")[2]
        if not t.startswith("["):
            t = t.split(":", 1)[0]
        host = t.strip(".")
        if host.startswith("www.") and "." in host[4:]:
            host = host[4:]

    host = host.strip().lower().rstrip(".")
    if not host:
        return ""
    if _IPV4_RE.match(host) or ":" in host:
        return host
    if not _DOMAIN_RE.match(host):
        return ""
    return host


def default_is_protected(domain: str) -> bool:
    """預設保護規則：網域本身或任一層父網域在 rules_config.TRUSTED_DOMAINS，或以 TRUSTED_SUFFIXES 結尾。"""
    try:
        from rules_config import TRUSTED_DOMAINS, TRUSTED_SUFFIXES
    except Exception:  # noqa: BLE001
        return False
    host = (domain or "").strip().lower().rstrip(".")
    if not host:
        return False
    labels = host.split(".")
    if any(".".join(labels[i:]) in TRUSTED_DOMAINS for i in range(len(labels))):
        return True
    dotted = "." + host
    return any(dotted.endswith(s) or host == s.lstrip(".") for s in TRUSTED_SUFFIXES)


def _atomic_write_json(path: str, data: Any) -> None:
    """原子寫入 JSON：寫到同目錄暫存檔、fsync，再以 os.replace 取代原檔。"""
    directory = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp-", suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        # Windows 上目標檔可能暫時被防毒或其他程序開啟，短暫重試
        for attempt in range(5):
            try:
                os.replace(tmp_path, path)
                return
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.05 * (attempt + 1))
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


class BlocklistStore:
    """
    管理「動態黑名單」與「網域回報累積次數」的讀寫。

    - dynamic_blocklist.json：使用者回報達門檻後新增、或由 165 開放資料匯入的網域，
      重啟伺服器仍會保留。
    - domain_reports.json：每個網域目前累積的回報次數（含尚未達門檻者）。
    """

    def __init__(
        self,
        base_dir: str,
        blocked_domains: Set[str],
        threshold: int,
        is_protected: Optional[Callable[[str], bool]] = None,
    ) -> None:
        self.dynamic_blocklist_path = os.path.join(base_dir, "dynamic_blocklist.json")
        self.domain_reports_path = os.path.join(base_dir, "domain_reports.json")
        self.blocked_domains = blocked_domains
        self.threshold = max(1, int(threshold))
        self.is_protected: Callable[[str], bool] = is_protected or default_is_protected
        self._lock = threading.RLock()
        self._xlocks: Dict[str, Any] = {}

    @contextmanager
    def _guard(self):
        """執行緒鎖 + 跨行程檔案鎖（鎖檔放在 domain_reports.json 同目錄）。"""
        with self._lock:
            if _FileLock is None:
                yield
                return
            lock_path = os.path.join(os.path.dirname(os.path.abspath(self.domain_reports_path)), ".truthmark.lock")
            xlock = self._xlocks.get(lock_path)
            if xlock is None:
                xlock = self._xlocks[lock_path] = _FileLock(lock_path, timeout=10)
            with xlock:  # 同一個 FileLock 物件可重入（內層呼叫不會死結）
                yield

    def _protected(self, domain: str) -> bool:
        try:
            return bool(self.is_protected(domain))
        except Exception:  # noqa: BLE001
            return False

    # ------------------------------------------------------------------
    # 共用讀檔
    # ------------------------------------------------------------------

    @staticmethod
    def _read_json(path: str, default: Any) -> Any:
        if not os.path.exists(path):
            return default
        try:
            with open(path, "r", encoding="utf-8-sig") as f:
                data = json.load(f)
            return data if isinstance(data, type(default)) else default
        except Exception as e:  # noqa: BLE001
            logger.warning("無法讀取 %s：%s", os.path.basename(path), e)
            return default

    # ------------------------------------------------------------------
    # 網域回報累積
    # ------------------------------------------------------------------

    def load_domain_reports(self) -> Dict[str, dict]:
        """讀取目前各網域累積的回報紀錄（尚未達門檻者也會保留在此）。"""
        with self._guard():
            return self._read_json(self.domain_reports_path, {})

    def save_domain_reports(self, reports: Dict[str, dict]) -> None:
        with self._guard():
            try:
                _atomic_write_json(self.domain_reports_path, reports)
            except Exception as e:  # noqa: BLE001
                logger.warning("無法寫入 domain_reports.json：%s", e)

    def register_domain_report(self, domain: str, note: str = "", reporter: Optional[str] = None) -> dict:
        """
        記錄一筆針對該網域的回報，並回傳目前累積狀態。
        只有累積回報數達到 self.threshold，才會真正把網域加入 blocked_domains。

        reporter：回報者指紋（security.reporter_fingerprint）。有帶時，同一指紋對同一網域
        只計一次（回傳 duplicate=True，不增加次數）；沒帶時每次回報 +1（v7.0 行為）。
        """
        normalized = normalize_domain(domain) or str(domain or "").strip().lower()
        note = str(note or "")[:300]

        with self._guard():
            reports = self.load_domain_reports()
            entry = reports.get(normalized) or {
                "count": 0,
                "notes": [],
                "first_reported": datetime.now().isoformat(),
            }

            try:
                prev_count = int(entry.get("count", 0))
            except (TypeError, ValueError):
                prev_count = 0
            duplicate = False
            if reporter:
                reporters = [r for r in (entry.get("reporters") or []) if isinstance(r, str)]
                if reporter in reporters:
                    duplicate = True
                    entry["count"] = max(prev_count, 1)
                else:
                    reporters.append(reporter)
                    entry["reporters"] = reporters[-MAX_REPORTERS_KEPT:]
                    entry["count"] = prev_count + 1
            else:
                entry["count"] = prev_count + 1
            if note and not duplicate:
                # 只保留最近 10 筆備註，避免檔案無限膨脹
                entry["notes"] = (list(entry.get("notes") or []) + [note])[-10:]
            entry["last_reported"] = datetime.now().isoformat()

            protected = self._protected(normalized)
            if protected:
                entry["protected"] = True
            reports[normalized] = entry
            self.save_domain_reports(reports)

            newly_blocked = False
            if (not protected and entry["count"] >= self.threshold
                    and normalized not in self.blocked_domains):
                self.blocked_domains.add(normalized)
                self.save_domain_to_dynamic_blocklist(normalized, note, source="user_report")
                newly_blocked = True
            elif protected and entry["count"] >= self.threshold:
                logger.warning("可信任網域 %s 累積回報 %d 次，未自動封鎖（需人工審核）", normalized, entry["count"])

            return {
                "domain": normalized,
                "count": entry["count"],
                "threshold": self.threshold,
                "blocked": normalized in self.blocked_domains,
                "newly_blocked": newly_blocked,
                "protected": protected,
                "duplicate": duplicate,
            }

    def remove_domain(self, domain: str) -> dict:
        """
        管理員移除誤封網域：從記憶體黑名單、dynamic_blocklist.json 移除，並把回報累積歸零
        （保留 removed_at 紀錄，避免同一批回報者立刻再把它封回去：指紋清單一併清空）。
        只能移除 dynamic_blocklist.json 內的網域；內建黑名單回傳 removed=False、reason="builtin"。
        """
        normalized = normalize_domain(domain) or str(domain or "").strip().lower()
        with self._guard():
            existing = self._load_dynamic_file()
            records = [d for d in existing["domains"] if isinstance(d, str)]
            in_dynamic = normalized in records
            in_memory = normalized in self.blocked_domains
            if not in_dynamic:
                return {"domain": normalized, "removed": False,
                        "reason": "builtin" if in_memory else "not_found"}
            records = [d for d in records if d != normalized]
            existing["domains"] = records
            existing["sources"].pop(normalized, None)
            existing["last_updated"] = datetime.now().isoformat()
            _atomic_write_json(self.dynamic_blocklist_path, existing)
            self.blocked_domains.discard(normalized)

            reports = self.load_domain_reports()
            if normalized in reports:
                reports[normalized] = {
                    "count": 0, "notes": [], "reporters": [],
                    "first_reported": reports[normalized].get("first_reported", datetime.now().isoformat()),
                    "removed_at": datetime.now().isoformat(),
                }
                self.save_domain_reports(reports)
            logger.info("管理員已移除誤封網域 %s", normalized)
            return {"domain": normalized, "removed": True, "reason": ""}

    # ------------------------------------------------------------------
    # 動態黑名單持久化
    # ------------------------------------------------------------------

    def _load_dynamic_file(self) -> Dict[str, Any]:
        data = self._read_json(self.dynamic_blocklist_path, {})
        if not isinstance(data.get("domains"), list):
            data["domains"] = []
        if not isinstance(data.get("sources"), dict):
            data["sources"] = {}
        return data

    def load_dynamic_blocklist(self) -> int:
        """啟動時讀取先前累積的動態黑名單，併入 self.blocked_domains；回傳讀入筆數。"""
        with self._guard():
            data = self._load_dynamic_file()
            loaded = 0
            for d in data.get("domains", []):
                normalized = normalize_domain(d)
                if normalized:
                    self.blocked_domains.add(normalized)
                    loaded += 1
            if loaded:
                logger.info("已載入動態黑名單 %d 筆", loaded)
            return loaded

    def save_domain_to_dynamic_blocklist(self, domain: str, note: str = "", source: str = "manual") -> None:
        """將新增的網域寫入 dynamic_blocklist.json（原子寫入），供下次啟動讀回。"""
        self._persist_domains([domain], source=source, note=note)

    def _persist_domains(self, domains: Iterable[str], source: str, note: str = "",
                         notes: Optional[Dict[str, str]] = None) -> int:
        with self._guard():
            try:
                existing = self._load_dynamic_file()
                records: List[str] = [d for d in existing["domains"] if isinstance(d, str)]
                known = set(records)
                sources: Dict[str, Any] = existing["sources"]
                now = datetime.now().isoformat()
                added = 0
                for domain in domains:
                    if not domain or domain in known:
                        continue
                    records.append(domain)
                    known.add(domain)
                    n = (notes or {}).get(domain) or note
                    sources[domain] = {"source": source, "added_at": now, **({"note": n[:200]} if n else {})}
                    added += 1
                if not added:
                    return 0
                existing["domains"] = records
                existing["sources"] = sources
                existing["last_updated"] = now
                _atomic_write_json(self.dynamic_blocklist_path, existing)
                return added
            except Exception as e:  # noqa: BLE001
                logger.warning("無法寫入 dynamic_blocklist.json：%s", e)
                return 0

    # ------------------------------------------------------------------
    # 165 開放資料匯入
    # ------------------------------------------------------------------

    def import_domains(self, domains: Iterable[Any], source: str = "165_opendata",
                       notes: Optional[Dict[str, str]] = None) -> Dict[str, int]:
        """
        批次匯入網域（例如 165 開放資料的詐騙網站清單），直接加入黑名單並持久化。
        domains 可為網域或完整網址；可信任網域會略過。
        notes：以「正規化後網域」為鍵的備註（例如 165 的網站性質、民國年月），寫入
        dynamic_blocklist.json 的 sources[domain].note。回傳 {total, added, already, invalid, protected}。
        """
        total = added_count = already = invalid = protected = 0
        new_domains: List[str] = []
        seen: Set[str] = set()
        for raw in domains:
            total += 1
            normalized = normalize_domain(raw)
            if not normalized:
                invalid += 1
                continue
            if normalized in seen:
                already += 1
                continue
            seen.add(normalized)
            if self._protected(normalized):
                protected += 1
                continue
            new_domains.append(normalized)

        with self._guard():
            fresh = [d for d in new_domains if d not in self.blocked_domains]
            already += len(new_domains) - len(fresh)
            for d in fresh:
                self.blocked_domains.add(d)
            added_count = len(fresh)
            if fresh:
                self._persist_domains(fresh, source=source, notes=notes)

        logger.info("黑名單匯入（%s）：共 %d 筆，新增 %d、已存在 %d、格式不符 %d、可信任網域略過 %d",
                    source, total, added_count, already, invalid, protected)
        return {"total": total, "added": added_count, "already": already, "invalid": invalid,
                "protected": protected}

    def import_file(self, path: str, source: Optional[str] = None) -> Dict[str, int]:
        """
        讀取 CSV / TXT 檔匯入黑名單。CSV 會自動偵測網址／網域欄位
        （欄名含 url、domain、網址、網域、WEBURL…；找不到時掃描每個儲存格）。
        """
        source = source or os.path.splitext(os.path.basename(path))[0]
        with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
            content = f.read()
        return self.import_domains(_iter_domain_cells(content), source=source)


_COLUMN_HINTS = ("weburl", "url", "domain", "網址", "網域", "域名", "website", "site")


def _iter_domain_cells(content: str) -> Iterable[str]:
    """從 CSV / 純文字內容中取出疑似網域或網址的儲存格。"""
    try:
        sample = content[:4096]
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;") if "," in sample or "\t" in sample else None
    except csv.Error:
        dialect = None
    if dialect is None:
        for line in content.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                yield line
        return

    reader = csv.reader(io.StringIO(content), dialect)
    rows = list(reader)
    if not rows:
        return
    header = [h.strip().lower() for h in rows[0]]
    columns = [i for i, h in enumerate(header) if any(hint in h for hint in _COLUMN_HINTS)]
    body = rows[1:] if columns else rows
    for row in body:
        cells = [row[i] for i in columns if i < len(row)] if columns else row
        for cell in cells:
            cell = (cell or "").strip()
            if cell and ("." in cell):
                yield cell


if __name__ == "__main__":  # 命令列：python blocklist_store.py import <檔案> [來源名稱]
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s：%(message)s")
    if len(sys.argv) >= 3 and sys.argv[1] == "import":
        store = BlocklistStore(os.path.dirname(os.path.abspath(__file__)), set(), threshold=3)
        store.load_dynamic_blocklist()
        print(json.dumps(store.import_file(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None),
                         ensure_ascii=False))
    else:
        print("用法：python blocklist_store.py import <165開放資料.csv> [來源名稱]")
