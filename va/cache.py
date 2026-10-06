# =============================================================================
# cache.py — TruthMark 快取層（v7.0）
# =============================================================================
# 目的：
#   高頻出現的「已知安全網域」（白名單命中、或近期已判定為低風險的網域）
#   若每次都重新跑一次完整特徵擷取 + AI 模型推論，會浪費 CPU／延遲時間。
#
#   本模組提供兩種快取機制：
#
#   1. TTLCache
#      執行緒安全的「模擬 Redis」記憶體快取，具備 TTL（存活時間）與 LRU
#      容量上限淘汰。介面刻意模仿 Redis 常用指令（get / set / delete /
#      exists），未來若要換成真正的 Redis（redis.asyncio），只需要實作
#      相同介面的另一個 class 並替換 main.py 中的實例即可。
#
#   2. make_domain_lru
#      針對「輸入是字串、輸出完全由輸入決定」的純函式，直接使用
#      functools.lru_cache 做函式層級快取。
#
# 修正紀錄（v7.0，依 CONTRACT §3）：
#   - 改用 OrderedDict 實作 O(1) LRU：get 命中時 move_to_end，容量滿時
#     popitem(last=False) 淘汰最久未使用者（舊版用 min() 掃全表，O(n)）。
#   - 新增「網域索引」與「群組索引」：set(..., domain=registered_domain,
#     group=canonical_url)；delete_domain(domain) / delete_group(group)
#     只刪除索引內的 key（O(k)）。
#     修正舊版 bug：main.py 的 /blocklist/add 用 delete_prefix(domain)，
#     但 key 是以完整 URL 開頭（https://…），永遠清不到任何項目。
#   - 新增 clear()、purge_expired()、更完整的 stats()（淘汰數、過期數、
#     索引網域數）。delete() 改回傳是否真的刪除。
# =============================================================================

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from threading import RLock
from typing import Any, Callable, Dict, Optional, Set, TypeVar

T = TypeVar("T")


def _norm_domain(domain: Optional[str]) -> str:
    """索引用的網域正規化：小寫、去尾端點、去開頭 www.。"""
    d = (domain or "").strip().lower().rstrip(".")
    if d.startswith("www.") and "." in d[4:]:
        d = d[4:]
    return d


@dataclass
class _CacheEntry:
    value: Any
    expires_at: float
    domain: str = ""
    group: str = ""


class TTLCache:
    """
    模擬 Redis 的記憶體 TTL + LRU 快取（所有操作 O(1)，索引失效 O(k)）。

    Redis 上線後的替換方式：
        實作一個具備相同 get/set/delete/delete_domain/delete_group/clear/stats
        介面、內部改呼叫 redis.asyncio.Redis 的類別（網域索引可用 Redis SET
        實作），換掉 main.py 建立實例的那一行即可。
    """

    def __init__(self, max_size: int = 2048, default_ttl_seconds: float = 600.0) -> None:
        self._store: "OrderedDict[str, _CacheEntry]" = OrderedDict()
        self._domain_index: Dict[str, Set[str]] = {}
        self._group_index: Dict[str, Set[str]] = {}
        self._lock = RLock()
        self._max_size = max(1, int(max_size))
        self._default_ttl = float(default_ttl_seconds)
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.expirations = 0

    # ------------------------------------------------------------------
    # 內部：索引維護（呼叫端必須持有 self._lock）
    # ------------------------------------------------------------------

    def _unindex(self, key: str, entry: _CacheEntry) -> None:
        for index, name in ((self._domain_index, entry.domain), (self._group_index, entry.group)):
            if not name:
                continue
            keys = index.get(name)
            if keys is not None:
                keys.discard(key)
                if not keys:
                    del index[name]

    def _remove(self, key: str) -> bool:
        entry = self._store.pop(key, None)
        if entry is None:
            return False
        self._unindex(key, entry)
        return True

    # ------------------------------------------------------------------
    # Redis 風格介面
    # ------------------------------------------------------------------

    def get(self, key: str) -> Optional[Any]:
        """取得快取值；不存在或已過期回傳 None（語意等同 Redis GET 未命中）。"""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self.misses += 1
                return None
            if entry.expires_at <= time.monotonic():
                self._remove(key)
                self.expirations += 1
                self.misses += 1
                return None
            self._store.move_to_end(key)
            self.hits += 1
            return entry.value

    def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[float] = None,
        domain: Optional[str] = None,
        group: Optional[str] = None,
    ) -> None:
        """
        寫入快取值（等同 Redis SETEX）。
        domain：registered domain，供 delete_domain() 批次失效。
        group：同一網址的不同回應變體（debug/compact…）共用的群組名稱，供 delete_group()。
        """
        ttl = self._default_ttl if ttl_seconds is None else float(ttl_seconds)
        if ttl <= 0:
            return
        entry = _CacheEntry(
            value=value,
            expires_at=time.monotonic() + ttl,
            domain=_norm_domain(domain),
            group=group or "",
        )
        with self._lock:
            self._remove(key)
            self._store[key] = entry
            if entry.domain:
                self._domain_index.setdefault(entry.domain, set()).add(key)
            if entry.group:
                self._group_index.setdefault(entry.group, set()).add(key)
            while len(self._store) > self._max_size:
                old_key, old_entry = self._store.popitem(last=False)
                self._unindex(old_key, old_entry)
                self.evictions += 1

    def delete(self, key: str) -> bool:
        with self._lock:
            return self._remove(key)

    def delete_domain(self, domain: str) -> int:
        """刪除以該 registered domain 建立索引的所有快取項目，回傳刪除數量。"""
        name = _norm_domain(domain)
        if not name:
            return 0
        with self._lock:
            keys = list(self._domain_index.get(name, ()))
            return sum(1 for k in keys if self._remove(k))

    def delete_group(self, group: str) -> int:
        """刪除同一群組（同一個正規化網址的所有回應變體）的快取項目。"""
        if not group:
            return 0
        with self._lock:
            keys = list(self._group_index.get(group, ()))
            return sum(1 for k in keys if self._remove(k))

    def delete_prefix(self, prefix: str) -> int:
        """
        刪除所有 key 以 prefix 開頭的快取項目（O(n)，保留給舊呼叫端相容）。
        新程式請改用 delete_domain() / delete_group()。
        """
        with self._lock:
            matched = [k for k in self._store if k.startswith(prefix)]
            for k in matched:
                self._remove(k)
            return len(matched)

    def clear(self) -> int:
        """清除全部快取（例如重新載入模型後），回傳清除數量。"""
        with self._lock:
            count = len(self._store)
            self._store.clear()
            self._domain_index.clear()
            self._group_index.clear()
            return count

    def purge_expired(self) -> int:
        """主動清除所有已過期項目（O(n)，可由排程呼叫；一般情況靠 get/LRU 自然淘汰）。"""
        now = time.monotonic()
        with self._lock:
            expired = [k for k, e in self._store.items() if e.expires_at <= now]
            for k in expired:
                self._remove(k)
            self.expirations += len(expired)
            return len(expired)

    def exists(self, key: str) -> bool:
        return self.get(key) is not None

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            size = len(self._store)
            domains = len(self._domain_index)
            hits, misses = self.hits, self.misses
            evictions, expirations = self.evictions, self.expirations
        total = hits + misses
        hit_rate = round(hits / total, 4) if total else 0.0
        return {
            "size": size,
            "max_size": self._max_size,
            "default_ttl_seconds": self._default_ttl,
            "hits": hits,
            "misses": misses,
            "hit_rate": hit_rate,
            "evictions": evictions,
            "expirations": expirations,
            "domains_indexed": domains,
        }


def make_domain_lru(func: Callable[[str], T], maxsize: int = 4096) -> Callable[[str], T]:
    """
    將一個「輸入字串 → 輸出由輸入唯一決定」的純函式包成 lru_cache 版本。
    讓同一個網域字串在快取有效期間內不會被重複計算。
    """
    return lru_cache(maxsize=maxsize)(func)
