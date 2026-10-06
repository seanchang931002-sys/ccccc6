/**
 * Truth Chrome Extension — background.js (service worker)
 * =============================================================================
 * 負責：
 * 1. 判斷後端 API 位址（本機 127.0.0.1:5500 / 8000 優先，都連不上則備援雲端）
 * 2. 提供 /predict 呼叫（給 popup.js 與 content.js 使用）
 * 3. 分頁網址自動檢查 + 工具列徽章（紅/黃/綠）
 * 4. 批次檢查頁面連結（給 content.js 的「掃描此頁連結」「掃描頁面廣告」功能）
 * 5. 偵測歷史紀錄（chrome.storage.local）與回報計數，供 popup 的統計儀表板使用
 *
 * 對齊 main.py v7.2 API 合約（向下相容 v5 / v6 後端）：
 *   POST /predict { url, debug:false } →
 *     { ok, url, risk_score, risk_level(high/medium/low), risk_label, verdict,
 *       source, reasons[], suggestion, triggered_rules[], degraded, degrade_reason ... }
 *   POST /predict/batch { urls:[…≤50], compact:true } →
 *     { ok, count, results: { 原樣輸入的 url: 單筆（compact）回應 } }
 *   錯誤回應：{ ok:false, error, message }（舊版後端可能只帶其中一個，這裡統一補齊）
 *
 * v5.2 修正紀錄：
 *   - apiBase 解析結果快取 60 秒（以設定值為簽章，設定變更即失效）；同時多個請求
 *     只會共用一次探測。請求連線失敗時讓快取失效、重新探測一次後再重試一次。
 *     （舊版每次 predict 都先打一次 /health，請求量加倍）
 *   - CHECK_LINKS 改呼叫 /predict/batch（compact，每批最多 50 條）；後端舊版回
 *     404 / 405 / 422 時自動退回逐條併發 /predict，並記住 10 分鐘不再嘗試 batch。
 *   - predict 一律送 { url, debug:false }（popup 需要完整欄位，不用 compact）。
 *   - 統一處理 ok:false（error || message 兩個欄位都補齊）；錯誤結果不快取。
 *   - 本地結果快取 key 正規化：去前後空白；純錨點 #fragment（#top、#section）去掉，
 *     SPA hash 路由（#/pages/…、#!/、帶 / = ? & 的片段）保留（後端 path_scam_route
 *     特徵會用到）。送給後端的網址與快取 key 使用同一份正規化結果，前後一致。
 *   - 本地快取加上筆數上限（LRU），避免 service worker 記憶體無限成長。
 *   - 歷史紀錄記錄後端回應的 degraded（降級模式）狀態。
 * =============================================================================
 */

'use strict';

const DEFAULT_PORTS = [5500, 8000];
// 本機埠號都探測失敗時，最後備援雲端部署版本（畢業展示 / 不在本機跑後端時仍可用）
const PROD_API_BASE = 'https://confirmtm-dvccd9d5hydrcjfy.japanwest-01.azurewebsites.net';
const CACHE_TTL_MS = 5 * 60 * 1000; // 同一網址 5 分鐘內不重複打 API
const MAX_CACHE_ENTRIES = 500;      // 本地結果快取上限（超過時淘汰最久未使用者）
const MAX_HISTORY = 300;

const API_BASE_TTL_MS = 60 * 1000;          // apiBase 探測成功的結果快取 60 秒
const API_BASE_NEGATIVE_TTL_MS = 10 * 1000; // 全部探測失敗時，10 秒內不重複整輪探測
const LOCAL_PROBE_TIMEOUT_MS = 1200;
const PROD_PROBE_TIMEOUT_MS = 8000;

const PREDICT_TIMEOUT_MS = 8000;
const BATCH_TIMEOUT_MS = 20000;
const WRITE_TIMEOUT_MS = 5000;  // /feedback、/blocklist/add

const BATCH_CHUNK_SIZE = 50;    // 對齊後端 /predict/batch 單批上限
const MAX_CHECK_URLS = 100;     // 單次 CHECK_LINKS 最多檢查的網址數
const FALLBACK_CONCURRENCY = 4; // 退回逐條模式時的併發數
const BATCH_FALLBACK_STATUSES = new Set([404, 405, 422]);
const BATCH_UNSUPPORTED_TTL_MS = 10 * 60 * 1000;

// 記憶體快取：`${正規化網址}::full|compact` -> { data, time, domain }
const resultCache = new Map();

// -----------------------------------------------------------------------
// 設定值（存放於 chrome.storage.local）
// -----------------------------------------------------------------------
const DEFAULT_SETTINGS = {
  apiBase: '',              // 空字串代表自動偵測
  apiPort: 5500,            // 手動指定埠號（apiBase 為空時使用）
  autoCheckTab: false,      // 是否自動檢查目前分頁網址並顯示徽章（v7.1 起預設關閉：避免未經同意就把瀏覽網址送到後端；使用者可在設定中開啟）
  autoScanLinks: false,     // 是否自動掃描頁面上的連結並標示風險
  useProdFallback: true,    // 本機都連不上時，是否自動備援雲端後端
};

const API_SETTING_KEYS = ['apiBase', 'apiPort', 'useProdFallback'];

async function getSettings() {
  const stored = await chrome.storage.local.get(Object.keys(DEFAULT_SETTINGS));
  return { ...DEFAULT_SETTINGS, ...stored };
}

// -----------------------------------------------------------------------
// 共用工具
// -----------------------------------------------------------------------

/** 建立統一格式的錯誤回應（error 與 message 同內容）。 */
function makeError(message, extra) {
  const text = String(message || '未知錯誤');
  return { ok: false, error: text, message: text, ...(extra || {}) };
}

/** 把 FastAPI 的 detail（字串或驗證錯誤陣列）轉成可讀文字。 */
function detailToText(detail) {
  if (!detail) return '';
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    return detail
      .map(d => (d && typeof d === 'object' ? (d.msg || JSON.stringify(d)) : String(d)))
      .join('；');
  }
  try {
    return JSON.stringify(detail);
  } catch (e) {
    return String(detail);
  }
}

/** ok:false 回應補齊 error / message 兩個欄位。 */
function normalizeErrorData(data, httpStatus) {
  const msg = data.error || data.message || detailToText(data.detail) ||
    (httpStatus ? `後端回應錯誤：HTTP ${httpStatus}` : '後端回傳失敗');
  const out = { ...data, ok: false, error: String(msg), message: String(msg) };
  if (httpStatus) out.http_status = httpStatus;
  return out;
}

/** 單筆回應正規化：ok:false 補齊錯誤欄位；非物件視為格式錯誤。 */
function normalizeResult(data) {
  if (!data || typeof data !== 'object' || Array.isArray(data)) {
    return makeError('後端回應格式錯誤');
  }
  if (data.ok === false) return normalizeErrorData(data);
  return data;
}

/**
 * 讀取後端 JSON：
 *   - 2xx：回傳資料（ok:false 時補齊 error/message）
 *   - 非 2xx 但帶 JSON 錯誤內容：回傳正規化後的 ok:false 物件（含 http_status）
 *   - 其他（非 JSON、空內容）：丟出例外
 */
async function readApiJson(res) {
  let data = null;
  try {
    data = await res.json();
  } catch (e) {
    data = null;
  }

  const isObj = data && typeof data === 'object' && !Array.isArray(data);

  if (res.ok) {
    if (!isObj) throw new Error('後端回應格式錯誤（不是有效的 JSON 物件）');
    return data.ok === false ? normalizeErrorData(data) : data;
  }

  if (isObj && (data.ok === false || data.error || data.message || data.detail)) {
    return normalizeErrorData(data, res.status);
  }
  throw new Error(`後端回應錯誤：HTTP ${res.status}`);
}

/** fetch + 逾時；逾時丟出帶 isTimeout 標記的錯誤。 */
async function fetchWithTimeout(url, init, timeoutMs) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    return await fetch(url, { ...(init || {}), signal: ctrl.signal });
  } catch (err) {
    if (err && err.name === 'AbortError') {
      const e = new Error(`逾時（超過 ${Math.round(timeoutMs / 1000)} 秒）`);
      e.isTimeout = true;
      throw e;
    }
    throw err;
  } finally {
    clearTimeout(timer);
  }
}

/**
 * 本地快取 / 送出用的網址正規化：
 *   - 去前後空白（含全形空白）
 *   - 純錨點 fragment（#top、#section-2、空的 #）去掉：不影響判斷
 *   - SPA hash 路由（#/pages/…、#!/、片段含 / = ? &）保留：後端會用來判斷假投資 H5 路由
 */
// -----------------------------------------------------------------------
// 隱私保護：URL 去識別化（v7.2 新增）
// -----------------------------------------------------------------------
// normalizeCacheUrl() 是整支程式唯一、統一的「送往後端前的網址正規化」入口
// （/predict、/predict/batch、/feedback、本地快取 key、偵測歷史紀錄都共用
// 同一份正規化結果），因此把去識別化邏輯放在這裡，能確保：
//   1. 送給後端的網址一律只剩 protocol + host + path，不含查詢參數
//      （?user_id=...、fbclid、gclid 等追蹤或個資參數）與 hash fragment；
//   2. 本地快取 key 與偵測歷史紀錄用的也是同一個去識別化後的網址，三者
//      不會因為「有的有參數、有的沒參數」而互相對不上。
// 無法用 URL() 解析的字串（例如使用者貼上不完整的網址片段）則盡量單純做
// 空白清理，交由後端既有的格式驗證處理，不在前端擋下。
function normalizeCacheUrl(url) {
  const raw = String(url == null ? '' : url).replace(/^[\s　]+|[\s　]+$/g, '');
  try {
    const u = new URL(raw);
    u.search = '';
    u.hash = '';
    return u.toString();
  } catch (e) {
    // 無法解析為合法 URL（例如 chrome:// 內部頁面、不完整字串）：
    // 退回舊版的「只去頭尾空白＋條件式移除 hash」邏輯，不強行處理。
    let s = raw;
    const hashIdx = s.indexOf('#');
    if (hashIdx >= 0) {
      const frag = s.slice(hashIdx + 1);
      if (!/[\/!=?&]/.test(frag)) s = s.slice(0, hashIdx);
    }
    return s;
  }
}

// 從網址粗略取出網域（去掉開頭的 www.），用於 /blocklist/add 與歷史紀錄
function extractDomain(url) {
  try {
    const hostname = new URL(url).hostname.toLowerCase();
    return hostname.startsWith('www.') ? hostname.slice(4) : hostname;
  } catch (e) {
    return null;
  }
}

// -----------------------------------------------------------------------
// 本地結果快取（LRU + TTL；錯誤回應不快取）
// -----------------------------------------------------------------------
function cacheGet(normUrl, variants) {
  const now = Date.now();
  for (const variant of variants) {
    const key = `${normUrl}::${variant}`;
    const entry = resultCache.get(key);
    if (!entry) continue;
    if (now - entry.time >= CACHE_TTL_MS) {
      resultCache.delete(key);
      continue;
    }
    // 重新插入，維持 Map 的插入順序 = 最近使用順序
    resultCache.delete(key);
    resultCache.set(key, entry);
    return entry.data;
  }
  return null;
}

function cacheSet(normUrl, variant, data) {
  if (!normUrl || !data || data.ok !== true) return;
  const key = `${normUrl}::${variant}`;
  resultCache.delete(key);
  resultCache.set(key, { data, time: Date.now(), domain: extractDomain(normUrl) });
  while (resultCache.size > MAX_CACHE_ENTRIES) {
    resultCache.delete(resultCache.keys().next().value);
  }
}

// 清除某網址在快取中的所有紀錄（完整 / 精簡兩種），讓下次分析不會被舊的快取結果擋住
function invalidateCacheForUrl(url) {
  const norm = normalizeCacheUrl(url);
  resultCache.delete(`${norm}::full`);
  resultCache.delete(`${norm}::compact`);
}

// 清除某網域（含子網域）的所有快取紀錄（加入黑名單後使用）
function invalidateCacheForDomain(domain) {
  if (!domain) return;
  const d = String(domain).toLowerCase();
  for (const [key, entry] of resultCache) {
    const host = entry.domain || '';
    if (host === d || host.endsWith(`.${d}`)) resultCache.delete(key);
  }
}

// -----------------------------------------------------------------------
// 自動偵測後端位址（結果快取 60 秒；同時多個請求共用一次探測）
// -----------------------------------------------------------------------
let apiBaseState = null;   // { base, reachable, manual, sig, time }
let apiBasePending = null; // { sig, promise }
let batchUnsupported = null; // { base, time }：該後端不支援 /predict/batch

function settingsSignature(settings) {
  return JSON.stringify([settings.apiBase || '', settings.apiPort, !!settings.useProdFallback]);
}

function invalidateApiBase() {
  apiBaseState = null;
}

async function probeBase(base, timeoutMs = LOCAL_PROBE_TIMEOUT_MS) {
  try {
    const res = await fetchWithTimeout(`${base}/health`, { cache: 'no-store' }, timeoutMs);
    if (res.ok) return base;
  } catch (e) {
    // 忽略，嘗試下一個
  }
  return null;
}

async function probeAll(settings) {
  // 優先嘗試使用者指定的埠
  const preferredPort = settings.apiPort;
  const ports = [preferredPort, ...DEFAULT_PORTS.filter(p => p !== preferredPort)];

  for (const port of ports) {
    const base = await probeBase(`http://127.0.0.1:${port}`);
    if (base) return { base, reachable: true };
  }

  // 本機都連不上，嘗試雲端備援（可在設定中關閉）
  if (settings.useProdFallback) {
    const prod = await probeBase(PROD_API_BASE, PROD_PROBE_TIMEOUT_MS);
    if (prod) return { base: prod, reachable: true };
  }

  // 都失敗就回傳預設值，讓呼叫端自然報錯並提示使用者
  return { base: `http://127.0.0.1:${preferredPort || 5500}`, reachable: false };
}

/**
 * 取得後端位址資訊 { base, reachable, manual }。
 *   - 使用者手動指定 apiBase：直接使用，不探測（reachable 為 null = 未知）
 *   - 自動偵測：成功結果快取 60 秒；全部失敗只快取 10 秒
 *   - force=true：忽略快取重新探測（連線失敗後、或使用者按重新整理）
 */
async function resolveApiBaseInfo({ force = false } = {}) {
  const settings = await getSettings();
  const sig = settingsSignature(settings);

  if (settings.apiBase) {
    return { base: String(settings.apiBase).trim().replace(/\/+$/, ''), reachable: null, manual: true };
  }

  if (!force && apiBaseState && apiBaseState.sig === sig) {
    const ttl = apiBaseState.reachable ? API_BASE_TTL_MS : API_BASE_NEGATIVE_TTL_MS;
    if (Date.now() - apiBaseState.time < ttl) return apiBaseState;
  }

  if (!apiBasePending || apiBasePending.sig !== sig) {
    const promise = probeAll(settings)
      .then(info => {
        const state = { ...info, manual: false, sig, time: Date.now() };
        apiBaseState = state;
        return state;
      })
      .finally(() => {
        if (apiBasePending && apiBasePending.promise === promise) apiBasePending = null;
      });
    apiBasePending = { sig, promise };
  }
  return apiBasePending.promise;
}

// 向下相容：回傳後端位址字串
async function resolveApiBase(options) {
  const info = await resolveApiBaseInfo(options);
  return info.base;
}

/**
 * 呼叫後端（自動帶入 apiBase）。
 *   - 連線失敗（後端關閉 / 換埠）：apiBase 快取失效 → 重新探測一次 → 重試一次
 *   - 逾時：apiBase 快取失效（下次重新探測），不重試，直接回報逾時
 * 回傳 { res, base }；HTTP 狀態碼由呼叫端判斷。
 */
async function apiRequest(path, { method = 'GET', body, timeoutMs = PREDICT_TIMEOUT_MS } = {}) {
  let info = await resolveApiBaseInfo();
  const init = { method, cache: 'no-store' };
  if (body !== undefined) {
    init.headers = { 'Content-Type': 'application/json' };
    init.body = JSON.stringify(body);
  }

  for (let attempt = 0; ; attempt++) {
    try {
      const res = await fetchWithTimeout(`${info.base}${path}`, init, timeoutMs);
      return { res, base: info.base };
    } catch (err) {
      invalidateApiBase();
      if (err && err.isTimeout) {
        throw new Error(`後端回應逾時（超過 ${Math.round(timeoutMs / 1000)} 秒），請確認後端服務是否正在執行（${info.base}）`);
      }
      if (attempt === 0) {
        if (info.manual) continue; // 手動指定位址：直接重試一次
        const next = await resolveApiBaseInfo({ force: true });
        if (next.reachable) {
          info = next;
          continue;
        }
      }
      throw new Error(`無法連線到後端（${info.base}）：${(err && err.message) || err}`);
    }
  }
}

// -----------------------------------------------------------------------
// /predict 呼叫（含本地快取）
// -----------------------------------------------------------------------
async function predictUrl(url) {
  const target = normalizeCacheUrl(url);
  if (!target) return makeError('URL 不可為空');

  const cached = cacheGet(target, ['full']);
  if (cached) return cached;

  const { res } = await apiRequest('/predict', {
    method: 'POST',
    body: { url: target, debug: false },
    timeoutMs: PREDICT_TIMEOUT_MS,
  });
  const data = await readApiJson(res);
  cacheSet(target, 'full', data); // ok:false 不會被快取
  return data;
}

// 單條精簡查詢（批次退回逐條模式時使用；舊版後端會忽略 compact 欄位，回完整結果亦可）
async function predictCompactSingle(normUrl) {
  const cached = cacheGet(normUrl, ['full', 'compact']);
  if (cached) return cached;

  const { res } = await apiRequest('/predict', {
    method: 'POST',
    body: { url: normUrl, debug: false, compact: true },
    timeoutMs: PREDICT_TIMEOUT_MS,
  });
  const data = await readApiJson(res);
  cacheSet(normUrl, 'compact', data);
  return data;
}

async function sendFeedback(url, label, aiScore, note = '') {
  // 回報同樣先去識別化，避免查詢參數中可能夾帶的個資／追蹤碼被寫進
  // 後端的 feedback.csv（見 normalizeCacheUrl 的說明）。
  const { res } = await apiRequest('/feedback', {
    method: 'POST',
    body: { url: normalizeCacheUrl(url), label, ai_score: aiScore, note },
    timeoutMs: WRITE_TIMEOUT_MS,
  });
  return readApiJson(res);
}

// -----------------------------------------------------------------------
// 立即黑名單覆寫（呼叫 main.py 的 /blocklist/add，執行期立即生效）
// -----------------------------------------------------------------------
async function addToBlocklist(domain, note = '') {
  const { res } = await apiRequest('/blocklist/add', {
    method: 'POST',
    body: { domain, note },
    timeoutMs: WRITE_TIMEOUT_MS,
  });
  return readApiJson(res);
}

// -----------------------------------------------------------------------
// 偵測歷史紀錄（供 popup 儀表板：已掃描 / 高風險 / 規則攔截 / 已回報）
// -----------------------------------------------------------------------
async function addHistoryEntry(url, data) {
  if (!data || !data.ok) return;

  const { scanHistory = [] } = await chrome.storage.local.get('scanHistory');

  const entry = {
    url,
    domain: extractDomain(url) || url,
    score: data.risk_score,
    level: data.risk_level,
    source: data.source,
    triggeredRules: data.triggered_rules || [],
    verdict: data.verdict || data.risk_label || '',
    degraded: !!data.degraded,
    time: Date.now(),
    reported: false,
  };
  if (data.degraded && data.degrade_reason) entry.degradeReason = String(data.degrade_reason);

  scanHistory.unshift(entry);
  const trimmed = scanHistory.slice(0, MAX_HISTORY);

  await chrome.storage.local.set({ scanHistory: trimmed });
}

async function markHistoryReported(url) {
  const { scanHistory = [] } = await chrome.storage.local.get('scanHistory');
  let changed = false;

  for (const entry of scanHistory) {
    if (entry.url === url && !entry.reported) {
      entry.reported = true;
      changed = true;
    }
  }

  if (changed) await chrome.storage.local.set({ scanHistory });
}

async function incrementFeedbackCount() {
  const { feedbackCount = 0 } = await chrome.storage.local.get('feedbackCount');
  await chrome.storage.local.set({ feedbackCount: feedbackCount + 1 });
}

async function getHistoryAndStats() {
  const { scanHistory = [], feedbackCount = 0 } = await chrome.storage.local.get([
    'scanHistory',
    'feedbackCount',
  ]);
  return { history: scanHistory, feedbackCount };
}

async function clearHistory() {
  await chrome.storage.local.set({ scanHistory: [], feedbackCount: 0 });
}

// -----------------------------------------------------------------------
// 批次檢查連結：優先 /predict/batch（compact），舊版後端退回逐條併發
// -----------------------------------------------------------------------
function isBatchUnsupported(base) {
  return !!(batchUnsupported &&
    batchUnsupported.base === base &&
    Date.now() - batchUnsupported.time < BATCH_UNSUPPORTED_TTL_MS);
}

/** 送一批（≤50）到 /predict/batch，回傳 { results } / { unsupported } / { fallback }。 */
async function requestBatch(chunk) {
  const { res, base } = await apiRequest('/predict/batch', {
    method: 'POST',
    body: { urls: chunk, debug: false, compact: true },
    timeoutMs: BATCH_TIMEOUT_MS,
  });

  if (BATCH_FALLBACK_STATUSES.has(res.status)) return { unsupported: true, base };

  let data;
  try {
    data = await readApiJson(res);
  } catch (e) {
    return { fallback: true, reason: e.message };
  }
  if (!data || data.ok === false || !data.results || typeof data.results !== 'object') {
    return { fallback: true, reason: data && data.error };
  }
  return { results: data.results };
}

/** 逐條併發查詢（限制併發數，避免瞬間打爆後端）。 */
async function fetchEachConcurrently(normUrls, out, concurrency = FALLBACK_CONCURRENCY) {
  let index = 0;

  async function worker() {
    while (index < normUrls.length) {
      const current = normUrls[index++];
      try {
        out[current] = normalizeResult(await predictCompactSingle(current));
      } catch (e) {
        out[current] = makeError(e.message);
      }
    }
  }

  const workers = Array.from({ length: Math.min(concurrency, normUrls.length) }, worker);
  await Promise.all(workers);
}

/** 對尚未快取的正規化網址取得 compact 結果：{ 正規化網址: 結果 }。 */
async function fetchCompactResults(normUrls) {
  const out = {};
  const leftovers = []; // 需要退回逐條查詢的網址

  for (let i = 0; i < normUrls.length; i += BATCH_CHUNK_SIZE) {
    const chunk = normUrls.slice(i, i + BATCH_CHUNK_SIZE);
    const info = await resolveApiBaseInfo();
    if (isBatchUnsupported(info.base)) {
      leftovers.push(...chunk);
      continue;
    }

    let outcome;
    try {
      outcome = await requestBatch(chunk);
    } catch (e) {
      // 連線層級失敗（apiRequest 內已重新探測並重試過一次）：逐條也只會再失敗，直接回報
      chunk.forEach(u => { out[u] = makeError(e.message); });
      continue;
    }

    if (outcome.unsupported) {
      batchUnsupported = { base: outcome.base, time: Date.now() };
      leftovers.push(...chunk);
      continue;
    }
    if (outcome.fallback) {
      leftovers.push(...chunk);
      continue;
    }

    for (const u of chunk) {
      const r = outcome.results[u];
      if (r === undefined) {
        leftovers.push(u); // 後端漏回某一條：補查
        continue;
      }
      const data = normalizeResult(r);
      out[u] = data;
      cacheSet(u, 'compact', data);
    }
  }

  if (leftovers.length) await fetchEachConcurrently(leftovers, out);
  return out;
}

/**
 * CHECK_LINKS：回傳 { 原樣輸入的 url: 結果 }（content.js 以原樣 href 對應錨點）。
 * 單筆失敗回 { ok:false, error, message }，不影響其他網址。
 */
async function checkUrlsBatch(urls) {
  const rawList = [...new Set((Array.isArray(urls) ? urls : []).filter(u => typeof u === 'string'))]
    .slice(0, MAX_CHECK_URLS);
  const results = {};
  const rawByNorm = new Map(); // 正規化網址 -> [原樣網址…]

  for (const raw of rawList) {
    const norm = normalizeCacheUrl(raw);
    if (!norm) {
      results[raw] = makeError('URL 不可為空');
      continue;
    }
    const cached = cacheGet(norm, ['full', 'compact']);
    if (cached) {
      results[raw] = cached;
      continue;
    }
    if (!rawByNorm.has(norm)) rawByNorm.set(norm, []);
    rawByNorm.get(norm).push(raw);
  }

  if (rawByNorm.size) {
    const fetched = await fetchCompactResults([...rawByNorm.keys()]);
    for (const [norm, raws] of rawByNorm) {
      const data = fetched[norm] || makeError('後端未回傳此網址的結果');
      raws.forEach(raw => { results[raw] = data; });
    }
  }

  return results;
}

// -----------------------------------------------------------------------
// 工具列徽章
// -----------------------------------------------------------------------
// 色票與網頁一致（danger / warn / safe 深一階），白字在三色底上對比皆 >= 4.5:1
const BADGE_COLOR = { high: '#be123c', medium: '#b45309', low: '#15803d' };
const BADGE_TEXT = { high: '!!', medium: '!', low: 'OK' }; // 徽章只吃純文字，不用裝飾符號

// chrome.action.* 在 MV3 回傳 Promise：分頁剛好關閉等情況會被拒絕，一併吞掉避免 unhandledrejection
function safeAction(fnName, details) {
  try {
    const fn = chrome.action && chrome.action[fnName];
    if (typeof fn !== 'function') return; // 不支援的瀏覽器版本略過
    Promise.resolve(fn.call(chrome.action, details)).catch(() => {});
  } catch (e) {
    // 忽略
  }
}

async function updateBadgeForTab(tabId, url) {
  if (!url || !/^https?:\/\//.test(url)) {
    safeAction('setBadgeText', { tabId, text: '' });
    return;
  }

  try {
    const data = await predictUrl(url);
    if (!data.ok) throw new Error(data.error || data.message || 'predict failed');

    safeAction('setBadgeBackgroundColor', {
      tabId,
      color: BADGE_COLOR[data.risk_level] || '#557689',
    });
    // 徽章文字用白色，在深一階的紅/橘/綠底上都清楚（setBadgeTextColor 的 Promise 一併 catch）
    safeAction('setBadgeTextColor', { tabId, color: '#ffffff' });
    safeAction('setBadgeText', {
      tabId,
      text: BADGE_TEXT[data.risk_level] || '',
    });
    safeAction('setTitle', {
      tabId,
      title: `Truth：風險分數 ${data.risk_score}／100（${data.risk_label || data.risk_level}）` +
        (data.degraded ? '［降級模式］' : ''),
    });

    // 自動檢查也記錄進歷史，讓「已掃描」統計反映實際瀏覽行為
    addHistoryEntry(url, data).catch(() => {});
  } catch (e) {
    safeAction('setBadgeText', { tabId, text: '' });
    safeAction('setTitle', { tabId, title: `Truth（無法取得判斷結果：${e.message}）` });
  }
}

chrome.tabs.onUpdated.addListener(async (tabId, changeInfo, tab) => {
  if (changeInfo.status !== 'complete' || !tab.url) return;
  const settings = await getSettings();
  if (!settings.autoCheckTab) return;
  updateBadgeForTab(tabId, tab.url);
});

chrome.tabs.onActivated.addListener(async ({ tabId }) => {
  const settings = await getSettings();
  if (!settings.autoCheckTab) return;
  try {
    const tab = await chrome.tabs.get(tabId);
    if (tab.url) updateBadgeForTab(tabId, tab.url);
  } catch (e) {
    // 分頁可能已關閉
  }
});

// 後端相關設定變更（含其他頁面直接寫 storage）時，讓 apiBase 快取失效
if (chrome.storage && chrome.storage.onChanged && chrome.storage.onChanged.addListener) {
  chrome.storage.onChanged.addListener((changes, areaName) => {
    if (areaName !== 'local' || !changes) return;
    if (API_SETTING_KEYS.some(k => Object.prototype.hasOwnProperty.call(changes, k))) {
      invalidateApiBase();
      batchUnsupported = null;
    }
  });
}

// -----------------------------------------------------------------------
// 訊息路由（popup.js / content.js 呼叫；協定與 v5.1 相同）
// -----------------------------------------------------------------------
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  (async () => {
    try {
      switch (message && message.type) {
        case 'PREDICT': {
          const data = await predictUrl(message.url);
          // 手動分析（popup 主動查詢）也記入歷史；網址用後端回傳的 url，
          // 與 popup 回報時送出的 lastUrl（= data.url）一致，「已回報」標記才對得上
          addHistoryEntry((data && data.url) || normalizeCacheUrl(message.url), data).catch(() => {});
          sendResponse({ ok: true, data });
          break;
        }
        case 'FEEDBACK': {
          const data = await sendFeedback(message.url, message.label, message.aiScore, message.note);

          if (data.ok) {
            incrementFeedbackCount().catch(() => {});
            markHistoryReported(message.url).catch(() => {});
            // 後端 /feedback 會清掉該網址的伺服器快取，本地快取一併清除
            // （data.url 可能已被後端正規化，與本地 key 不同 → 同網域一併清掉）
            invalidateCacheForUrl(message.url);
            invalidateCacheForDomain(extractDomain(message.url));
          }

          // 使用者回報「是詐騙」時：立即加入黑名單 + 清除該網域快取，
          // 讓下一次分析（無論自動徽章或手動分析）馬上反映最新判斷。
          // 注意：/blocklist/add 只在後端目前執行期有效，重啟伺服器後會消失。
          if (data.ok && message.label === 1) {
            const domain = extractDomain(message.url);
            if (domain) {
              try {
                await addToBlocklist(domain, `使用者於擴充功能回報（AI 分數 ${message.aiScore}）`);
              } catch (e) {
                // 黑名單加入失敗不影響原本的回報結果
              }
              invalidateCacheForDomain(domain);
            }
          }

          sendResponse({ ok: true, data });
          break;
        }
        case 'CHECK_LINKS': {
          const results = await checkUrlsBatch(message.urls || []);
          sendResponse({ ok: true, results });
          break;
        }
        case 'GET_SETTINGS': {
          const settings = await getSettings();
          sendResponse({ ok: true, settings });
          break;
        }
        case 'SET_SETTINGS': {
          await chrome.storage.local.set(message.settings || {});
          invalidateApiBase();
          batchUnsupported = null;
          sendResponse({ ok: true });
          break;
        }
        case 'GET_API_BASE': {
          const info = await resolveApiBaseInfo({ force: !!message.force });
          // reachable：true=探測成功、false=全部探測失敗、null=手動指定位址（未探測）
          sendResponse({ ok: true, apiBase: info.base, reachable: info.reachable });
          break;
        }
        case 'GET_HISTORY': {
          const { history, feedbackCount } = await getHistoryAndStats();
          sendResponse({ ok: true, history, feedbackCount });
          break;
        }
        case 'CLEAR_HISTORY': {
          await clearHistory();
          sendResponse({ ok: true });
          break;
        }
        default: {
          const msg = `未知訊息類型：${message && message.type}`;
          sendResponse({ ok: false, error: msg, message: msg });
        }
      }
    } catch (err) {
      const msg = (err && err.message) || String(err);
      sendResponse({ ok: false, error: msg, message: msg });
    }
  })();

  return true; // 保持通道開啟以支援非同步 sendResponse
});
