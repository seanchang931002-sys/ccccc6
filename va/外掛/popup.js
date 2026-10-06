/**
 * Truth Chrome Extension — popup.js
 * =============================================================================
 * 對齊 main.py v7.2 API 合約與 background.js v7.2 訊息協定（協定與 v5.1 起相容）。
 * v5.2：錯誤訊息一律讀 error || message；後端降級模式（degraded）在來源旁標示「降級模式」；
 *       狀態條依 GET_API_BASE 的 reachable 顯示是否真的連得上，重新整理鈕會強制重新探測。
 * 整合：
 *   - 主畫面即為分析輸入框（單一網址即時查詢，可一鍵帶入目前分頁網址）
 *   - 使用者回饋（是詐騙 / 是正常）→ /feedback，詐騙會即時進黑名單
 *   - 頁面連結掃描 / 頁面廣告掃描
 *   - 統計儀表板（已掃描 / 高風險 / 規則攔截 / 已回報）
 *   - 偵測歷史紀錄清單（自動檢查分頁 + 手動分析都會累積，存於 chrome.storage.local）
 * 圖示一律使用 icons.js（TruthIcons.svg），不使用 emoji。
 * =============================================================================
 */

'use strict';

const SOURCE_MAP = {
  trusted_domain: { icon: 'check-circle', text: '白名單直通' },
  blocklist: { icon: 'ban', text: '政府封鎖黑名單（165 / 刑事局）' },
  hybrid_ai_hard_rule: { icon: 'bot', text: 'AI + 硬規則綜合判斷' },
  ai_model: { icon: 'bot', text: 'AI 模型判定' },
  rules_only: { icon: 'alert-triangle', text: '模型未載入，改用硬規則判斷' },
};

// 後端回應 degraded=true（模型未載入 / 特徵版本不符而改走規則評分）時附加在來源旁的提示
const DEGRADED_TEXT = '降級模式';

/** 從回應物件取出錯誤訊息（後端與 background 的錯誤可能只帶 error 或 message 其一）。 */
function errText(obj, fallback) {
  return (obj && (obj.error || obj.message)) || fallback;
}

const SOURCE_SHORT = {
  trusted_domain: '白名單',
  blocklist: '黑名單',
  hybrid_ai_hard_rule: 'AI+硬規則',
  ai_model: 'AI 模型',
  rules_only: '硬規則',
};

const LEVEL_COLOR = { high: 'var(--danger)', medium: 'var(--warn)', low: 'var(--ok)' };
const LEVEL_ICON = { high: 'shield-alert', medium: 'alert-triangle', low: 'shield-check' };
const LEVEL_LABEL = { high: '高風險', medium: '中風險', low: '低風險' };

// ── DOM ──────────────────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);

const statusDot = $('status-dot');
const statusText = $('status-text');
const statusDetail = $('status-detail');

const refreshBtn = $('refreshBtn');

const tabActions = $('tabActions');
const tabStats = $('tabStats');
const panelActions = $('panelActions');
const panelStats = $('panelStats');

const scannedCountEl = $('scanned-count');
const detectedCountEl = $('detected-count');
const ruleCountEl = $('rule-count');
const feedbackCountEl = $('feedback-count');

const urlInput = $('urlInput');
const useTabBtn = $('useTabBtn');
const analyzeBtn = $('analyzeBtn');
const loading = $('loading');
const resultArea = $('resultArea');
const verdictBadge = $('verdictBadge');
const targetUrlText = $('targetUrlText');
const scoreLabel = $('scoreLabel');
const glowFill = $('glowFill');
const sourceBadge = $('sourceBadge');
const aiSummaryText = $('aiSummaryText');
const btnScam = $('btnScam');
const btnSafe = $('btnSafe');

const scanLinksBtn = $('scanLinksBtn');
const scanResult = $('scanResult');
const scanAdsBtn = $('scanAdsBtn');
const scanAdsResult = $('scanAdsResult');
const scanAdsSummary = $('scanAdsSummary');
const adAvgScore = $('adAvgScore');
const adGlowFill = $('adGlowFill');

const settingsToggle = $('settingsToggle');
const settingsRow = $('settingsRow');
const settingsPanel = $('settingsPanel');
const settingsLoadHint = $('settingsLoadHint');
const apiPortInput = $('apiPortInput');
const autoCheckTabToggle = $('autoCheckTabToggle');
const autoScanLinksToggle = $('autoScanLinksToggle');
const useProdFallbackToggle = $('useProdFallbackToggle');
const apiStatus = $('apiStatus');
const saveSettingsBtn = $('saveSettingsBtn');

const errorBox = $('errorBox');

const historyList = $('history-list');
const clearHistoryBtn = $('clear-history-btn');

let lastUrl = '';
let lastScore = 0;

// ── 圖示輔助 ─────────────────────────────────────────────────────────────
function icon(name, size, cls) {
  return TruthIcons.svg(name, { size: size || 16, cls: cls || '' });
}

/** 設定「圖示 + 文字」按鈕內容（文字一律 escape）。 */
function setIconText(el, name, text, size) {
  el.innerHTML = `${icon(name, size || 14)}<span>${escapeHtml(text)}</span>`;
}

/** 設定掃描結果備註；isError 時用錯誤樣式（紅字 + alert-circle 圖示）。 */
function setNote(el, text, isError) {
  el.classList.toggle('is-error', !!isError);
  if (isError) {
    el.innerHTML = `${icon('alert-circle', 14)}<span>${escapeHtml(text)}</span>`;
  } else {
    el.textContent = text;
  }
}

// ── 初始化 ───────────────────────────────────────────────────────────────
(async function init() {
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (tab && tab.url && /^https?:\/\//.test(tab.url)) urlInput.value = tab.url;
  } catch (e) { /* 忽略 */ }

  resetFeedbackButtons();

  // 讀不到設定時，設定面板維持預設值並顯示小提示
  try {
    const res = await sendMessage({ type: 'GET_SETTINGS' });
    if (!res || !res.ok) throw new Error('讀取設定失敗');
    const settings = res.settings;
    apiPortInput.value = settings.apiPort;
    autoCheckTabToggle.checked = !!settings.autoCheckTab;
    autoScanLinksToggle.checked = !!settings.autoScanLinks;
    useProdFallbackToggle.checked = !!settings.useProdFallback;
  } catch (e) {
    settingsLoadHint.hidden = false;
  }

  refreshStatusBar();
  refreshHistory();
})();

// ── 訊息輔助函式 ─────────────────────────────────────────────────────────
function sendMessage(msg) {
  return chrome.runtime.sendMessage(msg);
}

// ── 分段控制（動作 / 統計資料） ──────────────────────────────────────────
function showTab(name) {
  const isActions = name === 'actions';
  tabActions.classList.toggle('active', isActions);
  tabStats.classList.toggle('active', !isActions);
  tabActions.setAttribute('aria-selected', String(isActions));
  tabStats.setAttribute('aria-selected', String(!isActions));
  tabActions.tabIndex = isActions ? 0 : -1;
  tabStats.tabIndex = isActions ? -1 : 0;
  panelActions.hidden = !isActions;
  panelStats.hidden = isActions;
}

tabActions.addEventListener('click', () => showTab('actions'));
tabStats.addEventListener('click', () => showTab('stats'));
[tabActions, tabStats].forEach((btn) => {
  btn.addEventListener('keydown', (e) => {
    if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
    const toStats = btn === tabActions;
    showTab(toStats ? 'stats' : 'actions');
    (toStats ? tabStats : tabActions).focus();
  });
});

// ── 頁尾狀態條：顯示目前連線的後端 ──────────────────────────────────────
function setStatus(state, text, detail) {
  statusDot.className = `status-dot ${state}`;
  statusText.textContent = text;
  statusDetail.textContent = detail || '';
}

async function refreshStatusBar(force) {
  setStatus('pending', '偵測後端連線中…', '');
  try {
    const { ok, apiBase, reachable } = await sendMessage({ type: 'GET_API_BASE', force: !!force });
    if (ok && apiBase && reachable === false) {
      setStatus('err', '無法連線後端', apiBase.replace(/^https?:\/\//, ''));
    } else if (ok && apiBase) {
      setStatus('ok', '後端已連線', apiBase.replace(/^https?:\/\//, ''));
    } else {
      throw new Error('no api base');
    }
  } catch (e) {
    setStatus('err', '無法連線後端', '');
  }
}

refreshBtn.addEventListener('click', () => {
  refreshBtn.classList.remove('spinning');
  void refreshBtn.offsetWidth;
  refreshBtn.classList.add('spinning');
  refreshStatusBar(true);
  if (!settingsPanel.hidden) refreshApiStatus();
});

// ── 展開區塊輔助 ─────────────────────────────────────────────────────────
function setExpanded(row, body, open) {
  body.hidden = !open;
  row.setAttribute('aria-expanded', String(open));
}

// ── 分析 ─────────────────────────────────────────────────────────────────
analyzeBtn.addEventListener('click', () => analyze());
urlInput.addEventListener('keydown', e => { if (e.key === 'Enter') analyze(); });

// 「使用目前分頁」：只把目前分頁網址填入輸入框，不送出分析
useTabBtn.addEventListener('click', async () => {
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (tab && tab.url) urlInput.value = tab.url;
  } catch (e) { /* 忽略 */ }
  urlInput.focus();
});

async function analyze() {
  if (analyzeBtn.disabled) return; // 分析進行中，忽略重複觸發（例如連按 Enter）
  const raw = urlInput.value.trim();
  if (!raw) {
    urlInput.focus();
    return;
  }

  hideError();
  resultArea.hidden = true;
  loading.hidden = false;
  analyzeBtn.disabled = true;

  try {
    const response = await sendMessage({ type: 'PREDICT', url: raw, debug: false });
    if (!response || !response.ok) throw new Error(errText(response, '分析失敗'));

    const data = response.data;
    if (!data || !data.ok) throw new Error(errText(data, '後端回傳失敗'));

    renderResult(data);
    lastUrl = data.url;
    lastScore = Number(data.risk_score) || 0;

    refreshStatusBar();
    refreshHistory();
  } catch (err) {
    showError(err.message);
  } finally {
    loading.hidden = true;
    analyzeBtn.disabled = false;
  }
}

function renderResult(data) {
  const level = data.risk_level;
  const scoreNum = Number(data.risk_score) || 0;
  const color = LEVEL_COLOR[level] || 'var(--text)';

  resultArea.hidden = false;

  verdictBadge.innerHTML =
    `${icon(LEVEL_ICON[level] || 'help-circle', 22)}<span>${escapeHtml(data.verdict || data.risk_label || '')}</span>`;
  verdictBadge.style.color = color;

  targetUrlText.innerHTML = `${icon('link', 14)}<span>${escapeHtml(data.url)}</span>`;
  targetUrlText.title = data.url || '';

  scoreLabel.textContent = `${scoreNum}`;
  scoreLabel.style.color = color;

  glowFill.style.width = '0%';
  requestAnimationFrame(() => {
    glowFill.style.width = `${Math.min(100, Math.max(0, scoreNum))}%`;
    glowFill.style.background = LEVEL_COLOR[level] || 'var(--ok)';
  });

  const src = SOURCE_MAP[data.source];
  const degradedSuffix = data.degraded ? `・${DEGRADED_TEXT}` : '';
  if (src) {
    sourceBadge.innerHTML = `${icon(src.icon, 14)}<span>${escapeHtml(src.text + degradedSuffix)}</span>`;
  } else if (data.source) {
    sourceBadge.innerHTML = `${icon('info', 14)}<span>${escapeHtml(data.source + degradedSuffix)}</span>`;
  } else if (data.degraded) {
    sourceBadge.innerHTML = `${icon('alert-triangle', 14)}<span>${escapeHtml(DEGRADED_TEXT)}</span>`;
  } else {
    sourceBadge.innerHTML = '';
  }
  sourceBadge.title = data.degraded
    ? `後端目前為降級模式${data.degrade_reason ? `：${data.degrade_reason}` : ''}`
    : '';

  const reasons = data.reasons || [];
  let text = `詐騙風險指數 <strong>${scoreNum}／100</strong>，判定為 <strong>${escapeHtml(data.risk_label || level)}</strong>。`;

  const highReasons = reasons.filter(r => r.level === 'high').map(r => r.message);
  const mediumReasons = reasons.filter(r => r.level === 'medium').map(r => r.message);

  if (highReasons.length) {
    text += `<br><br><b>主要風險因素：</b><br>` + highReasons.map(m => `・${escapeHtml(m)}`).join('<br>');
  }
  if (mediumReasons.length) {
    text += `<br><br><b>次要疑慮：</b><br>` + mediumReasons.map(m => `・${escapeHtml(m)}`).join('<br>');
  }
  if (!reasons.length) {
    text += `<br><br>目前未觸發任何風險特徵。`;
  }
  if (data.suggestion) {
    text += `<br><br><strong>${escapeHtml(data.suggestion)}</strong>`;
  }

  aiSummaryText.innerHTML = text;
  resetFeedbackButtons();
}

// ── 回饋（主結果卡片） ───────────────────────────────────────────────────
btnScam.addEventListener('click', () => sendFeedback(lastUrl, lastScore, 1, btnScam, btnSafe));
btnSafe.addEventListener('click', () => sendFeedback(lastUrl, lastScore, 0, btnSafe, btnScam));

async function sendFeedback(url, score, label, clickedBtn, otherBtn) {
  if (!url) return;

  btnScam.disabled = true;
  btnSafe.disabled = true;

  try {
    const response = await sendMessage({
      type: 'FEEDBACK',
      url,
      label,
      aiScore: score / 100,
      note: '',
    });

    if (!response || !response.ok || !response.data || !response.data.ok) {
      throw new Error(errText(response && response.data, '') || errText(response, '回報失敗'));
    }

    setIconText(clickedBtn, 'check-circle', '已回報，感謝！');
    clickedBtn.classList.add('done');

    refreshHistory();
  } catch (err) {
    btnScam.disabled = false;
    btnSafe.disabled = false;
    setIconText(clickedBtn, 'alert-circle', '回報失敗，請重試');
    clickedBtn.classList.add('fail');
    setTimeout(resetFeedbackButtons, 2000);
  }
}

function resetFeedbackButtons() {
  btnScam.disabled = false;
  btnSafe.disabled = false;
  [btnScam, btnSafe].forEach(b => b.classList.remove('done', 'fail'));
  setIconText(btnScam, 'flag', '是詐騙');
  setIconText(btnSafe, 'check-circle', '是正常');
}

// ── 掃描頁面連結 ─────────────────────────────────────────────────────────
scanLinksBtn.addEventListener('click', async () => {
  scanLinksBtn.disabled = true;
  setNote(scanResult, '掃描中，請稍候...');

  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (!tab || !tab.id) throw new Error('找不到目前分頁');

    const response = await chrome.tabs.sendMessage(tab.id, { type: 'SCAN_PAGE_LINKS' });
    if (!response || !response.ok) {
      throw new Error(errText(response, '此頁面可能不支援內容腳本注入'));
    }

    const { checked, high, medium } = response.summary;
    if (checked === 0) {
      setNote(scanResult, '此頁面未偵測到可分析的連結。');
    } else {
      setNote(scanResult,
        `已檢查 ${checked} 條連結，發現 ${high} 條高風險、${medium} 條中風險連結` +
        (high + medium > 0 ? '（已在頁面上以外框與圖示標示）。' : '，目前看起來安全。'));
    }

    refreshHistory();
  } catch (err) {
    setNote(scanResult, `掃描失敗：${err.message}`, true);
  } finally {
    scanLinksBtn.disabled = false;
  }
});

// ── 掃描頁面廣告風險 ─────────────────────────────────────────────────────
scanAdsBtn.addEventListener('click', async () => {
  scanAdsBtn.disabled = true;
  scanAdsSummary.hidden = true;
  setNote(scanAdsResult, '掃描中，請稍候...');

  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (!tab || !tab.id) throw new Error('找不到目前分頁');

    const response = await chrome.tabs.sendMessage(tab.id, { type: 'SCAN_PAGE_ADS' });
    if (!response || !response.ok) {
      throw new Error(errText(response, '此頁面可能不支援內容腳本注入'));
    }

    const { checked, high, medium, low } = response.summary;
    const avgScore = Number(response.summary.avgScore) || 0;

    if (checked === 0) {
      setNote(scanAdsResult, '此頁面未偵測到可分析的廣告（iframe 廣告或標記為 ad/sponsor 的區塊）。');
      return;
    }

    const adColor = LEVEL_COLOR[avgScore >= 70 ? 'high' : avgScore >= 40 ? 'medium' : 'low'];
    scanAdsSummary.hidden = false;
    adAvgScore.textContent = `${avgScore} / 100`;
    adAvgScore.style.color = adColor;
    adGlowFill.style.width = '0%';
    requestAnimationFrame(() => {
      adGlowFill.style.width = `${Math.min(100, Math.max(0, avgScore))}%`;
      adGlowFill.style.background = adColor;
    });

    setNote(scanAdsResult,
      `已檢查 ${checked} 則廣告：${high} 則高風險、${medium} 則中風險、${low} 則未見異常` +
      (high + medium > 0 ? '（已在頁面上以虛線外框與角標標示）。' : '，整體看起來安全。'));

    refreshHistory();
  } catch (err) {
    setNote(scanAdsResult, `掃描失敗：${err.message}`, true);
  } finally {
    scanAdsBtn.disabled = false;
  }
});

// ── 設定面板 ─────────────────────────────────────────────────────────────
async function refreshApiStatus() {
  apiStatus.textContent = '偵測後端連線中...';
  try {
    const { ok, apiBase, reachable } = await sendMessage({ type: 'GET_API_BASE' });
    if (ok && reachable === false) {
      apiStatus.textContent = `無法連線後端（已嘗試本機各埠與備援位址，預設：${apiBase}）`;
    } else {
      apiStatus.textContent = ok ? `目前使用後端：${apiBase}` : '無法偵測後端位址';
    }
  } catch (e) {
    apiStatus.textContent = '無法偵測後端位址';
  }
}

function toggleSettings() {
  // 在「統計資料」頁按齒輪：切回動作頁並確保設定面板是展開的（不是把看不見的面板收起來）
  const open = panelActions.hidden || settingsPanel.hidden;
  if (open) showTab('actions');
  setExpanded(settingsRow, settingsPanel, open);
  settingsToggle.setAttribute('aria-expanded', String(open));
  if (open) {
    refreshApiStatus();
    settingsPanel.scrollIntoView({ block: 'nearest' });
  }
}

settingsToggle.addEventListener('click', toggleSettings);
settingsRow.addEventListener('click', toggleSettings);

saveSettingsBtn.addEventListener('click', async () => {
  const settings = {
    apiPort: parseInt(apiPortInput.value, 10) || 5500,
    apiBase: '', // 清空自動偵測快取，改用新埠號重新偵測
    autoCheckTab: autoCheckTabToggle.checked,
    autoScanLinks: autoScanLinksToggle.checked,
    useProdFallback: useProdFallbackToggle.checked,
  };

  try {
    const res = await sendMessage({ type: 'SET_SETTINGS', settings });
    if (!res || !res.ok) throw new Error('save failed');
  } catch (e) {
    setIconText(saveSettingsBtn, 'alert-circle', '儲存失敗，請重試', 16);
    setTimeout(() => { saveSettingsBtn.textContent = '儲存設定'; }, 2000);
    return;
  }

  settingsLoadHint.hidden = true;
  setIconText(saveSettingsBtn, 'check', '已儲存', 16);
  setTimeout(() => { saveSettingsBtn.textContent = '儲存設定'; }, 1500);

  refreshStatusBar();
});

// ── 偵測歷史紀錄 ─────────────────────────────────────────────────────────
async function refreshHistory() {
  const response = await sendMessage({ type: 'GET_HISTORY' });
  if (!response || !response.ok) return;

  renderStats(response.history, response.feedbackCount);
  renderHistoryList(response.history);
}

function renderStats(history, feedbackCount) {
  scannedCountEl.textContent = history.length;

  const highCount = history.filter(h => h.level === 'high').length;
  detectedCountEl.textContent = highCount;

  const ruleCount = history.filter(h =>
    h.source === 'blocklist' || h.source === 'hybrid_ai_hard_rule' || h.source === 'rules_only'
  ).length;
  ruleCountEl.textContent = ruleCount;

  feedbackCountEl.textContent = feedbackCount || 0;
}

function emptyStateHtml(message) {
  return `
    <div class="empty-state">
      ${icon('shield-check', 32, 'empty-icon')}
      <p>${escapeHtml(message)}</p>
    </div>`;
}

function renderHistoryList(history) {
  // 只顯示中/高風險紀錄，最多 30 筆，避免清單過長
  const risky = history.filter(h => h.level !== 'low').slice(0, 30);

  if (risky.length === 0) {
    historyList.innerHTML = emptyStateHtml('尚無中／高風險偵測紀錄');
    return;
  }

  historyList.innerHTML = '';

  for (const entry of risky) {
    const card = document.createElement('div');
    card.className = 'record-card';

    const level = LEVEL_LABEL[entry.level] ? entry.level : 'low';
    const sourceLabel = (SOURCE_SHORT[entry.source] || entry.source || '未知') +
      (entry.degraded ? `・${DEGRADED_TEXT}` : '');
    const score = Math.min(100, Math.max(0, Number(entry.score) || 0));
    const reportedAttr = entry.reported ? 'disabled' : '';

    card.innerHTML = `
      <span class="record-level ${level}">${icon(LEVEL_ICON[level], 22)}</span>
      <div class="record-main">
        <div class="record-domain" title="${escapeHtml(entry.url)}">${escapeHtml(entry.domain || entry.url)}</div>
        <div class="record-meta">
          <span class="dot ${level}"></span>
          <span class="record-level-label ${level}">${LEVEL_LABEL[level]} ${score}</span>
          <span>${escapeHtml(formatRelativeTime(entry.time))}</span>
          <span class="source-badge">${escapeHtml(sourceLabel)}</span>
        </div>
      </div>
      <div class="record-actions">
        <button class="btn-locate" type="button" title="定位（開啟此網址）" aria-label="定位（開啟此網址）">${icon('external-link', 16)}</button>
        <button class="btn-copy" type="button" title="複製網址" aria-label="複製網址">${icon('copy', 16)}</button>
        <button class="btn-report-scam" type="button" title="回報：確實是詐騙" aria-label="回報：確實是詐騙" ${reportedAttr}>${icon('flag', 16)}</button>
        <button class="btn-report-safe" type="button" title="回報：其實是正常網站（誤判）" aria-label="回報：其實是正常網站（誤判）" ${reportedAttr}>${icon('check-circle', 16)}</button>
      </div>
    `;

    card.querySelector('.btn-locate').addEventListener('click', () => {
      chrome.tabs.create({ url: entry.url });
    });

    card.querySelector('.btn-copy').addEventListener('click', async (e) => {
      const btn = e.currentTarget;
      try {
        await navigator.clipboard.writeText(entry.url);
        btn.innerHTML = icon('check', 16);
        btn.classList.add('ok');
        btn.title = '已複製';
        btn.setAttribute('aria-label', '已複製');
        setTimeout(() => {
          btn.innerHTML = icon('copy', 16);
          btn.classList.remove('ok');
          btn.title = '複製網址';
          btn.setAttribute('aria-label', '複製網址');
        }, 1200);
      } catch (err) { /* 忽略 */ }
    });

    const scamBtn = card.querySelector('.btn-report-scam');
    const safeBtn = card.querySelector('.btn-report-safe');

    scamBtn.addEventListener('click', async () => {
      scamBtn.disabled = true;
      safeBtn.disabled = true;
      await sendMessage({ type: 'FEEDBACK', url: entry.url, label: 1, aiScore: entry.score / 100, note: '' });
      refreshHistory();
    });

    safeBtn.addEventListener('click', async () => {
      scamBtn.disabled = true;
      safeBtn.disabled = true;
      await sendMessage({ type: 'FEEDBACK', url: entry.url, label: 0, aiScore: entry.score / 100, note: '' });
      refreshHistory();
    });

    historyList.appendChild(card);
  }
}

clearHistoryBtn.addEventListener('click', async () => {
  await sendMessage({ type: 'CLEAR_HISTORY' });
  refreshHistory();
});

// ── 錯誤顯示 ─────────────────────────────────────────────────────────────
function showError(message) {
  errorBox.innerHTML = `${icon('alert-triangle', 16)}<span>${escapeHtml(message)}</span>`;
  errorBox.hidden = false;
}
function hideError() {
  errorBox.hidden = true;
  errorBox.textContent = '';
}

// ── 工具函式 ─────────────────────────────────────────────────────────────
function escapeHtml(t) {
  const d = document.createElement('div');
  d.textContent = t == null ? '' : String(t);
  return d.innerHTML.replace(/"/g, '&quot;');
}

function formatRelativeTime(ts) {
  if (!ts) return '';
  const diffSec = Math.max(0, Math.floor((Date.now() - ts) / 1000));
  if (diffSec < 60) return '剛剛';
  const diffMin = Math.floor(diffSec / 60);
  if (diffMin < 60) return `${diffMin} 分鐘前`;
  const diffHr = Math.floor(diffMin / 60);
  if (diffHr < 24) return `${diffHr} 小時前`;
  const diffDay = Math.floor(diffHr / 24);
  if (diffDay < 7) return `${diffDay} 天前`;
  const d = new Date(ts);
  return `${d.getMonth() + 1}/${d.getDate()}`;
}
