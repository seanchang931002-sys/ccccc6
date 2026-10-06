/**
 * Truth Chrome Extension — content.js
 * =============================================================================
 * 在網頁中掃描 <a> 連結，交給 background.js 批次呼叫 /predict，
 * 並在高風險 / 中風險連結旁加上視覺標示（不會修改連結本身的行為）。
 *
 * 兩種觸發時機：
 * 1. 若設定「autoScanLinks」為 true，頁面載入完成後自動掃描一次。
 * 2. popup.js 送出 { type: 'SCAN_PAGE_LINKS' } 訊息時，手動觸發掃描。
 *
 * 新增：頁面廣告掃描（collectPageAds）
 *   - 第三方廣告 iframe（doubleclick / googlesyndication / taboola ... 等網域）
 *   - Google Publisher Tag（<ins class="adsbygoogle">）
 *   - 帶有廣告標記的容器（class/id 含 ad- / sponsor / advertisement，
 *     或 aria-label="Advertisement"／"廣告"），取其內部 <a href> 落地頁連結
 *   同樣送去 background.js 的 /predict 批次檢查，依風險等級標示「廣告」
*   （高風險紅色／中風險黃色／未見異常綠色）。
 *
 * v5.2：background.js 的 CHECK_LINKS 改走 /predict/batch（compact，舊版後端自動退回逐條）。
 *   回應格式不變：{ ok, results: { 原樣送出的 url: { ok, risk_level, risk_score, ... } } }，
 *   單筆失敗為 { ok:false, error, message }；這裡只讀 ok / risk_level / risk_score，與 compact 相容。
 * =============================================================================
 */

'use strict';

// 圖示名稱對應 icons.js（TruthIcons）；色票與網頁一致（danger / warn / safe）。
// 連結徽章：白底 + 風險色圖示；廣告角標：實心風險色底 + 白字（白字對比皆 >= 4.5:1，
// 安全色用深一階的 #15803d 才能讓白字達標）。
const BADGE_STYLE = {
  high: { icon: 'shield-alert', color: '#be123c', label: '高風險' },
  medium: { icon: 'alert-triangle', color: '#b45309', label: '中風險' },
};

const AD_BADGE_STYLE = {
  high: { icon: 'shield-alert', color: '#be123c', label: '廣告／高風險' },
  medium: { icon: 'alert-triangle', color: '#b45309', label: '廣告／中風險' },
  low: { icon: 'shield-check', color: '#15803d', label: '廣告／未見異常' },
};

let alreadyScanned = false;
let alreadyScannedAds = false;

// 建立圖示並以 CSSOM（!important）鎖定關鍵樣式，避免宿主頁的全域 svg 規則
// （如 svg{fill:currentColor}、svg{display:block;width:100%}）讓圖示變形。
// 圖示庫不可用時回傳 null，由呼叫端改用純文字替代；不會拋出例外。
function createLockedIcon(name, px) {
  try {
    if (!window.TruthIcons || typeof window.TruthIcons.el !== 'function') return null;
    const svg = window.TruthIcons.el(name, { size: px });
    if (!svg) return null;
    const sw = svg.getAttribute('stroke-width') || '1.75';
    const lock = (node, props) => {
      for (const [k, v] of Object.entries(props)) node.style.setProperty(k, v, 'important');
    };
    lock(svg, {
      fill: 'none',
      stroke: 'currentColor',
      'stroke-width': sw,
      'stroke-linecap': 'round',
      'stroke-linejoin': 'round',
      width: `${px}px`,
      height: `${px}px`,
      'min-width': `${px}px`,
      'max-width': `${px}px`,
      'min-height': `${px}px`,
      'max-height': `${px}px`,
      display: 'inline-block',
      flex: 'none',
      'vertical-align': 'middle',
      margin: '0',
      padding: '0',
      opacity: '1',
      visibility: 'visible',
      transform: 'none',
      color: 'inherit', // 圖示顏色一律取自外層徽章（風險色），不讓宿主頁的 .ic 之類 class 改色
    });
    for (const child of Array.from(svg.children)) {
      lock(child, {
        fill: child.getAttribute('fill') || 'none',
        stroke: 'currentColor',
        'stroke-width': sw,
        'stroke-linecap': 'round',
        'stroke-linejoin': 'round',
      });
    }
    return svg;
  } catch (e) {
    return null;
  }
}

function collectPageLinks(limit = 40) {
  const anchors = Array.from(document.querySelectorAll('a[href^="http"]'));
  const seen = new Set();
  const targets = [];

  for (const a of anchors) {
    const href = a.href;
    if (!href || seen.has(href)) continue;
    seen.add(href);
    targets.push(href);
    if (targets.length >= limit) break;
  }

  return targets;
}

// -----------------------------------------------------------------------
// 廣告偵測
// -----------------------------------------------------------------------

// 已知廣告聯播網網域（iframe src 命中即視為廣告來源）
const AD_IFRAME_HOST_PATTERNS = [
  /doubleclick\.net$/i,
  /googlesyndication\.com$/i,
  /googleadservices\.com$/i,
  /google\.com$/i, // 搭配 pagead 路徑判斷，見下方 isAdIframe
  /amazon-adsystem\.com$/i,
  /taboola\.com$/i,
  /outbrain\.com$/i,
  /criteo\.com$/i,
  /adnxs\.com$/i,
  /adsafeprotected\.com$/i,
  /media\.net$/i,
  /pubmatic\.com$/i,
  /rubiconproject\.com$/i,
  /openx\.net$/i,
  /smartadserver\.com$/i,
  /sharethrough\.com$/i,
  /indexexchange\.com$/i,
  /casalemedia\.com$/i,
  /adform\.net$/i,
  /adsrvr\.org$/i,
  /yahoo\.com$/i, // 搭配 ads 路徑判斷，見下方 isAdIframe
  /33across\.com$/i,
];

// 容器 class/id/aria-label 命中即視為廣告區塊
const AD_CONTAINER_SELECTOR = [
  // Google 廣告
  'ins.adsbygoogle',
  '[id^="google_ads_iframe"]',
  '[id^="div-gpt-ad"]',
  // 一般廣告版位命名慣例
  '[class*="ad-slot" i]',
  '[class*="ad-banner" i]',
  '[class*="ad-container" i]',
  '[class*="ad-wrapper" i]',
  '[class*="ad_unit" i]',
  '[class*="adunit" i]',
  '[class*="advertisement" i]',
  '[class*="advertorial" i]',
  '[class*="sponsor" i]',
  '[data-ad]',
  '[data-testid*="ad" i]',
  '[aria-label="Advertisement"]',
  '[aria-label="廣告"]',
  // 新聞網站常見「原生廣告／業配內容」聯播網（Taboola / Outbrain / Revcontent / MGID / Dianomi / Nativo）
  '[id*="taboola" i]',
  '[class*="taboola" i]',
  '[id*="outbrain" i]',
  '[class*="outbrain" i]',
  '[class*="trc_rbox" i]',
  '[class*="revcontent" i]',
  '[class*="rc-widget" i]',
  '[class*="mgid" i]',
  '[id*="mgid" i]',
  '[class*="dianomi" i]',
  '[id*="dianomi" i]',
  '[class*="nativo" i]',
  '[class*="native-ad" i]',
  '[class*="content-ad" i]',
  '[class*="sponsored-content" i]',
  '[class*="sponsored-post" i]',
  '[class*="promoted-content" i]',
  // 中文新聞網站常見的「業配／贊助／廣告特輯」標記
  '[class*="business-ad" i]',
  '[class*="paid-content" i]',
].join(',');

function isAdIframe(iframe) {
  let host = '';
  let path = '';
  try {
    const u = new URL(iframe.src, location.href);
    host = u.hostname;
    path = u.pathname;
  } catch (e) {
    return false;
  }
  if (host.includes('google.com') && !/pagead|adservice/i.test(path)) return false;
  if (host.includes('yahoo.com') && !/ads|adserver|serving/i.test(path)) return false;
  return AD_IFRAME_HOST_PATTERNS.some(re => re.test(host));
}

// 從廣告容器（非 iframe）內找出實際的落地頁連結
function extractAdLandingUrl(container) {
  const a = container.querySelector('a[href^="http"]');
  return a ? a.href : null;
}

// -----------------------------------------------------------------------
// 原生贊助貼文偵測（Facebook / Instagram / Threads 等社群平台）
// 這類廣告沒有 iframe，也沒有 ad-/sponsor class（class 是編譯後的亂碼），
// 唯一線索是貼文上顯示的「贊助」／「Sponsored」文字標籤。
// -----------------------------------------------------------------------
const SPONSORED_LABEL_PATTERNS = [
  // Facebook / Instagram / Threads（繁中・簡中・英文）
  /^贊助$/,
  /^已贊助$/,
  /^赞助$/,
  /^Sponsored$/i,
  /^Paid partnership.*$/i,
  /^配对广告$/,
  /^配對廣告$/,
  // 新聞網站常見的業配／廣編／推廣標示
  /^廣告$/,
  /^广告$/,
  /^業配$/,
  /^业配$/,
  /^廣編特輯$/,
  /^廣告特輯$/,
  /^贊助內容$/,
  /^赞助内容$/,
  /^Promoted$/i,
  /^Paid Content$/i,
  /^Paid Post$/i,
  /^Advertisement$/i,
  /^Advertiser Content$/i,
];

function findSponsoredLabelNodes(limit = 25) {
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, {
    acceptNode(node) {
      // 排除外掛自己插入的角標文字（例如「廣告」），避免二次掃描時自我命中
      if (node.parentElement && node.parentElement.closest('[data-veriad-ad-badge]')) {
        return NodeFilter.FILTER_SKIP;
      }
      const text = node.nodeValue.trim();
      if (!text || text.length > 20) return NodeFilter.FILTER_SKIP;
      return SPONSORED_LABEL_PATTERNS.some(re => re.test(text))
        ? NodeFilter.FILTER_ACCEPT
        : NodeFilter.FILTER_SKIP;
    },
  });
  const nodes = [];
  let n;
  while ((n = walker.nextNode()) && nodes.length < limit) nodes.push(n);
  return nodes;
}

// Instagram／Facebook 有時不會把「贊助」顯示成一般文字，而是放在
// aria-label（畫面上看不到，但螢幕閱讀器讀得到）。這裡額外掃一輪。
function findSponsoredAriaElements(limit = 25) {
  const candidates = document.querySelectorAll('[aria-label]');
  const found = [];
  for (const el of candidates) {
    if (found.length >= limit) break;
    const label = (el.getAttribute('aria-label') || '').trim();
    if (!label || label.length > 20) continue;
    if (SPONSORED_LABEL_PATTERNS.some(re => re.test(label))) found.push(el);
  }
  return found;
}

// 從「贊助」標籤節點往上找出整則貼文的容器
// 涵蓋 role="article"（多數平台）、以及 Facebook/Instagram 常見的
// data-pagelet（Feed 資料區塊）與 data-testid post 容器命名
function findPostContainer(node) {
  let el = node.nodeType === Node.TEXT_NODE ? node.parentElement : node;
  let hops = 0;
  while (el && hops < 15) {
    if (el.matches?.(
      '[role="article"], article, [data-pagelet*="FeedUnit" i], ' +
      '[data-testid*="post" i], [data-testid*="story" i]'
    )) return el;
    el = el.parentElement;
    hops++;
  }
  return null;
}

// 貼文容器內找出「廣告要導向的外部連結」（排除同網域的個人檔案/留言等連結）
function extractNativeAdLandingUrl(container) {
  const anchors = Array.from(container.querySelectorAll('a[href^="http"]'));
  for (const a of anchors) {
    try {
      const u = new URL(a.href, location.href);
      if (u.hostname && u.hostname !== location.hostname) return a.href;
    } catch (e) {
      // 忽略無效網址
    }
  }
  return null;
}

function collectNativeSponsoredAds(limit = 20) {
  const found = [];
  const seenContainers = new Set();

  // 來源 1：可見的「贊助」文字節點
  // 來源 2：藏在 aria-label 裡的贊助標記（IG 常見做法）
  const sponsoredMarkers = [...findSponsoredLabelNodes(), ...findSponsoredAriaElements()];

  for (const marker of sponsoredMarkers) {
    if (found.length >= limit) break;
    const container = findPostContainer(marker);
    if (!container || seenContainers.has(container)) continue;
    seenContainers.add(container);

    const landingUrl = extractNativeAdLandingUrl(container);
    if (!landingUrl) continue; // 站內贊助貼文找不到外部連結就跳過，避免誤判

    found.push({ el: container, kind: 'native', targetUrl: landingUrl, checkUrl: landingUrl });
  }

  return found;
}

/**
 * 掃描頁面廣告，回傳 [{ el, kind: 'iframe'|'container'|'native', targetUrl, checkUrl }]
 * checkUrl 是送去 /predict 判斷風險用的網址：
 *   - iframe 廣告：用 iframe.src 本身（含廣告聯播網網域，可反映風險）
 *   - 容器廣告：優先用內部落地頁連結；找不到就跳過（避免誤判）
 *   - 原生贊助貼文（native）：用貼文內第一個外部連結；找不到就跳過
 */
function collectPageAds(limit = 30) {
  const found = [];
  const seenEls = new Set();

  // 1) 廣告 iframe
  document.querySelectorAll('iframe[src]').forEach(iframe => {
    if (found.length >= limit) return;
    if (!isAdIframe(iframe)) return;
    if (seenEls.has(iframe)) return;
    seenEls.add(iframe);
    found.push({ el: iframe, kind: 'iframe', targetUrl: iframe.src, checkUrl: iframe.src });
  });

  // 2) 有廣告標記的容器
  document.querySelectorAll(AD_CONTAINER_SELECTOR).forEach(container => {
    if (found.length >= limit) return;
    if (seenEls.has(container)) return;
    // 避免容器內已計算過的 iframe 被重複標記
    if (container.querySelector('iframe[src]') &&
        Array.from(container.querySelectorAll('iframe[src]')).some(f => seenEls.has(f))) {
      return;
    }
    const landingUrl = extractAdLandingUrl(container);
    if (!landingUrl) return; // 沒有可檢查的連結就跳過
    seenEls.add(container);
    found.push({ el: container, kind: 'container', targetUrl: landingUrl, checkUrl: landingUrl });
  });

  // 3) 原生贊助貼文（Facebook / Instagram 等，無 iframe 也無 ad class）
  if (found.length < limit) {
    for (const ad of collectNativeSponsoredAds(limit - found.length)) {
      if (seenEls.has(ad.el)) continue;
      seenEls.add(ad.el);
      found.push(ad);
    }
  }

  return found;
}

function markAd(el, riskLevel, riskScore, targetUrl, kind = 'ad') {
  const style = AD_BADGE_STYLE[riskLevel];
  if (!style) return;
  if (el.dataset.veriadAdMarked) return;
  el.dataset.veriadAdMarked = '1';

  const badgeText = kind === 'native' ? '贊助貼文' : '廣告';

  el.style.outline = `3px dashed ${style.color}`;
  el.style.outlineOffset = '2px';
  el.style.borderRadius = '4px';
  el.title = `Truth ${badgeText}偵測：${style.label}（風險分數 ${riskScore}／100）\n目標：${targetUrl}`.trim();

  // iframe 內部無法插入元素（常跨網域），改在外層疊一個角標
  // 注意：不可對第三方頁面使用 innerHTML（Trusted Types），圖示一律用 DOM 元素
  const badge = document.createElement('span');
  const adIcon = createLockedIcon(style.icon, 13);
  if (adIcon) badge.appendChild(adIcon);
  const adText = document.createElement('span');
  adText.textContent = badgeText;
  badge.appendChild(adText);
  badge.setAttribute('aria-label', `Truth ${style.label}`);
  badge.dataset.veriadAdBadge = '1';
  badge.style.cssText = `
    position:absolute; z-index:2147483647; transform:translateY(-100%);
    display:inline-flex; align-items:center; gap:4px;
    background:${style.color}; color:#ffffff; font-size:11px; font-weight:700;
    line-height:1.3; font-family:"Segoe UI","Microsoft JhengHei","Noto Sans TC",system-ui,sans-serif;
    padding:2px 6px; border-radius:4px; pointer-events:none;
    border:1px solid ${style.color}; box-sizing:content-box;
    box-shadow:none;
  `;

  if (getComputedStyle(el).position === 'static') el.style.position = 'relative';
  el.insertAdjacentElement('beforebegin', badge);
}

async function scanAndMarkAds() {
  const ads = collectPageAds();
  if (ads.length === 0) return { checked: 0, high: 0, medium: 0, low: 0, avgScore: 0, maxScore: 0, items: [] };

  const urls = ads.map(a => a.checkUrl);
  const response = await chrome.runtime.sendMessage({ type: 'CHECK_LINKS', urls });
  if (!response || !response.ok) return { checked: 0, high: 0, medium: 0, low: 0, avgScore: 0, maxScore: 0, items: [] };

  let high = 0;
  let medium = 0;
  let low = 0;
  let scoreSum = 0;
  let scoredCount = 0;
  let maxScore = 0;
  const items = [];

  const adResults = response.results || {};
  for (const ad of ads) {
    const result = adResults[ad.checkUrl];
    if (!result || !result.ok) continue;
    const level = result.risk_level;
    const score = result.risk_score || 0;

    if (level === 'high') high++;
    else if (level === 'medium') medium++;
    else low++;

    scoreSum += score;
    scoredCount++;
    if (score > maxScore) maxScore = score;

    markAd(ad.el, level, score, ad.targetUrl, ad.kind);
    items.push({ url: ad.targetUrl, kind: ad.kind, level, score });
  }

  const avgScore = scoredCount ? Math.round(scoreSum / scoredCount) : 0;

  return { checked: ads.length, high, medium, low, avgScore, maxScore, items };
}

function markLink(anchor, riskLevel, riskScore) {
  const style = BADGE_STYLE[riskLevel];
  if (!style) return;

  // 避免重複標記
  if (anchor.dataset.veriadMarked) return;
  anchor.dataset.veriadMarked = '1';

  anchor.style.outline = `2px solid ${style.color}`;
  anchor.style.outlineOffset = '2px';
  anchor.style.borderRadius = '3px';
  anchor.title = `Truth 偵測：${style.label}（風險分數 ${riskScore}／100）\n${anchor.title || ''}`.trim();

  const badge = document.createElement('span');
  const linkIcon = createLockedIcon(style.icon, 14);
  if (linkIcon) {
    badge.appendChild(linkIcon);
  } else {
    // 圖示庫不可用：改放純文字替代，避免留下空方塊
    badge.textContent = riskLevel === 'high' ? '!!' : '!';
  }
  badge.setAttribute('aria-label', `Truth ${style.label}`);
  badge.setAttribute('role', 'img');
  badge.title = `Truth ${style.label}`;
  const textFallback = linkIcon
    ? 'padding:2px;line-height:0;'
    : 'padding:2px 5px;min-width:10px;font-size:11px;font-weight:700;line-height:14px;' +
      'font-family:"Segoe UI","Microsoft JhengHei","Noto Sans TC",system-ui,sans-serif;';
  badge.style.cssText = `
    display:inline-flex;align-items:center;justify-content:center;
    margin-left:4px;border-radius:4px;
    background:#ffffff;color:${style.color};
    box-sizing:content-box;
    border:1px solid rgba(22,102,141,.35);
    box-shadow:none;
    cursor:help;vertical-align:middle;
    ${textFallback}
  `;
  badge.dataset.veriadBadge = '1';
  anchor.insertAdjacentElement('afterend', badge);
}

async function scanAndMark() {
  const links = collectPageLinks();
  if (links.length === 0) return { checked: 0, high: 0, medium: 0 };

  const response = await chrome.runtime.sendMessage({ type: 'CHECK_LINKS', urls: links });
  if (!response || !response.ok) return { checked: 0, high: 0, medium: 0 };

  let high = 0;
  let medium = 0;

  const anchorsByHref = new Map();
  document.querySelectorAll('a[href^="http"]').forEach(a => {
    if (!anchorsByHref.has(a.href)) anchorsByHref.set(a.href, []);
    anchorsByHref.get(a.href).push(a);
  });

  for (const [url, result] of Object.entries(response.results || {})) {
    if (!result || !result.ok) continue;
    const level = result.risk_level;
    if (level !== 'high' && level !== 'medium') continue;

    if (level === 'high') high++;
    if (level === 'medium') medium++;

    const anchors = anchorsByHref.get(url) || [];
    anchors.forEach(a => markLink(a, level, result.risk_score));
  }

  return { checked: links.length, high, medium };
}

// -----------------------------------------------------------------------
// 自動掃描（依設定）
// -----------------------------------------------------------------------
(async () => {
  try {
    const { ok, settings } = await chrome.runtime.sendMessage({ type: 'GET_SETTINGS' });
    if (ok && settings.autoScanLinks && !alreadyScanned) {
      alreadyScanned = true;
      // 稍微延遲，避免與頁面初始渲染搶資源
      setTimeout(() => { scanAndMark().catch(() => {}); }, 1500);
    }
    // autoScanLinks 開啟時，一併掃描頁面廣告（沿用同一個開關；
    // 若想獨立控制，可在 DEFAULT_SETTINGS 加一個 autoScanAds 欄位）
    if (ok && settings.autoScanLinks && !alreadyScannedAds) {
      alreadyScannedAds = true;
      setTimeout(() => { scanAndMarkAds().catch(() => {}); }, 2000);
    }
  } catch (e) {
    // background 尚未就緒或頁面為受限頁面，靜默忽略
  }
})();

// -----------------------------------------------------------------------
// 接收 popup 手動觸發的掃描要求
// -----------------------------------------------------------------------
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === 'SCAN_PAGE_LINKS') {
    alreadyScanned = true;
    scanAndMark()
      .then(summary => sendResponse({ ok: true, summary }))
      .catch(err => sendResponse({ ok: false, error: err.message, message: err.message }));
    return true; // 非同步回應
  }
  if (message.type === 'SCAN_PAGE_ADS') {
    alreadyScannedAds = true;
    scanAndMarkAds()
      .then(summary => sendResponse({ ok: true, summary }))
      .catch(err => sendResponse({ ok: false, error: err.message, message: err.message }));
    return true; // 非同步回應
  }
});