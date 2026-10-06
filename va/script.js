/**
 * Truth Website — script.js v7.1
 * =============================================================================
 * 對齊 main.py / features.py v7.1（46 維特徵）的 API 合約。
 *
 * 修正紀錄
 *   v5.0：改為實際呼叫 POST /predict（不再用前端 calcScore 假分數）；source 對照表、
 *         triggered_rules、reasons、/feedback 欄位對齊 main.py。
 *   v7.1：FEATURE_MAP 與 localAssess 補上第 46 個特徵 sld_randomness（網域亂碼度）。
 *   v7.0（依 CONTRACT §4）：
 *     - FEATURE_MAP 補齊 features.FEATURE_NAMES 全部特徵；前 31 個說明依 v7 新定義
 *       更新（長度類不含 scheme 與 www.、英文詞改 token 比對、品牌類需「非官方網域」…）。
 *     - levenshtein_brand_dist 改正：0 = 與品牌同名但非官方（不是「正版」），
 *       1～2 = 拼字近似，99 = 無相似品牌或本身即官方網域。
 *     - 狀態標籤改用後端 data.feature_assessment[name]（features.assess_feature 是門檻的
 *       唯一來源）；只有後端沒回傳時才用本地 localAssess()（與 v7 門檻相同的備援）。
 *     - 卡片順序依後端 data.feature_names；後端有、FEATURE_MAP 沒有的特徵以通用卡片
 *       顯示（名稱／單位／說明取自 GET /features 的 specs）；FEATURE_MAP 有、後端沒回傳的
 *       特徵不顯示。
 *     - 特徵數量不再寫死：依 /features 的 count 或 feature_names.length 動態顯示。
 *     - 錯誤訊息一律讀 error || message（含 FastAPI 422 的 detail），HTTP 非 2xx 也先讀回應內容。
 *     - degraded = true（模型與特徵版本不符，改用規則＋特徵啟發式）時顯示「降級模式」提示。
 *     - 低調顯示 api_version / feature_version。
 * =============================================================================
 */

'use strict';

// ── API 位址 ─────────────────────────────────────────────────────────────
// 本機開發：對應 run_server.py（127.0.0.1:5500）。
// 直接執行 `python main.py` 也預設監聽 127.0.0.1:5500（可用環境變數 PORT 改埠，需同步修改下方埠號）。
const API_BASE = (location.hostname === 'localhost' || location.hostname === '127.0.0.1')
    ? 'http://127.0.0.1:5500'
    : location.origin;


// ── 45 個特徵定義（順序、名稱與 features.py FEATURE_NAMES 完全一致）─────────
// kind：count / ratio / bool / level / score（同 features.FEATURE_SPECS）。
// 狀態門檻不寫在這裡：以後端 feature_assessment 為準，見 localAssess() 的備援。
const FEATURE_MAP = {
    // ── 長度類 ──
    url_length: { name: '網址長度', unit: '字元', icon: 'ruler', kind: 'count',
        desc: '去掉 http(s):// 與 www. 後的網址字元數（路徑只有「/」時不計）。過長的網址可能夾帶混淆參數或隱藏真實目的地。' },
    domain_length: { name: '網域長度', unit: '字元', icon: 'globe', kind: 'count',
        desc: '主機名稱（小寫、去 www.、不含埠號）的字元數。超過 30 字元常見於混淆型網域。' },
    path_length: { name: '路徑長度', unit: '字元', icon: 'route', kind: 'count',
        desc: '網址路徑（不含 ? 參數與 # 片段，只有「/」視為 0）的字元數。路徑很長可能藏有混淆或追蹤資訊。' },
    query_length: { name: '參數長度', unit: '字元', icon: 'wrench', kind: 'count',
        desc: '? 之後查詢參數的字元數。超長參數常用來夾帶編碼後的跳轉或追蹤資料。' },

    // ── 字元比例類 ──
    digit_ratio: { name: '數字比例', unit: '%', icon: 'hash', kind: 'ratio',
        desc: '數字字元占網址（去 scheme 與 www.）的比例。隨機數字比例過高，常見於大量註冊的詐騙網域。' },
    hyphen_count: { name: '連字號數量', unit: '個', icon: 'minus', kind: 'count',
        desc: '網址中「-」的數量。大量連字號常見於「品牌-tw-login」這類拼接仿冒。' },
    dot_count: { name: '點號數量', unit: '個', icon: 'ellipsis', kind: 'count',
        desc: '網址中「.」的數量（scheme 與 www. 不計）。點號過多可能是以多層子網域混淆真實網域。' },
    special_chars: { name: '特殊字元數', unit: '個', icon: 'asterisk', kind: 'count',
        desc: '網址中非英數字元的數量（含 / ? = & # 等）。符號過多代表結構複雜或刻意干擾解析。' },

    // ── 結構類 ──
    subdomain_depth: { name: '子網域層數', unit: '層', icon: 'layers', kind: 'count',
        desc: '依公開後綴清單（PSL，含 github.io、web.app 等）切分後的子網域層數，www 不計（例如 sub.momo.com.tw = 1 層）。' },
    path_depth: { name: '路徑層數', unit: '層', icon: 'folder', kind: 'count',
        desc: '網址路徑以「/」分隔的非空段數。' },
    query_params: { name: '參數個數', unit: '個', icon: 'key', kind: 'count',
        desc: '查詢參數的種類數（含空值參數）。大量參數可能用於追蹤或藏匿跳轉資訊。' },

    // ── 協定 / 網域類 ──
    is_https: { name: 'HTTPS 加密', unit: '', icon: 'lock', isBool: true, kind: 'bool',
        desc: '明確寫 https:// 或未指定協定為 TRUE；只有明確寫 http:// 才是 FALSE（未加密連線）。' },
    is_ip_address: { name: 'IP 位址網域', unit: '', icon: 'monitor', isBool: true, kind: 'bool',
        desc: '主機名稱是 IPv4／IPv6／十進位整數 IP 而非網域名稱。IP 直連常見於臨時架設的詐騙頁面。' },
    suspicious_tld: { name: '可疑頂級域名', unit: '', icon: 'tag', isBool: true, kind: 'bool',
        desc: '頂級域名屬於常被濫用清單（TLD 風險等級 ≥ 1，例如 .top、.vip、.xyz、.cc）。這些 TLD 成本低，常被詐騙者大量註冊。' },

    // ── 語意類 ──
    has_scam_word: { name: '詐騙誘因詞', unit: '', icon: 'alert-triangle', isBool: true, kind: 'bool',
        desc: '解碼後網址含投資、獎勵、加密貨幣等誘因詞（英文以完整字詞比對，learn 不會算成 earn）；新聞、查核、政府、教育網域的路徑不計。' },
    brand_in_sld: { name: '品牌＋可疑 TLD', unit: '', icon: 'mask', isBool: true, kind: 'bool',
        desc: '網域主要名稱（SLD）含知名品牌字樣、使用可疑頂級域名，且不是該品牌官方網域。' },

    // ── 廣告追蹤參數類 ──
    has_utm: { name: 'UTM 追蹤參數', unit: '', icon: 'megaphone', isBool: true, kind: 'bool',
        desc: '網址帶有 utm_ 行銷追蹤參數。屬中性資訊：詐騙廣告也可能帶 UTM，不代表安全。' },
    has_gclid: { name: '廣告點擊 ID', unit: '', icon: 'target', isBool: true, kind: 'bool',
        desc: '網址帶有 gclid、gbraid、wbraid、msclkid、dclid 等搜尋廣告點擊 ID。屬中性資訊，不代表安全。' },

    // ── 混淆手法類 ──
    double_http: { name: '內嵌網址', unit: '', icon: 'repeat', isBool: true, kind: 'bool',
        desc: '解碼後網址出現兩次以上 http(s)://，可能是轉址或偽裝。後端會另外評估內嵌的目標網址並取較高分。' },
    long_domain: { name: '超長網域', unit: '', icon: 'move-horizontal', isBool: true, kind: 'bool',
        desc: '主機名稱超過 30 字元。超長網域常用來塞入品牌或混淆字串，讓人難以辨識真實網域。' },

    // ── 隨機性 ──
    domain_entropy: { name: '網域隨機度', unit: '', icon: 'shuffle', kind: 'score',
        desc: '網域主要名稱（SLD）的 Shannon 熵值。越高代表字元越隨機（如 k3x9pqr），可能是自動產生的詐騙網域。' },

    // ── 強化特徵 ──
    has_shortener: { name: '短網址', unit: '', icon: 'scissors', isBool: true, kind: 'bool',
        desc: '註冊網域屬於短網址服務（bit.ly、reurl.cc、ppt.cc 等），使用者無法預知點擊後的真實落點。' },
    gambling_number_pattern: { name: '博弈幸運數字', unit: '', icon: 'dice', isBool: true, kind: 'bool',
        desc: '主機名稱的數字段為 168、888、666、999、777 等博弈幸運數字，或 win888、888bet 類組合；純數字網域（如 1688.com、8591）不算。' },
    brand_typo_like: { name: '品牌拼字變形', unit: '', icon: 'type', isBool: true, kind: 'bool',
        desc: '主機名稱含品牌拼字變形（sh0pee、faebook、walmar 等同形字、截斷字或拼字近似），且不是該品牌官方網域；正牌網域不會命中。' },
    cloud_hosting: { name: '雲端／CDN 主機', unit: '', icon: 'cloud', isBool: true, kind: 'bool',
        desc: '主機名稱以雲端／CDN／PaaS 基礎設施後綴結尾（CloudFront、Azure、pages.dev 等）。詐騙者常用來快速部署並逃避封鎖。' },
    mobile_lure_path: { name: '行動版落地頁', unit: '', icon: 'smartphone', isBool: true, kind: 'bool',
        desc: '主機名稱首段為 m／h5／wap／app／sj／mobile，或路徑含 /h5、/wap 段（或以 /m 開頭）。詐騙廣告常為手機用戶設計專屬落地頁。' },
    suspicious_keyword_in_domain: { name: '網域含詐騙詞', unit: '', icon: 'eye', isBool: true, kind: 'bool',
        desc: '詐騙、博弈、假投資或加密貨幣強詞直接出現在主機名稱中（以完整字詞比對）。出現在網域本身比出現在路徑風險更高。' },
    many_subdomains: { name: '多層子網域', unit: '', icon: 'network', isBool: true, kind: 'bool',
        desc: '子網域層數達 3 層以上。大量子網域常用於規避封鎖清單或偽裝合法服務。' },
    contains_percent_encoding: { name: '百分比編碼', unit: '', icon: 'percent', isBool: true, kind: 'bool',
        desc: '網址含 %XX 編碼。中文網址常見，屬中性資訊；所有語意比對都會先解碼再進行，編碼無法藏住關鍵字。' },

    // ── v5.0：品牌相似度 & 網域新鮮度（v7 修正語意）──
    levenshtein_brand_dist: { name: '品牌相似距離', unit: '', icon: 'arrow-left-right', kind: 'score',
        desc: '網域與「非官方」品牌名稱的編輯距離：0 = 與品牌同名但不是官方網域（如 apple.xyz）、1～2 = 拼字近似、99 = 無相似品牌或本身就是官方網域。4 字以內的短品牌只接受距離 1，且需搭配其他風險訊號。' },
    newly_registered_like: { name: '疑似隨機新網域', unit: '', icon: 'sparkle', isBool: true, kind: 'bool',
        desc: '網域主要名稱為無意義的隨機字元組合（母音比例低、連續子音、高熵），不含品牌或常見字詞、也不在白名單，符合即用即棄型詐騙網域。' },

    // ── v7.0：165 新型態詐騙特徵 ──
    tld_risk_level: { name: 'TLD 風險等級', unit: '級', icon: 'gauge', kind: 'level',
        desc: '頂級域名風險：0 = 一般、1 = 常被濫用（如 .xyz、.cc、.info、.shop）、2 = 高度濫用或即用即棄（如 .top、.vip、.icu、.xin）。' },
    gambling_keyword: { name: '博弈詞', unit: '', icon: 'coins', isBool: true, kind: 'bool',
        desc: '含博弈英文、中文或拼音（casino、百家樂、娛樂城 yule、bocai、caipiao、葡京 pujing、遊戲商代號等），疑似非法博弈網站。' },
    investment_lure_keyword: { name: '假投資詞', unit: '', icon: 'trending-up', isBool: true, kind: 'bool',
        desc: '含飆股、投顧、老師帶單、當沖、IPO 抽籤、量化、AI 選股、tougu、licai、stock-vip 等假投資常見字詞。' },
    crypto_exchange_lure: { name: '假交易所／錢包詞', unit: '', icon: 'wallet', isBool: true, kind: 'bool',
        desc: '含 exchange、-ex、futures、OTC、staking、mining、airdrop、USDT、DeFi、wallet-connect 等假交易所或錢包盜取字詞（以完整字詞比對）。' },
    brand_impersonation: { name: '品牌冒用', unit: '', icon: 'badge-alert', isBool: true, kind: 'bool',
        desc: '網域使用知名電商、社群、銀行、券商、加密交易所或政府機關（如 165、npa）名稱，可能再加 tw、vip、數字等字樣，但不屬於該品牌官方網域。' },
    free_hosting_platform: { name: '免費架站平台', unit: '', icon: 'server', isBool: true, kind: 'bool',
        desc: '架在 web.app、firebaseapp.com、vercel.app、pages.dev、github.io、wixsite、notion.site 等免費架站或暫存平台。任何人都能免費建立，詐騙網站常用來快速上線。' },
    tunnel_or_ephemeral_host: { name: '臨時通道主機', unit: '', icon: 'plug-zap', isBool: true, kind: 'bool',
        desc: '使用 ngrok、trycloudflare、loca.lt、serveo 或 IPFS 閘道等臨時通道網址。網址隨時可換，常見於即用即棄的釣魚頁面。' },
    social_invite_link: { name: '社群群組邀請', unit: '', icon: 'users', isBool: true, kind: 'bool',
        desc: 'LINE（line.me/ti/g、lin.ee）、Telegram（t.me/+、joinchat）、WhatsApp、Discord 群組或官方帳號邀請連結。假投資詐騙常以廣告導流加入群組。' },
    punycode_domain: { name: '國際化網域', unit: '', icon: 'languages', isBool: true, kind: 'bool',
        desc: '網域含 xn--（punycode）或非 ASCII 字元。顯示文字可能與正牌網址極為相似（同形字仿冒），請仔細確認。' },
    url_has_at_symbol: { name: '@ 帳號偽裝', unit: '', icon: 'at-sign', isBool: true, kind: 'bool',
        desc: '網址在網域前放了「@」帳號欄位（例如 https://bank.com@evil.top/），瀏覽器實際連往 @ 後面的網域，屬典型偽裝手法。' },
    non_standard_port: { name: '非標準埠', unit: '', icon: 'door-open', isBool: true, kind: 'bool',
        desc: '網址明確指定 80／443 以外的連接埠（如 :8080）。合法的商業網站極少這樣做。' },
    sld_digit_count: { name: '網域數字個數', unit: '個', icon: 'binary', kind: 'count',
        desc: '網域主要名稱（SLD）中的數字個數。品牌加長串數字（如 fubon168888）常見於大量註冊的詐騙網域。' },
    sld_length: { name: '網域名稱長度', unit: '字元', icon: 'whole-word', kind: 'count',
        desc: '網域主要名稱（SLD，不含子網域與頂級域名）的字元數。' },
    path_scam_route: { name: '詐騙 App 路由', unit: '', icon: 'app-window', isBool: true, kind: 'bool',
        desc: '含 #/pages/、/h5/#/、#/register、invitecode／agentcode 等邀請碼參數或 download.html 下載誘導，常見於假投資／假交易所 H5 App。合法下載頁也可能命中，列為需注意。' },
    sld_randomness: { name: '網域亂碼度', unit: '', icon: 'shuffle', kind: 'score',
        desc: '以英文與漢語拼音的字母組合統計，衡量網域名稱是否像無法發音的亂碼（0 = 像一般字詞、1 = 像 xkqplbwz）；≥ 0.7 視為疑似亂碼。只看網域名稱本身。' },
};


// ── 本地備援門檻（只在後端沒有回傳 feature_assessment 時使用）──────────────
// 與 features.py v7 assess_feature() 相同；後端才是唯一來源，修改門檻請改後端。
const LOCAL_BOOL_LABELS = {
    is_https: ['safe', 'HTTPS／未指定', 'warn', '未加密 HTTP'],
    is_ip_address: ['alert', 'IP 位址', 'safe', '一般網域'],
    suspicious_tld: ['warn', '常被濫用 TLD', 'safe', '一般 TLD'],
    has_scam_word: ['warn', '含誘因詞', 'safe', '未偵測'],
    brand_in_sld: ['alert', '品牌＋可疑 TLD', 'safe', '未偵測'],
    has_utm: ['neutral', '含行銷追蹤', 'neutral', '無'],
    has_gclid: ['neutral', '含廣告點擊 ID', 'neutral', '無'],
    double_http: ['warn', '內嵌網址', 'safe', '無'],
    long_domain: ['warn', '網域過長', 'safe', '正常'],
    has_shortener: ['warn', '短網址', 'safe', '非短網址'],
    gambling_number_pattern: ['warn', '博弈數字', 'safe', '未偵測'],
    brand_typo_like: ['alert', '疑似拼字仿冒', 'safe', '未偵測'],
    cloud_hosting: ['warn', '雲端／CDN 主機', 'safe', '一般主機'],
    mobile_lure_path: ['warn', '行動版落地頁', 'safe', '未偵測'],
    suspicious_keyword_in_domain: ['alert', '網域含詐騙詞', 'safe', '未偵測'],
    many_subdomains: ['warn', '子網域過多', 'safe', '正常'],
    contains_percent_encoding: ['neutral', '含編碼', 'neutral', '無'],
    newly_registered_like: ['warn', '疑似隨機新網域', 'safe', '正常'],
    gambling_keyword: ['alert', '博弈詞', 'safe', '未偵測'],
    investment_lure_keyword: ['alert', '假投資詞', 'safe', '未偵測'],
    crypto_exchange_lure: ['alert', '假交易所詞', 'safe', '未偵測'],
    brand_impersonation: ['alert', '品牌冒用', 'safe', '未偵測'],
    free_hosting_platform: ['warn', '免費架站平台', 'safe', '未偵測'],
    tunnel_or_ephemeral_host: ['alert', '臨時通道', 'safe', '未偵測'],
    social_invite_link: ['warn', '群組邀請連結', 'safe', '未偵測'],
    punycode_domain: ['warn', '國際化網域', 'safe', '未偵測'],
    url_has_at_symbol: ['alert', '@ 偽裝', 'safe', '未偵測'],
    non_standard_port: ['warn', '非標準埠', 'safe', '標準埠'],
    path_scam_route: ['warn', '疑似詐騙 App 路由', 'safe', '未偵測'],
};
// [門檻（≥ 即超過）, 超過狀態, 文字, 未超過狀態, 文字]
const LOCAL_NUMERIC_LABELS = {
    url_length: [120, 'warn', '偏長', 'safe', '正常'],
    domain_length: [31, 'warn', '網域偏長', 'safe', '正常'],
    path_length: [100, 'warn', '路徑很長', 'neutral', '一般'],
    query_length: [200, 'warn', '參數很長', 'neutral', '一般'],
    digit_ratio: [0.3, 'warn', '數字偏多', 'safe', '正常'],
    hyphen_count: [4, 'warn', '連字號偏多', 'safe', '正常'],
    dot_count: [6, 'warn', '點號偏多', 'safe', '正常'],
    special_chars: [30, 'warn', '符號偏多', 'safe', '正常'],
    subdomain_depth: [3, 'warn', '子網域過多', 'safe', '正常'],
    path_depth: [6, 'warn', '路徑較深', 'neutral', '一般'],
    query_params: [8, 'warn', '參數較多', 'neutral', '一般'],
    sld_digit_count: [5, 'warn', '數字偏多', 'safe', '正常'],
    sld_length: [20, 'warn', '名稱偏長', 'neutral', '一般'],
};

function localAssess(name, value) {
    const v = Number.isFinite(Number(value)) ? Number(value) : 0;
    const b = LOCAL_BOOL_LABELS[name];
    if (b) return v >= 1 ? { status: b[0], text: b[1] } : { status: b[2], text: b[3] };
    if (name === 'levenshtein_brand_dist') {
        if (v === 0) return { status: 'alert', text: '與品牌同名但非官方' };
        if (v === 1) return { status: 'alert', text: '高度相似品牌' };
        if (v === 2) return { status: 'warn', text: '相似品牌' };
        return { status: 'safe', text: '無相似品牌' };
    }
    if (name === 'tld_risk_level') {
        if (v >= 2) return { status: 'alert', text: '高度濫用 TLD' };
        if (v >= 1) return { status: 'warn', text: '常被濫用 TLD' };
        return { status: 'safe', text: '一般 TLD' };
    }
    if (name === 'sld_randomness') {
        if (v >= 0.7) return { status: 'warn', text: '疑似亂碼網域' };
        if (v >= 0.5) return { status: 'neutral', text: '略不自然' };
        return { status: 'safe', text: '像一般字詞' };
    }
    if (name === 'domain_entropy') {
        if (v >= 3.5) return { status: 'warn', text: '隨機性高' };
        if (v >= 2.5) return { status: 'neutral', text: '中等' };
        return { status: 'safe', text: '規律' };
    }
    const n = LOCAL_NUMERIC_LABELS[name];
    if (n) return v >= n[0] ? { status: n[1], text: n[2] } : { status: n[3], text: n[4] };
    return { status: 'neutral', text: '資訊' };
}

const STATUS_CLASS = { alert: 'status-alert', safe: 'status-safe', warn: 'status-warn', neutral: 'status-neutral' };


// ── source 對照表（main.py build_response() 實際回傳的值）──────────────────
// 圖示一律來自內嵌的 TruthIcons（icons.js），圖示名稱 + 純文字，不使用 emoji。
const SOURCE_MAP = {
    trusted_domain:      { icon: 'shield-check',   color: 'var(--safe)',   text: '白名單直通' },
    blocklist:           { icon: 'ban',            color: 'var(--danger)', text: '政府封鎖黑名單（165 / 刑事局）' },
    hybrid_ai_hard_rule: { icon: 'sliders',        color: '#2a7ea6',       text: 'AI + 硬規則綜合判斷' },
    ai_model:            { icon: 'bot',            color: '#2a7ea6',       text: 'AI 模型判定' },
    rules_only:          { icon: 'alert-triangle', color: 'var(--warn)',   text: '硬規則＋特徵啟發式判斷（未使用 AI 模型）' },
    degraded:            { icon: 'alert-triangle', color: 'var(--warn)',   text: '降級模式（硬規則＋特徵啟發式）' },
};

const LEVEL_COLOR = { high: 'var(--danger)', medium: 'var(--warn)', low: 'var(--safe)' };
const LEVEL_ICON = { high: 'shield-alert', medium: 'alert-triangle', low: 'shield-check' };
const REASON_CHIP_CLASS = { high: 'chip-danger', medium: 'chip-warn', safe: 'chip-safe' };

// ── 圖示輔助 ─────────────────────────────────────────────────────────────
function ico(name, size, opts) {
    return (window.TruthIcons && window.TruthIcons.svg(name, Object.assign({ size: size || 16 }, opts || {}))) || '';
}

// 判斷來源：圖示 + 純文字（未知 source 只顯示轉義後的文字）
function sourceHtml(source, size, cls) {
    const m = SOURCE_MAP[source];
    if (!m) return escapeHtml(source == null ? '' : String(source));
    return ico(m.icon, size || 16, { cls: cls || 'src-ic', style: `color:${m.color}` }) + escapeHtml(m.text);
}

// 回報按鈕內容（圖示 + 文字）
const FB_LABEL = {
    scam: () => ico('flag', 16) + '回報為詐騙',
    safe: () => ico('check-circle', 16) + '回報為正常',
    done: () => ico('check-circle', 16) + '已回報，感謝！',
    fail: () => ico('alert-circle', 16) + '回報失敗，請重試',
};


// ── DOM ──────────────────────────────────────────────────────────────────
const featuresGrid = document.getElementById('featuresGridInner');
const resultArea    = document.getElementById('resultArea');
const loading        = document.getElementById('loading');
const glowFill       = document.getElementById('glowFill');
const scoreLabel     = document.getElementById('scoreLabel');
const analyzeBtn     = document.getElementById('analyzeBtn');
const targetInput    = document.getElementById('targetInput');
const btnScam        = document.getElementById('btnScam');
const btnSafe        = document.getElementById('btnSafe');

analyzeBtn.addEventListener('click', analyzeUrl);
targetInput.addEventListener('keydown', e => { if (e.key === 'Enter') analyzeUrl(); });

btnScam.addEventListener('click', () => sendFeedback(1));
btnSafe.addEventListener('click', () => sendFeedback(0));

let lastUrl = '';
let lastScore = 0;

// GET /features 取得的特徵規格（通用卡片用）；失敗時維持空物件
let featureSpecs = {};
let featureVersion = '';


// ── 特徵數量／版本：依 API 動態顯示（不寫死 31 / 45）────────────────────
function setFeatureCount(n) {
    const count = Number(n);
    if (!Number.isFinite(count) || count <= 0) return;
    document.querySelectorAll('[data-feature-count]').forEach(el => { el.textContent = `${count} 個`; });
}

function setFeatureVersion(v) {
    if (!v) return;
    featureVersion = String(v);
    document.querySelectorAll('[data-feature-version]').forEach(el => {
        el.textContent = ` · 特徵版本 ${featureVersion}`;
    });
}

async function loadFeatureMeta() {
    try {
        const res = await fetch(`${API_BASE}/features`, { signal: AbortSignal.timeout(5000) });
        if (!res.ok) return;
        const meta = await res.json();
        if (!meta || typeof meta !== 'object') return;
        if (meta.specs && typeof meta.specs === 'object') featureSpecs = meta.specs;
        setFeatureCount(meta.count || (Array.isArray(meta.features) ? meta.features.length : 0));
        setFeatureVersion(meta.version);
    } catch (e) {
        // 後端尚未啟動時不影響頁面；分析時會再由 /predict 回傳的 feature_names 更新
    }
}
loadFeatureMeta();


// ── 錯誤訊息：一律讀 error || message（FastAPI 422 讀 detail）────────────
class ApiError extends Error {
    constructor(message) { super(message); this.name = 'ApiError'; }
}

function apiErrorMessage(d) {
    if (!d || typeof d !== 'object') return '';
    if (d.error) return String(d.error);
    if (d.message) return String(d.message);
    if (Array.isArray(d.detail)) return d.detail.map(x => (x && x.msg) || '').filter(Boolean).join('；');
    if (d.detail) return String(d.detail);
    return '';
}


// ═══════════════════════════════════════════════════════════════════════
// 主分析函數 — 實際呼叫 /predict（真實模型 + 黑名單 + 硬規則 + 灰色地帶補強）
// ═══════════════════════════════════════════════════════════════════════
async function analyzeUrl() {
    const url = targetInput.value.trim();
    if (!url) { alert('請先輸入網址'); return; }

    lastUrl = url;
    loading.style.display = 'block';
    resultArea.style.display = 'none';
    resetFeedbackButtons();
    analyzeBtn.disabled = true;

    try {
        const res = await fetch(`${API_BASE}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            // debug:true 時後端額外回傳 features、feature_names、feature_assessment，供特徵矩陣顯示
            body: JSON.stringify({ url, debug: true }),
            signal: AbortSignal.timeout(12000),
        });

        let data = null;
        try { data = await res.json(); } catch (e) { data = null; }

        if (!res.ok || !data || data.ok === false) {
            const msg = apiErrorMessage(data);
            if (msg) throw new ApiError(msg);
            throw new ApiError(res.ok ? '分析失敗，請確認網址格式' : `伺服器回應異常 (HTTP ${res.status})`);
        }

        lastScore = Number(data.risk_score) || 0;
        renderResult(data, url);

    } catch (err) {
        renderError(err, url);
    } finally {
        analyzeBtn.disabled = false;
        loading.style.display = 'none';
        resultArea.style.display = 'block';
        setTimeout(() => resultArea.scrollIntoView({ behavior: 'smooth', block: 'start' }), 120);
    }
}


// ── 顯示結果 ─────────────────────────────────────────────────────────────
function ruleName(r) {
    if (r == null) return '';
    if (typeof r === 'object') return String(r.name || r.rule || r.key || '');
    return String(r);
}

function renderResult(data, url) {
    const score = Number.isFinite(Number(data.risk_score)) ? Number(data.risk_score) : 0;
    const level = data.risk_level; // 'high' | 'medium' | 'low'（main.py get_risk_level()）
    const color = LEVEL_COLOR[level] || 'var(--neutral)';
    const levelIcon = ico(LEVEL_ICON[level] || 'help-circle', 24, { style: `color:${color}` });

    document.getElementById('verdictLabel').innerHTML = `${levelIcon}<span>${escapeHtml(data.verdict || data.risk_label || '')}</span>`;
    document.getElementById('targetUrlText').innerHTML = ico('link', 16, { cls: 'ic-lead' }) + escapeHtml(url);

    scoreLabel.textContent = score + '%';
    scoreLabel.style.color = color;
    glowFill.style.width = Math.max(0, Math.min(100, score)) + '%';
    glowFill.style.background = color;
    glowFill.style.boxShadow = `0 0 16px ${color}`;

    // ── 判斷來源徽章：main.py 的 source + triggered_rules ──
    const sourceBadge = document.getElementById('sourceBadge');
    let badgeText = sourceHtml(data.source, 16);
    const rules = Array.isArray(data.triggered_rules) ? data.triggered_rules.map(ruleName).filter(Boolean) : [];
    if (rules.length) {
        badgeText += `（命中規則：${rules.map(escapeHtml).join('、')}）`;
    }
    if (data.degraded) badgeText += degradedHtml(data);
    badgeText += versionHtml(data);
    sourceBadge.innerHTML = badgeText;

    // ── 特徵矩陣（只有 debug:true 才有 data.features）──
    if (data.features && typeof data.features === 'object' && Object.keys(data.features).length) {
        renderFeatures(data);
    } else {
        featuresGrid.innerHTML = `<div class="empty-note" style="grid-column:1/-1;color:var(--muted);padding:12px;">${ico('info', 18)}<span>此結果來自白名單/黑名單直通，未執行完整特徵分析</span></div>`;
    }

    renderSummary(data, score, level, url);
}

// 降級模式提示（模型與特徵版本不符或模型未載入時，後端改用規則＋特徵啟發式評分）
function degradedHtml(data) {
    const reason = data.degrade_reason ? `：${escapeHtml(String(data.degrade_reason))}` : '';
    return `<div class="degraded-note">${ico('alert-triangle', 15)}<span>降級模式：AI 模型暫不可用，本次以硬規則＋特徵啟發式評分${reason}</span></div>`;
}

// API／特徵版本（低調顯示）
function versionHtml(data) {
    const parts = [];
    if (data.api_version) parts.push(`API ${escapeHtml(String(data.api_version))}`);
    if (data.feature_version) parts.push(`特徵 ${escapeHtml(String(data.feature_version))}`);
    if (data.feature_version) setFeatureVersion(data.feature_version);
    return parts.length ? `<div class="ver-tag">${parts.join(' · ')}</div>` : '';
}


// ── 渲染特徵卡片（data.features 是後端算好的真實值，非前端重算）──────────
// 順序依後端 feature_names；後端沒回傳的特徵不顯示；FEATURE_MAP 沒有的用通用卡片。
function renderFeatures(data) {
    const features = data.features;
    const assessment = (data.feature_assessment && typeof data.feature_assessment === 'object') ? data.feature_assessment : {};
    const names = [];
    const seen = new Set();
    (Array.isArray(data.feature_names) ? data.feature_names : []).forEach(n => {
        if (Object.prototype.hasOwnProperty.call(features, n) && !seen.has(n)) { names.push(n); seen.add(n); }
    });
    Object.keys(features).forEach(n => { if (!seen.has(n)) { names.push(n); seen.add(n); } });

    setFeatureCount(Array.isArray(data.feature_names) && data.feature_names.length ? data.feature_names.length : names.length);

    featuresGrid.innerHTML = names.map(key => featureCardHtml(key, features[key], assessment[key])).join('');
}

function featureConfig(key) {
    if (Object.prototype.hasOwnProperty.call(FEATURE_MAP, key)) return FEATURE_MAP[key];
    // 通用卡片：名稱／單位／說明取自 GET /features 的 specs（v7），沒有時顯示原始名稱
    const spec = (featureSpecs && featureSpecs[key]) || {};
    const kind = spec.kind || '';
    return {
        name: spec.zh || key,
        unit: spec.unit || '',
        icon: 'info',
        kind,
        isBool: kind === 'bool',
        desc: spec.desc || '後端新增的特徵（前端尚未建立專屬說明，數值與狀態以後端為準）。',
        generic: true,
    };
}

function formatFeatureValue(key, val, cfg) {
    const num = Number(val);
    if (val == null || val === '' || !Number.isFinite(num)) return String(val == null ? '—' : val);
    if (cfg.isBool) return num >= 1 ? 'TRUE' : 'FALSE';
    if (key === 'levenshtein_brand_dist' && num >= 99) return '—';
    if (cfg.kind === 'ratio') return Math.round(num * 100) + ' %';
    const shown = Number.isInteger(num) ? num : num.toFixed(2);
    return shown + (cfg.unit ? ' ' + cfg.unit : '');
}

function featureCardHtml(key, val, assessed) {
    const cfg = featureConfig(key);
    let a = (assessed && typeof assessed === 'object' && STATUS_CLASS[assessed.status]) ? assessed : null;
    if (!a) a = cfg.generic ? { status: 'neutral', text: '資訊' } : localAssess(key, val);
    const cls = STATUS_CLASS[a.status] || 'status-neutral';
    const display = formatFeatureValue(key, val, cfg);
    const descHtml = cfg.desc ? `<div class="feature-desc">${escapeHtml(cfg.desc)}</div>` : '';
    return `
            <div class="feature-card${cfg.generic ? ' feature-generic' : ''}">
                <div class="feature-head">
                    <span class="feature-ico ${cls}">${ico(cfg.icon, 16) || ico('info', 16)}</span>
                    <div class="feature-key">${String(key).toUpperCase().split('_').map(escapeHtml).join('_<wbr>')}</div>
                </div>
                <div class="feature-title">${escapeHtml(cfg.name)}</div>
                ${descHtml}
                <div class="feature-value-row">
                    <div class="feature-value">${escapeHtml(String(display))}</div>
                    <div class="feature-status ${cls}">${escapeHtml(String(a.text || ''))}</div>
                </div>
            </div>`;
}


// ── AI 摘要：直接使用後端 reasons + suggestion，不再前端猜測 ──────────────
function renderSummary(data, score, level, url) {
    const box   = document.getElementById('aiSummary');
    const body  = document.getElementById('aiSummaryText');
    const chips = document.getElementById('aiHighlight');
    box.style.display = 'block';

    const domain = (() => {
        try { return new URL(/^[a-z][a-z0-9+.-]*:\/\//i.test(url) ? url : 'https://' + url).hostname; }
        catch (e) { return url; }
    })();

    const reasons = Array.isArray(data.reasons) ? data.reasons.filter(r => r && typeof r === 'object') : [];
    const isDanger = level === 'high';
    const isWarn = level === 'medium';
    const label = data.risk_label || (isDanger ? '高風險' : isWarn ? '中風險' : '低風險');

    let text = `AI 分析 <strong>${escapeHtml(domain)}</strong> 後，詐騙風險指數為 <strong>${score}%</strong>，判定為<strong>${escapeHtml(label)}</strong>`;
    text += `（判斷來源：<span class="src-inline">${sourceHtml(data.source, 15, 'src-sm')}</span>）。`;
    if (data.degraded) {
        text += `<br>目前為降級模式，AI 模型暫不可用，分數由硬規則與特徵啟發式計算。`;
    }

    const highReasons = reasons.filter(r => r.level === 'high').map(r => r.message);
    const mediumReasons = reasons.filter(r => r.level === 'medium').map(r => r.message);

    if (highReasons.length) {
        text += `<br><br>主要風險因素：<br>` + highReasons.map(m => `・${escapeHtml(m)}`).join('<br>');
    }
    if (mediumReasons.length) {
        text += `<br><br>次要疑慮：<br>` + mediumReasons.map(m => `・${escapeHtml(m)}`).join('<br>');
    }
    if (!reasons.length) {
        text += `<br><br>目前未觸發任何風險特徵。`;
    }
    if (data.suggestion) {
        text += `<br><br><strong>${escapeHtml(data.suggestion)}</strong>`;
    }

    body.innerHTML = text;

    chips.innerHTML = reasons.map(r => {
        const cls = REASON_CHIP_CLASS[r.level] || 'chip-warn';
        return `<span class="ai-chip ${cls}">${escapeHtml(r.key)}</span>`;
    }).join('');
}


// ── 錯誤畫面 ─────────────────────────────────────────────────────────────
function renderError(err, url) {
    const isApi = err && err.name === 'ApiError';
    let message;
    if (err.name === 'TimeoutError' || err.name === 'AbortError') {
        message = '後端回應超時，請確認伺服器是否正常運作';
    } else if (isApi) {
        message = `分析失敗：${err.message}`;
    } else if (String(err.message).includes('Failed to fetch') || String(err.message).includes('NetworkError')) {
        message = `無法連線到後端（${API_BASE}），請確認已執行 python run_server.py`;
    } else {
        message = `分析失敗：${err.message}`;
    }

    document.getElementById('verdictLabel').innerHTML = isApi
        ? ico('alert-circle', 24, { style: 'color:var(--warn)' }) + '<span>無法分析</span>'
        : ico('plug', 24, { style: 'color:var(--danger)' }) + '<span>連線失敗</span>';
    document.getElementById('targetUrlText').innerHTML = ico('link', 16, { cls: 'ic-lead' }) + escapeHtml(url);
    scoreLabel.innerHTML = '—';
    scoreLabel.style.color = '';
    glowFill.style.width = '0%';
    document.getElementById('sourceBadge').innerHTML = escapeHtml(message);
    featuresGrid.innerHTML = '';

    const box = document.getElementById('aiSummary');
    box.style.display = 'block';
    document.getElementById('aiSummaryText').innerHTML = escapeHtml(message);
    document.getElementById('aiHighlight').innerHTML = '';
}


// ═══════════════════════════════════════════════════════════════════════
// 誤判回報 — 對齊 main.py FeedbackRequest { url, label, ai_score, note }
// ═══════════════════════════════════════════════════════════════════════
async function sendFeedback(isScam) {
    if (!lastUrl) return;

    const btn = isScam ? btnScam : btnSafe;
    btnScam.disabled = true;
    btnSafe.disabled = true;

    try {
        const res = await fetch(`${API_BASE}/feedback`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                url: lastUrl,
                label: isScam,               // 後端要求 0/1，不是 true/false 字串，JSON 的 1/0 可直接對應
                ai_score: lastScore / 100,    // 後端存的是 0~1 機率，前端 lastScore 是 0~100 分數
                note: '',
            }),
            signal: AbortSignal.timeout(5000),
        });

        let data = null;
        try { data = await res.json(); } catch (e) { data = null; }

        if (res.ok && data && data.ok) {
            btn.innerHTML = FB_LABEL.done();
            btn.classList.add('done');
        } else {
            throw new Error(apiErrorMessage(data) || '回報失敗');
        }
    } catch (err) {
        btnScam.disabled = false;
        btnSafe.disabled = false;
        btn.innerHTML = FB_LABEL.fail();
        btn.classList.add('fail');
        setTimeout(resetFeedbackButtons, 2000);
    }
}

function resetFeedbackButtons() {
    btnScam.disabled = false;
    btnSafe.disabled = false;
    [btnScam, btnSafe].forEach(b => {
        b.classList.remove('done', 'fail');
        b.style.borderColor = '';
        b.style.color = '';
    });
    btnScam.innerHTML = FB_LABEL.scam();
    btnSafe.innerHTML = FB_LABEL.safe();
}


// ── 工具函數 ─────────────────────────────────────────────────────────────
function escapeHtml(t) {
    const d = document.createElement('div');
    d.textContent = t == null ? '' : String(t);
    return d.innerHTML;
}
