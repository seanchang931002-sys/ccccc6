# =============================================================================
# rules_config.py — 白名單網域、政府黑名單網域、硬規則與詞庫設定（v7.0）
# =============================================================================
# 重構紀錄（v6）：把原本寫死在 main.py 裡的 TRUSTED_DOMAINS / HARD_RULES /
# BLOCKED_DOMAINS 抽到這個獨立模組。
#
# 修正紀錄（v7.0，依 CONTRACT §0～2 與 165 研究報告）：
#   - HardRule 新增 target 欄位（"url" / "decoded" / "host" / "path" / "any"），
#     並提供 HardRule.matches(targets) 與 match_hard_rules(targets) 輔助函式；
#     targets 由 features.get_rule_targets(url) 產生，main 不必自行拆字串。
#       url     = features.canonicalize_url(輸入)（補 https://、host 小寫、IDN 轉 punycode）
#       decoded = url 經 percent-decode（最多兩輪）＋ NFKC
#       host    = features.get_hostname(輸入)（小寫、去 www.、去埠號與 userinfo）
#       path    = 解碼後的 path + ?query + #fragment
#       any     = url 或 decoded 任一命中
#   - 修正 mobile_h5_wap_highrisk：舊 pattern 的 (m|h5)\. 會命中任何 .com.tw
#     （com. 的 m.），天下雜誌 cw.com.tw 直接 70 分。改為「h5./wap. 子網域或 /h5、/wap
#     路徑段」且「高風險 TLD」才觸發。
#   - 修正 brand_spoof_*：改為「hostname token 是品牌（＋詐騙常見前後綴/數字）且
#     不屬於該品牌官方網域」才觸發；寶雅 poyabuy.com.tw、康是美、屈臣氏等正牌網域
#     不再誤判。官方網域排除清單由 BRAND_OFFICIAL_DOMAINS ∪ TRUSTED_DOMAINS 自動產生。
#   - 修正子字串誤判：所有英文詞改為 token 邊界比對（(?<![a-z0-9]) … (?![a-z])），
#     learn/alphabet/better/method/1688/online/airline 不再命中。
#   - crypto_specific_pattern 改為只比對 hostname（舊版比對整串網址，會命中
#     幣圈新聞路徑）；double_http 排除廣告點擊/轉址服務（doubleclick 等）。
#   - suspicious_tld_random_domain 依名稱實作：高度濫用 TLD「且」SLD 含 4 個以上連續子音；
#     不再只看 TLD（ptt.cc、vocus.cc、abc.xyz 等合法站不會觸發）。
#   - 中文規則改比對 percent-decode 後字串，並排除新聞／查核／政府／教育網域。
#   - 新增 165 類規則：博弈拼音/娛樂城＋數字、假交易所品牌冒用、uni-app 邀請碼路由、
#     tunnel 通道、punycode、@ 偽裝、官方網域塞在子網域、政府機關冒用、社群群組邀請。
#   - 分數分級：確定性高的冒用/通道/@偽裝 85～92；典型 165 組合 75～88；
#     單一弱訊號 50～65（會把正常網站推到 ≥40 的弱訊號一律不做成硬規則）。
#   - 補 TRUSTED_DOMAINS / TRUSTED_SUFFIXES / BRAND_OFFICIAL_DOMAINS /
#     WHITELIST_OVERRIDE_SIGNALS（自 main.py 移入）/ WHITELIST_OVERRIDE_PATTERNS，
#     以及 features.py 共用的詞庫（GAMBLING_TERMS 等）、TLD_RISK_TIERS、
#     FREE_HOSTING_SUFFIXES、TUNNEL_SUFFIXES、SOCIAL_INVITE_PATTERNS。
#   - 修正註解：scsb.com.tw 是上海商業儲蓄銀行（新光銀行是 skbank.com.tw）。
#   注意：本模組不得 import features.py（features 會 import 本模組的詞庫）。
# =============================================================================

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Pattern, Set, Tuple, Union

RULES_VERSION = "7.1.0"


# =============================================================================
# 硬規則資料結構
# =============================================================================

VALID_RULE_TARGETS: Tuple[str, ...] = ("url", "decoded", "host", "path", "any")


@lru_cache(maxsize=512)
def _compile(pattern: str) -> Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


@dataclass(frozen=True)
class HardRule:
    """
    單一硬規則。

    target 指定 pattern 要比對哪一段字串（見檔頭說明）；main.py 的
    check_hard_rules() 取所有命中規則中最高的 score 作為硬規則分數，
    並用 message 組成使用者看得懂的原因。
    """

    name: str
    pattern: str
    score: int
    message: str
    target: str = "any"

    def compiled(self) -> Pattern[str]:
        return _compile(self.pattern)

    def matches(self, targets: Union[str, Mapping[str, str]]) -> bool:
        """
        targets：features.get_rule_targets(url) 的回傳值；若只給字串，
        則視為所有 target 都是同一個字串（相容舊呼叫方式）。永不丟例外。
        """
        try:
            regex = self.compiled()
            if isinstance(targets, str):
                return regex.search(targets) is not None
            if self.target == "any":
                return any(
                    regex.search(targets.get(key, "") or "") is not None
                    for key in ("url", "decoded")
                )
            return regex.search(targets.get(self.target, "") or "") is not None
        except Exception:
            return False


def match_hard_rules(targets: Union[str, Mapping[str, str]]) -> List["HardRule"]:
    """回傳所有命中的硬規則（依 HARD_RULES 順序）。"""
    return [rule for rule in HARD_RULES if rule.matches(targets)]


def _alt(words: Iterable[str]) -> str:
    """把字詞清單轉成 regex 交替式（長字優先，避免短字先吃掉）。"""
    uniq = sorted({w for w in words if w}, key=lambda s: (-len(s), s))
    return "|".join(re.escape(w) for w in uniq)


# =============================================================================
# TLD 風險分級（研究報告 §9）
# =============================================================================
# 2 = 高度濫用／即用即棄；1 = 常被濫用但合法使用也多；其餘 = 0。
# .cc 列 1（ptt.cc、vocus.cc、reurl.cc 皆為台灣合法大站）；.jp/.co/.ai/.io/.app/.me 列 0。

TLD_RISK_TIERS: Dict[int, FrozenSet[str]] = {
    2: frozenset({
        "xin", "win", "icu", "bond", "cfd", "sbs", "cyou", "qpon", "help", "run", "best",
        "pics", "wang", "vip", "top", "bid", "lol", "quest", "tk", "ml", "ga", "cf", "gq",
        "forex", "faith", "racing", "date", "download", "men", "party", "review", "stream",
        "gdn", "accountant", "cricket", "science", "zip", "mov", "kim",
    }),
    1: frozenset({
        "pro", "live", "world", "support", "shop", "store", "online", "site", "website",
        "xyz", "cc", "asia", "cn", "info", "club", "fun", "space", "fit", "tech", "link",
        "click", "work", "life", "cloud", "today", "ltd", "buzz", "biz", "ink", "trade",
        "finance", "fund", "money", "cash", "exchange", "markets", "loan", "pw", "hair",
        "beauty", "skin", "rest", "mom", "one", "bar", "cam", "monster", "su", "ws", "uno",
    }),
}

# 舊版相容：suspicious_tld 特徵 = TLD 風險等級 ≥ 1
SUSPICIOUS_TLDS: FrozenSet[str] = TLD_RISK_TIERS[2] | TLD_RISK_TIERS[1]

_TIER_ANY_ALT = _alt(SUSPICIOUS_TLDS)
_TIER2_ALT = _alt(TLD_RISK_TIERS[2])
_RISKY_CCTLDS = sorted(t for t in SUSPICIOUS_TLDS if len(t) == 2)


# =============================================================================
# 短網址 / 免費架站 / 臨時通道 / 社群邀請（研究報告 §6）
# =============================================================================

SHORTENER_DOMAINS: FrozenSet[str] = frozenset({
    # 台灣常用
    "reurl.cc", "lihi.cc", "lihi1.com", "lihi1.cc", "lihi2.cc", "lihi2.com", "lihi3.cc",
    "lihi3.com", "lihi.tv", "pse.is", "ppt.cc", "0rz.tw", "picsee.io",
    # 國際
    "bit.ly", "bitly.com", "tinyurl.com", "goo.gl", "t.co", "is.gd", "ow.ly", "cutt.ly",
    "shorturl.at", "rb.gy", "t.ly", "s.id", "qrco.de", "rebrand.ly", "tiny.cc", "buff.ly",
    "v.gd", "shorte.st", "adf.ly", "bl.ink", "short.io", "urlz.fr", "x.gd",
})

# 以 hostname 後綴比對（需有使用者子網域；sites.google.com 等「路徑型」平台另列）
FREE_HOSTING_SUFFIXES: Tuple[str, ...] = (
    # Google / Firebase
    ".web.app", ".firebaseapp.com", ".appspot.com", ".run.app", ".page.link",
    # Cloudflare
    ".pages.dev", ".workers.dev", ".r2.dev",
    # 其他 PaaS
    ".vercel.app", ".netlify.app", ".herokuapp.com", ".onrender.com", ".glitch.me",
    ".replit.app", ".repl.co", ".replit.dev", ".fly.dev", ".up.railway.app", ".deno.dev",
    ".surge.sh", ".github.io", ".gitlab.io", ".gitbook.io", ".ondigitalocean.app",
    ".koyeb.app", ".render.com",
    # Microsoft / AWS
    ".azurewebsites.net", ".blob.core.windows.net", ".web.core.windows.net",
    ".azurestaticapps.net", ".s3.amazonaws.com", ".amplifyapp.com", ".cloudfront.net",
    # 網站產生器
    ".wixsite.com", ".weebly.com", ".webflow.io", ".notion.site", ".carrd.co",
    ".framer.app", ".framer.website", ".godaddysites.com", ".square.site",
    ".mystrikingly.com", ".strikingly.com", ".jimdosite.com", ".my.canva.site",
    ".webnode.tw", ".webnode.page", ".blogspot.com", ".wordpress.com", ".000webhostapp.com",
    ".weebly.site", ".site123.me", ".tilda.ws",
    # 中國雲物件儲存（中文詐騙落地頁常見）
    ".aliyuncs.com", ".myqcloud.com", ".myhuaweicloud.com",
)
# 路徑型免費平台（hostname 完全相等即算）
FREE_HOSTING_HOSTS: FrozenSet[str] = frozenset({
    "sites.google.com", "storage.googleapis.com", "s3.amazonaws.com",
})
# S3 區域端點：bucket.s3.ap-northeast-1.amazonaws.com / bucket.s3-website-us-east-1.amazonaws.com
FREE_HOSTING_REGEXES: Tuple[str, ...] = (
    r"\.s3[.-][a-z0-9-]+\.amazonaws\.com$",
    r"\.s3-website[.-][a-z0-9-]+\.amazonaws\.com$",
)

# 「雲端 / CDN / PaaS 基礎設施」主機（cloud_hosting 特徵；hostname 後綴比對）
CLOUD_HOSTING_SUFFIXES: Tuple[str, ...] = (
    ".cloudfront.net", ".azurefd.net", ".azureedge.net", ".pages.dev", ".workers.dev",
    ".ondigitalocean.app", ".cdn.cloudflare.net", ".vercel.app", ".netlify.app",
    ".github.io", ".firebaseapp.com", ".web.app", ".herokuapp.com", ".onrender.com",
    ".azurewebsites.net", ".blob.core.windows.net", ".r2.dev", ".amazonaws.com",
    ".appspot.com", ".run.app", ".aliyuncs.com", ".myqcloud.com", ".b-cdn.net",
    ".akamaized.net", ".fastly.net",
)

TUNNEL_SUFFIXES: Tuple[str, ...] = (
    ".trycloudflare.com",
    ".ngrok.io", ".ngrok-free.app", ".ngrok.app", ".ngrok.dev", ".ngrok-free.dev",
    ".loca.lt", ".localtunnel.me", ".serveo.net", ".serveousercontent.com",
    ".lhr.life", ".localhost.run", ".pinggy.link", ".pinggy.online", ".pinggy.io",
    ".devtunnels.ms", ".share.zrok.io", ".loclx.io", ".bore.pub", ".tunnelmole.net",
    # IPFS 閘道
    ".ipfs.dweb.link", ".ipfs.w3s.link", ".ipfs.nftstorage.link", ".ipfs.4everland.io",
)
TUNNEL_HOSTS: FrozenSet[str] = frozenset({
    "nftstorage.link", "gateway.pinata.cloud", "cf-ipfs.com", "ipfs.io", "dweb.link",
    "w3s.link", "4everland.io",
})
# IPFS 內容路徑（任何閘道）
IPFS_PATH_PATTERN = r"/ipfs/(?:qm[1-9a-hj-np-z]{44}|bafy[a-z2-7]{50,})"

# 群組邀請（比對對象：hostname + 解碼後 path，re.I）
SOCIAL_INVITE_PATTERNS: List[str] = [
    r"(?:^|\.)line\.me/(?:r/)?ti/g2?/",         # LINE 群組 /ti/g/、LINE 社群 /ti/g2/
    r"(?:^|\.)lin\.ee/[a-z0-9]",                # LINE 官方帳號短連結（需查證）
    r"(?:^|\.)t\.me/(?:\+|joinchat/)",          # Telegram 私人群組邀請
    r"(?:^|\.)telegram\.(?:me|dog)/(?:\+|joinchat/)",
    r"(?:^|\.)chat\.whatsapp\.com/[a-z0-9]",    # WhatsApp 群組邀請
    r"(?:^|\.)discord(?:app)?\.(?:gg|com/invite)/",
]
# 加好友／私訊（非群組）只作說明參考，不進特徵
SOCIAL_CONTACT_WEAK: List[str] = [
    r"(?:^|\.)line\.me/(?:r/)?ti/p/",
    r"(?:^|\.)liff\.line\.me/",
    r"(?:^|\.)wa\.me/",
    r"api\.whatsapp\.com/send",
]

# 廣告點擊／轉址服務（double_http 排除；內嵌目標網址請 main 另外評估）
REDIRECTOR_DOMAINS: FrozenSet[str] = frozenset({
    "doubleclick.net", "googleadservices.com", "googlesyndication.com", "google.com",
    "google.com.tw", "facebook.com", "instagram.com", "youtube.com", "line.me", "t.co",
    "bing.com", "yahoo.com", "linkedin.com", "twitter.com", "x.com", "reddit.com",
    "pinterest.com", "criteo.com", "adnxs.com", "taboola.com", "outbrain.com",
    "appier.net", "linksynergy.com", "awin1.com", "adform.net", "dartsearch.net",
    "clickcease.com", "shopee.tw", "momoshop.com.tw", "pchome.com.tw", "books.com.tw",
    "safelinks.protection.outlook.com", "outlook.com", "microsoft.com", "apple.com",
    "amazon.com", "line-apps.com", "lin.ee", "branch.io", "app.link", "onelink.me",
    "adjust.com", "appsflyer.com", "tiktok.com",
})


# =============================================================================
# 新聞／查核／論壇／百科等內容平台（中文詞只出現在路徑時不採計）
# =============================================================================

CONTENT_PLATFORM_DOMAINS: FrozenSet[str] = frozenset({
    "yahoo.com", "ettoday.net", "ltn.com.tw", "udn.com", "chinatimes.com", "cna.com.tw",
    "setn.com", "tvbs.com.tw", "storm.mg", "bnext.com.tw", "technews.tw", "ithome.com.tw",
    "cnyes.com", "ctee.com.tw", "mygopen.com", "tfc-taiwan.org.tw", "cofacts.tw", "life.tw",
    "vocus.cc", "pixnet.net", "dcard.tw", "ptt.cc", "medium.com", "wikipedia.org",
    "wikimedia.org", "mirrormedia.mg", "pts.org.tw", "businessweekly.com.tw", "cw.com.tw",
    "nownews.com", "newtalk.tw", "ebc.net.tw", "ftvnews.com.tw", "ctwant.com", "upmedia.mg",
    "thenewslens.com", "blocktempo.com", "abmedia.io", "zombit.info", "coindesk.com",
    "cointelegraph.com", "bbc.com", "cnn.com", "reuters.com", "nytimes.com", "line.me",
    "google.com", "google.com.tw", "bing.com", "moneydj.com", "cmoney.tw", "stockfeel.com.tw",
    "wealth.com.tw", "gvm.com.tw", "commonhealth.com.tw", "nextapple.com", "appledaily.com.tw",
    "hk01.com", "rti.org.tw", "fact-checker.line.me", "g0v.tw", "hackmd.io", "fraud.tw",
})


# =============================================================================
# 博弈 / 假投資 / 假交易所 / 一般誘因詞庫（features.py 與 HARD_RULES 共用）
# =============================================================================
# 比對原則：英文一律以 token（以非英數切分、去除前後數字）比對；
# substring_terms 為「夠長且專一」的詞，允許出現在複合 token 內（例如 luckycasino88）。
# 中文（zh）對 percent-decode 後字串做子字串比對；只出現在內容平台路徑時不採計。

GAMBLING_TERMS: Dict[str, Any] = {
    "strong_tokens": frozenset({
        "casino", "casinos", "baccarat", "bacarat", "roulette", "blackjack", "sportsbook",
        "sportsbet", "slot", "slots", "poker", "jackpot", "keno", "sabong", "cockfight",
        "betting", "bettor", "pk10", "bjl",
        # 拼音
        "yule", "yulecheng", "bocai", "caipiao", "duchang", "dubo", "baijiale", "laohuji",
        "qipai", "buyu", "zhenren", "liuhecai", "shishicai", "kuaisan", "feiting", "saiche",
        "pujing", "xinpujing", "jinsha", "weinisi", "weinisiren", "yongli", "huangguan",
        "taiyangcheng", "suncity", "yabo", "kaiyun", "leyu", "jiuyou",
        # 知名博弈品牌
        "dafa", "dafabet", "bet365", "188bet", "fun88", "w88", "1xbet", "betway", "12bet",
    }),
    # 出現在路徑時仍採計的博弈詞（夠專一，正常網站路徑幾乎不會出現）
    "path_tokens": frozenset({
        "casino", "baccarat", "bocai", "caipiao", "yulecheng", "baijiale", "laohuji", "qipai",
        "zhenren", "liuhecai", "shishicai", "sportsbook", "sportsbet",
    }),
    "substring_terms": (
        "casino", "roulette", "blackjack", "sportsbook", "sportsbet", "jackpot", "yulecheng",
        "bocai", "caipiao", "baijiale", "laohuji", "qipai", "zhenren", "liuhecai", "shishicai",
        "pujing", "weinisi", "huangguan", "taiyangcheng", "dafabet",
    ),
    "weak_tokens": frozenset({
        "bet", "bets", "win", "wins", "vip", "club", "lucky", "fortune", "royal", "bingo",
        "lottery", "lotto", "game", "games", "gaming", "sports", "sport", "tiyu", "dianjing",
        "ag", "bg", "dg", "sa", "pg", "pp", "mg", "pt", "jdb", "cq9", "ky", "evo", "wm", "og",
        "fc", "bc", "cp", "yl", "hg", "pj", "js", "vns", "ty", "dj", "mgm", "live", "play",
    }),
    # 數字 + 博弈字組合（對單一 hostname token）
    "combo_regex": r"^(?:bet|win|slot|casino|vip|yl|bc|hg|cp|lucky)\d{2,}$|^\d{2,}(?:bet|win|vip|casino|slot|club)$",
    "tlds": frozenset({"bet", "casino", "poker", "bingo", "lotto"}),
    "zh": (
        "娛樂城", "娛樂場", "線上娛樂", "真人視訊", "真人娛樂", "百家樂", "老虎機", "電子遊藝",
        "捕魚機", "棋牌", "博弈", "博彩", "賭場", "賭城", "下注", "押注", "球版", "包網",
        "體驗金", "註冊送", "首儲", "首存", "儲值送", "返水", "反水", "秒出金", "百家樂代操",
        "帶牌", "時時彩", "北京賽車", "幸運飛艇", "六合彩", "骰寶", "德州撲克", "麻將胡了",
        "戰神賽特", "雷神之鎚", "威尼斯人", "葡京", "太陽城", "皇冠體育", "亞博", "開雲",
        "娱乐城", "百家乐", "老虎机", "电子游艺", "真人视讯", "彩票", "时时彩", "北京赛车",
        "幸运飞艇", "体育投注", "赌场", "首充", "注册送", "体验金", "出款", "太阳城", "亚博", "开云",
    ),
}

INVESTMENT_TERMS: Dict[str, Any] = {
    "strong_tokens": frozenset({
        "ipo", "forex", "xau", "xag", "quant", "aitrade", "aitrading", "copytrade",
        "copytrading", "mt4", "mt5", "metatrader", "cfd", "stockvip", "vipstock",
        # 拼音
        "gupiao", "tougu", "licai", "touzi", "qihuo", "waihui", "huangjin", "lianghua",
        "biaogu", "feigu", "mingpai", "daidan", "gendan", "chouqian", "shengou", "zhengquan",
        "quanshang", "dangchong", "jijin",
    }),
    # 核心金融詞（需搭配 lure 才算）
    "core_tokens": frozenset({
        "stock", "stocks", "invest", "investment", "investing", "trade", "trading", "trader",
        "broker", "fx", "futures", "fund", "funds", "capital", "wealth", "asset", "assets",
        "profit", "dividend", "gold", "oil", "nasdaq", "nyse", "etf", "securities", "finance",
        "markets", "forex", "quant", "ipo", "xau", "cfd",
    }),
    # 誘因詞（與核心詞以 - 或 . 分隔、同時出現於 hostname 才算，如 stock-vip、invest-tw-mentor）
    "lure_tokens": frozenset({
        "vip", "vvip", "svip", "pro", "ai", "club", "signal", "signals", "mentor", "teacher",
        "laoshi", "master", "king", "plus", "elite", "bonus", "profit",
    }),
    # 可與核心詞「黏在同一個 token」的誘因詞（stockvip、aistock）；刻意不含 group/master/win
    # 等字，避免 capitalgroup、trademaster、goldwin 這類合法品牌被誤判。
    "compound_lure_tokens": frozenset({
        "vip", "vvip", "svip", "ai", "signal", "signals", "mentor", "teacher", "laoshi",
    }),
    "zh": (
        "飆股", "明牌", "領取飆股", "免費飆股", "股票群組", "投資群組", "投顧老師", "老師帶單",
        "帶單", "跟單", "喊單", "當沖老師", "隔日沖", "保證中籤", "新股申購", "專用通道",
        "法人通道", "法人席位", "內部額度", "AI選股", "AI量化", "量化交易", "智能交易",
        "機器人交易", "高勝率", "保證獲利", "穩賺不賠", "月報酬", "日收益", "財富自由",
        "被動收入", "仙股", "港股仙股", "代操", "解凍金", "繳稅出金", "刷單", "搶單", "做單",
        "荐股", "牛股", "打新股", "中签", "跟单", "带单", "稳赚", "刷单", "抢单", "配资",
        "飙股", "保证获利",
    ),
}

CRYPTO_TERMS: Dict[str, Any] = {
    "strong_tokens": frozenset({
        "usdt", "usdc", "trc20", "erc20", "bep20", "defi", "web3", "dex", "cex", "otc", "p2p",
        "airdrop", "staking", "cloudmining", "hashrate", "walletconnect", "dapp", "drainer",
        "secondcontract", "jiaoyisuo", "jys", "qianbao", "wakuang", "kuangji", "zhiya",
        "kongtou", "heyue", "ex",
    }),
    "weak_tokens": frozenset({
        "exchange", "swap", "coin", "coins", "crypto", "token", "tokens", "btc", "bitcoin",
        "eth", "ether", "bnb", "xrp", "doge", "sol", "nft", "stake", "mining", "miner", "pool",
        "liquidity", "yield", "claim", "reward", "rewards", "wallet", "connect", "bridge",
        "futures", "contract", "perp", "perpetual", "leverage", "sync", "verify", "recover",
        "revoke", "validate", "bit",
    }),
    # 允許出現在複合 token 內的加密詞（≥4 字）
    "substring_terms": ("crypto", "bitcoin", "wallet", "mining", "token", "usdt", "defi", "web3", "coin"),
    "ex_suffix_regex": r"^[a-z0-9]{3,}ex$",
    "ex_exceptions": frozenset({
        "apex", "index", "latex", "rolex", "fedex", "timex", "kleenex", "durex", "pyrex",
        "amex", "simplex", "complex", "duplex", "convex", "vortex", "vertex", "cortex", "codex",
        "reflex", "annex", "alex", "flex", "perplex", "telex", "dex", "bitmex", "spandex",
        "cineplex", "multiplex", "tex", "rex", "lex", "vex", "hex", "yandex", "playtex",
        "unilex", "memex", "biotex", "goretex", "ibex", "murex", "silex", "sanex", "solex",
        "phemex", "remex", "medex", "vapex", "nanoflex", "airflex", "softex", "cosmex",
    }),
    "zh": (
        "虛擬貨幣", "虛擬幣", "加密貨幣", "數位貨幣", "泰達幣", "U幣", "幣商", "換U", "提幣",
        "充幣", "冷錢包", "雲挖礦", "雲端挖礦", "礦機", "礦池", "質押", "流動性挖礦", "空投",
        "領取空投", "秒合約", "量化機器人", "老用戶專屬", "数字货币", "虚拟币", "挖矿", "矿机",
        "质押", "秒合约", "充币", "提币",
    ),
}

# has_scam_word：一般詐騙誘因詞（token 比對）
SCAM_WORD_TOKENS: FrozenSet[str] = frozenset({
    "invest", "profit", "earn", "bonus", "prize", "lucky", "crypto", "forex", "slot",
    "casino", "bet", "vip", "loan", "claim", "reward", "rewards", "wallet", "web3", "usdt",
    "btc", "eth", "defi", "xau", "cfd", "trading", "keygen", "airdrop", "giveaway", "jackpot",
})
# 需相鄰出現的英文詞組（token bigram）
SCAM_WORD_BIGRAMS: Tuple[Tuple[str, str], ...] = (
    ("free", "download"), ("lifetime", "license"), ("office", "free"), ("excel", "free"),
    ("office", "crack"), ("office", "activate"), ("serial", "key"), ("free", "gift"),
)
SCAM_WORD_ZH: Tuple[str, ...] = (
    "中獎", "免費領", "賺錢", "博弈", "外匯", "穩賺", "保證獲利", "虛擬貨幣", "加密貨幣",
    "免費下載", "破解版", "序號產生", "激活碼", "永久授權", "日賺", "月入", "領獎",
    "紅包領取", "免費送", "投資獲利", "高報酬", "被動收入", "中奖", "赚钱", "稳赚", "免费领",
)


# newly_registered_like 用：SLD 含這些有意義字詞（≥4 字以子字串、≤3 字以 token 比對）
# 就不視為隨機新網域（pxmart 的 mart、ctbcbank 的 bank…）。
MEANINGFUL_WORDS: FrozenSet[str] = frozenset({
    "shop", "store", "mall", "mart", "bank", "news", "media", "tech", "info", "mail", "cloud",
    "host", "game", "play", "book", "books", "deal", "sale", "price", "order", "cart",
    "market", "trade", "invest", "crypto", "stock", "gold", "plus", "world", "global",
    "group", "home", "life", "live", "love", "care", "food", "travel", "tour", "hotel",
    "land", "house", "city", "town", "design", "studio", "photo", "video", "music", "film",
    "sport", "fitness", "health", "school", "learn", "service", "system", "soft", "data",
    "link", "star", "moon", "light", "smart", "power", "energy", "auto", "motor", "bike",
    "coffee", "wine", "beer", "fashion", "style", "beauty", "skin", "hair", "baby", "kids",
    "garden", "green", "black", "white", "silver", "line", "tube", "face", "gram", "business",
    "weekly", "daily", "times", "post", "press", "money", "finance", "capital", "fund",
    "lab", "labs", "hub", "app", "web", "net", "pay", "buy", "fit", "pet", "dog", "cat",
    "car", "tea", "sun", "art", "box", "fun", "job", "jobs", "work", "office", "online",
    "digital", "network", "software", "computer", "phone", "mobile", "print", "craft",
    "street", "park", "river", "ocean", "mountain", "forest", "flower", "sweet", "fresh",
    "first", "best", "super", "magic", "happy", "lucky", "royal", "prime", "pro",
    "schwab", "twitch", "strength", "rhythm",
    "club",   # v7.1：kfcclub、golfclub 這類「縮寫＋club」品牌
})


# =============================================================================
# 品牌 → 官方 registered domain（brand_impersonation / brand_typo_like / 硬規則共用）
# =============================================================================
# key 為 hostname token（小寫英數）；只列有把握的官方網域。
# max / gate / ace / core / edge / axi 這類短品牌名誤判太多，刻意不列為品牌 token。

_ECOM_SOCIAL_OFFICIAL: Dict[str, Set[str]] = {
    "shopee": {"shopee.tw", "shopee.com", "shopee.sg", "shopee.com.my", "shopee.ph",
               "shopee.co.id", "shopee.vn", "shopee.co.th", "shopee.com.br", "shopee.com.mx",
               "shopee.com.co", "shopee.cl", "shopee.cn", "shp.ee", "shopeemobile.com", "shopee.io"},
    "lazada": {"lazada.com", "lazada.sg", "lazada.com.my", "lazada.co.th", "lazada.com.ph",
               "lazada.co.id", "lazada.vn"},
    "momo": {"momoshop.com.tw", "momo.com.tw", "momomall.com.tw", "momo.vn"},
    "momoshop": {"momoshop.com.tw"},
    "pchome": {"pchome.com.tw"},
    "poya": {"poyabuy.com.tw", "poya.com.tw"},
    "poyabuy": {"poyabuy.com.tw"},
    "cosmed": {"cosmed.com.tw"},
    "watsons": {"watsons.com.tw", "watsons.com", "watsons.com.my", "watsons.com.sg",
                "watsons.com.hk", "watsons.co.th", "watsons.com.ph", "watsons.co.id", "watsons.vn"},
    "tiktok": {"tiktok.com", "tiktokv.com", "tiktokcdn.com", "tiktokshop.com", "tiktokglobalshop.com"},
    "facebook": {"facebook.com", "fb.com", "fb.me", "fbcdn.net", "facebook.net", "messenger.com",
                 "meta.com", "facebookmail.com"},
    "instagram": {"instagram.com", "cdninstagram.com", "ig.me", "threads.net", "threads.com"},  # Threads 為 Meta 官方
    "line": {"line.me", "lin.ee", "line-apps.com", "line-scdn.net", "linecorp.com",
             "linebank.com.tw", "line.biz", "linetv.tw", "lycorp.co.jp"},
    "apple": {"apple.com", "icloud.com", "apple.co", "me.com", "apple.news", "mzstatic.com",
              "cdn-apple.com"},
    "google": {"google.com", "google.com.tw", "gstatic.com", "youtube.com", "g.co", "forms.gle",
               "g.page", "googlevideo.com", "withgoogle.com", "google.dev", "android.com",
               "googleapis.com", "googleusercontent.com"},
    "microsoft": {"microsoft.com", "live.com", "office.com", "office365.com", "outlook.com",
                  "microsoftonline.com", "windows.com", "azure.com", "msn.com", "bing.com",
                  "xbox.com", "skype.com", "aka.ms", "microsoft365.com", "msauth.net", "msft.net",
                  "visualstudio.com", "windows.net", "sharepoint.com"},
    "amazon": {"amazon.com", "amazonaws.com", "amzn.to", "a2z.com", "media-amazon.com",
               "ssl-images-amazon.com", "primevideo.com", "amazon.dev", "awsstatic.com"},
    "paypal": {"paypal.com", "paypal.me", "paypalobjects.com", "paypal-community.com"},
    "netflix": {"netflix.com", "netflix.net", "nflxext.com", "nflximg.net", "nflxvideo.net"},
    "yahoo": {"yahoo.com", "yahoo.co.jp", "yimg.com", "yahoo.net", "yahooapis.com", "yahoo.com.tw"},
    "youtube": {"youtube.com", "youtu.be", "ytimg.com", "youtube-nocookie.com", "googlevideo.com"},
    "walmart": {"walmart.com", "walmart.ca", "walmartimages.com", "walmart.com.mx", "wal-mart.com"},
    "costco": {"costco.com", "costco.com.tw", "costco.co.jp", "costco.ca", "costco.co.uk",
               "costco.com.au", "costco.co.kr"},
    "rakuten": {"rakuten.com.tw", "rakuten.co.jp", "rakuten.com", "rakuten-bank.com.tw", "rakuten.tw"},
    "klook": {"klook.com"},
    "tokopedia": {"tokopedia.com", "tokopedia.net"},
    "taobao": {"taobao.com", "tmall.com", "alibaba.com", "alicdn.com", "1688.com"},
    "tmall": {"tmall.com", "tmall.hk"},
    "alibaba": {"alibaba.com", "alibabacloud.com", "aliyun.com", "alibaba-inc.com", "alicdn.com",
                "1688.com", "alibabagroup.com"},
    "aliexpress": {"aliexpress.com", "aliexpress.us", "aliexpress.ru"},
    "ebay": {"ebay.com", "ebay.co.uk", "ebay.de", "ebay.com.au", "ebay.ca", "ebay.fr", "ebay.it",
             "ebayimg.com"},
    "shein": {"shein.com", "shein.tw", "sheingroup.com", "shein.com.hk"},
    "ruten": {"ruten.com.tw"},
    "myship": {"7-11.com.tw"},
    "newebpay": {"newebpay.com"},
    "ecpay": {"ecpay.com.tw"},
    "jkopay": {"jkopay.com", "jkos.com"},
    "familymart": {"family.com.tw", "familymart.com.tw", "familymart.co.jp"},
}

_FINANCE_OFFICIAL: Dict[str, Set[str]] = {
    "ctbc": {"ctbcbank.com", "ctbcsec.com", "ctbcholding.com", "ctbcinvestments.com",
             "ctbclife.com", "chinatrust.com.tw", "ctbcinsurance.com"},
    "ctbcbank": {"ctbcbank.com"},
    "chinatrust": {"chinatrust.com.tw", "ctbcbank.com"},
    "esun": {"esunbank.com", "esunbank.com.tw", "esunsec.com.tw", "esunfhc.com", "esunfhc.com.tw",
             "esun.co", "esunlife.com.tw", "esb.page.link"},
    "esunbank": {"esunbank.com", "esunbank.com.tw"},
    "cathay": {"cathaybk.com.tw", "cathaysec.com.tw", "cathaylife.com.tw", "cathay-cube.com.tw",
               "cathayholdings.com", "cathayholdings.com.tw", "cathayfut.com.tw",
               "cathay-ins.com.tw", "cathaysite.com.tw", "cathaypacific.com"},
    "cathaybk": {"cathaybk.com.tw"},
    "fubon": {"fubon.com", "fubon.com.tw", "fbs.com.tw", "fubonlife.com.tw", "taipeifubon.com.tw",
              "fubon-ins.com.tw", "fubonfutures.com.tw"},
    "yuanta": {"yuanta.com.tw", "yuanta.com", "yuantabank.com.tw", "yuantafutures.com.tw",
               "yuantafunds.com", "yuantaetfs.com", "yuantalife.com.tw"},
    "sinopac": {"sinopac.com", "sinotrade.com.tw", "sinopacholdings.com"},
    "sinotrade": {"sinotrade.com.tw"},
    "kgi": {"kgi.com.tw", "kgi.com", "kgibank.com.tw", "kgieworld.com.tw", "kgif.com.tw", "kgilife.com.tw"},
    "megabank": {"megabank.com.tw", "emega.com.tw", "megaholdings.com.tw", "megafutures.com.tw"},
    "taishin": {"taishinbank.com.tw", "taishinholdings.com.tw", "richart.tw", "tssco.com.tw"},
    "richart": {"richart.tw", "taishinbank.com.tw"},
    "firstbank": {"firstbank.com.tw"},
    "hncb": {"hncb.com.tw"},
    "landbank": {"landbank.com.tw"},
    "skbank": {"skbank.com.tw"},
    "scsb": {"scsb.com.tw"},
    "ubot": {"ubot.com.tw"},
    "nextbank": {"nextbank.com.tw"},
    "twse": {"twse.com.tw"},
    "tpex": {"tpex.org.tw"},
    "taifex": {"taifex.com.tw"},
    "tdcc": {"tdcc.com.tw"},
    "masterlink": {"masterlink.com.tw"},
    "tsmc": {"tsmc.com", "tsmc.com.tw"},
    "citi": {"citi.com", "citibank.com", "citibank.com.tw", "citigroup.com"},
    "citibank": {"citi.com", "citibank.com", "citibank.com.tw"},
    "hsbc": {"hsbc.com", "hsbc.com.tw", "hsbc.com.hk", "hsbc.co.uk"},
    "cmegroup": {"cmegroup.com"},
    "nasdaq": {"nasdaq.com"},
    "blackrock": {"blackrock.com"},
    "etoro": {"etoro.com"},
    "robinhood": {"robinhood.com"},
    "bakkt": {"bakkt.com"},
}

_CRYPTO_OFFICIAL: Dict[str, Set[str]] = {
    "binance": {"binance.com", "binance.us", "bnbchain.org"}, "okx": {"okx.com"}, "okex": {"okx.com"},
    "bybit": {"bybit.com"}, "bitget": {"bitget.com"}, "kucoin": {"kucoin.com"},
    "coinbase": {"coinbase.com"}, "kraken": {"kraken.com"}, "mexc": {"mexc.com"},
    "htx": {"htx.com"}, "huobi": {"htx.com", "huobi.com"}, "bitfinex": {"bitfinex.com"},
    "bitflyer": {"bitflyer.com"}, "upbit": {"upbit.com"}, "bingx": {"bingx.com"},
    "maicoin": {"maicoin.com"}, "bitopro": {"bitopro.com"}, "hoyabit": {"hoyabit.com"},
    "xrex": {"xrex.io"}, "metamask": {"metamask.io"}, "trustwallet": {"trustwallet.com"},
    "tronlink": {"tronlink.org"}, "imtoken": {"token.im"}, "tokenpocket": {"tokenpocket.pro"},
    "safepal": {"safepal.com"}, "trezor": {"trezor.io"}, "uniswap": {"uniswap.org"},
    "pancakeswap": {"pancakeswap.finance"}, "opensea": {"opensea.io"},
    "etherscan": {"etherscan.io"}, "bscscan": {"bscscan.com"}, "tronscan": {"tronscan.org"},
    "coinmarketcap": {"coinmarketcap.com"}, "coingecko": {"coingecko.com"},
    "walletconnect": {"walletconnect.com", "walletconnect.network", "reown.com"},
    "coinw": {"coinw.com"}, "bitmart": {"bitmart.com"}, "phemex": {"phemex.com"},
}

_GOV_OFFICIAL: Dict[str, Set[str]] = {
    "165": {"npa.gov.tw", "165dashboard.tw"},
    "npa": {"npa.gov.tw"},
    "mof": {"mof.gov.tw"},
    "moi": {"moi.gov.tw"},
    "nhi": {"nhi.gov.tw"},
    "fsc": {"fsc.gov.tw"},
    "cib": {"npa.gov.tw"},
    "etax": {"nat.gov.tw"},
}

BRAND_OFFICIAL_DOMAINS: Dict[str, Set[str]] = {
    **_ECOM_SOCIAL_OFFICIAL, **_FINANCE_OFFICIAL, **_CRYPTO_OFFICIAL, **_GOV_OFFICIAL,
}

# 品牌分類（硬規則訊息用）
BRAND_CATEGORIES: Dict[str, str] = {
    **{b: "ecommerce" for b in _ECOM_SOCIAL_OFFICIAL},
    **{b: "finance" for b in _FINANCE_OFFICIAL},
    **{b: "crypto" for b in _CRYPTO_OFFICIAL},
    **{b: "gov" for b in _GOV_OFFICIAL},
}

# 這些品牌在許多國家 ccTLD 都有官方站（google.co.jp、amazon.de…）：
# SLD 等於品牌且 TLD 為低風險 2 字母 ccTLD 時視為官方。
MULTI_CC_BRANDS: FrozenSet[str] = frozenset({
    "google", "amazon", "apple", "microsoft", "yahoo", "shopee", "lazada", "ebay", "paypal",
    "costco", "watsons", "hsbc", "rakuten", "netflix", "facebook", "tiktok", "citibank",
})

# 本身是常見英文字的品牌：token 完全相等不算，必須帶前後綴/數字或高風險 TLD
BRAND_GENERIC_WORDS: FrozenSet[str] = frozenset({"line", "apple", "amazon", "momo", "citi", "shein"})

# 品牌 + 這些前後綴（可再接數字）= 詐騙常見組合（shopeemall、binance-tw、fubon-vip、kgi-vip888）
BRAND_AFFIX_TOKENS: FrozenSet[str] = frozenset({
    "tw", "taiwan", "vip", "vvip", "svip", "pro", "plus", "mall", "shop", "shops", "shopping",
    "store", "buy", "sale", "sales", "official", "app", "apps", "online", "global", "web",
    "web3", "defi", "btc", "usdt", "xau", "ex", "swap", "coin", "coins", "bank", "ebank",
    "netbank", "pay", "login", "signin", "verify", "verification", "secure", "security",
    "service", "services", "support", "help", "wallet", "trade", "trading", "invest",
    "futures", "exchange", "market", "markets", "finance", "fund", "funds", "securities",
    "sec", "stock", "stocks", "club", "group", "card", "cards", "loan", "loans", "refund",
    "bonus", "reward", "rewards", "gift", "gifts", "event", "promo", "claim", "airdrop", "h5",
    "asia", "hk", "cn", "us", "int", "ltd", "account", "member", "recover", "recovery", "sync",
    "connect", "update", "auth", "kyc", "kefu", "customer", "tax", "gov", "npa",
    "police", "fraud", "antifraud", "ec", "id", "hd", "center", "centre", "portal", "mobile",
    "cash", "money", "gold", "ai", "quant", "mining", "staking", "earn", "subsidy", "parcel",
    "payment", "notice", "fine", "penalty",
})
BRAND_PREFIX_TOKENS: FrozenSet[str] = frozenset({"tw", "vip", "my", "official", "secure", "login", "web", "app"})

# v7.1 黏字切分（features._segment_signatures）：詞庫詞以外，允許出現在同一個 token 裡的「填充詞」
# （biaogu+vip、188+bet+tw、kucoin+exchange+vip、tw+licai+plus）。只當填充，單獨出現不構成任何訊號。
SEGMENT_AFFIX_TOKENS: FrozenSet[str] = BRAND_AFFIX_TOKENS | BRAND_PREFIX_TOKENS | frozenset({
    "king", "kings", "best", "top", "go", "new", "hub", "live", "win", "world", "zone", "station",
    "life", "net", "star", "lucky", "royal", "asia", "88", "168", "666", "888",
})

# 明確的品牌拼寫變形（token 前綴比對；不含正牌名稱）
BRAND_TYPO_TERMS: Dict[str, str] = {
    "shoppe": "shopee", "shopeee": "shopee", "shoppee": "shopee", "sh0pee": "shopee",
    "lazadaa": "lazada", "faebook": "facebook", "faceb00k": "facebook", "facebok": "facebook",
    "fecebook": "facebook", "micros0ft": "microsoft", "m1crosoft": "microsoft",
    "mircosoft": "microsoft", "microsfot": "microsoft", "poyabyu": "poyabuy",
    "igshop": "instagram", "igproduct": "instagram", "tikmall": "tiktok", "walmar": "walmart",
    "amaz0n": "amazon", "amazom": "amazon", "g00gle": "google", "gooogle": "google",
    "paypa1": "paypal", "paypai": "paypal", "yah00": "yahoo", "netfiix": "netflix",
    "binanc": "binance", "binnace": "binance", "bianance": "binance", "coinbse": "coinbase",
    "metamsk": "metamask", "okexx": "okex", "bybitt": "bybit", "kucoln": "kucoin",
}

# 與短品牌編輯距離 1 的常見英文字：不視為拼字仿冒（live≈line、memo≈momo、pola≈poya）
LEVENSHTEIN_COMMON_WORDS: FrozenSet[str] = frozenset({
    "live", "love", "like", "lime", "link", "lane", "lone", "lint", "lion", "lina", "lino",
    "linx", "fine", "mine", "nine", "pine", "wine", "dine", "vine", "kine", "lines", "liner",
    "mono", "memo", "moto", "mojo", "demo", "mama", "momi", "moms", "polo", "pola", "pole",
    "poly", "soya", "toya", "roya", "yoga", "apply", "ample", "appl", "maple", "yahoos",
    "amazin", "amazing", "goggle", "googly", "watson", "costa", "cosmo", "cosmos",
    "klock", "lazed", "tiktak", "okey", "obex",
})

# Levenshtein 比對用品牌清單（舊版 25 個 + 主要交易所/台灣金融品牌）
BRAND_NAME_LIST: List[str] = [
    "apple", "google", "microsoft", "amazon", "paypal",
    "facebook", "instagram", "line", "shopee", "momo",
    "pchome", "yahoo", "netflix", "youtube", "lazada",
    "poya", "cosmed", "watsons", "tiktok", "tokopedia",
    "walmart", "allegro", "rakuten", "klook", "costco",
    "binance", "coinbase", "kucoin", "bybit", "bitget", "metamask", "maicoin", "bitopro",
    "huobi", "okex", "cathaybk", "esunbank", "ctbcbank", "sinopac", "yuanta", "fubon",
    "megabank", "taishin", "firstbank", "landbank",
]


# =============================================================================
# 白名單
# =============================================================================

TRUSTED_DOMAINS: Set[str] = {
    "google.com", "google.com.tw", "facebook.com", "instagram.com", "youtube.com", "line.me",
    "shopee.tw", "shopee.com", "shopee.sg", "shopee.com.my", "momo.com.tw", "momoshop.com.tw",
    "pchome.com.tw", "apple.com", "microsoft.com", "amazon.com", "netflix.com", "yahoo.com",
    "costco.com.tw", "books.com.tw", "rakuten.com.tw", "klook.com", "twse.com.tw",
    "165.npa.gov.tw", "165dashboard.tw",

    # --- 零售／藥妝／支付（官方網域；舊版 brand 規則曾誤判） ---
    "poyabuy.com.tw",         # 寶雅
    "cosmed.com.tw",          # 康是美
    "watsons.com.tw",         # 屈臣氏
    "ruten.com.tw",           # 露天
    "pxmart.com.tw",          # 全聯
    "7-11.com.tw",            # 統一超商（含 myship 賣貨便）
    "family.com.tw",          # 全家
    "newebpay.com",           # 藍新金流（165 最常被冒用的品牌）
    "ecpay.com.tw",           # 綠界
    "jkopay.com",             # 街口
    "easycard.com.tw",        # 悠遊卡

    # --- 台灣銀行／金融機構官方網域 ---
    # 這些網域的匯率查詢、外匯換算、投資理財頁面本來就會合法出現
    # forex / rate / crypto / invest 等字詞，因此明確列入白名單。
    "esunbank.com",           # 玉山銀行
    "esunbank.com.tw",
    "esun.co",                # 玉山官方短網址
    "esb.page.link",          # 玉山官方短網址
    "ctbcbank.com",           # 中國信託
    "cathaybk.com.tw",        # 國泰世華
    "cathay-cube.com.tw",     # 國泰世華 CUBE
    "firstbank.com.tw",       # 第一銀行
    "tcb-bank.com.tw",        # 合作金庫
    "megabank.com.tw",        # 兆豐銀行
    "taishinbank.com.tw",     # 台新銀行
    "richart.tw",             # 台新 Richart
    "sinopac.com",            # 永豐銀行
    "hncb.com.tw",            # 華南銀行
    "landbank.com.tw",        # 土地銀行
    "bot.com.tw",             # 臺灣銀行
    "ubot.com.tw",            # 聯邦銀行
    "scsb.com.tw",            # 上海商業儲蓄銀行（舊註解誤植為新光銀行）
    "skbank.com.tw",          # 新光銀行
    "sunnybank.com.tw",       # 陽信銀行
    "bankchb.com",            # 彰化銀行
    "feib.com.tw",            # 遠東商銀
    "yuantabank.com.tw",      # 元大銀行
    "kgibank.com.tw",         # 凱基銀行
    "o-bank.com",             # 王道銀行
    "tbb.com.tw",             # 臺灣企銀
    "linebank.com.tw",        # LINE Bank
    "nextbank.com.tw",        # 將來銀行
    "rakuten-bank.com.tw",    # 樂天國際商銀
    "hsbc.com.tw",            # 匯豐台灣
    "dbs.com.tw",             # 星展台灣
    "citibank.com.tw",        # 花旗台灣
    "fubon.com",              # 富邦金控／台北富邦銀行
    "post.gov.tw",            # 中華郵政
    "cbc.gov.tw",             # 中央銀行
    "fsc.gov.tw",             # 金融監督管理委員會

    # --- 證券／期貨／交易所與周邊 ---
    "yuanta.com.tw", "yuanta.com", "fbs.com.tw", "sinotrade.com.tw", "kgi.com.tw",
    "cathaysec.com.tw", "capital.com.tw", "masterlink.com.tw", "pscnet.com.tw",
    "entrust.com.tw", "emega.com.tw", "ctbcsec.com", "esunsec.com.tw",
    "capitalfutures.com.tw", "yuantafutures.com.tw", "megafutures.com.tw",
    "concordfutures.com.tw", "tsfutures.com.tw", "cathayfut.com.tw", "spf.com.tw",
    "kgif.com.tw", "ibff.com.tw", "pfcf.com.tw", "dcnf.com.tw",
    "tpex.org.tw", "taifex.com.tw", "tdcc.com.tw", "sitca.org.tw", "twsa.org.tw",
    "sfipc.org.tw",

    # --- 知名合法加密貨幣交易所／錢包／行情站（官方網域本身不是詐騙） ---
    "binance.com", "okx.com", "bybit.com", "bitget.com", "kucoin.com", "coinbase.com",
    "kraken.com", "maicoin.com", "bitopro.com", "hoyabit.com", "xrex.io", "metamask.io",
    "etherscan.io", "coingecko.com", "coinmarketcap.com", "tokenpocket.pro", "trustwallet.com",
}

# 以這些後綴結尾的 hostname 一律視為可信（政府／學術／軍方）
TRUSTED_SUFFIXES: Tuple[str, ...] = (".gov.tw", ".edu.tw", ".mil.tw", ".gov.taipei")

# 白名單網域即使命中，若網址帶有以下強烈詐騙信號，仍會送進完整預測流程（自 main.py 移入）。
# 注意：'forex' 刻意不列 —— 銀行官網的匯率查詢頁本來就會合法出現。
WHITELIST_OVERRIDE_SIGNALS: Tuple[str, ...] = (
    "free-download", "crack", "keygen", "serial-",
    "office-free", "excel-free", "office-crack", "office-activate",
    "usdt", "btc", "crypto", "claim", "prize",
    "免費下載", "破解", "序號", "激活",
)

# 白名單平台上的高風險內容（re.I；請對 canonical 與 decoded 網址各比對一次）
WHITELIST_OVERRIDE_PATTERNS: List[str] = [
    r"testflight\.apple\.com/join/",                      # 假投資 App 側載
    r"^itms-services:",                                   # 企業簽安裝
    r"docs\.google\.com/forms/", r"//forms\.gle/",        # 釣魚表單
    r"//sites\.google\.com/",                             # 免費架站
    r"drive\.google\.com/.+\.(?:apk|ipa)(?:[?#&]|$)",
    r"^[^?#]*\.(?:apk|ipa|mobileconfig)(?:[?#]|$)",
    r"//(?:[a-z0-9-]+\.)*line\.me/(?:r/)?ti/g2?/",        # LINE 群組／社群邀請
    r"//lin\.ee/[a-z0-9]",                                # LINE 官方帳號短連結
    r"(?:facebook|instagram)\.com/.+/(?:posts|groups)/.*(?:飆股|投資|娛樂城|百家樂)",
    # 內嵌跳轉到高風險 TLD（google.com/url?q=https://xxx.top/）
    r"(?:[?&][a-z0-9_.-]*=|/)https?://(?:[^/?#&@]*@)?[a-z0-9.-]*\.(?:" + _TIER_ANY_ALT + r")(?::\d+)?(?:[/?#&]|$)",
]


# =============================================================================
# 硬規則用的共用片段
# =============================================================================

_ALL_OFFICIAL: Set[str] = set(TRUSTED_DOMAINS)
for _domains in BRAND_OFFICIAL_DOMAINS.values():
    _ALL_OFFICIAL |= _domains
_ALL_OFFICIAL_ALT = _alt(_ALL_OFFICIAL)

# hostname 不是任何官方/可信網域（或其子網域），也不是政府/學術網域
_NOT_OFFICIAL_HOST = (
    r"^(?!(?:[a-z0-9-]+\.)*(?:" + _ALL_OFFICIAL_ALT + r")$)"
    r"(?!(?:[a-z0-9-]+\.)*(?:gov|edu|mil|ac|go|gob|gouv|govt)\.[a-z]{2,3}$)"
    r"(?!(?:[a-z0-9-]+\.)*(?:gov|edu|mil)$)"
    r"(?!(?:[a-z0-9-]+\.)*(?:" + _alt(MULTI_CC_BRANDS) + r")\.(?:com?\.)?"
    r"(?!(?:" + "|".join(_RISKY_CCTLDS) + r")$)[a-z]{2}$)"
)

# 新聞／查核／政府／教育網域（中文詞規則排除）：放在 scheme:// 之後的 authority lookahead
_NOT_CONTENT_AUTHORITY = (
    r"(?![^/?#]*?(?<![a-z0-9-])(?:" + _alt(CONTENT_PLATFORM_DOMAINS) + r")(?::\d+)?(?:[/?#]|$))"
    r"(?![^/?#]*\.(?:gov|edu)\.tw\.?(?::\d+)?(?:[/?#]|$))"
)
_URL_HEAD = r"^[a-z][a-z0-9+.-]*://"
_HOST_IN_URL = r"(?:[^/?#@]*@)?"   # 略過 userinfo


def _brand_spoof_pattern(brands: Iterable[str]) -> str:
    brands = sorted(set(brands))
    br_all = _alt(brands)
    br_long5 = _alt(b for b in brands if len(b) >= 5)
    br_long6 = _alt(b for b in brands if len(b) >= 6 and b not in BRAND_GENERIC_WORDS)
    affix = _alt(BRAND_AFFIX_TOKENS)
    prefix = _alt(BRAND_PREFIX_TOKENS)
    alternatives = [
        # A1：品牌 + 前後綴（binance-tw、shopeemall、fubon-vip、kgi-vip888）
        r"(?<![a-z0-9])(?:" + br_all + r")-?(?:" + affix + r")\d*(?![a-z])",
        # A2：品牌 + 數字（fubon168；短品牌不適用，避免 line6 這類合法品牌）
        r"(?<![a-z0-9])(?:" + br_long5 + r")\d+(?![a-z])",
        # B：前綴 + 品牌（tw-binance、vip-okx）
        r"(?<![a-z0-9])(?:" + prefix + r")-?(?:" + br_all + r")\d*(?![a-z])",
        # C：品牌直接換高風險 TLD（pchome.shop、binance.top）
        r"(?<![a-z0-9-])(?:" + br_all + r")\.(?:" + _TIER_ANY_ALT + r")$",
    ]
    if br_long6:
        # D：長品牌 + 1～3 個亂碼字母 + 後綴（kucoinx-futures）
        alternatives.append(
            r"(?<![a-z0-9])(?:" + br_long6 + r")[a-z]{1,3}-(?:" + affix + r")\d*(?![a-z])"
        )
    return _NOT_OFFICIAL_HOST + r".*?(?:" + "|".join(alternatives) + r")"


def _brands_of(category: str) -> List[str]:
    return [b for b, c in BRAND_CATEGORIES.items() if c == category]


# 截斷型拼字變形（walmar、faceboo）只對 ≥7 字且非一般英文字的品牌啟用
BRAND_TRUNCATION_TARGETS: Tuple[str, ...] = (
    "facebook", "microsoft", "instagram", "netflix", "youtube", "walmart", "binance",
    "coinbase", "metamask", "tokopedia", "rakuten", "bitopro", "maicoin",
)
# 品牌中間插入連字號（binanc-e、c-oinbase）：只取其中一側 ≤ 2 字的切法，避免 pc-home 這類一般詞
BRAND_HYPHEN_EXCEPTIONS: FrozenSet[str] = frozenset({"pc-home", "cost-co"})


def brand_hyphen_variants() -> List[str]:
    variants: List[str] = []
    for brand in sorted(BRAND_OFFICIAL_DOMAINS):
        if len(brand) < 6 or not brand.isalpha() or brand in BRAND_GENERIC_WORDS:
            continue
        for i in range(1, len(brand)):
            if min(i, len(brand) - i) > 2:
                continue
            variant = brand[:i] + "-" + brand[i:]
            if variant not in BRAND_HYPHEN_EXCEPTIONS:
                variants.append(variant)
    return variants


def _brand_typo_pattern() -> str:
    affix = _alt(BRAND_AFFIX_TOKENS)
    alternatives: List[str] = []
    for term, brand in sorted(BRAND_TYPO_TERMS.items()):
        guard = ""
        if brand.startswith(term) and len(brand) > len(term):
            guard = "(?!" + re.escape(brand[len(term):]) + ")"
        if len(term) >= 7:
            alternatives.append(re.escape(term) + guard)
        else:
            alternatives.append(re.escape(term) + guard + r"(?:\d+|-?(?:" + affix + r")\d*)?(?![a-z])")
    for brand in BRAND_TRUNCATION_TARGETS:
        alternatives.append(re.escape(brand[:-1]) + "(?!" + re.escape(brand[-1]) + ")")
    alternatives.append(r"(?:" + _alt(brand_hyphen_variants()) + r")(?![a-z])")
    return _NOT_OFFICIAL_HOST + r".*?(?<![a-z0-9])(?:" + "|".join(alternatives) + r")"


# /register?code=xxx（推廣碼註冊頁）
_REGISTER_CODE = (
    r"/(?:[^?#]*/)?(?:register|reg|signup)(?:\.html?|\.php)?\?(?:[^#]*&)?(?:code|ic|agent|inv|pid)=[a-z0-9]"
)


_GOV_AFFIX_ALT = _alt({
    "tax", "refund", "gov", "npa", "police", "fraud", "antifraud", "tw", "taiwan", "service",
    "verify", "subsidy", "cash", "parcel", "login", "pay", "payment", "fine", "penalty",
    "ticket", "notice", "case", "bonus", "money", "gift", "165",
})
_GOV_BRAND_ALT = _alt(_brands_of("gov"))

_FREE_HOST_ALT = _alt(s.lstrip(".") for s in FREE_HOSTING_SUFFIXES)
_TUNNEL_ALT = _alt(s.lstrip(".") for s in TUNNEL_SUFFIXES)
_SHORTENER_ALT = _alt(SHORTENER_DOMAINS)
_REDIRECTOR_ALT = _alt(REDIRECTOR_DOMAINS)

_FREE_HOST_LURE_ALT = _alt(set(BRAND_OFFICIAL_DOMAINS) | {
    "invest", "stock", "stocks", "vip", "usdt", "wallet", "airdrop", "casino", "bocai",
    "tougu", "licai", "gupiao", "touzi", "bank", "ebank", "loan", "verify", "login", "signin",
    "secure", "crypto", "exchange", "defi", "web3", "trade", "trading", "forex", "futures",
    "claim", "reward", "bonus", "refund", "tax", "gov", "kyc", "mining", "staking", "yule",
    "slot", "metamask", "bitcoin", "btc", "eth",
})

_ROUTE = r"(?:#/pages/[a-z]|/(?:h5|wap)/#/|#/(?:register|reg|signup|sign-up)(?:[/?&]|$))"
_INVITE_STRONG = r"(?:invite_?code|agent_?code|recommend_?code|share_?code|inviter_?code|tjm|yqm)"
_INVITE_ANY = r"(?:" + _INVITE_STRONG[3:-1] + r"|code|ic|agent|inv)"

_INV_CORE_ALT = _alt(INVESTMENT_TERMS["core_tokens"])
_INV_LURE_ALT = _alt(INVESTMENT_TERMS["lure_tokens"])
_INV_CLURE_ALT = _alt(INVESTMENT_TERMS["compound_lure_tokens"])

_CRYPTO_STRONG_HOST_ALT = _alt({
    "usdt", "usdc", "trc20", "erc20", "bep20", "airdrop", "cloudmining", "hashrate",
    "drainer", "jiaoyisuo", "jys", "qianbao", "wakuang", "kuangji", "kongtou", "heyue",
    "walletconnect", "secondcontract",
})
_CRYPTO_WEAK_ALT = _alt(
    (set(CRYPTO_TERMS["weak_tokens"]) - {"bit"})
    | {"defi", "web3", "dex", "otc", "p2p", "staking", "dapp", "ex"}
)
_CRYPTO_ONLY_ALT = _alt({
    "exchange", "swap", "coin", "coins", "crypto", "token", "tokens", "btc", "bitcoin", "eth",
    "bnb", "xrp", "doge", "nft", "stake", "staking", "mining", "miner", "liquidity", "wallet",
    "defi", "web3", "dex", "otc", "p2p", "dapp", "futures", "perp", "usdt",
})

_GAMBLING_STRONG_HOST_ALT = _alt({
    "casino", "casinos", "bocai", "caipiao", "yulecheng", "baijiale", "laohuji", "qipai",
    "zhenren", "liuhecai", "shishicai", "pujing", "xinpujing", "jinsha", "weinisi",
    "weinisiren", "huangguan", "taiyangcheng", "sportsbook", "sportsbet", "duchang", "dafa",
    "dafabet", "bet365", "188bet", "fun88", "w88", "1xbet", "betway", "12bet", "yule", "leyu",
    "kaiyun", "yabo", "jiuyou", "pk10", "bjl", "kuaisan", "feiting", "saiche",
})
_GAMBLING_SUBSTR_ALT = _alt({
    "casino", "bocai", "caipiao", "yulecheng", "baijiale", "laohuji", "qipai", "zhenren",
    "liuhecai", "shishicai", "pujing", "weinisi", "huangguan", "taiyangcheng",
    "sportsbook", "sportsbet", "dafabet",
})
_GAMBLING_TLD_TERM_ALT = _alt({
    "poker", "slot", "slots", "baccarat", "jackpot", "roulette", "blackjack", "bingo",
    "lottery", "lotto", "keno", "betting", "bet", "bets", "mgm", "sabong",
})

_ZH_GAMBLING_STRONG = _alt({
    "娛樂城", "娱乐城", "百家樂", "百家乐", "老虎機", "老虎机", "真人視訊", "真人视讯", "線上娛樂",
    "线上娱乐", "電子遊藝", "电子游艺", "博弈", "博彩", "時時彩", "时时彩", "北京賽車", "北京赛车",
    "幸運飛艇", "幸运飞艇", "體驗金", "体验金", "註冊送", "注册送", "首儲", "首存", "首充", "返水",
    "包網", "球版", "捕魚機", "六合彩", "賭場", "赌场",
})
_ZH_INVEST_STRONG = _alt({
    "飆股", "飙股", "明牌", "老師帶單", "老师带单", "帶單", "带单", "喊單", "跟單", "投顧老師",
    "股票群組", "投資群組", "保證獲利", "保证获利", "穩賺不賠", "稳赚不赔", "保證中籤", "專用通道",
    "法人通道", "內部額度", "AI選股", "AI量化", "當沖老師", "港股仙股", "代操", "刷單", "刷单",
    "搶單", "抢单", "荐股", "配资",
})
_ZH_CRYPTO_STRONG = _alt({
    "泰達幣", "U幣", "幣商", "換U", "雲挖礦", "雲端挖礦", "礦機", "礦池", "質押", "流動性挖礦",
    "領取空投", "秒合約", "量化機器人", "老用戶專屬", "虚拟币", "挖矿", "矿机", "质押", "秒合约",
    "充币", "提币",
})


# =============================================================================
# 硬規則：命中後會直接給較高風險分數
# =============================================================================

HARD_RULES: List[HardRule] = [
    # ---------------- 確定性高：IP / @ 偽裝 / 官方網域塞子網域 / 政府冒用 / 通道 ----------------
    HardRule(
        name="ip_address_url",
        pattern=r"^(?:\d{1,3}\.){3}\d{1,3}$|^[0-9a-f]{0,4}(?::[0-9a-f]{0,4}){2,7}$|^\d{8,10}$|^0x[0-9a-f]{8}$",
        score=92,
        message="網址直接使用 IP 位址，常見於高風險或臨時網站。",
        target="host",
    ),
    HardRule(
        name="url_userinfo_at_spoof",
        pattern=r"^[a-z][a-z0-9+.-]*://[^/?#]*@",
        score=90,
        message="網址在網域前放了「@」帳號欄位（例如 官方網址@詐騙網域），瀏覽器實際連往的是 @ 後面的網域，屬典型偽裝手法。",
        target="url",
    ),
    HardRule(
        name="official_domain_in_subdomain",
        pattern=r"^(?!(?:[a-z0-9-]+\.)*(?:" + _ALL_OFFICIAL_ALT + r")$)(?:[a-z0-9-]+\.)*?(?:"
                + _ALL_OFFICIAL_ALT + r")[.-](?![a-z]{2}$)",
        score=90,
        message="網址把知名官方網域（如 binance.com、esunbank.com.tw）塞進子網域或前綴，實際網域並非官方，屬高仿冒手法。",
        target="host",
    ),
    HardRule(
        name="gov_agency_spoof",
        pattern=r"^(?!(?:[a-z0-9-]+\.)*gov\.tw$)(?!(?:[a-z0-9-]+\.)*(?:165dashboard\.tw)$)"
                r"(?:(?:.*[.-])?gov[.-]?tw(?:[.-]|$)"
                r"|.*?(?<![a-z0-9])(?:" + _GOV_BRAND_ALT + r")-?(?:" + _GOV_AFFIX_ALT + r")\d*(?![a-z]))",
        score=90,
        message="網址冒用政府機關（165、警政署、財政部等）名稱或「gov-tw」字樣，但不是 .gov.tw 官方網域。",
        target="host",
    ),
    HardRule(
        name="tunnel_or_ephemeral_host",
        pattern=r"^(?:[a-z0-9-]+\.)+(?:" + _TUNNEL_ALT + r")$",
        score=88,
        message="網址使用 ngrok、trycloudflare 等臨時通道或 IPFS 閘道，常被用來架設即用即棄的詐騙頁面。",
        target="host",
    ),
    HardRule(
        name="ipfs_content_path",
        pattern=_URL_HEAD + r"[^?#]*/ipfs/(?:qm[1-9a-hj-np-z]{44}|bafy[a-z2-7]{50,})",
        score=80,
        message="網址指向 IPFS 分散式儲存內容，無法下架、常被用於釣魚與錢包盜取頁面。",
        target="url",
    ),

    # ---------------- 品牌冒用（與官方網域不符才觸發） ----------------
    HardRule(
        name="brand_spoof_finance",
        pattern=_brand_spoof_pattern(_brands_of("finance")),
        score=88,
        message="網址使用台灣銀行、券商或金融機構品牌（＋tw/vip/login 等字樣），但不是該機構的官方網域，疑似假冒金融機構。",
        target="host",
    ),
    HardRule(
        name="brand_spoof_crypto_exchange",
        pattern=_brand_spoof_pattern(_brands_of("crypto")),
        score=88,
        message="網址使用知名加密貨幣交易所或錢包品牌（如 Binance、OKX、MetaMask），但不是官方網域，疑似高仿交易所。",
        target="host",
    ),
    HardRule(
        name="brand_spoof_ecommerce",
        pattern=_brand_spoof_pattern(b for b in _brands_of("ecommerce") if b != "microsoft"),
        score=86,
        message="網址使用知名電商、社群或品牌名稱（＋mall/shop/vip 等字樣），但不是該品牌官方網域，疑似仿冒。",
        target="host",
    ),
    HardRule(
        name="brand_typo_squatting",
        pattern=_brand_typo_pattern(),
        score=86,
        message="網域為知名品牌的拼字變形（如 faebook、tikmall、walmar、binanc-e、sh0pee），且不是該品牌官方網域，屬高仿冒手法。",
        target="host",
    ),
    HardRule(
        name="brand_spoof_microsoft_office",
        pattern=_NOT_OFFICIAL_HOST + r".*?(?<![a-z0-9])(?:micros0ft|m1crosoft|mircosoft|microsfot|"
                r"microsoft-[a-z0-9-]+|ms-?office\d*|office365-[a-z0-9-]+|office-(?:free|crack|activate|key)|"
                r"excel-(?:free|crack|download))(?![a-z])",
        score=88,
        message="網址疑似仿冒 Microsoft Office / Excel，可能為軟體詐騙或釣魚頁面。",
        target="host",
    ),

    # ---------------- punycode ----------------
    HardRule(
        name="punycode_mixed_script",
        pattern=r"(?:^|\.)xn--[a-z0-9-]*[a-z][a-z0-9-]*-[a-z0-9]+(?:\.|$)",
        score=86,
        message="網域含「拉丁字母＋特殊字元」混合的國際化網域（punycode），常用於製造與正牌網址外觀幾乎相同的仿冒網域。",
        target="host",
    ),
    HardRule(
        name="punycode_risky_tld",
        pattern=r"(?:^|\.)xn--[a-z0-9-]+\.(?:" + _TIER_ANY_ALT + r")$",
        score=82,
        message="網域為中文等國際化網域（punycode）並使用高風險頂級域名，符合博弈／假投資網站常見手法。",
        target="host",
    ),
    HardRule(
        name="punycode_domain",
        pattern=r"(?:^|\.)xn--",
        score=58,
        message="網域為國際化網域（punycode，xn--），實際顯示文字可能與看起來不同，請確認來源。",
        target="host",
    ),

    # ---------------- 跳轉 ----------------
    HardRule(
        name="double_http",
        # 只看「路徑中直接接網址」或 url/goto/redirect/target 等轉址參數；
        # referrer、return、next、redirect_uri 等追蹤/登入回呼參數與廣告點擊服務不算。
        pattern=_URL_HEAD + r"(?![^/?#]*?(?<![a-z0-9-])(?:" + _REDIRECTOR_ALT + r")(?::\d+)?(?:[/?#]|$))"
                r"[^#]*?(?:/|[?&](?:url|u|q|redirect|redirect_?url|redirect_?to|goto|go|target|dest|"
                r"destination|link|to|out|jump|forward|site|r)=)https?://",
        score=72,
        message="網址以轉址參數或路徑內嵌另一個 http/https 網址，可能存在跳轉偽裝（廣告點擊追蹤服務已排除）。",
        target="decoded",
    ),
    HardRule(
        name="redirect_to_risky_target",
        pattern=r"(?:[?&][a-z0-9_.-]*=|/)https?://(?:[^/?#&@]*@)?[a-z0-9.-]*\.(?:" + _TIER_ANY_ALT
                + r")(?::\d+)?(?:[/?#&]|$)",
        score=82,
        message="網址內嵌的跳轉目標使用高風險頂級域名（如 .top、.vip、.xyz），疑似以轉址包裝詐騙網站。",
        target="decoded",
    ),

    # ---------------- 抽獎 / 中文誘因（解碼後；排除新聞／政府網域） ----------------
    HardRule(
        name="lottery_reward_pattern",
        pattern=_URL_HEAD + _NOT_CONTENT_AUTHORITY
                + r".*?(?:中獎|抽獎|免費領|領取|獲獎).{0,20}(?:點擊|立即|馬上|現在|填寫)",
        score=88,
        message="網址或文字包含抽獎、領取、立即點擊等高誘因詐騙模式。",
        target="decoded",
    ),

    # ---------------- 博弈 ----------------
    HardRule(
        name="gambling_strong_term_host",
        pattern=r"(?<![a-z])(?:" + _GAMBLING_STRONG_HOST_ALT + r")(?![a-z])|(?:" + _GAMBLING_SUBSTR_ALT
                + r")|(?<![a-z])baccarat(?=\d)|[a-z0-9]-baccarat",
        score=86,
        message="網域含博弈專用字詞或拼音（如 casino、yule 娛樂、bocai 博彩、pujing 葡京、jinsha 金沙），疑似非法博弈網站。",
        target="host",
    ),
    HardRule(
        name="gambling_vip_number_pattern",
        pattern=r"(?:168|888|666|999|777)[a-z0-9-]{0,15}?(?:vip|win|bet|casino|slot|game|coin|trade|fx)|"
                r"(?<![a-z])(?:vip|win|bet|casino|slot)[a-z0-9-]{0,15}?(?:168|888|666|999|777)",
        score=88,
        message="網域出現 168、888、VIP、bet、win 等博弈或金融詐騙常見組合。",
        target="host",
    ),
    HardRule(
        name="gambling_term_risky_tld",
        pattern=r"(?<![a-z])(?:" + _GAMBLING_TLD_TERM_ALT + r")(?![a-z])[a-z0-9.-]*\.(?:" + _TIER_ANY_ALT
                + r")$|(?<![a-z])(?:poker|slots?|jackpot|baccarat|roulette|bingo|lotto)\d{2,}(?![a-z])",
        score=84,
        message="網域含博弈字詞（poker、slot、jackpot、bet 等）並使用高風險頂級域名，疑似博弈網站。",
        target="host",
    ),
    HardRule(
        name="gambling_abbrev_number_risky_tld",
        pattern=r"(?:(?<![a-z])(?:bet|win|slot|vip|yl|bc|hg|cp|pj|js|vns|ty|dj|ky|ag|lucky)\d{2,}(?![a-z])"
                r"|(?<![a-z0-9])\d{2,}(?:bet|win|vip|slot|club)(?![a-z]))[a-z0-9.-]*\.(?:" + _TIER_ANY_ALT + r")$",
        score=80,
        message="網域為博弈常見縮寫＋數字（如 bc888 博彩、yl 娛樂、hg 皇冠）並使用高風險頂級域名。",
        target="host",
    ),
    HardRule(
        name="gambling_zh_term",
        pattern=_URL_HEAD + _NOT_CONTENT_AUTHORITY + r".*?(?:" + _ZH_GAMBLING_STRONG + r")",
        score=80,
        message="網址（解碼後）含娛樂城、百家樂、老虎機、體驗金等博弈用語。",
        target="decoded",
    ),

    # ---------------- 假投資 ----------------
    HardRule(
        name="investment_pinyin_host",
        pattern=r"(?<![a-z])(?:tougu|licai|gupiao|touzi|qihuo|waihui|huangjin|lianghua|biaogu|feigu|"
                r"mingpai|daidan|gendan|chouqian|shengou|zhengquan|quanshang|dangchong|jijin|stockvip|vipstock)(?![a-z])",
        score=85,
        message="網域含投顧、理財、股票、抽籤、帶單等中文拼音（如 tougu、licai、gupiao），符合假投資網站命名模式。",
        target="host",
    ),
    HardRule(
        name="investment_combo_host",
        pattern=_NOT_OFFICIAL_HOST + r".*?(?:"
                r"(?<![a-z0-9])(?:" + _INV_CORE_ALT + r")(?:" + _INV_CLURE_ALT + r")\d*(?![a-z])"
                r"|(?<![a-z0-9])(?:" + _INV_CLURE_ALT + r")(?:" + _INV_CORE_ALT + r")\d*(?![a-z])"
                r"|(?<![a-z0-9])(?:" + _INV_CORE_ALT + r"){1,2}\d{2,}(?![a-z])"
                r"|(?<![a-z0-9])(?:" + _INV_CORE_ALT + r")(?![a-z0-9])(?:[.-][a-z0-9]+)*?[.-](?:" + _INV_LURE_ALT
                + r")\d*(?=[.-])"
                r"|(?<![a-z0-9])(?:" + _INV_LURE_ALT + r")\d*(?![a-z0-9])(?:[.-][a-z0-9]+)*?[.-](?:" + _INV_CORE_ALT
                + r")(?=[.-]))",
        score=82,
        message="網域同時包含股票／投資／外匯等金融詞與 VIP、AI、老師、俱樂部等誘因詞，符合假投資網站命名模式。",
        target="host",
    ),
    HardRule(
        name="investment_guarantee_combo_pattern",
        # 'forex'／'crypto'／'profit' 是銀行匯率頁、財經新聞會合法出現的通用詞，
        # 必須同時搭配「保證獲利／穩賺／高報酬」等誇大話術才觸發。
        pattern=r"\b(?:forex|crypto|profit)\b.{0,25}"
                r"(?:guarantee|guaranteed|high-?return|get-?rich|risk-?free|"
                r"穩賺|保證獲利|高報酬|高獲利|暴賺|翻倍|穩賺不賠|包賺)|"
                r"(?:guarantee|guaranteed|high-?return|get-?rich|risk-?free|"
                r"穩賺|保證獲利|高報酬|高獲利|暴賺|翻倍|穩賺不賠|包賺)"
                r".{0,25}\b(?:forex|crypto|profit)\b",
        score=84,
        message="網址同時包含外匯／加密貨幣字詞與保證獲利、穩賺等誇大話術，屬常見投資詐騙模式。",
        target="decoded",
    ),
    HardRule(
        name="investment_zh_term",
        pattern=_URL_HEAD + _NOT_CONTENT_AUTHORITY + r".*?(?:" + _ZH_INVEST_STRONG + r")",
        score=80,
        message="網址（解碼後）含飆股、老師帶單、保證獲利、專用通道等假投資話術。",
        target="decoded",
    ),

    # ---------------- 假交易所 / 錢包 ----------------
    HardRule(
        name="crypto_specific_pattern",
        pattern=_NOT_OFFICIAL_HOST + r".*?(?<![a-z])(?:" + _CRYPTO_STRONG_HOST_ALT + r")(?![a-z])",
        score=80,
        message="網域含 USDT、空投、雲挖礦、交易所（jys）等加密貨幣詐騙常用字詞。",
        target="host",
    ),
    HardRule(
        name="crypto_weak_combo",
        pattern=_NOT_OFFICIAL_HOST + r"(?:.*?(?<![a-z])(?:" + _CRYPTO_WEAK_ALT + r")(?![a-z]).*?[.-](?:"
                + _CRYPTO_WEAK_ALT + r")\d*(?=[.-])"
                r"|.*?(?<![a-z])(?:" + _CRYPTO_ONLY_ALT + r")(?![a-z])[a-z0-9.-]*\.(?:" + _TIER_ANY_ALT + r")$)",
        score=82,
        message="網域同時含多個加密貨幣／錢包字詞（如 mining-reward、wallet-verify）或搭配高風險頂級域名，疑似假交易所或錢包盜取頁面。",
        target="host",
    ),
    HardRule(
        name="finance_term_high_risk_tld",
        pattern=r"(?<![a-z])(?:stock|stocks|invest|investment|trade|trading|forex|fx|futures|fund|funds|"
                r"wealth|profit|crypto|coin|coins|exchange|usdt|btc|wallet|bank|loan)(?![a-z])"
                r"[a-z0-9.-]*\.(?:" + _TIER2_ALT + r")$",
        score=72,
        message="網域含股票、交易、錢包、借貸等金融字詞，並使用高度濫用的頂級域名（如 .vip、.top），符合假投資網站命名模式。",
        target="host",
    ),
    HardRule(
        name="vip_token_risky_tld",
        pattern=r"(?<![a-z])v?s?vip(?:[a-z]+)?\d*(?![a-z])[a-z0-9.-]*\.(?:" + _TIER_ANY_ALT + r")$",
        score=72,
        message="網域含 VIP 字樣並使用常被濫用的頂級域名，常見於假投資刷單商城與博弈網站。",
        target="host",
    ),
    HardRule(
        name="crypto_ex_suffix_risky_tld",
        pattern=r"(?<![a-z0-9-])(?!(?:" + _alt(CRYPTO_TERMS["ex_exceptions"]) + r")\.)[a-z0-9]{3,}ex\.(?:"
                + _TIER_ANY_ALT + r")$",
        score=78,
        message="網域以「-ex」（exchange 縮寫）結尾並使用高風險頂級域名，符合假交易所命名模式（如 rainbowex.cc）。",
        target="host",
    ),
    HardRule(
        name="crypto_ex_token",
        pattern=r"(?<![a-z0-9])[a-z0-9]{2,}-ex\d*(?=[.-])",
        score=76,
        message="網域含獨立的「-ex」字樣（exchange 縮寫，如 hpro-ex），符合假交易所命名模式。",
        target="host",
    ),
    HardRule(
        name="crypto_zh_term",
        pattern=_URL_HEAD + _NOT_CONTENT_AUTHORITY + r".*?(?:" + _ZH_CRYPTO_STRONG + r")",
        score=78,
        message="網址（解碼後）含泰達幣、換U、雲挖礦、質押、秒合約等虛擬貨幣詐騙用語。",
        target="decoded",
    ),

    # ---------------- 免費架站＋誘因 ----------------
    HardRule(
        name="free_hosting_with_lure",
        pattern=r"^(?:[a-z0-9-]+\.)*?[a-z0-9-]*?(?<![a-z0-9])(?:" + _FREE_HOST_LURE_ALT
                + r")(?:vip|pro|tw)?\d*(?![a-z0-9])[a-z0-9-]*(?:\.[a-z0-9-]+)*?\.(?:" + _FREE_HOST_ALT + r")$",
        score=82,
        message="網址架在免費架站／雲端暫存平台（如 pages.dev、web.app、github.io），且子網域含品牌、金融或博弈字詞，疑似臨時詐騙頁面。",
        target="host",
    ),

    # ---------------- uni-app / H5 路由與 App 下載誘導 ----------------
    HardRule(
        name="uniapp_route_with_invite",
        pattern=r"^(?=.*" + _ROUTE + r")(?=.*[?&]" + _INVITE_ANY + r"=[a-z0-9])",
        score=86,
        message="網址為假投資／假交易所常用的 uni-app H5 路由（#/pages/、/h5/#/）並帶邀請碼／代理碼參數。",
        target="url",
    ),
    HardRule(
        name="uniapp_route_risky_host",
        pattern=_URL_HEAD + _HOST_IN_URL
                + r"(?:(?:h5|m|wap|app|sj)\.[^/?#:]*|[^/?#:]*\.(?:" + _TIER_ANY_ALT
                + r")|(?:\d{1,3}\.){3}\d{1,3}|[^/?#:]*xn--[^/?#:]*|[^/?#:]*[a-z][0-9]{3,}[^/?#:]*)"
                r"(?::\d+)?(?:[/?#].*?)?" + _ROUTE,
        score=80,
        message="網址為 uni-app H5 路由（#/pages/、/h5/#/），且主機為 h5./m. 子網域、高風險 TLD、IP 或含長數字，符合假投資 App 網頁版特徵。",
        target="url",
    ),
    HardRule(
        name="uniapp_hash_route",
        pattern=_ROUTE,
        score=62,
        message="網址為 uni-app／H5 單頁應用的登入或註冊路由（#/pages/、/h5/#/），假投資與假交易所網站大量使用此模板，請確認來源。",
        target="url",
    ),
    HardRule(
        name="register_invite_risky_tld",
        pattern=_URL_HEAD + _HOST_IN_URL + r"[^/?#:]*\.(?:" + _TIER_ANY_ALT + r")(?::\d+)?" + _REGISTER_CODE,
        score=75,
        message="網址為帶推廣碼的註冊頁（/register?code=），且網域為高風險頂級域名，符合假投資／博弈平台推廣模式。",
        target="url",
    ),
    HardRule(
        name="register_invite_code",
        pattern=_URL_HEAD + _HOST_IN_URL + r"[^/?#]*" + _REGISTER_CODE,
        score=60,
        message="網址為帶推廣碼的註冊頁（/register?code=），常見於假投資、博弈平台的推廣連結，請確認來源。",
        target="url",
    ),
    HardRule(
        name="mobile_h5_wap_highrisk",
        pattern=_URL_HEAD + _HOST_IN_URL + r"(?=(?:h5|wap)\.|[^?#]*/(?:h5|wap)(?:[/?#]|$))[^/?#:]*\.(?:"
                + _TIER_ANY_ALT + r")(?::\d+)?(?:[/?#]|$)",
        score=72,
        message="網址使用 h5、wap 行動版落地頁型態，且網域為高風險頂級域名。",
        target="url",
    ),
    HardRule(
        name="mobile_prefix_high_risk_tld",
        pattern=r"^(?:m|h5|wap|app|sj|trade|stock|stocks|dch|main)\.(?:[a-z0-9-]+\.)*[a-z0-9-]+\.(?:"
                + _TIER2_ALT + r")$",
        score=75,
        message="網址為 app./h5./m. 等行動版子網域，並使用高度濫用的頂級域名（如 .top、.vip），符合假投資 App 網頁版特徵。",
        target="host",
    ),
    HardRule(
        name="app_download_lure_risky",
        pattern=_URL_HEAD + _HOST_IN_URL
                + r"(?:(?:app|m|h5|dl|down|download)\.[^/?#:]*|[^/?#:]*\.(?:" + _TIER_ANY_ALT
                + r")|[^/?#:]*[a-z][0-9]{3,}[^/?#:]*)(?::\d+)?/(?:[^?#]*/)?"
                r"(?:download|down|appdown|app-download|app)\.(?:html?|php)(?:[?#]|$)",
        score=78,
        message="網址為 App 下載誘導頁（download.html），且網域為高風險 TLD 或可疑子網域，常見於假投資／博弈 App 側載。",
        target="url",
    ),
    HardRule(
        name="ios_profile_or_enterprise_install",
        pattern=r"^itms-services:|^[^?#]*\.mobileconfig(?:[?#]|$)",
        score=86,
        message="網址會安裝 iOS 描述檔或企業簽 App（mobileconfig／itms-services），常被假投資 App 用來繞過 App Store 審查。",
        target="url",
    ),
    HardRule(
        name="apk_ipa_direct_download",
        pattern=r"^[^?#]*\.(?:apk|ipa)(?:[?#]|$)",
        score=62,
        message="網址直接下載 APK／IPA 安裝檔，未經官方商店審核，請勿任意安裝。",
        target="url",
    ),
    HardRule(
        name="non_standard_port_risky",
        pattern=_URL_HEAD + _HOST_IN_URL + r"(?:[^/?#:]*\.(?:" + _TIER_ANY_ALT
                + r")|(?:\d{1,3}\.){3}\d{1,3}):(?!(?:80|443)(?:[/?#]|$))\d{2,5}(?:[/?#]|$)",
        score=78,
        message="網址使用非標準連接埠（如 :8443、:8080），且網域為高風險頂級域名或 IP，合法網站極少這樣做。",
        target="url",
    ),
    HardRule(
        name="suspicious_tld_random_domain",
        # 高度濫用 TLD + 無意義隨機字元 SLD（4 個以上連續子音，如 zltfm.top）
        pattern=r"(?<![a-z0-9-])(?=[a-z0-9]*[bcdfghjklmnpqrstvwxz]{4})[a-z0-9]{5,15}\.(?:" + _TIER2_ALT + r")$",
        score=62,
        message="網址使用高度濫用的頂級域名，且網域名稱為無意義的隨機字元組合，符合即用即棄型詐騙網域特徵。",
        target="host",
    ),

    # ---------------- 軟體破解 ----------------
    HardRule(
        name="software_piracy_pattern",
        pattern=r"(?<![a-z0-9])(?:office|excel|word|windows|win1[01]|photoshop|adobe|autocad|winrar|software|apk)"
                r"[-_ ]?(?:crack(?:ed)?|keygen|activat(?:e|or|ion)|patch|free-?download)(?![a-z])"
                r"|(?<![a-z0-9])(?:crack(?:ed)?|keygen)[-_ ]?(?:office|excel|windows|download|free|full|version|software)(?![a-z])"
                r"|(?<![a-z0-9])keygen(?![a-z])|(?<![a-z0-9])serial-?key(?![a-z])"
                r"|免費下載.{0,10}(?:office|excel|word)|破解版|序號產生器|激活碼",
        score=80,
        message="網址包含軟體破解、序號產生器、免費下載 Office 等字詞，常見於假冒軟體詐騙頁面。",
        target="decoded",
    ),

    # ---------------- 社群導流 / 邀請碼（中風險） ----------------
    HardRule(
        name="social_group_invite",
        pattern=_URL_HEAD + _HOST_IN_URL + r"(?:[a-z0-9-]+\.)*(?:line\.me/(?:r/)?ti/g2?/|t\.me/(?:\+|joinchat/)|"
                r"telegram\.(?:me|dog)/(?:\+|joinchat/)|chat\.whatsapp\.com/[a-z0-9]|discord(?:app)?\.(?:gg|com/invite)/)",
        score=55,
        message="網址為 LINE／Telegram／WhatsApp／Discord 群組邀請連結；假投資詐騙常以廣告導流加入群組，請先查證群組來源。",
        target="url",
    ),
    HardRule(
        name="line_official_shortlink",
        pattern=_URL_HEAD + _HOST_IN_URL + r"lin\.ee/[a-z0-9]",
        score=50,
        message="網址為 LINE 官方帳號短連結（lin.ee），無法從網址看出加入的是哪個帳號，請確認是否為品牌官方帳號。",
        target="url",
    ),
    HardRule(
        name="invite_code_param",
        pattern=r"[?&]" + _INVITE_STRONG + r"=[a-z0-9]",
        score=55,
        message="網址帶有邀請碼／代理碼參數（invite_code、agentcode 等），常見於假投資、博弈平台的推廣註冊連結。",
        target="url",
    ),
    HardRule(
        name="shortener_with_lure",
        pattern=_URL_HEAD + _HOST_IN_URL + r"(?:" + _SHORTENER_ALT + r")(?::\d+)?/[^#]*?(?<![a-z0-9])"
                r"(?:usdt|btc|eth|crypto|stockvip|vipstock|casino|bocai|tougu|licai|gupiao|airdrop|forex|"
                r"invitecode|invite_code|agentcode|yule|baccarat)(?![a-z])",
        score=68,
        message="短網址的路徑或參數含加密貨幣、投顧、博弈或邀請碼字詞，疑似以短網址隱藏詐騙目的地。",
        target="url",
    ),
]


def validate_hard_rules() -> List[str]:
    """回傳無法編譯或設定錯誤的規則清單（空 list = 全部正常）。"""
    problems: List[str] = []
    seen: Set[str] = set()
    for rule in HARD_RULES:
        if rule.name in seen:
            problems.append(f"{rule.name}: 名稱重複")
        seen.add(rule.name)
        if rule.target not in VALID_RULE_TARGETS:
            problems.append(f"{rule.name}: target={rule.target} 不合法")
        try:
            re.compile(rule.pattern, re.IGNORECASE)
        except re.error as exc:
            problems.append(f"{rule.name}: {exc}")
    return problems


# =============================================================================
# 政府封鎖黑名單初始種子清單（NPA 165 + 已知詐騙域名）
# =============================================================================
# 說明：此清單為離線快取版本，涵蓋常見詐騙域名。
# 完整、即時版本請參考：https://www.165.npa.gov.tw
#
# 執行期間透過 /blocklist/add 新增、達到回報門檻的網域，「不會」寫回這裡，
# 而是持久化在 dynamic_blocklist.json（見 blocklist_store.py），伺服器啟動時
# 會自動讀回並併入記憶體中的黑名單集合。若要手動長期加入某個已確認的詐騙
# 域名，才需要直接編輯這份清單。

INITIAL_BLOCKED_DOMAINS: Set[str] = {
    # ── 刑事局封鎖域名（直接申報入庫） ──────────────────────────────────────
    "hydrohfm.com",
    "hydrofx.net",
    "hydrafx.cc",
    "jfw168vip.com",
    "win888bet.cc",
    "lucky168vip.top",
    "bet99pro.com",
    "slot777win.xyz",
    "casino999.vip",
    "888cashwin.cc",
    "168profit.top",
    "tikmall.cc",
    "tiktokstore.shop",
    "igproduct.xyz",
    "sh0pee.com",
    "shoppe-tw.xyz",
    "lazadaa.com",
    "poyabuy.cc",
    "pchome-tw.top",
    "momo-official.xyz",
    "msoffice-free.com",
    "office-crack.net",
    "excel-free-download.com",
    # ── 新增：使用者回報後確認的詐騙域名 ────────────────────────────────────
    # （在此手動維護，格式：registered domain，如 evil.com 而非 m.evil.com）
    "zpvbjk.cc",
}
