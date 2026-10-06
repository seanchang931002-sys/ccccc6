# =============================================================================
# train_model.py — Truth（基於大數據之廣告行銷的稽核平台）AI 模型訓練腳本 v7.0
# =============================================================================
# 修正紀錄（v4.1）：
#   1. 移除 LogisticRegression 中不存在的 margin=0 參數（會導致 TypeError）
#   2. augment_scam_urls() 補充多樣化擴增格式（h5/wap、無HTTPS、博弈數字等），
#      避免模型只學到單一「品牌名+可疑TLD」模式
#
# 新增（v5.0）：
#   - 擴增 F 類樣本：高 entropy + 無意義 SLD 型詐騙域名（hydrohfm 類型）
#   - 擴增 G 類樣本：Levenshtein 距離 1~3 的品牌拼字仿冒樣本
#   - feature_count：29 → 31（levenshtein_brand_dist, newly_registered_like）
#
# 重寫（v7.0，依 CONTRACT §5；特徵 31 → 45 維，舊 scam_model.pkl 不相容）：
#   1. 資料：data_sources.py 統一以 features.canonicalize_url 正規化（無 scheme 補 https://）、
#      canonical 去重；主資料內部衝突排除；feedback 同網址以最新為準並覆蓋主資料、樣本權重 ×2；
#      可選匯入 data/ 下 165 開放資料（存在才讀、不連網）。
#   2. 擴增：
#      - 真實樣本「路徑／參數變體」：每筆真實網址各加一筆「形狀翻轉」（有路徑 → 裸網域；
#        裸網域 → 一般路徑）與一筆「廣告落地頁」（一般路徑 + utm/fbclid/gclid），兩類標籤
#        套用同一產生器、同機率加 http://，三筆共用原本的樣本權重 → 消除
#        「有路徑／有追蹤參數／https = 正常」捷徑（詐騙原本 96% 是裸網域）。
#      - 165 四大類示意樣本（博弈、假投資、假交易所／錢包、雲端暫存／通道）＋社群邀請、
#        銀行券商政府電商品牌冒用、網址混淆（@、IP、非標準埠、punycode）、隨機字母網域。
#      - hard negatives：易誤判子字串、.com.tw／新聞媒體（含中文詐騙詞報導路徑）、合法
#        銀行／券商／期貨商、正牌交易所與錢包、合法雲端文件與免費架站作品集、m./app./h5
#        官方站、帶 utm/gclid 的合法電商、政府學術、官方社群帳號、數字與縮寫網域。
#      - 模板列也各有一筆「形狀翻轉」雙胞胎（與原列平分權重；訊號在路徑的模板除外），
#        沒有任何風險訊號的詐騙模板列剔除（對模型只是標籤雜訊）。
#      - 擴增列標記 is_synthetic 與 group；與 tests/fixtures/url_cases.json（驗收清單）
#        相同網址的擴增列一律剔除，詐騙擴增列也不得使用驗收清單中的網域（避免洩漏）。
#      - 模板權重：詐騙 0.4、hard negative 0.6（合計約占三成，不壓倒真實資料）。
#   3. 驗證：StratifiedGroupKFold(5)（group = registered domain；同模板 = 同 group），
#      每折訓練用該折訓練組全部（含擴增），評估只算測試組的真實樣本；5 種候選模型各做
#      4～6 組小型超參數搜尋；選模＝OOF PR-AUC 最高者 1 個標準誤內、取「真實詐騙 vs 未見過
#      網域的 hard negative」AUC 最高者；回報 PR-AUC、ROC-AUC、0.40／0.70 門檻下
#      precision／recall／F1／F2、校準前後 Brier；sigmoid 與 isotonic 以巢狀 group-CV 擇優。
#   4. 校準採 prior shift：訓練資料詐騙比例約 47%，以目標比例 TARGET_PRIOR=0.20 加權校準
#      （--target-prior 0 可改回資料原始比例），讓沒有風險訊號的一般網站不會被推到 0.40 以上。
#   5. 最終模型：CalibratedClassifierCV(ensemble=True) 以全部資料、group 切分訓練，
#      每個子模型都以「未參與訓練的真實樣本」校準；存成 bundle（CONTRACT §3）。
#      估計器一律是標準 sklearn 類別（樹模型外包單步 Pipeline），main.py 直接 joblib.load 即可。
#   6. model_report.json：CV 結果、OOF 指標、校準、特徵重要度（含 permutation 前 15）、
#      url_cases.json 模型層級 FP／FN、scheme／路徑捷徑檢查、形狀穩健度、疑似標註錯誤清單。
#   7. verify_environment() 改為比對 feature_names 與 feature_schema_id。
#   8. 產物改為原子寫入（同目錄暫存檔 + os.replace）：訓練途中呼叫 /reload-model
#      不會讀到寫一半的 scam_model.pkl。
#
# 修正（v7.1，依 CONTRACT §5.1；特徵 45 → 46 維，新增 sld_randomness）：
#   1. 特徵：features.py v7.1（sld_randomness、newly_registered_like 改用字元 n-gram 可讀性、黏字切分）。
#   2. 校準目標比例（target prior）改以「獨立 benign 驗證集」選擇（--benign-validation，預設讀
#      環境變數 TRUTHMARK_BENIGN_VALIDATION）：候選 0.10／0.15／0.20／0.30／0.40／資料原始比例，
#      每個候選以巢狀 group-CV 求真實資料 OOF 召回 @0.40，並以全部資料訓練的校準模型求驗證集
#      模型層級誤判率 @0.40；準則＝誤判率 ≤ 3% 下召回最大（同分取 precision 高者）。取捨表寫進
#      model_meta.json（calibration.prior_selection）與 model_report.json（prior_selection）。
#      url_cases.json 是驗收集，不參與選擇。沒有提供驗證集時沿用 TARGET_PRIOR（上次選擇結果）。
#
# 使用方式：python train_model.py            （完整訓練，約 2～3 分鐘；含 prior 選擇約 8～10 分鐘）
#           python train_model.py --quick    （每個模型只試 2 組參數，開發用）
#           python train_model.py --no-save  （只評估、不覆寫模型與報告）
#           python train_model.py --benign-validation <benign_validation.json>
# =============================================================================

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import re
import string
import sys
import time
import warnings
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from urllib.parse import quote

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

import data_sources as DS
from features import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_ID,
    FEATURE_VERSION,
    canonicalize_url,
    extract_feature_dict,
    extract_features,
    get_hostname,
    get_registered_domain,
)
from rules_config import TRUSTED_DOMAINS


# =============================================================================
# 路徑設定
# =============================================================================

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))

RAW_DATA_PATH   = os.path.join(BASE_DIR, "data.csv")
CLEAN_DATA_PATH = os.path.join(BASE_DIR, "data_clean.csv")
FEEDBACK_PATH   = os.path.join(BASE_DIR, "feedback.csv")
OPENDATA_DIR    = os.path.join(BASE_DIR, "data")
URL_CASES_PATH  = os.path.join(BASE_DIR, "tests", "fixtures", "url_cases.json")

MODEL_PATH      = os.path.join(BASE_DIR, "scam_model.pkl")
META_PATH       = os.path.join(BASE_DIR, "model_meta.json")
REPORT_PATH     = os.path.join(BASE_DIR, "model_report.json")

URL_COL   = DS.URL_COL
LABEL_COL = DS.LABEL_COL

BUNDLE_FORMAT = "truthmark-model-bundle/1"
# main.py 的風險門檻（risk_score ≥ 40 medium、≥ 70 high）對應的模型機率
THRESHOLDS = {"medium": 0.40, "high": 0.70}
RANDOM_STATE = 42
N_SPLITS = 5

FEEDBACK_WEIGHT = 2.0      # 使用者回報（最新人工判斷）權重
# 模板擴增樣本權重（避免壓倒真實資料）。詐騙模板訊號明確、少量權重即可學到；hard negative
# 權重略高，用來抵銷真實資料「正常樣本幾乎都是 .com.tw」造成的「一般 gTLD = 詐騙」偏差
SYNTHETIC_WEIGHT = {1: 0.4, 0: 0.6}
VARIANT_HTTP_RATE = 0.05   # 真實樣本變體改成 http:// 的機率（兩類相同）
# 校準目標詐騙比例：訓練資料約 47% 是詐騙，遠高於實際廣告。v7.1 起以獨立 benign 驗證集選擇
# （見 select_target_prior；此常數為最近一次選擇的結果，沒有提供驗證集時使用）
TARGET_PRIOR = 0.20
PRIOR_CANDIDATES: Tuple[Optional[float], ...] = (0.10, 0.15, 0.20, 0.30, 0.40, None)   # None = 資料原始比例
BENIGN_VALIDATION_MAX_FPR = 0.03
BENIGN_VALIDATION_ENV = "TRUTHMARK_BENIGN_VALIDATION"
TRACKING_RATE = 0.35       # 擴增網址帶廣告追蹤參數的機率（兩類相同）


# =============================================================================
# 擴增：共用亂數工具與路徑產生器（兩類標籤共用同一套「一般路徑」）
# =============================================================================

_B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
_B32 = "abcdefghijklmnopqrstuvwxyz234567"
_CONSONANTS = "bcdfghjklmnpqrstvwxz"
_VOWELS = "aeiouy"


class _Gen:
    """可重現的擴增用亂數工具（固定 seed）。"""

    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)

    def choice(self, seq: Sequence[Any]) -> Any:
        return seq[self.rng.randrange(len(seq))]

    def chance(self, p: float) -> bool:
        return self.rng.random() < p

    def randint(self, a: int, b: int) -> int:
        return self.rng.randint(a, b)

    def chars(self, n: int, alphabet: str) -> str:
        return "".join(alphabet[self.rng.randrange(len(alphabet))] for _ in range(n))

    def alnum(self, n: int) -> str:
        return self.chars(n, string.ascii_letters + string.digits)

    def lower_alnum(self, n: int) -> str:
        return self.chars(n, string.ascii_lowercase + string.digits)

    def hexs(self, n: int) -> str:
        return self.chars(n, "0123456789abcdef")

    def digits(self, n: int) -> str:
        return str(self.rng.randint(1, 9)) + self.chars(n - 1, string.digits) if n > 0 else ""

    def random_sld(self, lo: int = 5, hi: int = 9) -> str:
        """165 公告中常見的「子音密集、無意義」SLD（hbvsgd、nhkutdp 類型）。"""
        n = self.randint(lo, hi)
        return "".join(self.choice(_VOWELS) if self.chance(0.18) else self.choice(_CONSONANTS)
                       for _ in range(n))

    def scheme(self) -> str:
        """兩類樣本共用的 scheme 分布：https 78%、未寫 14%、http 8%。"""
        r = self.rng.random()
        return "https://" if r < 0.78 else ("" if r < 0.92 else "http://")


NEUTRAL_SLUGS = [
    "summer-sale", "new-arrivals", "outlet", "weekly-deal", "skincare", "3c", "fashion", "home-living",
    "limited", "best-seller", "member-day", "anniversary", "spring", "autumn", "flash-sale", "beauty",
    "kids", "travel", "books", "kitchen", "pets", "outdoor", "coffee", "camera", "laptop", "shoes",
]
NEUTRAL_LANGS = ["zh-tw", "tw", "zh-TW", "zh-hant", "en", "zh_TW"]
TRACK_SOURCES = ["facebook", "google", "line", "instagram", "yahoo", "criteo", "newsletter", "dcard",
                 "youtube", "tiktok", "ptt", "edm"]
TRACK_MEDIUMS = ["cpc", "paid_social", "display", "social", "email", "banner", "news-feed", "cpm"]


def neutral_path(g: _Gen, allow_root: bool = True) -> str:
    """與標籤無關的一般網站路徑（兩類樣本共用；不含任何詐騙誘因詞）。"""
    n = g.randint(10, 9_999_999)
    slug = g.choice(NEUTRAL_SLUGS)
    lang = g.choice(NEUTRAL_LANGS)
    options: List[Callable[[], str]] = [
        lambda: "/index.html",
        lambda: f"/{lang}/",
        lambda: f"/{lang}/index.html",
        lambda: "/home",
        lambda: "/login",
        lambda: "/member/login",
        lambda: "/about",
        lambda: "/contact.html",
        lambda: f"/product/{n}",
        lambda: f"/products/{slug}",
        lambda: f"/goods/GoodsDetail.jsp?i_code={n}",
        lambda: f"/prod/{g.chars(6, string.ascii_uppercase + string.digits)}-{g.chars(9, string.ascii_uppercase + string.digits)}",
        lambda: f"/article/{n}",
        lambda: f"/news/{g.randint(2023, 2026)}{g.randint(1, 12):02d}{g.randint(1, 28):02d}/{n}.htm",
        lambda: f"/event/{g.randint(2023, 2026)}/{slug}",
        lambda: f"/category/{slug}",
        lambda: f"/collections/{slug}",
        lambda: f"/p/{g.hexs(8)}",
        lambda: f"/SalePage/Index/{n}",
        lambda: f"/search?q={slug}",
        lambda: f"/item.php?id={n}",
        lambda: f"/{lang}/{slug}/{n}",
        lambda: f"/blog/{slug}-{g.randint(1, 999)}",
        lambda: f"/shop/list?page={g.randint(2, 30)}&sort=new",
        # 新聞／商品常見的長數字 ID（兩類共用：避免 digit_ratio 被路徑數字帶偏）
        lambda: f"/{g.choice(['news', 'story', 'realtimenews', 'article', 'item'])}/{g.digits(g.randint(10, 20))}",
        lambda: f"/story/{g.digits(4)}/{g.digits(7)}",
        lambda: f"/{g.randint(2023, 2026)}{g.randint(1, 12):02d}{g.randint(1, 28):02d}{g.digits(6)}-{g.digits(6)}",
    ]
    if allow_root:
        options.append(lambda: "/")
    return g.choice(options)()


def tracking_query(g: _Gen) -> str:
    """廣告追蹤參數（兩類樣本共用；詐騙廣告的落地頁同樣帶 fbclid／gclid）。"""
    parts: List[Callable[[], str]] = [
        lambda: (f"utm_source={g.choice(TRACK_SOURCES)}&utm_medium={g.choice(TRACK_MEDIUMS)}"
                 f"&utm_campaign={g.choice(NEUTRAL_SLUGS).replace('-', '_')}_{g.randint(100, 9999)}"),
        lambda: "fbclid=IwAR" + g.chars(g.randint(40, 60), string.ascii_letters + string.digits + "_-"),
        lambda: "gclid=EAIaIQobChMI" + g.chars(g.randint(40, 70), string.ascii_letters + string.digits + "_-"),
        lambda: "gad_source=1&gclid=Cj0KCQ" + g.chars(g.randint(40, 60), string.ascii_letters + string.digits + "_"),
        lambda: "msclkid=" + g.hexs(32),
        lambda: "ttclid=" + g.alnum(g.randint(24, 40)),
        lambda: f"utm_content={g.choice(NEUTRAL_SLUGS)}&utm_term={g.choice(NEUTRAL_SLUGS)}",
        lambda: "cto_pld=" + g.chars(22, string.ascii_letters + string.digits + "-_"),
        lambda: f"ref={g.choice(TRACK_SOURCES)}",
    ]
    picked = g.rng.sample(range(len(parts)), g.randint(1, 3))
    query = "&".join(parts[i]() for i in picked)
    # gclid 兩種寫法不同時出現
    if query.count("gclid=") > 1:
        query = "&".join(p for p in query.split("&") if not p.startswith("gad_source"))
        seen, kept = set(), []
        for p in query.split("&"):
            name = p.split("=", 1)[0]
            if name not in seen:
                seen.add(name)
                kept.append(p)
        query = "&".join(kept)
    return query


def attach_query(path: str, query: str) -> str:
    """把 query 接在 path 後（若有 #fragment，接在 fragment 之前）。"""
    if not query:
        return path
    head, sep, frag = path.partition("#")
    if not head:
        head = "/"
    head += ("&" if "?" in head else "?") + query
    return head + (sep + frag if sep else "")


def with_tracking(g: _Gen, path: str, p: float) -> str:
    return attach_query(path, tracking_query(g)) if g.chance(p) else path


# ---- 詐騙特有路徑（研究報告 §2.2、§3.3、§4.3、§5.3） ----

UNIAPP_ROUTES = [
    "/#/pages/login/login", "/#/pages/register/register", "/#/pages/index/index", "/#/pages/mine/mine",
    "/#/pages/assets/index", "/#/pages/recharge/recharge", "/#/pages/withdraw/withdraw",
    "/#/pages/trade/index", "/#/pages/market/market", "/#/pages/ipo/index", "/#/pages/stock/detail",
    "/#/pages/subscribe/list",
]
H5_ROUTES = [
    "/h5/", "/h5/#/", "/h5/#/login", "/h5/#/register", "/wap/", "/wap/#/login", "/m/", "/app/",
    "/dist/", "/spc/", "/h5/index.html", "/wap/index",
]
DOWNLOAD_PATHS = [
    "/download.html", "/down.html", "/app.html", "/appdown.html", "/download.php", "/dl/app-release.apk",
    "/app/install.apk", "/ios/install.mobileconfig", "/static/app.ipa",
]
GAMBLING_PATHS = [
    "/promo", "/promotion", "/activity", "/vip", "/live", "/casino", "/slot", "/sports", "/lottery",
    "/fish", "/chess", "/game/lobby", "/zh-tw/promotions",
]
CRYPTO_ROUTES = [
    "/#/pages/contract/index", "/#/pages/second/index", "/#/pages/futures/index", "/#/kline",
    "/second-contract", "/#/pages/c2c/index", "/#/pages/otc/buy", "/#/pages/wallet/index",
    "/#/pages/assets/recharge", "/h5/#/pages/trade/kline",
]
DRAINER_PATHS = [
    "/claim", "/airdrop", "/connect", "/connect-wallet", "/walletconnect", "/approve", "/verify",
    "/sync", "/rectify", "/validate", "/migration", "/revoke", "/claim-reward",
]
PHISH_PATHS = [
    "/login", "/verify", "/member/verify", "/order/check", "/refund", "/parcel", "/ebank/login.html",
    "/account/update", "/kyc", "/card/activate", "/notice", "/payment/confirm",
]
INVITE_KEYS = ["invitecode", "inviteCode", "invite_code", "agentcode", "agent_code", "recommend_code",
               "share_code", "tjm", "yqm"]

GAMBLING_ZH = ["娛樂城", "百家樂", "真人視訊", "老虎機", "體驗金", "註冊送", "首儲", "返水", "線上娛樂",
               "電子遊藝", "百家乐", "娱乐城"]
INVESTMENT_ZH = ["飆股", "老師帶單", "投資群組", "抽籤", "保證獲利", "當沖", "AI選股", "新股申購", "財富自由",
                 "股票群組", "跟單", "穩賺不賠", "荐股", "带单"]
CRYPTO_ZH = ["虛擬貨幣", "泰達幣", "換U", "挖礦", "領取空投", "交易所", "錢包授權", "秒合約", "雲挖礦",
             "質押", "数字货币", "提币"]


def p_uniapp(g: _Gen) -> str:
    route = g.choice(UNIAPP_ROUTES)
    if g.chance(0.4):
        route += f"?{g.choice(INVITE_KEYS)}={g.alnum(g.randint(4, 8))}"
    return route


def p_h5(g: _Gen) -> str:
    route = g.choice(H5_ROUTES)
    if route.endswith("#/") and g.chance(0.4):
        route += g.choice(["register", "login", "pages/index/index"])
    return route


def p_register(g: _Gen) -> str:
    return g.choice([
        lambda: f"/register?code={g.digits(4)}",
        lambda: f"/register?{g.choice(INVITE_KEYS)}={g.alnum(6)}",
        lambda: f"/?agent={g.digits(4)}",
        lambda: f"/reg.html?ic={g.digits(5)}",
        lambda: f"/signup?{g.choice(INVITE_KEYS)}={g.digits(5)}",
        lambda: f"/#/register?code={g.digits(4)}",
        lambda: f"/?ic={g.digits(6)}",
    ])()


def p_download(g: _Gen) -> str:
    return g.choice(DOWNLOAD_PATHS)


def p_gambling(g: _Gen) -> str:
    path = g.choice(GAMBLING_PATHS)
    return path + (f"?agent={g.digits(4)}" if g.chance(0.3) else "")


def p_crypto(g: _Gen) -> str:
    return g.choice(CRYPTO_ROUTES)


def p_drainer(g: _Gen) -> str:
    return g.choice(DRAINER_PATHS)


def p_phish(g: _Gen) -> str:
    path = g.choice(PHISH_PATHS)
    return path + (f"?order={g.digits(8)}" if g.chance(0.25) else "")


def p_hex(g: _Gen) -> str:
    return "/" + g.hexs(8)


def p_mobile(g: _Gen) -> str:
    return g.choice(["/h5", "/wap", "/app", "/dist", "/spc", "/m/", "/h5/", "/wap/index.html"])


def p_zh(terms: Sequence[str]) -> Callable[[_Gen], str]:
    def make(g: _Gen) -> str:
        term = quote(g.choice(terms))
        return g.choice([
            lambda: f"/{term}",
            lambda: f"/{g.choice(['news', 'post', 'p', 'tw'])}/{term}",
            lambda: f"/?q={term}",
            lambda: f"/{term}?ref={g.digits(4)}",
        ])()
    return make


def _bare_or_root(g: _Gen, tracking: float) -> str:
    # 廣告直接導到首頁時也會帶 fbclid／gclid（https://x.top/?fbclid=…）
    return attach_query("/", tracking_query(g)) if g.chance(tracking) else g.choice(["", "/"])


def scam_path(g: _Gen, lure: Sequence[Callable[[_Gen], str]], bare: float = 0.30,
              neutral: float = 0.25, tracking: float = TRACKING_RATE) -> str:
    """詐騙擴增列的路徑：30% 裸網域、25% 一般路徑、45% 該類型特有路由；追蹤參數機率與正常樣本相同。"""
    r = g.rng.random()
    if r < bare:
        return _bare_or_root(g, tracking)
    if r < bare + neutral or not lure:
        return with_tracking(g, neutral_path(g), tracking)
    return with_tracking(g, g.choice(lure)(g), tracking)


def benign_path(g: _Gen, legit: Sequence[Callable[[_Gen], str]] = (), bare: float = 0.30,
                specific: float = 0.25, tracking: float = TRACKING_RATE) -> str:
    """hard negative 的路徑：30% 裸網域、其餘為一般路徑或該網站常見的合法路徑；追蹤參數機率同詐騙樣本。"""
    r = g.rng.random()
    if r < bare:
        return _bare_or_root(g, tracking)
    if r < bare + specific and legit:
        return with_tracking(g, g.choice(legit)(g), tracking)
    return with_tracking(g, neutral_path(g), tracking)


def compose(g: _Gen, host: str, path: str = "", scheme: Optional[str] = None) -> str:
    sch = g.scheme() if scheme is None else scheme
    return f"{sch}{host}{path}"


def maybe_www(g: _Gen, host: str, p: float = 0.3) -> str:
    return ("www." + host) if g.chance(p) and host.count(".") == 1 else host


# =============================================================================
# 擴增：165 四大類示意詐騙樣本（label=1）
# =============================================================================

LUCKY = ["168", "888", "666", "8899", "6688", "365", "99", "777", "518", "9988", "1314", "5566", "88",
         "8888", "2025", "6666", "998"]
GAM_PINYIN = ["yule", "yulecheng", "bocai", "caipiao", "duchang", "baijiale", "laohuji", "qipai", "zhenren",
              "pujing", "xinpujing", "jinsha", "weinisi", "weinisiren", "huangguan", "yongli", "taiyangcheng",
              "kaiyun", "leyu", "jiuyou", "yabo", "buyu", "liuhecai", "shishicai", "feiting"]
GAM_EN = ["casino", "baccarat", "slots", "slot", "poker", "jackpot", "sportsbook", "roulette", "blackjack",
          "betting", "keno", "sabong"]
GAM_TAILS = ["tw", "vip", "88", "168", "live", "club", "royal", "lucky", "asia", "win", "online", "666"]
GAM_ABBR = ["bc", "yl", "hg", "cp", "pj", "js", "vns", "ty", "dj", "ky", "ag", "mg", "pg"]
GAM_BRANDS = ["fun88", "w88", "12bet", "188bet", "1xbet", "betway", "bet365", "dafabet", "sportsbet"]
GAM_TLDS = ["vip", "top", "cc", "com", "net", "club", "xyz", "online", "bet", "casino", "win", "asia",
            "shop", "live", "fun", "site", "icu", "com", "vip", "cc"]
ZH_GAM_HOST = ["娛樂城", "百家樂", "真人視訊", "老虎機", "威尼斯人", "葡京", "太陽城", "博弈", "彩票", "棋牌",
               "捕魚機", "娱乐城"]

INV_PINYIN = ["tougu", "licai", "gupiao", "touzi", "qihuo", "waihui", "huangjin", "lianghua", "biaogu",
              "mingpai", "daidan", "gendan", "chouqian", "shengou", "zhengquan", "quanshang", "jijin", "feigu",
              "dangchong"]
INV_EN = ["stockvip", "vipstock", "aitrade", "aitrading", "quant", "ipo", "forex", "xau", "copytrade", "mt5",
          "mt4", "cfd", "bullion", "metatrader", "copytrading"]
INV_AFFIX = ["tw", "vip", "88", "168", "pro", "club", "ai", "666", "2025", "tw88", "plus", "king"]
INV_A = ["tw", "global", "smart", "alpha", "prime", "asia", "royal", "golden", "first", "star", "new"]
INV_B = ["stock", "stocks", "invest", "trade", "trading", "wealth", "capital", "fund", "finance",
         "securities", "markets", "futures", "asset"]
INV_C = ["vip", "pro", "club", "plus", "88", "168", "ai", "signal", "mentor", "elite", "master", "teacher"]
INV_TLDS = ["com", "vip", "top", "cc", "xyz", "club", "online", "shop", "net", "pro", "site", "cyou", "icu",
            "com", "vip", "top"]
SHUADAN_BRANDS = ["tmall", "taobao", "alibaba", "aliexpress", "amazon", "ebay", "shein", "lazada",
                  "tokopedia", "walmart"]
BROKER_SPOOF = ["cmegroup", "eightcap", "nasdaq", "nyse", "blackrock", "robinhood", "etoro", "bakkt",
                "interactivebrokers", "citi"]
BROKER_TAILS = ["xau", "btc", "pro", "vip", "tw", "global", "plus", "fx", "gold", "ai"]

EX_BRANDS = ["binance", "okx", "okex", "bybit", "bitget", "kucoin", "coinbase", "kraken", "mexc", "htx",
             "huobi", "bitfinex", "bitflyer", "upbit", "bingx", "maicoin", "bitopro", "hoyabit", "xrex",
             "bitmart", "phemex", "coinw"]
EX_AFFIX = ["pro", "vip", "tw", "app", "global", "web3", "defi", "ex", "swap", "plus", "88", "168", "hk",
            "asia", "exchange", "futures", "wallet", "online"]
EX_WORDS = ["dawn", "nova", "orbit", "lumi", "zen", "astro", "coinx", "bitz", "hpro", "megab", "sky", "tiger",
            "lotus", "aurora", "vega", "titan", "solar", "neon"]
CRYPTO_TERMS_HOST = ["usdt", "usdc", "trc20", "otc", "defi", "web3", "staking", "airdrop", "cloudmining",
                     "hashrate", "dapp", "p2p", "jiaoyisuo", "qianbao", "wakuang", "erc20"]
CRYPTO_TAILS = ["tw", "vip", "pro", "club", "88", "asia", "global", "hub", "online", "666", "plus"]
WALLETS = ["metamask", "trustwallet", "tronlink", "imtoken", "tokenpocket", "safepal", "ledger", "trezor",
           "uniswap", "pancakeswap", "opensea", "walletconnect"]
WALLET_ACTS = ["verify", "sync", "claim", "connect", "recover", "airdrop", "support", "validate", "rectify",
               "restore"]
EX_TLDS = ["com", "vip", "top", "cc", "xyz", "net", "pro", "online", "site", "shop", "icu", "lol", "quest",
           "com", "top", "vip"]

FREE_SUFFIXES = ["pages.dev", "web.app", "firebaseapp.com", "vercel.app", "netlify.app", "herokuapp.com",
                 "onrender.com", "glitch.me", "replit.app", "github.io", "gitlab.io", "weebly.com", "webflow.io",
                 "notion.site", "azurewebsites.net", "workers.dev", "000webhostapp.com", "appspot.com",
                 "framer.website", "carrd.co", "blogspot.com", "fly.dev", "deno.dev", "surge.sh",
                 "godaddysites.com", "mystrikingly.com"]
LURE_BRAND = ["binance", "okx", "fubon", "cathay", "esun", "ctbc", "shopee", "momo", "yuanta", "bybit",
              "sinopac", "post"]
LURE_STRONG = ["stock", "invest", "usdt", "casino", "airdrop", "wallet", "tougu", "yule", "bocai", "licai",
               "stockvip", "crypto", "defi", "web3", "ipo", "forex", "bonus", "refund", "loan"]
LURE_WEAK = ["tw", "vip", "login", "verify", "club", "pro", "app", "online", "service", "88", "168"]

BANK_BRANDS = ["fubon", "esun", "esunbank", "cathay", "cathaybk", "ctbc", "ctbcbank", "yuanta", "sinopac",
               "sinotrade", "kgi", "taishin", "megabank", "firstbank", "landbank", "hncb", "scsb", "skbank",
               "ubot", "richart", "nextbank", "citibank", "hsbc", "masterlink"]
BANK_AFFIX = ["vip", "tw", "bank", "login", "verify", "service", "online", "secure", "loan", "card", "ebank",
              "app", "168", "88", "netbank", "support", "member", "pay", "securities", "stock"]
GOV_A = ["165", "npa", "mof", "moi", "nhi", "fsc", "etax", "post", "gov"]
GOV_B = ["gov", "tw", "refund", "tax", "police", "antifraud", "fine", "subsidy", "fraud"]
GOV_C = ["tw", "service", "online", "center", "notice", "portal"]
ECOM_BRANDS = ["shopee", "momo", "momoshop", "pchome", "ruten", "poya", "poyabuy", "watsons", "cosmed",
               "costco", "myship", "newebpay", "ecpay", "jkopay", "taobao", "tmall", "alibaba", "amazon",
               "ebay", "shein", "lazada", "rakuten", "klook", "netflix", "apple", "microsoft", "familymart",
               "tiktok", "paypal"]
ECOM_AFFIX = ["mall", "shop", "vip", "tw", "official", "sale", "store", "88", "pay", "order", "verify",
              "refund", "service", "member", "event", "promo", "gift", "online", "support"]
BRAND_TLDS = ["com", "net", "top", "xyz", "cc", "vip", "online", "shop", "info", "site", "store", "app",
              "co", "com", "com", "top", "xyz", "live", "icu", "club"]
RISKY_TLDS = ["top", "xyz", "cc", "vip", "online", "shop", "icu", "sbs", "cyou", "site", "lol", "buzz"]
RANDOM_TLDS = ["com", "com", "com", "com", "vip", "top", "cc", "xyz", "shop", "store", "asia", "cn", "live",
               "pro", "icu", "sbs", "cyou", "lol", "quest", "online", "site", "net"]
LEGIT_FOR_AT = ["www.bot.com.tw", "www.landbank.com.tw", "www.cathaybk.com.tw", "www.momoshop.com.tw",
                "shopee.tw", "www.okx.com", "line.me", "www.fubon.com", "www.sinopac.com", "www.post.gov.tw",
                "accounts.google.com", "www.apple.com", "www.ctbcbank.com", "www.taishinbank.com.tw"]
OFFICIAL_FOR_SUB = ["cathaybk.com.tw", "esunbank.com", "momoshop.com.tw", "shopee.tw", "fubon.com", "line.me",
                    "okx.com", "bot.com.tw", "post.gov.tw", "pchome.com.tw", "bybit.com", "taishinbank.com.tw"]
HOMOGLYPH = {"a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "y": "у",
             "x": "х", "i": "і"}
HOMOGLYPH_BRANDS = ["binance", "shopee", "paypal", "apple", "esunbank", "cathaybk", "fubon", "coinbase",
                    "momoshop", "yahoo", "facebook", "microsoft", "bybit", "kucoin", "sinopac"]


def _sep(g: _Gen) -> str:
    return g.choice(["", "-", "", "-", ""])


def _join(g: _Gen, parts: Sequence[str]) -> str:
    sep = _sep(g)
    return sep.join(p for p in parts if p)


def _public_ip(g: _Gen) -> str:
    first = g.choice([23, 45, 47, 52, 103, 104, 107, 118, 139, 154, 156, 185, 203, 207, 210, 211, 220])
    return f"{first}.{g.randint(1, 254)}.{g.randint(1, 254)}.{g.randint(1, 254)}"


def _homoglyph(g: _Gen, word: str) -> str:
    idx = [i for i, ch in enumerate(word) if ch in HOMOGLYPH]
    if not idx:
        return word
    i = g.choice(idx)
    return word[:i] + HOMOGLYPH[word[i]] + word[i + 1:]


def gen_scam_gambling(g: _Gen) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    lure = [p_register, p_h5, p_gambling, p_download, p_uniapp, p_zh(GAMBLING_ZH)]
    for _ in range(22):
        host = _join(g, [g.choice(["", "", "tw", "vip"]), g.choice(GAM_PINYIN), g.choice(LUCKY)])
        rows.append((compose(g, maybe_www(g, f"{host}.{g.choice(GAM_TLDS)}"), scam_path(g, lure)),
                     "gam_pinyin_number"))
    for _ in range(14):
        host = _join(g, [g.choice(["", "tw", "vip", "best", "royal"]), g.choice(GAM_EN), g.choice(GAM_TAILS)])
        rows.append((compose(g, maybe_www(g, f"{host}.{g.choice(GAM_TLDS)}"), scam_path(g, lure)), "gam_english"))
    for _ in range(12):
        host = f"{g.choice(GAM_ABBR)}{g.choice(LUCKY)}{g.choice(['', '', g.choice(LUCKY)])}"
        rows.append((compose(g, maybe_www(g, f"{host}.{g.choice(RISKY_TLDS)}"), scam_path(g, lure)), "gam_abbrev_number"))
    for _ in range(6):
        host = g.choice(LUCKY) + g.digits(g.randint(2, 4))
        rows.append((compose(g, f"{host}.{g.choice(['cc', 'vip', 'top', 'com', 'xyz', 'net'])}",
                             scam_path(g, lure)), "gam_numeric"))
    for _ in range(8):
        host = f"{g.choice(ZH_GAM_HOST)}{g.choice(['', g.choice(LUCKY)])}"
        rows.append((compose(g, f"{host}.{g.choice(['vip', 'com', 'top', 'cc', 'net', 'xyz'])}",
                             scam_path(g, lure)), "gam_idn"))
    for _ in range(8):
        host = g.choice([g.random_sld(5, 8), f"{g.random_sld(3, 5)}{g.digits(3)}"])
        path = with_tracking(g, p_zh(GAMBLING_ZH)(g), 0.2)
        rows.append((compose(g, f"{host}.{g.choice(RANDOM_TLDS)}", path), "gam_zh_path"))
    for _ in range(6):
        host = _join(g, [g.choice(GAM_BRANDS), g.choice(["tw", "vip", "asia", "88", "app", "zh"])])
        rows.append((compose(g, maybe_www(g, f"{host}.{g.choice(GAM_TLDS)}"), scam_path(g, lure)), "gam_known_brand"))
    return rows


def gen_scam_investment(g: _Gen) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    lure = [p_uniapp, p_h5, p_register, p_download, p_zh(INVESTMENT_ZH), p_hex, p_mobile]
    for _ in range(18):
        host = _join(g, [g.choice(["", "", "tw", "vip"]), g.choice(INV_PINYIN), g.choice(INV_AFFIX)])
        rows.append((compose(g, maybe_www(g, f"{host}.{g.choice(INV_TLDS)}"), scam_path(g, lure)), "inv_pinyin"))
    for _ in range(12):
        host = _join(g, [g.choice(["", "tw", "my", "best"]), g.choice(INV_EN), g.choice(INV_AFFIX)])
        rows.append((compose(g, maybe_www(g, f"{host}.{g.choice(INV_TLDS)}"), scam_path(g, lure)), "inv_english"))
    for _ in range(12):
        sep = g.choice(["-", ""])
        host = sep.join([g.choice(INV_A), g.choice(INV_B), g.choice(INV_C)])
        rows.append((compose(g, maybe_www(g, f"{host}.{g.choice(INV_TLDS)}"), scam_path(g, lure)), "inv_weak_combo"))
    for _ in range(12):
        sub = g.choice(["app", "h5", "stocks", "trade", "m", "sj", "tw1", "zh", "main", f"{g.randint(2024, 2026)}edge"])
        host = f"{sub}.{g.random_sld(5, 8)}{g.choice(['', '', str(g.randint(1, 99))])}.{g.choice(['com', 'vip', 'top', 'xyz', 'cc', 'shop'])}"
        rows.append((compose(g, host, scam_path(g, [p_uniapp, p_h5, p_download, p_mobile], bare=0.25,
                                                 neutral=0.15)), "inv_random_subdomain"))
    for _ in range(10):
        kind = g.randint(0, 2)
        if kind == 0:
            host = f"{g.choice(SHUADAN_BRANDS)}{g.digits(g.randint(4, 8))}.{g.choice(['vip', 'com', 'shop', 'top', 'cc'])}"
        elif kind == 1:
            host = f"{g.choice(['shop', 'mall', 'store', 'seller'])}{g.choice(LUCKY)}.{g.choice(['life', 'shop', 'store', 'online', 'vip'])}"
        else:
            host = f"{g.choice(['emma', 'lucas', 'olivia', 'mason', 'chloe', 'ryan'])}shopping.online"
        rows.append((compose(g, maybe_www(g, host), scam_path(g, [p_uniapp, p_register, p_h5])), "inv_shuadan_mall"))
    for _ in range(8):
        host = g.choice([g.random_sld(5, 8), f"{g.choice(['tw', 'my', 'go'])}{g.random_sld(4, 6)}"])
        path = with_tracking(g, p_zh(INVESTMENT_ZH)(g), 0.3)
        rows.append((compose(g, f"{host}.{g.choice(RANDOM_TLDS)}", path), "inv_zh_path"))
    for _ in range(8):
        host = _join(g, [g.choice(BROKER_SPOOF), g.choice(BROKER_TAILS)])
        rows.append((compose(g, maybe_www(g, f"{host}.{g.choice(INV_TLDS)}"), scam_path(g, lure)), "inv_broker_spoof"))
    return rows


def gen_scam_exchange(g: _Gen) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    lure = [p_uniapp, p_h5, p_crypto, p_download, p_register, p_drainer, p_zh(CRYPTO_ZH)]
    for _ in range(24):
        parts = [g.choice(EX_BRANDS), g.choice(EX_AFFIX)]
        if g.chance(0.3):
            parts.append(g.choice(["tw", "vip", "88", "app"]))
        if g.chance(0.2):
            parts.reverse()
        rows.append((compose(g, maybe_www(g, f"{_join(g, parts)}.{g.choice(EX_TLDS)}"), scam_path(g, lure)),
                     "ex_brand_affix"))
    for _ in range(10):
        host = f"{g.choice(EX_WORDS)}{g.choice(['ex', '-ex', 'ex'])}"
        rows.append((compose(g, maybe_www(g, f"{host}.{g.choice(EX_TLDS)}"), scam_path(g, lure)), "ex_suffix"))
    for _ in range(14):
        host = _join(g, [g.choice(CRYPTO_TERMS_HOST), g.choice(CRYPTO_TAILS)])
        if g.chance(0.25):
            host = f"{g.choice(['tw', 'my', 'go'])}-{host}"
        rows.append((compose(g, maybe_www(g, f"{host}.{g.choice(EX_TLDS)}"), scam_path(g, lure)), "ex_crypto_terms"))
    for _ in range(12):
        host = f"{g.choice(WALLETS)}-{g.choice(WALLET_ACTS)}"
        rows.append((compose(g, f"{host}.{g.choice(EX_TLDS)}", scam_path(g, [p_drainer], bare=0.25, neutral=0.1)),
                     "ex_wallet_drainer"))
    for _ in range(8):
        kind = g.randint(0, 2)
        if kind == 0:
            host = f"m.{g.random_sld(2, 3)}jys.{g.choice(['top', 'vip', 'cc'])}"
        elif kind == 1:
            host = f"{g.choice(EX_BRANDS)}{g.lower_alnum(3)}.com"
        else:
            host = f"{g.random_sld(3, 5)}jiaoyisuo.{g.choice(['com', 'vip', 'top'])}"
        rows.append((compose(g, host, scam_path(g, [p_mobile, p_h5, p_crypto], bare=0.3, neutral=0.1)), "ex_jys_mobile"))
    return rows


def gen_scam_ephemeral(g: _Gen) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    lure = [p_uniapp, p_phish, p_drainer, lambda gg: "/invest", lambda gg: "/index.html"]
    for _ in range(30):
        parts = [g.choice(LURE_STRONG), g.choice(LURE_WEAK)]
        if g.chance(0.5):
            parts.insert(0, g.choice(LURE_BRAND))
        name = "-".join(parts) if g.chance(0.7) else "".join(parts)
        suffix = g.choice(FREE_SUFFIXES)
        path = scam_path(g, lure, bare=0.35, neutral=0.2)
        if suffix in ("weebly.com", "godaddysites.com", "mystrikingly.com") and g.chance(0.5):
            host = f"{name}.{suffix}"
        elif g.chance(0.12):
            # Wix：user.wixsite.com/<site>（誘因字在路徑，不做形狀翻轉）
            host, path = f"{g.random_sld(5, 8)}.wixsite.com", f"/{name}{path if path.startswith('/') else ''}"
            rows.append((compose(g, host, path, scheme="https://"), "host_free_lure_wix"))
            continue
        else:
            host = f"{name}.{suffix}"
        rows.append((compose(g, host, path, scheme="https://"), "host_free_lure"))
    for _ in range(4):
        name = "-".join([g.choice(LURE_STRONG), g.choice(LURE_WEAK)])
        rows.append((compose(g, f"{name}.s3.amazonaws.com", g.choice(["/index.html", "/login.html", "/claim.html"]),
                             scheme="https://"), "host_object_storage"))
        rows.append((compose(g, f"pub-{g.hexs(32)}.r2.dev", f"/{name}.html", scheme="https://"),
                     "host_object_storage"))
    words = ["gentle", "river", "quiet", "forest", "lamp", "silver", "ocean", "maple", "brave", "hidden", "lucky",
             "magic", "rapid", "solar", "tiny", "violet", "winter", "zebra", "canyon", "meadow"]
    for _ in range(14):
        kind = g.randint(0, 6)
        if kind == 0:
            host = "-".join(g.choice(words) for _ in range(4)) + ".trycloudflare.com"
        elif kind == 1:
            host = f"{g.hexs(4)}-{_public_ip(g).replace('.', '-')}.ngrok-free.app"
        elif kind == 2:
            host = f"{g.hexs(8)}.ngrok.io"
        elif kind == 3:
            host = f"{g.choice(words)}-{g.choice(words)}-{g.randint(10, 99)}.loca.lt"
        elif kind == 4:
            host = f"{g.hexs(12)}.serveo.net"
        elif kind == 5:
            host = f"{g.hexs(10)}.lhr.life"
        else:
            host = f"{g.lower_alnum(8)}-8080.asse.devtunnels.ms"
        rows.append((compose(g, host, scam_path(g, [p_phish, p_drainer, p_uniapp], bare=0.4, neutral=0.2),
                             scheme="https://"), "host_tunnel"))
    for _ in range(6):
        kind = g.randint(0, 3)
        if kind == 0:
            url = f"https://ipfs.io/ipfs/Qm{g.chars(44, _B58)}"
        elif kind == 1:
            url = f"https://cf-ipfs.com/ipfs/bafy{g.chars(55, _B32)}"
        elif kind == 2:
            url = f"https://bafy{g.chars(55, _B32)}.ipfs.dweb.link/"
        else:
            url = f"https://gateway.pinata.cloud/ipfs/Qm{g.chars(44, _B58)}/index.html"
        rows.append((url + g.choice(["", "#login", "?claim=1"]), "host_ipfs"))
    return rows


def gen_scam_social(g: _Gen) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    tails = ["", "", "?utm_source=invitation&utm_medium=link_copy", "?openQrModal=true", "?ref=ad"]
    makers: List[Callable[[], str]] = [
        lambda: f"https://line.me/ti/g2/{g.alnum(g.randint(16, 22))}",
        lambda: f"https://line.me/R/ti/g/{g.alnum(10)}",
        lambda: f"https://line.me/ti/g/{g.alnum(10)}",
        lambda: f"https://t.me/+{g.alnum(16)}",
        lambda: f"https://t.me/joinchat/{g.alnum(22)}",
        lambda: f"https://telegram.me/joinchat/{g.alnum(22)}",
        lambda: f"https://chat.whatsapp.com/{g.alnum(22)}",
        lambda: f"https://discord.gg/{g.alnum(8)}",
        lambda: f"https://discord.com/invite/{g.alnum(8)}",
        lambda: f"https://lin.ee/{g.alnum(7)}",
    ]
    for i in range(24):
        url = makers[i % len(makers)]() + g.choice(tails)
        rows.append((url, "social_group_invite"))
    return rows


def gen_scam_brand(g: _Gen) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    for _ in range(26):
        parts = [g.choice(BANK_BRANDS), g.choice(BANK_AFFIX)]
        if g.chance(0.3):
            parts.append(g.choice(["tw", "online", "service", "168"]))
        if g.chance(0.15):
            parts.reverse()
        host = _join(g, parts)
        rows.append((compose(g, maybe_www(g, f"{host}.{g.choice(BRAND_TLDS)}"),
                             scam_path(g, [p_phish, p_uniapp, p_download])), "brand_bank_broker"))
    for _ in range(12):
        host = "-".join([g.choice(GOV_A), g.choice(GOV_B), g.choice(GOV_C)])
        rows.append((compose(g, f"{host}.{g.choice(RISKY_TLDS + ['com', 'net', 'info'])}",
                             scam_path(g, [p_phish, lambda gg: "/refund", lambda gg: "/subsidy/apply"])), "brand_gov_spoof"))
    for _ in range(26):
        parts = [g.choice(ECOM_BRANDS), g.choice(ECOM_AFFIX)]
        if g.chance(0.3):
            parts.append(g.choice(["tw", "88", "vip", "official"]))
        if g.chance(0.15):
            parts.reverse()
        host = _join(g, parts)
        rows.append((compose(g, maybe_www(g, f"{host}.{g.choice(BRAND_TLDS)}"),
                             scam_path(g, [p_phish, p_uniapp, p_register])), "brand_ecommerce"))
    return rows


def gen_scam_obfuscation(g: _Gen) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    for _ in range(10):
        evil = f"{_join(g, [g.choice(LURE_BRAND + ['secure', 'account']), g.choice(['verify', 'login', 'secure', 'check', 'auth'])])}.{g.choice(RISKY_TLDS)}"
        rows.append((compose(g, f"{g.choice(LEGIT_FOR_AT)}@{evil}", g.choice(["/", "/login", "/verify", ""]),
                             scheme=g.choice(["https://", "http://"])), "obf_userinfo_at"))
    for _ in range(10):
        port = g.choice(["", "", ":8080", ":8888", ":9000", ":6688"])
        path = g.choice([p_h5, p_uniapp, p_phish, p_register, p_download])(g)
        rows.append((compose(g, f"{_public_ip(g)}{port}", path, scheme=g.choice(["http://", "http://", "https://"])),
                     "obf_ip_host"))
    for _ in range(8):
        host = f"{_join(g, [g.choice(['tw', 'vip', 'my']), g.choice(['stock', 'usdt', 'casino', 'invest', 'wallet'])])}.{g.choice(RISKY_TLDS)}"
        port = g.choice([":8080", ":8443", ":8888", ":9000", ":8000", ":8081", ":7777", ":6688"])
        rows.append((compose(g, host + port, scam_path(g, [p_uniapp, p_h5, p_phish], bare=0.3, neutral=0.1)),
                     "obf_non_standard_port"))
    for _ in range(8):
        brand = _homoglyph(g, g.choice(HOMOGLYPH_BRANDS))
        tld = g.choice(["com", "net", "top", "com.tw", "xyz", "co"])
        rows.append((compose(g, f"{maybe_www(g, brand + '.' + tld, 0.4)}", g.choice(["", "/", "/login", "/zh-tw/"]),
                             scheme="https://"), "obf_punycode_homoglyph"))
    for _ in range(8):
        host = (f"{g.choice(OFFICIAL_FOR_SUB)}.{g.choice(['secure', 'account', 'member', 'auth'])}-"
                f"{g.choice(['verify', 'login', 'check', 'update'])}.{g.choice(RISKY_TLDS)}")
        rows.append((compose(g, host, g.choice(["", "/", "/login", "/verify"]), scheme="https://"),
                     "obf_official_in_subdomain"))
    return rows


def gen_scam_random_domain(g: _Gen) -> List[Tuple[str, str]]:
    """165 公告約四成是「沒有任何關鍵字」的隨機字母網域（研究報告 §0-1、§2.1）。"""
    rows: List[Tuple[str, str]] = []
    subs = ["", "", "", "www.", "app.", "h5.", "m.", "stocks.", "trade.", "sj.", "tw.", "tw1.", "zh.", "main."]
    for _ in range(40):
        host = f"{g.choice(subs)}{g.random_sld(5, 9)}{g.choice(['', '', '', str(g.randint(1, 99))])}.{g.choice(RANDOM_TLDS)}"
        path = scam_path(g, [p_mobile, p_h5, p_uniapp, p_hex], bare=0.35, neutral=0.20)
        rows.append((compose(g, host, path), "random_sld"))
    return rows


SCAM_GENERATORS: List[Tuple[str, Callable[[_Gen], List[Tuple[str, str]]]]] = [
    ("scam_gambling", gen_scam_gambling),
    ("scam_investment", gen_scam_investment),
    ("scam_fake_exchange", gen_scam_exchange),
    ("scam_ephemeral_host", gen_scam_ephemeral),
    ("scam_social_invite", gen_scam_social),
    ("scam_brand_impersonation", gen_scam_brand),
    ("scam_url_obfuscation", gen_scam_obfuscation),
    ("scam_random_domain", gen_scam_random_domain),
]


# =============================================================================
# 擴增：hard negatives（label=0；網域刻意避開 url_cases.json 驗收清單）
# =============================================================================

HN_SUBSTRING = [
    "www.learningapps.org", "learnenglish.britishcouncil.org", "www.codecademy.com", "www.edx.org",
    "learn.adafruit.com", "www.earnest.com", "www.betaseries.com", "betanews.com", "www.tibet.net",
    "www.betterworldbooks.com", "www.betterhealth.vic.gov.au", "diabetesjournals.org", "www.diabetes.org.uk",
    "winmerge.org", "www.windowscentral.com", "www.winzip.com", "www.casio-intl.com", "www.casio.co.jp",
    "tw.portal-pokemon.com", "www.pokemon.com", "www.vip.com", "math.stackexchange.com", "superuser.com",
    "www.x-rates.com", "www.exchangerate-api.com", "outlook.office365.com", "www.fedex.com", "www.rolex.com",
    "www.dexcom.com", "www.latex-project.org", "www.timex.com", "www.kleenex.com", "www.mining.com",
    "www.miningweekly.com", "www.istockphoto.com", "stockx.com", "www.shutterstock.com", "www.goldengoose.com",
    "www.online-convert.com", "www.airlinequality.com", "linear.app", "lineageos.org", "www.ea.com",
    "www.knowledgehut.com", "www.sportslottery.com.tw", "article.sportslottery.com.tw", "www.methodhome.com",
    "www.togetherwecan.org.tw", "www.bettermoneyhabits.com", "www.alphabetagamma.com", "www.windriver.com",
]
HN_MEDIA_CONTENT = [   # CONTENT_PLATFORM_DOMAINS：路徑可帶中文詐騙詞（報導）
    "www.nownews.com", "www.ftvnews.com.tw", "news.ebc.net.tw", "newtalk.tw", "www.thenewslens.com",
    "www.ctee.com.tw", "technews.tw", "www.ithome.com.tw", "www.gvm.com.tw", "www.rti.org.tw",
    "www.ctwant.com", "www.upmedia.mg", "tw.nextapple.com", "www.mygopen.com", "tfc-taiwan.org.tw",
    "cofacts.tw", "vocus.cc", "www.blocktempo.com", "abmedia.io", "zombit.info", "www.coindesk.com",
    "cointelegraph.com", "www.hk01.com", "www.bbc.com", "www.reuters.com", "www.commonhealth.com.tw",
    "life.tw", "medium.com", "www.pixnet.net", "fraud.tw",
]
HN_MEDIA_OTHER = [     # 其他新聞／內容網站：只放一般路徑
    "www.twreporter.org", "www.inside.com.tw", "www.managertoday.com.tw", "www.taiwannews.com.tw",
    "www.nippon.com", "www.mobile01.com", "www.eprice.com.tw", "www.kocpc.com.tw", "www.techbang.com",
    "www.nikkei.com", "www.theguardian.com", "www.dw.com", "www.voacantonese.com", "www.cts.com.tw",
    "www.ttv.com.tw",
]
ZH_NEWS_TITLES = [
    "線上娛樂城詐騙手法曝光", "假投資網站騙走退休金", "飆股群組老師帶單是詐騙", "虛擬貨幣詐騙集團落網",
    "USDT換匯詐騙", "博弈集團洗錢遭起訴", "刷單詐騙兼職陷阱", "AI選股詐騙新手法", "假交易所出金失敗",
    "百家樂代操詐騙", "警方破獲假投資機房", "防詐宣導保證獲利都是騙局",
]
HN_FINANCE = [
    "www.hncb.com.tw", "www.bankchb.com", "www.tcb-bank.com.tw", "www.scsb.com.tw", "www.skbank.com.tw",
    "www.ubot.com.tw", "www.feib.com.tw", "www.yuantabank.com.tw", "www.kgibank.com.tw", "www.o-bank.com",
    "www.tbb.com.tw", "www.sunnybank.com.tw", "www.rakuten-bank.com.tw", "www.hsbc.com.tw", "www.sc.com",
    "www.citibank.com.tw", "www.masterlink.com.tw", "www.pscnet.com.tw", "www.entrust.com.tw",
    "www.emega.com.tw", "www.ctbcsec.com", "www.esunsec.com.tw", "www.capital.com.tw", "www.capitalfutures.com.tw",
    "www.yuantafutures.com.tw", "www.megafutures.com.tw", "www.concordfutures.com.tw", "www.tsfutures.com.tw",
    "www.cathayfut.com.tw", "www.spf.com.tw", "www.kgif.com.tw", "www.fbs.com.tw", "www.cathay-cube.com.tw",
    "esun.co", "www.ecpay.com.tw", "www.jkopay.com", "www.easycard.com.tw", "www.newebpay.com",
    "www.sfb.gov.tw", "www.banking.gov.tw", "www.cbc.gov.tw", "www.sfipc.org.tw", "www.twsa.org.tw",
    "www.wantgoo.com", "goodinfo.tw", "histock.tw", "www.investing.com", "tw.tradingview.com",
    "www.fundrich.com.tw", "www.money101.com.tw",
]
HN_CRYPTO = [
    "www.bitget.com", "www.kucoin.com", "www.mexc.com", "www.htx.com", "www.bitfinex.com", "bitflyer.com",
    "upbit.com", "bingx.com", "hoyabit.com", "xrex.io", "trustwallet.com", "www.tronlink.org", "token.im",
    "www.tokenpocket.pro", "www.safepal.com", "trezor.io", "www.ledger.com", "app.uniswap.org",
    "pancakeswap.finance", "opensea.io", "bscscan.com", "tronscan.org", "tether.to", "www.bitmart.com",
    "phemex.com", "www.coinw.com", "www.bakkt.com",
]
HN_CLOUD = [
    "fastapi.tiangolo.com", "docs.djangoproject.com", "developer.mozilla.org", "nodejs.org", "react.dev",
    "vuejs.org", "nextjs.org", "devcenter.heroku.com", "render.com", "docs.railway.app", "gitlab.com",
    "bitbucket.org", "readthedocs.org", "requests.readthedocs.io", "flask.palletsprojects.com",
    "pandas.pydata.org", "numpy.org", "pytorch.org", "scikit-learn.org", "hackmd.io", "g0v.hackmd.io",
    "www.postman.com", "codepen.io", "jupyter.org", "portal.azure.com",
]
HN_FREE_HOSTING_BENIGN = [
    "google.github.io", "facebook.github.io", "microsoft.github.io", "karpathy.github.io", "jakevdp.github.io",
    "ntu-csie-ml.github.io", "g0v.github.io", "taipei-opendata.github.io", "chen-portfolio.vercel.app",
    "nextjs-blog-demo.vercel.app", "taipei-bus-map.vercel.app", "weather-dashboard-tw.vercel.app",
    "hackathon-taipei.netlify.app", "design-system-docs.netlify.app", "my-portfolio-2025.netlify.app",
    "cat-gallery.pages.dev", "open-data-viz.pages.dev", "team-wiki.pages.dev", "recipe-share-demo.web.app",
    "bus-schedule-tw.web.app", "todo-api-demo.herokuapp.com", "acme-careers.notion.site",
    "docs-company.gitbook.io", "photo-club.weebly.com", "hiking-diary.blogspot.com",
]
HN_MOBILE = [
    "m.yes123.com.tw", "m.housefun.com.tw", "m.mobile01.com", "m.eslite.com", "m.sinyi.com.tw", "m.sogo.com.tw",
    "m.imdb.com", "m.bilibili.com", "m.weibo.cn", "h5.m.jd.com", "m.ctee.com.tw", "m.bahamut.com.tw",
    "mobile.twitter.com", "m.vip.com", "m.jd.com", "m.kkday.com", "m.agoda.com", "m.trip.com",
    "app.powerbi.com", "app.slack.com", "app.hubspot.com", "app.asana.com", "app.diagrams.net",
    "app.box.com", "wap.cht.com.tw", "m.plurk.com", "m.thsrc.com.tw", "mobile.yahoo.co.jp",
    "m.ebc.net.tw", "m.ftvnews.com.tw", "m.match.net.tw", "m.inside.com.tw", "m.nownews.com",
    "m.hk01.com", "m.mirror.co.uk", "m.economictimes.com", "m.timesofindia.com", "m.huffpost.com",
    "m.espn.com", "m.allrecipes.com", "m.wikihow.com", "m.ikea.com", "m.uniqlo.com", "m.pinkoi.com",
    "m.kkbox.com", "m.books.com.cn", "m.ctrip.com", "m.ly.com", "m.qunar.com", "m.dianping.com",
    "m.meituan.com", "m.suning.com", "m.vipshop.com", "m.zol.com.cn", "m.autohome.com.cn",
    "app.zoom.us", "app.clickup.com", "app.mural.co", "app.frame.io", "app.airtable.com", "app.datadoghq.com",
    "h5.qq.com", "h5.youzan.com", "wap.sogou.com", "m.mafengwo.cn",
]
HN_RETAIL = [
    "www.nike.com", "www.adidas.com.tw", "www.decathlon.tw", "www.ikea.com.tw", "www.muji.com",
    "www.sogo.com.tw", "www.hola.com.tw", "www.nitori-net.tw", "www.uniqlo.com.tw", "tw.coupang.com",
    "www.pinkoi.com", "www.kkday.com", "www.klook.com", "www.agoda.com", "www.booking.com", "tw.hotels.com",
    "www.eslite.com", "www.kingstone.com.tw", "www.sanmin.com.tw", "www.obdesign.com.tw", "www.lativ.com.tw",
    "www.gu-global.com", "www.zara.com", "www.dyson.com.tw", "www.asus.com", "www.acer.com", "www.mi.com",
    "www.lenovo.com", "www.dell.com", "www.sony.com.tw", "www.rt-mart.com.tw", "www.yesstyle.com",
]
HN_GOV = [
    "www.moea.gov.tw", "www.mohw.gov.tw", "www.nhi.gov.tw", "www.mvdis.gov.tw", "www.ris.gov.tw",
    "www.kcg.gov.tw", "www.tycg.gov.tw", "www.ntpc.gov.tw", "1999.gov.taipei", "10000.gov.tw",
    "165dashboard.tw", "www.moi.gov.tw", "www.ncc.gov.tw", "www.nhri.org.tw", "www.nccu.edu.tw",
    "www.ncku.edu.tw", "www.nycu.edu.tw", "www.fju.edu.tw", "www.ntust.edu.tw", "www.yzu.edu.tw",
    "fraudbuster.digiat.org.tw", "moda.gov.tw", "www.cy.gov.tw", "www.judicial.gov.tw", "www.taiwan.net.tw",
    "www.nkust.edu.tw", "www.tku.edu.tw", "www.tpech.gov.taipei", "www.mjib.gov.tw",
]
HN_SOCIAL_OFFICIAL = ["tw165", "cathaybk", "esunbank", "momoshop", "pchome24h", "7-eleven", "fubon", "taishin",
                      "sinopac", "ctbcbank", "poyabuy", "carrefour"]
HN_NUMERIC = [
    "www.1111.com.tw", "www.17life.com", "www.163.com", "www.4399.com", "www.17track.net", "www.123rf.com",
    "9gag.com", "www.mycard520.com.tw", "www.51job.com", "www.58.com", "99designs.com", "1password.com",
    "www.3m.com.tw", "37signals.com", "500px.com", "www.23andme.com", "www.360.cn", "www.88db.com",
    "www.7net.com.tw", "www.24h.com.vn",
]
HN_PLAIN_GTLD = [      # 一般 .com/.net/.org 合法網站：真實資料的 .com 詐騙比例偏高，避免「普通 .com = 詐騙」
    "www.officedepot.com", "www.homedepot.com", "www.bestbuy.com", "www.nationalgeographic.com",
    "www.theverge.com", "www.washingtonpost.com", "www.bloomberg.com", "www.economist.com", "www.healthline.com",
    "www.webmd.com", "www.allrecipes.com", "www.tripadvisor.com", "www.expedia.com", "www.airbnb.com",
    "www.dropbox.com", "www.salesforce.com", "www.atlassian.com", "www.zendesk.com", "mailchimp.com",
    "www.squarespace.com", "wordpress.org", "www.mozilla.org", "www.python.org", "www.rust-lang.org",
    "www.cloudflare.com", "www.digitalocean.com", "www.adobe.com", "www.canva.com", "www.figma.com",
    "www.spotify.com", "www.timeanddate.com", "weather.com", "www.accuweather.com", "www.speedtest.net",
    "www.grammarly.com", "www.khanacademy.org", "www.ted.com", "www.udemy.com", "www.goodreads.com",
    "www.rottentomatoes.com", "www.lonelyplanet.com", "www.nintendo.com", "www.playstation.com",
    "www.epicgames.com", "www.beanfun.com", "www.uber.com", "www.starbucks.com", "www.target.com",
    "www.etsy.com", "www.wayfair.com", "www.newegg.com", "www.bhphotovideo.com", "www.logitech.com",
    "www.garmin.com", "www.nikon.com", "www.fujifilm.com", "www.yamaha.com", "www.toyota.com",
    "www.giant-bicycles.com", "www.trendmicro.com", "www.kaspersky.com", "www.malwarebytes.com",
    "www.virustotal.com", "www.namecheap.com", "www.hubspot.com", "www.surveymonkey.com", "www.eventbrite.com",
    "www.meetup.com", "www.scribd.com", "www.slideshare.net", "www.sourceforge.net", "www.speedcurve.com",
    "www.notion.com", "www.trello.com", "www.miro.com", "www.zoom.com", "www.webex.com", "www.calendly.com",
    "www.typeform.com", "www.shopify.com", "www.stripe.com", "www.twilio.com", "www.okta.com", "www.docusign.com",
    "www.quora.com", "www.pinterest.com", "www.tumblr.com", "www.flickr.com", "www.vimeo.com", "www.dailymotion.com",
    "www.archive.org", "www.gutenberg.org", "www.w3.org", "www.ietf.org", "www.icann.org", "www.unicef.org",
    "www.who.int", "www.oecd.org", "www.worldbank.org", "www.imf.org", "www.nature.com", "www.science.org",
    "www.sciencedirect.com", "www.springer.com", "www.jstor.org", "www.researchgate.net", "arxiv.org",
    "www.mit.edu", "www.stanford.edu", "www.harvard.edu", "www.cmu.edu", "www.berkeley.edu", "www.ox.ac.uk",
    "www.ft.com", "www.wsj.com", "www.forbes.com", "www.cnbc.com", "www.marketwatch.com", "www.morningstar.com",
    "www.zillow.com", "www.indeed.com", "www.glassdoor.com", "www.linkedin.cn", "www.gov.cn", "www.tsinghua.edu.cn",
    "www.sina.com.cn", "www.zhihu.com", "www.douban.com", "www.bilibili.com", "www.iqiyi.com", "www.rakuten.co.jp",
    "www.yodobashi.com", "www.muji.net", "www.uniqlo.com", "www.daiso-sangyo.co.jp", "www.coupang.com",
    "www.naver.com", "www.kakaocorp.com", "www.line-cdn.net", "www.grab.com", "www.shopback.com",
    "www.agoda.net", "www.trivago.com", "www.skyscanner.net", "www.kayak.com", "www.hostelworld.com",
]
HN_RANDOM_LOOKING = [  # 合法但字面像隨機字串的品牌（特殊拼字／縮寫；v7.0 newly_registered_like 的已知誤判）
    "www.kktix.com", "www.kfcclub.com.tw", "www.sfgate.com", "www.knightfrank.com", "www.sprinklr.com",
    "www.zscaler.com", "www.mcdonalds.com", "www.jcpenney.com", "www.bkstr.com", "www.rspca.org.uk",
    "www.hrblock.com", "www.zdnet.com", "www.tcfst.org.tw",
]
HN_ACRONYM = [
    "www.hbo.com", "www.cpc.com.tw", "www.dhl.com.tw", "www.bmw.com.tw", "www.kkbox.com", "www.sgs.com",
    "www.nvidia.com", "www.wwf.org.tw", "www.pwc.tw", "www.kpmg.com", "www.bcg.com", "www.thsrc.com.tw",
    "www.cht.com.tw", "www.fetnet.net", "www.taiwanmobile.com", "www.ftv.com.tw", "www.tsrc.com.tw",
    "www.hkjc.com", "www.mtr.com.hk", "www.ptv.com.tw",
]


def _legit(*paths: str) -> List[Callable[[_Gen], str]]:
    return [lambda g, p=p: p for p in paths]


def gen_hard_negatives(g: _Gen) -> List[Tuple[str, str, str]]:
    """回傳 [(url, category, template)]。"""
    rows: List[Tuple[str, str, str]] = []

    def add(cat: str, tpl: str, url: str) -> None:
        rows.append((url, cat, tpl))

    for host in HN_SUBSTRING:
        add("benign_substring_trap", "hn_substring",
            compose(g, host, benign_path(g, _legit("/learn/python", "/questions/12345/how-to-parse-url",
                                                    "/zh-tw/home.html", "/en/watches", "/tw/zh/", "/games",
                                                    "/news/lottery-results", "/download/"))))
    for host in HN_MEDIA_CONTENT:
        for _ in range(2):
            title = quote(g.choice(ZH_NEWS_TITLES))
            n = g.randint(100000, 9999999)
            path = g.choice([
                f"/news/{n}/{title}", f"/article/{n}", f"/{title}-{n}.html", f"/news/{n}",
                f"/zh-tw/{title}", f"/p/{n}?utm_source=line&utm_medium=share", f"/tag/{title}",
            ])
            add("benign_tw_media", "hn_media_content", compose(g, host, path if g.chance(0.85) else ""))
    for host in HN_MEDIA_OTHER:
        add("benign_tw_media", "hn_media_other", compose(g, host, benign_path(g, _legit("/news/12345", "/article/556677"))))
    finance_paths = _legit("/personal/deposit", "/ebank/login", "/AntiFraud/Index", "/zh-tw/ipo/subscribe",
                           "/futures/quote", "/stock/2330", "/etf/0050", "/tw/zh/personal-banking/",
                           "/credit-card/apply", "/wps/portal/", "/news/anti-fraud", "/zh-tw/securities/trade",
                           "/StockInfo/ShowSaleMonChart.asp?STOCK_ID=2330", "/symbols/TWSE-2330/",
                           "/equities/taiwan-semiconductor", "/fund/overview", "/ch/home.jsp?id=95")
    for host in HN_FINANCE:
        add("benign_finance", "hn_finance", compose(g, host, benign_path(g, finance_paths, specific=0.45)))
    crypto_paths = [
        lambda gg: f"/zh-TC/futures/{gg.choice(['BTCUSDT', 'ETHUSDT', 'SOLUSDT'])}",
        lambda gg: f"/zh-hant/trade/{gg.choice(['btc_usdt', 'eth_usdt'])}",
        lambda gg: gg.choice(["/staking", "/earn", "/airdrop", "/markets", "/swap", "/download", "/wallet",
                              "/zh-TW/support/faq", "/learn/what-is-usdt", "/nft"]),
        lambda gg: f"/register?ref={gg.alnum(8)}",
        lambda gg: f"/token/0x{gg.hexs(40)}",
        lambda gg: f"/assets/{gg.choice(['ethereum', 'bnb', 'tron'])}",
    ]
    for host in HN_CRYPTO:
        for _ in range(2 if g.chance(0.4) else 1):
            add("benign_crypto_exchange", "hn_crypto_official", compose(g, host, benign_path(g, crypto_paths, specific=0.6)))
    for host in HN_CLOUD:
        add("benign_cloud_docs", "hn_cloud_docs",
            compose(g, host, benign_path(g, _legit("/docs/", "/en/stable/", "/tutorial/", "/guide/introduction",
                                                    "/zh-TW/docs/Web/HTTP", "/api/reference", "/learn"))))
    for host in HN_FREE_HOSTING_BENIGN:
        add("benign_cloud_docs", "hn_free_hosting_benign",
            compose(g, host, benign_path(g, _legit("/docs/", "/blog/2025/hello-world", "/projects/", "/about",
                                                    "/__/auth/handler?apiKey=AIza" + g.alnum(20)), specific=0.5),
                    scheme="https://"))
    for _ in range(4):
        add("benign_cloud_docs", "hn_cloud_assets", g.choice([
            f"https://storage.googleapis.com/{g.choice(['tw-report', 'acme-public', 'open-data-2025'])}/report-{g.randint(1, 99)}.pdf",
            f"https://d{g.lower_alnum(13)}.cloudfront.net/static/js/main.{g.hexs(8)}.js",
            f"https://{g.choice(['acme-assets', 'shop-static'])}.s3.amazonaws.com/assets/app.{g.hexs(6)}.css",
            f"https://{g.choice(['acme', 'contoso'])}-api.azurewebsites.net/api/health",
            f"https://sites.google.com/view/{g.choice(['ntu-photo-club', 'cgu-cs-lab', 'tp-hs-music'])}/home",
        ]))
    mobile_paths = _legit("/h5/event/2025-anniversary", "/wap/index.html", "/client", "/home", "/wiki/%E5%8F%B0%E7%81%A3",
                          "/m/index", "/promo/h5/summer")
    for host in HN_MOBILE:
        add("benign_mobile_official", "hn_mobile_official", compose(g, host, benign_path(g, mobile_paths, specific=0.35),
                                                                   scheme="https://"))
    for host in HN_RETAIL:
        for _ in range(2 if g.chance(0.45) else 1):
            path = attach_query(neutral_path(g, allow_root=False), tracking_query(g))
            if host in ("www.nike.com", "www.muji.com", "www.zara.com", "www.gu-global.com", "www.mi.com",
                        "www.asus.com", "www.acer.com", "www.lenovo.com", "www.dell.com"):
                path = "/tw" + path
            add("benign_retail_tracking", "hn_retail_tracking", compose(g, host, path, scheme="https://"))
    gov_paths = [
        lambda gg: f"/News_Content.aspx?n={gg.randint(100, 9999)}&s={gg.randint(100000, 999999)}",
        lambda gg: f"/cp.aspx?n={gg.hexs(16).upper()}",
        lambda gg: "/" + quote(gg.choice(["防詐專區", "防制詐騙", "反詐騙宣導", "最新消息"])),
        lambda gg: f"/News.aspx?n={gg.randint(100, 9999)}&sms={gg.randint(1000, 99999)}",
        lambda gg: "/ch/index",
        lambda gg: f"/p/412-1000-{gg.randint(1000, 99999)}.php",
    ]
    for host in HN_GOV:
        add("benign_gov_edu", "hn_gov_edu", compose(g, host, benign_path(g, gov_paths, specific=0.5, tracking=0.1)))
    for acct in HN_SOCIAL_OFFICIAL:
        add("benign_social", "hn_social_official", f"https://line.me/R/ti/p/@{acct}")
    social_makers: List[Callable[[], str]] = [
        lambda: f"https://page.line.me/{g.lower_alnum(8)}",
        lambda: f"https://t.me/s/{g.choice(['bbcchinese', 'cnalive', 'tginfo', 'durov', 'pythontw'])}",
        lambda: f"https://www.plurk.com/{g.choice(['cwmagazine', 'udnnews', 'taipei_travel'])}",
        lambda: f"https://mastodon.social/@{g.choice(['g0v', 'python', 'nasa'])}",
        lambda: f"https://bsky.app/profile/{g.choice(['cna', 'pts', 'nasa'])}.bsky.social",
        lambda: f"https://www.mobile01.com/topicdetail.php?f={g.randint(100, 900)}&t={g.randint(6000000, 7000000)}",
        lambda: f"https://www.reddit.com/r/{g.choice(['taiwan', 'python', 'investing'])}/",
        lambda: f"https://www.threads.com/@{g.choice(['ettoday', 'udn', 'cwmagazine'])}",
    ]
    for i in range(16):
        add("benign_social", "hn_social_page", social_makers[i % len(social_makers)]())
    for host in HN_NUMERIC:
        add("benign_numeric_domain", "hn_numeric", compose(g, host, benign_path(g)))
    for host in HN_ACRONYM:
        add("benign_acronym_domain", "hn_acronym", compose(g, host, benign_path(g)))
    for host in HN_PLAIN_GTLD:
        add("benign_plain_gtld", "hn_plain_gtld", compose(g, host, benign_path(g)))
    for host in HN_RANDOM_LOOKING:
        for _ in range(2):
            add("benign_acronym_domain", "hn_random_looking", compose(g, host, benign_path(g)))
    return rows


# =============================================================================
# 資料載入與擴增組裝
# =============================================================================

def load_url_cases(path: str = URL_CASES_PATH) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            data = data.get("cases", [])
        return [c for c in data if isinstance(c, dict) and c.get("url")]
    except Exception as exc:  # noqa: BLE001
        print(f"讀取 url_cases.json 失敗，略過驗收清單評估：{exc}")
        return []


def load_dataset(opendata_dir: str = OPENDATA_DIR, max_opendata: Optional[int] = 4000,
                 verbose: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    讀入真實資料（主資料 + feedback + 可選的 165 開放資料），回傳標準欄位 DataFrame：
    url（canonical key）、raw_url、label、source、weight、time、note、registered_domain。
    """
    log = print if verbose else (lambda *a, **k: None)
    log("\n[1] 載入真實資料")
    main_df, main_info = DS.load_main_dataset(CLEAN_DATA_PATH, RAW_DATA_PATH, verbose=verbose)
    fb_df, fb_info = DS.load_feedback(FEEDBACK_PATH, weight=FEEDBACK_WEIGHT, verbose=verbose)
    od_df, od_info = DS.load_165_opendata(opendata_dir, max_rows=max_opendata, verbose=verbose)
    merged, merge_info = DS.merge_sources(main_df, fb_df, od_df)

    if merge_info["feedback_override"]:
        log(f"  feedback 覆蓋主資料標籤 {len(merge_info['feedback_override'])} 筆：{merge_info['feedback_override'][:5]}")
    if merge_info["opendata_conflicts"]:
        log(f"  165 開放資料與主資料（標為正常）衝突 {len(merge_info['opendata_conflicts'])} 筆，保留主資料標籤：")
        for u in merge_info["opendata_conflicts"][:10]:
            log("   -", u)
    counts = merged["label"].value_counts().to_dict()
    log(f"  真實資料合計 {len(merged)} 筆（正常 {counts.get(0, 0)}、詐騙 {counts.get(1, 0)}；"
        f"feedback 新增 {merge_info['feedback_new']}、165 開放資料新增 {merge_info['opendata_new']}）")
    info = {"main": main_info, "feedback": fb_info, "opendata": od_info,
            "merge": {k: (v if not isinstance(v, list) else v[:50]) for k, v in merge_info.items()},
            "real_count": int(len(merged)),
            "real_label_counts": {str(k): int(v) for k, v in counts.items()}}
    return merged, info


SIGNAL_FLAGS = (
    "gambling_keyword", "investment_lure_keyword", "crypto_exchange_lure", "brand_impersonation",
    "free_hosting_platform", "tunnel_or_ephemeral_host", "social_invite_link", "punycode_domain",
    "url_has_at_symbol", "non_standard_port", "path_scam_route", "newly_registered_like",
    "gambling_number_pattern", "brand_typo_like", "mobile_lure_path", "has_scam_word",
    "suspicious_keyword_in_domain", "is_ip_address", "brand_in_sld", "cloud_hosting", "has_shortener",
    "double_http",
)


def has_risk_signal(url: str) -> bool:
    """特徵層級是否有任何風險訊號（TLD 風險、品牌近似或任一風險旗標）。"""
    d = extract_feature_dict(url)
    if d["tld_risk_level"] >= 1 or d["levenshtein_brand_dist"] <= 2:
        return True
    return any(d[k] > 0 for k in SIGNAL_FLAGS)


def _split_canonical(url: str) -> Optional[Tuple[str, str]]:
    m = re.match(r"^(https?)://([^/?#]*)(.*)$", url)
    if not m:
        return None
    return m.group(2), m.group(3)


def real_path_pool(real: pd.DataFrame) -> List[str]:
    """真實正常網址的「路徑＋參數」（不含內嵌網址與任何風險訊號），供兩類樣本的 flip 變體共用。"""
    pool = []
    for url in real.loc[real["label"] == 0, "url"]:
        parts = _split_canonical(url)
        if not parts or parts[1] in ("", "/") or "http" in parts[1].lower() or len(parts[1]) > 300:
            continue
        if has_risk_signal("https://www.example.com" + parts[1]):
            continue
        pool.append(parts[1])
    return pool


# 這些模板的關鍵訊號在路徑（或主機本身是共用平台），不產生「形狀翻轉」雙胞胎
NO_TWIN_TEMPLATES = {
    "social_group_invite", "host_ipfs", "host_object_storage", "host_free_lure_wix", "gam_zh_path",
    "inv_zh_path", "hn_social_official", "hn_social_page", "hn_cloud_assets", "hn_media_content",
}


def shape_twin(g: _Gen, url: str, pool: Sequence[str]) -> Optional[str]:
    """
    模板列的「形狀翻轉」雙胞胎：有路徑／參數 → 裸網域；裸網域 → 真實正常路徑或一般路徑（可帶追蹤參數）。
    兩類標籤一律套用，讓每個擴增網域在「裸網域」與「有路徑」兩種寫法下各占一半權重。
    """
    parts = _split_canonical(canonicalize_url(url))
    if parts is None or not parts[0]:
        return None
    authority, rest = parts
    if rest not in ("", "/"):
        return g.choice(["https://", "", "https://"]) + authority + g.choice(["", "/"])
    path = g.choice(pool) if pool and g.chance(0.5) else neutral_path(g, allow_root=False)
    return "https://" + authority + with_tracking(g, path, TRACKING_RATE)


def build_real_variants(real: pd.DataFrame, g: _Gen) -> pd.DataFrame:
    """
    每筆真實網址加兩個變體（兩類標籤使用同一套產生器）：
      flip：有路徑／參數 → 裸網域；裸網域 → 真實正常網址的路徑（60%）或一般路徑（5% 機率改成 http://）
      ad  ：一般路徑 + 廣告追蹤參數（utm／fbclid／gclid…）
    原始列與兩個變體共用原本的樣本權重（各 1/3），讓每個網域在兩種「形狀」下的權重相等；
    flip 的路徑取自真實正常網址，讓「有路徑」的長度／參數分布在兩類之間一致。
    """
    pool = real_path_pool(real)
    rows = []
    for rec in real.itertuples(index=False):
        parts = _split_canonical(rec.url)
        if parts is None:
            continue
        authority, rest = parts
        if not authority:
            continue
        has_rest = rest not in ("", "/")
        scheme = "http://" if g.chance(VARIANT_HTTP_RATE) else "https://"
        if has_rest:
            flip = (scheme if g.chance(0.5) else "") + authority + ("/" if g.chance(0.5) else "")
        else:
            path = g.choice(pool) if pool and g.chance(0.6) else neutral_path(g, allow_root=False)
            flip = scheme + authority + path
        ad = "https://" + authority + attach_query(neutral_path(g), tracking_query(g))
        for kind, url in (("flip", flip), ("ad", ad)):
            rows.append({"url": url, "label": int(rec.label), "category": f"real_variant_{kind}",
                         "template": f"variant_{kind}", "source": "variant", "parent": rec.url,
                         "group": rec.group})
    return pd.DataFrame(rows)


def build_training_frame(real: pd.DataFrame, url_cases: List[Dict[str, Any]], seed: int = RANDOM_STATE,
                         verbose: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """組合真實資料、真實樣本變體、165 模板擴增與 hard negatives，並做洩漏防護與 group 指派。"""
    if verbose:
        print("\n[2] 資料擴增")
    g = _Gen(seed)

    real = real.copy()
    real["group"] = "rd:" + real["registered_domain"].fillna("").astype(str)
    real.loc[real["group"] == "rd:", "group"] = "url:" + real["url"]
    real["is_synthetic"] = 0
    real["category"] = "real:" + real["source"].str.split(":").str[0]
    real["template"] = "real"
    real["parent"] = real["url"]
    real_keys = set(real["url"])
    rd_to_group = dict(zip(real["registered_domain"], real["group"]))

    variants = build_real_variants(real, g)

    synth_rows = []
    for category, fn in SCAM_GENERATORS:
        for url, tpl in fn(g):
            synth_rows.append({"url": url, "label": 1, "category": category, "template": tpl, "source": "synthetic"})
    for url, category, tpl in gen_hard_negatives(g):
        synth_rows.append({"url": url, "label": 0, "category": category, "template": tpl, "source": "synthetic"})
    # 形狀翻轉雙胞胎（與原列共用權重），消除擴增資料內的「有路徑 = 正常」偏差
    pool = real_path_pool(real)
    twins = []
    for fam, row in enumerate(synth_rows):
        row["fam"] = fam
        if row["template"] in NO_TWIN_TEMPLATES:
            continue
        twin = shape_twin(g, row["url"], pool)
        if twin:
            twins.append({**row, "url": twin, "template": row["template"]})
    synth = pd.DataFrame(synth_rows + twins)

    # ---- 洩漏防護：驗收清單（url_cases.json）不得進入擴增 ----
    case_keys = {DS.dedupe_key(c["url"]) for c in url_cases}
    platform = set(TRUSTED_DOMAINS) | {
        "line.me", "lin.ee", "t.me", "telegram.me", "whatsapp.com", "discord.gg", "discord.com", "ipfs.io",
        "google.com", "bit.ly", "reurl.cc", "tinyurl.com",
    }
    platform |= set(FREE_SUFFIXES) | {"wixsite.com", "weebly.com", "blogspot.com", "godaddysites.com",
                                      "mystrikingly.com", "000webhostapp.com", "amazonaws.com", "r2.dev"}
    case_scam = [c["url"] for c in url_cases if c.get("expect") == "scam"]
    platform |= {"chat.whatsapp.com", "discord.com"}
    case_scam_hosts = {get_hostname(u) for u in case_scam} - {""} - platform
    case_scam_domains = {get_registered_domain(u) for u in case_scam}
    case_scam_domains = {d for d in case_scam_domains if d and d not in platform}
    case_domains = {get_registered_domain(c["url"]) for c in url_cases}

    synth["key"] = synth["url"].map(DS.dedupe_key)
    synth["registered_domain"] = synth["url"].map(get_registered_domain)
    synth["hostname"] = synth["url"].map(get_hostname)
    drop_case = synth["key"].isin(case_keys)
    drop_case_domain = (synth["label"] == 1) & (synth["registered_domain"].isin(case_scam_domains)
                                                | synth["hostname"].isin(case_scam_hosts))
    drop_real = synth["key"].isin(real_keys)
    synth = synth[~(drop_case | drop_case_domain | drop_real)].copy()
    after_guard = len(synth)
    synth = synth.drop_duplicates(subset=["key"], keep="first")
    # 詐騙模板列若「沒有任何風險訊號」（例如 twlicaiplus.com 這種黏在一起的字特徵抓不到），
    # 對模型只是標籤雜訊，會讓「一般 .com 長網域」被學成詐騙 → 剔除
    no_signal = (synth["label"] == 1) & ~synth["url"].map(has_risk_signal)
    no_signal_examples = synth.loc[no_signal, "url"].head(20).tolist()
    synth = synth[~no_signal].copy()
    leak = {
        "dropped_same_url_as_url_cases": int(drop_case.sum()),
        "dropped_scam_domain_in_url_cases": int((drop_case_domain & ~drop_case).sum()),
        "dropped_same_url_as_real": int((drop_real & ~drop_case & ~drop_case_domain).sum()),
        "dropped_duplicates": int(after_guard - int(len(synth) + no_signal.sum())),
        "dropped_scam_without_signal": int(no_signal.sum()),
        "dropped_scam_without_signal_examples": no_signal_examples,
        "hard_negative_domains_overlapping_url_cases": sorted(
            set(synth.loc[synth["label"] == 0, "registered_domain"]) & case_domains),
    }

    # group：與真實資料同網域 → 共用真實 group；詐騙模板 → 同模板同 group；hard negative → registered domain
    def synth_group(rec: pd.Series) -> str:
        rd = rec["registered_domain"]
        if rd in rd_to_group:
            return rd_to_group[rd]
        if rec["label"] == 1:
            return f"tpl:{rec['template']}"
        return f"rd:{rd}" if rd else f"tpl:{rec['template']}"

    synth["group"] = synth.apply(synth_group, axis=1)
    # 雙胞胎必須與原列同一 group（同網域，否則會跨折洩漏）
    synth["group"] = synth.groupby("fam")["group"].transform("first")
    synth["is_synthetic"] = 1
    fam_size = synth.groupby("fam")["url"].transform("size").astype(float)
    synth["weight"] = synth["label"].map(SYNTHETIC_WEIGHT).astype(float) / fam_size
    synth["parent"] = ""

    # 真實樣本與其變體：權重平分（原始 1/3、flip 1/3、ad 1/3）
    variants["key"] = variants["url"].map(DS.dedupe_key)
    variants = variants[~variants["key"].isin(real_keys) & ~variants["key"].isin(case_keys)].copy()
    variants = variants.drop_duplicates(subset=["key"], keep="first")
    fam_size = variants.groupby("parent").size()
    weight_of = dict(zip(real["url"], real["weight"]))
    real["family_size"] = real["url"].map(lambda u: 1 + int(fam_size.get(u, 0)))
    real["weight"] = real["weight"] / real["family_size"]
    variants["weight"] = variants["parent"].map(lambda p: weight_of[p] / (1 + int(fam_size.get(p, 0))))
    variants["is_synthetic"] = 1
    variants["registered_domain"] = variants["url"].map(get_registered_domain)

    real_part = real.assign(key=real["url"])[
        ["url", "key", "label", "weight", "is_synthetic", "group", "category", "template", "source", "parent",
         "registered_domain"]]
    cols = list(real_part.columns)
    frame = pd.concat([real_part, variants[cols], synth[cols]], ignore_index=True)
    frame["label"] = frame["label"].astype(int)
    frame["is_synthetic"] = frame["is_synthetic"].astype(int)
    frame["weight"] = frame["weight"].astype(float)

    stats = summarize_frame(frame)
    stats["leakage_guard"] = leak
    if verbose:
        print_frame_summary(frame, stats)
    return frame.reset_index(drop=True), stats


def summarize_frame(frame: pd.DataFrame) -> Dict[str, Any]:
    by_cat = (frame.groupby(["category", "label"]).size().unstack(fill_value=0)
              .rename(columns={0: "benign", 1: "scam"}))
    weight_by = frame.groupby(["is_synthetic", "label"])["weight"].sum()
    return {
        "rows": int(len(frame)),
        "real_rows": int((frame["is_synthetic"] == 0).sum()),
        "synthetic_rows": int((frame["is_synthetic"] == 1).sum()),
        "variant_rows": int(frame["category"].str.startswith("real_variant").sum()),
        "template_rows": int(((frame["is_synthetic"] == 1) & ~frame["category"].str.startswith("real_variant")).sum()),
        "label_counts": {str(k): int(v) for k, v in frame["label"].value_counts().items()},
        "groups": int(frame["group"].nunique()),
        "by_category": {cat: {k: int(v) for k, v in row.items()} for cat, row in by_cat.iterrows()},
        "by_template": {k: int(v) for k, v in frame[frame["is_synthetic"] == 1]["template"].value_counts().items()},
        "weight_sum": {f"{'synthetic' if s else 'real'}_{'scam' if lab else 'benign'}": round(float(v), 2)
                       for (s, lab), v in weight_by.items()},
        "weight_sum_real_family": round(float(frame[frame["category"].str.startswith(("real", "real_variant"))]["weight"].sum()), 2),
        "weight_sum_templates": round(float(frame[(frame["is_synthetic"] == 1)
                                                  & ~frame["category"].str.startswith("real_variant")]["weight"].sum()), 2),
    }


def print_frame_summary(frame: pd.DataFrame, stats: Dict[str, Any]) -> None:
    print(f"  訓練列數 {stats['rows']}（真實 {stats['real_rows']}、真實變體 {stats['variant_rows']}、"
          f"模板擴增 {stats['template_rows']}）；group 數 {stats['groups']}")
    print("  各類數量（benign / scam）：")
    for cat, row in sorted(stats["by_category"].items()):
        print(f"    {cat:<28} {row.get('benign', 0):>5} / {row.get('scam', 0):>5}")
    print(f"  樣本權重合計：真實＋變體 {stats['weight_sum_real_family']:.1f}、模板擴增 {stats['weight_sum_templates']:.1f}"
          f"（模板擴增占 {stats['weight_sum_templates'] / max(stats['weight_sum_real_family'] + stats['weight_sum_templates'], 1e-9):.1%}）")
    leak = stats.get("leakage_guard", {})
    if leak:
        print(f"  洩漏防護：剔除與 url_cases 同網址 {leak['dropped_same_url_as_url_cases']} 筆、"
              f"使用 url_cases 詐騙網域 {leak['dropped_scam_domain_in_url_cases']} 筆、與真實資料重複 "
              f"{leak['dropped_same_url_as_real']} 筆；hard negative 與 url_cases 共用網域 "
              f"{len(leak['hard_negative_domains_overlapping_url_cases'])} 個"
              f"（{', '.join(leak['hard_negative_domains_overlapping_url_cases'][:12])}）")
        print(f"  剔除無任何風險訊號的詐騙模板列 {leak['dropped_scam_without_signal']} 筆"
              f"（例：{', '.join(leak['dropped_scam_without_signal_examples'][:4])}）")


def build_feature_matrix(urls: Any):
    """
    URL 清單 → len(FEATURE_NAMES) 維（v7.1 為 46）特徵矩陣（float64，順序同 FEATURE_NAMES）。
    相容舊介面：傳入含 URL／Label 欄位的 DataFrame 時回傳 (X, y)。
    """
    if isinstance(urls, pd.DataFrame):
        url_col = "url" if "url" in urls.columns else URL_COL
        label_col = "label" if "label" in urls.columns else LABEL_COL
        X = np.array([extract_features(u) for u in urls[url_col]], dtype=float)
        return X, urls[label_col].astype(int).to_numpy()
    return np.array([extract_features(u) for u in urls], dtype=float)


# =============================================================================
# 模型與交叉驗證
# =============================================================================

def make_estimator(name: str, params: Dict[str, Any]):
    if name == "HistGradientBoosting":
        base = dict(class_weight="balanced", early_stopping=False, random_state=RANDOM_STATE)
        base.update(params)
        return HistGradientBoostingClassifier(**base)
    if name == "GradientBoosting":
        base = dict(random_state=RANDOM_STATE)
        base.update(params)
        return GradientBoostingClassifier(**base)
    if name == "RandomForest":
        base = dict(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE)
        base.update(params)
        return RandomForestClassifier(**base)
    if name == "ExtraTrees":
        base = dict(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE)
        base.update(params)
        return ExtraTreesClassifier(**base)
    if name == "LogisticRegression":
        base = dict(max_iter=5000, class_weight="balanced", random_state=RANDOM_STATE)
        base.update(params)
        return Pipeline([
            ("log1p", FunctionTransformer(np.log1p)),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(**base)),
        ])
    raise ValueError(name)


def get_search_space(quick: bool = False) -> Dict[str, List[Dict[str, Any]]]:
    """每個候選模型 4～6 組小型超參數（--quick 時只取前 2 組）。"""
    space = {
        "HistGradientBoosting": [
            {"learning_rate": 0.05, "max_iter": 300, "max_leaf_nodes": 15, "min_samples_leaf": 20, "l2_regularization": 0.0},
            {"learning_rate": 0.05, "max_iter": 300, "max_leaf_nodes": 31, "min_samples_leaf": 20, "l2_regularization": 1.0},
            {"learning_rate": 0.1, "max_iter": 200, "max_leaf_nodes": 15, "min_samples_leaf": 10, "l2_regularization": 1.0},
            {"learning_rate": 0.1, "max_iter": 200, "max_leaf_nodes": 31, "min_samples_leaf": 30, "l2_regularization": 0.0},
            {"learning_rate": 0.03, "max_iter": 500, "max_leaf_nodes": 15, "min_samples_leaf": 20, "l2_regularization": 0.5, "max_depth": 6},
            {"learning_rate": 0.05, "max_iter": 300, "max_leaf_nodes": 7, "min_samples_leaf": 40, "l2_regularization": 0.0},
        ],
        "GradientBoosting": [
            {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 3, "subsample": 0.8, "max_features": "sqrt"},
            {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 4, "subsample": 0.8, "max_features": "sqrt"},
            {"n_estimators": 400, "learning_rate": 0.05, "max_depth": 3, "subsample": 0.8, "max_features": 0.5},
            {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 5, "subsample": 0.7, "max_features": "sqrt", "min_samples_leaf": 5},
            {"n_estimators": 500, "learning_rate": 0.03, "max_depth": 4, "subsample": 0.8, "max_features": "sqrt", "min_samples_leaf": 3},
        ],
        "RandomForest": [
            {"max_depth": None, "min_samples_leaf": 1, "max_features": "sqrt"},
            {"max_depth": None, "min_samples_leaf": 2, "max_features": "sqrt"},
            {"max_depth": 16, "min_samples_leaf": 1, "max_features": 0.4},
            {"max_depth": 12, "min_samples_leaf": 3, "max_features": "sqrt"},
            {"max_depth": None, "min_samples_leaf": 1, "max_features": 0.3, "class_weight": "balanced_subsample"},
        ],
        "ExtraTrees": [
            {"max_depth": None, "min_samples_leaf": 1, "max_features": "sqrt"},
            {"max_depth": None, "min_samples_leaf": 2, "max_features": 0.5},
            {"max_depth": 16, "min_samples_leaf": 1, "max_features": 0.5},
            {"max_depth": 12, "min_samples_leaf": 2, "max_features": "sqrt"},
        ],
        "LogisticRegression": [
            {"C": 0.1}, {"C": 0.3}, {"C": 1.0}, {"C": 3.0}, {"C": 10.0},
        ],
    }
    if quick:
        space = {k: v[:2] for k, v in space.items()}
    return space


def get_candidate_models() -> Dict[str, Any]:
    """（相容舊介面）回傳五個候選模型（各取搜尋空間的第一組參數）。"""
    return {name: make_estimator(name, grid[0]) for name, grid in get_search_space().items()}


def get_feature_importance(model) -> List[Dict[str, Any]]:
    """（相容舊介面）樹模型的 feature_importances_ 或線性模型 |coef|，由高到低；校準模型取各子模型平均。"""
    if isinstance(model, CalibratedClassifierCV):
        return model_feature_importance(model)["values"]
    final = model.steps[-1][1] if isinstance(model, Pipeline) else model
    if hasattr(final, "feature_importances_"):
        values = np.asarray(final.feature_importances_, dtype=float)
    elif hasattr(final, "coef_"):
        values = np.abs(np.asarray(final.coef_[0], dtype=float))
    else:
        return []
    order = np.argsort(-values)
    return [{"feature": FEATURE_NAMES[i], "importance": float(values[i])} for i in order]


def _fit_weights(estimator, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """GradientBoosting 沒有 class_weight：以 compute_sample_weight('balanced') × 樣本權重補上。"""
    final = estimator.steps[-1][1] if isinstance(estimator, Pipeline) else estimator
    if isinstance(final, GradientBoostingClassifier):
        return w * compute_sample_weight("balanced", y)
    return w


def _fit_kwargs(estimator, w: np.ndarray) -> Dict[str, Any]:
    if isinstance(estimator, Pipeline):
        return {f"{estimator.steps[-1][0]}__sample_weight": w}
    return {"sample_weight": w}


def fit_estimator(estimator, X: np.ndarray, y: np.ndarray, w: np.ndarray):
    estimator.fit(X, y, **_fit_kwargs(estimator, _fit_weights(estimator, y, w)))
    return estimator


def make_group_splits(y: np.ndarray, groups: np.ndarray, is_real: np.ndarray, n_splits: int = N_SPLITS,
                      seed: int = RANDOM_STATE, real_only_test: bool = True) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    StratifiedGroupKFold；回傳 (訓練索引＝訓練組全部列, 測試索引)。
    real_only_test=True 時測試索引只含測試組的真實列（校準與正式評估用）；
    False 時含測試組全部列（另外統計「未見過模板」的擴增樣本表現，只供參考）。
    """
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = []
    for train_idx, test_idx in sgkf.split(np.zeros(len(y)), y, groups):
        splits.append((train_idx, test_idx[is_real[test_idx]] if real_only_test else test_idx))
    return splits


def threshold_metrics(y: np.ndarray, p: np.ndarray, thr: float) -> Dict[str, Any]:
    pred = p >= thr
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    f2 = 5 * precision * recall / (4 * precision + recall) if 4 * precision + recall else 0.0
    return {"threshold": thr, "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "f2": round(f2, 4), "fpr": round(fp / (fp + tn), 4) if fp + tn else 0.0,
            "accuracy": round((tp + tn) / max(len(y), 1), 4), "confusion": [[tn, fp], [fn, tp]]}


def prob_metrics(y: np.ndarray, p: np.ndarray, prior_w: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    機率指標。brier／log_loss 以真實資料原始比例計算；prior_w（目標比例權重）存在時另外回報
    brier_target_prior／log_loss_target_prior（把詐騙比例換算成部署目標比例後的期望值）。
    precision／recall 為實際筆數（未加權）。
    """
    p = np.clip(p, 1e-6, 1 - 1e-6)
    out: Dict[str, Any] = {
        "n": int(len(y)), "positives": int(y.sum()),
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
        "log_loss": round(float(log_loss(y, p, labels=[0, 1])), 4),
        "at_0.40": threshold_metrics(y, p, THRESHOLDS["medium"]),
        "at_0.70": threshold_metrics(y, p, THRESHOLDS["high"]),
    }
    if prior_w is not None:
        out["brier_target_prior"] = round(float(brier_score_loss(y, p, sample_weight=prior_w)), 4)
        out["log_loss_target_prior"] = round(float(log_loss(y, p, sample_weight=prior_w, labels=[0, 1])), 4)
        a = float(prior_w[y == 1][0]) if np.any(y == 1) else 1.0
        b = float(prior_w[y == 0][0]) if np.any(y == 0) else 1.0
        for key in ("at_0.40", "at_0.70"):
            (tn, fp), (fn, tp) = out[key]["confusion"]
            denom = tp * a + fp * b
            out[key]["precision_target_prior"] = round(tp * a / denom, 4) if denom else 0.0
    best = max((threshold_metrics(y, p, t) for t in np.round(np.arange(0.05, 0.96, 0.05), 2)),
               key=lambda m: m["f2"])
    out["best_f2_threshold"] = {"threshold": float(best["threshold"]), "f2": best["f2"],
                                "precision": best["precision"], "recall": best["recall"]}
    return out


def cross_val_oof(name: str, params: Dict[str, Any], X: np.ndarray, y: np.ndarray, w: np.ndarray,
                  splits: List[Tuple[np.ndarray, np.ndarray]], real_mask: np.ndarray):
    """回傳 (oof 機率（測試列；未被任何折測試的列為 NaN）, 每折真實樣本 PR-AUC)。"""
    oof = np.full(len(y), np.nan)
    fold_scores = []
    for train_idx, test_idx in splits:
        est = fit_estimator(make_estimator(name, params), X[train_idx], y[train_idx], w[train_idx])
        p = est.predict_proba(X[test_idx])[:, 1]
        oof[test_idx] = p
        real_test = real_mask[test_idx]
        fold_scores.append(float(average_precision_score(y[test_idx][real_test], p[real_test])))
    return oof, fold_scores


def template_metrics(y: np.ndarray, oof: np.ndarray, template_mask: np.ndarray,
                     real_mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    「整個模板／網域 group 未參與訓練」時，擴增列在 0.40 門檻的表現（只供參考）。
    hard_negative_auc：真實詐騙 vs 未見過網域的 hard negative 的 ROC-AUC——衡量模型能否把真實詐騙
    排在「沒看過的合法網站」之前（低誤判的泛化能力），用於 PR-AUC 相近時的選模。
    """
    yt, pt = y[template_mask], oof[template_mask]
    ok = ~np.isnan(pt)
    yt, pt = yt[ok], pt[ok]
    scam, benign = pt[yt == 1], pt[yt == 0]
    out: Dict[str, Any] = {
        "n": int(len(yt)),
        "scam_recall_0.40": round(float(np.mean(scam >= THRESHOLDS["medium"])), 4) if len(scam) else None,
        "benign_fpr_0.40": round(float(np.mean(benign >= THRESHOLDS["medium"])), 4) if len(benign) else None,
    }
    if real_mask is not None and len(benign):
        real_scam = oof[real_mask & (y == 1)]
        real_scam = real_scam[~np.isnan(real_scam)]
        if len(real_scam):
            yy = np.r_[np.ones(len(real_scam)), np.zeros(len(benign))]
            out["hard_negative_auc"] = round(float(roc_auc_score(yy, np.r_[real_scam, benign])), 4)
    return out


def shape_metrics(y: np.ndarray, oof: np.ndarray, masks: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """
    形狀穩健度：同一批真實網域在「原始寫法」「形狀翻轉」「廣告落地頁」三種寫法下的 OOF PR-AUC。
    變體與原始列同 group，因此同樣未參與該折訓練。三者差距越小，代表越不依賴「有路徑／裸網域」捷徑
    （原始寫法本身帶有標籤相關的形狀，只看它會高估依賴捷徑的模型）。
    """
    out: Dict[str, Any] = {}
    for key, mask in masks.items():
        ok = mask & ~np.isnan(oof)
        if ok.sum() and len(np.unique(y[ok])) == 2:
            out[key] = round(float(average_precision_score(y[ok], oof[ok])), 4)
    return out


def search_models(X, y, w, splits, real_mask, template_mask, quick: bool = False,
                  shape_masks: Optional[Dict[str, np.ndarray]] = None
                  ) -> Tuple[Dict[str, Any], Tuple[str, Dict[str, Any]], np.ndarray]:
    print("\n" + "=" * 70)
    print(f"[3] 模型比較：StratifiedGroupKFold({N_SPLITS})，評估只用測試組真實樣本（OOF）")
    print("=" * 70)
    results: Dict[str, Any] = {}
    oofs: Dict[Tuple[str, int], np.ndarray] = {}
    space = get_search_space(quick)
    for name, grid in space.items():
        results[name] = []
        for i, params in enumerate(grid):
            t0 = time.time()
            oof, folds = cross_val_oof(name, params, X, y, w, splits, real_mask)
            yr, pr = y[real_mask], oof[real_mask]
            m = prob_metrics(yr, pr)
            tm = template_metrics(y, oof, template_mask, real_mask)
            sm = shape_metrics(y, oof, shape_masks or {})
            entry = {"params": params, "oof": m, "fold_pr_auc_mean": round(float(np.mean(folds)), 4),
                     "fold_pr_auc_std": round(float(np.std(folds)), 4), "heldout_templates": tm,
                     "shape_robustness": sm, "seconds": round(time.time() - t0, 1)}
            results[name].append(entry)
            oofs[(name, i)] = oof
            print(f"  {name:<20} #{i} PR-AUC {m['pr_auc']:.4f}（折 {entry['fold_pr_auc_mean']:.4f}±{entry['fold_pr_auc_std']:.4f}）"
                  f" ROC {m['roc_auc']:.4f} R/P@.40 {m['at_0.40']['recall']:.3f}/{m['at_0.40']['precision']:.3f}"
                  f" | flip/ad {sm.get('flip')}/{sm.get('ad')} | 模板 R {tm['scam_recall_0.40']} FPR {tm['benign_fpr_0.40']}"
                  f" HN-AUC {tm.get('hard_negative_auc')}  {entry['seconds']}s")

    # 選模：OOF PR-AUC 最高者的「一個標準誤」內視為不相上下（折間差異 ≈ 0.01，0.003 的差距不具意義），
    # 其中再選 hard_negative_auc 最高者（對沒看過的合法網站誤判最少），最後以 F2@0.40 決勝
    flat = [(name, i, e) for name, entries in results.items() for i, e in enumerate(entries)]
    top = max(flat, key=lambda t: t[2]["oof"]["pr_auc"])
    se = top[2]["fold_pr_auc_std"] / math.sqrt(len(splits))
    pool = [t for t in flat if t[2]["oof"]["pr_auc"] >= top[2]["oof"]["pr_auc"] - se]
    chosen = max(pool, key=lambda t: (t[2]["heldout_templates"].get("hard_negative_auc") or 0.0,
                                      t[2]["oof"]["at_0.40"]["f2"]))
    name, i, entry = chosen
    best_params = space[name][i]
    print(f"\nPR-AUC 最高：{top[0]} #{top[1]}（{top[2]['oof']['pr_auc']:.4f}，1 SE = {se:.4f}）；"
          f"1 SE 內候選 {len(pool)} 個")
    print(f"最佳候選：{name} #{i} {best_params}（OOF PR-AUC {entry['oof']['pr_auc']:.4f}、"
          f"HN-AUC {entry['heldout_templates'].get('hard_negative_auc')}、F2@0.40 {entry['oof']['at_0.40']['f2']:.3f}）")
    summary = {
        name_: max(entries, key=lambda e: e["oof"]["pr_auc"]) for name_, entries in results.items()
    }
    selection = {"rule": "OOF PR-AUC 最高者 1 個標準誤內，取 hard_negative_auc 最高、再取 F2@0.40",
                 "top_pr_auc": {"model": top[0], "index": top[1], "pr_auc": top[2]["oof"]["pr_auc"]},
                 "one_se": round(se, 4),
                 "candidates_within_1se": [{"model": t[0], "index": t[1], "pr_auc": t[2]["oof"]["pr_auc"],
                                            "hard_negative_auc": t[2]["heldout_templates"].get("hard_negative_auc")}
                                           for t in pool],
                 "chosen": {"model": name, "index": i}}
    return {"all": results, "best_per_model": summary, "selection": selection}, (name, best_params), oofs[(name, i)]


# =============================================================================
# 機率校準（巢狀 group-CV 選 sigmoid / isotonic）
# =============================================================================

def prior_weights(y: np.ndarray, is_real: np.ndarray, target_prior: Optional[float]) -> Optional[np.ndarray]:
    """
    校準用樣本權重：把真實資料的詐騙比例（約 47%）換算成部署時的目標比例 target_prior
    （prior probability shift；Saerens et al., 2002）。None 表示不調整。
    """
    if not target_prior:
        return None
    pi = float(np.mean(y[is_real])) if np.any(is_real) else float(np.mean(y))
    pi = min(max(pi, 1e-3), 1 - 1e-3)
    return np.where(y == 1, target_prior / pi, (1 - target_prior) / (1 - pi)).astype(float)


def make_calibrated(name: str, params: Dict[str, Any], method: str,
                    inner_splits: List[Tuple[np.ndarray, np.ndarray]]) -> CalibratedClassifierCV:
    base = make_estimator(name, params)
    if not isinstance(base, Pipeline):
        # 包一層單步 Pipeline：CalibratedClassifierCV 的 sample_weight 只用於校準器（目標比例權重），
        # 訓練權重改由 model__sample_weight 傳給模型本身，兩者互不干擾（全部是標準 sklearn 類別，可安全 pickle）
        base = Pipeline([("model", base)])
    return CalibratedClassifierCV(base, method=method, cv=inner_splits, ensemble=True)


def fit_calibrated(name: str, params: Dict[str, Any], method: str, X, y, w, groups, is_real,
                   target_prior: Optional[float], seed: int = RANDOM_STATE) -> CalibratedClassifierCV:
    """
    以 group 切分訓練 CalibratedClassifierCV（ensemble）：每個子模型用訓練組全部列（含擴增、樣本權重）
    訓練，再以測試組的真實列校準（依 target_prior 加權）。
    """
    inner = make_group_splits(y, groups, is_real, N_SPLITS, seed)
    cal = make_calibrated(name, params, method, inner)
    w_fit = _fit_weights(make_estimator(name, params), y, w)
    w_cal = prior_weights(y, is_real, target_prior)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cal.fit(X, y, sample_weight=w_cal, model__sample_weight=w_fit)
    return cal


def nested_calibration(name, params, X, y, w, groups, is_real, splits, real_mask, template_mask,
                       target_prior: Optional[float], methods: Sequence[str] = ("sigmoid", "isotonic"),
                       verbose: bool = True) -> Dict[str, Any]:
    label = f"目標詐騙比例 {target_prior:.0%}" if target_prior else "資料原始比例"
    print_ = print if verbose else (lambda *a, **k: None)
    print_(f"\n[4] 機率校準：巢狀 group-CV 比較 {' / '.join(methods)}（{label}）")
    out: Dict[str, Any] = {}
    for method in methods:
        t0 = time.time()
        oof = np.full(len(y), np.nan)
        for k, (train_idx, test_idx) in enumerate(splits):
            cal = fit_calibrated(name, params, method, X[train_idx], y[train_idx], w[train_idx],
                                 groups[train_idx], is_real[train_idx], target_prior, seed=RANDOM_STATE + 1 + k)
            oof[test_idx] = cal.predict_proba(X[test_idx])[:, 1]
        yr = y[real_mask]
        m = prob_metrics(yr, oof[real_mask], prior_weights(yr, np.ones(len(yr), bool), target_prior))
        tm = template_metrics(y, oof, template_mask, real_mask)
        out[method] = {"oof": m, "heldout_templates": tm, "oof_proba": oof, "seconds": round(time.time() - t0, 1)}
        weighted = f"（目標比例加權 {m['brier_target_prior']:.4f}）" if "brier_target_prior" in m else ""
        print_(f"  {method:<9} Brier {m['brier']:.4f}{weighted}"
              f"  PR-AUC {m['pr_auc']:.4f}"
              f"  P/R@0.40 {m['at_0.40']['precision']:.3f}/{m['at_0.40']['recall']:.3f}"
              f"  P/R@0.70 {m['at_0.70']['precision']:.3f}/{m['at_0.70']['recall']:.3f}"
              f" | 模板 R {tm['scam_recall_0.40']} FPR {tm['benign_fpr_0.40']}  {out[method]['seconds']}s")
    return out


# =============================================================================
# 校準目標比例選擇（獨立 benign 驗證集；v7.1）
# =============================================================================

def load_benign_validation(path: Optional[str]) -> List[Dict[str, Any]]:
    """讀取獨立 benign 驗證集：{"urls": [{"url", "category"}…]} 或網址字串陣列；不存在時回傳空清單。"""
    if not path or not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        items = data.get("urls", []) if isinstance(data, dict) else data
        out = []
        for item in items:
            if isinstance(item, str):
                out.append({"url": item, "category": ""})
            elif isinstance(item, dict) and item.get("url"):
                out.append({"url": item["url"], "category": item.get("category", "")})
        return out
    except Exception as exc:  # noqa: BLE001
        print(f"讀取 benign 驗證集失敗，略過 prior 選擇：{exc}")
        return []


def _bare_variant(url: str) -> str:
    parts = _split_canonical(canonicalize_url(url))
    return parts[0] if parts and parts[0] else url


def evaluate_benign_validation(model, items: List[Dict[str, Any]], thr: float = THRESHOLDS["medium"]) -> Dict[str, Any]:
    """模型層級（不含硬規則／白名單）在獨立 benign 驗證集上的誤判率；另附「裸網域寫法」的誤判率供參考。"""
    if not items:
        return {"available": False}
    urls = [i["url"] for i in items]
    p = model.predict_proba(build_feature_matrix(urls))[:, 1]
    p_bare = model.predict_proba(build_feature_matrix([_bare_variant(u) for u in urls]))[:, 1]
    fps = sorted(({"url": u, "category": i["category"], "proba": round(float(v), 4)}
                  for u, i, v in zip(urls, items, p) if v >= thr), key=lambda d: -d["proba"])
    return {
        "available": True, "n": len(items), "threshold": thr,
        "fp": int(np.sum(p >= thr)), "fp_rate": round(float(np.mean(p >= thr)), 4),
        "fp_at_0.70": int(np.sum(p >= THRESHOLDS["high"])),
        "fp_rate_bare_domain": round(float(np.mean(p_bare >= thr)), 4),
        "mean_proba": round(float(np.mean(p)), 4), "p95_proba": round(float(np.percentile(p, 95)), 4),
        "false_positives": fps,
    }


def select_target_prior(name: str, params: Dict[str, Any], method: str, X, y, w, groups, is_real, splits,
                        is_template, benign_items: List[Dict[str, Any]],
                        candidates: Sequence[Optional[float]] = PRIOR_CANDIDATES,
                        max_fpr: float = BENIGN_VALIDATION_MAX_FPR) -> Dict[str, Any]:
    """
    依「真實資料 OOF 召回 @0.40 最大，且獨立 benign 驗證集模型層級誤判率 @0.40 ≤ max_fpr」選 target prior。
    每個候選：巢狀 group-CV（同 nested_calibration）求 OOF 指標；全部資料訓練的校準模型求驗證集誤判率。
    """
    print(f"\n[4b] 校準目標比例選擇：候選 {[c if c else '原始' for c in candidates]}；"
          f"獨立 benign 驗證集 {len(benign_items)} 筆，誤判上限 {max_fpr:.0%}")
    yr = y[is_real]
    rows = []
    for prior in candidates:
        t0 = time.time()
        res = nested_calibration(name, params, X, y, w, groups, is_real, splits, is_real, is_template,
                                 prior, methods=(method,), verbose=False)[method]
        model = fit_calibrated(name, params, method, X, y, w, groups, is_real, prior)
        bv = evaluate_benign_validation(model, benign_items)
        m = res["oof"]
        row = {
            "target_prior": prior, "label": f"{prior:.2f}" if prior else f"原始（{np.mean(yr):.2f}）",
            "oof_pr_auc": m["pr_auc"],
            "oof_recall_0.40": m["at_0.40"]["recall"], "oof_precision_0.40": m["at_0.40"]["precision"],
            "oof_fpr_0.40": m["at_0.40"]["fpr"],
            "oof_recall_0.70": m["at_0.70"]["recall"], "oof_precision_0.70": m["at_0.70"]["precision"],
            "benign_validation_fp_rate_0.40": bv.get("fp_rate"), "benign_validation_fp_0.40": bv.get("fp"),
            "benign_validation_fp_rate_bare_domain": bv.get("fp_rate_bare_domain"),
            "benign_validation_fp_0.70": bv.get("fp_at_0.70"),
            "benign_validation_false_positives": bv.get("false_positives", [])[:10],
            "heldout_template_recall_0.40": res["heldout_templates"].get("scam_recall_0.40"),
            "heldout_template_fpr_0.40": res["heldout_templates"].get("benign_fpr_0.40"),
            "seconds": round(time.time() - t0, 1),
        }
        rows.append(row)
        print(f"  prior {row['label']:<10} OOF PR-AUC {row['oof_pr_auc']:.4f}  R/P@.40 {row['oof_recall_0.40']:.3f}/"
              f"{row['oof_precision_0.40']:.3f}  R/P@.70 {row['oof_recall_0.70']:.3f}/{row['oof_precision_0.70']:.3f}"
              f" | 驗證集誤判 @.40 {row['benign_validation_fp_0.40']}/{len(benign_items)}"
              f"（{row['benign_validation_fp_rate_0.40']:.1%}；裸網域 {row['benign_validation_fp_rate_bare_domain']:.1%}）"
              f"  {row['seconds']}s")
    ok = [r for r in rows if r["benign_validation_fp_rate_0.40"] is not None
          and r["benign_validation_fp_rate_0.40"] <= max_fpr]
    if ok:
        chosen = max(ok, key=lambda r: (r["oof_recall_0.40"], r["oof_precision_0.40"]))
        reason = (f"驗證集誤判率 ≤ {max_fpr:.0%} 的候選中 OOF 召回 @0.40 最高"
                  f"（{chosen['oof_recall_0.40']:.3f}；誤判 {chosen['benign_validation_fp_rate_0.40']:.1%}）")
    else:
        chosen = min(rows, key=lambda r: (r["benign_validation_fp_rate_0.40"], -r["oof_recall_0.40"]))
        reason = f"沒有候選符合誤判率 ≤ {max_fpr:.0%}，改取誤判率最低者"
    print(f"  → 選用 target prior {chosen['label']}：{reason}")
    return {"rule": f"真實資料 OOF 召回 @0.40 最大，且獨立 benign 驗證集模型層級誤判 @0.40 ≤ {max_fpr:.0%}"
                    "（同分取 precision 高者）；url_cases 不參與",
            "method": method, "benign_validation_count": len(benign_items),
            "candidates": rows, "chosen": chosen["target_prior"], "chosen_label": chosen["label"], "reason": reason}


# =============================================================================
# 特徵重要度
# =============================================================================

def fold_permutation_importance(name, params, X, y, w, splits, real_mask, n_repeats: int = 5) -> List[Dict[str, Any]]:
    """每折以該折模型在「測試組真實樣本」上算 permutation importance（PR-AUC 下降量），再取平均。"""
    acc = np.zeros(len(FEATURE_NAMES))
    acc_sq = np.zeros(len(FEATURE_NAMES))
    for k, (train_idx, test_idx) in enumerate(splits):
        test_idx = test_idx[real_mask[test_idx]]
        est = fit_estimator(make_estimator(name, params), X[train_idx], y[train_idx], w[train_idx])
        res = permutation_importance(est, X[test_idx], y[test_idx], scoring="average_precision",
                                     n_repeats=n_repeats, random_state=RANDOM_STATE + k, n_jobs=1)
        acc += res.importances_mean
        acc_sq += res.importances_mean ** 2
    mean = acc / len(splits)
    std = np.sqrt(np.maximum(acc_sq / len(splits) - mean ** 2, 0))
    order = np.argsort(-mean)
    return [{"feature": FEATURE_NAMES[i], "importance": round(float(mean[i]), 5),
             "fold_std": round(float(std[i]), 5)} for i in order]


def model_feature_importance(calibrated: CalibratedClassifierCV) -> Dict[str, Any]:
    """最終模型（各校準子模型平均）的 impurity importance 或 LR 係數。"""
    imps, coefs = [], []
    for cc in getattr(calibrated, "calibrated_classifiers_", []):
        est = cc.estimator
        final = est.steps[-1][1] if isinstance(est, Pipeline) else est
        if hasattr(final, "feature_importances_"):
            imps.append(np.asarray(final.feature_importances_, dtype=float))
        elif hasattr(final, "coef_"):
            coefs.append(np.asarray(final.coef_[0], dtype=float))
    if imps:
        mean = np.mean(imps, axis=0)
        order = np.argsort(-mean)
        return {"kind": "impurity", "values": [{"feature": FEATURE_NAMES[i], "importance": round(float(mean[i]), 5)}
                                               for i in order]}
    if coefs:
        mean = np.mean(coefs, axis=0)
        order = np.argsort(-np.abs(mean))
        return {"kind": "logistic_coef(log1p+標準化後)", "values": [
            {"feature": FEATURE_NAMES[i], "importance": round(float(abs(mean[i])), 5), "coef": round(float(mean[i]), 5)}
            for i in order]}
    return {"kind": "unavailable（HistGradientBoosting 無 impurity importance，請看 permutation）", "values": []}


# =============================================================================
# 驗收清單、捷徑檢查、標註檢查
# =============================================================================

def evaluate_url_cases(model, cases: List[Dict[str, Any]], frame: pd.DataFrame,
                       oof_by_key: Dict[str, float]) -> Dict[str, Any]:
    """在 url_cases.json 上只看模型機率（不含硬規則／白名單）列出 FP／FN。"""
    if not cases:
        return {"available": False}
    X = build_feature_matrix([c["url"] for c in cases])
    probs = model.predict_proba(X)[:, 1]
    train_keys_real = set(frame.loc[frame["is_synthetic"] == 0, "key"])
    train_keys_all = set(frame["key"])
    train_domains = set(frame["registered_domain"])
    items, per_cat = [], {}
    for case, p in zip(cases, probs):
        key = DS.dedupe_key(case["url"])
        rd = get_registered_domain(case["url"])
        expect = case.get("expect")
        item = {
            "url": case["url"], "expect": expect, "category": case.get("category", ""),
            "min_score": case.get("min_score"), "model_proba": round(float(p), 4),
            "in_training": "real" if key in train_keys_real else ("synthetic" if key in train_keys_all else ""),
            "domain_in_training": rd in train_domains,
        }
        if key in oof_by_key:
            item["oof_proba"] = round(float(oof_by_key[key]), 4)
        items.append(item)
        cat = per_cat.setdefault(item["category"], {"n": 0, "flagged_0.40": 0, "flagged_0.70": 0, "expect": expect})
        cat["n"] += 1
        cat["flagged_0.40"] += int(p >= THRESHOLDS["medium"])
        cat["flagged_0.70"] += int(p >= THRESHOLDS["high"])
    fp = [i for i in items if i["expect"] == "benign" and i["model_proba"] >= THRESHOLDS["medium"]]
    fn = [i for i in items if i["expect"] == "scam" and i["model_proba"] < THRESHOLDS["medium"]]
    high_miss = [i for i in items if i["expect"] == "scam" and (i.get("min_score") or 0) >= 70
                 and i["model_proba"] < THRESHOLDS["high"]]
    benign = [i for i in items if i["expect"] == "benign"]
    scam = [i for i in items if i["expect"] == "scam"]
    summary = {
        "benign": len(benign), "scam": len(scam),
        "fp_at_0.40": len(fp), "fn_at_0.40": len(fn), "high_conf_miss_at_0.70": len(high_miss),
        "benign_fp_rate_0.40": round(len(fp) / max(len(benign), 1), 4),
        "scam_recall_0.40": round(1 - len(fn) / max(len(scam), 1), 4),
        "scam_recall_0.70": round(sum(i["model_proba"] >= 0.70 for i in scam) / max(len(scam), 1), 4),
        "note": "只看模型機率；main.py 另有硬規則、白名單覆寫與內嵌跳轉評分，最終 risk_score 以 API 為準。"
                "category=dataset 的案例本身就是訓練資料，oof_proba 為該列在交叉驗證中未參與訓練時的機率。",
    }
    return {"available": True, "summary": summary, "per_category": per_cat,
            "false_positives": sorted(fp, key=lambda i: -i["model_proba"]),
            "false_negatives": sorted(fn, key=lambda i: i["model_proba"]),
            "high_confidence_misses": sorted(high_miss, key=lambda i: i["model_proba"])}


# v7.1 特徵設計實驗紀錄（寫進 model_report.feature_experiments；重現腳本與清單見交接說明）
FEATURE_EXPERIMENTS: Dict[str, Any] = {
    "char_scope": {
        "question": "dot_count／digit_ratio／special_chars 的計算範圍",
        "design": "GradientBoosting（v7.0 最佳參數）× 5 個模型種子 × 2 種 group 切分＝10 次；"
                  "OOF PR-AUC 只算真實樣本；漂移＝同一真實網址家族（原始／形狀翻轉／廣告落地頁）OOF 機率最大差",
        "columns": ["pr_auc_original", "pr_auc_flip", "pr_auc_ad", "family_drift_mean", "family_drift_p90"],
        "V0_full_url": {"mean": [0.9376, 0.9256, 0.9200, 0.0908, 0.2444], "std": [0.0016, 0.0024, 0.0032, 0.0014, 0.0043]},
        "V1_host_only": {"mean": [0.9370, 0.9363, 0.9317, 0.0615, 0.1745], "std": [0.0026, 0.0032, 0.0024, 0.0014, 0.0035]},
        "V2_host_path_no_query": {"mean_3_seeds": [0.9310, 0.9271, 0.9204, 0.0885, 0.2447]},
        "V3_dot_digit_host_special_host_path": {"mean": [0.9379, 0.9359, 0.9309, 0.0662, 0.1865],
                                                "std": [0.0028, 0.0022, 0.0026, 0.0012, 0.0051]},
        "probe_spread_max_3_seeds": {"V0": 0.5131, "V1": 0.1323, "V2": 0.3637, "V3": 0.2061},
        "decision": "V1（只算主機名稱）：原始寫法 PR-AUC 差 −0.0006（小於 1 個標準差），形狀翻轉／廣告落地頁 "
                    "PR-AUC 各 +0.011，家族漂移平均 −32%、P90 −29%，探針（同網域 5 種寫法）最大差 0.51 → 0.13",
    },
    "newly_registered_like": {
        "rule_v70": "母音比例低／罕見子音連接比例／高熵（_random_like_sld）",
        "rule_v71": "sld_randomness ≥ 0.70（且不含品牌、有意義字詞、不在白名單）",
        "legit_list_a": {"n": 259, "v70": 12, "v71": 2,
                         "note": "自建真實合法網域（不含 url_cases／data*.csv／feedback／白名單），特徵設計用"},
        "handover_known_fp": {"n": 11, "v70": 11, "v71": 0,
                              "domains": "threads.net、flickr、tumblr、scribd、zdnet、mcdonalds、kktix、kfcclub、"
                                         "hrblock、sprinklr、zscaler"},
        "real_benign_data_clean": {"n": 572, "v70": 7, "v71": 6},
        "real_scam_data_clean": {"n": 497, "v70": 152, "v71": 129},
        "benign_validation_list_b": {"n": 215, "v70": 8, "v71": 2, "note": "獨立驗證集（未參與特徵設計）"},
    },
}

SHORTCUT_PROBES = [
    # (網域, 預期) — 詐騙：真實資料樣本與 165 公告型態；正常：一般官方網站
    ("az6rxm.com", "scam"), ("w1.hhdshhfh.cc", "scam"), ("www.axizak.com", "scam"),
    ("vylsb.com", "scam"), ("hpro-ex.net", "scam"), ("tw-fxmart.top", "scam"),
    ("www.momoshop.com.tw", "benign"), ("www.cw.com.tw", "benign"), ("www.104.com.tw", "benign"),
    ("www.tatung.com.tw", "benign"),
]


def feature_signal_rates(X: np.ndarray, y: np.ndarray, is_real: np.ndarray,
                         benign_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """真實資料（正常／詐騙）與獨立 benign 驗證集上的主要旗標觸發率（v7.1 特徵檢查）。"""
    keys = ["newly_registered_like", "gambling_keyword", "investment_lure_keyword", "crypto_exchange_lure",
            "brand_impersonation", "suspicious_keyword_in_domain"]
    idx = {k: FEATURE_NAMES.index(k) for k in keys + ["sld_randomness"]}

    def summarize(M: np.ndarray) -> Dict[str, Any]:
        if not len(M):
            return {"n": 0}
        out: Dict[str, Any] = {"n": int(len(M))}
        for k in keys:
            out[k] = round(float(np.mean(M[:, idx[k]] >= 1)), 4)
        r = M[:, idx["sld_randomness"]]
        out["sld_randomness_mean"] = round(float(np.mean(r)), 4)
        out["sld_randomness_ge_0.70"] = round(float(np.mean(r >= 0.70)), 4)
        return out

    res = {"real_benign": summarize(X[is_real & (y == 0)]), "real_scam": summarize(X[is_real & (y == 1)])}
    if benign_items:
        res["benign_validation"] = summarize(build_feature_matrix([i["url"] for i in benign_items]))
    return res


def shortcut_check(model) -> Dict[str, Any]:
    """
    檢查「scheme／路徑捷徑」是否消除：同一網域的裸網域、https://…/、http://…/、一般路徑＋廣告參數
    的模型機率差異（neutral_spread）應很小；加上 /h5/#/ 這類詐騙路由後機率不應下降。
    """
    variants = {
        "bare": "{h}",
        "https_root": "https://{h}/",
        "http_root": "http://{h}/",
        "https_path_utm": "https://{h}/zh-tw/index.html?utm_source=facebook&utm_medium=cpc&fbclid=IwAR0abcDEF123",
        "https_h5_hash": "https://{h}/h5/#/",
    }
    rows = []
    for host, expect in SHORTCUT_PROBES:
        urls = {k: v.format(h=host) for k, v in variants.items()}
        probs = model.predict_proba(build_feature_matrix(list(urls.values())))[:, 1]
        pm = {k: round(float(p), 4) for k, p in zip(urls, probs)}
        neutral = [pm["bare"], pm["https_root"], pm["https_path_utm"]]
        rows.append({"host": host, "expect": expect, "proba": pm,
                     "neutral_spread": round(max(neutral) - min(neutral), 4),
                     "h5_minus_bare": round(pm["https_h5_hash"] - pm["bare"], 4)})
    spreads = [r["neutral_spread"] for r in rows]
    return {"rows": rows, "max_neutral_spread": round(max(spreads), 4),
            "mean_neutral_spread": round(float(np.mean(spreads)), 4)}


def label_review(frame: pd.DataFrame, oof: np.ndarray, top: int = 15) -> Dict[str, Any]:
    """真實樣本中 OOF 機率與標籤差距最大的列（疑似標註錯誤，供人工確認）。"""
    real = frame[frame["is_synthetic"] == 0].copy()
    real["oof"] = oof[real.index.to_numpy()]
    benign = real[real["label"] == 0].sort_values("oof", ascending=False).head(top)
    scam = real[real["label"] == 1].sort_values("oof").head(top)
    pick = lambda d: [{"url": u, "oof_proba": round(float(p), 4)} for u, p in zip(d["url"], d["oof"])]  # noqa: E731
    watch = real[real["url"].str.contains("gh168-bank-loan-money", na=False)]
    return {"benign_with_high_proba": pick(benign), "scam_with_low_proba": pick(scam),
            "manual_check": pick(watch)}


# =============================================================================
# 儲存
# =============================================================================

def save_bundle(model, best_name: str, params: Dict[str, Any], calibration: Dict[str, Any],
                metrics: Dict[str, Any], trained_at: str) -> None:
    bundle = {
        "format": BUNDLE_FORMAT,
        "estimator": model,
        "feature_names": list(FEATURE_NAMES),
        "feature_version": FEATURE_VERSION,
        "feature_schema_id": FEATURE_SCHEMA_ID,
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "thresholds": dict(THRESHOLDS),
        "trained_at": trained_at,
        "best_model": best_name,
        "params": params,
        "calibration": calibration,
        "metrics": metrics,
    }
    _atomic_write(MODEL_PATH, lambda tmp: joblib.dump(bundle, tmp))
    print(f"\n模型 bundle 已儲存：{MODEL_PATH}")


def _atomic_write(path: str, writer: Callable[[str], Any]) -> None:
    """先寫同目錄暫存檔再 os.replace（main.py 重新載入時不會讀到寫一半的檔案）。"""
    tmp = os.path.join(os.path.dirname(os.path.abspath(path)),
                       f".tmp-{os.path.basename(path)}-{os.getpid()}")
    try:
        writer(tmp)
        for attempt in range(5):   # Windows 上目標檔可能暫時被防毒或其他行程鎖住
            try:
                os.replace(tmp, path)
                break
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.1 * (attempt + 1))
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _write_json(path: str, obj: Any, **kw: Any) -> None:
    def writer(tmp: str) -> None:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, **kw)
    _atomic_write(path, writer)


def save_metadata(meta: Dict[str, Any]) -> None:
    """儲存訓練環境資訊，供 main.py 啟動時 verify_environment() 使用。"""
    _write_json(META_PATH, meta)
    print(f"model_meta.json 已儲存：{META_PATH}")


def save_report(report: Dict[str, Any]) -> None:
    """儲存完整評估報告（可直接放進專題報告）。"""
    _write_json(REPORT_PATH, report, default=_json_default)
    print(f"model_report.json 已儲存：{REPORT_PATH}")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if math.isnan(float(obj)) else float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def verify_environment() -> bool:
    """
    給 main.py 啟動時呼叫，比對訓練時與目前環境：
      - feature_names 必須與 features.FEATURE_NAMES 完全相同（名稱與順序）
      - feature_schema_id 必須與 features.FEATURE_SCHEMA_ID 相同
      - sklearn 版本不同時視為不一致（pickle 可能無法正確載入）
    全部一致才回傳 True。
    """
    if not os.path.exists(META_PATH):
        print("找不到 model_meta.json，無法驗證環境")
        return False

    try:
        with open(META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)

        ok = True

        trained_sklearn = meta.get("sklearn_version")
        if trained_sklearn != sklearn.__version__:
            print(
                f"警告：sklearn 版本不符！"
                f"訓練版本={trained_sklearn}，目前版本={sklearn.__version__}。"
                f"請修改 requirements.txt：scikit-learn=={trained_sklearn}"
            )
            ok = False

        names = meta.get("feature_names")
        if names != list(FEATURE_NAMES):
            n_meta = len(names) if isinstance(names, list) else meta.get("feature_count")
            print(
                f"警告：特徵名稱或順序不符！"
                f"訓練時={n_meta} 個，目前={len(FEATURE_NAMES)} 個。"
                f"請重新執行 train_model.py。"
            )
            ok = False

        schema = meta.get("feature_schema_id")
        if schema != FEATURE_SCHEMA_ID:
            print(f"警告：feature_schema_id 不符（訓練時={schema}，目前={FEATURE_SCHEMA_ID}），請重新執行 train_model.py。")
            ok = False

        if ok:
            print(f"環境驗證通過（sklearn {sklearn.__version__}，{len(FEATURE_NAMES)} 個特徵，schema {FEATURE_SCHEMA_ID}）")

        return ok

    except Exception as e:
        print(f"環境驗證失敗：{e}")
        return False


# =============================================================================
# 主程式
# =============================================================================

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="TruthMark v7 模型訓練")
    parser.add_argument("--quick", action="store_true", help="每個模型只試 2 組超參數（開發用）")
    parser.add_argument("--opendata-dir", default=OPENDATA_DIR, help="165 開放資料資料夾（預設 data/）")
    parser.add_argument("--max-opendata", type=int, default=4000, help="165 開放資料最多取幾筆")
    parser.add_argument("--no-save", action="store_true", help="只評估、不寫入模型與報告（開發用）")
    parser.add_argument("--target-prior", type=float, default=TARGET_PRIOR,
                        help=f"校準目標詐騙比例（預設 {TARGET_PRIOR}；0 表示沿用資料原始比例）；"
                             "提供 --benign-validation 時改由驗證集選擇")
    parser.add_argument("--benign-validation", default=os.environ.get(BENIGN_VALIDATION_ENV, ""),
                        help=f"獨立 benign 驗證集 JSON（選 target prior 用；預設讀環境變數 {BENIGN_VALIDATION_ENV}）")
    parser.add_argument("--no-prior-select", action="store_true", help="不做 target prior 選擇（沿用 --target-prior）")
    args = parser.parse_args(argv)

    try:
        sys.stdout.reconfigure(errors="replace")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

    t_start = time.time()
    print("=" * 70)
    print("Truth AI 模型訓練 v7.1")
    print(f"scikit-learn {sklearn.__version__} / numpy {np.__version__} / Python {platform.python_version()}")
    print(f"特徵數量：{len(FEATURE_NAMES)}（feature_version {FEATURE_VERSION}、schema {FEATURE_SCHEMA_ID}）")
    print("=" * 70)

    url_cases = load_url_cases()
    real, data_info = load_dataset(args.opendata_dir, args.max_opendata)
    frame, frame_stats = build_training_frame(real, url_cases)

    X = build_feature_matrix(frame["url"].tolist())
    y = frame["label"].to_numpy(dtype=int)
    w = frame["weight"].to_numpy(dtype=float)
    groups = frame["group"].to_numpy()
    is_real = (frame["is_synthetic"] == 0).to_numpy()
    print(f"\n特徵矩陣：{X.shape[0]} 筆 × {X.shape[1]} 維")
    if X.shape[1] != len(FEATURE_NAMES):
        raise RuntimeError("特徵維度與 FEATURE_NAMES 不一致")

    is_template = ((frame["is_synthetic"] == 1) & ~frame["category"].str.startswith("real")).to_numpy()
    splits = make_group_splits(y, groups, is_real, real_only_test=False)
    fold_info = [{"train_rows": int(len(tr)), "test_rows": int(len(te)),
                  "test_real_rows": int(is_real[te].sum()),
                  "test_real_scam": int(y[te][is_real[te]].sum())} for tr, te in splits]

    shape_masks = {"original": is_real,
                   "flip": (frame["category"] == "real_variant_flip").to_numpy(),
                   "ad": (frame["category"] == "real_variant_ad").to_numpy()}
    search, (best_name, best_params), oof_raw = search_models(X, y, w, splits, is_real, is_template,
                                                              quick=args.quick, shape_masks=shape_masks)
    default_prior = args.target_prior if args.target_prior and 0 < args.target_prior < 1 else None
    yr = y[is_real]

    # 校準方法（sigmoid／isotonic）以預設目標比例的巢狀 group-CV Brier 選擇
    calib = nested_calibration(best_name, best_params, X, y, w, groups, is_real, splits, is_real, is_template,
                               default_prior)
    method_key = "brier_target_prior" if default_prior else "brier"
    method = min(("sigmoid", "isotonic"), key=lambda m: (calib[m]["oof"][method_key], calib[m]["oof"]["log_loss"]))
    print(f"  選用 {method}（以目標比例 {default_prior if default_prior else '原始'} 的 Brier 比較）")

    # 目標比例：以獨立 benign 驗證集選擇（url_cases 不參與）
    benign_items = load_benign_validation(args.benign_validation)
    prior_selection: Optional[Dict[str, Any]] = None
    target_prior = default_prior
    if benign_items and not args.no_prior_select:
        prior_selection = select_target_prior(best_name, best_params, method, X, y, w, groups, is_real, splits,
                                              is_template, benign_items)
        target_prior = prior_selection["chosen"]
    else:
        print(f"\n[4b] 未提供獨立 benign 驗證集（--benign-validation 或環境變數 {BENIGN_VALIDATION_ENV}），"
              f"沿用 target prior {default_prior}")
    chosen_calib = (calib[method] if target_prior == default_prior else
                    nested_calibration(best_name, best_params, X, y, w, groups, is_real, splits, is_real,
                                       is_template, target_prior, methods=(method,))[method])
    real_prior_w = prior_weights(yr, np.ones(len(yr), bool), target_prior)
    raw_metrics = prob_metrics(yr, oof_raw[is_real], real_prior_w)
    brier_key = "brier_target_prior" if target_prior else "brier"
    cal_metrics = chosen_calib["oof"]
    print(f"  最終：{method}、target prior {target_prior if target_prior else '原始'}；Brier"
          f"（{'目標比例加權' if target_prior else '原始比例'}）{raw_metrics[brier_key]:.4f}（未校準）→ "
          f"{cal_metrics[brier_key]:.4f}")
    calib_dataset_prior = None
    if target_prior:
        calib_dataset_prior = nested_calibration(best_name, best_params, X, y, w, groups, is_real, splits,
                                                 is_real, is_template, None, methods=(method,))[method]

    print("\n[5] 特徵重要度（各折 permutation importance，PR-AUC 下降量）")
    perm = fold_permutation_importance(best_name, best_params, X, y, w, splits, is_real)
    for item in perm[:15]:
        print(f"  {item['feature']:<30} {item['importance']:+.4f} ± {item['fold_std']:.4f}")

    print("\n[6] 最終模型：全部資料 + group 切分校準（CalibratedClassifierCV ensemble）")
    final_model = fit_calibrated(best_name, best_params, method, X, y, w, groups, is_real, target_prior)
    trained_at = datetime.now().isoformat(timespec="seconds")

    real_idx = np.where(is_real)[0]
    oof_cal = chosen_calib["oof_proba"]
    oof_by_key = {frame.at[i, "key"]: float(oof_cal[i]) for i in real_idx}
    cases_eval = evaluate_url_cases(final_model, url_cases, frame, oof_by_key)
    shortcut = shortcut_check(final_model)
    review = label_review(frame, oof_cal)
    benign_eval = evaluate_benign_validation(final_model, benign_items)
    if benign_eval.get("available"):
        print(f"\n[6b] 獨立 benign 驗證集（模型層級）：誤判 @0.40 {benign_eval['fp']}/{benign_eval['n']}"
              f"（{benign_eval['fp_rate']:.1%}；裸網域寫法 {benign_eval['fp_rate_bare_domain']:.1%}）")
        for item in benign_eval["false_positives"][:10]:
            print(f"   FP {item['proba']:.3f} {item['category']:<20} {item['url'][:80]}")

    if cases_eval.get("available"):
        s = cases_eval["summary"]
        print(f"\n[7] url_cases.json（模型層級）：benign {s['benign']}、scam {s['scam']}；"
              f"FP@0.40 {s['fp_at_0.40']}、FN@0.40 {s['fn_at_0.40']}、"
              f"recall@0.40 {s['scam_recall_0.40']:.3f}、recall@0.70 {s['scam_recall_0.70']:.3f}")
        for item in cases_eval["false_positives"][:10]:
            print(f"   FP {item['model_proba']:.3f} {item['category']:<24} {item['url'][:80]}")
        for item in cases_eval["false_negatives"][:15]:
            print(f"   FN {item['model_proba']:.3f} {item['category']:<24} {item['url'][:80]}")
    print(f"\n[8] 捷徑檢查：同網域「裸網域 / https / 一般路徑+utm」機率最大差 {shortcut['max_neutral_spread']:.4f}"
          f"（平均 {shortcut['mean_neutral_spread']:.4f}）")
    for r in shortcut["rows"]:
        pm = r["proba"]
        print(f"   {r['expect']:<6} {r['host']:<22} bare {pm['bare']:.3f}  https {pm['https_root']:.3f}  "
              f"http {pm['http_root']:.3f}  path+utm {pm['https_path_utm']:.3f}  /h5/#/ {pm['https_h5_hash']:.3f}")

    metrics = {
        "evaluation": "OOF（StratifiedGroupKFold 5 折，只算測試組真實樣本；校準為巢狀 group-CV）",
        "pr_auc": cal_metrics["pr_auc"], "roc_auc": cal_metrics["roc_auc"],
        "brier": cal_metrics["brier"], "brier_uncalibrated": raw_metrics["brier"],
        "log_loss": cal_metrics["log_loss"],
        "at_0.40": cal_metrics["at_0.40"], "at_0.70": cal_metrics["at_0.70"],
        "best_f2_threshold": cal_metrics["best_f2_threshold"],
        "real_samples": int(is_real.sum()),
        "real_scam_prior": round(float(np.mean(yr)), 4),
        "target_prior": target_prior,
        "shape_robustness_pr_auc": shape_metrics(y, chosen_calib["oof_proba"], shape_masks),
        "heldout_templates": chosen_calib["heldout_templates"],
    }
    if benign_eval.get("available"):
        metrics["benign_validation"] = {k: benign_eval[k] for k in
                                        ("n", "fp", "fp_rate", "fp_at_0.70", "fp_rate_bare_domain", "mean_proba")}
    if target_prior:
        metrics["brier_target_prior"] = cal_metrics["brier_target_prior"]
        metrics["brier_uncalibrated_target_prior"] = raw_metrics["brier_target_prior"]
    if calib_dataset_prior is not None:
        metrics["dataset_prior_calibration"] = {
            k: calib_dataset_prior["oof"][k] for k in ("brier", "log_loss", "at_0.40", "at_0.70")}
    calibration_info = {"method": method, "ensemble": True, "n_splits": N_SPLITS,
                        "target_prior": target_prior,
                        "real_scam_prior": round(float(np.mean(yr)), 4),
                        "calibration_data": "每個子模型以未參與訓練之 group 的真實樣本校準"
                                            + ("（依目標詐騙比例加權，prior shift）" if target_prior else ""),
                        "brier_uncalibrated": raw_metrics[brier_key],
                        "brier": cal_metrics[brier_key],
                        "method_selection": {"target_prior": default_prior, "metric": method_key,
                                             "sigmoid": calib["sigmoid"]["oof"][method_key],
                                             "isotonic": calib["isotonic"]["oof"][method_key]},
                        "brier_metric": brier_key}
    if prior_selection is not None:
        calibration_info["prior_selection"] = {
            k: prior_selection[k] for k in ("rule", "benign_validation_count", "chosen", "chosen_label", "reason")}
        calibration_info["prior_selection"]["table"] = [
            {k: v for k, v in row.items() if k != "benign_validation_false_positives"}
            for row in prior_selection["candidates"]]
    label_counts = {str(k): int(v) for k, v in frame["label"].value_counts().items()}
    meta = {
        "project": "Truth",
        "model_file": "scam_model.pkl",
        "format": BUNDLE_FORMAT,
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "feature_count": len(FEATURE_NAMES),
        "feature_names": list(FEATURE_NAMES),
        "feature_version": FEATURE_VERSION,
        "feature_schema_id": FEATURE_SCHEMA_ID,
        "best_model": best_name,
        "params": best_params,
        "calibration": calibration_info,
        "thresholds": dict(THRESHOLDS),
        "metrics": metrics,
        "data_count": int(len(frame)),
        "label_counts": label_counts,
        "real_count": int(is_real.sum()),
        "real_label_counts": {str(k): int(v) for k, v in frame.loc[is_real, "label"].value_counts().items()},
        "synthetic_count": int((~is_real).sum()),
        "synthetic_breakdown": {"real_variants": frame_stats["variant_rows"], "templates": frame_stats["template_rows"]},
        "trained_at": trained_at,
    }
    report = {
        "generated_at": trained_at,
        "versions": {"feature_version": FEATURE_VERSION, "feature_schema_id": FEATURE_SCHEMA_ID,
                     "sklearn": sklearn.__version__, "numpy": np.__version__},
        "data": {"sources": data_info, "training_frame": frame_stats},
        "cv_design": {
            "splitter": f"StratifiedGroupKFold(n_splits={N_SPLITS}, shuffle=True, random_state={RANDOM_STATE})",
            "group": "registered domain；同模板擴增列共用同一 group；擴增列與真實資料同網域時共用該網域 group",
            "train": "訓練組全部列（真實＋變體＋模板擴增，含樣本權重）",
            "evaluate": "只算測試組的真實（is_synthetic=0）樣本",
            "folds": fold_info,
            "selection": "OOF PR-AUC 最高者 1 個標準誤內，取 hard_negative_auc（真實詐騙 vs 未見過網域的合法網站）"
                         "最高者，再以 F2@0.40 決勝",
            "heldout_templates": "擴增列以其 group（模板或網域）整組留出時的表現，只供參考",
        },
        "cross_validation": search,
        "best": {"model": best_name, "params": best_params},
        "oof_uncalibrated": raw_metrics,
        "calibration": {"chosen": method, "target_prior": target_prior,
                        "method_selection_prior": default_prior,
                        "sigmoid": {**calib["sigmoid"]["oof"], "heldout_templates": calib["sigmoid"]["heldout_templates"]},
                        "isotonic": {**calib["isotonic"]["oof"], "heldout_templates": calib["isotonic"]["heldout_templates"]},
                        "chosen_with_dataset_prior": (None if calib_dataset_prior is None else {
                            **calib_dataset_prior["oof"], "heldout_templates": calib_dataset_prior["heldout_templates"]})},
        "prior_selection": prior_selection if prior_selection is not None else {
            "skipped": True, "target_prior": target_prior,
            "note": f"未提供獨立 benign 驗證集，沿用 TARGET_PRIOR；請以 --benign-validation 或 {BENIGN_VALIDATION_ENV} 重新選擇"},
        "benign_validation": benign_eval,
        "feature_experiments": FEATURE_EXPERIMENTS,
        "feature_signal_rates": feature_signal_rates(X, y, is_real, benign_items),
        "oof_metrics": metrics,
        "permutation_importance_top15": perm[:15],
        "permutation_importance_all": perm,
        "model_feature_importance": model_feature_importance(final_model),
        "url_cases": cases_eval,
        "shortcut_check": shortcut,
        "label_review": review,
        "elapsed_seconds": round(time.time() - t_start, 1),
    }
    dp = metrics.get("dataset_prior_calibration")
    report["notes"] = [
        f"校準採 prior shift：真實資料詐騙比例 {metrics['real_scam_prior']:.0%}，以目標比例 "
        f"{(target_prior or metrics['real_scam_prior']):.0%} 加權校準。"
        + (f"同一模型若沿用資料原始比例，@0.40 precision／recall = {dp['at_0.40']['precision']:.3f}／"
           f"{dp['at_0.40']['recall']:.3f}（見 oof_metrics.dataset_prior_calibration）。" if dp else ""),
        "真實樣本的「原始寫法」本身帶有標籤相關的形狀（詐騙 96% 為裸網域）；shape_robustness_pr_auc 列出"
        "同一批網域在原始／形狀翻轉／廣告落地頁三種寫法下的 OOF PR-AUC，三者接近代表沒有依賴路徑捷徑。",
        "heldout_templates 為擴增列以整個模板／網域 group 留出時的表現，只供參考，不代表真實分布。",
        "url_cases 為驗收清單，未用於訓練；category=dataset 的案例本身就是訓練資料，請看 oof_proba。",
        "label_review 列出 OOF 機率與標籤差距最大的真實樣本（疑似標註錯誤），請人工確認後修正 data.csv。",
    ]
    if prior_selection is not None:
        report["notes"].insert(1, f"target prior 以獨立 benign 驗證集（{len(benign_items)} 筆）選擇："
                                  f"{prior_selection['chosen_label']}；{prior_selection['reason']}。取捨表見 prior_selection。")
    # 舊欄位相容（舊版報告頁面讀 holdout_report / top_features）
    report["top_features"] = perm[:15]
    report["holdout_report"] = {"note": "v7 改用 group-CV OOF 評估，請看 oof_metrics", **{
        k: metrics[k] for k in ("pr_auc", "roc_auc", "brier")}}

    if args.no_save:
        print("\n--no-save：不寫入模型與報告")
    else:
        save_bundle(final_model, best_name, best_params, calibration_info, metrics, trained_at)
        save_metadata(meta)
        save_report(report)

    print("\n" + "=" * 70)
    print(f"訓練完成（{time.time() - t_start:.0f} 秒）：{best_name} + {method} 校準")
    print(f"OOF PR-AUC {metrics['pr_auc']:.4f}、ROC-AUC {metrics['roc_auc']:.4f}、Brier {metrics['brier']:.4f}"
          + (f"（目標比例 {target_prior:.0%} 加權 {metrics['brier_target_prior']:.4f}）" if target_prior else ""))
    print(f"@0.40 precision {metrics['at_0.40']['precision']:.3f} recall {metrics['at_0.40']['recall']:.3f} "
          f"F2 {metrics['at_0.40']['f2']:.3f}；@0.70 precision {metrics['at_0.70']['precision']:.3f} "
          f"recall {metrics['at_0.70']['recall']:.3f}")
    print(f"部署提醒：請確認 requirements.txt 中 scikit-learn=={sklearn.__version__}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
