# =============================================================================
# tools/build_ngram_table.py — 產生 sld_randomness 用的字元 n-gram 統計表（v7.1）
# =============================================================================
# 用途：features.py 的第 46 個特徵 sld_randomness 以「字元 trigram 平均驚奇度」衡量網域名稱
#       （SLD）像不像可讀的英文／漢語拼音。本腳本以「本機離線語料」重新產生統計表：
#
#   (a) Python 標準庫原始碼（sysconfig.get_paths()["stdlib"]，排除 test／site-packages）的
#       註解與 docstring，抽出英文字母詞（排除緊鄰 . _ ( 的程式識別字、camelCase、無母音字、只出現 1 次的字）
#   (b) 以「聲母 × 韻母」依拼音拼寫規則程式化產生的漢語拼音音節，以及固定亂數種子組合的雙音節詞；
#       另以「子音 × 母音」產生日文羅馬拼音音節與 2～3 音節組合（台灣常見日系品牌，權重 10%）
#   (c) rules_config 的品牌 token 與官方網域 SLD（只當少量補充，權重占比約 4%）
#
#   刻意不使用 data.csv／data_clean.csv／feedback.csv／tests/fixtures/url_cases.json 或任何
#   訓練、驗收網址（避免特徵把標籤資訊帶進模型）。
#
# 模型：字母 a～z + 起始「^」+ 結束「$」的 trigram，Witten-Bell 插值平滑（trigram → bigram →
#       unigram → 均勻分布）。輸出：
#       bi  ：完整 bigram 條件機率 log2 P(c|b)（含平滑；28×27 格）
#       tri ：加權次數 ≥ MIN_TRIGRAM_COUNT 的 trigram 之 log2 P(c|ab)（已含插值）
#       bo  ：trigram context 的 backoff 權重 log2(1-λ(ab))（查無 trigram 時 = bo[ab] + bi[bc]）
#       ref ：語料本身的平均驚奇度統計（供 features.py 對照用，非必要）
#
# 執行：python tools/build_ngram_table.py            （寫入 va/char_ngram_table.json）
#       python tools/build_ngram_table.py --check    （只產生並印出統計，不寫檔）
# 同一版 Python 標準庫重跑結果完全相同（固定排序與亂數種子）。
# =============================================================================

from __future__ import annotations

import argparse
import ast
import io
import json
import math
import os
import random
import re
import sys
import sysconfig
import tokenize
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Tuple

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
VA_DIR = os.path.dirname(TOOLS_DIR)
OUTPUT_PATH = os.path.join(VA_DIR, "char_ngram_table.json")

TABLE_FORMAT = "truthmark-char-ngram/1"
ALPHABET = "abcdefghijklmnopqrstuvwxyz"
START, END = "^", "$"
VIRTUAL_TOTAL = 1e6              # 全部語料字元的加權總量換算成 100 萬個「虛擬字元」
MIN_TRIGRAM_COUNT = 20.0         # 虛擬次數低於此值（相對頻率 < 2e-5）的 trigram 不輸出（改走 backoff）
WEIGHT_SHARE = {"english": 0.62, "pinyin": 0.24, "romaji": 0.10, "brand": 0.04}
PINYIN_PAIRS = 6000              # 雙音節組合數（固定種子）
ROMAJI_WORDS = 3000              # 日文羅馬拼音 2～3 音節組合數（固定種子）
MIN_WORD_COUNT = 2               # 英文字至少出現次數（只出現 1 次的多半是識別字或拼字錯誤）
SEED = 20261003

_SKIP_DIRS = {"site-packages", "test", "tests", "idle_test", "__pycache__", "lib2to3", "turtledemo"}
# 只取「獨立的英文字」：前後不得緊鄰英數、底線、點號或左括號（排除 os.path、fd_open、getattr( 等識別字）
_WORD_RE = re.compile(r"(?<![A-Za-z0-9_.])([A-Za-z]{2,18})(?![A-Za-z0-9_(])")
_HAS_VOWEL_RE = re.compile(r"[aeiouy]")


# =============================================================================
# (a) 英文：標準庫註解與 docstring
# =============================================================================

def _iter_stdlib_files(root: str) -> Iterable[str]:
    for base, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS)
        for name in sorted(files):
            if name.endswith(".py"):
                yield os.path.join(base, name)


def _docstrings(source: str) -> List[str]:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return []
    out = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc:
                out.append(doc)
    return out


def _comments(source: str) -> List[str]:
    out = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type == tokenize.COMMENT:
                out.append(tok.string.lstrip("#"))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    return out


def _is_plain_word(word: str) -> bool:
    """全小寫或首字大寫（排除 camelCase／全大寫縮寫）。"""
    return word.islower() or (word[0].isupper() and word[1:].islower())


def english_counts(stdlib_dir: str) -> Counter:
    counts: Counter = Counter()
    for path in _iter_stdlib_files(stdlib_dir):
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                source = fh.read()
        except OSError:
            continue
        for text in _docstrings(source) + _comments(source):
            for word in _WORD_RE.findall(text):
                if _is_plain_word(word):
                    counts[word.lower()] += 1
    # 只出現 1～2 次的多半是識別字、拼字錯誤或專有名詞；無母音的多半是縮寫（fd、pwd、cwd）
    return Counter({w: c for w, c in counts.items() if c >= MIN_WORD_COUNT and _HAS_VOWEL_RE.search(w)})


# =============================================================================
# (b) 漢語拼音：聲母 × 韻母
# =============================================================================

_INITIALS = ("b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "j", "q", "x",
             "zh", "ch", "sh", "r", "z", "c", "s")
_OPEN_FINALS = ("a", "o", "e", "ai", "ei", "ao", "ou", "an", "en", "ang", "eng", "ong")
_I_FINALS = ("i", "ia", "ie", "iao", "iu", "ian", "in", "iang", "ing", "iong")
_U_FINALS = ("u", "ua", "uo", "uai", "ui", "uan", "un", "uang")
_V_FINALS_JQX = ("u", "ue", "uan", "un")          # ü 在 j／q／x 後寫成 u
_LABIALS = {"b", "p", "m", "f"}
_RETROFLEX_DENTAL = {"zh", "ch", "sh", "r", "z", "c", "s"}


def pinyin_syllables() -> List[str]:
    """依拼音拼寫規則產生音節（近似；少量多產生的組合不影響統計）。"""
    out = set()
    for ini in _INITIALS:
        if ini in ("j", "q", "x"):
            finals = _I_FINALS + _V_FINALS_JQX
        else:
            finals = tuple(f for f in _OPEN_FINALS
                           if not (f == "o" and ini not in _LABIALS)
                           and not (f == "ong" and ini in _LABIALS))
            if ini in _LABIALS:
                finals += ("u",) + (() if ini == "f" else _I_FINALS[:-1])
            elif ini in _RETROFLEX_DENTAL:
                finals += ("i",) + _U_FINALS
            elif ini in ("d", "t", "n", "l"):
                finals += _I_FINALS[:-1] + _U_FINALS
                if ini in ("n", "l"):
                    finals += ("v", "ve", "ue")
            else:  # g k h
                finals += _U_FINALS
        out.update(ini + f for f in finals)
    # 零聲母（y／w 開頭的寫法）
    out.update(("a", "o", "e", "ai", "ei", "ao", "ou", "an", "en", "ang", "eng", "er"))
    out.update("y" + f for f in ("a", "e", "ao", "ou", "an", "in", "ang", "ing", "ong", "i", "u",
                                  "ue", "uan", "un"))
    out.update("w" + f for f in ("a", "o", "ai", "ei", "an", "en", "ang", "eng", "u"))
    return sorted(out)


def pinyin_counts() -> Counter:
    syllables = pinyin_syllables()
    rng = random.Random(SEED)
    counts: Counter = Counter({s: 3.0 for s in syllables})
    for _ in range(PINYIN_PAIRS):
        counts[rng.choice(syllables) + rng.choice(syllables)] += 1.0
    return counts


# =============================================================================
# (b') 日文羅馬拼音：子音 × 母音（台灣常見日系品牌：mihoyo、shiseido、kanebo、suntory）
# =============================================================================

_ROMAJI_CONSONANTS = ("", "k", "s", "t", "n", "h", "m", "y", "r", "w", "g", "z", "d", "b", "p")
_ROMAJI_VOWELS = ("a", "i", "u", "e", "o")
_ROMAJI_SPECIAL = ("shi", "chi", "tsu", "fu", "ji", "sha", "shu", "sho", "cha", "chu", "cho", "ja", "ju", "jo",
                   "kya", "kyu", "kyo", "nya", "nyu", "nyo", "hya", "hyu", "hyo", "mya", "myu", "myo",
                   "rya", "ryu", "ryo", "gya", "gyu", "gyo", "bya", "byu", "byo", "pya", "pyu", "pyo")
_ROMAJI_INVALID = {"yi", "ye", "wi", "wu", "we", "si", "ti", "tu", "hu", "zi", "di", "du"}


def romaji_syllables() -> List[str]:
    out = {c + v for c in _ROMAJI_CONSONANTS for v in _ROMAJI_VOWELS} - _ROMAJI_INVALID
    out.update(_ROMAJI_SPECIAL)
    return sorted(out)


def romaji_counts() -> Counter:
    """單音節＋固定種子的 2～3 音節組合（可再接撥音 n），模擬日文詞的字母分布。"""
    syllables = romaji_syllables()
    rng = random.Random(SEED + 2)
    counts: Counter = Counter({s: 3.0 for s in syllables})
    for _ in range(ROMAJI_WORDS):
        word = "".join(rng.choice(syllables) for _ in range(rng.randint(2, 3)))
        if rng.random() < 0.25:
            word += "n"
        counts[word] += 1.0
    return counts


# =============================================================================
# (c) 品牌 token（少量補充）
# =============================================================================

def brand_counts() -> Counter:
    sys.path.insert(0, VA_DIR)
    try:
        import rules_config as R  # noqa: WPS433（工具腳本延遲載入）
    finally:
        sys.path.pop(0)
    words: Counter = Counter()
    for brand, domains in R.BRAND_OFFICIAL_DOMAINS.items():
        for token in [brand] + [d.split(".", 1)[0] for d in domains]:
            for part in re.findall(r"[a-z]+", token.lower()):
                if len(part) >= 3:
                    words[part] += 1.0
    return words


# =============================================================================
# trigram 模型（Witten-Bell 插值）
# =============================================================================

def _scaled(counts: Counter, share: float) -> Dict[str, float]:
    """類別權重：每個字以 log2(1+次數) 計（避免 the／of 主導），再把整類縮放到指定占比。"""
    raw = {w: math.log2(1.0 + c) for w, c in counts.items()}
    total_chars = sum(v * (len(w) + 1) for w, v in raw.items()) or 1.0
    return {w: v * share / total_chars for w, v in raw.items()}


def build_model(weighted_words: Dict[str, float]) -> Dict[str, object]:
    symbols = ALPHABET + END
    uni: Dict[str, float] = defaultdict(float)
    bi: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    tri: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    scale = VIRTUAL_TOTAL   # 把占比換成「虛擬次數」，讓 MIN_TRIGRAM_COUNT 有意義
    for word, w in weighted_words.items():
        w *= scale
        padded = START + START + word + END
        for i in range(2, len(padded)):
            a, b, c = padded[i - 2], padded[i - 1], padded[i]
            uni[c] += w
            bi[b][c] += w
            tri[a + b][c] += w

    n_uni = sum(uni.values())
    t_uni = sum(1 for v in uni.values() if v > 0)
    lam_uni = n_uni / (n_uni + t_uni)
    p_uni = {c: lam_uni * uni.get(c, 0.0) / n_uni + (1 - lam_uni) / len(symbols) for c in symbols}

    p_bi: Dict[str, Dict[str, float]] = {}
    for b in START + ALPHABET:
        row = bi.get(b, {})
        n, t = sum(row.values()), sum(1 for v in row.values() if v > 0)
        lam = n / (n + t) if n else 0.0
        p_bi[b] = {c: (lam * row.get(c, 0.0) / n if n else 0.0) + (1 - lam) * p_uni[c] for c in symbols}

    tri_out: Dict[str, float] = {}
    bo_out: Dict[str, float] = {}
    for ctx in sorted(tri):
        row = tri[ctx]
        n, t = sum(row.values()), sum(1 for v in row.values() if v > 0)
        lam = n / (n + t)
        # backoff 權重：未輸出的 trigram 一律以 (1-λ)·P(c|b) 計
        bo_out[ctx] = round(math.log2(1 - lam), 3)
        for c, v in row.items():
            if v < MIN_TRIGRAM_COUNT:
                continue
            p = lam * v / n + (1 - lam) * p_bi[ctx[1]][c]
            tri_out[ctx + c] = round(math.log2(p), 2)
    bi_out = {b + c: round(math.log2(p), 2) for b, row in p_bi.items() for c, p in row.items()}
    return {"bi": bi_out, "tri": tri_out, "bo": bo_out}


def surprisal_bits(word: str, table: Dict[str, object]) -> List[float]:
    """與 features.py 相同的查表方式（供 ref 統計與 --check 使用）。"""
    bi, tri, bo = table["bi"], table["tri"], table["bo"]
    padded = START + START + word + END
    out = []
    for i in range(2, len(padded)):
        a, b, c = padded[i - 2], padded[i - 1], padded[i]
        lp = tri.get(a + b + c)  # type: ignore[union-attr]
        if lp is None:
            lp = bo.get(a + b, 0.0) + bi.get(b + c, -12.0)  # type: ignore[union-attr]
        out.append(-lp)
    return out


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="產生 char_ngram_table.json（sld_randomness 用）")
    parser.add_argument("--check", action="store_true", help="只產生並印出統計，不寫檔")
    parser.add_argument("--output", default=OUTPUT_PATH)
    args = parser.parse_args(argv)

    stdlib_dir = sysconfig.get_paths()["stdlib"]
    eng = english_counts(stdlib_dir)
    pin = pinyin_counts()
    rom = romaji_counts()
    brands = brand_counts()
    weighted: Dict[str, float] = defaultdict(float)
    for counts, share in ((eng, WEIGHT_SHARE["english"]), (pin, WEIGHT_SHARE["pinyin"]),
                          (rom, WEIGHT_SHARE["romaji"]), (brands, WEIGHT_SHARE["brand"])):
        for w, v in _scaled(counts, share).items():
            weighted[w] += v
    table = build_model(dict(sorted(weighted.items())))

    # 語料本身的平均驚奇度（每字元 bits），features.py 不依賴，只供對照
    def mean_bits(words: Iterable[str]) -> float:
        bits = [b for w in words for b in surprisal_bits(w, table)]
        return round(sum(bits) / max(len(bits), 1), 3)

    rng = random.Random(SEED + 1)
    random_words = ["".join(rng.choice(ALPHABET) for _ in range(rng.randint(5, 10))) for _ in range(2000)]
    ref = {
        "english_mean_bits": mean_bits(sorted(eng)[:20000]),
        "pinyin_mean_bits": mean_bits(sorted(pin)),
        "uniform_random_mean_bits": mean_bits(random_words),
    }
    out = {
        "format": TABLE_FORMAT,
        "order": 3,
        "alphabet": ALPHABET,
        "start": START,
        "end": END,
        "smoothing": "Witten-Bell interpolation（trigram→bigram→unigram→uniform）；log2 機率",
        "corpus": {
            "english": f"Python {sys.version_info.major}.{sys.version_info.minor} 標準庫註解與 docstring 英文字（{len(eng)} 種）",
            "pinyin": f"聲母×韻母產生的拼音音節 {len(pinyin_syllables())} 個＋雙音節組合 {PINYIN_PAIRS} 個（seed {SEED}）",
            "romaji": f"子音×母音產生的日文羅馬拼音音節 {len(romaji_syllables())} 個＋2～3 音節組合 {ROMAJI_WORDS} 個",
            "brand": f"rules_config 品牌 token／官方網域 SLD（{len(brands)} 種，權重 {WEIGHT_SHARE['brand']:.0%}）",
            "weight_share": WEIGHT_SHARE,
            "excluded": "未使用 data*.csv、feedback.csv、url_cases.json 或任何訓練／驗收網址",
        },
        "min_trigram_count": MIN_TRIGRAM_COUNT,
        "ref": ref,
        **table,
    }
    print(f"英文字 {len(eng)} 種、拼音 {len(pin)} 種、羅馬拼音 {len(rom)} 種、品牌 {len(brands)} 種；"
          f"bigram {len(table['bi'])}、trigram {len(table['tri'])}、context {len(table['bo'])}")
    print(f"平均驚奇度（bits/字元）：英文 {ref['english_mean_bits']}、拼音 {ref['pinyin_mean_bits']}、"
          f"均勻亂數 {ref['uniform_random_mean_bits']}")
    if args.check:
        return 0
    text = json.dumps(out, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    tmp = args.output + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, args.output)
    print(f"已寫入 {args.output}（{len(text) / 1024:.1f} KB）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
