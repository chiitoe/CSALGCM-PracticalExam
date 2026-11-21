"""
File: variant_solver.py // EDITED VERSION
Calibrated against Sample Variants.xlsx & Oco et al. (2024).
Supports hyphenated Tagalog words (dinami-dami/dinami-rami), prefix+d/r shifts
(magdadaan/magdaraan, mandidigma/mandirigma), and blocks English -ed/-er false positives.

Usage:
  python variant_solver.py Raw.txt Variants.csv --lang fil --min-freq 11
  python variant_solver.py Raw.txt Variants.csv --lang both --min-freq 1
  
  FOR MacOS:
  python3 variant_solver.py Raw.txt Variants.csv --lang fil --min-freq 11
  
  
  NOTE:
  Try changing the --min-freq 11 to --min-freq5
  
  python variant_solver.py Raw.txt Variants.csv --lang fil --min-freq 5
  python3 variant_solver.py Raw.txt Variants.csv --lang fil --min-freq 5
"""
import argparse
import csv
import random
import re
from collections import Counter, defaultdict
from typing import List, Tuple, Optional

try:
    from rapidfuzz.distance import Levenshtein as RFLev
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

VOWELS = set("aeiou")
FILIPINO_SUB_PAIRS = { # Added different pairs to catch more pairs
    frozenset(('o', 'u')), frozenset(('e', 'i')), frozenset(('d', 'r')),
    frozenset(('c', 'k')), frozenset(('v', 'b')), frozenset(('i', 'y')),
    frozenset(('o', 'w')), frozenset(('f', 'p')),
    frozenset(('s', 'z')), frozenset(('h', 'j')), frozenset(('u', 'w')),
}

# Explicit semantic non-variants from Oco et al. (2024) Table 2, Table 3 & Sample Variants
NON_VARIANT_BLACKLIST = {
    frozenset(("opo", "upo")), frozenset(("pare", "pari")),
    frozenset(("madikit", "marikit")), frozenset(("pauwi", "pawi")),
    frozenset(("paiyak", "payak")), frozenset(("aabutan", "aabutin")),
    frozenset(("pagduda", "pagdura")), frozenset(("mag-date", "mag-rate")),
    frozenset(("explodes", "explores")), frozenset(("robber", "rubber")),
    frozenset(("polling", "pulling")),
}


def is_vowel(ch: str) -> bool:
    return ch in VOWELS


def is_consonant(ch: str) -> bool:
    return ch.isalpha() and ch not in VOWELS


# --- 1. DIVIDE AND CONQUER: MERGE SORT ---
def merge_sort_vocab(words: List[str]) -> List[str]:
    if len(words) <= 1:
        return words
    mid = len(words) // 2
    left = merge_sort_vocab(words[:mid])
    right = merge_sort_vocab(words[mid:])
    merged, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        if (len(left[i]), left[i]) <= (len(right[j]), right[j]):
            merged.append(left[i]); i += 1
        else:
            merged.append(right[j]); j += 1
    merged.extend(left[i:]); merged.extend(right[j:])
    return merged


# --- 2. DYNAMIC PROGRAMMING: BANDED LEVENSHTEIN DISTANCE ---
def banded_dp_distance(s1: str, s2: str, max_k: int = 1) -> int:
    if abs(len(s1) - len(s2)) > max_k:
        return -1
    if HAS_RAPIDFUZZ:
        d = RFLev.distance(s1, s2, score_cutoff=max_k)
        return d if d <= max_k else -1

    if len(s1) < len(s2):
        s1, s2 = s2, s1
    m, n = len(s1), len(s2)
    INF = max_k + 1
    prev = [j if j <= max_k else INF for j in range(n + 1)]
    for i in range(1, m + 1):
        curr = [INF] * (n + 1)
        if i <= max_k:
            curr[0] = i
        row_min = curr[0]
        for j in range(max(1, i - max_k), min(n, i + max_k) + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
            if curr[j] < row_min:
                row_min = curr[j]
        if row_min > max_k:
            return -1
        prev = curr
    return prev[n] if prev[n] <= max_k else -1


# --- 3. GREEDY ALGORITHMS: FILIPINO & ENGLISH ORTHOGRAPHIC RULES ---
def check_filipino_pair(w1: str, w2: str, c1_cnt: int, c2_cnt: int, min_freq: int = 1) -> Optional[str]:
    if min(c1_cnt, c2_cnt) < min_freq:
        return None
    if frozenset((w1, w2)) in NON_VARIANT_BLACKLIST:
        return None
    if banded_dp_distance(w1, w2, max_k=1) != 1:
        return None

    # Reject words with double consonants (polittico, robber, grabbed/grabber, planned/planner)
    for w in (w1, w2):
        for k in range(len(w) - 1):
            if w[k] == w[k + 1] and is_consonant(w[k]):
                return None

    # Same-length substitutions
    if len(w1) == len(w2):
        idx = next((i for i in range(len(w1)) if w1[i] != w2[i]), -1)
        if idx == -1:
            return None
        c1, c2 = w1[idx], w2[idx]
        pair = frozenset((c1, c2))
        if pair not in FILIPINO_SUB_PAIRS:
            return None

        # Tier 1: Oco et al. (2024) Table 5 100%-Precision Context Rules (ehVC/ihVC, omC/umC)
        if pair == frozenset(('e', 'i')) and idx + 3 < len(w1):
            if w1[idx + 1] == 'h' and is_vowel(w1[idx + 2]) and is_consonant(w1[idx + 3]):
                return f"{c1} vs. {c2}"
        if pair == frozenset(('o', 'u')) and idx + 2 < len(w1):
            if w1[idx + 1] == 'm' and is_consonant(w1[idx + 2]):
                return f"{c1} vs. {c2}"

        # Tier 2: General Filipino Phonological Shifts
        if len(w1) < 6:
            return None

        # Reject Filipino verb suffix focus shifts (-an vs -in, e.g., aabutan/aabutin)
        if idx == len(w1) - 2 and w1.endswith('n'):
            return None

        # d vs. r Rules (calibrated to Sample Variants.xlsx):
        # 1. Must be followed by a vowel (blocks word-final English -ed vs -er like decided/decider, managed/manager)
        # 2. Must be preceded by a vowel, a hyphen (dinami-dami/dinami-rami), or a prefix consonant 'g'/'n' (magdadaan/mandidigma)
        if pair == frozenset(('d', 'r')):
            if not (0 < idx < len(w1) - 1 and is_vowel(w1[idx + 1])):
                return None
            prev_ch = w1[idx - 1]
            if not (is_vowel(prev_ch) or prev_ch in ('-', 'g', 'n')):
                return None

        return f"{c1} vs. {c2}"

    # Length differs by 1: Glide Reductions (CuwV <-> CwV, CiyV <-> CyV)
    longer, shorter = (w1, w2) if len(w1) > len(w2) else (w2, w1)
    if len(shorter) < 5:
        return None
    idx = 0
    while idx < len(shorter) and longer[idx] == shorter[idx]:
        idx += 1
    if 0 < idx and idx + 2 < len(longer):
        prev_c, extra_v, glide_c, next_v = longer[idx - 1], longer[idx], longer[idx + 1], longer[idx + 2]
        if is_consonant(prev_c) and is_vowel(next_v):
            if (extra_v == 'u' and glide_c == 'w') or (extra_v == 'i' and glide_c == 'y'):
                full_glide = extra_v + glide_c
                return f"{full_glide} vs. {glide_c}" if len(w1) > len(w2) else f"{glide_c} vs. {full_glide}"
    return None


def generate_english_candidates(word: str) -> List[Tuple[str, str]]:
    cands = []
    for m in re.finditer(r"our", word):
        if m.start() >= 2:
            cands.append((word[:m.start()] + "or" + word[m.start() + 3:], "our vs. or"))
    for s_uk, s_us in [
        ("isation", "ization"), ("isations", "izations"), ("ising", "izing"),
        ("ised", "ized"), ("ise", "ize"), ("ises", "izes"),
        ("yse", "yze"), ("ysing", "yzing"), ("ysed", "yzed"),
    ]:
        if word.endswith(s_uk) and len(word) >= len(s_uk) + 2:
            cands.append((word[:-len(s_uk)] + s_us, "s vs. z"))
    if word.endswith("ence") and len(word) >= 6:
        cands.append((word[:-4] + "ense", "c vs. s"))
    if word.endswith("re") and len(word) >= 6 and word[-3] in "tmbgl":
        cands.append((word[:-2] + "er", "re vs. er"))
    for m in re.finditer(r"ll(ing|ed|er|ers|or|ors|y|ous)", word):
        if m.start() >= 3:
            cands.append((word[:m.start()] + "l" + word[m.start() + 2:], "ll vs. l"))
    for dg in ("ae", "oe"):
        if dg in word and len(word) >= 6:
            cands.append((word.replace(dg, "e", 1), f"{dg} vs. e"))
    if word.endswith("ogue") and len(word) >= 7:
        cands.append((word[:-2], "ogue vs. og"))
    return cands


# --- 4. EXTRACTION PIPELINE ---
def run_solver(input_file: str, output_csv: str, lang: str = "both", min_freq: int = 11):
    counts = Counter()
    # Includes optional inner hyphens for Tagalog reduplications (e.g., dinami-dami)
    pattern = re.compile(r"[a-zA-ZñÑ]+(?:-[a-zA-ZñÑ]+)*")
    with open(input_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            counts.update(pattern.findall(line.lower()))

    vocab = merge_sort_vocab([w for w in counts if len(w) >= 5 and counts[w] >= min_freq])
    vocab_set = set(vocab)
    results, seen = [], set()

    if lang in ("fil", "both"):
        masked_buckets = defaultdict(list)
        for w in vocab:
            for i in range(len(w)):
                if w[i] in "aeioucdrkvwyfpbszhj": # i added b,s,z,h,j to the mix
                    masked_buckets[(len(w), i, w[:i] + "*" + w[i + 1:])].append(w)

        for bucket in masked_buckets.values():
            if 1 < len(bucket) <= 60:
                for i in range(len(bucket)):
                    for j in range(i + 1, len(bucket)):
                        w1, w2 = sorted((bucket[i], bucket[j]))
                        pair_key = (w1, w2)
                        if pair_key not in seen:
                            cdiff = check_filipino_pair(w1, w2, counts[w1], counts[w2], min_freq)
                            if cdiff:
                                seen.add(pair_key)
                                results.append((w1, w2, counts[w1], counts[w2], cdiff))

        for w_long in vocab:
            if len(w_long) >= 6:
                for i in range(1, len(w_long) - 2):
                    if w_long[i:i + 2] in ("uw", "iy"):
                        w_short = w_long[:i] + w_long[i + 1:]
                        if w_short in vocab_set:
                            w1, w2 = sorted((w_long, w_short))
                            pair_key = (w1, w2)
                            if pair_key not in seen:
                                cdiff = check_filipino_pair(w1, w2, counts[w1], counts[w2], min_freq)
                                if cdiff:
                                    seen.add(pair_key)
                                    results.append((w1, w2, counts[w1], counts[w2], cdiff))

    if lang in ("eng", "both"):
        for w_raw in vocab:
            for w_cand, cdiff in generate_english_candidates(w_raw):
                if w_cand in vocab_set and w_raw != w_cand:
                    if banded_dp_distance(w_raw, w_cand, max_k=2) in (1, 2):
                        w1, w2 = sorted((w_raw, w_cand))
                        pair_key = (w1, w2)
                        if pair_key not in seen:
                            # Ensure cdiff direction matches sorted (w1, w2)
                            if w1 != w_raw and " vs. " in cdiff:
                                left_c, right_c = cdiff.split(" vs. ")
                                cdiff = f"{right_c} vs. {left_c}"
                            seen.add(pair_key)
                            results.append((w1, w2, counts[w1], counts[w2], cdiff))

    # Sort alphabetically by (word1, word2) to match Sample Variants.xlsx
    results.sort(key=lambda r: (r[0], r[1]))

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["word1", "word2", "word1_count", "word2_count", "character_difference"])
        writer.writerows(results)
    print(f"Extracted {len(results)} pairs to {output_csv} (mode={lang}, min_freq={min_freq}).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", default="dataset.txt")
    parser.add_argument("output", nargs="?", default="Variants.csv")
    parser.add_argument("--lang", choices=["fil", "eng", "both"], default="fil")
    parser.add_argument("--min-freq", type=int, default=11)
    args = parser.parse_args()
    run_solver(args.input, args.output, args.lang, args.min_freq)