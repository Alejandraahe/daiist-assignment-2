"""
Shared settings and logic for train.py and app.py.

The decision: a Philippine bank's cash-processing centre receives the peso
notes that its counting machines rejected as "suspicious". Today a teller
looks at every one. The model scores each note photo, and each note gets one
of three outcomes:

    PASS    -> accepted as genuine
    REVIEW  -> a person checks it (UV lamp, watermark, security thread)
    FAIL    -> retained as counterfeit and surrendered to the central bank (BSP)

Positive class = COUNTERFEIT. Everything that costs money is defined here,
once, so the training sweep and the dashboard use exactly the same numbers.
All amounts are in euros.
"""

import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
WEIGHTS = ROOT / "weights"
DATASETS = ROOT / "datasets"

# --- Roboflow Universe dataset (used as-is, never forked) -------------------
WORKSPACE = "thesis2024"
PROJECT = "ccm-2024"
VERSION = 2
UNIVERSE_URL = f"https://universe.roboflow.com/{WORKSPACE}/{PROJECT}/dataset/{VERSION}"

POSITIVE_KEYWORD = "counterfeit"   # any class whose name contains this is a fake note

# --- Business assumptions (all defended in REPORT.md) -----------------------
EUR_PER_PHP = 1 / 65          # about 65 pesos per euro
COST_FP = 3.00                # a GENUINE note we FAIL: customer complaint, re-check, goodwill
COST_TP = 0.50                # a FAKE we FAIL: paperwork to surrender it to the central bank
COST_TN = 0.00                # a genuine note we PASS: business as usual
COST_REVIEW = 0.10            # one human check: ~90 s of a teller at ~4 EUR/hour loaded cost
REVIEW_ACCURACY = 1.0         # a teller with UV lamp + watermark check is treated as correct
# COST_FN is not one number: a fake we PASS loses its face value (see note_value).

BASE_RATE = 0.05              # share of fakes among machine-rejected notes (assumption)
NOTES_PER_DAY = 2000          # suspicious notes reaching the centre per day
SECONDS_PER_REVIEW = 90
TELLER_HOURS_PER_DAY = 6.5    # productive hours in an 8-hour shift

WORDS = {"twenty": 20, "fifty": 50, "one hundred": 100, "two hundred": 200,
         "five hundred": 500, "one thousand": 1000}


def is_positive(name: str) -> bool:
    return POSITIVE_KEYWORD in str(name).lower()


def peso_value(name: str) -> int:
    """Face value in pesos from a class name like 'Counterfeit five hundred peso'."""
    n = str(name).lower()
    for words in sorted(WORDS, key=len, reverse=True):   # 'five hundred' before 'hundred'
        if words in n:
            return WORDS[words]
    digits = re.findall(r"\d+", n)
    return int(digits[0]) if digits else 0


def fn_cost_eur(value_php) -> np.ndarray:
    """Passing a fake loses its face value: the bank cannot pass it on or claim it back."""
    return np.asarray(value_php, dtype=float) * EUR_PER_PHP


def outcome_counts(scores, labels, low, high):
    """score < low -> PASS; low <= score < high -> REVIEW; score >= high -> FAIL."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    pass_, fail = s < low, s >= high
    review = ~pass_ & ~fail
    return {
        "fake_pass": int((pass_ & (y == 1)).sum()), "fake_review": int((review & (y == 1)).sum()),
        "fake_fail": int((fail & (y == 1)).sum()),
        "real_pass": int((pass_ & (y == 0)).sum()), "real_review": int((review & (y == 0)).sum()),
        "real_fail": int((fail & (y == 0)).sum()),
    }


def cost_per_1000(scores, labels, values, low, high, base_rate=BASE_RATE, fn_scale=1.0,
                  c_fp=COST_FP, c_tp=COST_TP, c_tn=COST_TN, c_review=COST_REVIEW):
    """Expected cost per 1,000 notes at the PRODUCTION base rate.

    The test split has far more fakes than production, so we measure the
    average cost per fake note and per genuine note separately, then re-weight
    them by the base rate we expect in production.
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    v = np.asarray(values, dtype=float)
    pass_, fail = s < low, s >= high
    review = ~pass_ & ~fail
    a = REVIEW_ACCURACY
    fn = fn_cost_eur(v) * fn_scale
    per_fake = np.where(pass_, fn, np.where(fail, c_tp, c_review + a * c_tp + (1 - a) * fn))
    per_real = np.where(pass_, c_tn, np.where(fail, c_fp, c_review + a * c_tn + (1 - a) * c_fp))
    fakes, reals = y == 1, y == 0
    cost_fake = per_fake[fakes].mean() if fakes.any() else 0.0
    cost_real = per_real[reals].mean() if reals.any() else 0.0
    f_rev = review[fakes].mean() if fakes.any() else 0.0
    r_rev = review[reals].mean() if reals.any() else 0.0
    return {
        "cost_per_1000": float(1000 * (base_rate * cost_fake + (1 - base_rate) * cost_real)),
        "review_share": float(base_rate * f_rev + (1 - base_rate) * r_rev),
        "fake_caught": float((fail[fakes].mean() + a * f_rev) if fakes.any() else 0.0),
        "genuine_failed": float(fail[reals].mean() if reals.any() else 0.0),
        **outcome_counts(s, y, low, high),
    }


def staffing(review_share, notes_per_day=NOTES_PER_DAY):
    per_day = review_share * notes_per_day
    hours = per_day * SECONDS_PER_REVIEW / 3600
    return {"reviews_per_day": per_day, "review_hours": hours, "tellers": hours / TELLER_HOURS_PER_DAY}
