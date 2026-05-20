"""
NAB Scoring — faithful implementation of the official NAB scoring methodology.

Per Numenta paper "Evaluating Real-time Anomaly Detection Algorithms" (Lavin & Ahmad, 2015):

Each labeled anomaly has a *window* around it. Detections inside the window get positive
credit via a scaled sigmoid (earlier = more credit). Detections outside any window are
false positives with a fixed penalty. The score for a profile is:

    S_profile = sum_over_anomaly_windows( sigmoid_credit(earliest_detection) )
                + A_FP * (num_false_positives)
                + A_FN * (num_missed_windows)

Three official profiles (weighting TP, FP, FN, TN differently):

    Standard:        A_TP=1.0   A_FP=-0.11  A_FN=-1.0
    Reward low FP:   A_TP=1.0   A_FP=-0.22  A_FN=-1.0
    Reward low FN:   A_TP=1.0   A_FP=-0.11  A_FN=-2.0

Raw scores are normalized so a perfect detector = 100, a null detector (no detections) = 0.

This implementation:
  - Takes per-row anomaly scores and a threshold
  - Reads NAB combined_windows.json for ground truth windows
  - Computes the three official profile scores
"""

import json
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


# Official NAB application profiles
PROFILES = {
    "standard": {"A_TP": 1.0, "A_FP": -0.11, "A_FN": -1.0},
    "reward_low_FP": {"A_TP": 1.0, "A_FP": -0.22, "A_FN": -1.0},
    "reward_low_FN": {"A_TP": 1.0, "A_FP": -0.11, "A_FN": -2.0},
}


def scaled_sigmoid(relative_position: float) -> float:
    """
    NAB's sigmoid credit function.
    relative_position is in [-1, 0] for detections inside the window:
        -1 = at start of window (full credit)
         0 = at end of window (less credit)
    Returns a credit value in (0, 1].

    Implementation per the NAB paper: y = 1 / (1 + exp(5*relative_position))
    Adjusted so detection at window start ~ 1.0 and at window end ~ small positive.
    """
    if relative_position > 0:
        # Detection after window end — still some credit decaying to 0
        return 2.0 * (1.0 / (1.0 + math.exp(5.0 * relative_position))) - 1.0
    # Detection inside window: credit decays from start to end
    return 2.0 * (1.0 / (1.0 + math.exp(5.0 * relative_position))) - 1.0


@dataclass
class FileResult:
    file_key: str
    raw_scores: Dict[str, float]   # per profile
    n_windows: int
    n_tp_windows: int
    n_fp: int


def score_single_file(
    timestamps: pd.DatetimeIndex,
    anomaly_scores: np.ndarray,
    threshold: float,
    windows: List[Tuple[pd.Timestamp, pd.Timestamp]],
    min_gap_steps: int = 0,
) -> Dict[str, float]:
    """
    Score one file's detector output against its ground-truth anomaly windows.

    Returns raw score per profile (un-normalized).

    min_gap_steps: enforce a minimum gap between consecutive detections.
                   Common practice in streaming anomaly detection — counts an
                   event once, not every threshold crossing within a sustained
                   anomaly. NAB harness does this via its per-window matching.
    """
    # Detections: indices where score >= threshold, with minimum gap enforced
    raw_detections = np.where(anomaly_scores >= threshold)[0]
    if min_gap_steps > 0 and len(raw_detections) > 0:
        filtered = [raw_detections[0]]
        for d in raw_detections[1:]:
            if d - filtered[-1] >= min_gap_steps:
                filtered.append(d)
        detections = np.array(filtered)
    else:
        detections = raw_detections
    detection_times = timestamps[detections] if len(detections) > 0 else pd.DatetimeIndex([])

    # For each window, find earliest detection inside (or after) the window
    tp_credits = []
    matched_detection_idxs = set()

    for w_start, w_end in windows:
        w_length = (w_end - w_start).total_seconds()
        if w_length <= 0:
            continue

        # Find detections in window
        in_window_mask = (detection_times >= w_start) & (detection_times <= w_end)
        in_window_idxs = np.where(in_window_mask)[0]

        if len(in_window_idxs) > 0:
            # Earliest detection in window
            earliest_idx = in_window_idxs[0]
            earliest_time = detection_times[earliest_idx]
            # Relative position: -1 at start, 0 at end
            position = (earliest_time - w_start).total_seconds() / w_length
            relative_position = position - 1.0  # shift to [-1, 0]
            credit = scaled_sigmoid(relative_position)
            tp_credits.append(max(0.0, credit))
            # Mark all in-window detections as "matched" (don't count as FP)
            for didx in in_window_idxs:
                matched_detection_idxs.add(int(detections[didx]))
        else:
            tp_credits.append(None)  # window missed = FN

    # False positives: detections outside any window
    n_fp = 0
    for i, didx in enumerate(detections):
        if int(didx) not in matched_detection_idxs:
            # Verify it's truly outside all windows (defensive)
            t = timestamps[didx]
            in_any = any(w_start <= t <= w_end for w_start, w_end in windows)
            if not in_any:
                n_fp += 1

    n_fn = sum(1 for c in tp_credits if c is None)
    n_tp = sum(1 for c in tp_credits if c is not None)
    tp_sum = sum(c for c in tp_credits if c is not None)

    # Compute per-profile raw score
    results = {}
    for name, p in PROFILES.items():
        raw = p["A_TP"] * tp_sum + p["A_FP"] * n_fp + p["A_FN"] * n_fn
        results[name] = raw

    return {
        "scores": results,
        "n_windows": len(windows),
        "n_tp": n_tp,
        "n_fn": n_fn,
        "n_fp": n_fp,
        "n_detections": int(len(detections)),
        "tp_sum": float(tp_sum),
    }


def null_detector_score(windows: List[Tuple[pd.Timestamp, pd.Timestamp]]) -> Dict[str, float]:
    """A detector that never fires: zero TP, zero FP, all windows missed."""
    n_fn = len(windows)
    out = {}
    for name, p in PROFILES.items():
        out[name] = p["A_FN"] * n_fn
    return out


def perfect_detector_score(windows: List[Tuple[pd.Timestamp, pd.Timestamp]]) -> Dict[str, float]:
    """
    A 'perfect' detector: fires exactly once at the start of each window, no false positives.
    Credit per window ~ scaled_sigmoid(-1) ≈ 0.987 (effectively 1.0).
    """
    credit_per_window = scaled_sigmoid(-1.0)
    out = {}
    n_windows = len(windows)
    for name, p in PROFILES.items():
        out[name] = p["A_TP"] * credit_per_window * n_windows
    return out


def normalize_score(
    raw_score: float,
    null_score: float,
    perfect_score: float,
) -> float:
    """
    NAB normalization: a perfect detector = 100, a null detector = 0.
    Score above 100 is impossible by definition; below 0 means worse than no detection.
    """
    if perfect_score - null_score == 0:
        return 0.0
    return 100.0 * (raw_score - null_score) / (perfect_score - null_score)


def aggregate_corpus(
    per_file_results: List[Dict],
    per_file_nulls: List[Dict[str, float]],
    per_file_perfects: List[Dict[str, float]],
) -> Dict[str, float]:
    """Aggregate across all files: sum raw, sum null, sum perfect, then normalize."""
    out = {}
    for profile in PROFILES.keys():
        raw_total = sum(r["scores"][profile] for r in per_file_results)
        null_total = sum(n[profile] for n in per_file_nulls)
        perfect_total = sum(p[profile] for p in per_file_perfects)
        out[profile] = normalize_score(raw_total, null_total, perfect_total)
    return out
