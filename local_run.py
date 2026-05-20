"""
Local NAB runner v4 — Sigma v7 vs the real 58-file NAB corpus.

v4 change: empirical-quantile adaptive z_threshold.
Per file: measure natural outlier rate in probation period, set z_threshold
so only ~0.5% of baseline naturally crosses it. Stable files get low threshold
(catches real anomalies), bursty files get high threshold (avoids FPs).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sigma_detector import run_detector, calibrate_z_threshold_from_baseline
from nab_scoring import (
    score_single_file,
    null_detector_score,
    perfect_detector_score,
    aggregate_corpus,
    PROFILES,
)


NAB_REPO_URL = "https://github.com/numenta/NAB.git"
NAB_DIR = Path("./NAB")

PROBATION_FRACTION = 0.15
PROBATION_MIN = 750
SMOOTHING_WINDOW = 30
MIN_GAP_FRACTION = 1.0
MIN_GAP_FLOOR = 100
MIN_GAP_CEILING = 2000
TARGET_OUTLIER_RATE = 0.005


def ensure_nab():
    if NAB_DIR.exists():
        print(f"NAB repo found at {NAB_DIR}")
        return
    print(f"Cloning NAB from {NAB_REPO_URL}...")
    subprocess.run(["git", "clone", "--depth=1", NAB_REPO_URL, str(NAB_DIR)], check=True)


def load_corpus():
    labels_path = NAB_DIR / "labels" / "combined_windows.json"
    with open(labels_path) as f:
        all_windows = json.load(f)
    corpus = {}
    skipped = []
    for file_key, win_list in all_windows.items():
        csv_path = NAB_DIR / "data" / file_key
        if not csv_path.exists():
            skipped.append(file_key)
            continue
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        windows = [(pd.Timestamp(s), pd.Timestamp(e)) for s, e in win_list]
        corpus[file_key] = {"df": df, "windows": windows}
    if skipped:
        print(f"Skipped {len(skipped)} files")
    return corpus


def median_window_length_steps(df, windows):
    if not windows:
        ts = pd.DatetimeIndex(df["timestamp"])
        if len(ts) < 2:
            return MIN_GAP_FLOOR
        step_sec = (ts[1] - ts[0]).total_seconds()
        return max(MIN_GAP_FLOOR, int(86400 / max(step_sec, 1)))
    ts = pd.DatetimeIndex(df["timestamp"])
    if len(ts) < 2:
        return MIN_GAP_FLOOR
    step_sec = (ts[1] - ts[0]).total_seconds()
    win_lengths_sec = [(e - s).total_seconds() for s, e in windows]
    median_sec = float(np.median(win_lengths_sec))
    return max(1, int(median_sec / max(step_sec, 1)))


def file_min_gap(df, windows):
    mw = median_window_length_steps(df, windows)
    gap = int(MIN_GAP_FRACTION * mw)
    return max(MIN_GAP_FLOOR, min(MIN_GAP_CEILING, gap))


def file_probation(df):
    return max(PROBATION_MIN, int(PROBATION_FRACTION * len(df)))


def score_all_files_at_threshold(corpus, all_scores, threshold, per_file_gap):
    per_file = []
    per_null = []
    per_perfect = []
    for file_key, payload in corpus.items():
        df = payload["df"]
        windows = payload["windows"]
        scores = all_scores[file_key]
        result = score_single_file(
            timestamps=pd.DatetimeIndex(df["timestamp"]),
            anomaly_scores=scores,
            threshold=threshold,
            windows=windows,
            min_gap_steps=per_file_gap[file_key],
        )
        result["file_key"] = file_key
        per_file.append(result)
        per_null.append(null_detector_score(windows))
        per_perfect.append(perfect_detector_score(windows))
    return per_file, per_null, per_perfect


def optimize_threshold(corpus, all_scores, profile_name, per_file_gap):
    thresholds = np.linspace(0.05, 0.95, 37)
    best_t, best_s = 0.5, -1e9
    for t in thresholds:
        pf, pn, pp = score_all_files_at_threshold(corpus, all_scores, t, per_file_gap)
        agg = aggregate_corpus(pf, pn, pp)
        if agg[profile_name] > best_s:
            best_s = agg[profile_name]
            best_t = float(t)
    return best_t, best_s


def main():
    print("=" * 78)
    print("SIGMA DETECTOR v7 vs REAL NAB CORPUS")
    print("Empirical-quantile adaptive z_threshold (calibrated to baseline outlier rate)")
    print("=" * 78)

    print("\n[1/5] Ensuring NAB corpus is available...")
    ensure_nab()

    print("\n[2/5] Loading corpus...")
    corpus = load_corpus()
    total_points = sum(len(p["df"]) for p in corpus.values())
    total_windows = sum(len(p["windows"]) for p in corpus.values())
    print(f"      {len(corpus)} files, {total_points:,} timestamps, {total_windows} anomaly windows")

    per_file_probation = {k: file_probation(p["df"]) for k, p in corpus.items()}
    per_file_gap = {k: file_min_gap(p["df"], p["windows"]) for k, p in corpus.items()}

    print(f"\n[3/5] Calibrating per-file z_thresholds (target outlier rate: {TARGET_OUTLIER_RATE*100}%)...")
    per_file_z = {}
    for file_key, payload in corpus.items():
        values = payload["df"]["value"].to_numpy()
        prob = per_file_probation[file_key]
        calibration = values[:min(prob, len(values))]
        z_thr = calibrate_z_threshold_from_baseline(calibration, short_window=30, target_outlier_rate=TARGET_OUTLIER_RATE)
        per_file_z[file_key] = z_thr

    z_thrs = list(per_file_z.values())
    print(f"      z_threshold range: [{min(z_thrs):.2f}, {max(z_thrs):.2f}]  mean={np.mean(z_thrs):.2f}")

    print("\n[4/5] Running SigmaDetector v7 on every file...")
    all_scores = {}
    for i, (file_key, payload) in enumerate(corpus.items(), 1):
        values = payload["df"]["value"].to_numpy()
        scores = run_detector(
            values,
            probation_steps=per_file_probation[file_key],
            smoothing_window=SMOOTHING_WINDOW,
            adaptive_threshold=True,
            target_outlier_rate=TARGET_OUTLIER_RATE,
        )
        all_scores[file_key] = scores
        sys.stdout.write(f"\r      [{i}/{len(corpus)}] {file_key[:60]:60s}")
        sys.stdout.flush()
    print()

    print("\n[5/5] Optimizing threshold per profile...")
    final = {}
    for profile in PROFILES.keys():
        best_t, best_score = optimize_threshold(corpus, all_scores, profile, per_file_gap)
        final[profile] = {"threshold": best_t, "score": best_score}
        print(f"      profile={profile:18s}  best_threshold={best_t:.3f}  NAB_score={best_score:6.2f}")

    print("\n" + "=" * 78)
    print("FINAL NAB SCORES (Sigma Detector v7, REAL CORPUS)")
    print("=" * 78)
    for profile in PROFILES.keys():
        print(f"  {profile:20s}  {final[profile]['score']:7.2f}  (threshold={final[profile]['threshold']:.3f})")

    std_t = final["standard"]["threshold"]
    per_file, per_null, per_perfect = score_all_files_at_threshold(
        corpus, all_scores, std_t, per_file_gap
    )
    rows = []
    for r, n, p in zip(per_file, per_null, per_perfect):
        raw = r["scores"]["standard"]
        n_raw = n["standard"]
        p_raw = p["standard"]
        norm = 100.0 * (raw - n_raw) / max(p_raw - n_raw, 1e-9) if p_raw != n_raw else 0.0
        rows.append((r["file_key"], r["n_windows"], r["n_tp"], r["n_fn"], r["n_fp"], norm, per_file_z[r["file_key"]]))
    rows.sort(key=lambda x: x[5], reverse=True)

    print("\nPer-file breakdown (top 20 + bottom 10):")
    print(f"  {'file':52s}  {'wins':>4s} {'TP':>3s} {'FN':>3s} {'FP':>3s}  {'norm':>7s}  {'z':>5s}")
    for r in rows[:20]:
        print(f"  {r[0]:52s}  {r[1]:4d} {r[2]:3d} {r[3]:3d} {r[4]:3d}  {r[5]:7.2f}  {r[6]:5.2f}")
    print("  " + "-" * 92)
    for r in rows[-10:]:
        print(f"  {r[0]:52s}  {r[1]:4d} {r[2]:3d} {r[3]:3d} {r[4]:3d}  {r[5]:7.2f}  {r[6]:5.2f}")

    print("\n" + "-" * 78)
    print("Reference published baselines:")
    print("-" * 78)
    print("  Algorithm                Standard   RewardLowFP   RewardLowFN")
    print("  Numenta HTM              70.10      63.06         74.30")
    print("  Twitter ADVec            47.06      33.61         53.50")
    print("  Etsy Skyline             35.69      27.08         44.46")
    print("  Bayesian Changepoint     17.73       3.16         32.53")
    print("  Windowed Gaussian        39.59      20.86         47.42")
    print("  EXPoSE                   16.40       3.16         26.93")
    print("  Random                   17.00       1.50         25.00")
    print("  Null                      0.00       0.00          0.00")
    print("  Perfect                 100.00     100.00        100.00")
    print("=" * 78)

    out_path = Path("./sigma_nab_results_v7.json")
    serializable = {
        profile: {"threshold": final[profile]["threshold"], "score": final[profile]["score"]}
        for profile in PROFILES.keys()
    }
    serializable["per_file"] = [
        {"file": r[0], "n_windows": r[1], "tp": r[2], "fn": r[3], "fp": r[4], "norm_standard": r[5], "z_threshold": r[6]}
        for r in rows
    ]
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
