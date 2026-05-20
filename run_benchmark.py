"""
Main runner: Sigma detector vs synthetic NAB-style corpus.

Steps:
  1. Build the synthetic corpus
  2. Run SigmaDetector across each file to get per-row anomaly scores
  3. For each scoring profile, sweep thresholds to find the optimum (the official
     NAB harness also does this — each algorithm picks one threshold per profile)
  4. Score at the optimal threshold
  5. Compare against published NAB baselines
"""

import sys
import numpy as np
import pandas as pd

from sigma_detector import run_detector, SigmaDetector
from nab_scoring import (
    score_single_file,
    null_detector_score,
    perfect_detector_score,
    aggregate_corpus,
    PROFILES,
)
from synth_corpus import build_corpus


def score_all_files_at_threshold(corpus, all_scores, threshold, min_gap_steps=200):
    """Return per-file results dict for one threshold across the whole corpus."""
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
            min_gap_steps=min_gap_steps,
        )
        result["file_key"] = file_key
        per_file.append(result)
        per_null.append(null_detector_score(windows))
        per_perfect.append(perfect_detector_score(windows))
    return per_file, per_null, per_perfect


def optimize_threshold(corpus, all_scores, profile_name):
    """Sweep thresholds and find the one that maximizes corpus-wide NAB score for the profile."""
    thresholds = np.linspace(0.05, 0.95, 37)  # 37 points, step ~0.025
    best_threshold = 0.5
    best_score = -1e9
    for t in thresholds:
        per_file, per_null, per_perfect = score_all_files_at_threshold(corpus, all_scores, t)
        agg = aggregate_corpus(per_file, per_null, per_perfect)
        if agg[profile_name] > best_score:
            best_score = agg[profile_name]
            best_threshold = float(t)
    return best_threshold, best_score


def main():
    print("=" * 78)
    print("SIGMA DETECTOR vs NAB-style synthetic corpus")
    print("Shadow Theory anomaly detection benchmark")
    print("=" * 78)

    print("\n[1/4] Building synthetic NAB-style corpus...")
    corpus = build_corpus()
    total_points = sum(len(p["df"]) for p in corpus.values())
    total_windows = sum(len(p["windows"]) for p in corpus.values())
    print(f"      {len(corpus)} files, {total_points:,} timestamps, {total_windows} anomaly windows")

    print("\n[2/4] Running SigmaDetector on every file...")
    all_scores = {}
    for file_key, payload in corpus.items():
        values = payload["df"]["value"].to_numpy()
        n = len(values)
        probation = max(750, int(0.15 * n))
        scores = run_detector(values, probation_steps=probation, smoothing_window=30)
        all_scores[file_key] = scores
        non_zero = (scores > 0.01).sum()
        max_score = scores.max()
        print(
            f"      {file_key:60s}  n={len(values):6d}  "
            f"sigma_max={max_score:.3f}  active_pts={non_zero}"
        )

    print("\n[3/4] Optimizing threshold per profile (NAB harness behavior)...")
    final = {}
    for profile in PROFILES.keys():
        best_t, best_score = optimize_threshold(corpus, all_scores, profile)
        final[profile] = {"threshold": best_t, "score": best_score}
        print(f"      profile={profile:18s}  best_threshold={best_t:.3f}  NAB_score={best_score:6.2f}")

    print("\n[4/4] Per-file breakdown at standard-profile threshold:")
    std_t = final["standard"]["threshold"]
    per_file, per_null, per_perfect = score_all_files_at_threshold(corpus, all_scores, std_t)
    print(
        f"      {'file':60s}  {'wins':>4s}  {'TP':>3s}  {'FN':>3s}  {'FP':>3s}  "
        f"{'raw':>7s}  {'null':>7s}  {'norm':>7s}"
    )
    raw_total = 0.0
    null_total = 0.0
    perfect_total = 0.0
    for r, n, p in zip(per_file, per_null, per_perfect):
        raw = r["scores"]["standard"]
        n_raw = n["standard"]
        p_raw = p["standard"]
        norm = 0.0
        if p_raw - n_raw > 0:
            norm = 100.0 * (raw - n_raw) / (p_raw - n_raw)
        print(
            f"      {r['file_key']:60s}  {r['n_windows']:4d}  "
            f"{r['n_tp']:3d}  {r['n_fn']:3d}  {r['n_fp']:3d}  "
            f"{raw:7.2f}  {n_raw:7.2f}  {norm:7.2f}"
        )
        raw_total += raw
        null_total += n_raw
        perfect_total += p_raw

    print("\n" + "=" * 78)
    print("FINAL NAB SCORES (Sigma Detector)")
    print("=" * 78)
    for profile in PROFILES.keys():
        print(f"  {profile:20s}  {final[profile]['score']:7.2f}  (threshold={final[profile]['threshold']:.3f})")

    # Reference baselines from Lavin & Ahmad 2015 (Numenta NAB paper)
    # Published scores on the full 58-file NAB corpus
    print("\n" + "-" * 78)
    print("Reference published baselines (full 58-file NAB corpus, Lavin & Ahmad 2015 + updates):")
    print("-" * 78)
    print("  Algorithm                Standard   RewardLowFP   RewardLowFN")
    print("  Numenta HTM              70.10      63.06         74.30")
    print("  Numenta HTM (Java)       69.74      62.59         74.10")
    print("  Twitter ADVec            47.06      33.61         53.50")
    print("  Etsy Skyline             35.69      27.08         44.46")
    print("  Bayesian Changepoint     17.73       3.16         32.53")
    print("  Windowed Gaussian        39.59      20.86         47.42")
    print("  EXPoSE                   16.40       3.16         26.93")
    print("  Random                   17.00      ~ 1.50        25.00")
    print("  Null detector             0.00       0.00          0.00")
    print("  Perfect detector        100.00     100.00        100.00")
    print()
    print("Note: This run is on a 10-file SYNTHETIC NAB-equivalent subset to validate")
    print("the pipeline. The real benchmark must be run on the actual NAB CSVs locally")
    print("(GitHub is blocked from this sandbox). The local-run package is shipped with this.")
    print("=" * 78)

    return final


if __name__ == "__main__":
    main()
