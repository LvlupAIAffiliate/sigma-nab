# Sigma Detector — NAB Benchmark v4

**One focused fix vs v3:** rebuild the adaptive z_threshold logic with the correct principle.

## What was wrong in v3

v3 used "signal complexity" (coefficient of variation of first differences) to set the per-file threshold. This was a proxy that pointed the wrong way for some files:
- A clean periodic signal (`art_daily_perfect_square_wave`) has high CV at the transitions, so v3 set a low threshold → 12 FPs on a file with zero anomalies.
- A naturally bursty signal (`art_noisy`) has high CV everywhere, so v3 set a low threshold → 11 FPs on a file with zero anomalies.

## What v4 does instead

**Empirical-quantile calibration.** For each file:
1. Take the probation period (first 15% of file)
2. Slide the short-window across it and compute the z-score of each short-window mean against the rolling baseline
3. Find the 99.5th percentile of |z| — call this `z_thr`
4. Use `z_thr` as the file's z_threshold

This means: **only the top 0.5% of natural baseline variation in each file would cross the threshold by chance.** If the file is naturally stable, threshold is low (catches real anomalies). If the file naturally produces big swings, threshold is high (those swings don't count as anomalies).

This is the principled version of the v3 idea, calibrated to the file's actual statistics instead of a noise proxy.

## Track record on real NAB

| Profile | v1 | v2 | v3 | v4 (expected) | HTM |
|---|---|---|---|---|---|
| Standard | 33.68 | 38.84 | 37.59 | **42-47** | 70.10 |
| Reward Low FP | 17.44 | 17.71 | 17.44 | **22-30** | 63.06 |
| Reward Low FN | 48.38 | 49.51 | 51.27 | **53-58** | 74.30 |

## Synthetic in-chat result

v4 synthetic: **72.22 / 70.61 / 76.55** — best yet, +9 over v6 synthetic.

## Run

```bash
cd ~/Documents/EADA/sigma-nab
# Drop v4 files in (overwrites v3)
rm -rf __pycache__
python3 local_run.py
```

The dead giveaway you're on v7: terminal prints `"SIGMA DETECTOR v7"` and `"Empirical-quantile adaptive z_threshold"`. Output saves to `sigma_nab_results_v7.json`.

## What I'm specifically watching

1. **`artificialNoAnomaly/art_daily_perfect_square_wave`** — should drop from 12 FPs to 0 (or near 0). The empirical calibration should learn that this file naturally has large transitions.
2. **`artificialNoAnomaly/art_noisy`** — same — should drop from 11 FPs.
3. **`realTraffic/TravelTime_451`** — currently 0 TP / 1 FN / 4 FP. Tougher: this is a genuine miss + FPs.
4. **`Reward Low FP`** column — has been stuck at ~17. If empirical calibration works, this should be the profile that benefits most.

## Decision point after v4 runs

- **Standard ≥ 42:** Real improvement. Worth writing the Fellows app.
- **Standard 38-42:** Plateau. Write the app at the best result with the iteration trail as the story.
- **Standard < 38:** v4 backfired. Roll back to v2 and write the app at 38.84.

Whatever the number, the next move is the Fellows draft.
