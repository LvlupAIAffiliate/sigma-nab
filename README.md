# Sigma Detector — A Coherence-Based Streaming Anomaly Detector

A streaming univariate anomaly detector derived from **Shadow Theory**, a resolution-invariant sign topology framework based on Robinson's nonstandard analysis. Benchmarked on the official **Numenta Anomaly Benchmark (NAB)** with no task-specific training.

**Author:** Leonard Benson
**Organization:** Altus Level Up AI LLC, Adelanto, California
**Date of public release:** May 19, 2026

---

## Intellectual Property Notice

This repository implements methods covered by:

- **USPTO Provisional Patent Application #64/036,030** — Filed April 10, 2026. Shadow Theory framework. Production Sigma formulation and detector architectures are subject to additional patent applications pending. Sole inventor: Leonard Benson.
- **Zenodo Publication, DOI [10.5281/zenodo.19476061](https://doi.org/10.5281/zenodo.19476061)** — Shadow Theory: Resolution-Invariant Sign Topology Framework.
- **Mathematical Logic Quarterly** — Manuscript submitted, under review.

**Citation required.** See [CITATION.cff](./CITATION.cff). Commercial use requires written license — contact 
lvlupaiaffiliate@mail.com.

This public release establishes the priority date for the empirical-quantile calibration methodology (May 19, 2026) and the Sigma Detector's NAB benchmark result.

---

## Headline Result

Sigma Detector v4 on the official NAB corpus (58 files, 365,558 timestamps, 116 anomaly windows):

| Profile | **Sigma v4** | Numenta HTM | Twitter ADVec | Etsy Skyline | Windowed Gaussian | Bayesian Changepoint | EXPoSE | Random |
|---|---|---|---|---|---|---|---|---|
| Standard | **42.80** | 70.10 | 47.06 | 35.69 | 39.59 | 17.73 | 16.40 | 17.00 |
| Reward Low FP | **19.65** | 63.06 | 33.61 | 27.08 | 20.86 | 3.16 | 3.16 | 1.50 |
| Reward Low FN | **52.72** | 74.30 | 53.50 | 44.46 | 47.42 | 32.53 | 26.93 | 25.00 |

**Sigma v4 outperforms** Etsy Skyline, Windowed Gaussian, Bayesian Changepoint, EXPoSE, and Random on the Standard profile.
**Sigma v4 is within 4.26 of Twitter ADVec** with no NAB-specific training.
**Sigma v4 uses 7 fixed parameters and ~150 lines of code**, vs Numenta HTM's online per-file learning with hundreds of parameters.

Full results: [`sigma_nab_results_v7.json`](./sigma_nab_results_v7.json).

---

## The Method

### Core Formula

The Sigma score for a streaming univariate time-series is:
Where:
- **phi** is an outlier-calibrated logistic activation, shifted by a per-file `z_threshold`
- **dSignal** is the standardized deviation of the short-window median from the long baseline median
- **dVariance** is the standardized deviation of the short-window MAD-derived std from baseline std
- **dTrend** is the standardized deviation of the short-window slope from baseline slope
- **P(v)** is an exponentially-decayed accumulator of persistent component firing

This structure derives from a coherence framework originally developed for electrical fault diagnostics (ShadowGrid) and ported directly to univariate time-series without task-specific retraining.

### Cross-Domain Origin

The detector was not designed for NAB. Its formula, weights, and persistence dynamics were developed for:

- Electrical infrastructure fault detection (ShadowGrid)
- EV battery health prognostication (Domain XVII, NASA dataset, 116-cycle lead time)
- Clinical deterioration detection (ShadowDx, MIMIC-IV open-access demo v2.2; credentialed validation in progress)
- Vector-host transmission dynamics (Domain XIX)

The NAB result demonstrates that the underlying coherence framework transfers to a benchmark in a completely separate domain without task-specific adaptation.

### v4 Key Innovation: Empirical-Quantile Calibration

For each file, the detector calibrates its `z_threshold` based on the file's own natural outlier rate during the probation period:

1. Take the first 15% of the file (probation period)
2. Slide the short-window across it, compute z-score of each window mean vs rolling baseline
3. Set `z_threshold` to the 99.5th percentile of |z|

This means only ~0.5% of natural baseline variation would cross the threshold by chance, **per file**. Quiet/stable files get low thresholds (catches real anomalies). Bursty/noisy files get high thresholds (avoids false positives).

Observed `z_threshold` range in production: **2.50 (stable files) to 5.00 (perfect periodic files)**.

---

## Iteration Trail

Four documented methodology iterations:

| Version | Standard | Reward Low FP | Reward Low FN | Change Applied |
|---|---|---|---|---|
| v1 | 33.68 | 17.44 | 48.38 | Direct port from ShadowGrid (electrical fault detection) |
| v2 | 38.84 | 17.71 | 49.51 | + Probation period, score smoothing, adaptive min-gap |
| v3 | 37.59 | 17.44 | 51.27 | + Multi-scale baselines, trend dominance bonus |
| **v4** | **42.80** | **19.65** | **52.72** | + Empirical-quantile z_threshold calibration |

Total gain v1 to v4: **+9.12 Standard, +2.21 Reward Low FP, +4.34 Reward Low FN**.

---

## Reproducing the Result

```bash
git clone https://github.com/LvlupAIAffiliate/sigma-nab.git
cd sigma-nab
pip3 install numpy pandas
python3 local_run.py
```

Expected output: `sigma_nab_results_v7.json` with full per-file scores. Runtime ~5-15 minutes on Apple M-series silicon.

---


---


---

## Mathematical Positioning (May 2026)

A survey of publicly-available NAB-targeted anomaly detectors reveals that current approaches fall primarily into two categories: classical statistical methods (e.g. `iandanforth/NAB-detectors` from 2019, now abandoned) and deep learning architectures (Transformer-based, autoencoder-based, VAE-based — see TranAD, Tuli et al. VLDB 2022 for the modern reference).

**To the author's knowledge, no other publicly-available streaming anomaly detector uses Robinson's nonstandard analysis as a mathematical foundation.** Shadow Theory's sign-topology framework is the distinguishing contribution of this work, independent of the specific NAB result.

## Files

| File | Purpose |
|---|---|
| `sigma_detector.py` | Core detector (~150 lines, 7 parameters) |
| `nab_scoring.py` | Faithful implementation of NAB's three official scoring profiles |
| `local_run.py` | Main runner — clones NAB, scores all 58 files, optimizes thresholds |
| `synth_corpus.py` | Synthetic NAB-style data generator (for sandbox validation) |
| `run_benchmark.py` | Runs detector against synthetic corpus |
| `sigma_nab_results_v7.json` | Full per-file results at v4 |

---

## License

This work is released under a **non-commercial research-use license**. See [LICENSE](./LICENSE).

Commercial use, including incorporation into commercial AI safety products, anomaly detection services, or paid software, requires written license from Altus Level Up AI LLC. Contact lvlupaiaffiliate@mail.com for licensing inquiries.

The underlying Shadow Theory framework is covered by USPTO Provisional Patent Application #64/036,030.

---

## Citing This Work

If you use Sigma Detector or Shadow Theory methodology in your research, please cite both the Zenodo publication and this repository. See [CITATION.cff](./CITATION.cff) for machine-readable metadata.

---

## Related Work

- **Numenta NAB:** Lavin, A., & Ahmad, S. (2015). *Evaluating Real-time Anomaly Detection Algorithms — the Numenta Anomaly Benchmark.* ICMLA 2015.
- **Robinson's Nonstandard Analysis:** Robinson, A. (1966). *Non-standard Analysis.* North-Holland Publishing.
- **Shadow Theory Foundational Paper:** Benson, L. (2026). Zenodo DOI [10.5281/zenodo.19476061](https://doi.org/10.5281/zenodo.19476061).

---

*"Be a careful steward of what you're given before asking to be trusted with more."*


## Patent Status

Reference implementation of the Sigma anomaly-detection method derived
from Shadow Theory (Benson II, L., Zenodo DOI: 10.5281/zenodo.19476061).

Foundational Shadow Theory framework — including the Shadow Sign Function,
Shadow State classification, and transition detection methods — is covered
under USPTO Provisional Application #64/036,030 (filed April 10, 2026) by
Leonard Benson II / Altus Level Up AI LLC, Adelanto, California.

Production implementations, including the full-fidelity Sigma scoring engine
and tiered deployment architectures, are subject to additional patent
applications pending. Commercial use requires licensing inquiry to
Altus Level Up AI LLC.

### Cross-Domain Applications

The Shadow Theory framework has been applied across multiple domains, each
subject to its own intellectual property protections. Domain-specific
applications referenced in this repository — including but not limited to
electrical infrastructure fault detection (ShadowGrid), electric vehicle
battery health prognostication (Domain XVII), clinical deterioration
detection (ShadowDx), vector-host transmission dynamics (Domain XIX), and
streaming anomaly detection — are each covered by separate provisional
patent applications, pending continuations, or filings in preparation by
Altus Level Up AI LLC. Inclusion of cross-domain validation results in
this public repository constitutes reference disclosure of method
performance and does not grant license to any specific domain application.

