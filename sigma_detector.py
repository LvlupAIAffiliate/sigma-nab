"""
Sigma Detector v7 — Shadow Theory anomaly detector, NAB-tuned.

v7 fix over v6: adaptive z_threshold now measures the file's NATURAL
outlier rate during probation period, then calibrates z_threshold to
the file's own statistics — not to noise level.

Logic: if a file naturally produces large z-scores frequently in its
baseline period, raise the threshold so those don't count as anomalies.
If a file is genuinely stable in its baseline, keep the threshold low
so real anomalies (which differ from baseline) get caught.

This is a more principled version of "adaptive sensitivity": we calibrate
to the file's own empirical distribution, not to an arbitrary noise proxy.
"""

import numpy as np
from collections import deque


class SigmaDetector:
    def __init__(
        self,
        baseline_window_fast: int = 600,
        baseline_window_slow: int = 2400,
        short_window: int = 30,
        w1: float = 0.40,
        w2: float = 0.25,
        w3: float = 0.20,
        gamma: float = 0.15,
        z_threshold: float = 3.0,
        phi_steepness: float = 1.0,
        priority_decay: float = 0.97,
        priority_fire_threshold: float = 0.3,
        trend_dominance_threshold: float = 2.0,
        trend_dominance_bonus: float = 0.15,
        eps: float = 1e-6,
    ):
        self.bw_fast = baseline_window_fast
        self.bw_slow = baseline_window_slow
        self.sw = short_window
        self.w1, self.w2, self.w3, self.gamma = w1, w2, w3, gamma
        self.z_thr = z_threshold
        self.k = phi_steepness
        self.priority_decay = priority_decay
        self.fire_thr = priority_fire_threshold
        self.trend_dom_thr = trend_dominance_threshold
        self.trend_dom_bonus = trend_dominance_bonus
        self.eps = eps

        self.buffer = deque(maxlen=baseline_window_slow)
        self.priority = 0.0

    def _phi(self, z: float) -> float:
        z_shifted = self.k * (abs(z) - self.z_thr)
        z_clipped = np.clip(z_shifted, -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-z_clipped))

    def _robust_stats(self, arr: np.ndarray):
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))
        sd = 1.4826 * mad + self.eps
        x = np.arange(len(arr), dtype=float)
        x_mean = x.mean()
        denom = ((x - x_mean) ** 2).sum() + self.eps
        slope = float(((x - x_mean) * (arr - med)).sum() / denom)
        return med, sd, slope

    def score(self, value: float) -> float:
        self.buffer.append(float(value))

        if len(self.buffer) < self.sw + 20:
            return 0.0

        arr_full = np.asarray(self.buffer, dtype=float)
        arr_b_fast = arr_full[-min(self.bw_fast, len(arr_full)):]
        arr_b_slow = arr_full
        arr_s = arr_full[-self.sw:]

        mu_bf, sd_bf, slope_bf = self._robust_stats(arr_b_fast)
        mu_bs, sd_bs, slope_bs = self._robust_stats(arr_b_slow)
        mu_s, sd_s, slope_s = self._robust_stats(arr_s)

        z_signal_fast = (mu_s - mu_bf) / sd_bf
        z_signal_slow = (mu_s - mu_bs) / sd_bs
        z_signal = z_signal_fast if abs(z_signal_fast) >= abs(z_signal_slow) else z_signal_slow

        z_variance_fast = (sd_s - sd_bf) / sd_bf
        z_variance_slow = (sd_s - sd_bs) / sd_bs
        z_variance = z_variance_fast if abs(z_variance_fast) >= abs(z_variance_slow) else z_variance_slow

        slope_scale_slow = sd_bs / max(self.sw, 1) + self.eps
        z_trend = (slope_s - slope_bs) / slope_scale_slow

        c_signal = self._phi(z_signal)
        c_variance = self._phi(z_variance)
        c_trend = self._phi(z_trend)

        if abs(z_trend) > self.trend_dom_thr * abs(z_signal) and abs(z_trend) > self.z_thr:
            c_trend = min(1.0, c_trend + self.trend_dom_bonus)

        max_component = max(c_signal, c_variance, c_trend)
        firing_intensity = max_component if max_component > self.fire_thr else 0.0
        self.priority = self.priority_decay * self.priority + (1.0 - self.priority_decay) * firing_intensity

        sigma = (
            self.w1 * c_signal
            + self.w2 * c_variance
            + self.w3 * c_trend
            + self.gamma * self.priority
        )
        return float(np.clip(sigma, 0.0, 1.0))


def _rolling_max(arr: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return arr.copy()
    n = len(arr)
    out = np.empty(n, dtype=float)
    dq = deque()
    for i in range(n):
        while dq and dq[0] <= i - window:
            dq.popleft()
        while dq and arr[dq[-1]] <= arr[i]:
            dq.pop()
        dq.append(i)
        out[i] = arr[dq[0]]
    return out


def calibrate_z_threshold_from_baseline(
    baseline_values: np.ndarray,
    short_window: int = 30,
    target_outlier_rate: float = 0.005,
    z_min: float = 2.5,
    z_max: float = 5.0,
) -> float:
    """
    Calibrate z_threshold to the file's NATURAL outlier rate during baseline.

    Method: slide the short_window across baseline period, compute z-score of
    each short-window mean against the rolling baseline median/MAD. Find the
    z-value at the (1 - target_outlier_rate) quantile of |z|. That's the
    z_threshold where roughly target_outlier_rate fraction of baseline data
    would naturally fire.

    Quiet/stable signal -> low natural |z| -> low calibrated threshold (catches real anomalies)
    Bursty/jumpy signal -> high natural |z| -> high calibrated threshold (avoids FPs)

    This is the OPPOSITE of v6's complexity logic — and the correct direction.
    """
    n = len(baseline_values)
    if n < 200:
        return 3.0  # default

    # Use full baseline period stats
    med_full = float(np.median(baseline_values))
    mad_full = float(np.median(np.abs(baseline_values - med_full)))
    sd_full = 1.4826 * mad_full + 1e-6

    # Slide short_window across baseline, compute z of each short-window mean
    z_scores = []
    step = max(1, short_window // 4)  # 75% overlap
    for i in range(short_window, n, step):
        window = baseline_values[i - short_window:i]
        mu_s = float(np.median(window))
        z = abs(mu_s - med_full) / sd_full
        z_scores.append(z)

    if not z_scores:
        return 3.0

    z_arr = np.array(z_scores)
    # The threshold above which only target_outlier_rate of baseline points sit
    quantile = 1.0 - target_outlier_rate
    z_thr = float(np.quantile(z_arr, quantile))
    return float(np.clip(z_thr, z_min, z_max))


def run_detector(
    values: np.ndarray,
    probation_steps: int = 0,
    smoothing_window: int = 0,
    adaptive_threshold: bool = True,
    target_outlier_rate: float = 0.005,
    **kwargs,
) -> np.ndarray:
    """
    Run SigmaDetector with empirical-quantile adaptive z_threshold.

    The new adaptive logic: calibrate z_threshold so that the natural baseline
    only crosses it target_outlier_rate (0.5% by default) of the time.
    """
    if adaptive_threshold:
        # Use probation period as the baseline calibration sample
        if probation_steps > 0 and probation_steps < len(values):
            calibration_sample = values[:probation_steps]
        else:
            # Fall back to first 15% of values
            calibration_sample = values[:max(750, int(0.15 * len(values)))]

        z_thr = calibrate_z_threshold_from_baseline(
            calibration_sample,
            short_window=kwargs.get("short_window", 30),
            target_outlier_rate=target_outlier_rate,
        )
        kwargs.setdefault("z_threshold", z_thr)

    det = SigmaDetector(**kwargs)
    out = np.empty(len(values), dtype=float)
    for i, v in enumerate(values):
        out[i] = det.score(v)

    if probation_steps > 0:
        out[:min(probation_steps, len(out))] = 0.0

    if smoothing_window > 1:
        out = _rolling_max(out, smoothing_window)

    return out


# Backward-compat exports for runners
def estimate_signal_complexity(values):
    """Deprecated — kept for compatibility."""
    return 0.5


def adaptive_z_threshold(values, base=3.0, lo=2.5, hi=4.5):
    """Deprecated — kept for compatibility; returns midpoint."""
    return calibrate_z_threshold_from_baseline(values[:max(750, int(0.15 * len(values)))])
