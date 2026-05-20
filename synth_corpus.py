"""
Synthesize NAB-style time-series matching the documented characteristics of
real NAB files. Used for in-chat validation. The real benchmark must be run
on the actual NAB CSVs locally — this is a sanity test that the pipeline works.

We replicate the labeled anomaly windows from NAB's combined_windows.json
and generate synthetic signals with the right kind of anomaly (level shift,
variance increase, periodic disruption, gradual drift) injected into those windows.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict


def synth_periodic_with_anomalies(
    start: str,
    end: str,
    period_hours: float,
    base_amplitude: float,
    noise_std: float,
    baseline: float,
    anomaly_windows: List[Tuple[str, str]],
    anomaly_type: str = "spike",
    freq_minutes: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a periodic signal (e.g. daily-cycle) with anomalies injected in given windows.

    anomaly_type:
      'spike'    : sharp amplitude increase
      'shift'    : sustained level shift
      'variance' : variance increase (noisier in window)
      'drop'     : sustained level drop
      'drift'    : gradual upward drift across window
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start=start, end=end, freq=f"{freq_minutes}min")
    n = len(idx)

    # Base periodic signal
    t_hours = (idx - idx[0]).total_seconds().to_numpy() / 3600.0
    signal = baseline + base_amplitude * np.sin(2 * np.pi * t_hours / period_hours)
    # Weekly modulation for realism
    signal += 0.3 * base_amplitude * np.sin(2 * np.pi * t_hours / (24 * 7))
    # Noise
    signal += rng.normal(0, noise_std, size=n)

    # Inject anomalies
    for w_start_str, w_end_str in anomaly_windows:
        w_start = pd.Timestamp(w_start_str)
        w_end = pd.Timestamp(w_end_str)
        mask = (idx >= w_start) & (idx <= w_end)
        m_n = int(mask.sum())
        if m_n == 0:
            continue

        if anomaly_type == "spike":
            signal[mask] += rng.normal(3 * base_amplitude, base_amplitude, size=m_n)
        elif anomaly_type == "shift":
            # Soften edges: ramp in over first 20%, ramp out over last 20%
            envelope = np.ones(m_n)
            ramp = max(int(m_n * 0.2), 1)
            envelope[:ramp] = np.linspace(0, 1, ramp)
            envelope[-ramp:] = np.linspace(1, 0, ramp)
            signal[mask] += 2.5 * base_amplitude * envelope
        elif anomaly_type == "variance":
            signal[mask] += rng.normal(0, 3.0 * noise_std, size=m_n)
        elif anomaly_type == "drop":
            envelope = np.ones(m_n)
            ramp = max(int(m_n * 0.2), 1)
            envelope[:ramp] = np.linspace(0, 1, ramp)
            envelope[-ramp:] = np.linspace(1, 0, ramp)
            signal[mask] -= 2.0 * base_amplitude * envelope
        elif anomaly_type == "drift":
            drift = np.linspace(0, 2.0 * base_amplitude, m_n)
            signal[mask] += drift
        else:
            raise ValueError(f"Unknown anomaly_type: {anomaly_type}")

    return pd.DataFrame({"timestamp": idx, "value": signal})


# A representative subset of NAB files with real anomaly windows from combined_windows.json
# Each entry: (file_key, generator_kwargs, anomaly_type)
NAB_LIKE_FILES = [
    # NYC taxi - 30-min buckets, strong daily/weekly cycle, 5 anomaly windows
    (
        "realKnownCause/nyc_taxi.csv",
        {
            "start": "2014-07-01 00:00:00",
            "end": "2015-01-31 23:30:00",
            "period_hours": 24.0,
            "base_amplitude": 8000,
            "noise_std": 1200,
            "baseline": 15000,
            "freq_minutes": 30,
            "anomaly_windows": [
                ("2014-10-30 15:30:00", "2014-11-03 22:30:00"),  # NYC Marathon
                ("2014-11-25 12:00:00", "2014-11-29 19:00:00"),  # Thanksgiving
                ("2014-12-23 11:30:00", "2014-12-27 18:30:00"),  # Christmas
                ("2014-12-29 21:30:00", "2015-01-03 04:30:00"),  # New Year
                ("2015-01-24 20:30:00", "2015-01-29 03:30:00"),  # Snow storm
            ],
            "seed": 1,
        },
        "drop",  # holiday-style drops + variance
    ),
    # Machine temperature failure - sustained variance/level changes
    (
        "realKnownCause/machine_temperature_system_failure.csv",
        {
            "start": "2013-12-02 21:15:00",
            "end": "2014-02-19 15:25:00",
            "period_hours": 12.0,
            "base_amplitude": 4,
            "noise_std": 1.5,
            "baseline": 85,
            "freq_minutes": 5,
            "anomaly_windows": [
                ("2013-12-10 06:25:00", "2013-12-12 05:35:00"),
                ("2013-12-15 17:50:00", "2013-12-17 17:00:00"),
                ("2014-01-27 14:20:00", "2014-01-29 13:30:00"),
                ("2014-02-07 14:55:00", "2014-02-09 14:05:00"),
            ],
            "seed": 2,
        },
        "shift",
    ),
    # Ambient temperature - 2 wide anomaly windows
    (
        "realKnownCause/ambient_temperature_system_failure.csv",
        {
            "start": "2013-07-04 00:00:00",
            "end": "2014-05-28 00:00:00",
            "period_hours": 24.0,
            "base_amplitude": 8,
            "noise_std": 1.0,
            "baseline": 72,
            "freq_minutes": 60,
            "anomaly_windows": [
                ("2013-12-15 07:00:00", "2013-12-30 09:00:00"),
                ("2014-03-29 15:00:00", "2014-04-20 22:00:00"),
            ],
            "seed": 3,
        },
        "drift",
    ),
    # CPU utilization ASG misconfig - single sustained anomaly
    (
        "realKnownCause/cpu_utilization_asg_misconfiguration.csv",
        {
            "start": "2014-07-04 00:04:00",
            "end": "2014-07-18 23:59:00",
            "period_hours": 24.0,
            "base_amplitude": 12,
            "noise_std": 3.0,
            "baseline": 45,
            "freq_minutes": 5,
            "anomaly_windows": [
                ("2014-07-10 12:29:00", "2014-07-15 17:19:00"),
            ],
            "seed": 4,
        },
        "shift",
    ),
    # EC2 latency system failure - 3 narrow anomalies
    (
        "realKnownCause/ec2_request_latency_system_failure.csv",
        {
            "start": "2014-03-07 03:46:00",
            "end": "2014-04-05 22:36:00",
            "period_hours": 24.0,
            "base_amplitude": 5,
            "noise_std": 1.0,
            "baseline": 30,
            "freq_minutes": 5,
            "anomaly_windows": [
                ("2014-03-14 03:31:00", "2014-03-14 14:41:00"),
                ("2014-03-18 17:06:00", "2014-03-19 04:16:00"),
                ("2014-03-20 21:26:00", "2014-03-21 03:41:00"),
            ],
            "seed": 5,
        },
        "spike",
    ),
    # Twitter volume AAPL - bursty social media
    (
        "realTweets/Twitter_volume_AAPL.csv",
        {
            "start": "2015-02-26 21:42:53",
            "end": "2015-04-22 21:52:53",
            "period_hours": 24.0,
            "base_amplitude": 80,
            "noise_std": 30,
            "baseline": 150,
            "freq_minutes": 5,
            "anomaly_windows": [
                ("2015-03-03 04:37:53", "2015-03-04 13:37:53"),
                ("2015-03-09 01:02:53", "2015-03-10 10:02:53"),
                ("2015-03-15 10:27:53", "2015-03-16 19:27:53"),
                ("2015-03-30 10:57:53", "2015-03-31 19:57:53"),
            ],
            "seed": 6,
        },
        "spike",
    ),
    # EC2 CPU utilization 24ae8d - 2 anomaly windows
    (
        "realAWSCloudwatch/ec2_cpu_utilization_24ae8d.csv",
        {
            "start": "2014-02-14 14:30:00",
            "end": "2014-02-28 14:30:00",
            "period_hours": 24.0,
            "base_amplitude": 10,
            "noise_std": 2.0,
            "baseline": 40,
            "freq_minutes": 5,
            "anomaly_windows": [
                ("2014-02-26 13:45:00", "2014-02-27 06:25:00"),
                ("2014-02-27 08:55:00", "2014-02-28 01:35:00"),
            ],
            "seed": 7,
        },
        "shift",
    ),
    # Artificial flatmiddle - known synthetic
    (
        "artificialWithAnomaly/art_daily_flatmiddle.csv",
        {
            "start": "2014-04-01 00:00:00",
            "end": "2014-04-15 23:55:00",
            "period_hours": 24.0,
            "base_amplitude": 30,
            "noise_std": 2.0,
            "baseline": 50,
            "freq_minutes": 5,
            "anomaly_windows": [
                ("2014-04-10 07:15:00", "2014-04-11 16:45:00"),
            ],
            "seed": 8,
        },
        "shift",
    ),
    # Artificial no anomaly - should produce no FPs ideally
    (
        "artificialNoAnomaly/art_daily_small_noise.csv",
        {
            "start": "2014-04-01 00:00:00",
            "end": "2014-04-15 23:55:00",
            "period_hours": 24.0,
            "base_amplitude": 30,
            "noise_std": 2.0,
            "baseline": 50,
            "freq_minutes": 5,
            "anomaly_windows": [],
            "seed": 9,
        },
        "shift",
    ),
    # Rogue agent key hold - narrow windows
    (
        "realKnownCause/rogue_agent_key_hold.csv",
        {
            "start": "2014-07-14 00:00:00",
            "end": "2014-07-19 12:00:00",
            "period_hours": 24.0,
            "base_amplitude": 0.2,
            "noise_std": 0.05,
            "baseline": 1.0,
            "freq_minutes": 5,
            "anomaly_windows": [
                ("2014-07-15 04:35:00", "2014-07-15 13:25:00"),
                ("2014-07-17 05:50:00", "2014-07-18 06:45:00"),
            ],
            "seed": 10,
        },
        "spike",
    ),
]


def build_corpus() -> Dict[str, Dict]:
    """Build the synthetic NAB-like corpus, return dict file_key -> {df, windows}."""
    corpus = {}
    for file_key, kwargs, atype in NAB_LIKE_FILES:
        windows = kwargs["anomaly_windows"]
        df = synth_periodic_with_anomalies(anomaly_type=atype, **kwargs)
        corpus[file_key] = {
            "df": df,
            "windows": [(pd.Timestamp(s), pd.Timestamp(e)) for s, e in windows],
        }
    return corpus
