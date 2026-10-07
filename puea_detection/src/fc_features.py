"""
Fusion-center (FC) features for the adaptive-PUEA experiments.

Input: the per-SU sensing vector of a slot (RSS in dBm and each SU's noise
floor estimate). Output: the five statistics used by Chhetry & Marchang
(2021) - mean, variance, median, upper and lower quartile of the RSS vector -
plus the mean SNR. These are permutation-invariant: they discard *which* SU
saw *what*, which is exactly the spatial information the physical
consistency check (physical_consistency.py) uses.

``spatial_features`` returns the raw per-SU RSS vector, used only by the
reference "Spatial-XGBoost" system, which tests whether a classifier given
SU identity can learn location implicitly.
"""

from __future__ import annotations

import numpy as np

FEATURE_NAMES = ["rss_mean", "rss_var", "rss_median", "rss_q75", "rss_q25", "snr_mean"]


def fc_features(rss_dbm: np.ndarray, noise_est_dbm: np.ndarray) -> np.ndarray:
    """
    rss_dbm, noise_est_dbm: (N,) for one slot or (T, N) for a batch.
    Returns (6,) or (T, 6) in the order of ``FEATURE_NAMES``.
    """
    rss = np.atleast_2d(rss_dbm)
    snr = rss - np.atleast_2d(noise_est_dbm)
    q25, med, q75 = np.percentile(rss, [25, 50, 75], axis=1)
    out = np.column_stack([
        rss.mean(axis=1),
        rss.var(axis=1),
        med,
        q75,
        q25,
        snr.mean(axis=1),
    ])
    return out[0] if np.ndim(rss_dbm) == 1 else out


def spatial_features(rss_dbm: np.ndarray) -> np.ndarray:
    """Raw per-SU RSS (SU identity preserved)."""
    return np.asarray(rss_dbm, dtype=float)
