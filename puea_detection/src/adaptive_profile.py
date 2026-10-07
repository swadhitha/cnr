"""
Adaptive legitimate-PU profile and the update gate.

The profile is a Gaussian model (mean, covariance) of the FC feature vector
z_t (see fc_features.py) for legitimate PU transmissions, plus tracked
summaries (estimated PU power, physical residual) kept for display. An
observation's anomaly score is its squared Mahalanobis distance

    d^2(z) = (z - mu_t)^T Sigma_t^{-1} (z - mu_t)

and it is "profile-anomalous" when d^2 > tau, where tau is learnt from
PU-only calibration data (see :func:`calibrate_profile`).

Updating (EWMA, forgetting factor lambda):

    mu_{t+1}    = (1 - lambda) mu_t + lambda z_t
    Sigma_{t+1} = (1 - lambda) Sigma_t + lambda (z_t - mu_t)(z_t - mu_t)^T

The profile never decides by itself whether to update - that is the job of
:func:`update_allowed`, the gate under study:

* naive gate:     update if the observation is PU-like
* anchored gate:  update if the observation is PU-like AND physically
                  consistent with the known PU location (and verifiable)

The profile never sees labels, attacker information or future slots.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def update_allowed(pu_like: bool, physically_consistent: bool | None, require_physical: bool) -> bool:
    """
    The adaptive update gate.

    pu_like: the classifier stage considers the observation legitimate.
    physically_consistent: result of the physical check (None = unavailable
        / not verifiable). Treated as failing: an observation that cannot be
        verified is never allowed to modify the profile (fail closed).
    require_physical: False = naive adaptation, True = physically anchored.
    """
    if not pu_like:
        return False
    if not require_physical:
        return True
    return bool(physically_consistent) if physically_consistent is not None else False


@dataclass
class AdaptivePUProfile:
    mean: np.ndarray
    cov: np.ndarray
    threshold: float
    ewma_lambda: float
    ridge: float = 1e-6
    feature_names: list = field(default_factory=list)
    n_updates: int = 0
    tracked: dict = field(default_factory=dict)  # EWMA summaries for display only

    def __post_init__(self):
        self.mean = np.asarray(self.mean, dtype=float).copy()
        self.cov = np.asarray(self.cov, dtype=float).copy()
        self.initial_mean = self.mean.copy()
        self.initial_cov = self.cov.copy()
        self._inv0 = self._regularised_inverse(self.initial_cov)
        self._refresh_inverse()

    def _regularised_inverse(self, cov):
        d = cov.shape[0]
        scale = max(np.trace(cov) / d, 1e-12)
        return np.linalg.inv(cov + self.ridge * scale * np.eye(d))

    def _refresh_inverse(self):
        self._inv = self._regularised_inverse(self.cov)

    def copy(self) -> "AdaptivePUProfile":
        p = AdaptivePUProfile(self.initial_mean, self.initial_cov, self.threshold,
                              self.ewma_lambda, self.ridge, list(self.feature_names))
        p.mean, p.cov = self.mean.copy(), self.cov.copy()
        p.n_updates, p.tracked = self.n_updates, dict(self.tracked)
        p._refresh_inverse()
        return p

    def score(self, z) -> np.ndarray:
        """Squared Mahalanobis distance; z is (d,) or (T, d)."""
        diff = np.atleast_2d(z) - self.mean
        d2 = np.einsum("ij,jk,ik->i", diff, self._inv, diff)
        return d2[0] if np.ndim(z) == 1 else d2

    def is_anomalous(self, z):
        return self.score(z) > self.threshold

    def update(self, z, tracked: dict | None = None, weight: float = 1.0) -> None:
        """
        EWMA update with one accepted observation. ``weight`` in [0, 1] scales
        the forgetting factor (soft update: weight = posterior probability that
        the observation is legitimate; 1 = ordinary update).
        """
        z = np.asarray(z, dtype=float)
        lam = self.ewma_lambda * float(np.clip(weight, 0.0, 1.0))
        diff = z - self.mean
        self.mean = self.mean + lam * diff
        self.cov = (1.0 - lam) * self.cov + lam * np.outer(diff, diff)
        self._refresh_inverse()
        self.n_updates += 1
        for k, v in (tracked or {}).items():
            if v is None or not np.isfinite(v):
                continue
            self.tracked[k] = v if k not in self.tracked else (1 - lam) * self.tracked[k] + lam * v

    def displacement(self, reference_mean=None) -> float:
        """Mahalanobis distance (in the *initial* covariance metric) of the current mean from a reference."""
        ref = self.initial_mean if reference_mean is None else np.asarray(reference_mean)
        diff = self.mean - ref
        return float(np.sqrt(diff @ self._inv0 @ diff))

    def state(self) -> dict:
        s = {f"mu_{n}": float(v) for n, v in zip(self.feature_names, self.mean)}
        s.update({f"tracked_{k}": float(v) for k, v in self.tracked.items()})
        s["n_updates"] = self.n_updates
        s["drift_from_initial"] = self.displacement()
        return s

    def to_dict(self) -> dict:
        return {
            "mean": self.mean.tolist(), "cov": self.cov.tolist(), "threshold": self.threshold,
            "ewma_lambda": self.ewma_lambda, "ridge": self.ridge, "feature_names": self.feature_names,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AdaptivePUProfile":
        return cls(np.asarray(d["mean"]), np.asarray(d["cov"]), d["threshold"], d["ewma_lambda"],
                   d.get("ridge", 1e-6), d.get("feature_names", []))


def calibrate_profile(cal_features: np.ndarray, cfg: dict, feature_names=None) -> AdaptivePUProfile:
    """
    Build the initial profile from PU-only calibration features:

    1. robust mean / covariance (MinCovDet) on the first half;
    2. threshold = empirical (1 - calibration_false_alarm) quantile of d^2
       on the held-out second half.
    """
    from sklearn.covariance import MinCovDet

    ap = cfg["adaptive_profile"]
    half = cal_features.shape[0] // 2
    mcd = MinCovDet(random_state=0).fit(cal_features[:half])
    prof = AdaptivePUProfile(mcd.location_, mcd.covariance_, np.inf, float(ap["ewma_lambda"]),
                             float(ap["covariance_ridge"]), list(feature_names or []))
    d2 = prof.score(cal_features[half:])
    prof.threshold = float(np.quantile(d2, 1.0 - float(ap["calibration_false_alarm"]), method="higher"))
    return prof
