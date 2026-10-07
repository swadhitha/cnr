"""
Physical consistency of a multi-SU sensing vector with the known PU location.

Question answered per slot: *could this spatial RSS pattern have been
produced by a transmitter at the legitimate PU's location?* This is a
different question from "does this look PU-like?" (classifier / profile):
a power-matched attacker can reproduce PU-like *statistics* of the RSS
vector, but not the *spatial pattern* across SUs unless it is co-located
with the PU.

Model
-----
For SU i (with SNR above ``min_snr_db`` so the signal estimate is reliable),
the signal estimate in dB is

    s_i = 10 log10( max(E_i - N_i, eps) )     E_i: measured energy, N_i: noise estimate

and under "transmitter at location x":

    s_i = P - 10 * alpha * log10(d_i(x)) + e_i,      e_i ~ N(0, sigma^2)

with P (transmit power) unknown and fitted per slot by least squares:
P_hat(x) = mean_i(s_i + 10 alpha log10 d_i(x)), SSR(x) = sum_i residual_i^2.
alpha is estimated once from PU-only calibration data (by default not per
slot, so a transmitter elsewhere cannot hide by bending alpha).

Statistics
----------
* ``glrt`` (default): Lambda = n * ln( SSR(x_PU) / min_x SSR(x) ), with the
  minimum over a grid covering the area (and x_PU itself). This is the
  generalised likelihood-ratio statistic for "source at x_PU" vs "source
  anywhere"; asymptotically chi^2 with 2 dof under H0. Its *null
  distribution* depends on sigma only weakly (through the grid
  discretisation and the geometry; empirically the 99th percentile moves
  from ~9.8 to ~8.8 as sigma goes 4 -> 8 dB for 40 SUs), so a
  legitimate increase in shadowing variance does not by itself shift the
  calibrated false-rejection rate. (Individual values do change with the
  noise realisation; the invariance is distributional.)
* ``chi2``: SSR(x_PU) / sigma_hat^2 / (n - 1), the reduced chi-square of
  the fit at the PU location. It needs sigma_hat (estimated from
  calibration) and grows when shadowing drifts upward - kept for comparison.

Outputs
-------
``physical_residual``: the selected statistic (larger = less consistent).
``physical_consistency_score``: in [0, 1], the empirical survival
probability of the residual under the calibration distribution of the
legitimate PU (1 = typical of the PU, ~0 = never seen from the PU). It is
not a probability that the transmitter is the PU.

Thresholds are learnt only from PU-only calibration data - see
:func:`calibrate_physical_model`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

EPS_MW = 1e-12


def signal_estimate_db(rss_dbm, noise_est_dbm, min_snr_db: float):
    """
    Noise-subtracted signal estimate (dB) and validity mask.
    SUs whose measured SNR is below ``min_snr_db`` are masked out.
    """
    rss = np.atleast_2d(rss_dbm)
    noise = np.atleast_2d(noise_est_dbm)
    e = np.power(10.0, rss / 10.0)
    n = np.power(10.0, noise / 10.0)
    s = 10.0 * np.log10(np.maximum(e - n, EPS_MW))
    valid = (rss - noise) >= min_snr_db
    return s, valid


def log_distance_matrix(points_xy, su_xy, min_distance: float = 1.0):
    """(G, N) matrix of -10*log10(d) from each candidate point to each SU (alpha applied later)."""
    d = np.linalg.norm(np.asarray(points_xy)[:, None, :] - su_xy[None, :, :], axis=-1)
    return -10.0 * np.log10(np.maximum(d, min_distance))


def weighted_ssr(s, w, x):
    """
    Least-squares fit of s_ti = P_tg + x_gi (P free per slot/point) with 0/1
    weights w_ti. Returns SSR of shape (T, G) and P_hat of shape (T, G).

    Uses sum_i w (s - x - P)^2 = S2 - 2 S*X + W*X^2 - (S1 - W*X)^2 / n,
    all computed as matrix products (no per-slot Python loop).
    """
    w = w.astype(float)
    n = np.maximum(w.sum(axis=1, keepdims=True), 1.0)  # (T,1)
    ws = w * s
    s1 = ws.sum(axis=1, keepdims=True)  # (T,1)
    s2 = (ws * s).sum(axis=1, keepdims=True)  # (T,1)
    sx = ws @ x.T  # (T,G)
    wx = w @ x.T  # (T,G)
    wx2 = w @ (x * x).T  # (T,G)
    u1 = s1 - wx
    ssr = s2 - 2.0 * sx + wx2 - u1 * u1 / n
    return np.maximum(ssr, 1e-9), u1 / n


@dataclass
class PhysicalModel:
    """Calibrated physical model and threshold (serialisable to JSON)."""

    pu_xy: tuple
    alpha: float
    sigma_db: float
    statistic: str
    threshold: float
    min_snr_db: float
    min_valid_sus: int
    grid_step: float
    area_size: float
    min_distance: float
    fit_alpha_per_slot: bool
    calibration_false_alarm: float
    calibration_residuals_sorted: list  # held-out calibration residuals (for the consistency score)
    theoretical_threshold: float  # chi^2 quantile, reported for comparison only

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PhysicalModel":
        return cls(**d)


def _grid(area_size: float, step: float) -> np.ndarray:
    ax = np.arange(0.0, area_size + 1e-9, step)
    gx, gy = np.meshgrid(ax, ax)
    return np.column_stack([gx.ravel(), gy.ravel()])


class PhysicalConsistencyChecker:
    """Computes physical residuals for sensing vectors given a calibrated model."""

    def __init__(self, model: PhysicalModel, su_xy: np.ndarray):
        self.model = model
        self.su_xy = np.asarray(su_xy, dtype=float)
        pu = np.asarray(model.pu_xy, dtype=float)[None, :]
        self._x_pu = log_distance_matrix(pu, self.su_xy, model.min_distance)  # (1,N)
        grid = _grid(model.area_size, model.grid_step)
        self.grid_xy = np.vstack([pu, grid])  # row 0 is the PU itself
        self._x_grid = log_distance_matrix(self.grid_xy, self.su_xy, model.min_distance)
        self._cal = np.asarray(model.calibration_residuals_sorted, dtype=float)

    def compute(self, rss_dbm, noise_est_dbm) -> dict:
        """
        rss_dbm, noise_est_dbm: (N,) or (T, N). Returns a dict of arrays (T,):
        physical_residual, physical_consistency_score, physically_consistent,
        verifiable, n_valid_sus, pu_power_hat_dbm, source_x_hat, source_y_hat,
        glrt, chi2_reduced.
        """
        m = self.model
        s, valid = signal_estimate_db(rss_dbm, noise_est_dbm, m.min_snr_db)
        n = valid.sum(axis=1)
        verifiable = n >= m.min_valid_sus

        if m.fit_alpha_per_slot:
            ssr_pu, p_pu, ssr_all = self._fit_free_alpha(s, valid)
        else:
            ssr_all, p_all = weighted_ssr(s, valid, m.alpha * self._x_grid)
            ssr_pu, p_pu = ssr_all[:, 0], p_all[:, 0]
        best = ssr_all.argmin(axis=1)
        ssr_min = ssr_all[np.arange(len(best)), best]

        n_f = n.astype(float)
        glrt = n_f * np.log(ssr_pu / ssr_min)
        dof = np.maximum(n_f - (2 if m.fit_alpha_per_slot else 1), 1.0)
        chi2_red = ssr_pu / (m.sigma_db ** 2) / dof
        residual = glrt if m.statistic == "glrt" else chi2_red
        residual = np.where(verifiable, residual, np.nan)

        return {
            "physical_residual": residual,
            "physical_consistency_score": self.consistency_score(residual),
            "physically_consistent": verifiable & (residual <= m.threshold),
            "verifiable": verifiable,
            "n_valid_sus": n,
            "pu_power_hat_dbm": p_pu,
            "source_x_hat": self.grid_xy[best, 0],
            "source_y_hat": self.grid_xy[best, 1],
            "glrt": np.where(verifiable, glrt, np.nan),
            "chi2_reduced": np.where(verifiable, chi2_red, np.nan),
        }

    def consistency_score(self, residual) -> np.ndarray:
        """Empirical survival probability of ``residual`` under the PU calibration distribution."""
        r = np.atleast_1d(np.asarray(residual, dtype=float))
        if self._cal.size == 0:
            return np.full(r.shape, np.nan)
        rank = np.searchsorted(self._cal, r, side="left")
        score = 1.0 - rank / self._cal.size
        return np.where(np.isnan(r), 0.0, score)

    def _fit_free_alpha(self, s, valid):
        """Per-slot fit of both P and alpha at every grid point (slower; optional)."""
        T, G = s.shape[0], self._x_grid.shape[0]
        ssr = np.empty((T, G))
        p0 = np.empty(T)
        for t in range(T):
            w = valid[t]
            st = s[t, w]
            x = self._x_grid[:, w]  # (G, n)
            xc = x - x.mean(axis=1, keepdims=True)
            sc = st - st.mean()
            beta = (xc @ sc) / np.maximum((xc * xc).sum(axis=1), 1e-12)
            res = sc[None, :] - beta[:, None] * xc
            ssr[t] = np.maximum((res * res).sum(axis=1), 1e-9)
            p0[t] = st.mean() - beta[0] * x[0].mean()
        return ssr[:, 0], p0, ssr


# ---------------------------------------------------------------------------
# Calibration (PU-only data; never test data)
# ---------------------------------------------------------------------------

def estimate_alpha_sigma(rss_dbm, noise_est_dbm, su_xy, pu_xy, min_snr_db, min_distance=1.0):
    """
    Pooled least-squares estimate of the path-loss exponent and shadowing
    std from PU-only slots, with a free power intercept per slot:
    s_ti = P_t - 10*alpha*log10(d_i) + e_ti.
    """
    s, valid = signal_estimate_db(rss_dbm, noise_est_dbm, min_snr_db)
    x = log_distance_matrix(np.asarray(pu_xy, float)[None, :], np.asarray(su_xy, float), min_distance)[0]
    w = valid.astype(float)
    n = np.maximum(w.sum(axis=1, keepdims=True), 1.0)
    s_c = s - (w * s).sum(axis=1, keepdims=True) / n
    x_c = x[None, :] - (w * x[None, :]).sum(axis=1, keepdims=True) / n
    alpha = float((w * s_c * x_c).sum() / (w * x_c * x_c).sum())
    resid = (s_c - alpha * x_c) * w
    dof = w.sum() - w.shape[0] - 1
    sigma = float(np.sqrt((resid ** 2).sum() / max(dof, 1.0)))
    return alpha, sigma


def calibrate_physical_model(cal_rss, cal_noise_est, su_xy, cfg: dict) -> PhysicalModel:
    """
    Calibrate on PU-only calibration slots:

    1. alpha_hat, sigma_hat from the first half (pooled regression);
    2. compute the statistic on the second (held-out) half;
    3. threshold = empirical (1 - calibration_false_alarm) quantile of those
       held-out residuals (a "higher" quantile, so at most that fraction of
       held-out legitimate slots exceeds it).

    The held-out residuals are also stored to turn a residual into a
    consistency score. The chi^2 quantile is recorded for comparison but
    not used.
    """
    from scipy.stats import chi2

    pc = cfg["physical_consistency"]
    geo = cfg["geometry"]
    half = cal_rss.shape[0] // 2
    alpha, sigma = estimate_alpha_sigma(
        cal_rss[:half], cal_noise_est[:half], su_xy, geo["pu_xy"], pc["min_snr_db"], geo["min_su_distance"]
    )
    fa = float(pc["calibration_false_alarm"])
    model = PhysicalModel(
        pu_xy=tuple(geo["pu_xy"]),
        alpha=alpha,
        sigma_db=sigma,
        statistic=pc["statistic"],
        threshold=np.inf,
        min_snr_db=float(pc["min_snr_db"]),
        min_valid_sus=int(pc["min_valid_sus"]),
        grid_step=float(pc["grid_step"]),
        area_size=float(geo["area_size"]),
        min_distance=float(geo["min_su_distance"]),
        fit_alpha_per_slot=bool(pc["fit_alpha_per_slot"]),
        calibration_false_alarm=fa,
        calibration_residuals_sorted=[],
        theoretical_threshold=float("nan"),
    )
    checker = PhysicalConsistencyChecker(model, su_xy)
    out = checker.compute(cal_rss[half:], cal_noise_est[half:])
    r = out["physical_residual"]
    r = np.sort(r[~np.isnan(r)])
    if r.size == 0:
        raise ValueError("No verifiable calibration slots - check min_snr_db / min_valid_sus.")
    model.threshold = float(np.quantile(r, 1.0 - fa, method="higher"))
    model.calibration_residuals_sorted = r.tolist()
    n_med = float(np.median(out["n_valid_sus"]))
    if model.statistic == "glrt":
        model.theoretical_threshold = float(chi2.ppf(1.0 - fa, 2))
    else:
        dof = n_med - (2 if model.fit_alpha_per_slot else 1)
        model.theoretical_threshold = float(chi2.ppf(1.0 - fa, dof) / dof)
    return model
