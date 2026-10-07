"""
Sequential, drift-invariant location test for the legitimate PU.

The per-slot GLRT in physical_consistency.py asks "could THIS slot have come
from the PU's location?". An attacker a few metres from the PU passes most
single-slot tests (anchor resolution limit). But poisoning an adaptive
profile requires the attacker to transmit in many slots, and each of those
slots carries a small, *consistently signed* location mismatch. This module
accumulates that evidence across slots.

Per-slot location score (P profiled out)
----------------------------------------
With signal estimates s_i and the model s_i = P + alpha * x_i(p) + e_i,
x_i(p) = -10 log10 d_i(p), e_i ~ N(0, sigma^2), the gradient of the model
w.r.t. the source position p, evaluated at the PU, is

    g_i = d(alpha x_i)/dp = -(10 alpha / ln 10) (p_PU - q_i) / d_i^2       (q_i: SU position)

and the efficient score of the location with P profiled out is

    u = (1/sigma^2) sum_i w_i r_i (g_i - g_bar),   r_i = s_i - P_hat - alpha x_i(p_PU)

with Fisher information J = (1/sigma^2) sum_i w_i (g_i - g_bar)(g_i - g_bar)^T.
Under H0 (PU transmitting): E[u] = 0, Cov[u] = J. For a source at
p_PU + delta (small delta): E[u] ~= J delta.

Drift invariance:
* transmit-power drift - P is re-fitted every slot, so u does not change;
* noise-floor drift   - enters through the per-slot noise estimates
  (already subtracted) and the SNR validity mask (handled per slot);
* shadowing drift     - rescales u by 1/sigma; sigma is tracked from the
  residual at the *best-fit* location, which does not depend on where the
  source is, so a displaced single transmitter cannot inflate it.
The PU location and alpha are never adapted.

Standardised score: v_t = J_t^{-1/2} u_t ~ N(0, I_2) under H0.

Sequential test
---------------
Window-limited GLR-CUSUM for a mean shift of unknown direction:

    W_t = max_{t-M < k <= t}  || sum_{tau=k..t} v_tau ||^2 / (2 (t - k + 1))

W_t > h raises the alarm and opens an *alarm episode*. The episode (the
``seq_alarm`` output) stays open until W_t has been <= h for ``window``
consecutive verifiable slots - the same window length, so no extra
parameter. Without this hysteresis the alarm flickers around h under
temporally correlated shadowing; that was found in the first evaluation
and fixed afterwards (docs/adaptive_puea.md section 11, post-hoc change).
``argmax k`` at the start of the episode is the estimated change point. h is set by
Monte Carlo on N(0, I_2) for a target per-slot false-alarm probability. No
test data is used. Before entering the CUSUM, v is

* AR(1)-prewhitened, e_t = (v_t - phi v_{t-1}) / sqrt(1 - phi^2), with phi
  fitted on PU-only calibration data (phi ~ 0 for i.i.d. shadowing; needed
  when shadowing is correlated in time, otherwise sums of v are
  over-dispersed and the false-alarm rate explodes), and
* whitened with the empirical covariance of e on calibration data
  (absorbs model mismatch such as SNR masking and noise-estimate error).

Attribution
-----------
After an alarm, the slots since the estimated change point are a mixture of
PU slots and attacker slots. ``fit_mixture`` fits a two-source model:
each slot is from the PU (prob. 1 - rho) or from one fixed unknown location
x_A (prob. rho), with P profiled out per slot, by maximising the mixture
likelihood over a candidate grid (rho by EM per candidate). Each slot is then
used in two ways:

* profile updates during an alarm are weighted by the slot's posterior
  probability of coming from the PU (the EM-style soft update under the
  mixture), so a resolvable attacker does not stop legitimate drift tracking;
* a slot is flagged if its log-likelihood ratio (fitted location vs PU)
  exceeds the Neyman-Pearson threshold for the configured per-slot false-alarm
  probability on PU slots. Under H0 that LLR is Gaussian with mean -lam_A/2
  and variance lam_A, lam_A being the noncentrality of the fitted location,
  so tau = -lam_A/2 + z_{1-pfa} sqrt(lam_A) (closed form, no tuning).

Analytic predictions (used to check the theory against the simulation)
---------------------------------------------------------------------
* single-slot GLRT rejection: Lambda ~ noncentral chi^2(2, lam(delta)),
  lam(delta) = min_P sum_i (alpha (x_i(p_A) - x_i(p_PU)) - P)^2 / sigma^2
  (exact), ~= delta^T J delta (small delta);
* sequential detection delay: on an attacker slot the standardised score
  has mean mu_v = J^{-1/2} E[u], with ||mu_v||^2 = ``score_noncentrality``
  (closed form from the geometry; equals lam for small delta). Two
  predictions are provided:
  - ``predicted_delay``: first-order drift approximation
    delay ~= h / (rho^2 ||mu_v||^2 (1 - phi) / (2 (1 + phi))) - a scaling
    law that ignores noise in W (so it overestimates the delay);
  - ``predicted_delay_model``: median first-passage time of the same
    GLR-CUSUM run on the idealised score model v_t = n_t + B_t mu_v,
    B_t ~ Bernoulli(rho), n_t unit-variance AR(1)(phi) - geometry, rho, phi
    and h only, no RF simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from physical_consistency import log_distance_matrix, signal_estimate_db, weighted_ssr

LN10 = np.log(10.0)


# ---------------------------------------------------------------------------
# Geometry: gradients, Fisher information, noncentrality
# ---------------------------------------------------------------------------

def location_gradients(pu_xy, su_xy, alpha: float, min_distance: float = 1.0) -> np.ndarray:
    """(N, 2) gradient of alpha * (-10 log10 d_i(p)) w.r.t. p at p = pu_xy."""
    diff = np.asarray(pu_xy, float)[None, :] - np.asarray(su_xy, float)
    d2 = np.maximum((diff ** 2).sum(axis=1), min_distance ** 2)
    return -(10.0 * alpha / LN10) * diff / d2[:, None]


def fisher_information(g: np.ndarray, w: np.ndarray, sigma: float) -> np.ndarray:
    """J for 0/1 weights w; g is (N, 2), w is (N,) or (T, N). Returns (2,2) or (T,2,2)."""
    w2 = np.atleast_2d(w).astype(float)
    n = np.maximum(w2.sum(axis=1, keepdims=True), 1.0)
    gbar = (w2 @ g) / n  # (T,2)
    m2 = np.einsum("ti,ij,ik->tjk", w2, g, g)  # sum w g g^T
    J = (m2 - n[:, :, None] * np.einsum("tj,tk->tjk", gbar, gbar)) / sigma ** 2
    return J[0] if np.ndim(w) == 1 else J


def noncentrality(att_xy, pu_xy, su_xy, alpha: float, sigma: float, min_distance: float = 1.0,
                  w=None) -> float:
    """Exact single-slot GLRT noncentrality of a source at att_xy (P profiled out)."""
    su_xy = np.asarray(su_xy, float)
    xa = log_distance_matrix(np.atleast_2d(att_xy), su_xy, min_distance)[0]
    xp = log_distance_matrix(np.atleast_2d(pu_xy), su_xy, min_distance)[0]
    diff = alpha * (xa - xp)
    w = np.ones(len(su_xy)) if w is None else np.asarray(w, float)
    c = diff - (w * diff).sum() / w.sum()
    return float((w * c * c).sum() / sigma ** 2)


def score_noncentrality(att_xy, pu_xy, su_xy, alpha: float, sigma: float, min_distance: float = 1.0) -> float:
    """||J^{-1/2} E[u]||^2 for a source at att_xy (all SUs valid, P profiled out)."""
    su_xy = np.asarray(su_xy, float)
    xa = log_distance_matrix(np.atleast_2d(att_xy), su_xy, min_distance)[0]
    xp = log_distance_matrix(np.atleast_2d(pu_xy), su_xy, min_distance)[0]
    diff = alpha * (xa - xp)
    diff = diff - diff.mean()
    g = location_gradients(pu_xy, su_xy, alpha, min_distance)
    J = fisher_information(g, np.ones(len(su_xy)), sigma)
    eu = (g - g.mean(axis=0)).T @ diff / sigma ** 2
    return float(eu @ np.linalg.solve(J, eu))


def predicted_delay_model(mu2: float, rho: float, threshold: float, window: int, phi: float = 0.0,
                          horizon: int = 3000, n_rep: int = 200, seed: int = 0) -> float:
    """Median first-passage time of the window-limited GLR-CUSUM on the idealised score model."""
    rng = np.random.default_rng(seed)
    mu = np.array([np.sqrt(max(mu2, 0.0)), 0.0])
    z = rng.standard_normal((n_rep, horizon + 1, 2))
    n = np.empty_like(z)
    n[:, 0] = z[:, 0]
    for t in range(1, horizon + 1):
        n[:, t] = phi * n[:, t - 1] + np.sqrt(1.0 - phi ** 2) * z[:, t]
    v = n + (rng.uniform(size=(n_rep, horizon + 1, 1)) < rho) * mu
    e = (v[:, 1:] - phi * v[:, :-1]) / np.sqrt(1.0 - phi ** 2)
    cs = np.concatenate([np.zeros((n_rep, 1, 2)), np.cumsum(e, axis=1)], axis=1)
    first = np.full(n_rep, np.inf)
    for t in range(horizon):
        L = np.arange(1, min(t + 1, window) + 1)
        sums = cs[:, t + 1, None, :] - cs[:, t + 1 - L, :]
        w = ((sums ** 2).sum(axis=2) / (2.0 * L)).max(axis=1)
        newly = (w > threshold) & ~np.isfinite(first)
        first[newly] = t
        if np.isfinite(first).all():
            break
    return float(np.median(first))


def predicted_rejection(lam, threshold: float) -> np.ndarray:
    """P(noncentral chi^2(2, lam) > threshold)."""
    from scipy.stats import ncx2, chi2

    lam = np.asarray(lam, float)
    out = np.where(lam > 1e-12, ncx2.sf(threshold, 2, np.maximum(lam, 1e-12)), chi2.sf(threshold, 2))
    return out


def predicted_delay(mu2: float, rho: float, threshold: float, phi: float = 0.0) -> float:
    """First-order (drift-only) prediction of the sequential detection delay (slots)."""
    drift = 0.5 * rho ** 2 * mu2 * (1.0 - phi) / (1.0 + phi)
    return float(threshold / drift) if drift > 0 else np.inf


# ---------------------------------------------------------------------------
# Per-slot scores
# ---------------------------------------------------------------------------

def slot_scores(rss_dbm, noise_est_dbm, su_xy, pu_xy, alpha: float, min_snr_db: float,
                min_distance: float, area_size: float, grid_step: float):
    """
    Unscaled per-slot quantities (sigma = 1):
    u0 (T,2), J0 (T,2,2), sigma2_slot (T,) - residual variance at the best-fit
    location (location-invariant), the signal estimates s (T,N) and the
    validity mask (T,N).
    """
    su_xy = np.asarray(su_xy, float)
    s, valid = signal_estimate_db(rss_dbm, noise_est_dbm, min_snr_db)
    w = valid.astype(float)
    n = np.maximum(w.sum(axis=1), 1.0)
    x_pu = log_distance_matrix(np.atleast_2d(pu_xy), su_xy, min_distance)[0]
    r = s - alpha * x_pu[None, :]
    p_hat = (w * r).sum(axis=1) / n
    r = (r - p_hat[:, None]) * w
    g = location_gradients(pu_xy, su_xy, alpha, min_distance)
    u0 = r @ g  # sum w r g  (sum w r = 0, so centring g is implicit)
    J0 = fisher_information(g, valid, 1.0)

    ax = np.arange(0.0, area_size + 1e-9, grid_step)
    gx, gy = np.meshgrid(ax, ax)
    grid = np.vstack([np.atleast_2d(pu_xy), np.column_stack([gx.ravel(), gy.ravel()])])
    ssr, _ = weighted_ssr(s, valid, alpha * log_distance_matrix(grid, su_xy, min_distance))
    sigma2 = ssr.min(axis=1) / np.maximum(n - 3.0, 1.0)
    return u0, J0, sigma2, s, valid


def _inv_sqrt_2x2(J: np.ndarray) -> np.ndarray:
    """Batched symmetric inverse square root of (T,2,2) matrices."""
    vals, vecs = np.linalg.eigh(J)
    vals = np.maximum(vals, 1e-12)
    return np.einsum("tij,tj,tkj->tik", vecs, 1.0 / np.sqrt(vals), vecs)


def standardise(u0, J0, sigma2) -> np.ndarray:
    """v = J^{-1/2} u with J = J0/sigma^2, u = u0/sigma^2  ->  v = J0^{-1/2} u0 / sigma."""
    v = np.einsum("tij,tj->ti", _inv_sqrt_2x2(J0), u0)
    return v / np.sqrt(np.asarray(sigma2, float))[:, None]


# ---------------------------------------------------------------------------
# GLR-CUSUM
# ---------------------------------------------------------------------------

def glr_statistic(v_window: np.ndarray):
    """W and argmax start index for the window (rows = slots, oldest first)."""
    m = v_window.shape[0]
    rev = np.cumsum(v_window[::-1], axis=0)  # sums over the last 1..m slots
    lengths = np.arange(1, m + 1)
    w = (rev ** 2).sum(axis=1) / (2.0 * lengths)
    j = int(np.argmax(w))
    return float(w[j]), m - 1 - j  # index (within window) where the change starts


def calibrate_threshold(pfa_per_slot: float, window: int, n_slots: int = 200_000, seed: int = 0) -> float:
    """
    h such that the probability that W_t > h in a given slot under H0
    (v ~ N(0, I_2) i.i.d.) is pfa_per_slot. Monte Carlo on one long stream.
    """
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n_slots + window, 2))
    cs = np.vstack([np.zeros(2), np.cumsum(v, axis=0)])
    best = np.zeros(n_slots)
    for L in range(1, window + 1):
        t = np.arange(window, window + n_slots)
        s = cs[t + 1] - cs[t + 1 - L]
        best = np.maximum(best, (s ** 2).sum(axis=1) / (2.0 * L))
    return float(np.quantile(best, 1.0 - pfa_per_slot))


def candidate_grid(pu_xy, area_size: float, coarse_step: float, local_radius: float, local_step: float,
                   min_offset: float) -> np.ndarray:
    """Candidate attacker locations: coarse grid over the area + fine grid near the PU (PU itself excluded)."""
    ax = np.arange(0.0, area_size + 1e-9, coarse_step)
    gx, gy = np.meshgrid(ax, ax)
    coarse = np.column_stack([gx.ravel(), gy.ravel()])
    loc = np.arange(-local_radius, local_radius + 1e-9, local_step)
    lx, ly = np.meshgrid(loc, loc)
    local = np.column_stack([lx.ravel(), ly.ravel()]) + np.asarray(pu_xy, float)
    local = local[np.hypot(*(local - pu_xy).T) <= local_radius]
    pts = np.vstack([coarse, local])
    pts = pts[np.all((pts >= 0) & (pts <= area_size), axis=1)]
    return pts[np.hypot(*(pts - np.asarray(pu_xy, float)).T) >= min_offset]


def fit_mixture(s, valid, sigma2: float, pu_xy, su_xy, alpha: float, min_distance: float, cand: np.ndarray,
                n_iter: int = 30) -> dict:
    """
    Two-source mixture over the given slots: PU (prob 1 - rho) vs one fixed
    location x_A (prob rho). Returns x_A, rho, the log-likelihood gain over
    "PU only", and the per-slot posteriors.
    """
    pts = np.vstack([np.atleast_2d(pu_xy), cand])
    ssr, _ = weighted_ssr(s, valid, alpha * log_distance_matrix(pts, np.asarray(su_xy, float), min_distance))
    llr = np.clip(-(ssr[:, 1:] - ssr[:, :1]) / (2.0 * sigma2), -50.0, 50.0)  # (T,G) log LR attacker vs PU
    lr = np.exp(llr)
    rho = np.full(lr.shape[1], 0.1)
    for _ in range(n_iter):
        post = rho * lr / (1.0 - rho + rho * lr)
        rho = np.clip(post.mean(axis=0), 1e-4, 1 - 1e-4)
    ll = np.log(1.0 - rho + rho * lr).sum(axis=0)
    g = int(np.argmax(ll))
    post = rho[g] * lr[:, g] / (1.0 - rho[g] + rho[g] * lr[:, g])
    lam_a = noncentrality(cand[g], pu_xy, su_xy, alpha, np.sqrt(sigma2), min_distance)
    return {"x_a": cand[g].copy(), "rho": float(rho[g]), "ll_gain": float(ll[g]), "posterior": post,
            "lam_a": lam_a}


def np_llr_threshold(lam_a: float, pfa: float) -> float:
    """Neyman-Pearson LLR threshold for per-slot false-alarm probability pfa on PU slots."""
    from scipy.stats import norm

    return float(-0.5 * lam_a + norm.ppf(1.0 - pfa) * np.sqrt(max(lam_a, 1e-12)))


def slot_posterior(s, valid, sigma2: float, fit: dict, pu_xy, su_xy, alpha: float, min_distance: float):
    """(posterior, llr) that a single slot came from the fitted attacker location rather than the PU."""
    pts = np.vstack([np.atleast_2d(pu_xy), fit["x_a"][None, :]])
    ssr, _ = weighted_ssr(np.atleast_2d(s), np.atleast_2d(valid),
                          alpha * log_distance_matrix(pts, np.asarray(su_xy, float), min_distance))
    llr = float(np.clip(-(ssr[0, 1] - ssr[0, 0]) / (2.0 * sigma2), -50.0, 50.0))
    lr, rho = np.exp(llr), fit["rho"]
    return float(rho * lr / (1.0 - rho + rho * lr)), llr


_THRESHOLD_CACHE: dict = {}


@dataclass
class SequentialCalibration:
    phi: float  # AR(1) coefficient of v on calibration data
    whitening: np.ndarray  # (2,2): e_white = whitening @ e
    threshold: float
    window: int
    pfa_per_slot: float
    sigma2_ref: float
    sigma_lambda: float
    cal_e_cov: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"phi": self.phi, "whitening": self.whitening.tolist(), "threshold": self.threshold,
                "window": self.window, "pfa_per_slot": self.pfa_per_slot, "sigma2_ref": self.sigma2_ref,
                "sigma_lambda": self.sigma_lambda, "cal_e_cov": self.cal_e_cov}


def calibrate_sequential(cal_rss, cal_noise, su_xy, phys_model, cfg: dict) -> SequentialCalibration:
    """
    PU-only calibration (first half of the calibration stream): sigma
    reference, AR(1) coefficient phi of v and the empirical covariance of the
    prewhitened score (whitening). Threshold by Monte Carlo for the
    configured per-slot false-alarm probability.
    """
    sc = cfg["sequential"]
    u0, J0, s2, _, valid = slot_scores(cal_rss, cal_noise, su_xy, phys_model.pu_xy, phys_model.alpha,
                                       phys_model.min_snr_db, phys_model.min_distance, phys_model.area_size,
                                       phys_model.grid_step)
    ok = valid.sum(axis=1) >= phys_model.min_valid_sus
    half = len(u0) // 2
    sel = ok & (np.arange(len(u0)) < half)
    sigma2_ref = float(np.median(s2[sel]))
    v = standardise(u0[sel], J0[sel], np.full(sel.sum(), sigma2_ref))
    phi = float(np.clip((v[1:] * v[:-1]).sum() / (v[:-1] ** 2).sum(), 0.0, 0.99))
    e = (v[1:] - phi * v[:-1]) / np.sqrt(1.0 - phi ** 2)
    cov = np.cov(e.T)
    vals, vecs = np.linalg.eigh(cov)
    whitening = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T
    key = (float(sc["pfa_per_slot"]), int(sc["window"]), int(sc.get("mc_seed", 0)))
    if key not in _THRESHOLD_CACHE:
        _THRESHOLD_CACHE[key] = calibrate_threshold(*key[:2], seed=key[2])
    return SequentialCalibration(phi, whitening, _THRESHOLD_CACHE[key], int(sc["window"]),
                                 float(sc["pfa_per_slot"]), sigma2_ref, float(sc["sigma_lambda"]), cov.tolist())


class SequentialSpatialMonitor:
    """
    Causal per-slot monitor. ``step`` takes the slot's unscaled score and
    returns W_t, the alarm flag and the estimated change-point slot. It keeps
    the last ``window`` verifiable slots (prewhitened scores, signal
    estimates, validity masks) for the GLR statistic and for attribution.
    sigma^2 is tracked by EWMA from the location-invariant best-fit residual,
    only on slots that are not in alarm.
    """

    def __init__(self, cal: SequentialCalibration, n_sus: int):
        self.cal = cal
        self.sigma2 = cal.sigma2_ref
        m = cal.window
        self.e = np.zeros((m, 2))
        self.s = np.zeros((m, n_sus))
        self.valid = np.zeros((m, n_sus), dtype=bool)
        self.slot_of = np.zeros(m, dtype=int)
        self.n = 0
        self.v_prev = None
        self.in_episode = False
        self.clear_run = 0
        self.episode_change = -1

    def _order(self):
        m = min(self.n, self.cal.window)
        return np.arange(self.n - m, self.n) % self.cal.window

    def window_since(self, slot: int):
        """(s, valid) of buffered slots with slot index >= ``slot`` (oldest first)."""
        o = self._order()
        keep = o[self.slot_of[o] >= slot]
        return self.s[keep], self.valid[keep]

    def step(self, t: int, u0, J0, sigma2_slot: float, s, valid, verifiable: bool) -> dict:
        if not verifiable:
            return {"seq_stat": np.nan, "seq_alarm": self.in_episode, "seq_raw_alarm": False,
                    "seq_change_slot": self.episode_change, "seq_episode_start": False}
        v = standardise(np.asarray(u0)[None], np.asarray(J0)[None], np.array([self.sigma2]))[0]
        phi = self.cal.phi
        prev = self.v_prev if self.v_prev is not None else phi * v  # first slot: no prewhitening memory
        e = self.cal.whitening @ ((v - phi * prev) / np.sqrt(1.0 - phi ** 2))
        self.v_prev = v
        k = self.n % self.cal.window
        self.e[k], self.s[k], self.valid[k], self.slot_of[k] = e, s, valid, t
        self.n += 1
        o = self._order()
        w, j = glr_statistic(self.e[o])
        raw = w > self.cal.threshold
        started = False
        if raw:
            self.clear_run = 0
            if not self.in_episode:
                self.in_episode, started = True, True
                self.episode_change = int(self.slot_of[o[j]])
        elif self.in_episode:
            self.clear_run += 1
            if self.clear_run >= self.cal.window:
                self.in_episode, self.episode_change = False, -1
        if not self.in_episode and np.isfinite(sigma2_slot):
            lam = self.cal.sigma_lambda
            self.sigma2 = (1 - lam) * self.sigma2 + lam * sigma2_slot
        return {"seq_stat": w, "seq_alarm": self.in_episode, "seq_raw_alarm": bool(raw),
                "seq_change_slot": self.episode_change, "seq_episode_start": started}
