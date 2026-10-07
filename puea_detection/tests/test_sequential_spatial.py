"""
Unit tests for sequential_spatial.py.

Run from the repository root:
    python -m unittest discover -s puea_detection/tests -v
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from physical_consistency import log_distance_matrix  # noqa: E402
from sequential_spatial import (  # noqa: E402
    SequentialCalibration,
    SequentialSpatialMonitor,
    calibrate_threshold,
    candidate_grid,
    fisher_information,
    fit_mixture,
    glr_statistic,
    location_gradients,
    noncentrality,
    np_llr_threshold,
    slot_posterior,
    score_noncentrality,
    slot_scores,
    standardise,
)

ALPHA = 3.0
SIGMA = 4.0
PU = np.array([15.0, 85.0])
NOISE_DBM = -120.0  # far below the signal, so the noise-subtracted estimate is exact


def geometry(n=40, seed=0):
    return np.random.default_rng(seed).uniform(0, 100, size=(n, 2))


def rss(tx_xy, power_dbm, su_xy, rng, sigma=SIGMA, n_slots=1):
    x = log_distance_matrix(np.atleast_2d(tx_xy), su_xy, 1.0)[0]
    return power_dbm + ALPHA * x[None, :] + sigma * rng.standard_normal((n_slots, len(su_xy)))


def scores(r, su_xy):
    noise = np.full_like(r, NOISE_DBM)
    return slot_scores(r, noise, su_xy, PU, ALPHA, 3.0, 1.0, 100.0, 2.5)


class TestScore(unittest.TestCase):
    def test_power_drift_invariance(self):
        """A common power offset (legitimate power drift) leaves the location score unchanged."""
        su = geometry()
        r = rss(PU, 30.0, su, np.random.default_rng(1), n_slots=5)
        u_a, _, s2_a, _, _ = scores(r, su)
        u_b, _, s2_b, _, _ = scores(r + 6.0, su)
        np.testing.assert_allclose(u_a, u_b, atol=1e-6)
        np.testing.assert_allclose(s2_a, s2_b, rtol=1e-6)

    def test_standardised_score_is_standard_normal_under_h0(self):
        su = geometry()
        r = rss(PU, 30.0, su, np.random.default_rng(2), n_slots=4000)
        u0, J0, _, _, _ = scores(r, su)
        v = standardise(u0, J0, np.full(len(u0), SIGMA ** 2))
        np.testing.assert_allclose(v.mean(axis=0), 0.0, atol=0.06)
        np.testing.assert_allclose(np.cov(v.T), np.eye(2), atol=0.08)

    def test_score_mean_matches_noncentrality_for_attacker(self):
        """E||v||^2 on attacker slots ~= 2 + score noncentrality."""
        su = geometry()
        att = PU + np.array([3.0, -2.0])
        r = rss(att, 30.0, su, np.random.default_rng(3), n_slots=4000)
        u0, J0, _, _, _ = scores(r, su)
        v = standardise(u0, J0, np.full(len(u0), SIGMA ** 2))
        mu2 = score_noncentrality(att, PU, su, ALPHA, SIGMA)
        self.assertAlmostEqual(float((v.mean(axis=0) ** 2).sum()), mu2, delta=0.1 * mu2 + 0.05)

    def test_fisher_matches_exact_noncentrality_for_small_offsets(self):
        su = geometry()
        g = location_gradients(PU, su, ALPHA)
        J = fisher_information(g, np.ones(len(su)), SIGMA)
        delta = np.array([0.5, 0.3])
        lam = noncentrality(PU + delta, PU, su, ALPHA, SIGMA)
        self.assertAlmostEqual(float(delta @ J @ delta), lam, delta=0.05 * lam)
        self.assertAlmostEqual(score_noncentrality(PU + delta, PU, su, ALPHA, SIGMA), lam, delta=0.05 * lam)

    def test_location_invariant_sigma_estimate(self):
        """The best-fit residual variance does not grow when the source moves."""
        su = geometry()
        rng = np.random.default_rng(4)
        _, _, s2_pu, _, _ = scores(rss(PU, 30.0, su, rng, n_slots=2000), su)
        _, _, s2_att, _, _ = scores(rss(np.array([60.0, 40.0]), 30.0, su, rng, n_slots=2000), su)
        self.assertAlmostEqual(np.median(s2_pu) / np.median(s2_att), 1.0, delta=0.1)


class TestGLR(unittest.TestCase):
    def test_glr_statistic_and_change_point(self):
        v = np.zeros((10, 2))
        v[6:] = [2.0, 0.0]
        w, j = glr_statistic(v)
        self.assertEqual(j, 6)
        self.assertAlmostEqual(w, (4 * 2.0) ** 2 / (2 * 4))

    def test_threshold_false_alarm_rate(self):
        h = calibrate_threshold(0.01, 50, n_slots=40_000, seed=1)
        rng = np.random.default_rng(7)
        v = rng.standard_normal((40_050, 2))
        cs = np.vstack([np.zeros(2), np.cumsum(v, axis=0)])
        best = np.zeros(40_000)
        t = np.arange(50, 40_050)
        for L in range(1, 51):
            s = cs[t + 1] - cs[t + 1 - L]
            best = np.maximum(best, (s ** 2).sum(axis=1) / (2.0 * L))
        self.assertAlmostEqual(float((best > h).mean()), 0.01, delta=0.004)

    def test_monitor_alarms_on_near_pu_attacker_not_on_pu(self):
        su = geometry()
        rng = np.random.default_rng(5)
        cal = SequentialCalibration(0.0, np.eye(2), calibrate_threshold(1e-3, 200, n_slots=50_000), 200, 1e-3,
                                    SIGMA ** 2, 0.01)
        pu = rss(PU, 30.0, su, rng, n_slots=600)
        att = rss(PU + np.array([4.0, -3.0]), 30.0, su, rng, n_slots=600)
        is_att = rng.uniform(size=600) < 0.3
        stream = np.vstack([pu[:300], np.where(is_att[300:, None], att[300:], pu[300:])])
        u0, J0, s2, s, valid = scores(stream, su)
        mon = SequentialSpatialMonitor(cal, len(su))
        alarms = np.array([mon.step(t, u0[t], J0[t], s2[t], s[t], valid[t], True)["seq_alarm"]
                           for t in range(600)])
        self.assertFalse(alarms[:300].any())
        self.assertTrue(alarms[300:].any())


    def test_alarm_episode_hysteresis(self):
        """An episode opens on one exceedance and closes only after `window` clear slots."""
        cal = SequentialCalibration(0.0, np.eye(2), 5.0, 4, 1e-3, 1.0, 0.01)
        mon = SequentialSpatialMonitor(cal, 3)
        eye, z = np.eye(2), np.zeros(3)
        u = [np.zeros(2)] * 3 + [np.array([4.0, 0.0])] + [np.zeros(2)] * 10
        out = [mon.step(t, u[t], eye, 1.0, z, z.astype(bool), True) for t in range(len(u))]
        raw = [o["seq_raw_alarm"] for o in out]
        ep = [o["seq_alarm"] for o in out]
        self.assertEqual(raw.index(True), 3)
        self.assertTrue(out[3]["seq_episode_start"])
        last_raw = max(i for i, r in enumerate(raw) if r)
        self.assertTrue(all(ep[3:last_raw + cal.window]))
        self.assertFalse(ep[last_raw + cal.window])  # closes on the window-th clear slot
        self.assertEqual(sum(o["seq_episode_start"] for o in out), 1)


class TestAttribution(unittest.TestCase):
    def test_mixture_recovers_attacker_location_and_rate(self):
        su = geometry()
        rng = np.random.default_rng(6)
        att_xy = PU + np.array([6.0, -4.0])
        n = 400
        is_att = rng.uniform(size=n) < 0.3
        r = np.where(is_att[:, None], rss(att_xy, 30.0, su, rng, n_slots=n), rss(PU, 30.0, su, rng, n_slots=n))
        _, _, _, s, valid = scores(r, su)
        cand = candidate_grid(PU, 100.0, 2.5, 15.0, 1.0, 0.5)
        fit = fit_mixture(s, valid, SIGMA ** 2, PU, su, ALPHA, 1.0, cand)
        self.assertLess(np.linalg.norm(fit["x_a"] - att_xy), 3.0)
        self.assertAlmostEqual(fit["rho"], is_att.mean(), delta=0.1)
        pred = fit["posterior"] > 0.5
        self.assertGreater((pred == is_att).mean(), 0.8)

    def test_np_threshold_controls_false_alarms_on_pu_slots(self):
        su = geometry()
        rng = np.random.default_rng(8)
        fit = {"x_a": PU + np.array([5.0, 0.0]), "rho": 0.3}
        lam_a = noncentrality(fit["x_a"], PU, su, ALPHA, SIGMA)
        tau = np_llr_threshold(lam_a, 0.01)
        _, _, _, s, valid = scores(rss(PU, 30.0, su, rng, n_slots=4000), su)
        llr = np.array([slot_posterior(s[t], valid[t], SIGMA ** 2, fit, PU, su, ALPHA, 1.0)[1] for t in range(4000)])
        self.assertAlmostEqual(float((llr > tau).mean()), 0.01, delta=0.005)

    def test_candidate_grid_excludes_pu(self):
        cand = candidate_grid(PU, 100.0, 2.5, 15.0, 1.0, 0.5)
        self.assertGreaterEqual(np.hypot(*(cand - PU).T).min(), 0.5)
        self.assertTrue(np.all((cand >= 0) & (cand <= 100)))


class TestRollbackContamination(unittest.TestCase):
    def test_weighted_update_scales_contamination(self):
        from run_adaptive_experiment import _evaluation_columns
        from adaptive_profile import AdaptivePUProfile
        from simulation import load_config

        lam = 0.5
        rows = pd.DataFrame({"rollback_to_slot": [-1, -1], "update_weight": [1.0, 0.2]})
        prof = AdaptivePUProfile(np.zeros(6), np.eye(6), 1.0, lam)
        out = _evaluation_columns(rows, np.array([0, 1]), np.zeros((2, 6)), np.zeros((2, 6)),
                                  np.array([True, True]), lam, load_config(), prof)
        self.assertAlmostEqual(out["profile_contamination"].iloc[1], 0.1)

    def test_contamination_returns_to_checkpoint_value(self):
        from run_adaptive_experiment import _evaluation_columns
        from adaptive_profile import AdaptivePUProfile
        from simulation import load_config

        n, lam = 6, 0.5
        labels = np.array([0, 1, 1, 1, 0, 0])
        accepted = np.array([True, True, True, False, False, True])
        rows = pd.DataFrame({"rollback_to_slot": [-1, -1, -1, 0, -1, -1]})
        prof = AdaptivePUProfile(np.zeros(6), np.eye(6), 1.0, lam)
        out = _evaluation_columns(rows, labels, np.zeros((n, 6)), np.zeros((n, 6)), accepted, lam,
                                  load_config(), prof)
        c = out["profile_contamination"].to_numpy()
        np.testing.assert_allclose(c[:3], [0.0, 0.5, 0.75])
        self.assertEqual(c[3], 0.0)  # rolled back to the value after slot 0
        self.assertEqual(c[5], 0.0)  # clean update afterwards keeps it at 0


if __name__ == "__main__":
    unittest.main()
