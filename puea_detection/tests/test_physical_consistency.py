"""
Unit tests for physical_consistency.py.

Run from the repository root:
    python -m unittest discover -s puea_detection/tests -v
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from physical_consistency import (  # noqa: E402
    PhysicalConsistencyChecker,
    PhysicalModel,
    estimate_alpha_sigma,
    log_distance_matrix,
    signal_estimate_db,
    weighted_ssr,
)

ALPHA = 3.0
PU = np.array([15.0, 85.0])


def make_model(threshold=10.0, statistic="glrt", cal=None):
    return PhysicalModel(
        pu_xy=tuple(PU), alpha=ALPHA, sigma_db=4.0, statistic=statistic, threshold=threshold,
        min_snr_db=3.0, min_valid_sus=8, grid_step=2.5, area_size=100.0, min_distance=1.0,
        fit_alpha_per_slot=False, calibration_false_alarm=0.01,
        calibration_residuals_sorted=sorted(cal if cal is not None else np.linspace(0, 10, 101).tolist()),
        theoretical_threshold=9.21,
    )


def rss_from(tx_xy, power_dbm, su_xy, shadow=None, noise_dbm=-120.0):
    """Noise-free-ish RSS vector from the path-loss model (noise far below signal)."""
    d = np.maximum(np.linalg.norm(su_xy - tx_xy, axis=1), 1.0)
    pr = power_dbm - 10 * ALPHA * np.log10(d) + (0 if shadow is None else shadow)
    rss = 10 * np.log10(10 ** (pr / 10) + 10 ** (noise_dbm / 10))
    return rss, np.full_like(rss, noise_dbm)


class TestWeightedSSR(unittest.TestCase):
    def test_matches_direct_least_squares(self):
        rng = np.random.default_rng(0)
        s = rng.normal(size=(3, 12))
        x = rng.normal(size=(5, 12))
        w = (rng.uniform(size=(3, 12)) > 0.3).astype(float)
        ssr, p = weighted_ssr(s, w, x)
        for t in range(3):
            m = w[t] > 0
            for g in range(5):
                u = s[t, m] - x[g, m]
                self.assertAlmostEqual(p[t, g], u.mean(), places=8)
                self.assertAlmostEqual(ssr[t, g], ((u - u.mean()) ** 2).sum(), places=6)

    def test_exact_fit_gives_near_zero_ssr(self):
        x = np.array([[0.0, -3.0, -7.0, -10.0]])
        s = x + 42.0
        ssr, p = weighted_ssr(s, np.ones_like(s), x)
        self.assertLess(ssr[0, 0], 1e-6)
        self.assertAlmostEqual(p[0, 0], 42.0, places=8)


class TestSignalEstimate(unittest.TestCase):
    def test_noise_subtraction_and_mask(self):
        # 3 dB above noise => signal equal to noise power => s == noise_dbm
        rss = np.array([-47.0 + 10 * np.log10(2), -49.0])
        noise = np.array([-47.0, -50.0])
        s, valid = signal_estimate_db(rss, noise, min_snr_db=3.0)
        self.assertAlmostEqual(s[0, 0], -47.0, places=6)
        self.assertTrue(valid[0, 0])
        self.assertFalse(valid[0, 1])


class TestChecker(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.su = rng.uniform(0, 100, size=(40, 2))
        self.checker = PhysicalConsistencyChecker(make_model(), self.su)

    def test_pu_without_shadowing_is_consistent(self):
        rss, noise = rss_from(PU, 30.0, self.su)
        out = self.checker.compute(rss, noise)
        self.assertTrue(out["verifiable"][0])
        self.assertLess(out["physical_residual"][0], 1e-3)
        self.assertTrue(out["physically_consistent"][0])
        self.assertAlmostEqual(out["pu_power_hat_dbm"][0], 30.0, places=3)

    def test_pu_power_change_is_absorbed(self):
        """Legitimate power drift must not change the residual (P is fitted per slot)."""
        a = self.checker.compute(*rss_from(PU, 30.0, self.su))["physical_residual"][0]
        b = self.checker.compute(*rss_from(PU, 36.0, self.su))["physical_residual"][0]
        self.assertAlmostEqual(a, b, places=4)

    def test_displaced_power_matched_transmitter_is_inconsistent(self):
        att = np.array([75.0, 25.0])
        d_att = np.maximum(np.linalg.norm(self.su - att, axis=1), 1.0)
        d_pu = np.maximum(np.linalg.norm(self.su - PU, axis=1), 1.0)
        matched = 30.0 + 10 * ALPHA * (np.log10(d_att).mean() - np.log10(d_pu).mean())
        rss, noise = rss_from(att, matched, self.su)
        # the mean RSS matches the PU's ...
        rss_pu, _ = rss_from(PU, 30.0, self.su)
        self.assertAlmostEqual(rss.mean(), rss_pu.mean(), places=3)
        # ... but the spatial pattern does not
        out = self.checker.compute(rss, noise)
        self.assertGreater(out["physical_residual"][0], 10.0)
        self.assertFalse(out["physically_consistent"][0])
        # and the best-fit grid point is near the attacker
        est = np.array([out["source_x_hat"][0], out["source_y_hat"][0]])
        self.assertLess(np.linalg.norm(est - att), 5.0)

    def test_glrt_null_distribution_is_sigma_free(self):
        """
        Under H0 the upper tail of the GLRT (what the threshold uses) stays
        roughly stable when shadowing std doubles, whereas the chi^2
        statistic would grow ~4x. (At very small sigma the grid
        discretisation makes the GLRT exactly 0 more often - not tested.)
        """
        rng = np.random.default_rng(2)

        def q99(sigma):
            rows = [rss_from(PU, 30.0, self.su, shadow=rng.normal(0, sigma, 40)) for _ in range(1000)]
            g = self.checker.compute(np.vstack([r for r, _ in rows]), np.vstack([n for _, n in rows]))["glrt"]
            return np.quantile(g, 0.99)

        lo, hi = q99(4.0), q99(8.0)
        self.assertLess(abs(lo - hi), 0.25 * lo)

    def test_chi2_statistic_grows_with_shadowing(self):
        chk = PhysicalConsistencyChecker(make_model(statistic="chi2"), self.su)
        rng = np.random.default_rng(3)
        z = rng.normal(size=40)
        a = chk.compute(*rss_from(PU, 30.0, self.su, shadow=2.0 * z))["physical_residual"][0]
        b = chk.compute(*rss_from(PU, 30.0, self.su, shadow=4.0 * z))["physical_residual"][0]
        self.assertAlmostEqual(b / a, 4.0, places=2)

    def test_unverifiable_slot(self):
        rss, noise = rss_from(PU, 30.0, self.su)
        noise = rss - 1.0  # every SU below min_snr_db
        out = self.checker.compute(rss, noise)
        self.assertFalse(out["verifiable"][0])
        self.assertFalse(out["physically_consistent"][0])
        self.assertTrue(np.isnan(out["physical_residual"][0]))
        self.assertEqual(out["physical_consistency_score"][0], 0.0)

    def test_consistency_score_is_empirical_survival(self):
        s = self.checker.consistency_score(np.array([-1.0, 5.0, 100.0]))
        self.assertAlmostEqual(s[0], 1.0)
        self.assertAlmostEqual(s[1], 1.0 - 50 / 101, places=6)
        self.assertAlmostEqual(s[2], 0.0)

    def test_batch_equals_single(self):
        rng = np.random.default_rng(4)
        rows = [rss_from(PU, 30.0, self.su, shadow=rng.normal(0, 4, 40)) for _ in range(4)]
        batch = self.checker.compute(np.vstack([r for r, _ in rows]), np.vstack([n for _, n in rows]))
        for i, (r, n) in enumerate(rows):
            single = self.checker.compute(r, n)
            self.assertAlmostEqual(single["physical_residual"][0], batch["physical_residual"][i], places=8)


class TestAlphaSigmaEstimate(unittest.TestCase):
    def test_recovers_parameters(self):
        rng = np.random.default_rng(5)
        su = rng.uniform(0, 100, size=(40, 2))
        rows = [rss_from(PU, 30.0 + rng.normal(), su, shadow=rng.normal(0, 4, 40)) for _ in range(400)]
        alpha, sigma = estimate_alpha_sigma(np.vstack([r for r, _ in rows]), np.vstack([n for _, n in rows]),
                                            su, PU, min_snr_db=3.0)
        self.assertAlmostEqual(alpha, ALPHA, delta=0.05)
        self.assertAlmostEqual(sigma, 4.0, delta=0.15)


class TestLogDistance(unittest.TestCase):
    def test_min_distance_clip(self):
        x = log_distance_matrix(np.array([[0.0, 0.0]]), np.array([[0.0, 0.0], [10.0, 0.0]]), 1.0)
        np.testing.assert_allclose(x, [[0.0, -10.0]])


if __name__ == "__main__":
    unittest.main()
