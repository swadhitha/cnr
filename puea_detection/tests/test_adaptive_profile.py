"""
Unit tests for the adaptive update gate, the PU profile and the detection systems.

Run from the repository root:
    python -m unittest discover -s puea_detection/tests -v
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from adaptive_profile import AdaptivePUProfile, calibrate_profile, update_allowed  # noqa: E402
from adaptive_systems import SYSTEMS_BY_NAME, DetectionSystem  # noqa: E402


class TestUpdateGate(unittest.TestCase):
    def test_naive_gate_updates_every_pu_like_slot(self):
        self.assertTrue(update_allowed(True, False, require_physical=False))
        self.assertTrue(update_allowed(True, None, require_physical=False))
        self.assertFalse(update_allowed(False, True, require_physical=False))

    def test_anchored_gate_requires_physical_consistency(self):
        self.assertTrue(update_allowed(True, True, require_physical=True))
        self.assertFalse(update_allowed(True, False, require_physical=True))
        self.assertFalse(update_allowed(False, True, require_physical=True))

    def test_anchored_gate_fails_closed_when_unverifiable(self):
        self.assertFalse(update_allowed(True, None, require_physical=True))


class TestProfile(unittest.TestCase):
    def setUp(self):
        self.p = AdaptivePUProfile(np.zeros(2), np.eye(2), threshold=9.0, ewma_lambda=0.1)

    def test_score_is_squared_mahalanobis(self):
        self.assertAlmostEqual(float(self.p.score(np.array([3.0, 4.0]))), 25.0, places=4)
        self.assertTrue(self.p.is_anomalous(np.array([3.0, 4.0])))
        self.assertFalse(self.p.is_anomalous(np.array([1.0, 1.0])))

    def test_ewma_update(self):
        self.p.update(np.array([1.0, 0.0]))
        np.testing.assert_allclose(self.p.mean, [0.1, 0.0])
        np.testing.assert_allclose(self.p.cov, 0.9 * np.eye(2) + 0.1 * np.array([[1.0, 0.0], [0.0, 0.0]]))
        self.assertEqual(self.p.n_updates, 1)

    def test_copy_is_independent(self):
        q = self.p.copy()
        q.update(np.array([5.0, 5.0]))
        np.testing.assert_allclose(self.p.mean, [0.0, 0.0])
        self.assertEqual(self.p.n_updates, 0)

    def test_displacement_uses_initial_metric(self):
        self.p.update(np.array([10.0, 0.0]))
        self.assertAlmostEqual(self.p.displacement(), 1.0, places=4)

    def test_calibration_threshold_is_held_out_quantile(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(2000, 3))
        cfg = {"adaptive_profile": {"ewma_lambda": 0.01, "calibration_false_alarm": 0.05,
                                    "covariance_ridge": 1e-6}}
        prof = calibrate_profile(X, cfg)
        held_out_fa = (prof.score(X[1000:]) > prof.threshold).mean()
        self.assertLessEqual(held_out_fa, 0.05)
        self.assertGreater(held_out_fa, 0.03)


class _Evidence:
    """Minimal stand-in for adaptive_systems.SlotEvidence (one slot)."""

    def __init__(self, z, p_xgb, consistent, verifiable=True, resid=1.0):
        self.z = np.atleast_2d(z)
        self.p_xgb = np.array([p_xgb])
        self.p_spatial = np.array([p_xgb])
        self.phys = {
            "verifiable": np.array([verifiable]),
            "physically_consistent": np.array([consistent]),
            "physical_residual": np.array([resid]),
            "physical_consistency_score": np.array([0.5]),
            "pu_power_hat_dbm": np.array([30.0]),
            "source_x_hat": np.array([0.0]),
            "source_y_hat": np.array([0.0]),
        }


class TestSystems(unittest.TestCase):
    def setUp(self):
        self.profile = AdaptivePUProfile(np.zeros(2), np.eye(2), threshold=9.0, ewma_lambda=0.1)

    def system(self, name):
        return DetectionSystem(SYSTEMS_BY_NAME[name], self.profile, xgb_threshold=0.5, phys_threshold=10.0)

    def test_pu_like_but_physically_inconsistent(self):
        """The key distinction: PU-like != physically consistent with the PU."""
        ev = _Evidence(z=[0.5, 0.5], p_xgb=0.1, consistent=False, resid=50.0)
        naive = self.system("Naive-Adaptive").step(ev, 0)
        anchored = self.system("Physically-Anchored-Adaptive").step(ev, 0)
        self.assertTrue(naive["pu_like"] and anchored["pu_like"])
        self.assertTrue(naive["accepted_for_update"])
        self.assertFalse(naive["attack_detected"])
        self.assertFalse(anchored["accepted_for_update"])
        self.assertTrue(anchored["rejected_for_update"])
        self.assertTrue(anchored["attack_detected"])

    def test_pu_like_and_consistent_updates_anchored(self):
        ev = _Evidence(z=[0.5, 0.5], p_xgb=0.1, consistent=True)
        sys_ = self.system("Physically-Anchored-Adaptive")
        r = sys_.step(ev, 0)
        self.assertTrue(r["accepted_for_update"])
        self.assertFalse(r["attack_detected"])
        self.assertEqual(sys_.profile.n_updates, 1)
        self.assertEqual(self.profile.n_updates, 0)  # shared calibration profile untouched

    def test_update_only_variant_ignores_physics_for_decision(self):
        ev = _Evidence(z=[0.5, 0.5], p_xgb=0.1, consistent=False, resid=50.0)
        r = self.system("Anchored-Update-Only").step(ev, 0)
        self.assertFalse(r["attack_detected"])
        self.assertFalse(r["accepted_for_update"])

    def test_alarm_needs_both_xgb_and_profile(self):
        inside = _Evidence(z=[0.5, 0.5], p_xgb=0.9, consistent=True)
        outside = _Evidence(z=[5.0, 5.0], p_xgb=0.9, consistent=True)
        self.assertFalse(self.system("Naive-Adaptive").step(inside, 0)["attack_detected"])
        self.assertTrue(self.system("Naive-Adaptive").step(outside, 0)["attack_detected"])
        self.assertTrue(self.system("Baseline-XGBoost").step(inside, 0)["attack_detected"])

    def test_static_systems_never_update(self):
        ev = _Evidence(z=[0.5, 0.5], p_xgb=0.1, consistent=True)
        for name in ["Baseline-XGBoost", "XGB+Physical", "XGB+StaticProfile", "Physical-Only"]:
            self.assertFalse(self.system(name).step(ev, 0)["accepted_for_update"], name)

    def test_unverifiable_never_updates_anchored_and_raises_no_physical_alarm(self):
        ev = _Evidence(z=[0.5, 0.5], p_xgb=0.1, consistent=False, verifiable=False, resid=np.nan)
        r = self.system("Physically-Anchored-Adaptive").step(ev, 0)
        self.assertFalse(r["accepted_for_update"])
        self.assertFalse(r["attack_detected"])


if __name__ == "__main__":
    unittest.main()
