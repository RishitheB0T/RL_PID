# tests/test_v6.py : checks of improved_v6 (anchored gains, band bonus) and the validation tests
# ---------------------------------------------------------------
# Run from the project folder:   .\run -m unittest discover tests
# ---------------------------------------------------------------
import json, unittest
import numpy as np
from cases import load_case
from core.env import RLPIDEnv
from core.tools import fixed, results_path, run_test, score
from core.validation import last_step_metrics, selection_score, validation_metrics, anchor_metrics
from core.versions import VERSIONS, gain_ranges, BAND_BONUS

CASES = [load_case(name) for name in ("cs1", "cs2", "cs3")]
V6 = VERSIONS["improved_v6"]


class TestAnchoredGains(unittest.TestCase):
    def test_slider_zero_is_the_anchor(self):
        for case in CASES:
            g = V6.to_gains([0, 0, 0], gain_ranges("improved_v6", case))
            self.assertEqual(tuple(map(float, g)), tuple(map(float, case.ANCHOR_PID)) + (1.0,), case.ID)

    def test_anchor_is_the_grid_search_result(self):
        for case in CASES:
            with open(results_path(f"{case.ID}_fixed_gains.json")) as f:
                best = json.load(f)["best_fixed_pid"]
            np.testing.assert_allclose(case.ANCHOR_PID, (best["Kp"], best["tau_i"], best["tau_d"]), rtol=1e-12)

    def test_slider_ends(self):
        case = CASES[2]                                    # CS3: anchor (42.5, 3.75, 0.15), span 0.15
        ranges = gain_ranges("improved_v6", case)
        np.testing.assert_allclose(V6.to_gains([-1, -1, -1], ranges)[:3], (21.25, 1.25, 0.0), atol=1e-12)
        np.testing.assert_allclose(V6.to_gains([1, 1, 1], ranges)[:3], (85.0, 11.25, 0.30), atol=1e-12)


class TestBandBonus(unittest.TestCase):
    def test_bonus_counts_samples_in_band(self):
        """Same fixed gains: v6 reward - v5 reward = BAND_BONUS x (samples inside the 2 % band)."""
        case = CASES[0]
        totals = {}
        for version in ("improved_v5", "improved_v6"):
            env = RLPIDEnv(case, version, test=True)
            env.reset()
            total, done = 0.0, False
            while not done:
                _, r, terminated, truncated, _ = env.step_gains(*case.ANCHOR_PID)
                total += r
                done = terminated or truncated
            totals[version] = total
        e = env.r[case.WARMUP_STEPS:] - env.y[case.WARMUP_STEPS:]
        in_band = int(np.sum(np.abs(e) <= 0.02))
        self.assertAlmostEqual(totals["improved_v6"] - totals["improved_v5"], BAND_BONUS * in_band, places=6)


class TestValidation(unittest.TestCase):
    def test_last_step_metrics_equals_score_on_the_step_test(self):
        for case in CASES:
            env, _ = run_test(case, fixed(*case.ANCHOR_PID))
            np.testing.assert_allclose(last_step_metrics(env), score(env), rtol=1e-12)

    def test_anchor_scores_one(self):
        for case in CASES:
            m = validation_metrics(case, fixed(*case.ANCHOR_PID))
            self.assertAlmostEqual(selection_score(m, anchor_metrics(case), case.DT), 1.0, places=12)

    def test_validation_is_not_the_final_test(self):
        """The validation steps differ from the final 0 -> 1 step test."""
        from core.validation import VALIDATION_LEVELS
        self.assertNotIn([1.0], VALIDATION_LEVELS)


if __name__ == "__main__":
    unittest.main()
