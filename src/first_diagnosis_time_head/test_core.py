"""Focused endpoint, likelihood, sampling and leakage tests."""
import math
import unittest

import core


class EndpointTests(unittest.TestCase):
    def example(self, tokens, ages, cut, followup=1000, horizon=100):
        return core.example(tokens, ages, cut, followup, {1: 0, 2: 1}, horizon)

    def test_diagnosis_not_first_clinical_event(self):
        r = self.example([1, 9, 9, 2], [0, 10, 12, 30], 1)
        self.assertEqual((r["event"], r["duration"]), (True, 20))

    def test_same_day_and_repeated_diagnosis_kept(self):
        r = self.example([1, 1, 2], [10, 10, 50], 0)
        self.assertEqual(r["duration"], 0)
        self.assertTrue(r["event"])

    def test_exact_horizon_included(self):
        self.assertTrue(self.example([1, 2], [0, 100], 0)["event"])
        self.assertFalse(self.example([1, 2], [0, 101], 0)["event"])

    def test_censoring_not_future_count(self):
        r = self.example([1, 9], [10, 20], 0, followup=25)
        self.assertFalse(r["event"])
        self.assertEqual(r["duration"], 15)
        self.assertTrue(r["loss_eligible"])

    def test_no_exposure_no_loss(self):
        r = self.example([1], [10], 0, followup=8)
        self.assertFalse(r["loss_eligible"])
        self.assertEqual(r["duration"], 0)

    def test_empty_diagnosis_history(self):
        r = self.example([9, 2], [10, 20], 0)
        self.assertEqual(r["history"], [])
        self.assertEqual(core.covariates(core.model_input(r))[-1], 1)

    def test_future_changes_do_not_change_input(self):
        a = self.example([1, 9, 1], [10, 20, 30], 1, followup=100)
        b = self.example([1, 9, 2, 2], [10, 20, 80, 90], 1, followup=35)
        self.assertEqual(core.model_input(a), core.model_input(b))
        self.assertEqual(core.covariates(a), core.covariates(b))
        self.assertNotEqual(a["duration"], b["duration"])

    def test_cut_clock_not_last_diagnosis_clock(self):
        r = self.example([1, 9, 2], [10, 40, 50], 1)
        self.assertEqual(r["duration"], 10)
        self.assertAlmostEqual(core.covariates(r)[2], math.log1p(30))

    def test_unordered_time_rejected(self):
        with self.assertRaises(ValueError):
            self.example([1, 2], [10, 8], 0)

    def test_distribution_output_not_clipped_to_window(self):
        self.assertEqual(core.predictions(math.log1p(200), 1, [0], 100), [None])
        self.assertEqual(core.predictions(-1, 1, [0], 100), [0.])

    def test_reproducible_patient_specific_draws(self):
        a = core.normal_draws(4, "val", 7, 9, 20)
        self.assertEqual(a, core.normal_draws(4, "val", 7, 9, 20))
        self.assertNotEqual(a, core.normal_draws(4, "test", 7, 9, 20))
        self.assertTrue(all(math.isfinite(v) for v in a))

    def test_patient_averaging_and_availability(self):
        rows = [dict(event=True, duration=10, samples=[0., 20.], median=10.),
                dict(event=True, duration=10, samples=[11., None], median=12.),
                dict(event=False, duration=50, samples=[20., 30.], median=25.)]
        m = core.summarize(rows, 8, 2)
        self.assertEqual(m["sampled_time_mae_days"], 5.5)
        self.assertEqual(m["paired_rollout_availability"], .5)
        self.assertEqual(m["paired_patients"], 2)


class LikelihoodTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        from head import nll
        self.nll = nll

    def test_likelihood_three_observation_types(self):
        t = self.torch
        mu = t.zeros(3, dtype=t.float64, requires_grad=True)
        sigma = t.ones(3, dtype=t.float64)
        duration = t.tensor([0., math.e-1, math.e-1], dtype=t.float64)
        observed = t.tensor([True, True, False])
        eligible = t.ones(3, dtype=t.bool)
        loss = self.nll(mu, sigma, duration, observed, eligible)
        expected = (math.log(2) + .5 + .5*math.log(2*math.pi) + 1 -
                    math.log(.5*math.erfc(1/math.sqrt(2)))) / 3
        self.assertAlmostEqual(float(loss), expected, places=10)
        loss.backward()
        self.assertTrue(bool(t.isfinite(mu.grad).all()))

    def test_censored_zero_exposure_excluded(self):
        t = self.torch
        loss = self.nll(t.zeros(2), t.ones(2), t.zeros(2),
                        t.tensor([True, False]), t.tensor([True, False]))
        self.assertAlmostEqual(float(loss), math.log(2), places=10)

    def test_head_has_independent_weights(self):
        import torch
        from head import TimeHead
        x = torch.tensor([[0., 1.], [1., 2.], [3., 2.]])
        a = TimeHead(x, torch.tensor([0., 3., 8.]), torch.tensor([True]*3))
        b = TimeHead(x, torch.tensor([0., 3., 8.]), torch.tensor([True]*3))
        self.assertNotEqual(next(a.parameters()).data_ptr(), next(b.parameters()).data_ptr())
        mu, sigma = a(x)
        self.assertTrue(bool(torch.isfinite(mu).all() and (sigma >= .05).all()))


if __name__ == "__main__":
    unittest.main()
