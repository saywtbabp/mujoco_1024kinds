"""Correctness checks for the information-only multi-Probe experiment."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from navlab import runtime
import numpy as np
from navlab.belief import consistency_update
from navlab.map import MapSpec
from navlab.physics import World, NumericalFailure
from navlab.multi_probe import (load_settings, load_baseline, sample_times, exact_rows,
                               simulate_response, belief_evolution, clear_run, pca_geometry,
                               cache_manifest, digest, response_worker, build_response_cache)


class MultiProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings, cls.cfg = load_settings()
        cls.library = load_baseline(cls.cfg)

    def synthetic(self):
        times = np.array([0., .05, .10, .15, .20])
        # Repeated cumulative observation must not make the posterior sharper.
        predictions = np.zeros((4, len(times), 2))
        predictions[:, 1:, 0] = np.array([.8, 1., 1.2, 1.4])[:, None]
        response = np.zeros((len(times), 2)); response[1:, 0] = 1.
        stream = dict(probe_id="test", force_limit_N=1., times=times, predictions=predictions)
        obs = dict(probe_id="test", force_limit_N=1., times=times.copy(), response=response)
        theta = np.array([[.1, .2], [.2, .3], [.3, .2], [.4, .4]])
        return stream, obs, theta

    def test_prior_normalization_and_no_repeated_evidence(self):
        stream, obs, theta = self.synthetic()
        prior = np.full(4, .25)
        history = belief_evolution(stream, obs, prior, theta, np.array([0, 1, 1, 0]), self.cfg, self.settings)
        np.testing.assert_array_equal(history["weights"][0], prior)
        for j in range(1, len(stream["times"])):
            expected, _ = consistency_update(prior, stream["predictions"][:, j], obs["response"][j], self.cfg)
            np.testing.assert_array_equal(history["weights"][j], expected)
        np.testing.assert_allclose(history["weights"].sum(axis=1), 1., atol=1e-14)
        wrong, _ = consistency_update(history["weights"][1], stream["predictions"][:, 2], obs["response"][2], self.cfg)
        self.assertGreater(np.max(abs(wrong-history["weights"][2])), .01)

    def test_probe_id_force_and_time_must_match(self):
        stream, obs, theta = self.synthetic()
        for key, value in [("probe_id", "wrong"), ("force_limit_N", 2.2), ("times", obs["times"]+.01)]:
            bad = dict(obs); bad[key] = value
            with self.assertRaises(ValueError):
                belief_evolution(stream, bad, np.full(4, .25), theta, np.array([0, 1, 1, 0]), self.cfg, self.settings)

    def test_true_parameters_not_an_update_input(self):
        import inspect
        params = inspect.signature(belief_evolution).parameters
        self.assertNotIn("truth", params)
        self.assertNotIn("theta_truth", params)

    def test_persistence_and_confirmation_and_not_reached(self):
        t = np.arange(8)*.05
        p = np.array([.5, .95, .5, .95, .95, .95, .95, .5])
        d = 4*p*(1-p)
        clear = clear_run(t, p, d, .36, 4)
        self.assertAlmostEqual(clear["t_clear"], .15)
        self.assertAlmostEqual(clear["t_confirmed"], .30)
        self.assertTrue(clear["later_relapse"])
        self.assertIsNone(clear_run(t, p, d, .1, 4)["t_clear"])
        # Low D on alternating Push/Avoid sides is not stable agreement.
        p[3:7] = [.95, .05, .95, .05]
        self.assertIsNone(clear_run(t, p, 4*p*(1-p), .36, 4)["t_clear"])
        self.assertIsNone(clear_run(t[:4], np.ones(4)*.95, np.ones(4)*.19, .36, 4)["t_clear"])

    def test_pca_known_weighted_line(self):
        theta = np.array([[0., 0.], [1., 2.], [2., 4.]])
        g = pca_geometry(theta, np.array([.25, .5, .25]))
        np.testing.assert_allclose(g["mean"], [1., 2.])
        np.testing.assert_allclose(g["covariance"], [[.5, 1.], [1., 2.]])
        self.assertAlmostEqual(g["slope"], 2.)
        self.assertAlmostEqual(g["angle_deg"], np.degrees(np.arctan(2.)))
        self.assertAlmostEqual(g["eigenvalues"][1], 0.)

    def test_exact_time_samples_reject_interpolation(self):
        with self.assertRaises(ValueError):
            exact_rows(np.array([[0., 0., 0.], [.1, 1., .1]]), np.array([0., .05, .1]))

    def test_standard_instrumentation_and_duration_prefix(self):
        theta = self.library["theta"][20]
        world = World(MapSpec(), self.cfg)
        response, _, _ = simulate_response(world, self.library["approach"], theta, sample_times(self.settings))
        np.testing.assert_array_equal(response[-1], self.library["z_hat"][20])
        short = World(MapSpec(), replace(self.cfg, probe_duration=.6))
        short.restore(self.library["approach"], theta)
        z, _ = short.probe()
        np.testing.assert_allclose(z, response[12], atol=1e-9, rtol=0)

    def test_cache_identity_and_partial_resume(self):
        settings = self.settings
        snapshot = dict(source_hashes={"test": "v1"}, mujoco_version="test", numpy_version=np.__version__,
                        contact=dict(lever=.835, direction=[0, 1, 0]))
        probe = settings["probes"][0]
        small = dict(self.library, theta=self.library["theta"][:4])
        manifest = cache_manifest(probe, self.cfg, settings, small, snapshot)
        altered = dict(probe, force_limit_N=1.4)
        self.assertNotEqual(digest(manifest), digest(cache_manifest(altered, self.cfg, settings, small, snapshot)))
        self.assertNotEqual(digest(manifest), digest(cache_manifest(probe, replace(self.cfg, probe_velocity_damping=1.3), settings, small, snapshot)))
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            cfg = replace(self.cfg, workers=1)
            first = build_response_cache(output, probe, cfg, settings, small, snapshot)
            path = first["path"]
            with np.load(path, allow_pickle=False) as saved:
                arrays = {k: saved[k].copy() for k in saved.files}
            arrays["done"][2:] = False
            arrays["predictions"][2:] = np.nan
            from navlab.library import atomic_npz
            atomic_npz(path, **arrays)
            second = build_response_cache(output, probe, cfg, settings, small, snapshot)
            np.testing.assert_array_equal(first["predictions"], second["predictions"])
            with patch("navlab.multi_probe.simulate_response", side_effect=AssertionError("must reuse cache")):
                build_response_cache(output, probe, cfg, settings, small, snapshot)
            with self.assertRaises(ValueError):
                build_response_cache(output, altered, cfg, settings, small, snapshot)

    def test_numerical_anomaly_is_not_normal_response(self):
        with patch("navlab.multi_probe._worker", (None, None, None)), patch("navlab.multi_probe.simulate_response", side_effect=NumericalFailure("test warning")):
            index, response, wall, status, error = response_worker((7, [.1, .2]))
        self.assertEqual(index, 7)
        self.assertIsNone(response)
        self.assertIsNone(wall)
        self.assertEqual(status, "numerical_anomaly")
        self.assertEqual(error, "test warning")


if __name__ == "__main__":
    unittest.main(verbosity=2)
