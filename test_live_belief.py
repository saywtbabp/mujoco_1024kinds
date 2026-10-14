"""Validate real prefix evidence and prevent repeated-frame overconfidence."""
import unittest
from navlab import runtime
import numpy as np
from navlab.config import load_config
from navlab.map import MapSpec
from navlab.library import build_library
from navlab.probe_stream import build_probe_stream,predictions_at,capture_probe
from navlab.physics import World
from navlab.live_belief import LiveBelief
from navlab.belief import consistency_update


class LiveBeliefEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config()
        cls.library = build_library(MapSpec(),cls.cfg)
        cls.stream = build_probe_stream(MapSpec(),cls.cfg,cls.library)

    def test_complete_prefix_library_matches_terminal_predictions(self):
        prediction = self.stream["predictions"]
        self.assertEqual(prediction.shape[0],1024)
        self.assertTrue(np.isfinite(prediction).all())
        np.testing.assert_allclose(prediction[:,-1],self.library["z_hat"],atol=1e-9,rtol=0)
        self.assertTrue(np.all(np.diff(prediction[:,:,0],axis=1)>=-1e-9))
        np.testing.assert_array_equal(predictions_at(self.stream,0.),np.zeros((1024,2)))

    def test_partial_evidence_uses_matching_time_and_original_prior(self):
        world = World(MapSpec(),self.cfg)
        world.restore(self.library["approach"],[.437,2.473])
        observer = LiveBelief(self.cfg,self.library,self.stream,self.library["labels"],7.)
        observer.begin_probe(world)
        world.callback = observer.on_world
        _,z,_ = capture_probe(world)
        records = [r for r in observer.records if r["phase"]=="PROBE"]
        self.assertGreaterEqual(len(records),20)
        for row in records:
            expected,_ = consistency_update(observer.prior,predictions_at(self.stream,row["elapsed"]),
                                           row["observation"],self.cfg)
            np.testing.assert_allclose(row["weights"],expected,atol=1e-14,rtol=0)
            self.assertAlmostEqual(row["weights"].sum(),1.,places=12)
        np.testing.assert_allclose(records[0]["weights"],observer.prior,atol=1e-12,rtol=0)
        self.assertGreater(np.linalg.norm(records[-1]["weights"]-observer.prior,ord=1),.5)
        final,_ = observer.finish_probe(z,world.data.time)
        old_final,_ = consistency_update(observer.prior,self.library["z_hat"],z,self.cfg)
        np.testing.assert_array_equal(final,old_final)

    def test_changed_labels_do_not_reapply_observation(self):
        observer = LiveBelief(self.cfg,self.library,self.stream,self.library["labels"],7.)
        observer.finish_probe(self.library["z_hat"][200],1.2)
        before = observer.weights.copy()
        observer.set_continuation(1-self.library["labels"],"Avoid",1.2)
        np.testing.assert_array_equal(before,observer.weights)
        self.assertAlmostEqual(observer.p,float(before@(1-self.library["labels"])),places=12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
