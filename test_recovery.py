"""Actuator feasibility, state gate, timeout and physical recovery regressions."""
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
from dataclasses import replace
from navlab import runtime
import numpy as np
from navlab.config import Config
from navlab.map import MapSpec,Planner
from navlab.physics import World,prepare,bounded_allocation,NumericalFailure
from navlab import library
from navlab.config import ROOT
from navlab.sampling import candidate_worlds


class RecoveryTests(unittest.TestCase):
    def test_feasible_motor_requests_are_preserved(self):
        world = World(MapSpec(),Config())
        rng = np.random.default_rng(1729)
        for motors in rng.uniform(world.limits[:,0],world.limits[:,1],(500,4)):
            request = np.linalg.solve(world.mixer,motors)
            actual = bounded_allocation(world.mixer,world.limits,request[0],request[1:])
            np.testing.assert_allclose(actual,motors,atol=1e-12,rtol=0)

    def test_saturated_allocation_keeps_collective_and_roll_pitch_direction(self):
        world = World(MapSpec(),Config())
        rng = np.random.default_rng(20)
        for _ in range(500):
            thrust = float(rng.uniform(-10,60))
            torque = rng.uniform(-5,5,3)
            motors = bounded_allocation(world.mixer,world.limits,thrust,torque)
            self.assertTrue(np.all(motors >= world.limits[:,0]-1e-12))
            self.assertTrue(np.all(motors <= world.limits[:,1]+1e-12))
            achieved = np.linalg.solve(world.mixer,motors)
            self.assertAlmostEqual(achieved[0],np.clip(thrust,0,40),places=11)
            self.assertAlmostEqual(achieved[1]*torque[1]-achieved[2]*torque[0],0.,places=11)
            self.assertGreaterEqual(float(achieved[1:3]@torque[:2]),-1e-12)

    def test_contact_and_clearance_block_flight(self):
        world = World(MapSpec(),Config())
        world.force = 1.
        ready,checks,path = world.crossing_ready()
        self.assertFalse(ready)
        self.assertFalse(checks["contact_free"])
        self.assertIsNone(path)
        world.force = 0.
        ready,checks,path = world.crossing_ready()
        self.assertFalse(ready)
        self.assertFalse(checks["clearance"])

    def test_recovery_timeout_cannot_enter_flight(self):
        world,start,approach,dt = prepare(MapSpec(),replace(Config(),recovery_timeout=.01))
        stages = []
        world.callback = lambda w: stages.append(w.stage)
        result = world.route("door",approach,[.4015625,.7590625],dt)
        self.assertFalse(result.success)
        self.assertEqual(result.failure_category,"recovery_timeout")
        self.assertTrue(np.isinf(result.time))
        self.assertGreater(result.elapsed_time,dt)
        self.assertNotIn("FLIGHT",stages)

    def test_id20_releases_contact_and_reaches_common_merge(self):
        cfg = Config()
        world,start,approach,dt = prepare(MapSpec(),cfg)
        result = world.route("door",approach,[.4015625,.7590625],dt)
        self.assertTrue(result.success,result.reason)
        self.assertEqual(result.time,result.elapsed_time)
        self.assertEqual(result.wall_peak,0.)
        self.assertTrue(result.recovery["success"])
        self.assertGreaterEqual(result.recovery["dwell"],cfg.recovery_dwell-1e-10)
        c = result.recovery["checks"]
        self.assertEqual(c["door_force_N"],0.)
        self.assertTrue(c["clearance"])
        self.assertLess(c["speed_m_s"],cfg.recovery_speed_max)
        self.assertFalse(Planner(MapSpec(),cfg,door_angle=result.final_angle).collides(np.array(c["position"])[:2])[0])
        self.assertLess(result.final_speed,cfg.merge_speed_tolerance)

    def test_numerical_failure_is_not_a_push_timeout(self):
        world = World(MapSpec(),Config())
        state = world.snapshot()
        def invalid_step(*args,**kwargs):
            raise NumericalFailure("injected invalid simulation")
        world.step = invalid_step
        result = world.route("door",state,[.4,7.])
        self.assertFalse(result.success)
        self.assertEqual(result.failure_category,"numerical_anomaly")
        self.assertTrue(np.isinf(result.time))

    def test_post_probe_bypass_restores_route_heading(self):
        world,start,approach,dt = prepare(MapSpec(),Config())
        world.restore(approach,[.071,.035])
        world.probe()
        result = world.route("bypass",world.snapshot(),[.05,.01])
        self.assertTrue(result.success,result.reason)
        self.assertLess(result.time,world.cfg.route_timeout)
        self.assertTrue(result.recovery["success"])

    def test_case_a_post_probe_door_keeps_height_while_recovering(self):
        cfg = Config()
        _,start,approach,dt = prepare(MapSpec(),cfg)
        world = World(MapSpec(),cfg)
        truth = [.071,.035]
        world.restore(start,truth)
        self.assertTrue(world.fly([world.spec.approach]))
        world.probe()
        result = world.route("door",world.snapshot(),truth)
        self.assertTrue(result.success,result.reason)
        self.assertTrue(result.recovery["success"])
        self.assertEqual(result.recovery["checks"]["door_force_N"],0.)
        self.assertEqual(result.wall_peak,0.)

    def test_interrupted_continuations_resume_only_unfinished_worlds(self):
        cfg = replace(Config(),workers=1)
        spec = MapSpec()
        world,start,approach,dt = prepare(spec,cfg)
        world.restore(approach,[.071,.035]); world.probe()
        state = world.snapshot()
        theta = candidate_worlds(cfg)[:2]
        original = library.parallel_rows
        def interrupted(items,args,workers):
            yield next(iter(original(items,args,workers)))
            raise RuntimeError("intentional test interruption")
        with tempfile.TemporaryDirectory(dir=ROOT/"results"/"recovery_validation") as temporary:
            folder = Path(temporary)
            with patch.object(library,"parallel_rows",side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError,"intentional test interruption"):
                    library.continuation_labels(spec,cfg,state,theta,folder)
            with np.load(folder/"continuation_costs.npz") as saved:
                np.testing.assert_array_equal(saved["done"],[True,False])
                self.assertTrue(np.isnan(saved["J_door"][1]))
            def resumed(items,args,workers):
                self.assertEqual([i for i,t in items],[1])
                yield from original(items,args,workers)
            with patch.object(library,"parallel_rows",side_effect=resumed):
                labels,jd,jb = library.continuation_labels(spec,cfg,state,theta,folder)
            np.testing.assert_array_equal(labels,(jd<jb).astype(int))
            with np.load(folder/"continuation_costs.npz") as saved:
                self.assertTrue(np.all(saved["done"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
