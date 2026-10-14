"""Meaningful numerical/physics invariants; PyCharm Run, no pytest dependency."""
import unittest
from dataclasses import replace
from navlab import runtime
import numpy as np
from navlab.config import Config
from navlab.sampling import sobol_2d,candidate_worlds,check_truth
from navlab.map import MapSpec,geometry,Planner
from navlab.physics import World,prepare
from navlab.belief import consistency_update,support,decide


class ResearchInvariants(unittest.TestCase):
    def test_sobol_known_prefix_and_1024_net(self):
        expected = np.array([[0,0],[.5,.5],[.75,.25],[.25,.75],
                             [.375,.375],[.875,.875],[.625,.125],[.125,.625]])
        np.testing.assert_array_equal(sobol_2d(3),expected)
        points = sobol_2d(10)
        self.assertEqual(len(np.unique(points,axis=0)),1024)
        self.assertTrue(np.all((points>=0)&(points<1)))
        # A 2-D Sobol (0,10,2) net has one point in every 1/32 x 1/32 box.
        bins = (points*32).astype(int)
        self.assertEqual(len(np.unique(bins,axis=0)),1024)
        for p in range(11):
            counts = np.bincount((points[:,0]*(2**p)).astype(int),minlength=2**p)
            np.testing.assert_array_equal(counts,np.full(2**p,2**(10-p)))

    def test_feasible_domain_and_off_grid_truth(self):
        cfg = Config()
        theta = candidate_worlds(cfg)
        check_truth([.437,2.473],theta,cfg)
        with self.assertRaises(ValueError):
            check_truth(theta[100],theta,cfg)
        with self.assertRaises(ValueError):
            check_truth([.9,2.],theta,cfg)

    def test_weighted_disagreement_and_gate(self):
        cfg = Config()
        p,d = support([.9,.1],[1,0])
        self.assertAlmostEqual(p,.9)
        self.assertAlmostEqual(d,.36)
        self.assertEqual(decide(2.,.5,1.,cfg)[0],"Avoid")
        self.assertEqual(decide(7.,.99,.0396,cfg)[0],"Push")
        self.assertEqual(decide(7.,.01,.0396,cfg)[0],"Avoid")
        self.assertEqual(decide(7.,.5,1.,cfg)[0],"Probe")
        self.assertEqual(decide(7.,.5,1.,cfg,probe_used=True)[0],"Avoid")

    def test_soft_update_stable_under_extreme_error(self):
        cfg = Config()
        prediction = np.array([[0.,0.],[100.,100.],[1000.,1000.]])
        weights,error = consistency_update([1,1,0],prediction,[10000.,10000.],cfg)
        self.assertTrue(np.all(np.isfinite(weights)))
        self.assertAlmostEqual(weights.sum(),1.)
        self.assertEqual(weights[2],0.)
        weights,error = consistency_update([1,1,1],prediction,[100.,100.],cfg)
        self.assertAlmostEqual(weights[1],1.)
        self.assertTrue(np.all(error>=0))

    def test_astar_navigation_value_and_clearance(self):
        cfg = Config()
        big,small = geometry(MapSpec(),cfg),geometry(MapSpec("small",1.05),cfg)
        self.assertGreater(big["G"],cfg.geometry_threshold)
        self.assertLess(small["G"],cfg.geometry_threshold)
        self.assertGreater(big["G"],small["G"])
        for door in (True,False):
            planner = Planner(MapSpec(),cfg,door_angle=None if door else 0.,force_door=door)
            path,_ = planner.plan(MapSpec().start,MapSpec().merge)
            self.assertTrue(all(planner.clear(a,b) for a,b in zip(path[:-1],path[1:])))

    def test_exact_hinge_inertia_and_unchanged_appearance(self):
        world = World(MapSpec(),Config())
        sizes = world.model.geom_size.copy()
        rgba = world.model.geom_rgba.copy()
        for theta in ((.05,.01),(.317,4.135),(.8,8.)):
            world.reset(theta)
            self.assertAlmostEqual(world.hinge_inertia(),theta[0],places=12)
            self.assertAlmostEqual(world.model.dof_frictionloss[world.da],theta[1],places=12)
            np.testing.assert_array_equal(world.model.geom_size,sizes)
            np.testing.assert_array_equal(world.model.geom_rgba,rgba)
            self.assertEqual(world.data.qpos[world.qa],0.)

    def test_all_three_original_drone_models_fly(self):
        for source in ("Mujoco_0","Mujoco_1","Mujoco_2"):
            cfg = replace(Config(),model_source=source)
            world,start,approach,dt = prepare(MapSpec(),cfg)
            self.assertGreater(world.model.ntendon,0)
            self.assertEqual(len(world.motor_ids),4)
            self.assertLess(np.linalg.norm(world.observed()[0]-MapSpec().approach),.08)
            self.assertGreater(dt,0)

    def test_probe_response_and_replay(self):
        cfg = Config()
        world,start,approach,dt = prepare(MapSpec(),cfg)
        results = []
        for theta in ((.071,.035),(.673,6.317)):
            world.restore(approach,theta)
            z,_ = world.probe()
            world.restore(approach,theta)
            replay,_ = world.probe()
            np.testing.assert_allclose(z,replay,atol=1e-10,rtol=0.)
            results.append(z)
        self.assertGreater(results[0][1],results[1][1]+.2)
        self.assertGreater(results[1][0],results[0][0]+5.)


if __name__ == "__main__":
    unittest.main(verbosity=2)
