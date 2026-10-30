"""Physical, cache and belief invariants for the five temporal profiles."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
from navlab import runtime
import numpy as np
from navlab.config import load_config
from navlab.force_profiles import ForceProfile, ForceSegment, ProfileControlConfig, simulate_profile_response
from navlab.profile_comparison import (load_profile_settings, profile_budget_audit, profile_belief_evolution,
                                       build_profile_cache, profile_worker, flag_fastest)
from navlab.profile_report import axial_difference, direction_reliable
from navlab.multi_probe import load_baseline, sample_times, simulate_response
from navlab.physics import World, NumericalFailure
from navlab.map import MapSpec
from navlab.library import atomic_npz


class ForceProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings, cls.cfg, profiles = load_profile_settings()
        cls.profiles = {p.id:p for p in profiles}
        cls.library = load_baseline(cls.cfg)
        cls.times = sample_times(cls.settings)

    def test_all_analytical_and_discrete_budgets(self):
        rows = profile_budget_audit(list(self.profiles.values()), self.cfg.timestep)
        self.assertEqual(len(rows),5)
        self.assertEqual(sum(r["matched_budget"] for r in rows),4)
        for r in rows:
            self.assertAlmostEqual(r["analytical_budget_Ns"],1.92 if r["matched_budget"] else 2.64,places=12)
            self.assertAlmostEqual(r["timestep_integrated_budget_Ns"],r["analytical_budget_Ns"],places=12)

    def test_switches_are_right_continuous(self):
        for pid,t,before,after in [("low_to_high_1p0_2p2",.6,1.,2.2),("high_to_low_2p2_1p0",.6,2.2,1.),("pulse_hold_2p2_1p48",.2,2.2,1.48)]:
            p=self.profiles[pid]
            self.assertEqual(p.force_limit_at(np.nextafter(t,-np.inf)),before)
            self.assertEqual(p.force_limit_at(t),after)
            self.assertEqual(p.force_limit_at(np.nextafter(t,np.inf)),after)
            self.assertEqual(p.force_limit_at(1.2),after)

    def test_profile_validation_rejects_gap_wrong_budget_and_off_tick(self):
        p=self.profiles["low_to_high_1p0_2p2"]
        for bad in [replace(p,command_force_budget_Ns=1.),
                    replace(p,segments=(ForceSegment(0.,.59,1.),ForceSegment(.6,1.2,2.2))),
                    replace(p,segments=(ForceSegment(0.,.601,1.),ForceSegment(.601,1.2,2.2)))]:
            with self.assertRaises(ValueError):bad.validate(.002)

    def test_adapter_only_changes_cap_and_uses_elapsed_clock(self):
        clock=SimpleNamespace(step_count=299)
        p=self.profiles["low_to_high_1p0_2p2"]
        config=ProfileControlConfig(self.cfg,p,clock)
        self.assertEqual(config.probe_force_max,1.)
        clock.step_count=300
        self.assertEqual(config.probe_force_max,2.2)
        self.assertEqual(config.probe_velocity_damping,self.cfg.probe_velocity_damping)
        self.assertEqual(config.timestep,self.cfg.timestep)
        self.assertEqual(self.cfg.probe_force_max,2.2)

    def test_standard_matches_original_uninstrumented_scalar_probe(self):
        world=World(MapSpec(),self.cfg)
        r,_,diag,commands=simulate_profile_response(world,self.library["approach"],self.library["theta"][20],self.times,self.profiles["standard_2p2N"],True)
        np.testing.assert_array_equal(r[-1],self.library["z_hat"][20])
        np.testing.assert_array_equal(commands[:,1],np.full(600,2.2))
        self.assertEqual(diag.shape,(25,6))
        self.assertIs(world.cfg,self.cfg)
        self.assertNotIn("step",world.__dict__)

    def test_constant_adapter_matches_previous_constant_probe(self):
        p=self.profiles["constant_1p6N"]
        r,_,_,_=simulate_profile_response(World(MapSpec(),self.cfg),self.library["approach"],self.library["theta"][20],self.times,p)
        old_cfg=replace(self.cfg,probe_force_max=1.6)
        old,_,_=simulate_response(World(MapSpec(),old_cfg),self.library["approach"],self.library["theta"][20],self.times)
        np.testing.assert_array_equal(r,old)

    def test_identical_profile_prefixes_match_until_transition(self):
        theta=self.library["theta"][20]
        standard,_,_,_=simulate_profile_response(World(MapSpec(),self.cfg),self.library["approach"],theta,self.times,self.profiles["standard_2p2N"])
        for pid,j in [("high_to_low_2p2_1p0",12),("pulse_hold_2p2_1p48",4)]:
            r,_,_,_=simulate_profile_response(World(MapSpec(),self.cfg),self.library["approach"],theta,self.times,self.profiles[pid])
            np.testing.assert_array_equal(r[:j+1],standard[:j+1])
        low,_,_,_=simulate_profile_response(World(MapSpec(),self.cfg),self.library["approach"],theta,self.times,self.profiles["low_to_high_1p0_2p2"])
        old_cfg=replace(self.cfg,probe_force_max=1.)
        old,_,_=simulate_response(World(MapSpec(),old_cfg),self.library["approach"],theta,self.times)
        np.testing.assert_array_equal(low[:13],old[:13])

    def test_actual_step_and_motor_tick_switching(self):
        p=self.profiles["low_to_high_1p0_2p2"]
        world=World(MapSpec(),self.cfg)
        ticks=[];control=world.control
        def measured_control(horizontal=None,yaw=None):
            ticks.append((world.step_count,world.cfg.probe_force_max))
            return control(horizontal,yaw)
        world.control=measured_control
        _,_,_,commands=simulate_profile_response(world,self.library["approach"],self.library["theta"][20],self.times,p,True)
        self.assertEqual(commands[299,1],1.)
        self.assertEqual(commands[300,1],2.2)
        self.assertIn((295,1.),ticks)
        self.assertIn((300,2.2),ticks)
        self.assertEqual(len(ticks),120)

    def test_adapter_cleanup_on_failure(self):
        world=World(MapSpec(),self.cfg)
        with patch("navlab.force_profiles.capture_probe",side_effect=NumericalFailure("injected")):
            with self.assertRaises(NumericalFailure):
                simulate_profile_response(world,self.library["approach"],self.library["theta"][0],self.times,self.profiles["constant_1p6N"],True)
        self.assertIs(world.cfg,self.cfg)
        self.assertIsNone(world.callback)
        self.assertNotIn("step",world.__dict__)

    def test_segment_identity_rejects_same_id_and_peak_but_different_order(self):
        p=self.profiles["low_to_high_1p0_2p2"]
        q=replace(self.profiles["high_to_low_2p2_1p0"],id=p.id,label=p.label)
        self.assertNotEqual(p.signature,q.signature)
        stream=dict(profile_signature=p.signature)
        obs=dict(profile_signature=q.signature)
        with self.assertRaises(ValueError):
            profile_belief_evolution(stream,obs,None,None,None,self.cfg,self.settings)

    def test_incremental_cache_and_profile_change_rejection(self):
        p=self.profiles["low_to_high_1p0_2p2"]
        cfg=replace(self.cfg,workers=1)
        library=dict(self.library,theta=self.library["theta"][:3])
        snapshot=dict(source_hashes={"test":"v1"},mujoco_version="test",numpy_version=np.__version__,contact={"lever":.835,"direction":[0,1,0]})
        with tempfile.TemporaryDirectory() as tmp:
            first=build_profile_cache(Path(tmp),p,cfg,self.settings,library,snapshot)
            with np.load(first["path"],allow_pickle=False) as saved:
                arrays={k:saved[k].copy() for k in saved.files}
            arrays["done"][1:]=False;arrays["predictions"][1:]=np.nan
            atomic_npz(first["path"],**arrays)
            second=build_profile_cache(Path(tmp),p,cfg,self.settings,library,snapshot)
            np.testing.assert_array_equal(first["predictions"],second["predictions"])
            with patch("navlab.profile_comparison.profile_worker",side_effect=AssertionError("cached worlds rerun")):
                build_profile_cache(Path(tmp),p,cfg,self.settings,library,snapshot)
            q=replace(self.profiles["high_to_low_2p2_1p0"],id=p.id,label=p.label)
            with self.assertRaises(ValueError):
                build_profile_cache(Path(tmp),q,cfg,self.settings,library,snapshot)

    def test_numerical_anomaly_is_separate_and_blocks_normal_result(self):
        with patch("navlab.profile_comparison._worker",(None,None,None,None)),patch("navlab.profile_comparison.simulate_profile_response",side_effect=NumericalFailure("warning")):
            result=profile_worker((2,[.1,.2]))
        self.assertEqual(result[4],"numerical_anomaly")
        self.assertIsNone(result[1])

    def test_axial_angles_and_reliability(self):
        self.assertEqual(axial_difference(89,-89),2)
        self.assertEqual(axial_difference(80,-80),20)
        self.assertEqual(axial_difference(0,180),0)
        m=dict(ESS=1024,normalized_geometry=dict(anisotropy_ratio=1.))
        self.assertFalse(direction_reliable(m,self.settings))
        m=dict(ESS=2,normalized_geometry=dict(anisotropy_ratio=100.))
        self.assertFalse(direction_reliable(m,self.settings))

    def test_fastest_flags_preserve_ties_and_not_reached(self):
        histories=[dict(t_clear=t,probe=dict(matched_budget=m),summary={}) for t,m in [(.8,False),(.9,True),(.9,True),(None,True)]]
        flag_fastest(histories)
        self.assertTrue(histories[0]["summary"]["fastest_overall"])
        self.assertTrue(histories[1]["summary"]["fastest_matched_budget"])
        self.assertTrue(histories[2]["summary"]["fastest_matched_budget"])
        self.assertFalse(histories[3]["summary"]["fastest_matched_budget"])


if __name__=="__main__":
    unittest.main(verbosity=2)
