"""Validate an ACTUALLY completed default A-E study, not fabricated fixtures."""
import argparse
import csv
import json
from pathlib import Path
from navlab import runtime
import numpy as np
from navlab.config import ROOT
from navlab.sampling import check_truth
from navlab.config import Config
from navlab.belief import decide
from navlab.library import signature
from navlab.map import MapSpec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args()
    output = Path(args.output) if args.output else Path((ROOT/"results"/"LATEST.txt").read_text(encoding="utf-8").strip())
    results = json.loads((output/"results.json").read_text(encoding="utf-8"))
    by_case = {r["case"]:r for r in results}
    assert set(by_case)==set("ABCDE"),"requires a complete default A-E run"
    for r in results:
        assert r["N"]==1024 and r["truth_is_off_grid"]
        assert r["mission_success"] and r["shared_tail_success"]
        cfg = Config(**r["config"])
        assert r["library_signature"] == signature(MapSpec(),cfg),"results do not match current controller source"
        with np.load(Path(r["library_folder"])/"hypotheses.npz",allow_pickle=False) as library:
            assert np.all(library["done"]) and library["theta"].shape==(1024,2)
            assert str(library["signature"]) == r["library_signature"]
            assert not np.any(library["failure_category"] == "numerical_anomaly")
            np.testing.assert_array_equal(library["success"],np.isfinite(library["J_door"]))
            check_truth(r["theta_truth_evaluation_only"],library["theta"],cfg)
        with (output/("case_"+r["case"])/"weights.csv").open(encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        w = np.array([float(v["w_after"]) for v in rows])
        y = np.array([int(v["y_after"]) for v in rows])
        np.testing.assert_allclose([float(v["w_initial"]) for v in rows],1/1024,atol=0,rtol=0)
        np.testing.assert_array_equal([v["w_initial"] for v in rows],[v["w_visual"] for v in rows])
        assert np.isfinite(w).all() and np.all(w>=0)
        assert abs(w.sum()-1)<1e-12
        p = r.get("posterior",r["prior"])["p_D"]
        assert abs(float(w@y)-p)<1e-12
        prior = r["prior"]
        assert prior["action"] == decide(r["geometry"]["G"],prior["p_D"],prior["D"],cfg)[0]
        selected = r.get("posterior",prior)
        assert r["final_action"] == decide(r["geometry"]["G"],selected["p_D"],selected["D"],cfg,probe_used=r["probe_used"])[0]
        mission = r["mission"]
        assert mission["time"] == mission["elapsed_time"] and not mission["failure_category"]
        if mission["recovery"]:
            recovery = mission["recovery"]
            assert recovery["success"] and recovery["dwell"] >= cfg.recovery_dwell-1e-10
            assert all(recovery["checks"][k] for k in ("contact_free","attitude","speed","height","clearance"))
        if r["probe_used"]:
            post = r["posterior"]
            assert post["consistency_error_posterior"] < post["consistency_error_prior"]
            assert post["normalized_parameter_distance_posterior"] < post["normalized_parameter_distance_prior"]
            assert 0<=post["D"]<=1
            with np.load(Path(r["continuation_cache"])/"continuation_costs.npz",allow_pickle=False) as continuation:
                assert str(continuation["signature"]) == r["library_signature"]
                assert np.all(continuation["done"])
                assert not np.any(np.isnan(continuation["J_door"])) and not np.any(np.isnan(continuation["J_bypass"]))
                np.testing.assert_array_equal(continuation["labels"],(continuation["J_door"]<continuation["J_bypass"]).astype(int))
                assert not np.any(continuation["door_category"] == "numerical_anomaly")
                assert not np.any(continuation["bypass_category"] == "numerical_anomaly")
    for case in "BCE":
        assert by_case[case]["prior"] == by_case["A"]["prior"],"same map/domain/prior cannot read truth"
    print("Completed 1024-world A-E results verified:",output)


if __name__=="__main__":
    main()
