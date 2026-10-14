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
        with np.load(Path(r["library_folder"])/"hypotheses.npz",allow_pickle=False) as library:
            assert np.all(library["done"]) and library["theta"].shape==(1024,2)
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
        if r["probe_used"]:
            post = r["posterior"]
            assert post["consistency_error_posterior"] < post["consistency_error_prior"]
            assert post["normalized_parameter_distance_posterior"] < post["normalized_parameter_distance_prior"]
            assert 0<=post["D"]<=1
    assert by_case["A"]["final_action"]=="Push"
    assert by_case["B"]["final_action"]=="Avoid"
    assert by_case["C"]["prior"]["action"]=="Probe"
    assert by_case["D"]["final_action"]=="Avoid" and not by_case["D"]["probe_used"]
    assert by_case["E"]["posterior"]["mass_radius_0_10_posterior"] > by_case["E"]["posterior"]["mass_radius_0_10_prior"]
    assert all(by_case[k]["posterior"]["D"] < by_case[k]["prior"]["D"] for k in "ABCE")
    print("Completed 1024-world A-E results verified:",output)


if __name__=="__main__":
    main()
