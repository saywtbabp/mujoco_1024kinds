"""Optional 1024-world physical verification of both no-Probe agreement branches.

These are explicitly DIFFERENT feasible-domain experiments, all initially equal
weighted. They do not inject truth-dependent priors into the default A-E study.
"""
import multiprocessing
import sys
from dataclasses import replace
from navlab.config import load_config
from navlab.map import MapSpec
from navlab.library import build_library
from navlab.experiments import run_case,new_output,save_json
from navlab.report import write_report


def main():
    base = load_config()
    output = new_output()
    experiments = [
        ("easy_feasible_domain","A",replace(base,inertia_min=.05,inertia_max=.35,
                                              friction_min=.01,friction_max=.15,workers=8),"Push"),
        ("resistant_feasible_domain","B",replace(base,friction_min=6.,friction_max=8.,workers=8),"Avoid")
    ]
    results = []
    for domain,name,cfg,expected in experiments:
        cfg.validate()
        folder = output/domain; folder.mkdir(parents=True,exist_ok=True)
        library = build_library(MapSpec(),cfg)
        result = run_case(name,cfg,library,folder)
        result["separate_feasible_domain_experiment"] = domain
        save_json(folder/"results.json",[result])
        write_report(folder,[result],library,cfg)
        if result["prior"]["action"] != expected or result["probe_used"] or not result["mission_success"]:
            raise RuntimeError("no-Probe physical agreement check failed: " + domain)
        results.append(result)
    save_json(output/"agreement_verification.json",results)
    print("Both physical agreement branches passed:",output)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    if hasattr(sys.stdout,"reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
