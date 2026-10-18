"""Compare actual preserved and current default libraries, never infer labels from tau_f."""
import json
from collections import Counter
from pathlib import Path
from navlab import runtime
import numpy as np
from navlab.config import ROOT
from navlab.experiments import save_json


def old_category(reason):
    if reason == "reached common merge state":
        return "success"
    if "push timeout" in reason:
        return "push_timeout"
    if "A*" in reason:
        return "planning_clearance_failure"
    return "crossing_failure"


def main():
    old_output = ROOT/"results"/"20261017_230029"
    output = Path((ROOT/"results"/"LATEST.txt").read_text(encoding="utf-8").strip())
    old_results = json.loads((old_output/"results.json").read_text(encoding="utf-8"))
    results = json.loads((output/"results.json").read_text(encoding="utf-8"))
    with np.load(Path(old_results[0]["library_folder"])/"hypotheses.npz") as f:
        old = {k:f[k].copy() for k in f.files}
    with np.load(Path(results[0]["library_folder"])/"hypotheses.npz") as f:
        new = {k:f[k].copy() for k in f.files}
    np.testing.assert_array_equal(old["theta"],new["theta"])
    old_c = np.array([old_category(r) for r in old["reason"]])
    new_c = np.where(new["success"],"success",new["failure_category"])
    categories = ["success","push_timeout","recovery_timeout","planning_clearance_failure",
                  "recovery_failure","crossing_failure","numerical_anomaly"]
    distributions = {k:dict(before=int(np.sum(old_c==k)),after=int(np.sum(new_c==k))) for k in categories}
    bands = []
    for lo,hi in ((.01,2.),(2.,4.),(4.,8.01)):
        mask = (new["theta"][:,1]>=lo)&(new["theta"][:,1]<hi)
        bands.append(dict(friction_range=[lo,hi],N=int(mask.sum()),
            before=dict(Counter(old_c[mask])),after=dict(Counter(new_c[mask]))))
    regression = np.flatnonzero(old["success"] & ~new["success"])
    rescued = np.flatnonzero(~old["success"] & new["success"])
    both = old["success"] & new["success"]
    delta = new["J_door"][both]-old["J_door"][both]
    probe_diff = float(np.max(np.abs(old["z_hat"]-new["z_hat"])))
    assert probe_diff < 1e-9,"standard Probe changed"
    support0 = new["J_door"]<new["J_bypass"]
    old_support = old["J_door"]<old["J_bypass"]
    payload = dict(old_output=old_output,new_output=output,old_signature=str(old["signature"]),
        new_signature=str(new["signature"]),distributions=distributions,friction_bands=bands,
        regressed_ids=regression,rescued_ids=rescued,probe_max_absolute_difference=probe_diff,
        regression_details=[dict(id=int(i),theta=new["theta"][i],before_time_s=old["J_door"][i],
                                 reason=new["reason"][i],elapsed_s=new["elapsed_time"][i],
                                 recovery_exit=new["recovery_exit"][i],wall_peak_N=new["route_wall_peak"][i])
                            for i in regression],
        before_door_support=int(old_support.sum()),after_door_support=int(support0.sum()),
        successful_time_delta_quantiles_s=np.quantile(delta,[0,.5,1]) if len(delta) else [],
        time_quantiles_s=np.quantile(new["J_door"][new["success"]],[0,.5,1]),
        wall_contact_worlds=int(np.sum(new["route_wall_peak"]>.1)),
        successful_wall_contact_worlds=int(np.sum(new["success"] & (new["route_wall_peak"]>.1))),
        cases=[dict(case=r["case"],prior=r["prior"],posterior=r.get("posterior"),
                    final_action=r["final_action"],time=r["total_time_to_M_s"],
                    success=r["mission_success"],tail_success=r["shared_tail_success"]) for r in results])
    payload["continuations"] = {}
    for r in results:
        if not r["probe_used"]:
            continue
        with np.load(Path(r["continuation_cache"])/"continuation_costs.npz",allow_pickle=False) as data:
            payload["continuations"][r["case"]] = {
                route:dict(Counter(np.where(data[route+"_success"],"success",data[route+"_category"])))
                for route in ("door","bypass")}
    save_json(output/"recovery_audit.json",payload)
    rows = ["# Recovery control comparison", "", f"Old: `{old_output}`",f"New: `{output}`", "",
            "| Outcome | Before | After |","| --- | ---: | ---: |"]
    rows += [f"| {k} | {v['before']} | {v['after']} |" for k,v in distributions.items()]
    rows += ["",f"Door support: {old_support.sum()} -> {support0.sum()} / 1024.",
             f"Standard Probe maximum absolute difference: {probe_diff}.",
             f"Previously successful but now failed IDs: {regression.tolist()}.",
             f"Rescued worlds: {len(rescued)}. Wall contact >0.1 N worlds: {payload['wall_contact_worlds']}.","",
             "| Case | Prior p_D | Prior D | Initial | Final p_D | Final D | Final | Time to M / s |", "| --- | ---: | ---: | --- | ---: | ---: | --- | ---: |"]
    for r in results:
        p = r.get("posterior",r["prior"])
        rows.append(f"| {r['case']} | {r['prior']['p_D']:.9f} | {r['prior']['D']:.9f} | {r['prior']['action']} | {p['p_D']:.9f} | {p['D']:.9f} | {r['final_action']} | {r['total_time_to_M_s']:.3f} |")
    (output/"recovery_audit.md").write_text("\n".join(rows)+"\n",encoding="utf-8")
    print(json.dumps(payload,ensure_ascii=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x),indent=2))


if __name__ == "__main__":
    main()
