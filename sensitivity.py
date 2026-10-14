"""Optional deterministic geometric/consistency sensitivity study (no new model).

Uses the same 1024 default hypotheses and baseline route labels. This isolates
the gate/weighting effects; it does not claim conditional mission performance.
"""
import csv
import multiprocessing
import sys
from dataclasses import replace
from navlab import runtime
import numpy as np
from navlab.config import load_config
from navlab.map import MapSpec,geometry
from navlab.physics import prepare
from navlab.library import build_library
from navlab.belief import support,decide,consistency_update
from navlab.experiments import new_output,save_json


def main():
    cfg = load_config()
    library = build_library(MapSpec(),cfg)
    output = new_output()
    weights = np.ones(len(library["theta"]))/len(library["theta"])
    rows = []
    for edge in (.95,1.05,1.3,1.6,2.,2.4,3.2625,4.2,5.4):
        spec = MapSpec("geometry_sweep",edge)
        geo = geometry(spec,cfg)
        world,start,_,_ = prepare(spec,cfg)
        bypass = world.route("bypass",start,library["theta"][0])
        if not bypass.success or bypass.contact_peak > 1e-8:
            raise RuntimeError("geometry sweep Bypass invalid")
        labels = (library["J_door"] < bypass.time).astype(int)
        p,d = support(weights,labels)
        action,reason = decide(geo["G"],p,d,cfg)
        rows.append(dict(right_block_edge_m=edge,G_m=geo["G"],J_bypass_s=bypass.time,
                         p_D=p,D=d,action=action,reason=reason))
    with (output/"geometry_sweep.csv").open("w",newline="",encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    world,start,approach,_ = prepare(MapSpec(),cfg)
    world.restore(approach,[.437,2.473]); z,_ = world.probe()
    distance = np.linalg.norm((library["theta"]-[.437,2.473])/
                             [cfg.inertia_max-cfg.inertia_min,cfg.friction_max-cfg.friction_min],axis=1)
    rows2 = []
    for factor in (.5,1.,2.):
        for alpha in (.5,1.,2.,4.):
            local_cfg = replace(cfg,force_scale=cfg.force_scale*factor,
                                angle_scale=cfg.angle_scale*factor,alpha=alpha)
            post,_ = consistency_update(weights,library["z_hat"],z,local_cfg)
            p,d = support(post,library["labels"])
            rows2.append(dict(tolerance_multiplier=factor,alpha=alpha,p_D_weights_only=p,
                              D_weights_only=d,parameter_distance=float(post@distance),
                              neighborhood_mass=float(post[distance<=.1].sum())))
    with (output/"consistency_sweep.csv").open("w",newline="",encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows2[0])); writer.writeheader(); writer.writerows(rows2)
    save_json(output/"sensitivity.json",dict(config=cfg.__dict__,library_signature=library["signature"],
                                              geometry=rows,consistency=rows2,z_truth=z,
                                              caveat="weighting uses baseline labels; no conditional mission claim"))
    print("Saved:",output)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    if hasattr(sys.stdout,"reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
