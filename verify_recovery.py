"""Physical recovery regression, compared with the preserved pre-change source."""
import argparse
import importlib.util
import json
import time
from dataclasses import asdict
from navlab import runtime
import numpy as np
from navlab.config import ROOT,load_config
from navlab.physics import prepare
from navlab.map import MapSpec
from navlab.sampling import candidate_worlds
from navlab.experiments import save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline",action="store_true")
    parser.add_argument("--ids",default="20,0,4,6,12,16,24,28,32,48,2,1")
    args = parser.parse_args()
    cfg = load_config()
    output = ROOT/"results"/"recovery_validation"
    output.mkdir(parents=True,exist_ok=True)
    make = prepare
    if args.baseline:
        spec = importlib.util.spec_from_file_location("navlab.physics_before",output/"physics_before.py")
        module = importlib.util.module_from_spec(spec)
        import sys
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        make = module.prepare
    world,start,approach,approach_time = make(MapSpec(),cfg)
    theta = candidate_worlds(cfg)
    rows = []
    for i in map(int,args.ids.split(",")):
        began = time.perf_counter()
        samples = []
        def record(w):
            pos,vel,R,gyro = w.observed()
            samples.append(dict(time=float(w.data.time),stage=w.stage,position=pos.tolist(),
                speed=float(np.linalg.norm(vel)),tilt_deg=float(np.degrees(np.arccos(np.clip(R[2,2],-1,1)))),
                gyro=float(np.linalg.norm(gyro)),door_force=w.force,wall_force=w.wall_force,
                angle=float(w.data.qpos[w.qa]),motors=w.ctrl.tolist()))
        world.callback = record
        result = world.route("door",approach,theta[i],approach_time)
        rows.append(dict(id=i,theta=theta[i],result=asdict(result),samples=samples))
        save_json(output/("baseline.json" if args.baseline else "current.json"),rows)
        print(i,theta[i],result,"wall_s",round(time.perf_counter()-began,2),flush=True)


if __name__ == "__main__":
    main()
