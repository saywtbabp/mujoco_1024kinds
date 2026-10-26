"""Recorded physical checks of early braking from baseline and post-Probe states."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict,replace
import argparse
import math
from navlab import runtime
import numpy as np
from navlab.config import ROOT,load_config
from navlab.physics import World,prepare
from navlab.map import MapSpec
from navlab.sampling import candidate_worlds
from navlab.experiments import CASES,save_json

_world = None


def init(cfg):
    global _world
    _world = World(MapSpec(),cfg)


def rollout(job):
    name,theta,state,prefix,kind,trace = job
    samples = []
    def record(w):
        pos,vel,R,gyro = w.observed()
        samples.append(dict(time=float(w.data.time),stage=w.stage,position=pos.copy(),velocity=vel.copy(),
            angle_deg=math.degrees(float(w.data.qpos[w.qa])),omega_rad_s=float(w.data.qvel[w.da]),
            tilt_deg=math.degrees(math.acos(float(np.clip(R[2,2],-1,1)))),
            door_force_N=w.force,wall_force_N=w.wall_force,motors=w.ctrl.copy()))
    _world.callback = record if trace else None
    result = _world.route(kind,state,theta,prefix)
    return dict(name=name,theta=theta,kind=kind,result=asdict(result),samples=samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--brake-angle',type=float,help='diagnostic uniform controller setting')
    parser.add_argument('--out',default='small_after')
    args = parser.parse_args()
    cfg = replace(load_config(),workers=args.workers)
    if args.brake_angle is not None:
        cfg = replace(cfg,push_brake_angle_deg=args.brake_angle).validate()
    _,start,approach,dt = prepare(MapSpec(),cfg)
    theta = candidate_worlds(cfg)
    low = np.flatnonzero(theta[:,1]<2.)
    low = low[np.argsort(theta[low,0])]
    ids = sorted(set([20,312,432,228,984,0,2,4,6,16,28,48,1,3,7,15]
        +low[np.linspace(0,len(low)-1,12).astype(int)].tolist()))
    jobs = [(f'baseline_{i}',theta[i],approach,dt,'door',i in (20,312,432)) for i in ids]
    for name in ('A','B','C','E'):
        world = World(MapSpec(),cfg)
        truth = CASES[name]['theta']
        world.restore(start,truth)
        if not world.fly([world.spec.approach]):
            raise RuntimeError('standard approach failed')
        world.probe()
        state = world.snapshot()
        jobs.append((name+'_truth',truth,state,0.,'door',True))
        for i in (0,20,312,432,2,4,16,1):
            jobs.append((name+f'_candidate_{i}',theta[i],state,0.,'door',False))
    rows = []
    folder = ROOT/'results'/'anticipatory_push_validation'
    folder.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=cfg.workers,initializer=init,initargs=(cfg,)) as pool:
        for row in pool.map(rollout,jobs):
            rows.append(row)
            r = row['result']
            print(row['name'],r['success'],round(r['elapsed_time'],3),r['failure_category'],
                  'brake/resume',r['push'].get('brake_count'),r['push'].get('resume_count'),flush=True)
    save_json(folder/(args.out+'.json'),rows)
    failures = [r for r in rows if r['result']['failure_category'] == 'numerical_anomaly']
    if failures:
        raise RuntimeError('invalid numerical rollout')
    for name in ('baseline_20','baseline_312','baseline_432','A_truth','C_truth','E_truth'):
        row = next(r for r in rows if r['name']==name)
        assert row['result']['success'],row
        assert row['result']['wall_peak']==0.,row
    assert next(r for r in rows if r['name']=='baseline_1')['result']['failure_category']=='push_timeout'


if __name__ == '__main__':
    main()
