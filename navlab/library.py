"""Offline, versioned forward library and actual MuJoCo route labels."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import csv
import hashlib
import json
import os
import time
from collections import Counter
from . import runtime
import mujoco
import numpy as np
from .config import ROOT
from .sampling import candidate_worlds
from .physics import World, prepare
from .map import geometry

_worker = None


def init_worker(spec,cfg,approach_state,approach_time,mode="baseline",observed_state=None,bypass_state=None):
    global _worker
    _worker = (World(spec,cfg),approach_state,approach_time,mode,observed_state,bypass_state)


def simulate_candidate(item):
    index,theta = item
    world,approach,approach_time,mode,observed,bypass_state = _worker
    if mode == "baseline":
        world.restore(approach,theta)
        observation,wall = world.probe()
        result = world.route("door",approach,theta,approach_time)
        return index,observation,result,wall
    # Ideal observed mechanical state, shared by every hypothesis. No true I or tau_f.
    result = world.route("door",observed,theta)
    bypass = world.route("bypass",observed,theta)
    return index,result,bypass


def parallel_rows(items,initializer_args,workers):
    if workers == 1:
        init_worker(*initializer_args)
        for item in items:
            yield simulate_candidate(item)
        return
    with ProcessPoolExecutor(max_workers=workers,initializer=init_worker,initargs=initializer_args) as pool:
        yield from pool.map(simulate_candidate,items,chunksize=8)


def signature(spec,cfg):
    digest = hashlib.sha256()
    config = asdict(cfg); config.pop("workers",None)
    digest.update(json.dumps(dict(map=asdict(spec),config=config,mujoco=mujoco.__version__,
                                  numpy=np.__version__,schema=3),sort_keys=True).encode())
    for name in ("physics.py","map.py","sampling.py","library.py","experiments.py"):
        digest.update((ROOT/"navlab"/name).read_bytes())
    digest.update((ROOT/"models"/cfg.model_source/"tensegrity_real_model.xml").read_bytes())
    return digest.hexdigest()


def atomic_npz(path,**arrays):
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream,**arrays)
    os.replace(tmp,path)


def build_library(spec,cfg,force=False):
    key = signature(spec,cfg)
    folder = ROOT/"results"/"libraries"/(spec.name+"_"+key[:12])
    folder.mkdir(parents=True,exist_ok=True)
    path = folder/"hypotheses.npz"
    theta = candidate_worlds(cfg)
    n = len(theta)
    geo = geometry(spec,cfg)
    world,start,approach,approach_time = prepare(spec,cfg)
    # Door is never contacted on Bypass, so its hidden mass/friction cannot affect this rollout.
    bypass = world.route("bypass",start,theta[0])
    if not bypass.success or bypass.contact_peak > 1e-8 or bypass.wall_peak > .1:
        raise RuntimeError("baseline Bypass must reach M without touching door or walls: " + str(bypass))
    arrays = dict(theta=theta,z_hat=np.full((n,2),np.nan),J_door=np.full(n,np.nan),
                  success=np.zeros(n,dtype=bool),reason=np.full(n,"",dtype="U180"),
                  failure_category=np.full(n,"",dtype="U40"),elapsed_time=np.zeros(n),
                  recovery_time=np.zeros(n),recovery_success=np.zeros(n,dtype=bool),
                  recovery_exit=np.full((n,7),np.nan),
                  probe_wall_peak=np.zeros(n),route_wall_peak=np.zeros(n),route_contact_peak=np.zeros(n),done=np.zeros(n,dtype=bool))
    if path.exists() and not force:
        with np.load(path,allow_pickle=False) as saved:
            if str(saved["signature"]) != key or not np.array_equal(saved["theta"],theta):
                raise RuntimeError("cache signature/candidates do not match")
            for name in arrays:
                arrays[name] = saved[name].copy()
        print(f"Library cache: {arrays['done'].sum()}/{n} worlds",flush=True)
    def save():
        atomic_npz(path,signature=np.array(key),J_bypass=np.full(n,bypass.time),
                   approach_time=np.array(approach_time),**arrays)
    pending = [(i,theta[i]) for i in range(n) if not arrays["done"][i]]
    started = time.perf_counter()
    if pending:
        print(f"Simulating {len(pending)} candidate worlds; {cfg.workers} worker processes",flush=True)
        completed = int(arrays["done"].sum())
        try:
            for row in parallel_rows(pending,(spec,cfg,approach,approach_time),cfg.workers):
                i,z,result,probe_wall = row
                arrays["z_hat"][i],arrays["J_door"][i] = z,result.time
                arrays["success"][i],arrays["reason"][i] = result.success,result.reason
                arrays["failure_category"][i],arrays["elapsed_time"][i] = result.failure_category,result.elapsed_time
                arrays["recovery_time"][i] = result.recovery.get("duration",0.)
                arrays["recovery_success"][i] = result.recovery.get("success",False)
                if result.recovery.get("checks"):
                    c = result.recovery["checks"]
                    arrays["recovery_exit"][i] = [c["door_force_N"],c["wall_force_N"],c["tilt_deg"],
                                                  c["gyro"],c["speed_m_s"],c["position"][2],float(c["clearance"])]
                arrays["probe_wall_peak"][i],arrays["route_wall_peak"][i] = probe_wall,result.wall_peak
                arrays["route_contact_peak"][i] = result.contact_peak
                arrays["done"][i] = True
                completed += 1
                if completed % 32 == 0 or completed == n:
                    save()
                    print(f"  {completed:4d}/{n}  elapsed={time.perf_counter()-started:.1f}s",flush=True)
        finally:
            save()
    if not np.all(arrays["done"]) or not np.all(np.isfinite(arrays["z_hat"])):
        raise RuntimeError("incomplete library; rerun to resume")
    if np.any(arrays["failure_category"] == "numerical_anomaly"):
        raise RuntimeError("numerical anomalies saved for diagnosis; library is invalid for decisions")
    labels = (arrays["J_door"] < bypass.time).astype(int)
    with (folder/"hypotheses.csv").open("w",newline="",encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id","I_hinge_kg_m2","tau_f_Nm","F_peak_N","Delta_theta_rad",
                         "J_door_s","J_bypass_s","y_i","door_success","door_reason",
                         "probe_wall_peak_N","route_wall_peak_N","failure_category","elapsed_s",
                         "recovery_s","recovery_success","route_contact_peak_N"])
        for i in range(n):
            writer.writerow([i,*theta[i],*arrays["z_hat"][i],arrays["J_door"][i],bypass.time,
                             labels[i],int(arrays["success"][i]),arrays["reason"][i],
                             arrays["probe_wall_peak"][i],arrays["route_wall_peak"][i],
                             arrays["failure_category"][i],arrays["elapsed_time"][i],
                             arrays["recovery_time"][i],int(arrays["recovery_success"][i]),arrays["route_contact_peak"][i]])
    metadata = dict(signature=key,config=asdict(cfg),map=asdict(spec),N=n,geometry=geo,
                    baseline_bypass=asdict(bypass),approach_time_s=approach_time,
                    bypass_reuse="identical no-door-contact rollout shared across physical hypotheses",
                    successful_door_worlds=int(arrays["success"].sum()),
                    door_supported_worlds=int(labels.sum()),
                    failure_categories=dict(Counter(arrays["failure_category"][~arrays["success"]])),
                    route_contact_peak_range_N=[float(arrays["route_contact_peak"].min()),float(arrays["route_contact_peak"].max())],
                    successful_time_quantiles_s=np.quantile(arrays["J_door"][arrays["success"]],[0,.5,1]).tolist() if arrays["success"].any() else [],
                    recovery_exit_columns=["door_force_N","wall_force_N","tilt_deg","gyro_rad_s","speed_m_s","height_m","clearance_ok"],
                    probe_force_range_N=[float(arrays["z_hat"][:,0].min()),float(arrays["z_hat"][:,0].max())],
                    probe_angle_range_rad=[float(arrays["z_hat"][:,1].min()),float(arrays["z_hat"][:,1].max())])
    (folder/"metadata.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"Door support: {labels.sum()}/{n}; J_bypass={bypass.time:.3f}s; G={geo['G']:.3f}m",flush=True)
    return dict(**arrays,J_bypass=np.full(n,bypass.time),labels=labels,geometry=geo,
                start=start,approach=approach,approach_time=approach_time,folder=folder,signature=key)


def continuation_labels(spec,cfg,state,theta,folder):
    n = len(theta)
    jd,jb = np.full(n,np.nan),np.full(n,np.nan)
    sd,sb = np.zeros(n,dtype=bool),np.zeros(n,dtype=bool)
    rd,rb = np.full(n,"",dtype="U180"),np.full(n,"",dtype="U180")
    cd,cb = np.full(n,"",dtype="U40"),np.full(n,"",dtype="U40")
    ed,eb = np.zeros(n),np.zeros(n)
    done = np.zeros(n,dtype=bool)
    key = signature(spec,cfg)
    path = folder/"continuation_costs.npz"
    if path.exists():
        with np.load(path,allow_pickle=False) as saved:
            if (str(saved["signature"]) == key and np.array_equal(saved["theta"],theta)
                    and "done" in saved):
                for name,array in (("J_door",jd),("J_bypass",jb),("door_success",sd),("bypass_success",sb),
                                   ("door_reason",rd),("bypass_reason",rb),("door_category",cd),("bypass_category",cb),
                                   ("door_elapsed",ed),("bypass_elapsed",eb),("done",done)):
                    array[:] = saved[name]
                print(f"  continuation checkpoint: {int(done.sum())}/{n}",flush=True)
    def save():
        atomic_npz(path,theta=theta,J_door=jd,J_bypass=jb,signature=np.array(key),
                   labels=(jd<jb).astype(int),door_success=sd,bypass_success=sb,
                   door_reason=rd,bypass_reason=rb,door_category=cd,bypass_category=cb,
                   door_elapsed=ed,bypass_elapsed=eb,done=done)
    started = time.perf_counter()
    print(f"Recomputing {n} Door/Bypass continuations from the SAME observed post-Probe state",flush=True)
    pending = [(i,theta[i]) for i in range(n) if not done[i]]
    try:
        for row in parallel_rows(pending,(spec,cfg,None,0.,"continuation",state),cfg.workers):
            i,door,bypass = row
            jd[i],jb[i],sd[i],sb[i],rd[i],rb[i] = door.time,bypass.time,door.success,bypass.success,door.reason,bypass.reason
            cd[i],cb[i],ed[i],eb[i] = door.failure_category,bypass.failure_category,door.elapsed_time,bypass.elapsed_time
            done[i] = True
            count = int(done.sum())
            if count % 32 == 0 or count == n:
                save()
                print(f"  continuation {count}/{n} elapsed={time.perf_counter()-started:.1f}s",flush=True)
    finally:
        save()
    if not np.all(done) or np.any(np.isnan(jd)) or np.any(np.isnan(jb)):
        raise RuntimeError("incomplete post-Probe route evaluation")
    labels = (jd < jb).astype(int)
    with (folder/"continuation_costs.csv").open("w",newline="",encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id","I","tau_f","J_door_remaining_s","J_bypass_remaining_s","y_i_after",
                         "door_reason","bypass_reason","door_failure_category","bypass_failure_category",
                         "door_elapsed_s","bypass_elapsed_s"])
        for i in range(n):
            writer.writerow([i,*theta[i],jd[i],jb[i],labels[i],rd[i],rb[i],cd[i],cb[i],ed[i],eb[i]])
    if np.any(cd == "numerical_anomaly") or np.any(cb == "numerical_anomaly"):
        raise RuntimeError("invalid numerical continuations preserved; refusing action labels")
    return labels,jd,jb
