from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import csv
import hashlib
import json
import shutil
import time
from . import runtime
import numpy as np
from .config import ROOT
from .map import MapSpec, geometry
from .sampling import check_truth
from .belief import support,decide
from .physics import World,prepare
from .library import continuation_labels,atomic_npz
from .live_belief import LiveBelief

CASES = {
    "A": dict(theta=(.071,.035),description="low hinge resistance; large geometric shortcut",map=MapSpec()),
    "B": dict(theta=(.673,6.317),description="large hinge resistance; large geometric shortcut",map=MapSpec()),
    "C": dict(theta=(.317,2.735),description="valuable shortcut and initial decision disagreement",map=MapSpec()),
    "D": dict(theta=(.437,2.473),description="same physical uncertainty; low geometric value",map=MapSpec("small_shortcut",1.05)),
    "E": dict(theta=(.437,2.473),description="interior off-grid truth; consistency weighting",map=MapSpec())
}


def json_safe(value):
    if isinstance(value,dict):
        return {k:json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple,np.ndarray)):
        return [json_safe(v) for v in value]
    if isinstance(value,(np.bool_,bool)):
        return bool(value)
    if isinstance(value,(np.integer,int)):
        return int(value)
    if isinstance(value,(np.floating,float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value,Path):
        return str(value)
    return value


def save_json(path,value):
    path.write_text(json.dumps(json_safe(value),ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")


def save_trace(path,trace):
    with path.open("w",newline="",encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time_s","stage","x_m","y_m","z_m","door_normal_force_N","door_angle_rad","speed_m_s"])
        writer.writerows(trace)


def continue_cached(spec,cfg,state,library):
    digest = hashlib.sha256(library["signature"].encode())
    for key in ("qpos","qvel","act"):
        digest.update(state[key].tobytes())
    digest.update(str((state["time"],state["yaw_reference"])).encode())
    cache = ROOT/"results"/"continuations"/digest.hexdigest()[:20]
    cache.mkdir(parents=True,exist_ok=True)
    path = cache/"continuation_costs.npz"
    if path.exists():
        with np.load(path,allow_pickle=False) as data:
            if ("signature" in data and str(data["signature"]) == library["signature"]
                    and np.array_equal(data["theta"],library["theta"])
                    and "done" in data and np.all(data["done"])
                    and not np.any(np.isnan(data["J_door"])) and not np.any(np.isnan(data["J_bypass"]))
                    and not np.any(data["door_category"] == "numerical_anomaly")
                    and not np.any(data["bypass_category"] == "numerical_anomaly")):
                print("Post-Probe continuation cache:",cache,flush=True)
                return data["labels"].copy(),data["J_door"].copy(),data["J_bypass"].copy(),cache
    labels,jd,jb = continuation_labels(spec,cfg,state,library["theta"],cache)
    return labels,jd,jb,cache


def run_case(name,cfg,library,output,show_viewer=False,live=None,probe_library=None,speed=1.):
    case = CASES[name]
    spec = case["map"]
    theta = check_truth(case["theta"],library["theta"],cfg)
    folder = output/("case_"+name)
    folder.mkdir(parents=True,exist_ok=True)
    geo = geometry(spec,cfg)
    weights = np.full(len(library["theta"]),1./len(library["theta"]))
    # A changed map requires its own actual Bypass time. Door approach/passage geometry is identical.
    evaluator,start,approach,approach_time = prepare(spec,cfg)
    ground_door = evaluator.route("door",approach,theta,approach_time)
    ground_bypass = evaluator.route("bypass",start,theta)
    if not ground_bypass.success:
        raise RuntimeError("ground-truth Bypass failed")
    if spec == MapSpec():
        bypass_prediction = float(library["J_bypass"][0])
    else:
        bypass_prediction = evaluator.route("bypass",start,library["theta"][0]).time
    labels0 = (library["J_door"] < bypass_prediction).astype(int)
    p0,d0 = support(weights,labels0)
    action0,reason0 = decide(geo["G"],p0,d0,cfg)
    print(f"\nCase {name}: {case['description']}\n  G={geo['G']:.3f}m p_D={p0:.6f} D={d0:.6f} -> {action0}",flush=True)
    if action0 == "Probe" and probe_library is None:
        from .probe_stream import build_probe_stream
        probe_library = build_probe_stream(MapSpec(),cfg,library)
    # Ground-truth parameters are only used to construct/evaluate this physical world.
    # The decision functions receive G, weights, action labels and contact observation only.
    world = World(spec,cfg)
    world.restore(start,theta)
    owned_live = show_viewer and live is None
    if owned_live:
        from .live_views import LiveViews
        live = LiveViews(speed=speed)
    observer = LiveBelief(cfg,library,probe_library,labels0,geo["G"])
    observer.action = action0
    observer.records[0]["time"] = float(start["time"])
    def on_world(current):
        observer.on_world(current)
        if live:
            live.on_world(current)
    world.callback = on_world
    world.trace = []
    if live:
        live.begin_case(name,world,observer,theta,folder)
    result = dict(case=name,description=case["description"],config=asdict(cfg),map=asdict(spec),
                  theta_truth_evaluation_only=theta,truth_is_off_grid=True,N=len(weights),
                  geometry=geo,prior=dict(p_D=p0,D=d0,action=action0,reason=reason0),
                  ground_truth_door=asdict(ground_door),ground_truth_bypass=asdict(ground_bypass),
                  oracle_action="Push" if ground_door.time < ground_bypass.time else "Avoid",
                  library_signature=library["signature"],library_folder=library["folder"],
                  probe_used=False)
    labels_after,posterior = labels0,weights
    try:
        if action0 == "Probe":
            # Replaying approach physically makes the demonstration continuous from S.
            if not world.fly([spec.approach]):
                raise RuntimeError("true world failed the standard approach")
            observer.begin_probe(world)
            z,wall = world.probe()
            observed_state = world.snapshot()
            probe_time = float(observed_state["time"]-start["time"])
            save_trace(folder/"probe_trace.csv",world.trace)
            posterior,error = observer.finish_probe(z,world.data.time)
            if live:
                live.update(world,force=True)
            pw,dw = support(posterior,labels0)
            print(f"  z*=[{z[0]:.6f} N, {z[1]:.6f} rad]; weights-only p_D={pw:.6f}, D={dw:.6f}",flush=True)
            # Keep the GUI responsive while models predict remaining routes; never advance the real world here.
            if live and live.running:
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=1) as prediction_thread:
                    future = prediction_thread.submit(continue_cached,spec,cfg,observed_state,library)
                    while not future.done():
                        live.waiting("正在计算 / 读取 Probe 后路线预测；物理世界保持当前状态")
                        time.sleep(.03)
                    labels_after,jd,jb,continuation_folder = future.result()
            else:
                labels_after,jd,jb,continuation_folder = continue_cached(spec,cfg,observed_state,library)
            shutil.copy2(continuation_folder/"continuation_costs.csv",folder/"continuation_costs.csv")
            p1,d1 = support(posterior,labels_after)
            final,reason = decide(geo["G"],p1,d1,cfg,probe_used=True)
            observer.set_continuation(labels_after,final,world.data.time)
            if live:
                live.update(world,force=True)
            print(f"  state-conditioned p_D={p1:.6f}, D={d1:.6f} -> {final} ({reason})",flush=True)
            world.trace = []
            mission = world.route("door" if final=="Push" else "bypass",observed_state,theta)
            prefix = probe_time
            after_truth_door = evaluator.route("door",observed_state,theta)
            after_truth_bypass = evaluator.route("bypass",observed_state,theta)
            oracle_after = "Push" if after_truth_door.time < after_truth_bypass.time else "Avoid"
            spread = np.array([cfg.inertia_max-cfg.inertia_min,cfg.friction_max-cfg.friction_min])
            distance = np.linalg.norm((library["theta"]-theta)/spread,axis=1)
            order = np.argsort(-posterior)
            result.update(probe_used=True,observation=z,probe_wall_peak_N=wall,
                          probe_total_time_s=probe_time,probe_action_duration_s=cfg.probe_duration,
                          posterior=dict(p_D=p1,D=d1,action=final,reason=reason,
                                         p_D_weights_only=pw,D_weights_only=dw,
                                         D_decreased=bool(d1 < d0),resolved=bool(d1 < cfg.disagreement_threshold),
                                         effective_hypotheses=float(1./np.sum(posterior**2)),
                                         consistency_error_prior=float(weights@error),
                                         consistency_error_posterior=float(posterior@error),
                                         normalized_parameter_distance_prior=float(weights@distance),
                                         normalized_parameter_distance_posterior=float(posterior@distance),
                                         mass_radius_0_10_prior=float(weights[distance<=.1].sum()),
                                         mass_radius_0_10_posterior=float(posterior[distance<=.1].sum()),
                                         both_routes_failed_count=int(np.sum(~np.isfinite(jd)&~np.isfinite(jb))),
                                         both_routes_failed_weight=float(posterior[~np.isfinite(jd)&~np.isfinite(jb)].sum()),
                                         top_hypotheses=[dict(id=int(i),theta=library["theta"][i],weight=float(posterior[i]),
                                                              error=float(error[i])) for i in order[:10]]),
                          continuation_cache=continuation_folder,
                          post_probe_state_assumption="ideal observed robot mechanical state + door angle/velocity; I and tau_f never read by policy",
                          oracle_after_probe=oracle_after,
                          ground_truth_door_after_probe=asdict(after_truth_door),
                          ground_truth_bypass_after_probe=asdict(after_truth_bypass),
                          action_matches_conditional_oracle=bool(final == oracle_after))
            atomic_npz(folder/"observed_post_probe_state.npz",**observed_state)
        else:
            final = action0
            observer.action = final
            prefix = 0.
            if final == "Push":
                if not world.fly([spec.approach]):
                    raise RuntimeError("true world failed Door approach")
                mission = world.route("door",world.snapshot(),theta,approach_time)
            else:
                mission = world.route("bypass",start,theta)
            result["action_matches_conditional_oracle"] = bool(final == result["oracle_action"])
        result.update(final_action=final,mission=asdict(mission),
                      total_time_to_M_s=prefix+mission.time,
                      regret_vs_no_probe_oracle_s=prefix+mission.time-min(ground_door.time,ground_bypass.time),
                      mission_success=mission.success,
                      visual_weights_unchanged=True)
        if mission.success:
            # This public shared tail is outside J_door/J_bypass. Same M pose/velocity/yaw tolerances.
            tail_start = world.data.time
            world.yaw_target = 0.
            success_g = world.fly([spec.goal])
            result["shared_tail_success"] = success_g
            result["shared_tail_time_s"] = float(world.data.time-tail_start)
        save_trace(folder/"mission_trace.csv",world.trace)
        observer.save(folder)
        result["live_weight_history"] = dict(path="belief_history.npz",frames=len(observer.records),
            update_method="original prior times consistency with matched cumulative Probe-prefix predictions",
            endpoint_matches_original_update=True,
            formal_push_weights="retained after Probe; execution force/angle/scene continue refreshing")
        with (folder/"weights.csv").open("w",newline="",encoding="utf-8-sig") as stream:
            writer = csv.writer(stream)
            writer.writerow(["id","I","tau_f","w_initial","w_visual","w_after","y_before","y_after"])
            for i in range(len(weights)):
                writer.writerow([i,*library["theta"][i],weights[i],weights[i],posterior[i],labels0[i],labels_after[i]])
        save_json(folder/"summary.json",result)
        if live:
            live.finish_case(result)
        print(f"  M success={mission.success}; total={prefix+mission.time:.3f}s; no-Probe oracle regret={result['regret_vs_no_probe_oracle_s']:.3f}s",flush=True)
        return result
    finally:
        if owned_live:
            live.close()


def new_output():
    # Local client date/time, explicitly UTC+8 instead of depending on host configuration.
    from datetime import timezone,timedelta
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
    output = ROOT/"results"/stamp
    output.mkdir(parents=True,exist_ok=True)
    return output
