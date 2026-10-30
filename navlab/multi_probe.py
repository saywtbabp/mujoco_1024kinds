"""Isolated Probe-information experiment. Existing navigation policy is untouched.

Fixed baseline route labels deliberately isolate the information effect. There
is no continuation-label recomputation, new policy, or Probe selection policy.
Every posterior uses the original prior and exactly one cumulative observation.
"""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
import csv
import hashlib
import json
import re
import shutil
import time
from . import runtime
import mujoco
import numpy as np
from .belief import consistency_update, support, decide
from .config import ROOT, Config, load_config
from .experiments import CASES, save_json, json_safe
from .library import signature, atomic_npz, build_library
from .map import MapSpec, HINGE
from .physics import World, prepare, NumericalFailure, FlightFailure
from .probe_stream import capture_probe
from .sampling import candidate_worlds, check_truth

_worker = None


def digest(value):
    return hashlib.sha256(json.dumps(json_safe(value), sort_keys=True,
                                    allow_nan=False).encode()).hexdigest()


def load_settings(path=None):
    path = Path(path).resolve() if path else ROOT / "probe_comparison_config.json"
    settings = json.loads(path.read_text(encoding="utf-8"))
    cfg = load_config(path.parent / settings["base_config"])
    validate_settings(settings, cfg)
    return settings, cfg


def validate_settings(settings, cfg):
    if settings["schema"] != 1:
        raise ValueError("unsupported comparison schema")
    # capture_probe observes at the original controller's 25-step cadence.
    cadence = 25 * cfg.timestep
    for key in ("maximum_probe_time_s", "sample_interval_s"):
        value = settings[key]
        if not np.isfinite(value) or value <= 0 or not np.isclose(value/cadence, round(value/cadence)):
            raise ValueError(key + " must be a positive multiple of 25 MuJoCo steps")
    duration, interval = settings["maximum_probe_time_s"], settings["sample_interval_s"]
    if not np.isclose(duration/interval, round(duration/interval)):
        raise ValueError("maximum time must be divisible by sample interval")
    if duration != cfg.probe_duration or cfg.probe_duration != 1.2 or cfg.probe_force_max != 2.2:
        raise ValueError("retain the original 2.20 N / 1.20 s Standard Probe")
    frames = settings["persistence_frames"]
    if not isinstance(frames, int) or not 1 <= frames <= round(duration/interval):
        raise ValueError("invalid persistence_frames")
    if not 0 < settings["D_clear"] <= 1 or not 0 < settings["credible_mass"] < 1:
        raise ValueError("invalid clear threshold or credible mass")
    if (not 0 < settings["broad_posterior_rms_ratio"] <= 1
            or not 0 < settings["broad_posterior_marginal_sd_ratio"] <= 1 or settings["baseline_atol"] <= 0):
        raise ValueError("invalid analysis thresholds")
    probes = settings["probes"]
    ids = [p["id"] for p in probes]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Probe IDs must be unique")
    for p in probes:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", p["id"]):
            raise ValueError("unsafe Probe ID")
        if not np.isfinite(p["force_limit_N"]) or p["force_limit_N"] <= 0:
            raise ValueError("invalid Probe force")
    standard = [p for p in probes if p["standard"]]
    if len(standard) != 1 or standard[0]["force_limit_N"] != cfg.probe_force_max:
        raise ValueError("exactly one original Standard Probe is required")
    if not settings["cases"] or any(c not in ("A", "B", "C", "E") for c in settings["cases"]):
        raise ValueError("comparison cases must be existing Probe cases A/B/C/E")
    return settings


def sample_times(settings):
    return np.arange(round(settings["maximum_probe_time_s"] / settings["sample_interval_s"]) + 1) * settings["sample_interval_s"]


def exact_rows(rows, times):
    """Select actual sampled frames; never interpolate a response in time."""
    indices = [int(np.argmin(abs(rows[:, 0] - t))) for t in times]
    if len(set(indices)) != len(times) or not np.allclose(rows[indices, 0], times, atol=1e-9, rtol=0):
        raise ValueError("Probe observation timestamps do not match the requested time grid")
    result = rows[indices, 1:3].copy()
    if not np.isfinite(result).all() or np.any(np.diff(result[:, 0]) < -1e-9):
        raise NumericalFailure("nonfinite response or decreasing cumulative peak")
    return result


def load_baseline(cfg):
    """Read the old versioned library without rewriting any historical file."""
    spec = MapSpec()
    key = signature(spec, cfg)
    folder = ROOT / "results" / "libraries" / (spec.name + "_" + key[:12])
    path = folder / "hypotheses.npz"
    if not path.exists():
        return build_library(spec, cfg)  # new signature directory only
    with np.load(path, allow_pickle=False) as saved:
        data = {k: saved[k].copy() for k in saved.files}
    theta = candidate_worlds(cfg)
    if str(data["signature"]) != key or not np.array_equal(theta, data["theta"]):
        raise RuntimeError("baseline signature or Sobol candidates mismatch")
    if not data["done"].all() or not np.isfinite(data["z_hat"]).all() or np.any(data["failure_category"] == "numerical_anomaly"):
        raise RuntimeError("baseline library incomplete/invalid; use build_library.py to resume")
    _, start, approach, approach_time = prepare(spec, cfg)
    data.update(labels=(data["J_door"] < data["J_bypass"]).astype(int),
                folder=folder, signature=key, start=start, approach=approach, approach_time=approach_time)
    return data


def initialize_output(output, settings, cfg, library, source_files=()):
    if output is None:
        prefix = "force_profile_comparison_" if settings.get("study") == "force_profile" else "probe_comparison_"
        output = ROOT / "results" / (prefix + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output = Path(output).resolve()
    marker = output / "config_snapshot.json"
    if output.exists() and not marker.exists() and any(output.iterdir()):
        raise ValueError("refusing to write into a non-comparison directory")
    output.mkdir(parents=True, exist_ok=True)
    cases = {c: dict(theta=CASES[c]["theta"], map=asdict(CASES[c]["map"]),
                     description=CASES[c]["description"]) for c in settings["cases"]}
    source_names = ["navlab/physics.py", "navlab/map.py", "navlab/config.py", "navlab/sampling.py",
                    "navlab/library.py", "navlab/experiments.py", "navlab/belief.py", "navlab/probe_stream.py",
                    "navlab/multi_probe.py", "navlab/multi_probe_report.py", "navlab/multi_probe_replay.js",
                    "probe_comparison.py", "probe_comparison_config.json", "experiment_config.json",
                    "test_probe_comparison.py", "verify_probe_report.js", "verify_probe_report_browser.js", "PROBE_COMPARISON.md"]
    source_names += list(source_files)
    source_names += [p.relative_to(ROOT).as_posix() for p in sorted((ROOT/"models"/cfg.model_source).rglob("*")) if p.is_file()]
    hashes = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in source_names}
    snapshot = dict(schema=1, settings=settings, base_config=asdict(cfg), cases=cases,
                    mujoco_version=mujoco.__version__, numpy_version=np.__version__,
                    candidate_library_signature=library["signature"], candidate_count=len(library["theta"]),
                    source_hashes=hashes, probe_configs={p["id"]: asdict(replace(cfg, probe_force_max=p["force_limit_N"],
                        probe_duration=settings["maximum_probe_time_s"])) for p in settings["probes"]},
                    sampling="unscrambled deterministic Sobol; seed recorded but no random sampling or observation noise",
                    labels="fixed original J_door < J_bypass, information-only comparison; no continuation relabeling",
                    posterior="consistency_update(original_uniform_prior, same_probe_predictions_at_t, observation_at_t)",
                    t_clear_definition="start of first run of persistence_frames consecutive samples with D < D_clear and same support side; confirmation at end of run; later relapses recorded",
                    pca="population weighted covariance in physical coordinates; also domain-normalized PCA; axes sign chosen with nonnegative I component",
                    contact=dict(approach_xyz=MapSpec().approach, hinge_xy=HINGE,
                                 nominal_lever_arm_m=float(MapSpec().approach[0]-HINGE[0]),
                                 push_direction_world=[0, 1, 0], contact_geometry="unchanged original tensegrity drone and door"))
    if marker.exists():
        if json.loads(marker.read_text(encoding="utf-8")) != json_safe(snapshot):
            raise ValueError("config/source changed: use a new output directory; previous experiment remains intact")
    else:
        save_json(marker, snapshot)
        save_json(output/"approach_state.json", library["approach"])
        atomic_npz(output/"candidates.npz", theta=library["theta"], labels=library["labels"],
                   J_door=library["J_door"], J_bypass=library["J_bypass"], z_hat_standard=library["z_hat"],
                   signature=np.array(library["signature"]))
        write_csv(output/"candidates.csv", [dict(world_id=i, I=t[0], tau_f=t[1], y=int(library["labels"][i]))
                                            for i, t in enumerate(library["theta"])])
        for name in source_names:
            dest = output/"source_snapshot"/name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/name, dest)
    return output, snapshot


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def cache_manifest(probe, cfg, settings, library, snapshot):
    return dict(schema=1, probe=probe, config=asdict(replace(cfg, probe_force_max=probe["force_limit_N"])),
                duration_s=settings["maximum_probe_time_s"], times=sample_times(settings),
                map=asdict(MapSpec()), contact=snapshot["contact"], timestep=cfg.timestep,
                candidate_library_signature=library["signature"], theta_sha256=hashlib.sha256(library["theta"].tobytes()).hexdigest(),
                common_initial_state_sha256=digest(library["approach"]),
                source_hashes=snapshot["source_hashes"], mujoco=snapshot["mujoco_version"], numpy=snapshot["numpy_version"])


def init_worker(spec, cfg, approach, times):
    global _worker
    _worker = World(spec, cfg), approach, times


def simulate_response(world, approach, theta, times):
    world.restore(approach, theta)
    diagnostics = []
    start = float(world.data.time)
    def observe(current):
        _, _, R, gyro = current.observed()
        diagnostics.append([float(current.data.time-start), float(current.data.qvel[current.da]),
                            float(current.force), float(np.arccos(np.clip(R[2, 2], -1, 1))),
                            float(np.linalg.norm(gyro)), float(current.force > 0)])
    world.callback = observe
    try:
        rows, z, wall = capture_probe(world)  # calls the untouched World.probe()
    finally:
        world.callback = None
    if np.any(world.data.warning.number) or not np.isfinite(world.data.qpos).all() or not np.isfinite(world.data.qvel).all():
        raise NumericalFailure("terminal MuJoCo warning/nonfinite mechanical state")
    prediction = exact_rows(rows, times)
    np.testing.assert_allclose(prediction[-1], z, atol=1e-9, rtol=0)
    # Diagnostics use the same timestamps; t=0 is recorded explicitly.
    return prediction, wall, np.asarray(diagnostics)


def response_worker(item):
    index, theta = item
    world, approach, times = _worker
    try:
        response, wall, _ = simulate_response(world, approach, theta, times)
        return index, response, wall, "ok", ""
    except (NumericalFailure, FlightFailure) as exc:
        category = "numerical_anomaly" if isinstance(exc, NumericalFailure) else "physical_failure"
        return index, None, None, category, str(exc)


def build_response_cache(output, probe, cfg, settings, library, snapshot):
    folder = output/"candidate_responses"/probe["id"]
    folder.mkdir(parents=True, exist_ok=True)
    manifest = cache_manifest(probe, cfg, settings, library, snapshot)
    key = digest(manifest)
    manifest_path = folder/"manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != json_safe(dict(signature=key, **manifest)):
        raise ValueError("existing cache manifest mismatch; refusing to overwrite")
    times, n = sample_times(settings), len(library["theta"])
    world_keys = np.array([digest(dict(cache=key, world_id=i, theta=t)) for i, t in enumerate(library["theta"])])
    path = folder/"responses.npz"
    arrays = dict(predictions=np.full((n, len(times), 2), np.nan), done=np.zeros(n, dtype=bool),
                  wall_peak=np.full(n, np.nan), status=np.full(n, "pending", dtype="U32"),
                  error=np.full(n, "", dtype="U400"))
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if str(saved["signature"]) != key or not np.array_equal(saved["world_keys"], world_keys) or not np.array_equal(saved["times"], times):
                raise ValueError("Probe/world/time cache mismatch; refusing reuse")
            arrays = {name: saved[name].copy() for name in arrays}
    if not manifest_path.exists():
        save_json(manifest_path, dict(signature=key, **manifest))
    def save():
        atomic_npz(path, signature=np.array(key), probe_id=np.array(probe["id"]),
                   force_limit_N=np.array(probe["force_limit_N"]), times=times, world_keys=world_keys, **arrays)
    pending = [(i, t) for i, t in enumerate(library["theta"]) if not arrays["done"][i]]
    print(f"{probe['label']}: {int(arrays['done'].sum())}/{n} cached, {len(pending)} pending", flush=True)
    pcfg = replace(cfg, probe_force_max=probe["force_limit_N"], probe_duration=settings["maximum_probe_time_s"])
    pool = None
    start = time.perf_counter()
    if pending:
        if cfg.workers == 1:
            init_worker(MapSpec(), pcfg, library["approach"], times)
            results = map(response_worker, pending)
        else:
            pool = ProcessPoolExecutor(max_workers=cfg.workers, initializer=init_worker,
                                       initargs=(MapSpec(), pcfg, library["approach"], times))
            results = pool.map(response_worker, pending, chunksize=8)
        try:
            for count, (i, response, wall, status, error) in enumerate(results, 1):
                arrays["status"][i], arrays["error"][i] = status, error
                if status == "ok":
                    arrays["predictions"][i], arrays["wall_peak"][i], arrays["done"][i] = response, wall, True
                else:
                    with (output/"anomalies.jsonl").open("a", encoding="utf-8") as f:
                        f.write(json.dumps(dict(probe_id=probe["id"], world_id=i, theta=library["theta"][i].tolist(),
                                                category=status, message=error), ensure_ascii=False) + "\n")
                if count % 32 == 0 or count == len(pending):
                    save()
                if count % 128 == 0 or count == len(pending):
                    print(f"  {int(arrays['done'].sum())}/{n}, elapsed={time.perf_counter()-start:.1f}s", flush=True)
        finally:
            save()
            if pool:
                pool.shutdown(wait=True, cancel_futures=True)
    if not arrays["done"].all() or not np.isfinite(arrays["predictions"]).all() or np.any(arrays["status"] != "ok"):
        raise RuntimeError("invalid/incomplete Probe cache: see anomalies.jsonl; no posterior will be computed")
    return dict(probe_id=probe["id"], force_limit_N=probe["force_limit_N"], times=times,
                predictions=arrays["predictions"], signature=key, path=path)


def pca_geometry(theta, weights):
    mean = weights @ theta
    centered = theta - mean
    covariance = (centered * weights[:, None]).T @ centered
    eigenvalues, axes = np.linalg.eigh(covariance)
    eigenvalues, axes = np.maximum(eigenvalues[::-1], 0), axes[:, ::-1]
    if axes[0, 0] < 0:
        axes[:, 0] *= -1
    if np.linalg.det(axes) < 0:
        axes[:, 1] *= -1
    return dict(mean=mean, covariance=covariance, eigenvalues=eigenvalues,
                principal_axis=axes[:, 0], secondary_axis=axes[:, 1],
                angle_deg=float(np.degrees(np.arctan2(axes[1, 0], axes[0, 0]))),
                slope=float(axes[1, 0]/axes[0, 0]) if abs(axes[0, 0]) > 1e-12 else None,
                anisotropy_ratio=float(eigenvalues[0]/eigenvalues[1]) if eigenvalues[1] > 1e-30 else None,
                major_sd=float(np.sqrt(eigenvalues[0])), minor_sd=float(np.sqrt(eigenvalues[1])))


def clear_run(times, p, d, threshold, persistence):
    for i in range(1, len(times)-persistence+1):  # t=0 has no Probe evidence
        window = slice(i, i+persistence)
        if np.all(d[window] < threshold) and np.all((p[window] > .5) == (p[i] > .5)):
            confirmation = i+persistence-1
            relapse = (d[confirmation+1:] >= threshold) | ((p[confirmation+1:] > .5) != (p[i] > .5))
            return dict(t_clear=float(times[i]), t_confirmed=float(times[confirmation]),
                        clear_index=i, confirmed_index=confirmation, later_relapse=bool(np.any(relapse)))
    return dict(t_clear=None, t_confirmed=None, clear_index=None, confirmed_index=None, later_relapse=False)


def belief_evolution(stream, observation, prior, theta, labels, cfg, settings):
    """No true door parameters are accepted by this function."""
    if stream["probe_id"] != observation["probe_id"] or stream["force_limit_N"] != observation["force_limit_N"]:
        raise ValueError("real response and prediction Probe IDs/forces must match")
    if not np.array_equal(stream["times"], observation["times"]):
        raise ValueError("real response and prediction timestamps must match exactly")
    if stream["predictions"].shape != (len(prior), len(stream["times"]), 2) or observation["response"].shape != (len(stream["times"]), 2):
        raise ValueError("invalid response dimensions")
    if not np.array_equal(observation["response"][0], [0., 0.]):
        raise ValueError("initial cumulative response must be zero")
    lower = np.array([cfg.inertia_min, cfg.friction_min])
    spans = np.array([cfg.inertia_max-cfg.inertia_min, cfg.friction_max-cfg.friction_min])
    unit_theta = (theta-lower)/spans
    prior_cov = pca_geometry(unit_theta, prior)["covariance"]
    weights, metrics = [], []
    for j, t in enumerate(stream["times"]):
        # One prefix, one update, always from the original prior.
        w, _ = consistency_update(prior, stream["predictions"][:, j], observation["response"][j], cfg)
        p, d = support(w, labels)
        raw, unit = pca_geometry(theta, w), pca_geometry(unit_theta, w)
        order = np.argsort(w)[::-1]
        credible_ids = order[:np.searchsorted(np.cumsum(w[order]), settings["credible_mass"])+1]
        # Existing A/B/C/E have G >= geometry_threshold. Use that same branch.
        action, reason = decide(cfg.geometry_threshold, p, d, cfg, probe_used=True)
        metrics.append(dict(time_s=float(t), p_D=p, D=d, ESS=float(1/np.dot(w, w)),
                            judgment=action if d < cfg.disagreement_threshold else "Uncertain",
                            existing_policy_action=action, existing_policy_reason=reason,
                            geometry=raw, normalized_geometry=unit,
                            normalized_rms_ratio=float(np.sqrt(np.trace(unit["covariance"])/np.trace(prior_cov))),
                            marginal_sd_ratio=np.sqrt(np.diag(unit["covariance"])/np.diag(prior_cov)),
                            credible_count=len(credible_ids), credible_push_count=int(np.sum(labels[credible_ids])),
                            credible_avoid_count=int(np.sum(1-labels[credible_ids])),
                            credible_parameter_min=theta[credible_ids].min(axis=0),
                            credible_parameter_max=theta[credible_ids].max(axis=0)))
        weights.append(w)
    weights = np.asarray(weights)
    np.testing.assert_allclose(weights.sum(axis=1), 1., atol=1e-12, rtol=0)
    np.testing.assert_array_equal(weights[0], prior)
    p, d = np.array([m["p_D"] for m in metrics]), np.array([m["D"] for m in metrics])
    clear = clear_run(stream["times"], p, d, settings["D_clear"], settings["persistence_frames"])
    return dict(weights=weights, metrics=metrics, **clear)


def flat_metrics(metric):
    raw, unit = metric["geometry"], metric["normalized_geometry"]
    out = {k: v for k, v in metric.items() if k not in ("geometry", "normalized_geometry", "marginal_sd_ratio", "credible_parameter_min", "credible_parameter_max")}
    out.update(mu_I=raw["mean"][0], mu_tau_f=raw["mean"][1], cov_II=raw["covariance"][0][0],
               cov_I_tau=raw["covariance"][0][1], cov_tau_tau=raw["covariance"][1][1],
               pca_angle_deg=raw["angle_deg"], pca_slope=raw["slope"], lambda1=raw["eigenvalues"][0],
               lambda2=raw["eigenvalues"][1], anisotropy_ratio=raw["anisotropy_ratio"],
               normalized_pca_angle_deg=unit["angle_deg"], normalized_major_sd=unit["major_sd"],
               normalized_minor_sd=unit["minor_sd"])
    for key in ("marginal_sd_ratio", "credible_parameter_min", "credible_parameter_max"):
        for j, name in enumerate(("I", "tau_f")):
            out[key+"_"+name] = metric[key][j]
    return out


def run_case_probe(output, name, probe, cfg, settings, library, stream):
    case = CASES[name]
    truth = check_truth(case["theta"], library["theta"], cfg)
    pcfg = replace(cfg, probe_force_max=probe["force_limit_N"])
    folder = output/("case_"+name)/probe["id"]
    folder.mkdir(parents=True, exist_ok=True)
    case_key = digest(dict(stream=stream["signature"], case=name, truth=truth, map=asdict(case["map"])))
    actual_path = folder/"actual_response.npz"
    if actual_path.exists():
        with np.load(actual_path, allow_pickle=False) as saved:
            if str(saved["signature"]) != case_key:
                raise ValueError("actual Case/Probe response cache mismatch")
            if (str(saved["probe_id"]) != probe["id"] or float(saved["force_limit_N"]) != probe["force_limit_N"]
                    or not np.array_equal(saved["times"], stream["times"])):
                raise ValueError("cached actual response Probe/time mismatch")
            response = saved["response"].copy()
    else:
        world = World(case["map"], pcfg)
        try:
            response, wall, diagnostics = simulate_response(world, library["approach"], truth, stream["times"])
        except (NumericalFailure, FlightFailure) as exc:
            with (output/"anomalies.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(dict(case=name, probe_id=probe["id"], category=type(exc).__name__, message=str(exc)))+"\n")
            raise
        atomic_npz(actual_path, signature=np.array(case_key), response=response, times=stream["times"],
                   probe_id=np.array(probe["id"]), force_limit_N=np.array(probe["force_limit_N"]), diagnostics=diagnostics,
                   wall_peak=np.array(wall))
        write_csv(folder/"diagnostics.csv", [dict(zip(["time_s", "door_omega_rad_s", "contact_force_N", "uav_tilt_rad",
                                                       "uav_gyro_rad_s", "contact"], row)) for row in diagnostics])
    observation = dict(probe_id=probe["id"], force_limit_N=probe["force_limit_N"], times=stream["times"], response=response)
    prior = np.full(len(library["theta"]), 1/len(library["theta"]))
    history = belief_evolution(stream, observation, prior, library["theta"], library["labels"], cfg, settings)
    final = history["metrics"][-1]
    summary = dict(case=name, probe_id=probe["id"], label=probe["label"], standard=probe["standard"],
                   force_limit_N=probe["force_limit_N"], max_simulated_time_s=float(stream["times"][-1]),
                   **{k: history[k] for k in ("t_clear", "t_confirmed", "later_relapse")},
                   final_p_D=final["p_D"], final_D=final["D"], final_action=final["judgment"],
                   final_ESS=final["ESS"], final_pca_angle_deg=final["geometry"]["angle_deg"],
                   final_normalized_rms_ratio=final["normalized_rms_ratio"])
    payload = dict(case=name, probe_id=probe["id"], probe=probe, times=stream["times"], observation=response,
                   weights=history["weights"], metrics=history["metrics"], summary=summary,
                   t_clear=history["t_clear"], t_confirmed=history["t_confirmed"],
                   clear_index=history["clear_index"], confirmed_index=history["confirmed_index"],
                   later_relapse=history["later_relapse"], cache_signature=stream["signature"])
    save_json(folder/"belief_history.json", payload)
    atomic_npz(folder/"belief_history.npz", times=stream["times"], weights=history["weights"],
               p_D=np.array([m["p_D"] for m in history["metrics"]]), D=np.array([m["D"] for m in history["metrics"]]))
    rows = []
    for i, metric in enumerate(history["metrics"]):
        rows.append(dict(**flat_metrics(metric), F_peak_sofar_N=response[i, 0], delta_theta_rad=response[i, 1],
                         t_clear=history["t_clear"], t_confirmed=history["t_confirmed"]))
    write_csv(folder/"belief_metrics.csv", rows)
    write_csv(folder/"decision_curves.csv", [dict(time_s=m["time_s"], p_D=m["p_D"], D=m["D"]) for m in history["metrics"]])
    write_csv(folder/"weights.csv", [dict(time_s=float(t), **{"w_"+str(i): w for i, w in enumerate(weights)})
                                      for t, weights in zip(stream["times"], history["weights"])])
    print(f"Case {name}, {probe['label']}: t_clear={history['t_clear']}, p_D={final['p_D']:.6f}, D={final['D']:.6f}", flush=True)
    return summary


def baseline_regression(output, standard, library, settings):
    max_final = float(np.max(abs(standard["predictions"][:, -1] - library["z_hat"])))
    np.testing.assert_allclose(standard["predictions"][:, -1], library["z_hat"], atol=settings["baseline_atol"], rtol=0)
    audit = dict(candidate_count=len(library["theta"]), terminal_max_absolute_difference=max_final,
                 terminal_matches=True, fixed_labels_unchanged=True, old_prefix_available=False)
    path = library["folder"]/"probe_prefix.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as old:
            if old["done"].all():
                indices = [int(np.argmin(abs(old["times"]-t))) for t in standard["times"]]
                np.testing.assert_allclose(old["times"][indices], standard["times"], atol=1e-9, rtol=0)
                diff = float(np.max(abs(old["predictions"][:, indices] - standard["predictions"])))
                np.testing.assert_allclose(old["predictions"][:, indices], standard["predictions"], atol=settings["baseline_atol"], rtol=0)
                audit.update(old_prefix_available=True, prefix_max_absolute_difference=diff, prefix_matches=True)
    save_json(output/"baseline_regression.json", audit)
    return audit


def run_comparison(settings, cfg, output=None, library_only=False):
    np.random.seed(settings["random_seed"])
    library = load_baseline(cfg)
    output, snapshot = initialize_output(output, settings, cfg, library)
    print("Comparison output:", output, flush=True)
    streams = {p["id"]: build_response_cache(output, p, cfg, settings, library, snapshot) for p in settings["probes"]}
    standard_id = next(p["id"] for p in settings["probes"] if p["standard"])
    baseline_regression(output, streams[standard_id], library, settings)
    if library_only:
        return output
    summaries = []
    for name in settings["cases"]:
        for probe in settings["probes"]:
            summaries.append(run_case_probe(output, name, probe, cfg, settings, library, streams[probe["id"]]))
    save_json(output/"probe_summary.json", summaries)
    write_csv(output/"probe_summary.csv", summaries)
    from .multi_probe_report import generate_report, validate_saved
    validate_saved(output)
    generate_report(output)
    return output
