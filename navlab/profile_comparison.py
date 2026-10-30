"""Five force profiles built on the existing multi-Probe belief experiment."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import time
from . import runtime
import numpy as np
from .config import ROOT, load_config
from .experiments import CASES, save_json, json_safe
from .force_profiles import ForceProfile, simulate_profile_response, DIAGNOSTIC_COLUMNS, COMMAND_COLUMNS
from .map import MapSpec
from .physics import World, NumericalFailure, FlightFailure
from .library import atomic_npz
from .sampling import check_truth
from .multi_probe import (validate_settings, sample_times, load_baseline, initialize_output, cache_manifest,
                          digest, write_csv, belief_evolution, flat_metrics, baseline_regression)

_worker = None
SOURCE_FILES = ["README.md", "build_probe_stream.py", "force_profile_comparison.py", "force_profile_comparison_config.json",
                "navlab/force_profiles.py", "navlab/profile_comparison.py", "navlab/profile_report.py",
                "test_force_profiles.py", "verify_profile_report.js", "FORCE_PROFILE_COMPARISON.md"]


def load_profile_settings(path=None):
    path = Path(path).resolve() if path else ROOT/"force_profile_comparison_config.json"
    settings = json.loads(path.read_text(encoding="utf-8"))
    cfg = load_config(path.parent/settings["base_config"])
    profiles = [ForceProfile.from_dict(p).validate(cfg.timestep) for p in settings["profiles"]]
    if len(profiles) != 5 or len({p.id for p in profiles}) != 5:
        raise ValueError("this experiment requires exactly the five distinct configured profiles")
    standard = [p for p in profiles if p.standard]
    if (len(standard) != 1 or standard[0].id != "standard_2p2N" or len(standard[0].segments) != 1
            or standard[0].segments[0].force_limit_N != cfg.probe_force_max or standard[0].matched_budget):
        raise ValueError("retain the original constant Standard profile")
    if (cfg.sobol_power, cfg.inertia_min, cfg.inertia_max, cfg.friction_min, cfg.friction_max) != (10, .05, .80, .01, 8.):
        raise ValueError("the original 1024-point physical parameter domain must remain unchanged")
    if (cfg.force_scale, cfg.angle_scale, cfg.alpha, cfg.disagreement_threshold) != (.30, .035, 2., .36):
        raise ValueError("the original likelihood and decision thresholds must remain unchanged")
    if settings["D_clear"] != .36 or settings["persistence_frames"] != 4 or settings["sample_interval_s"] != .05:
        raise ValueError("retain the previous sampling and clear-confirmation settings")
    if len([p for p in profiles if p.matched_budget]) != 4:
        raise ValueError("exactly four matched-budget profiles required")
    for p in profiles:
        if p.duration_s != settings["maximum_probe_time_s"]:
            raise ValueError("all profiles must have the same duration")
        if p.matched_budget and not np.isclose(p.budget_until(p.duration_s), settings["matched_budget_Ns"], atol=1e-12, rtol=0):
            raise ValueError("matched-budget profile has the wrong integral")
    # Compatibility metadata for the previous belief/replay schema; full segments are retained.
    settings["probes"] = [dict(p.as_dict(), force_limit_N=p.peak_limit_N, profile_signature=p.signature) for p in profiles]
    validate_settings(settings, cfg)
    return settings, cfg, profiles


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protected_inputs(library):
    files = list(library["folder"].glob("*"))
    for folder in (ROOT/"results").glob("probe_comparison_*"):
        files += [p for p in folder.rglob("*") if p.is_file()]
    return {str(p.resolve()): file_hash(p) for p in sorted(files) if p.is_file()}


def profile_budget_audit(profiles, timestep):
    rows = []
    for p in profiles:
        ts = np.arange(round(p.duration_s/timestep))*timestep
        discrete = float(sum(p.force_limit_at(t) for t in ts)*timestep)
        analytical = p.budget_until(p.duration_s)
        np.testing.assert_allclose([analytical, discrete], p.command_force_budget_Ns, atol=1e-12, rtol=0)
        rows.append(dict(profile_id=p.id, matched_budget=p.matched_budget,
                         configured_budget_Ns=p.command_force_budget_Ns, analytical_budget_Ns=analytical,
                         timestep_integrated_budget_Ns=discrete, mean_cap_N=analytical/p.duration_s,
                         duration_s=p.duration_s, passed=True))
    return rows


def init_profile_worker(spec, cfg, approach, times, profile):
    global _worker
    _worker = World(spec, cfg), approach, times, profile


def profile_worker(item):
    i, theta = item
    world, approach, times, profile = _worker
    try:
        response, wall, diagnostics, _ = simulate_profile_response(world, approach, theta, times, profile)
        return i, response, wall, diagnostics, "ok", ""
    except (NumericalFailure, FlightFailure) as exc:
        status = "numerical_anomaly" if isinstance(exc, NumericalFailure) else "physical_failure"
        return i, None, None, None, status, str(exc)


def build_profile_cache(output, profile, cfg, settings, library, snapshot):
    probe = next(p for p in settings["probes"] if p["id"] == profile.id)
    manifest = dict(cache_manifest(probe, cfg, settings, library, snapshot),
                    profile=profile.as_dict(), profile_signature=profile.signature,
                    diagnostic_columns=DIAGNOSTIC_COLUMNS)
    key = digest(manifest)
    folder = Path(output)/"candidate_responses"/profile.id
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path, path = folder/"manifest.json", folder/"responses.npz"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != json_safe(dict(signature=key, **manifest)):
        raise ValueError("profile segments/config/cache manifest mismatch; use a new output")
    times, n = sample_times(settings), len(library["theta"])
    world_keys = np.array([digest(dict(cache=key, candidate_id=i, theta=theta)) for i, theta in enumerate(library["theta"])])
    arrays = dict(predictions=np.full((n, len(times), 2), np.nan), diagnostics=np.full((n, len(times), len(DIAGNOSTIC_COLUMNS)), np.nan),
                  wall_peak=np.full(n, np.nan), done=np.zeros(n, dtype=bool),
                  status=np.full(n, "pending", dtype="U32"), error=np.full(n, "", dtype="U400"))
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if (str(saved["signature"]) != key or str(saved["profile_signature"]) != profile.signature
                    or not np.array_equal(saved["world_keys"], world_keys) or not np.array_equal(saved["times"], times)):
                raise ValueError("profile/candidate/time cache mismatch")
            arrays = {k: saved[k].copy() for k in arrays}
    if not manifest_path.exists():
        save_json(manifest_path, dict(signature=key, **manifest))
    def save():
        atomic_npz(path, signature=np.array(key), profile_id=np.array(profile.id), probe_id=np.array(profile.id),
                   profile_signature=np.array(profile.signature), force_limit_N=np.array(profile.peak_limit_N),
                   times=times, world_keys=world_keys, **arrays)
    pending = [(i, theta) for i, theta in enumerate(library["theta"]) if not arrays["done"][i]]
    print(f"{profile.id}: {int(arrays['done'].sum())}/{n} cached, {len(pending)} pending", flush=True)
    pool, started = None, time.perf_counter()
    if pending:
        if cfg.workers == 1:
            init_profile_worker(MapSpec(), cfg, library["approach"], times, profile)
            rows = map(profile_worker, pending)
        else:
            pool = ProcessPoolExecutor(max_workers=cfg.workers, initializer=init_profile_worker,
                                       initargs=(MapSpec(), cfg, library["approach"], times, profile))
            rows = pool.map(profile_worker, pending, chunksize=8)
        try:
            for count, (i, response, wall, diagnostics, status, error) in enumerate(rows, 1):
                arrays["status"][i], arrays["error"][i] = status, error
                if status == "ok":
                    arrays["predictions"][i], arrays["diagnostics"][i], arrays["wall_peak"][i] = response, diagnostics, wall
                    arrays["done"][i] = True
                else:
                    with (Path(output)/"anomalies.jsonl").open("a", encoding="utf-8") as f:
                        f.write(json.dumps(dict(profile_id=profile.id, candidate_id=i, category=status, message=error))+"\n")
                if count % 32 == 0 or count == len(pending):
                    save()
                if count % 128 == 0 or count == len(pending):
                    print(f"  {int(arrays['done'].sum())}/{n}, elapsed={time.perf_counter()-started:.1f}s", flush=True)
        finally:
            save()
            if pool:
                pool.shutdown(wait=True, cancel_futures=True)
    if (not arrays["done"].all() or np.any(arrays["status"] != "ok")
            or not np.isfinite(arrays["predictions"]).all() or not np.isfinite(arrays["diagnostics"]).all()):
        raise RuntimeError("invalid/incomplete profile response library; diagnostics preserved, posterior blocked")
    return dict(probe_id=profile.id, profile_id=profile.id, profile_signature=profile.signature,
                force_limit_N=profile.peak_limit_N, times=times, predictions=arrays["predictions"],
                signature=key, path=path)


def profile_belief_evolution(stream, observation, prior, theta, labels, cfg, settings):
    if stream["profile_signature"] != observation["profile_signature"]:
        raise ValueError("observation and predictions must have identical profile segments, not just the same peak force")
    return belief_evolution(stream, observation, prior, theta, labels, cfg, settings)


def run_case_profile(output, name, profile, cfg, settings, library, stream):
    folder = Path(output)/("case_"+name)/profile.id
    folder.mkdir(parents=True, exist_ok=True)
    case = CASES[name]
    truth = check_truth(case["theta"], library["theta"], cfg)
    case_key = digest(dict(stream=stream["signature"], profile_signature=profile.signature,
                           case=name, truth=truth, map=asdict(case["map"])))
    actual_path = folder/"actual_response.npz"
    if actual_path.exists():
        with np.load(actual_path, allow_pickle=False) as saved:
            if (str(saved["signature"]) != case_key or str(saved["profile_signature"]) != profile.signature
                    or str(saved["probe_id"]) != profile.id or not np.array_equal(saved["times"], stream["times"])):
                raise ValueError("actual Case/profile/time cache mismatch")
            response, diagnostics, commands = (saved[k].copy() for k in ("response", "diagnostics", "commands"))
    else:
        world = World(case["map"], cfg)
        try:
            response, wall, diagnostics, commands = simulate_profile_response(world, library["approach"], truth,
                                                                              stream["times"], profile, True)
        except (NumericalFailure, FlightFailure) as exc:
            with (Path(output)/"anomalies.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(dict(case=name, profile_id=profile.id, category=type(exc).__name__, message=str(exc)))+"\n")
            raise
        atomic_npz(actual_path, signature=np.array(case_key), profile_id=np.array(profile.id), probe_id=np.array(profile.id),
                   profile_signature=np.array(profile.signature), force_limit_N=np.array(profile.peak_limit_N),
                   times=stream["times"], response=response, diagnostics=diagnostics, commands=commands, wall_peak=np.array(wall))
    observation = dict(probe_id=profile.id, profile_signature=profile.signature, force_limit_N=profile.peak_limit_N,
                       times=stream["times"], response=response)
    prior = np.full(len(library["theta"]), 1/len(library["theta"]))
    history = profile_belief_evolution(stream, observation, prior, library["theta"], library["labels"], cfg, settings)
    final = history["metrics"][-1]
    probe = next(p for p in settings["probes"] if p["id"] == profile.id)
    summary = dict(case=name, profile_id=profile.id, probe_id=profile.id, label=profile.label,
                   standard=profile.standard, matched_budget=profile.matched_budget,
                   command_force_budget_Ns=profile.command_force_budget_Ns,
                   mean_command_cap_N=profile.budget_until(profile.duration_s)/profile.duration_s,
                   max_simulated_time_s=profile.duration_s,
                   **{k: history[k] for k in ("t_clear", "t_confirmed", "later_relapse")},
                   final_p_D=final["p_D"], final_D=final["D"], final_action=final["judgment"], final_ESS=final["ESS"],
                   final_normalized_pca_angle_deg=final["normalized_geometry"]["angle_deg"],
                   final_normalized_major_sd=final["normalized_geometry"]["major_sd"],
                   final_normalized_minor_sd=final["normalized_geometry"]["minor_sd"],
                   final_normalized_rms_width=float(np.sqrt(np.trace(final["normalized_geometry"]["covariance"]))),
                   final_normalized_rms_ratio=final["normalized_rms_ratio"])
    write_csv(folder/"diagnostics.csv", [dict(zip(DIAGNOSTIC_COLUMNS, row)) for row in diagnostics])
    write_csv(folder/"command_trace.csv", [dict(zip(COMMAND_COLUMNS, row)) for row in commands])
    physical = [dict(time_s=float(t), force_limit_N=profile.force_limit_at(float(t)),
                     command_budget_sofar_Ns=profile.budget_until(float(t)), F_peak_sofar_N=response[j, 0],
                     delta_theta_rad=response[j, 1], door_omega_rad_s=diagnostics[j, 1], contact_force_N=diagnostics[j, 2])
                for j, t in enumerate(stream["times"])]
    write_csv(folder/"physical_response.csv", physical)
    print(f"Case {name} / {profile.id}: clear={history['t_clear']}, confirmed={history['t_confirmed']}, "
          f"p_D={final['p_D']:.6f}, D={final['D']:.6f}", flush=True)
    return dict(case=name, profile_id=profile.id, probe_id=profile.id, profile_signature=profile.signature, probe=probe,
                profile=profile.as_dict(), times=stream["times"], observation=response, diagnostics=diagnostics,
                command_force_limits=[profile.force_limit_at(float(t)) for t in stream["times"]],
                command_budgets_sofar=[profile.budget_until(float(t)) for t in stream["times"]],
                weights=history["weights"], metrics=history["metrics"], summary=summary,
                **{k: history[k] for k in ("t_clear", "t_confirmed", "clear_index", "confirmed_index", "later_relapse")},
                cache_signature=stream["signature"])


def save_case_history(output, history):
    folder = Path(output)/("case_"+history["case"])/history["profile_id"]
    save_json(folder/"belief_history.json", history)
    atomic_npz(folder/"belief_history.npz", times=history["times"], weights=history["weights"],
               p_D=np.array([m["p_D"] for m in history["metrics"]]), D=np.array([m["D"] for m in history["metrics"]]))
    write_csv(folder/"belief_metrics.csv", [dict(**flat_metrics(m), force_limit_N=history["command_force_limits"][j],
              command_budget_sofar_Ns=history["command_budgets_sofar"][j], F_peak_sofar_N=history["observation"][j, 0],
              delta_theta_rad=history["observation"][j, 1], door_omega_rad_s=history["diagnostics"][j, 1],
              contact_force_N=history["diagnostics"][j, 2],
              normalized_anisotropy_ratio=m["normalized_geometry"]["anisotropy_ratio"],
              t_clear=history["t_clear"], t_confirmed=history["t_confirmed"])
              for j, m in enumerate(history["metrics"])])
    write_csv(folder/"decision_curves.csv", [dict(time_s=m["time_s"], p_D=m["p_D"], D=m["D"]) for m in history["metrics"]])
    write_csv(folder/"weights.csv", [dict(time_s=float(t), **{"w_"+str(i): w for i, w in enumerate(weights)})
                                    for t, weights in zip(history["times"], history["weights"])])


def flag_fastest(histories):
    for matched_only, name in ((False, "fastest_overall"), (True, "fastest_matched_budget")):
        eligible = [h for h in histories if h["t_clear"] is not None and (not matched_only or h["probe"]["matched_budget"])]
        first = min((h["t_clear"] for h in eligible), default=None)
        for h in histories:
            h["summary"][name] = bool(h["t_clear"] is not None and h["t_clear"] == first and
                                      (not matched_only or h["probe"]["matched_budget"]))


def run_profile_comparison(settings, cfg, profiles, output=None, library_only=False):
    np.random.seed(settings["random_seed"])
    library = load_baseline(cfg)  # physics.py/library.py signatures and labels remain untouched
    settings = dict(settings, protected_input_hashes=protected_inputs(library))
    output, snapshot = initialize_output(output, settings, cfg, library, SOURCE_FILES)
    print("Force-profile output:", output, flush=True)
    save_json(output/"budget_validation.json", profile_budget_audit(profiles, cfg.timestep))
    streams = {p.id: build_profile_cache(output, p, cfg, settings, library, snapshot) for p in profiles}
    standard = next(p for p in profiles if p.standard)
    baseline_regression(output, streams[standard.id], library, settings)
    if library_only:
        return output
    summaries = []
    for case in settings["cases"]:
        histories = [run_case_profile(output, case, p, cfg, settings, library, streams[p.id]) for p in profiles]
        flag_fastest(histories)
        for history in histories:
            save_case_history(output, history)
            summaries.append(history["summary"])
    save_json(output/"profile_summary.json", summaries)
    write_csv(output/"profile_summary.csv", summaries)
    from .profile_report import validate_profile_saved, generate_profile_report
    validate_profile_saved(output)
    answers = generate_profile_report(output)
    for answer in answers:
        print(answer, flush=True)
    return output
