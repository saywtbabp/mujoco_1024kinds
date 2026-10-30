"""Matched-budget, transition and posterior geometry analysis of saved profiles."""
from itertools import combinations
from pathlib import Path
import csv
import html
import json
from . import runtime
import numpy as np
from .config import Config
from .experiments import json_safe, save_json
from .force_profiles import ForceProfile
from .multi_probe import write_csv
from .multi_probe_report import load_saved, read_json, validate_saved, fnum, COLORS
from .profile_comparison import profile_budget_audit, file_hash


def axial_difference(a, b):
    """PCA axes are unoriented: 89 and -89 degrees differ by 2, not 178."""
    return abs((a-b+90.) % 180.-90.)


def direction_reliable(metric, settings):
    ratio = metric["normalized_geometry"]["anisotropy_ratio"]
    return (metric["ESS"] >= settings["direction_min_ESS"] and
            (ratio is None or ratio >= settings["direction_min_anisotropy"]))


def pair_differences(data):
    settings = data["snapshot"]["settings"]
    rows, summary = [], []
    for case, histories in data["cases"].items():
        for a, b in combinations(settings["probes"], 2):
            ha, hb = histories[a["id"]], histories[b["id"]]
            pair_rows = []
            for j, t in enumerate(ha["times"]):
                ma, mb = ha["metrics"][j], hb["metrics"][j]
                row = dict(case=case, reference_id=a["id"], candidate_id=b["id"], time_s=t,
                           both_matched_budget=a["matched_budget"] and b["matched_budget"],
                           L1=float(np.sum(abs(np.asarray(ha["weights"][j])-hb["weights"][j]))),
                           abs_D_difference=abs(ma["D"]-mb["D"]),
                           normalized_axis_difference_deg=axial_difference(ma["normalized_geometry"]["angle_deg"], mb["normalized_geometry"]["angle_deg"]),
                           reliable_axes=direction_reliable(ma, settings) and direction_reliable(mb, settings),
                           normalized_mean_distance=float(np.linalg.norm(np.asarray(ma["normalized_geometry"]["mean"])-mb["normalized_geometry"]["mean"])))
                rows.append(row); pair_rows.append(row)
            reliable = [r for r in pair_rows if r["reliable_axes"]]
            peak = max(pair_rows, key=lambda r: r["L1"])
            summary.append(dict(case=case, reference_id=a["id"], candidate_id=b["id"],
                                both_matched_budget=a["matched_budget"] and b["matched_budget"],
                                final_L1=pair_rows[-1]["L1"], max_L1=peak["L1"], max_L1_time_s=peak["time_s"],
                                final_abs_D_difference=pair_rows[-1]["abs_D_difference"],
                                max_abs_D_difference=max(r["abs_D_difference"] for r in pair_rows),
                                final_normalized_axis_difference_deg=pair_rows[-1]["normalized_axis_difference_deg"],
                                final_axes_reliable=pair_rows[-1]["reliable_axes"],
                                max_reliable_axis_difference_deg=max((r["normalized_axis_difference_deg"] for r in reliable), default=None)))
    return rows, summary


def transition_rows(data):
    settings = data["snapshot"]["settings"]
    rows = []
    for case, histories in data["cases"].items():
        for definition in settings["profiles"]:
            p = ForceProfile.from_dict(definition)
            h = histories[p.id]
            for switch in p.switch_times:
                j = int(np.argmin(abs(np.asarray(h["times"])-switch)))
                assert abs(h["times"][j]-switch) < 1e-9 and 0 < j < len(h["times"])-1
                pre, at, post = (h["metrics"][k] for k in (j-1, j, j+1))
                rows.append(dict(case=case, profile_id=p.id, switch_time_s=switch,
                                 force_before_N=p.force_limit_at(h["times"][j-1]), force_after_N=p.force_limit_at(switch),
                                 pre_time_s=h["times"][j-1], post_time_s=h["times"][j+1],
                                 D_pre=pre["D"], D_at=at["D"], D_post=post["D"],
                                 D_slope_before=(at["D"]-pre["D"])/settings["sample_interval_s"],
                                 D_slope_after=(post["D"]-at["D"])/settings["sample_interval_s"],
                                 ESS_pre=pre["ESS"], ESS_at=at["ESS"], ESS_post=post["ESS"],
                                 normalized_angle_pre=pre["normalized_geometry"]["angle_deg"],
                                 normalized_angle_post=post["normalized_geometry"]["angle_deg"],
                                 axis_change_deg=axial_difference(pre["normalized_geometry"]["angle_deg"],post["normalized_geometry"]["angle_deg"]),
                                 reliable_axes=direction_reliable(pre,settings) and direction_reliable(post,settings),
                                 L1_at_to_post=float(np.sum(abs(np.asarray(h["weights"][j+1])-h["weights"][j]))),
                                 delta_theta_pre=h["observation"][j-1][1], delta_theta_post=h["observation"][j+1][1],
                                 door_omega_pre=h["diagnostics"][j-1][1], door_omega_post=h["diagnostics"][j+1][1]))
    return rows


def validate_profile_saved(output):
    output = Path(output)
    data = load_saved(output)
    snapshot, settings = data["snapshot"], data["snapshot"]["settings"]
    cfg = Config(**snapshot["base_config"])
    profiles = [ForceProfile.from_dict(p).validate(cfg.timestep) for p in settings["profiles"]]
    budgets = profile_budget_audit(profiles, cfg.timestep)
    assert budgets == read_json(output/"budget_validation.json")
    audit = validate_saved(output, summary_name="profile_summary")
    for p in profiles:
        with np.load(output/"candidate_responses"/p.id/"responses.npz", allow_pickle=False) as saved:
            assert str(saved["profile_signature"]) == p.signature
            assert str(saved["profile_id"]) == p.id
            assert np.isfinite(saved["diagnostics"]).all()
            assert saved["diagnostics"].shape == (1024, len(saved["times"]), 6)
        for case, histories in data["cases"].items():
            h, folder = histories[p.id], output/("case_"+case)/p.id
            assert h["profile_signature"] == p.signature
            with np.load(folder/"actual_response.npz", allow_pickle=False) as saved:
                assert str(saved["profile_signature"]) == p.signature
                np.testing.assert_array_equal(saved["diagnostics"], h["diagnostics"])
                command = saved["commands"].copy()
            ticks = np.arange(round(p.duration_s/cfg.timestep))*cfg.timestep
            np.testing.assert_array_equal(command[:,0], ticks)
            np.testing.assert_array_equal(command[:,1], [p.force_limit_at(t) for t in ticks])
            assert np.all(command[:,2] <= command[:,1]+1e-12)
            np.testing.assert_allclose(command[:,1].sum()*cfg.timestep, p.command_force_budget_Ns, atol=1e-12, rtol=0)
            for filename, array in (("command_trace.csv", command), ("diagnostics.csv", np.asarray(h["diagnostics"]))):
                with (folder/filename).open(encoding="utf-8-sig") as f:
                    rows = list(csv.DictReader(f))
                np.testing.assert_array_equal([[float(v) for v in r.values()] for r in rows], array)
            with (folder/"physical_response.csv").open(encoding="utf-8-sig") as f:
                rows = list(csv.DictReader(f))
            for j, row in enumerate(rows):
                assert float(row["time_s"]) == h["times"][j]
                assert float(row["force_limit_N"]) == p.force_limit_at(h["times"][j])
                assert float(row["command_budget_sofar_Ns"]) == p.budget_until(h["times"][j])
                assert float(row["F_peak_sofar_N"]) == h["observation"][j][0]
                assert float(row["delta_theta_rad"]) == h["observation"][j][1]
                assert float(row["door_omega_rad_s"]) == h["diagnostics"][j][1]
                assert float(row["contact_force_N"]) == h["diagnostics"][j][2]
    with np.load(output/"candidates.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["labels"], (saved["J_door"] < saved["J_bypass"]).astype(int))
    for filename, expected in settings["protected_input_hashes"].items():
        assert file_hash(filename) == expected, "historical file changed: "+filename
    old_case_regression = []
    # Compare against every available prior magnitude experiment with the same candidate set.
    for filename in settings["protected_input_hashes"]:
        path = Path(filename)
        if path.name != "belief_history.json" or path.parent.name != "probe_standard_2p2N":
            continue
        case = path.parent.parent.name.removeprefix("case_")
        if case not in data["cases"]:
            continue
        old, current = read_json(path), data["cases"][case]["standard_2p2N"]
        np.testing.assert_array_equal(old["weights"], current["weights"])
        np.testing.assert_allclose(old["times"], current["times"], atol=1e-12, rtol=0)
        np.testing.assert_array_equal(old["observation"], current["observation"])
        assert old["t_clear"] == current["t_clear"] and old["t_confirmed"] == current["t_confirmed"]
        old_case_regression.append(dict(case=case, source=str(path), max_weight_difference=0., passed=True))
    audit.update(profile_identity_and_segments=True, command_switches_checked_at_every_physics_step=True,
                 matched_budgets_passed=True, fixed_original_labels=True, candidate_physical_diagnostics=True,
                 physical_response_csv_agreement=True, historical_files_unchanged=True,
                 protected_historical_file_count=len(settings["protected_input_hashes"]),
                 standard_case_regression=old_case_regression,
                 likelihood=dict(force_scale=cfg.force_scale, angle_scale=cfg.angle_scale, alpha=cfg.alpha),
                 D_clear=settings["D_clear"], persistence_frames=settings["persistence_frames"])
    save_json(output/"validation.json", audit)
    return audit


def research_answers(data, pairs):
    settings, cases = data["snapshot"]["settings"], data["cases"]
    labels = {p["id"]:p["label"] for p in settings["probes"]}
    matched = [p["id"] for p in settings["probes"] if p["matched_budget"]]
    answers = []
    matched_pairs = [p for p in pairs if p["both_matched_budget"]]
    answers.append("Q1 不同 profile 的 posterior 是否不同：同预算组完整 1.20 s 的最大 L1 差为 "
                   f"{max(p['final_L1'] for p in matched_pairs):.4f}（范围 0–2）。下面按 Case 列出同期与终点差异。")
    for c, hs in cases.items():
        related = [p for p in matched_pairs if p["case"]==c and p["reference_id"]=="constant_1p6N"]
        detail = "；".join(f"{labels[p['candidate_id']]}: final L1={p['final_L1']:.4f}, max L1={p['max_L1']:.4f}" for p in related)
        answers.append(f"Q2 Case {c}，相同完整预算 1.92 N·s，Constant 1.6 N 对各 profile：{detail}。")
    sequence = [p for p in matched_pairs if p["reference_id"]=="low_to_high_1p0_2p2" and p["candidate_id"]=="high_to_low_2p2_1p0"]
    answers.append("Q3 仅交换高低力的时间顺序："+"；".join(f"Case {p['case']}: final L1={p['final_L1']:.4f}，max |ΔD|={p['max_abs_D_difference']:.4f}" for p in sequence)+"。两者完整预算及力值/持续时间相同，中途累计预算不同。")
    faster, near = [], []
    for c, hs in cases.items():
        base = hs["standard_2p2N"]
        for pid in matched:
            h=hs[pid]
            if h["t_clear"] is None or base["t_clear"] is None:
                continue
            delta = h["t_clear"]-base["t_clear"]
            note = "" if h["summary"]["final_action"]==base["summary"]["final_action"] else "（终点判断与 Standard 不同）"
            if delta < -1e-9:
                faster.append(f"Case {c} {labels[pid]}: {h['t_clear']:.2f} s vs Standard {base['t_clear']:.2f} s"+note)
            if delta <= settings["near_standard_clear_tolerance_s"]+1e-9:
                near.append(f"Case {c} {labels[pid]}: Δt_clear={delta:+.2f} s"+note)
    answers.append("Q4 比 Standard 的实际 early-stop 更早者："+("；".join(faster) if faster else "本次未发现")+"。所有比较均使用连续确认后的稳定区间起点。")
    answers.append(f"Q5 预算从 2.64 降至 1.92 N·s，平均上限从 2.20 降至 1.60 N；与 Standard 相差不超过 {settings['near_standard_clear_tolerance_s']:.2f} s 的 matched profile："+("；".join(near) if near else "无")+"。")
    reliable = [p for p in matched_pairs if p["max_reliable_axis_difference_deg"] is not None]
    if reliable:
        peak=max(reliable,key=lambda p:p["max_reliable_axis_difference_deg"])
        answers.append(f"Q6 域归一化主轴最大可靠夹角差为 {peak['max_reliable_axis_difference_deg']:.2f}°，出现在 Case {peak['case']} 的 {labels[peak['reference_id']]} 与 {labels[peak['candidate_id']]}。"
                       f"以报告阈值 {settings['direction_change_report_deg']:.1f}° 衡量，"+
                       ("观察到方向变化。" if peak["max_reliable_axis_difference_deg"]>=settings["direction_change_report_deg"] else "没有观察到达到该幅度的方向变化。")+
                       f"此比较要求双方 ESS≥{settings['direction_min_ESS']:.0f}、归一化各向异性≥{settings['direction_min_anisotropy']:.1f}，避开近各向同性 prior 和粒子过少时的角度解释。")
    else:
        answers.append("Q6 没有同时满足方向解释条件的 matched profile 帧，不能据 PCA 数字判断辨识方向发生了可靠变化。")
    winners=[]
    for c, hs in cases.items():
        all_best=[h["probe"]["label"] for h in hs.values() if h["summary"]["fastest_overall"]]
        group_best=[h["probe"]["label"] for h in hs.values() if h["summary"]["fastest_matched_budget"]]
        winners.append(f"Case {c}: overall={', '.join(all_best) or '未达到'}；matched={', '.join(group_best) or '未达到'}")
    answers.append("Q7 最快者（保留并列，仅作实验描述）："+"；".join(winners)+"。")
    order_results=[]
    for c, hs in cases.items():
        lo, hi=hs["low_to_high_1p0_2p2"],hs["high_to_low_2p2_1p0"]
        if lo["t_clear"] is not None and (hi["t_clear"] is None or lo["t_clear"]<hi["t_clear"]-1e-9):
            result="Low→High 更早明确"
        elif hi["t_clear"] is not None and (lo["t_clear"] is None or hi["t_clear"]<lo["t_clear"]-1e-9):
            result="High→Low 更早明确"
        else:
            result="明确时间并列" if lo["t_clear"] is not None else "两者均未明确"
        order_results.append(f"Case {c}: {result}（{fnum(lo['t_clear'],2)} / {fnum(hi['t_clear'],2)} s），终点 D={lo['summary']['final_D']:.4g} / {hi['summary']['final_D']:.4g}")
    answers.append("Q8 时间顺序对照："+"；".join(order_results)+"。差异来自本模型下输入时序导致的响应差异，不代表已证明独立辨识 I 和 τ_f。")
    return answers


def line_chart(series, title, xmax=1.2, bounds=None, switches=(), threshold=None, markers=(), angle=False):
    W,H,L,R,T,B=650,255,60,18,30,38
    values=[v for s in series for _,v in s["points"] if v is not None and np.isfinite(v)]
    lo,hi=bounds if bounds else ((min(values),max(values)) if values else (0.,1.))
    if hi-lo<1e-12: hi=lo+1
    if bounds is None:
        pad=.05*(hi-lo);lo-=pad;hi+=pad
    x=lambda t:L+t/xmax*(W-L-R)
    y=lambda v:H-B-(v-lo)/(hi-lo)*(H-T-B)
    parts=[f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(title)}"><title>{html.escape(title)}</title>',
           f'<text x="{L}" y="18" class="chart-title">{html.escape(title)}</text>']
    for v in np.linspace(lo,hi,5):
        parts += [f'<line x1="{L}" x2="{W-R}" y1="{y(v):.3f}" y2="{y(v):.3f}" stroke="#e2e8f0"/>',
                  f'<text x="{L-6}" y="{y(v)+4:.3f}" text-anchor="end" class="tick">{v:.2f}</text>']
    for t in np.linspace(0,xmax,5):
        parts.append(f'<text x="{x(t):.3f}" y="{H-15}" text-anchor="middle" class="tick">{t:.2f}</text>')
    if threshold is not None:
        parts.append(f'<line x1="{L}" x2="{W-R}" y1="{y(threshold):.3f}" y2="{y(threshold):.3f}" stroke="#64748b" stroke-dasharray="4 4"/>')
    for t,label in switches:
        parts.append(f'<line x1="{x(t):.3f}" x2="{x(t):.3f}" y1="{T}" y2="{H-B}" stroke="#475569" stroke-dasharray="3 5"><title>{html.escape(label)}</title></line>')
        parts.append(f'<text x="{x(t)+3:.3f}" y="{T+11}" class="tick">{t:.2f}s switch</text>')
    for s in series:
        chunks,chunk=[],[]
        last=None
        for t,v in s["points"]:
            if v is None or (angle and last is not None and abs(v-last)>90):
                if chunk: chunks.append(chunk)
                chunk=[]
            if v is not None:
                chunk.append(f"{x(t):.5f},{y(v):.5f}")
            last=v
        if chunk: chunks.append(chunk)
        for chunk in chunks:
            parts.append(f'<polyline points="{" ".join(chunk)}" fill="none" stroke="{s["color"]}" stroke-width="{3 if s.get("standard") else 1.8}"><title>{html.escape(s["label"])}</title></polyline>')
    for t,value,color,label in markers:
        parts.append(f'<circle cx="{x(t):.4f}" cy="{y(value):.4f}" r="4.5" fill="white" stroke="{color}" stroke-width="2"><title>{html.escape(label)}</title></circle>')
    parts.append(f'<line class="time-cursor" data-left="{L}" data-right="{W-R}" x1="{L}" x2="{L}" y1="{T}" y2="{H-B}" stroke="#0f172a" stroke-dasharray="3 3" opacity=".45"/></svg>')
    return "".join(parts)


def generate_profile_report(output):
    output=Path(output)
    data=load_saved(output)
    settings=data["snapshot"]["settings"]
    profiles=[ForceProfile.from_dict(p) for p in settings["profiles"]]
    pairs,pair_summary=pair_differences(data)
    transitions=transition_rows(data)
    answers=research_answers(data,pair_summary)
    write_csv(output/"profile_pair_differences.csv",pairs)
    write_csv(output/"profile_pair_summary.csv",pair_summary)
    save_json(output/"profile_pair_differences.json",pairs)
    save_json(output/"profile_pair_summary.json",pair_summary)
    write_csv(output/"transition_analysis.csv",transitions)
    save_json(output/"transition_analysis.json",transitions)
    save_json(output/"research_answers.json",answers)
    (output/"research_answers.md").write_text("\n\n".join(answers)+"\n",encoding="utf-8")
    data["pair_summary"]=pair_summary
    switches=sorted({s for p in profiles for s in p.switch_times})
    switch_labels=[(t," / ".join(p.label for p in profiles if t in p.switch_times)) for t in switches]
    color={p.id:COLORS[i%len(COLORS)] for i,p in enumerate(profiles)}
    force_series=[]
    for p in profiles:
        points=[]
        for s in p.segments: points.extend([(s.start_s,s.force_limit_N),(s.end_s,s.force_limit_N)])
        force_series.append(dict(label=p.label,color=color[p.id],standard=p.standard,points=points))
    force_plot=line_chart(force_series,"F_cmd,max(t) / N · command cap",bounds=(0,2.4),switches=switch_labels)
    legend=''.join(f'<span style="color:{color[p.id]}">━ {html.escape(p.label)}</span>' for p in profiles)
    cards=[]
    for case,histories in data["cases"].items():
        def plot(metric,title,bounds=None,threshold=None,angle=False):
            series,markers=[],[]
            for p in profiles:
                h=histories[p.id]
                if metric=="angle":
                    vals=[m["normalized_geometry"]["angle_deg"] if m["normalized_geometry"]["anisotropy_ratio"] is None or m["normalized_geometry"]["anisotropy_ratio"]>=settings["direction_min_anisotropy"] else None for m in h["metrics"]]
                elif metric in ("major_sd","minor_sd"):
                    vals=[m["normalized_geometry"][metric] for m in h["metrics"]]
                else:
                    vals=[m[metric] for m in h["metrics"]]
                series.append(dict(label=p.label,color=color[p.id],standard=p.standard,points=list(zip(h["times"],vals))))
                if metric in ("D","p_D") and h["clear_index"] is not None:
                    j=h["clear_index"]; markers.append((h["times"][j],vals[j],color[p.id],p.label+" t_clear"))
            return line_chart(series,title,bounds=bounds,switches=switch_labels,threshold=threshold,markers=markers,angle=angle)
        def physical_plot(key,title):
            series=[]
            for p in profiles:
                h=histories[p.id]
                if key=="theta": vals=[v[1] for v in h["observation"]]
                elif key=="peak": vals=[v[0] for v in h["observation"]]
                else: vals=[v[1 if key=="omega" else 2] for v in h["diagnostics"]]
                series.append(dict(label=p.label,color=color[p.id],standard=p.standard,points=list(zip(h["times"],vals))))
            return line_chart(series,title,switches=switch_labels)
        def summary_rows(matched_only):
            rows=[]
            for p in profiles:
                if matched_only and not p.matched_budget: continue
                s=histories[p.id]["summary"]
                badges=("Standard · " if p.standard else "")+("fastest overall · " if s["fastest_overall"] else "")+("fastest matched" if s["fastest_matched_budget"] else "")
                rows.append(f'<tr><td>{html.escape(p.label)}<br><small>{badges}</small></td><td>{p.command_force_budget_Ns:.2f}</td><td>{fnum(s["t_clear"],2)}</td><td>{fnum(s["t_confirmed"],2)}</td>'
                            f'<td>{s["final_p_D"]:.6f}</td><td>{s["final_D"]:.4g}</td><td>{s["final_action"]}</td><td>{s["final_ESS"]:.2f}</td><td>{s["final_normalized_pca_angle_deg"]:.2f}°</td>'
                            f'<td>{s["final_normalized_major_sd"]:.4f} / {s["final_normalized_minor_sd"]:.4f}</td><td>{s["final_normalized_rms_ratio"]:.3f}</td><td>{"Yes" if s["later_relapse"] else "No"}</td>'
                            f'<td><a href="case_{case}/{p.id}/belief_history.json">JSON</a> · <a href="case_{case}/{p.id}/physical_response.csv">physical CSV</a></td></tr>')
            return ''.join(rows)
        table_head='<tr><th>Profile</th><th>Budget / N·s</th><th>t_clear / s</th><th>confirm / s</th><th>final p_D</th><th>final D</th><th>action</th><th>ESS</th><th>normalized PCA</th><th>major / minor SD</th><th>RMS / prior</th><th>relapse</th><th>data</th></tr>'
        options=lambda selected:''.join(f'<option value="{p.id}"{" selected" if p.id==selected else ""}>{html.escape(p.label)}</option>' for p in profiles)
        pair_rows=[p for p in pair_summary if p["case"]==case and p["both_matched_budget"]]
        pair_table=''.join(f'<tr><td>{html.escape(p["reference_id"])}</td><td>{html.escape(p["candidate_id"])}</td><td>{p["final_L1"]:.4f}</td><td>{p["max_L1"]:.4f}</td><td>{p["max_abs_D_difference"]:.4f}</td><td>{p["final_normalized_axis_difference_deg"]:.2f}°</td><td>{fnum(p["max_reliable_axis_difference_deg"],2)}</td></tr>' for p in pair_rows)
        transition_table=''.join(f'<tr><td>{html.escape(r["profile_id"])}</td><td>{r["switch_time_s"]:.2f}</td><td>{r["force_before_N"]:.2f}→{r["force_after_N"]:.2f}</td><td>{r["D_pre"]:.4f} / {r["D_at"]:.4f} / {r["D_post"]:.4f}</td><td>{r["D_slope_before"]:.3f} / {r["D_slope_after"]:.3f}</td><td>{r["ESS_pre"]:.1f}→{r["ESS_post"]:.1f}</td><td>{r["axis_change_deg"]:.2f}° {"" if r["reliable_axes"] else "(低可靠性)"}</td></tr>' for r in transitions if r["case"]==case)
        truth=data["snapshot"]["cases"][case]["theta"]
        cards.append(f'''<section class="case-panel" data-case="{case}"><h2>Case {case} · θ*={truth}</h2>
<div class="controls"><label>Reference Profile <select class="reference-selector">{options('standard_2p2N')}</select></label><label>Candidate Profile <select class="probe-selector">{options('constant_1p6N')}</select></label>
<button class="play">播放权重变化</button><button class="first">初始等权</button><button class="last">1.20 s</button><select class="speed"><option value="200">1×</option><option value="400">0.5×</option><option value="100">2×</option></select></div>
<label class="slider-label">同步时间 <input class="time-slider" type="range" min="0" max="24" step="1" value="0"></label><div class="replay-status"></div><div class="pair-metrics note"></div>
<div class="legend"><span class="push">● Push</span><span class="avoid">● Avoid</span><span>＋真值</span><span>PCA 第一主轴 ±1 SD</span></div>
<div class="grid"><div><h3 class="reference-title">Reference · Standard</h3><svg class="standard-scatter" viewBox="0 0 610 360"></svg><div class="standard-metrics metrics"></div></div><div><h3 class="candidate-title">Candidate</h3><svg class="candidate-scatter" viewBox="0 0 610 360"></svg><div class="candidate-metrics metrics"></div></div></div>
<p class="note">全部 1024 粒子始终显示；两图共用坐标和点大小/透明度尺度。可把 Reference 切到 Constant 1.6 N 直接比较时间结构。真值仅用于仿真与十字显示。</p>
<div class="legend">{legend}</div><div class="grid">{force_plot}{plot('D','D(t) · 空心圆 = t_clear',(0,1),settings['D_clear'])}{plot('p_D','p_D(t)',(0,1))}{plot('angle','Domain-normalized PCA angle(t) / °',(-90,90),angle=True)}{plot('ESS','ESS(t)')}{plot('normalized_rms_ratio','Normalized RMS width / prior')}{plot('major_sd','Normalized major-axis SD')}{plot('minor_sd','Normalized minor-axis SD')}</div>
<p class="note">PCA 是无向轴：夹角按模 180° 计算。角度图在 ±90° 跨界处分段，并隐藏近各向同性帧；低 ESS 下的角度只作描述。可靠方向比较要求 ESS≥{settings['direction_min_ESS']:.0f} 且各向异性≥{settings['direction_min_anisotropy']:.1f}。</p>
<h3>Comparison A · 所有 profile 与 Standard early-stop</h3><div class="scroll"><table>{table_head}{summary_rows(False)}</table></div>
<h3>Matched command-force budget comparison · Comparison B</h3><p>下表四种 profile 的完整预算均为 1.92 N·s、平均上限均为 1.60 N。Standard 的 2.64 N·s 作为独立强激励基线。</p><div class="scroll"><table>{table_head}{summary_rows(True)}</table></div>
<h3>Matched-budget profile pair differences</h3><div class="scroll"><table><tr><th>Reference</th><th>Candidate</th><th>final L1</th><th>max L1</th><th>max |ΔD|</th><th>final axis difference</th><th>max reliable axis difference</th></tr>{pair_table}</table></div>
<h3>Force transition 前后</h3><p class="note">pre=s−0.05，at=s，post=s+0.05。at 时刻响应来自切换前的 [0,s) 输入；切换后的第一帧响应为 post。D 斜率是相邻采样差分，不把切换附近变化直接宣称为独立参数辨识。</p><div class="scroll"><table><tr><th>Profile</th><th>switch / s</th><th>cap / N</th><th>D pre / at / post</th><th>D slope before / after</th><th>ESS pre→post</th><th>axis change</th></tr>{transition_table}</table></div>
<h3>Input → physical response → belief</h3><div class="grid">{physical_plot('theta','Door Δθ(t) / rad')}{physical_plot('omega','Door angular velocity / rad/s')}{physical_plot('contact','Actual contact force / N · 0.05 s samples')}{physical_plot('peak','Cumulative contact F_peak,sofar / N')}</div>
<p class="note">瞬时接触力为 20 Hz 诊断采样，累计 F_peak 在每个 0.002 s 物理步上计算。两者均为实际 MuJoCo 响应；指令 cap 和 B_F 是输入配置，不是 contact impulse。</p></section>''')
    budgets=read_json(output/"budget_validation.json")
    budget_table=''.join(f'<tr><td>{html.escape(p.id)}</td><td>{"；".join(f"[{s.start_s:.2f},{s.end_s:.2f}) s: {s.force_limit_N:.2f} N" for s in p.segments)}</td><td>{p.command_force_budget_Ns:.2f}</td><td>{p.budget_until(p.duration_s)/p.duration_s:.2f}</td></tr>' for p in profiles)
    encoded=json.dumps(json_safe(data),ensure_ascii=False,allow_nan=False,separators=(',',':')).replace('<','\\u003c')
    js=(output/"source_snapshot"/"navlab"/"multi_probe_replay.js").read_text(encoding="utf-8")
    baseline=read_json(output/"baseline_regression.json")
    doc=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Force Profile · Matched Budget Belief Experiment</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#edf2f7;color:#172b42;font:15px/1.65 system-ui,"Microsoft YaHei",sans-serif}}main{{max-width:1480px;margin:auto;padding:28px}}h1{{margin:0;font-size:32px}}h2{{margin-top:0}}h3{{margin-bottom:8px}}section{{background:white;border:1px solid #dbe4ef;border-radius:12px;padding:24px;margin:22px 0}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}svg{{width:100%;display:block}}.controls,.legend{{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin:12px 0}}button,select{{font:inherit;padding:6px 10px;border:1px solid #cbd5e1;border-radius:5px;background:#f8fafc;color:#172b42}}button{{cursor:pointer}}.play{{background:#0f766e;color:white}}.slider-label{{display:flex;align-items:center;gap:15px}}input[type=range]{{flex:1;accent-color:#0f766e}}.metrics{{padding:10px;background:#f1f5f9;border-radius:8px;min-height:120px;font-variant-numeric:tabular-nums}}.replay-status{{font-weight:600;padding:10px 0}}.clear{{background:#dcfce7;color:#166534;padding:2px 5px}}.note,small{{font-size:13px;color:#64748b}}.push{{color:#059669}}.avoid{{color:#dc2626}}.tick{{font-size:11px;fill:#64748b}}.chart-title{{font-size:14px;fill:#172b42;font-weight:600}}.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}}td,th{{padding:9px;border-bottom:1px solid #e2e8f0;text-align:left;white-space:nowrap}}th{{background:#f1f5f9}}a{{color:#0369a1}}li{{margin:12px 0}}@media(max-width:850px){{main{{padding:10px}}section{{padding:14px}}.grid{{grid-template-columns:1fr}}}}
</style></head><body><main><p class="note">1024 SOBOL WORLDS · SAME LIKELIHOOD / LABELS / PRIOR</p><h1>Force 的时间结构是否改变 Belief？</h1><section>
<p>固定原初始状态、接近方式、接触几何、+Y 推进反馈和四旋翼控制，只改变 F_cmd,max(t)。每个 Case × profile 只运行一条完整 1.20 s stream，所有 duration 使用其真实采样前缀。</p>
<p>B_F = ∫ F_cmd,max(t)dt，是水平指令力上限的时间积分。它不是实际接触冲量，也不是实际反馈指令的积分。四个 matched profile 的完整预算相同；在中间时刻已投入的 budget 可以不同，slider 显示各自累计预算。</p>
<table><tr><th>Profile ID</th><th>时间定义</th><th>Budget / N·s</th><th>Mean cap / N</th></tr>{budget_table}</table>
<p>权重每帧由原始 uniform prior 重算，观测仍为 [累计 F_peak, 当前 Δθ]；0.30 N / 0.035 rad，α=2。固定原 y_i，p_D=Σwᵢyᵢ，D=4p_D(1−p_D)。D_clear=0.36、连续 4 帧；t_clear 为稳定区间起点，在线确认时间晚 0.15 s。末尾不足确认帧记 Not reached，后续反复单独标注。</p>
<p>Standard 全库终点最大差 {baseline['terminal_max_absolute_difference']:.3g}，旧 prefix 最大差 {baseline.get('prefix_max_absolute_difference',0):.3g}。所有匹配预算积分均通过检查。原导航任务的 continuation 和决策流程保持原样，本报告固定标签隔离信息效应。</p>
<a href="profile_summary.csv">Summary CSV</a> · <a href="profile_summary.json">JSON</a> · <a href="config_snapshot.json">完整配置</a> · <a href="validation.json">验证</a> · <a href="budget_validation.json">Budget 验证</a> · <a href="profile_pair_summary.csv">Pair differences</a> · <a href="transition_analysis.csv">Transition data</a></section>
<section><h2>实际实验回答 Q1–Q8</h2><ol>{''.join('<li>'+html.escape(a)+'</li>' for a in answers)}</ol><p class="note">这是当前确定性 MuJoCo 模型、固定标签及 0.05 s 采样下的结果；没有测量噪声重复实验，不作统计显著性或最优算法结论。方向解释阈值仅用于报告，不参与 likelihood、p_D/D 或 Probe 控制。</p></section>
{''.join(cards)}<script type="application/json" id="comparison-data">{encoded}</script><script>{js}</script></main></body></html>'''
    (output/"report.html").write_text(doc,encoding="utf-8")
    embedded=json.loads(doc.split('<script type="application/json" id="comparison-data">',1)[1].split('</script>',1)[0])
    assert embedded==json_safe(data)
    save_json(output/"html_data_validation.json",dict(passed=True,embedded_payload_equals_saved_json=True,
              fields=["weights","time","p_D","D","t_clear","t_confirmed","PCA","ESS","profile","physical_response"]))
    return answers
