"""Self-contained replay/report generated exclusively from saved experiment data."""
from pathlib import Path
import csv
import html
import json
from . import runtime
import numpy as np
from .config import ROOT, Config
from .experiments import save_json, json_safe
from .multi_probe import belief_evolution, flat_metrics

COLORS = ["#2563eb", "#d97706", "#9333ea", "#0f766e", "#be123c"]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_saved(output):
    output = Path(output)
    snapshot = read_json(output/"config_snapshot.json")
    with np.load(output/"candidates.npz", allow_pickle=False) as saved:
        theta, labels = saved["theta"].copy(), saved["labels"].copy()
    data = dict(snapshot=snapshot, theta=theta, labels=labels, cases={})
    for case in snapshot["settings"]["cases"]:
        data["cases"][case] = {p["id"]: read_json(output/("case_"+case)/p["id"]/"belief_history.json")
                               for p in snapshot["settings"]["probes"]}
    return data


def validate_saved(output):
    """Recompute saved results from actual/predicted responses, including all CSVs."""
    output = Path(output)
    data = load_saved(output)
    snapshot, theta, labels = data["snapshot"], data["theta"], data["labels"]
    cfg, settings = Config(**snapshot["base_config"]), snapshot["settings"]
    prior = np.full(len(theta), 1/len(theta))
    total_frames = 0
    summaries = []
    for p in settings["probes"]:
        with np.load(output/"candidate_responses"/p["id"]/"responses.npz", allow_pickle=False) as saved:
            assert saved["done"].all() and np.all(saved["status"] == "ok")
            assert str(saved["probe_id"]) == p["id"] and float(saved["force_limit_N"]) == p["force_limit_N"]
            stream = dict(probe_id=str(saved["probe_id"]), force_limit_N=float(saved["force_limit_N"]),
                          times=saved["times"].copy(), predictions=saved["predictions"].copy())
        for case, histories in data["cases"].items():
            history = histories[p["id"]]
            folder = output/("case_"+case)/p["id"]
            with np.load(folder/"actual_response.npz", allow_pickle=False) as saved:
                obs = dict(probe_id=str(saved["probe_id"]), force_limit_N=float(saved["force_limit_N"]),
                           times=saved["times"].copy(), response=saved["response"].copy())
            recomputed = belief_evolution(stream, obs, prior, theta, labels, cfg, settings)
            np.testing.assert_array_equal(history["times"], stream["times"])
            np.testing.assert_array_equal(history["observation"], obs["response"])
            np.testing.assert_array_equal(history["weights"], recomputed["weights"])
            assert history["metrics"] == json_safe(recomputed["metrics"])
            for key in ("t_clear", "t_confirmed", "clear_index", "confirmed_index", "later_relapse"):
                assert history[key] == recomputed[key], key
            with np.load(folder/"belief_history.npz", allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["weights"], history["weights"])
                np.testing.assert_array_equal(saved["times"], history["times"])
                for key in ("p_D", "D"):
                    np.testing.assert_array_equal(saved[key], [m[key] for m in history["metrics"]])
            with (folder/"belief_metrics.csv").open(encoding="utf-8-sig") as f:
                rows = list(csv.DictReader(f))
            assert len(rows) == len(history["times"])
            for row, metric in zip(rows, history["metrics"]):
                for key, value in flat_metrics(metric).items():
                    if isinstance(value, (int, float, np.number)):
                        assert float(row[key]) == value, (case, p["id"], key)
                for key in ("t_clear", "t_confirmed"):
                    assert row[key] == ("" if history[key] is None else str(history[key]))
            with (folder/"weights.csv").open(encoding="utf-8-sig") as f:
                weight_rows = list(csv.DictReader(f))
            np.testing.assert_array_equal([[float(r["w_"+str(i)]) for i in range(len(theta))] for r in weight_rows], history["weights"])
            with (folder/"decision_curves.csv").open(encoding="utf-8-sig") as f:
                curve = list(csv.DictReader(f))
            for row, metric in zip(curve, history["metrics"]):
                for key in ("time_s", "p_D", "D"):
                    assert float(row[key]) == metric[key]
            total_frames += len(rows)
    for case in settings["cases"]:
        summaries += [data["cases"][case][p["id"]]["summary"] for p in settings["probes"]]
    assert summaries == read_json(output/"probe_summary.json")
    with (output/"probe_summary.csv").open(encoding="utf-8-sig") as f:
        saved_summary = list(csv.DictReader(f))
    for row, summary in zip(saved_summary, summaries):
        for key, value in summary.items():
            assert row[key] == ("" if value is None else str(value)), key
    audit = dict(passed=True, cases=len(data["cases"]), probes=len(settings["probes"]),
                 posterior_frames_checked=total_frames, particles_per_frame=len(theta),
                 checks=["original uniform prior", "all posterior sums = 1", "same-Probe/same-time cumulative responses",
                         "every posterior independently recomputed from original prior", "fixed baseline labels",
                         "all metrics/PCA/ESS and clear times recomputed", "JSON/NPZ/CSV agreement",
                         "all candidate rollouts completed without numerical anomalies"])
    save_json(output/"validation.json", audit)
    print("Saved-data validation passed:", audit["posterior_frames_checked"], "posterior frames", flush=True)
    return audit


def fnum(value, digits=3):
    return "Not reached" if value is None else f"{value:.{digits}f}"


def comparison_answers(data):
    settings, cases = data["snapshot"]["settings"], data["cases"]
    standard = next(p["id"] for p in settings["probes"] if p["standard"])
    answers = []
    baseline = "；".join(f"Case {c}: {fnum(h[standard]['t_clear'], 2)}" + (" s" if h[standard]["t_clear"] is not None else "") for c, h in cases.items())
    answers.append("Q1 Standard 首次明确时间："+baseline+"。起点需再经过连续采样确认，确认时间见表。")
    for c, histories in cases.items():
        base = histories[standard]
        reached = [h for h in histories.values() if h["t_clear"] is not None]
        earliest = min((h["t_clear"] for h in reached), default=None)
        fastest = [h["probe"]["label"] for h in reached if h["t_clear"] == earliest]
        faster = [h["probe"]["label"] for h in reached if not h["probe"]["standard"] and
                  base["t_clear"] is not None and h["t_clear"] < base["t_clear"]]
        lower = [h["probe"]["label"] for h in reached if h["probe"]["force_limit_N"] < base["probe"]["force_limit_N"] and
                 h["summary"]["final_D"] < settings["D_clear"] and
                 h["summary"]["final_action"] == base["summary"]["final_action"] and base["summary"]["final_D"] < settings["D_clear"]]
        max_D_spread = float(np.max(np.ptp([[m["D"] for m in h["metrics"]] for h in histories.values()], axis=0)))
        max_l1 = max(float(np.max(np.sum(abs(np.asarray(h["weights"])-base["weights"]), axis=1))) for h in histories.values())
        angles = [h["metrics"][-1]["normalized_geometry"]["angle_deg"] for h in histories.values()]
        widths = [h["metrics"][-1]["normalized_geometry"]["minor_sd"] for h in histories.values()]
        axes = [h["metrics"][-1]["geometry"]["angle_deg"] for h in histories.values()]
        broad = []
        for h in histories.values():
            candidates = [m for m in h["metrics"] if m["D"] < settings["D_clear"] and
                          (m["normalized_rms_ratio"] >= settings["broad_posterior_rms_ratio"] or
                           max(m["marginal_sd_ratio"]) >= settings["broad_posterior_marginal_sd_ratio"])]
            if candidates:
                m = candidates[0]
                broad.append(f"{h['probe']['label']} 在 {m['time_s']:.2f} s：D={m['D']:.4f}，参数 RMS 宽度为 prior 的 {m['normalized_rms_ratio']:.1%}，"
                             f"最宽边际标准差为 prior 的 {max(m['marginal_sd_ratio']):.1%}，ESS={m['ESS']:.1f}，"
                             f"90% 高权重集合的 τ_f 跨度 [{m['credible_parameter_min'][1]:.3f}, {m['credible_parameter_max'][1]:.3f}] N m；"
                             f"其终点 D={h['metrics'][-1]['D']:.3g}，τ_f 仍跨 [{h['metrics'][-1]['credible_parameter_min'][1]:.3f}, {h['metrics'][-1]['credible_parameter_max'][1]:.3f}] N m")
        answers.append(f"Case {c}（Q2–Q6）：最早明确为 {', '.join(fastest) if fastest else '均未达到'}，t={fnum(earliest,2)}；"
                       f"优于 Standard 已分析的 early-stop 起点：{', '.join(faster) if faster else '无'}；"
                       f"更小指令力且最终与 Standard 同样明确、同一方向：{', '.join(lower) if lower else '无'}。"
                       f"同期 D 最大差 {max_D_spread:.4f}，与 Standard 的权重 L1 最大差 {max_l1:.4f}。"
                       f"最终物理坐标 PCA 角 {min(axes):.2f}–{max(axes):.2f}°，归一化坐标 PCA 角 {min(angles):.2f}–{max(angles):.2f}°，"
                       f"归一化短轴标准差 {min(widths):.4f}–{max(widths):.4f}。")
        answers.append(f"Case {c}（Q7）："+("；".join(broad) if broad else "本次未出现达到配置宽度阈值的低 D posterior")+"。")
    fastest_sets = []
    for c, histories in cases.items():
        reached = [h for h in histories.values() if h["t_clear"] is not None]
        earliest = min((h["t_clear"] for h in reached), default=None)
        fastest_sets.append((c, {h["probe_id"] for h in reached if h["t_clear"] == earliest}))
    common = set.intersection(*(ids for _, ids in fastest_sets))
    names = [p["label"] for p in settings["probes"] if p["id"] in common]
    detail = "；".join(c+": "+(", ".join(p["label"] for p in settings["probes"] if p["id"] in ids) or "未达到") for c, ids in fastest_sets)
    answers.append("Q6 各 Case 最快或并列最快的 Probe："+detail+"。共同最快者："+(", ".join(names) or "无")+
                   "。仅在已运行 Cases 与当前采样分辨率内解释。当前为确定性仿真，没有重复噪声试验，因此不作统计显著性结论。")
    return answers


def chart_svg(histories, probes, key, title, settings, bounds=None):
    width, height, left, right, top, bottom = 650, 255, 54, 20, 30, 40
    xmax = settings["maximum_probe_time_s"]
    def get(m):
        return m["normalized_geometry"][key.split(".")[1]] if key.startswith("normalized.") else m[key]
    series = [[get(m) for m in histories[p["id"]]["metrics"]] for p in probes]
    ymin, ymax = bounds if bounds else (min(min(v) for v in series), max(max(v) for v in series))
    if ymax-ymin < 1e-12:
        ymax = ymin + 1
    if bounds is None:
        pad = .05*(ymax-ymin)
        ymin, ymax = ymin-pad, ymax+pad
    x = lambda t: left + t/xmax*(width-left-right)
    y = lambda v: height-bottom - (v-ymin)/(ymax-ymin)*(height-top-bottom)
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}"><title>{html.escape(title)}</title>',
             f'<text x="{left}" y="18" class="chart-title">{html.escape(title)}</text>']
    for v in np.linspace(ymin, ymax, 5):
        parts += [f'<line x1="{left}" x2="{width-right}" y1="{y(v):.3f}" y2="{y(v):.3f}" stroke="#e2e8f0"/>',
                  f'<text x="{left-6}" y="{y(v)+4:.3f}" text-anchor="end" class="tick">{v:.2f}</text>']
    for t in np.linspace(0, xmax, 5):
        parts.append(f'<text x="{x(t):.3f}" y="{height-16}" text-anchor="middle" class="tick">{t:.2f}</text>')
    parts.append(f'<text x="{width-right}" y="{height-2}" text-anchor="end" class="tick">Probe time / s</text>')
    if key == "D":
        parts.append(f'<line x1="{left}" x2="{width-right}" y1="{y(settings["D_clear"]):.3f}" y2="{y(settings["D_clear"]):.3f}" stroke="#64748b" stroke-dasharray="5 4"/>')
    for k, (p, values) in enumerate(zip(probes, series)):
        h = histories[p["id"]]
        points = " ".join(f"{x(t):.5f},{y(v):.5f}" for t, v in zip(h["times"], values))
        parts.append(f'<polyline points="{points}" fill="none" stroke="{COLORS[k%len(COLORS)]}" stroke-width="{3 if p["standard"] else 1.8}"/>')
        if key in ("D", "p_D") and h["t_clear"] is not None:
            j = h["clear_index"]
            parts.append(f'<circle cx="{x(h["t_clear"]):.3f}" cy="{y(values[j]):.3f}" r="5" fill="white" stroke="{COLORS[k%len(COLORS)]}" stroke-width="2"><title>{html.escape(p["label"])} t_clear={h["t_clear"]:.2f} s</title></circle>')
    parts.append(f'<line class="time-cursor" data-left="{left}" data-right="{width-right}" x1="{left}" x2="{left}" y1="{top}" y2="{height-bottom}" stroke="#0f172a" stroke-dasharray="3 3" opacity=".5"/></svg>')
    return "".join(parts)


def generate_report(output):
    output = Path(output)
    data = load_saved(output)
    settings = data["snapshot"]["settings"]
    probes = settings["probes"]
    standard = next(p for p in probes if p["standard"])
    answers = comparison_answers(data)
    save_json(output/"research_answers.json", answers)
    (output/"research_answers.md").write_text("\n\n".join(answers)+"\n", encoding="utf-8")
    sections = []
    for case, histories in data["cases"].items():
        rows = []
        for p in sorted(probes, key=lambda p: (not p["standard"], -p["force_limit_N"])):
            s = histories[p["id"]]["summary"]
            rows.append(f'<tr><td>{html.escape(p["label"])}</td><td>{p["force_limit_N"]:.2f} N</td><td>{s["max_simulated_time_s"]:.2f} s</td>'
                        f'<td>{fnum(s["t_clear"],2)}</td><td>{fnum(s["t_confirmed"],2)}</td><td>{s["final_p_D"]:.6f}</td>'
                        f'<td>{s["final_D"]:.6f}</td><td>{s["final_action"]}</td><td>{s["final_ESS"]:.1f}</td>'
                        f'<td>{"Yes" if s["later_relapse"] else "No"}</td>'
                        f'<td><a href="case_{case}/{p["id"]}/belief_history.json">JSON</a> · <a href="case_{case}/{p["id"]}/belief_metrics.csv">CSV</a></td></tr>')
        options = "".join(f'<option value="{p["id"]}">{html.escape(p["label"])}</option>' for p in probes)
        diffuse = [(m["normalized_rms_ratio"], h["probe_id"], i) for h in histories.values()
                   for i, m in enumerate(h["metrics"]) if m["D"] < settings["D_clear"]]
        showcase = max(diffuse) if diffuse else None
        showcase_button = (f'<button class="showcase" data-probe="{showcase[1]}" data-index="{showcase[2]}">查看低 D、参数仍分散帧</button>' if showcase else '')
        legend = "".join(f'<span style="color:{COLORS[i%len(COLORS)]}">━ {html.escape(p["label"])}</span>' for i, p in enumerate(probes))
        plots = "".join(chart_svg(histories, probes, k, title, settings, bounds) for k, title, bounds in
                        [("D", "D(t) · 空心圆 = t_clear", (0, 1)), ("p_D", "p_D(t) · Push support", (0, 1))])
        geometry_plots = "".join(chart_svg(histories, probes, k, title, settings) for k, title in
                                 [("ESS", "ESS(t)"), ("normalized.angle_deg", "Domain-normalized PCA angle / degree"),
                                  ("normalized.minor_sd", "Domain-normalized minor-axis SD"), ("normalized_rms_ratio", "Parameter RMS width / prior RMS width")])
        truth = data["snapshot"]["cases"][case]["theta"]
        sections.append(f'''<section class="case-panel" data-case="{case}"><h2>Case {case}</h2>
<p>真实 θ* = [{truth[0]}, {truth[1]}]，仅用于 MuJoCo 真值模拟及十字标记。Push / Avoid 颜色始终来自原库固定标签。</p>
<div class="controls"><label>Candidate Probe <select class="probe-selector">{options}</select></label>
<button class="play">播放权重变化</button><button class="first">初始等权</button><button class="last">1.20 s</button>{showcase_button}
<label>速度 <select class="speed"><option value="200">1×</option><option value="400">0.5×</option><option value="100">2×</option></select></label></div>
<label class="slider-label">同步时间 <input class="time-slider" type="range" min="0" max="{len(next(iter(histories.values()))['times'])-1}" value="0" step="1"></label>
<div class="replay-status" aria-live="polite"></div><div class="legend"><span class="push">● Push</span><span class="avoid">● Avoid</span><span>＋ 真实 θ*</span><span>线段：加权 PCA 第一主轴 ±1 SD</span></div>
<div class="scatter-grid"><div><h3>Standard · {standard['label']}</h3><svg class="standard-scatter" viewBox="0 0 610 360"></svg><div class="standard-metrics metrics"></div></div>
<div><h3 class="candidate-title">Candidate</h3><svg class="candidate-scatter" viewBox="0 0 610 360"></svg><div class="candidate-metrics metrics"></div></div></div>
<p class="note">全部 {len(data['theta'])} 个候选保持可见；点面积与透明度随相对权重变化，两图坐标范围及显示尺度完全相同。当前角度为物理坐标 PCA；其数值依赖单位。另存域归一化 PCA 供方向和宽度对照。</p>
<div class="legend">{legend}</div><div class="chart-grid">{plots}</div>
<details><summary>ESS / posterior 方向及宽度随时间变化</summary><div class="chart-grid">{geometry_plots}</div></details>
<div class="scroll"><table><thead><tr><th>Probe</th><th>Force limit</th><th>Max simulated time</th><th>t_clear / s</th><th>确认 / s</th><th>final p_D</th><th>final D</th><th>final judgment</th><th>final ESS</th><th>后续反复</th><th>数据</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>''')
    encoded = json.dumps(json_safe(data), ensure_ascii=False, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")
    script_path = output/"source_snapshot"/"navlab"/"multi_probe_replay.js"
    js = script_path.read_text(encoding="utf-8")
    baseline = read_json(output/"baseline_regression.json")
    validation = read_json(output/"validation.json")
    document = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Multi-Probe · Belief Evolution</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#edf2f7;color:#172b42;font:15px/1.6 system-ui,"Microsoft YaHei",sans-serif}}main{{max-width:1450px;margin:auto;padding:30px}}
h1{{font-size:32px;margin:0}}h2{{margin-top:0}}h3{{font-size:16px}}section,.intro{{background:white;border:1px solid #dbe4ef;border-radius:14px;padding:24px;margin:22px 0;box-shadow:0 3px 14px #20334b08}}
.eyebrow{{font-size:12px;letter-spacing:2px;color:#0f766e;font-weight:700}}.note{{color:#64748b;font-size:13px}}.controls,.legend{{display:flex;gap:18px;align-items:center;flex-wrap:wrap;margin:14px 0}}
button,select{{font:inherit;background:#f8fafc;border:1px solid #cbd5e1;border-radius:6px;padding:6px 12px;color:#172b42}}button{{cursor:pointer}}button.play{{background:#0f766e;color:white;border-color:#0f766e}}
.slider-label{{display:flex;align-items:center;gap:16px}}input[type=range]{{flex:1;accent-color:#0f766e}}.scatter-grid,.chart-grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}svg{{width:100%;display:block}}
.metrics{{font-variant-numeric:tabular-nums;padding:10px;background:#f1f5f9;border-radius:8px;min-height:105px}}.replay-status{{padding:10px 0;font-weight:600;min-height:42px}}.clear{{background:#dcfce7;color:#166534;border-radius:4px;padding:2px 6px}}
.push{{color:#059669}}.avoid{{color:#dc2626}}.tick{{font-size:11px;fill:#64748b}}.chart-title{{font-size:14px;fill:#172b42;font-weight:600}}.scroll{{overflow:auto;margin-top:20px}}table{{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}}th,td{{border-bottom:1px solid #e2e8f0;text-align:left;padding:9px;white-space:nowrap}}th{{background:#f1f5f9}}a{{color:#0369a1}}details{{margin:16px 0}}summary{{cursor:pointer}}li{{margin:12px 0}}@media(max-width:850px){{main{{padding:12px}}section{{padding:14px}}.scatter-grid,.chart-grid{{grid-template-columns:1fr}}}}
</style></head><body><main><div class="eyebrow">1024 SOBOL WORLDS · MUJOCO · INFORMATION-ONLY COMPARISON</div><h1>不同 Probe 怎样改变 Belief？</h1>
<div class="intro"><p>只改变水平指令力上限 F_max。固定初始状态、接近方式、接触位置、+Y 推动方向、控制律、1024 个 Sobol 候选与原 Push/Avoid 标签。四档 Probe 各模拟一次 1.20 s，所有短时结果来自同条 stream 前缀。</p>
<p>z(t) = [累计峰值接触力, 当前门角变化]。每帧调用原 consistency_update(prior, prediction(t), observation(t))，未逐帧累乘。尺度 {data['snapshot']['base_config']['force_scale']} N / {data['snapshot']['base_config']['angle_scale']} rad，α={data['snapshot']['base_config']['alpha']}。</p>
<p>p_D=Σwᵢyᵢ；D=4p_D(1−p_D)。统一 D_clear={settings['D_clear']}、连续 {settings['persistence_frames']} 帧、采样 {settings['sample_interval_s']:.2f} s；必须支持同一决策方向。t_clear 为稳定区间起点，确认时间为其最后一帧。t_clear 是离线回溯标记，在线停止最早应使用确认时间；末尾不足连续帧数不记达标。后续反复单独列出。Uncertain 显示原 D 门控未明确；原导航策略此时的保守动作仍是 Avoid。</p>
<p>本报告固定原路线标签，研究信息效果；没有改变或运行原任务的 Probe 后剩余路线重算和任务执行，也不自动选择最优 Probe。指令力上限不等于实际峰值接触力，可在 CSV 查看实际响应。</p>
<p>Standard 对旧库终点最大绝对差 {baseline['terminal_max_absolute_difference']:.3g}，旧前缀最大差 {baseline.get('prefix_max_absolute_difference',float('nan')):.3g}；验证通过 {validation['posterior_frames_checked']} 帧 × {validation['particles_per_frame']} 粒子。</p>
<a href="probe_summary.csv">Probe summary CSV</a> · <a href="probe_summary.json">JSON</a> · <a href="config_snapshot.json">完整配置</a> · <a href="validation.json">验证</a> · <a href="baseline_regression.json">Baseline 对照</a></div>
<section><h2>实验问题与实际结果</h2><ol>{''.join('<li>'+html.escape(a)+'</li>' for a in answers)}</ol><p class="note">“参数较宽”观察标准：域归一化 RMS 宽度至少为 prior 的 {settings['broad_posterior_rms_ratio']:.0%}，或任一参数边际标准差至少为 prior 的 {settings['broad_posterior_marginal_sd_ratio']:.0%}。后者保留带状 posterior 的单轴不确定性。这些指标不参与 belief 更新或 Probe 决策。</p></section>
{''.join(sections)}<p class="note">全部图表来自保存的 JSON/CSV。HTML 自包含，可直接离线打开。参数 posterior 不必唯一定位真值；D 低仅表示原库路线标签基本一致。</p>
<script type="application/json" id="comparison-data">{encoded}</script><script>{js}</script></main></body></html>'''
    (output/"report.html").write_text(document, encoding="utf-8")
    # The browser consumes the exact saved histories, not rounded chart/table numbers.
    embedded = json.loads(document.split('<script type="application/json" id="comparison-data">', 1)[1].split('</script>', 1)[0])
    assert embedded == json_safe(data)
    save_json(output/"html_data_validation.json", dict(passed=True, embedded_payload_equals_saved_json=True,
                                                       fields=["weights", "times", "p_D", "D", "t_clear", "t_confirmed", "PCA", "ESS"]))
    return output/"report.html"
