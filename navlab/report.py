"""Self-contained experiment report; stdlib SVG/HTML, no plotting dependency."""
import csv
import html
import json
from collections import Counter
from pathlib import Path
from . import runtime
import numpy as np
from .map import MapSpec
from .experiments import json_safe


def number(value,precision=4):
    return "—" if value is None or not np.isfinite(value) else f"{value:.{precision}f}"


def map_svg(spec,geo):
    width,height = 760,290
    xmin,xmax,ymin,ymax = spec.bounds
    scale = min((width-40)/(xmax-xmin),(height-40)/(ymax-ymin))
    def xy(point):
        return (20+(point[0]-xmin)*scale,height-20-(point[1]-ymin)*scale)
    elements = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="A star route geometry">']
    for _,center,half in spec.walls():
        a = xy(center+[-half[0],half[1]])
        elements.append(f'<rect x="{a[0]}" y="{a[1]}" width="{2*half[0]*scale}" height="{2*half[1]*scale}" fill="#64748b"/>')
    for field,color in (("door_path","#059669"),("bypass_path","#2563eb")):
        points = " ".join(f"{x},{y}" for x,y in map(xy,geo[field]))
        elements.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
    a,b = xy([-.61,1.0625]),xy([.5725,1.0625])
    elements.append(f'<line x1="{a[0]}" y1="{a[1]}" x2="{b[0]}" y2="{b[1]}" stroke="#b45309" stroke-width="5"/>')
    for name,point in (("S",spec.start),("M",spec.merge),("G",spec.goal)):
        x,y = xy(point)
        elements.append(f'<circle cx="{x}" cy="{y}" r="5" fill="#0f172a"/><text x="{x+7}" y="{y-7}">{name}</text>')
    elements.append('</svg>')
    return "".join(elements)


def scatter(theta,weights,labels,truth,cfg):
    width,height = 560,300
    x = 48+(theta[:,0]-cfg.inertia_min)/(cfg.inertia_max-cfg.inertia_min)*470
    y = 258-(theta[:,1]-cfg.friction_min)/(cfg.friction_max-cfg.friction_min)*220
    wmax = max(float(weights.max()),1e-12)
    out = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Physical hypothesis weights">',
           '<path d="M48 32 V258 H530" fill="none" stroke="#94a3b8"/>',
           '<text x="185" y="291">I about hinge (kg m²)</text>',
           '<text x="12" y="22">τ_f (N m)</text>']
    # Less-supported points render first; high weights remain visible.
    for i in np.argsort(weights):
        ratio = float(weights[i])/wmax
        radius = 1.4+4.*np.sqrt(ratio)
        color = "#059669" if labels[i] else "#dc2626"
        opacity = .08+.9*ratio
        tip = html.escape(f"i={i}, I={theta[i,0]:.5f}, tau_f={theta[i,1]:.5f}, w={weights[i]:.8f}, y={labels[i]}")
        out.append(f'<circle cx="{x[i]:.2f}" cy="{y[i]:.2f}" r="{radius:.2f}" fill="{color}" opacity="{opacity:.3f}"><title>{tip}</title></circle>')
    tx = 48+(truth[0]-cfg.inertia_min)/(cfg.inertia_max-cfg.inertia_min)*470
    ty = 258-(truth[1]-cfg.friction_min)/(cfg.friction_max-cfg.friction_min)*220
    out.append(f'<path d="M{tx-7} {ty} H{tx+7} M{tx} {ty-7} V{ty+7}" stroke="#0f172a" stroke-width="2"><title>off-grid truth (evaluation only)</title></path>')
    out.append('</svg>')
    return "".join(out)


def response_svg(predictions,observation=None):
    low,high = predictions.min(axis=0),predictions.max(axis=0)
    span = np.maximum(high-low,1e-6)
    points = (predictions-low)/span
    out = ['<svg viewBox="0 0 560 300" role="img" aria-label="Probe forward response library">',
           '<path d="M48 32 V258 H530" fill="none" stroke="#94a3b8"/>',
           '<text x="190" y="291">F_peak (N)</text><text x="12" y="22">Δθ (rad)</text>']
    for i,(u,v) in enumerate(points):
        out.append(f'<circle cx="{48+470*u:.2f}" cy="{258-220*v:.2f}" r="2" fill="#0284c7" opacity=".4"><title>i={i}; F={predictions[i,0]:.5f} N; delta={predictions[i,1]:.5f} rad</title></circle>')
    if observation is not None:
        u,v = (np.asarray(observation)-low)/span
        x,y = 48+470*u,258-220*v
        out.append(f'<path d="M{x-8} {y} H{x+8} M{x} {y-8} V{y+8}" stroke="#dc2626" stroke-width="3"><title>observed z*</title></path>')
    out.append(f'<text x="45" y="275">{low[0]:.2f}</text><text x="495" y="275">{high[0]:.2f}</text><text x="12" y="260">{low[1]:.3f}</text><text x="12" y="45">{high[1]:.3f}</text></svg>')
    return "".join(out)


def history_view(case,folder,theta,truth,cfg):
    path = folder/"belief_history.npz"
    if not path.exists():
        return None
    with np.load(path,allow_pickle=False) as data:
        payload = {name:data[name].tolist() for name in data.files}
    payload.update(theta=theta.tolist(),truth=list(truth),
                   limits=[cfg.inertia_min,cfg.inertia_max,cfg.friction_min,cfg.friction_max])
    encoded = json.dumps(payload,ensure_ascii=False,separators=(",",":"))
    encoded = encoded.replace("<","\\u003c")
    return f'''<div class="belief-replay" data-case="{case}">
<svg viewBox="0 0 560 300" class="history-svg" role="img" aria-label="记录的候选权重变化"></svg>
<div class="replay-controls"><button data-command="first">初始等权</button><button data-command="play">播放权重变化</button><button data-command="probe">Probe 结束</button><button data-command="last">最终决策</button></div>
<input class="history-slider" type="range" min="0" max="{len(payload['time'])-1}" value="{len(payload['time'])-1}" step="1" style="width:100%">
<div class="history-status"></div><p class="history-note">运行时在实时面板观看；这里按实际记录回放。每帧点大小按该帧最大权重归一化。正式执行期间保留 Probe 后权重。</p>
<script type="application/json" class="history-data">{encoded}</script></div>'''


REPLAY_JS = r'''
document.querySelectorAll('.belief-replay').forEach(panel=>{
 const data=JSON.parse(panel.querySelector('.history-data').textContent), svg=panel.querySelector('svg');
 const slider=panel.querySelector('input'), status=panel.querySelector('.history-status');
 const NS='http://www.w3.org/2000/svg', [imin,imax,fmin,fmax]=data.limits;
 const node=(tag,attrs={},text='')=>{const el=document.createElementNS(NS,tag);Object.entries(attrs).forEach(([k,v])=>el.setAttribute(k,v));if(text)el.textContent=text;return el;};
 const xy=t=>[48+(t[0]-imin)/(imax-imin)*470,258-(t[1]-fmin)/(fmax-fmin)*220];
 svg.append(node('path',{d:'M48 32 V258 H530',fill:'none',stroke:'#94a3b8'}));
 for(let j=0;j<=4;j++){let f=j/4,x=48+470*f,y=258-220*f;svg.append(node('text',{x,y:277,'text-anchor':'middle'},(imin+f*(imax-imin)).toFixed(2)));svg.append(node('text',{x:40,y:y+4,'text-anchor':'end'},(fmin+f*(fmax-fmin)).toFixed(2)));}
 svg.append(node('text',{x:180,y:297},'I about hinge (kg m²)'));svg.append(node('text',{x:12,y:20},'τ_f (N m)'));
 const group=node('g');svg.append(group);
 const dots=data.theta.map((t,i)=>{const [x,y]=xy(t),el=node('circle',{cx:x,cy:y});el.append(node('title'));group.append(el);return el;});
 const [tx,ty]=xy(data.truth);svg.append(node('path',{d:`M${tx-7} ${ty} H${tx+7} M${tx} ${ty-7} V${ty+7}`,stroke:'#0f172a','stroke-width':2}));
 let index=data.time.length-1,timer=null;
 const phases={INITIAL:'初始等权',PROBE:'Probe 中：累计观测实时更新',PROBE_COMPLETE:'Probe 结束',STATE_CONDITIONED:'同一 Probe 后状态：重算剩余路线标签'};
 function draw(i){index=i;slider.value=i;const weights=data.weights[i],labels=data.labels[i],maximum=Math.max(...weights);
  const order=weights.map((w,k)=>k).sort((a,b)=>weights[a]-weights[b]);
  order.forEach(k=>{const el=dots[k],ratio=weights[k]/Math.max(maximum,1e-15);el.setAttribute('r',1.4+4*Math.sqrt(ratio));el.setAttribute('fill',labels[k]?'#059669':'#dc2626');el.setAttribute('opacity',.08+.9*ratio);el.firstChild.textContent=`i=${k}; I=${data.theta[k][0].toFixed(5)}; τ_f=${data.theta[k][1].toFixed(5)}; w=${weights[k].toPrecision(7)}; y=${labels[k]}`;group.append(el);});
  status.textContent=`${i+1}/${data.time.length} 帧｜${phases[data.phase[i]]||data.phase[i]}｜t=${data.time[i].toFixed(2)} s，Probe=${data.probe_elapsed[i].toFixed(2)} s｜p_D=${data.p_D[i].toFixed(6)}，D=${data.D[i].toFixed(6)}｜最大权重=${maximum.toFixed(6)}`;
 }
 function stop(){if(timer)clearInterval(timer);timer=null;panel.querySelector('[data-command="play"]').textContent='播放权重变化';}
 slider.addEventListener('input',()=>{stop();draw(Number(slider.value));});
 panel.querySelectorAll('button').forEach(button=>button.addEventListener('click',()=>{const command=button.dataset.command;
  if(command==='play'){if(timer){stop();return;}if(index===data.time.length-1)draw(0);button.textContent='暂停回放';timer=setInterval(()=>{if(index<data.time.length-1)draw(index+1);else stop();},150);return;}
  stop();if(command==='first')draw(0);if(command==='last')draw(data.time.length-1);if(command==='probe'){const i=data.phase.lastIndexOf('PROBE_COMPLETE');draw(i<0?0:i);}
 }));
 if(data.time.length===1){panel.querySelector('[data-command="play"]').disabled=true;panel.querySelector('[data-command="probe"]').disabled=true;panel.querySelector('.history-note').textContent='本场景没有 Probe，1024 个候选保持初始等权；执行画面与接触力仍在实时面板显示。';}
 draw(index);
});
'''


def write_report(output,results,library,cfg):
    n = len(library["theta"])
    cards,rows = [],[]
    distribution = html.escape(json.dumps(dict(Counter(library["failure_category"][~library["success"]])),ensure_ascii=False))
    audit_html = ""
    audit_path = output/"recovery_audit.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        comparison = ''.join(f'<tr><td>{html.escape(k)}</td><td>{v["before"]}</td><td>{v["after"]}</td></tr>'
                             for k,v in audit["distributions"].items())
        audit_html = f'<section><h2>恢复控制与保留基线对照</h2><table><tr><th>1024 世界结果</th><th>修改前</th><th>本版本</th></tr>{comparison}</table><p>支持 Door：{audit["before_door_support"]} → {audit["after_door_support"]}。标准 Probe 全库最大绝对差：{audit["probe_max_absolute_difference"]}。原成功世界变为失败：{len(audit["regressed_ids"])} 个。<a href="recovery_audit.md">详细对照</a> · <a href="recovery_audit.json">机器可读审计</a></p></section>'
    for r in results:
        folder = output/("case_"+r["case"])
        with (folder/"weights.csv").open(encoding="utf-8-sig") as stream:
            records = list(csv.DictReader(stream))
        weights = np.array([float(v["w_after"]) for v in records])
        labels = np.array([int(v["y_after"]) for v in records])
        post = r.get("posterior",{})
        pd,d = post.get("p_D",r["prior"]["p_D"]),post.get("D",r["prior"]["D"])
        rows.append(f'<tr><td>{r["case"]}</td><td>{number(r["geometry"]["G"],3)}</td><td>{number(r["prior"]["p_D"])}</td><td>{number(r["prior"]["D"])}</td><td>{r["prior"]["action"]}</td><td>{number(pd)}</td><td>{number(d)}</td><td>{r["final_action"]}</td><td>{r["mission_success"]}</td><td>{number(r["total_time_to_M_s"],3)}</td><td>{number(r["regret_vs_no_probe_oracle_s"],3)}</td></tr>')
        metrics = ""
        if post:
            metrics = f'<p>只更新权重：p_D={number(post["p_D_weights_only"])}, D={number(post["D_weights_only"])}。随后从统一 Probe 后状态重算标签。</p><p>归一化参数距离：{number(post["normalized_parameter_distance_prior"])} → {number(post["normalized_parameter_distance_posterior"])}；半径 0.10 内权重：{number(post["mass_radius_0_10_prior"])} → {number(post["mass_radius_0_10_posterior"])}。D 是否下降：{post["D_decreased"]}；是否明确：{post["resolved"]}。</p>'
            with np.load(Path(r["continuation_cache"])/"continuation_costs.npz",allow_pickle=False) as remaining:
                counts = {route:dict(Counter(remaining[route+"_category"][~remaining[route+"_success"]]))
                          for route in ("door","bypass")}
            metrics += f'<p>1024 个共同观测状态的剩余路线失败类别：{html.escape(json.dumps(counts,ensure_ascii=False))}。两路均失败 {post["both_routes_failed_count"]} 个，后验权重 {number(post["both_routes_failed_weight"],6)}。</p>'
            metrics += f'<details><summary>Probe 前向响应覆盖与真实观测</summary>{response_svg(library["z_hat"],r["observation"])}<p>蓝点为 1024 个预测，红十字为 z*。不同物理状态可能具有相近响应，不要求唯一参数辨识。</p></details>'
        recovery = r["mission"].get("recovery",{})
        if recovery:
            c = recovery.get("checks",{})
            metrics += f'<p>实际恢复 {number(recovery.get("duration"),3)} s；完成={recovery.get("success")}；出口门接触={number(c.get("door_force_N"),3)} N，倾角={number(c.get("tilt_deg"),2)}°，速率={number(c.get("speed_m_s"),3)} m/s，净空达标={c.get("clearance")}。阶段时间：{html.escape(str(r["mission"].get("phase_times",{})))}。</p>'
            metrics += f'<p>完整执行的门接触峰值 {number(r["mission"]["contact_peak"],3)} N，墙接触峰值 {number(r["mission"]["wall_peak"],3)} N；接触峰值是物理碰撞响应，区别于 2.2 N 推进指令上限。</p>'
        chart = history_view(r["case"],folder,library["theta"],r["theta_truth_evaluation_only"],cfg)
        if chart is None:
            chart = scatter(library["theta"],weights,labels,r["theta_truth_evaluation_only"],cfg)
        snapshot = ''
        if (folder/"live_views"/"final_dashboard.png").exists():
            snapshot = f'<details><summary>实时面板保存画面</summary><img style="width:100%" src="case_{r["case"]}/live_views/final_dashboard.png" alt="实际 MuJoCo 实时面板保存画面"></details>'
        cards.append(f'<section><h2>Case {r["case"]}</h2><p>{html.escape(r["description"])}；θ*={r["theta_truth_evaluation_only"]}，不在候选集中。</p><div class="grid"><div>{map_svg(MapSpec(**r["map"]),r["geometry"])}</div><div>{chart}</div></div>{metrics}{snapshot}<p>最终 {r["final_action"]}；完整任务到 M：{number(r["total_time_to_M_s"],3)} s；真实环境直接路线的最优选择：{r["oracle_action"]}；Probe 后剩余任务最优选择：{r.get("oracle_after_probe","—")}。</p><a href="case_{r["case"]}/summary.json">详细 JSON</a> · <a href="case_{r["case"]}/weights.csv">候选权重 CSV</a> · <a href="case_{r["case"]}/live_timeline.csv">实时记录</a> · <a href="case_{r["case"]}/mission_trace.csv">真实执行轨迹</a></section>')
    doc = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>1024 候选世界主动交互导航</title>
<style>body{{max-width:1280px;margin:32px auto;padding:0 24px;font:16px/1.65 "Microsoft YaHei",system-ui;color:#0f172a;background:#f8fafc}}h1{{font-size:28px}}h2{{font-size:20px}}section{{background:white;padding:22px;margin:24px 0;border:1px solid #e2e8f0;border-radius:10px}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}svg{{width:100%;background:#fff}}text{{font:12px system-ui;fill:#334155}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{border:1px solid #cbd5e1;padding:8px;text-align:center}}th{{background:#e2e8f0}}a{{color:#0369a1}}.scroll{{overflow:auto}}.replay-controls{{display:flex;gap:6px;flex-wrap:wrap}}button{{padding:6px 10px;border:1px solid #cbd5e1;border-radius:5px;background:#f1f5f9;cursor:pointer}}.history-status,.history-note{{font-size:12px;color:#475569}}@media(max-width:850px){{.grid{{grid-template-columns:1fr}}}}</style>
<h1>物理候选世界 + 几何价值 G + 动作决策分歧 D</h1>
<p>N={n}，二维 Sobol 覆盖 Θ；初始及视觉权重均为 1/N。Θ 是允许的物理参数域，不是真实门的概率分布。绿色点支持 Door，红色点支持 Bypass；圆点大小和透明度表示当前权重，十字表示仅用于评估的 θ*。悬停可读候选参数与权重。</p>
<p>G = J_bypass^geo − J_door^geo（m）；p_D = Σ w_i y_i；D = 4p_D(1−p_D)。阈值 G≥{cfg.geometry_threshold} m 且 D≥{cfg.disagreement_threshold} 时执行一次标准 Probe。观测为峰值总法向接触力及门角变化；一致性 soft weighting 使用数值稳定实现。</p>
<div class="scroll"><table><tr><th>Case</th><th>G / m</th><th>初始 p_D</th><th>初始 D</th><th>初始动作</th><th>最终 p_D</th><th>最终 D</th><th>最终动作</th><th>到达 M</th><th>含 Probe 时间 / s</th><th>对无 Probe oracle 的后悔 / s</th></tr>{''.join(rows)}</table></div>
<p>时间来自原张拉结构无人机的四旋翼 MuJoCo rollout。推门、制动恢复、退离、门后安全航点和到 M 的全部执行时间计入路线。失败记为 +∞；CSV 保留 inf，JSON 用 null 与 success=false 表示，并单独保存实际 elapsed_time。M→G 公共段单独计时。</p>
<p>本版本候选 Door 完成 {int(library['success'].sum())}/{n}；支持 Door {int(library['labels'].sum())}/{n}。失败类别：{distribution}。数值异常会阻止库用于决策。达到 {cfg.open_angle_deg}° 即转 BACKOFF，接触、姿态、速度、高度与原 {cfg.clearance} m 规划净空连续达标 {cfg.recovery_dwell} s 后才进入 FLIGHT；恢复超时 {cfg.recovery_timeout} s。</p>
<p>首阶段假设已知建筑地图、固定高度、理想自身状态及理想接触/门角测量；未实现视觉检测网络。决策器不会读取真实 I、τ_f。两观测不保证唯一辨识参数，也不保证一次 Probe 必然降低 D。若 Probe 后仍有较大分歧，明确记录并按单次 Probe 保守绕行。</p>
{audit_html}{''.join(cards)}<p>库签名：{library['signature']}。CSV 和 JSON 是主要研究数据；本报告可直接离线打开。</p><script>{REPLAY_JS}</script></html>'''
    (output/"report.html").write_text(doc,encoding="utf-8")
