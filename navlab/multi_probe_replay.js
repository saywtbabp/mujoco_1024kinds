/* Offline replay of the exact saved experiment payload. No posterior arithmetic here. */
(() => {
  'use strict';
  const data = JSON.parse(document.querySelector('#comparison-data').textContent);
  const cfg = data.snapshot.base_config, settings = data.snapshot.settings;
  const standard = settings.probes.find(p => p.standard);
  const NS = 'http://www.w3.org/2000/svg';
  const node = (name, attrs = {}, text = '') => {
    const el = document.createElementNS(NS, name);
    for (const [key, value] of Object.entries(attrs)) el.setAttribute(key, value);
    el.textContent = text;
    return el;
  };
  const fmt = (v, n = 3) => v === null ? 'Not reached' : Number(v).toFixed(n);
  function scatter(svg, truth) {
    const x = v => 58 + (v-cfg.inertia_min)/(cfg.inertia_max-cfg.inertia_min)*524;
    const y = v => 312 - (v-cfg.friction_min)/(cfg.friction_max-cfg.friction_min)*284;
    svg.append(node('rect', {x:58, y:28, width:524, height:284, fill:'#f8fafc', stroke:'#cbd5e1'}));
    for(let j=0; j<5; j++) {
      const a = cfg.inertia_min + j/4*(cfg.inertia_max-cfg.inertia_min);
      const b = cfg.friction_min + j/4*(cfg.friction_max-cfg.friction_min);
      svg.append(node('text', {x:x(a), y:333, 'text-anchor':'middle', class:'tick'}, a.toFixed(2)));
      svg.append(node('text', {x:50, y:y(b)+4, 'text-anchor':'end', class:'tick'}, b.toFixed(2)));
    }
    svg.append(node('text', {x:322, y:354, 'text-anchor':'middle', class:'tick'}, 'I / kg m²'));
    svg.append(node('text', {x:58, y:16, class:'tick'}, 'τ_f / N m'));
    const group = node('g', {class:'particles'});
    const dots = data.theta.map((theta, i) => {
      const el = node('circle', {cx:x(theta[0]), cy:y(theta[1]), 'data-world-id':i});
      el.append(node('title'));
      group.append(el);
      return el;
    });
    svg.append(group);
    const axis = node('line', {stroke:'#334155', 'stroke-width':2, 'pointer-events':'none'});
    svg.append(axis);
    svg.append(node('path', {d:`M${x(truth[0])-8},${y(truth[1])}h16 M${x(truth[0])},${y(truth[1])-8}v16`, stroke:'#020617', 'stroke-width':2.5, fill:'none'}));
    return (history, index, commonMaximum) => {
      const weights = history.weights[index], metric = history.metrics[index];
      const order = weights.map((_, i) => i).sort((a,b) => weights[a]-weights[b]);
      for(const i of order) {
        const ratio = weights[i]/Math.max(commonMaximum, 1e-30), el = dots[i];
        el.setAttribute('r', 1.1 + 4.1*Math.sqrt(ratio));
        el.setAttribute('opacity', 0.07 + 0.91*ratio);
        el.setAttribute('fill', data.labels[i] ? '#059669' : '#dc2626');
        el.firstChild.textContent = `i=${i}; I=${data.theta[i][0]}; τ_f=${data.theta[i][1]}; w=${weights[i]}; y=${data.labels[i]}`;
        group.append(el);
      }
      const geom = metric.geometry, mu = geom.mean, a = geom.principal_axis, sd = geom.major_sd;
      const endpoint = sign => [Math.max(cfg.inertia_min,Math.min(cfg.inertia_max,mu[0]+sign*a[0]*sd)),
                                Math.max(cfg.friction_min,Math.min(cfg.friction_max,mu[1]+sign*a[1]*sd))];
      const lo = endpoint(-1), hi = endpoint(1);
      axis.setAttribute('x1', x(lo[0])); axis.setAttribute('y1', y(lo[1]));
      axis.setAttribute('x2', x(hi[0])); axis.setAttribute('y2', y(hi[1]));
    };
  }
  for(const panel of document.querySelectorAll('.case-panel')) {
    const name = panel.dataset.case, histories = data.cases[name];
    const baseline = histories[standard.id], truth = data.snapshot.cases[name].theta;
    const selector = panel.querySelector('.probe-selector'), slider = panel.querySelector('.time-slider');
    const referenceSelector = panel.querySelector('.reference-selector');
    const play = panel.querySelector('.play'), speed = panel.querySelector('.speed');
    const status = panel.querySelector('.replay-status');
    const left = scatter(panel.querySelector('.standard-scatter'), truth);
    const right = scatter(panel.querySelector('.candidate-scatter'), truth);
    let index = 0, timer = null;
    const stop = () => { if(timer !== null) clearInterval(timer); timer = null; play.textContent = '播放权重变化'; };
    function metrics(target, history, i) {
      const m = history.metrics[i];
      let clear = '';
      if(history.clear_index !== null && i >= history.clear_index) {
        clear = `<span class="clear">达到 t_clear=${fmt(history.t_clear,2)} s${i >= history.confirmed_index ? ' · 已连续确认' : ' · 离线起点，尚待连续确认'}</span>`;
      }
      if(history.later_relapse) clear += ' · 后续存在反复';
      target.innerHTML = `p_D=<b>${fmt(m.p_D,6)}</b> · D=<b>${fmt(m.D,6)}</b> · <b>${m.judgment}</b><br>` +
        `ESS=${fmt(m.ESS,1)} · normalized PCA=${fmt(m.normalized_geometry.angle_deg,2)}° · physical PCA=${fmt(m.geometry.angle_deg,2)}° · width/prior=${fmt(m.normalized_rms_ratio,3)}<br>${clear}`;
      if(history.command_force_limits) target.innerHTML += `<br>当前指令上限=${fmt(history.command_force_limits[i],2)} N · 已用 command-force budget=${fmt(history.command_budgets_sofar[i],3)} N·s`;
      target.dataset.p = String(m.p_D); target.dataset.d = String(m.D); target.dataset.time = String(history.times[i]);
      target.dataset.tClear = String(history.t_clear); target.dataset.ess = String(m.ESS);
      target.dataset.angle = String(m.normalized_geometry.angle_deg);
    }
    function draw(i) {
      index = Math.max(0,Math.min(i,baseline.times.length-1)); slider.value = index;
      const chosen = histories[selector.value];
      const reference = referenceSelector ? histories[referenceSelector.value] : baseline;
      const maximum = Math.max(...reference.weights[index], ...chosen.weights[index]);
      left(reference,index,maximum); right(chosen,index,maximum);
      const referenceTitle = panel.querySelector('.reference-title');
      if(referenceTitle) referenceTitle.textContent = 'Reference · '+reference.probe.label;
      panel.querySelector('.candidate-title').textContent = 'Candidate · '+chosen.probe.label;
      metrics(panel.querySelector('.standard-metrics'),reference,index);
      metrics(panel.querySelector('.candidate-metrics'),chosen,index);
      status.textContent = `Case ${name} · ${chosen.probe.label} · t=${fmt(chosen.times[index],2)} s · ${index+1}/${chosen.times.length} 帧`;
      status.dataset.time = String(chosen.times[index]); status.dataset.probeId = chosen.probe_id;
      status.dataset.referenceId = reference.probe_id;
      const pair = panel.querySelector('.pair-metrics');
      if(pair) {
        const l1 = chosen.weights[index].reduce((sum,w,k) => sum+Math.abs(w-reference.weights[index][k]),0);
        const delta = Math.abs(((chosen.metrics[index].normalized_geometry.angle_deg-reference.metrics[index].normalized_geometry.angle_deg+270)%180)-90);
        pair.textContent = `当前两图权重 L1=${fmt(l1,6)} · D 差=${fmt(Math.abs(chosen.metrics[index].D-reference.metrics[index].D),6)} · 归一化主轴夹角=${fmt(delta,2)}°`;
        pair.dataset.l1 = String(l1);
      }
      for(const cursor of panel.querySelectorAll('.time-cursor')) {
        const x = Number(cursor.dataset.left) + chosen.times[index]/settings.maximum_probe_time_s*(Number(cursor.dataset.right)-Number(cursor.dataset.left));
        cursor.setAttribute('x1',x); cursor.setAttribute('x2',x);
      }
    }
    selector.addEventListener('change', () => { stop(); draw(index); });
    if(referenceSelector) referenceSelector.addEventListener('change', () => { stop(); draw(index); });
    slider.addEventListener('input', () => { stop(); draw(Number(slider.value)); });
    panel.querySelector('.first').addEventListener('click', () => { stop(); draw(0); });
    panel.querySelector('.last').addEventListener('click', () => { stop(); draw(baseline.times.length-1); });
    const showcase = panel.querySelector('.showcase');
    if(showcase) showcase.addEventListener('click', () => {
      stop(); selector.value = showcase.dataset.probe; draw(Number(showcase.dataset.index));
    });
    play.addEventListener('click', () => {
      if(timer !== null) { stop(); return; }
      if(index === baseline.times.length-1) draw(0);
      play.textContent = '暂停';
      timer = setInterval(() => { draw(index+1); if(index === baseline.times.length-1) stop(); }, Number(speed.value));
    });
    speed.addEventListener('change', () => { stop(); });
    draw(0);
  }
})();
