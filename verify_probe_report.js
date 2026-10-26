// Execute the report's own replay JS against a small DOM, as verify_report.js does.
// Test all saved Case x Probe x time frames; no browser/network packages needed.
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
const output=path.resolve(process.argv[2]||'');
const html=fs.readFileSync(path.join(output,'report.html'),'utf8');
const payload=JSON.parse(html.match(/<script type="application\/json" id="comparison-data">([\s\S]*?)<\/script>/)[1]);
const js=[...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].at(-1)[1];
class Element {
  constructor(){this.attrs={};this.children=[];this.handlers={};this.textContent='';this.innerHTML='';this.value='';this.dataset={};}
  setAttribute(k,v){this.attrs[k]=String(v);}
  append(el){if(el.parent)el.parent.children=el.parent.children.filter(x=>x!==el);el.parent=this;this.children.push(el);}
  addEventListener(k,f){this.handlers[k]=f;}
  get firstChild(){return this.children[0];}
  click(){this.handlers.click?.();}
}
class Panel extends Element {
  constructor(name){
    super();this.dataset.case=name;
    const classes=['probe-selector','time-slider','play','speed','replay-status','standard-scatter','candidate-scatter','candidate-title','standard-metrics','candidate-metrics','first','last','showcase'];
    this.elements=Object.fromEntries(classes.map(c=>['.'+c,new Element()]));
    this.elements['.probe-selector'].value=payload.snapshot.settings.probes[0].id;
    this.elements['.speed'].value='200';
    this.cursors=Array.from({length:6},()=>{const e=new Element();e.dataset={left:'54',right:'630'};return e;});
    const diffuse=Object.values(payload.cases[name]).flatMap(h=>h.metrics.map((m,i)=>({m,h,i}))).filter(r=>r.m.D<payload.snapshot.settings.D_clear).sort((a,b)=>b.m.normalized_rms_ratio-a.m.normalized_rms_ratio)[0];
    if(diffuse)this.elements['.showcase'].dataset={probe:diffuse.h.probe_id,index:String(diffuse.i)};
    else delete this.elements['.showcase'];
  }
  querySelector(selector){return this.elements[selector]||null;}
  querySelectorAll(selector){assert.equal(selector,'.time-cursor');return this.cursors;}
}
const panels=Object.keys(payload.cases).map(c=>new Panel(c)),intervals=new Map();let timerId=0;
vm.runInNewContext(js,{
  document:{querySelector:s=>{assert.equal(s,'#comparison-data');return {textContent:JSON.stringify(payload)};},
            querySelectorAll:s=>{assert.equal(s,'.case-panel');return panels;},createElementNS:()=>new Element()},
  setInterval:f=>{intervals.set(++timerId,f);return timerId;},clearInterval:id=>intervals.delete(id)
},{timeout:30000});
const standard=payload.snapshot.settings.probes.find(p=>p.standard);
let checked=0;
function check(panel,index){
  const histories=payload.cases[panel.dataset.case],candidate=histories[panel.querySelector('.probe-selector').value],baseline=histories[standard.id];
  assert.equal(Number(panel.querySelector('.time-slider').value),index);
  for(const [cls,h,metrics] of [['.standard-scatter',baseline,'.standard-metrics'],['.candidate-scatter',candidate,'.candidate-metrics']]){
    const svg=panel.querySelector(cls),group=svg.children.find(e=>e.attrs.class==='particles');
    assert.equal(group.children.length,payload.theta.length);
    let mass=0,p=0;
    for(const dot of group.children){
      const id=Number(dot.attrs['data-world-id']),w=h.weights[index][id];
      assert.ok(dot.firstChild.textContent.includes(`w=${w};`));
      assert.equal(dot.attrs.fill,payload.labels[id]?'#059669':'#dc2626');
      assert.ok(Number(dot.attrs.opacity)>=.07);
      mass+=w;p+=w*payload.labels[id];
    }
    assert.ok(Math.abs(mass-1)<1e-12);assert.ok(Math.abs(p-h.metrics[index].p_D)<1e-12);
    const status=panel.querySelector(metrics);
    assert.equal(Number(status.dataset.p),h.metrics[index].p_D);
    assert.equal(Number(status.dataset.d),h.metrics[index].D);
    assert.equal(Number(status.dataset.ess),h.metrics[index].ESS);
    assert.equal(Number(status.dataset.time),h.times[index]);
    assert.equal(status.dataset.tClear,String(h.t_clear));
    assert.ok(status.innerHTML.includes(h.metrics[index].judgment));
    if(h.clear_index!==null && index>=h.clear_index)assert.ok(status.innerHTML.includes('达到 t_clear='));
  }
  const status=panel.querySelector('.replay-status');
  assert.equal(Number(status.dataset.time),candidate.times[index]);assert.equal(status.dataset.probeId,candidate.probe_id);
  for(const cursor of panel.cursors)assert.equal(Number(cursor.attrs.x1),54+candidate.times[index]/payload.snapshot.settings.maximum_probe_time_s*576);
}
for(const panel of panels){
  const histories=payload.cases[panel.dataset.case];
  for(const probe of payload.snapshot.settings.probes){
    const select=panel.querySelector('.probe-selector');select.value=probe.id;select.handlers.change();
    for(let i=0;i<histories[probe.id].times.length;i++){
      const slider=panel.querySelector('.time-slider');slider.value=i;slider.handlers.input();check(panel,i);checked++;
    }
  }
  panel.querySelector('.first').click();check(panel,0);
  const last=histories[standard.id].times.length-1;
  panel.querySelector('.last').click();check(panel,last);
  const showcase=panel.querySelector('.showcase');
  if(showcase){showcase.click();check(panel,Number(showcase.dataset.index));assert.equal(panel.querySelector('.probe-selector').value,showcase.dataset.probe);}
  panel.querySelector('.last').click();panel.querySelector('.play').click();check(panel,0);
  for(let i=1;i<=last;i++){for(const f of [...intervals.values()])f();check(panel,i);}
  assert.equal(intervals.size,0);
  panel.querySelector('.play').click();panel.querySelector('.play').click();assert.equal(intervals.size,0);
  console.log(`Case ${panel.dataset.case}: all Probe selectors, synchronized frames, particles, metrics, clear markers, showcase and playback passed`);
}
const audit={passed:true,cases:panels.length,probes:payload.snapshot.settings.probes.length,sliderFramesChecked:checked,
             playbackPassed:true,allParticlesChecked:true,method:'report replay JS executed in DOM substitute; real browser is optional'};
fs.writeFileSync(path.join(output,'replay_validation.json'),JSON.stringify(audit,null,2));
console.log(JSON.stringify(audit,null,2));
