// Execute the actual report replay script with all particles and both reference choices.
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
const output=path.resolve(process.argv[2]||'');
const html=fs.readFileSync(path.join(output,'report.html'),'utf8');
const data=JSON.parse(html.match(/<script type="application\/json" id="comparison-data">([\s\S]*?)<\/script>/)[1]);
const js=[...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].at(-1)[1];
for(const text of ['Matched command-force budget comparison','F_cmd,max(t)','Force transition','Door angular velocity','Actual contact force'])assert.ok(html.includes(text),text);
class Element{
 constructor(){this.attrs={};this.children=[];this.handlers={};this.dataset={};this.textContent='';this.innerHTML='';this.value='';}
 setAttribute(k,v){this.attrs[k]=String(v);}
 append(e){if(e.parent)e.parent.children=e.parent.children.filter(x=>x!==e);e.parent=this;this.children.push(e);}
 addEventListener(k,f){this.handlers[k]=f;}
 get firstChild(){return this.children[0];}
 click(){this.handlers.click?.();}
}
class Panel extends Element{
 constructor(c){
  super();this.dataset.case=c;
  this.elements=Object.fromEntries(['probe-selector','reference-selector','time-slider','play','speed','replay-status','standard-scatter','candidate-scatter','candidate-title','reference-title','standard-metrics','candidate-metrics','first','last','pair-metrics'].map(k=>['.'+k,new Element()]));
  this.elements['.probe-selector'].value='constant_1p6N';this.elements['.reference-selector'].value='standard_2p2N';this.elements['.speed'].value='200';
  this.cursors=Array.from({length:12},()=>{const e=new Element();e.dataset={left:'60',right:'632'};return e;});
 }
 querySelector(s){return this.elements[s]||null;}
 querySelectorAll(s){assert.equal(s,'.time-cursor');return this.cursors;}
}
const panels=Object.keys(data.cases).map(c=>new Panel(c)),timers=new Map();let timerId=0;
vm.runInNewContext(js,{document:{querySelector:s=>{assert.equal(s,'#comparison-data');return{textContent:JSON.stringify(data)};},querySelectorAll:s=>{assert.equal(s,'.case-panel');return panels;},createElementNS:()=>new Element()},setInterval:f=>{timers.set(++timerId,f);return timerId;},clearInterval:id=>timers.delete(id)},{timeout:30000});
let checked=0;
function check(panel,j){
 const histories=data.cases[panel.dataset.case],candidate=histories[panel.querySelector('.probe-selector').value],reference=histories[panel.querySelector('.reference-selector').value];
 assert.equal(Number(panel.querySelector('.time-slider').value),j);
 const commonMaximum=Math.max(...reference.weights[j],...candidate.weights[j]);
 for(const [svgClass,metricClass,h] of [['.standard-scatter','.standard-metrics',reference],['.candidate-scatter','.candidate-metrics',candidate]]){
  const group=panel.querySelector(svgClass).children.find(e=>e.attrs.class==='particles');assert.equal(group.children.length,1024);
  let p=0,mass=0;
  for(const dot of group.children){
   const id=Number(dot.attrs['data-world-id']),w=h.weights[j][id];
   assert.ok(dot.firstChild.textContent.includes(`w=${w};`));assert.equal(dot.attrs.fill,data.labels[id]?'#059669':'#dc2626');
   assert.equal(Number(dot.attrs.r),1.1+4.1*Math.sqrt(w/commonMaximum));
   assert.ok(Number(dot.attrs.opacity)>=.07);p+=w*data.labels[id];mass+=w;
  }
  assert.ok(Math.abs(mass-1)<1e-12);assert.ok(Math.abs(p-h.metrics[j].p_D)<1e-12);
  const status=panel.querySelector(metricClass),m=h.metrics[j];
  assert.equal(Number(status.dataset.p),m.p_D);assert.equal(Number(status.dataset.d),m.D);assert.equal(Number(status.dataset.ess),m.ESS);
  assert.equal(Number(status.dataset.angle),m.normalized_geometry.angle_deg);assert.equal(Number(status.dataset.time),h.times[j]);
  assert.equal(status.dataset.tClear,String(h.t_clear));assert.ok(status.innerHTML.includes(m.judgment));
  assert.ok(status.innerHTML.includes(h.command_force_limits[j].toFixed(2)+' N'));
  assert.ok(status.innerHTML.includes(h.command_budgets_sofar[j].toFixed(3)+' N·s'));
  if(h.clear_index!==null&&j>=h.clear_index)assert.ok(status.innerHTML.includes('达到 t_clear='));
 }
 const l1=candidate.weights[j].reduce((s,w,k)=>s+Math.abs(w-reference.weights[j][k]),0);
 assert.equal(Number(panel.querySelector('.pair-metrics').dataset.l1),l1);
 const status=panel.querySelector('.replay-status');assert.equal(status.dataset.referenceId,reference.profile_id);assert.equal(status.dataset.probeId,candidate.profile_id);assert.equal(Number(status.dataset.time),candidate.times[j]);
 for(const cursor of panel.cursors)assert.equal(Number(cursor.attrs.x1),60+candidate.times[j]/1.2*572);
}
for(const panel of panels){
 for(const ref of ['standard_2p2N','constant_1p6N']){
  panel.querySelector('.reference-selector').value=ref;panel.querySelector('.reference-selector').handlers.change();
  for(const p of data.snapshot.settings.probes){
   panel.querySelector('.probe-selector').value=p.id;panel.querySelector('.probe-selector').handlers.change();
   for(let j=0;j<25;j++){panel.querySelector('.time-slider').value=j;panel.querySelector('.time-slider').handlers.input();check(panel,j);checked++;}
  }
 }
 // Other profiles can also serve as a Reference without losing synchronization.
 for(const ref of data.snapshot.settings.probes){panel.querySelector('.reference-selector').value=ref.id;panel.querySelector('.reference-selector').handlers.change();check(panel,24);}
 panel.querySelector('.first').click();check(panel,0);panel.querySelector('.last').click();check(panel,24);
 panel.querySelector('.play').click();check(panel,0);
 for(let j=1;j<=24;j++){for(const f of [...timers.values()])f();check(panel,j);}
 assert.equal(timers.size,0);
 panel.querySelector('.play').click();panel.querySelector('.play').click();assert.equal(timers.size,0);
 console.log(`Case ${panel.dataset.case}: profile/reference selectors, 25 synchronized frames, all particles, budgets, metrics, clear markers and playback passed`);
}
const result={passed:true,cases:panels.length,profiles:data.snapshot.settings.probes.length,referenceProfilesTested:5,sliderFramesChecked:checked,allParticlesChecked:true,playbackPassed:true,method:'actual report JS in a DOM substitute'};
fs.writeFileSync(path.join(output,'replay_validation.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result,null,2));
