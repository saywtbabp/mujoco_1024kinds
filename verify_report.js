// Test the report's OWN replay script and controls with a small DOM substitute.
// No browser/network packages are required; no alternate plotting implementation.
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
const root=__dirname,latest=fs.readFileSync(path.join(root,'results','LATEST.txt'),'utf8').trim();
const html=fs.readFileSync(path.join(latest,'report.html'),'utf8');
const payloads=[...html.matchAll(/<script type="application\/json" class="history-data">([\s\S]*?)<\/script>/g)].map(m=>JSON.parse(m[1]));
assert.equal(payloads.length,5);
class Element {
 constructor(){this.attrs={};this.children=[];this.handlers={};this.textContent='';this.value='';this.disabled=false;}
 setAttribute(k,v){this.attrs[k]=String(v);}
 append(el){if(el.parent){el.parent.children=el.parent.children.filter(x=>x!==el);}el.parent=this;this.children.push(el);}
 addEventListener(k,f){this.handlers[k]=f;}
 get firstChild(){return this.children[0];}
 click(){if(!this.disabled)this.handlers.click();}
}
class Panel extends Element {
 constructor(data){super();this.data=data;this.svg=new Element();this.slider=new Element();this.status=new Element();this.script=new Element();this.script.textContent=JSON.stringify(data);this.note=new Element();this.buttons=['first','play','probe','last'].map(c=>{const b=new Element();b.dataset={command:c};return b;});}
 querySelector(selector){if(selector==='svg')return this.svg;if(selector==='input')return this.slider;if(selector==='.history-status')return this.status;if(selector==='.history-data')return this.script;if(selector==='.history-note')return this.note;const m=selector.match(/data-command="(\w+)"/);if(m)return this.buttons.find(b=>b.dataset.command===m[1]);throw Error(selector);}
 querySelectorAll(selector){assert.equal(selector,'button');return this.buttons;}
}
const panels=payloads.map(d=>new Panel(d));
const script=[...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].at(-1)[1];
const intervals=new Map();let timerId=0;
const context={document:{querySelectorAll:s=>{assert.equal(s,'.belief-replay');return panels;},createElementNS:()=>new Element()},setInterval:f=>{intervals.set(++timerId,f);return timerId;},clearInterval:id=>intervals.delete(id)};
vm.runInNewContext(script,context,{timeout:10000});
function check(panel,index){
 assert.equal(Number(panel.slider.value),index);
 const group=panel.svg.children.find(e=>e.children.length===1024),data=panel.data;
 assert.ok(group,'all 1024 hypotheses must be rendered');
 let p=0;for(const dot of group.children){const id=Number(dot.firstChild.textContent.match(/i=(\d+)/)[1]);if(dot.attrs.fill==='#059669')p+=data.weights[index][id];}
 assert.ok(Math.abs(p-data.p_D[index])<1e-10);
 assert.ok(panel.status.textContent.includes(`D=${data.D[index].toFixed(6)}`));
}
panels.forEach((panel,k)=>{
 const data=panel.data,last=data.time.length-1;
 check(panel,last);panel.buttons[0].click();check(panel,0);
 panel.slider.value=Math.floor(last/2);panel.slider.handlers.input();check(panel,Math.floor(last/2));
 panel.buttons[2].click();check(panel,Math.max(0,data.phase.lastIndexOf('PROBE_COMPLETE')));
 panel.buttons[3].click();check(panel,last);
 if(last>0){panel.buttons[1].click();check(panel,0);for(let i=0;i<=last;i++){for(const f of [...intervals.values()])f();}check(panel,last);assert.equal(intervals.size,0);}
 else assert.ok(panel.buttons[1].disabled);
 console.log(`Case ${'ABCDE'[k]}: slider, phase controls, weighted support and playback passed`);
});
