// Optional real Chromium/Edge smoke test, using Node's DevTools WebSocket client.
// No npm install; this launches an isolated headless browser, never the user's session.
const fs = require('node:fs'), os = require('node:os'), path = require('node:path');
const {spawn} = require('node:child_process'), {pathToFileURL} = require('node:url');
const assert = require('node:assert/strict');
const output = path.resolve(process.argv[2] || '');
assert.ok(fs.existsSync(path.join(output, 'report.html')), 'pass a comparison output directory');
const browserExe = process.argv[3] || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
assert.ok(fs.existsSync(browserExe), 'pass the path to installed Chrome/Edge as the second argument');
const profile = fs.mkdtempSync(path.join(os.tmpdir(),'probe_report_browser_'));
const proc = spawn(browserExe, ['--headless=new', '--disable-gpu', '--no-first-run', '--no-default-browser-check',
  '--remote-debugging-port=0', '--window-size=1440,1100', '--force-device-scale-factor=1',
  `--user-data-dir=${profile}`, 'about:blank'], {stdio:'ignore', windowsHide:true});
const pause = ms => new Promise(r => setTimeout(r,ms));
let socket, nextId=0;
const pending = new Map(), errors=[];
function send(method, params={}) {
  const id=++nextId;
  return new Promise((resolve,reject) => {
    const timer=setTimeout(() => {pending.delete(id);reject(Error('CDP timeout: '+method));},60000);
    pending.set(id,{resolve,reject,timer}); socket.send(JSON.stringify({id,method,params}));
  });
}
async function evaluate(expression) {
  const result=await send('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});
  if(result.exceptionDetails) throw Error(JSON.stringify(result.exceptionDetails));
  return result.result.value;
}
(async()=>{
  try {
    const portFile=path.join(profile,'DevToolsActivePort');
    for(let i=0;i<100 && !fs.existsSync(portFile);i++) await pause(100);
    assert.ok(fs.existsSync(portFile),'headless browser must expose its test endpoint');
    const port=fs.readFileSync(portFile,'utf8').split('\n')[0];
    const pages=await (await fetch(`http://127.0.0.1:${port}/json`)).json();
    const page=pages.find(p=>p.type==='page'); assert.ok(page);
    socket=new WebSocket(page.webSocketDebuggerUrl);
    await new Promise((resolve,reject)=>{socket.onopen=resolve;socket.onerror=reject;});
    socket.onmessage=event=>{
      const message=JSON.parse(event.data);
      if(message.id && pending.has(message.id)) {
        const p=pending.get(message.id); pending.delete(message.id); clearTimeout(p.timer);
        message.error?p.reject(Error(JSON.stringify(message.error))):p.resolve(message.result);
      }
      if(message.method==='Runtime.exceptionThrown') errors.push(message.params);
    };
    await send('Page.enable');await send('Runtime.enable');
    await send('Page.navigate',{url:pathToFileURL(path.join(output,'report.html')).href});
    for(let i=0;i<100;i++) {
      if(await evaluate("document.readyState === 'complete' && document.querySelectorAll('.particles circle').length > 0")) break;
      await pause(100);
    }
    const result=await evaluate(`(${function(){
      const data=JSON.parse(document.querySelector('#comparison-data').textContent),standard=data.snapshot.settings.probes.find(p=>p.standard);
      const check=(condition,message)=>{if(!condition)throw Error(message);};
      let checked=0;
      for(const panel of document.querySelectorAll('.case-panel')) {
        const c=panel.dataset.case,selector=panel.querySelector('.probe-selector'),slider=panel.querySelector('.time-slider');
        const histories=data.cases[c],baseline=histories[standard.id];
        const left=panel.querySelector('.standard-scatter'),right=panel.querySelector('.candidate-scatter');
        check(left.querySelectorAll('.particles circle').length===data.theta.length,'all baseline particles');
        check(right.querySelectorAll('.particles circle').length===data.theta.length,'all candidate particles');
        for(const probe of data.snapshot.settings.probes) {
          selector.value=probe.id;selector.dispatchEvent(new Event('change'));
          const h=histories[probe.id];
          for(let j=0;j<h.times.length;j++) {
            slider.value=j;slider.dispatchEvent(new Event('input'));
            const current=panel.querySelector('.candidate-metrics'),base=panel.querySelector('.standard-metrics');
            check(Number(current.dataset.p)===h.metrics[j].p_D,'display p_D equals JSON');
            check(Number(current.dataset.d)===h.metrics[j].D,'display D equals JSON');
            check(Number(current.dataset.ess)===h.metrics[j].ESS,'display ESS equals JSON');
            check(Number(base.dataset.p)===baseline.metrics[j].p_D,'standard uses synchronized time');
            check(current.dataset.tClear===String(h.t_clear),'display t_clear equals JSON');
            check(Number(panel.querySelector('.replay-status').dataset.time)===h.times[j],'status time equals JSON');
            for(const id of [0,20,200,512,1023]) {
              const el=right.querySelector('[data-world-id="'+id+'"]');
              check(el.querySelector('title').textContent.includes('w='+h.weights[j][id]+';'),'particle weight equals JSON');
              check(el.getAttribute('fill')===(data.labels[id]?'#059669':'#dc2626'),'fixed Push/Avoid color');
              check(Number(el.getAttribute('opacity'))>=.07,'all particles visible');
            }
            checked++;
          }
        }
        panel.querySelector('.first').click();check(Number(slider.value)===0,'first button');
        panel.querySelector('.last').click();check(Number(slider.value)===baseline.times.length-1,'last button');
        panel.querySelector('.showcase')?.click();
        const selected=histories[selector.value];
        check(selected.metrics[Number(slider.value)].D<data.snapshot.settings.D_clear,'low-D showcase');
      }
      return {cases:Object.keys(data.cases).length,probes:data.snapshot.settings.probes.length,sliderFramesChecked:checked};
    }} )()`);
    // Verify actual asynchronous playback reaches the last frame and stops.
    await evaluate("document.querySelector('.case-panel .last').click(); document.querySelector('.case-panel .speed').value='1'; document.querySelector('.case-panel .play').click()");
    for(let i=0;i<100;i++) {
      if(await evaluate("document.querySelector('.case-panel .play').textContent === '播放权重变化' && Number(document.querySelector('.case-panel .time-slider').value) > 0"))break;
      await pause(100);
    }
    assert.ok(await evaluate("Number(document.querySelector('.case-panel .time-slider').value) === Number(document.querySelector('.case-panel .time-slider').max)"),'playback reaches last frame');
    await evaluate("(()=>{const p=document.querySelector('.case-panel');p.querySelector('.probe-selector').value='probe_1p0N';p.querySelector('.probe-selector').dispatchEvent(new Event('change'));p.querySelector('.time-slider').value=13;p.querySelector('.time-slider').dispatchEvent(new Event('input'));p.scrollIntoView();})()");
    const shot=await send('Page.captureScreenshot',{format:'png'});
    fs.writeFileSync(path.join(output,'report_browser_preview.png'),Buffer.from(shot.data,'base64'));
    assert.equal(errors.length,0,'no browser JS exceptions');
    const audit={passed:true,...result,playbackPassed:true,browserExceptions:errors.length};
    fs.writeFileSync(path.join(output,'browser_validation.json'),JSON.stringify(audit,null,2));
    console.log(JSON.stringify(audit,null,2));
  } finally {
    if(socket)socket.close();proc.kill();await pause(500);
    // Verify the absolute target before recursively removing this test's own temp profile.
    if(path.dirname(path.resolve(profile))!==path.resolve(os.tmpdir()) || !path.basename(profile).startsWith('probe_report_browser_'))throw Error('unsafe temp cleanup target');
    try{fs.rmSync(profile,{recursive:true,force:true,maxRetries:3,retryDelay:100});}catch(e){console.warn('Temporary browser profile retained:',profile);}
  }
})().catch(error=>{console.error(error);process.exitCode=1;});
