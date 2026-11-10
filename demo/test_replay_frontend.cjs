const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const read=name=>fs.readFileSync(path.join(__dirname,name),'utf8').replace(/\r\n/g,'\n');

function page(){
  const elements=new Map();
  function element(selector){if(!elements.has(selector))elements.set(selector,{innerHTML:'',textContent:'',value:'',title:'',style:{},handlers:{},hidden:true,clientWidth:1260,clientHeight:600,
    addEventListener(name,cb){this.handlers[name]=cb;},getBoundingClientRect:()=>({left:0,top:0}),setPointerCapture(){},
    showModal(){this.open=true;},close(){this.open=false;}});return elements.get(selector);}
  const document={querySelector:element,handlers:{},addEventListener(name,cb){this.handlers[name]=cb;}};
  const ctx=vm.createContext({document,window:{addEventListener(){}},
    fetch(){throw Error('No backend request is allowed');},setTimeout(){throw Error('No playback timer is allowed');}});
  vm.runInContext(read('static/replay-data.js'),ctx);vm.runInContext(read('static/replay.js'),ctx);
  return {element,document,run:code=>vm.runInContext(code,ctx)};
}

test('each saved query renders its own graph and all recorded scores immediately',()=>{
  const p=page();assert.equal(p.run('bundle.queries.length'),5);
  assert.equal((p.element('#query-select').innerHTML.match(/<option/g)||[]).length,5);
  for(const id of p.run('bundle.queries.map(q=>q.id)')){
    p.element('#query-select').handlers.change({target:{value:id}});
    const svg=p.element('#world').innerHTML;
    assert.equal(p.element('#query-select').value,id);
    assert.equal(p.element('#query-select').title,p.run('data.question'));
    assert.equal((svg.match(/data-node=/g)||[]).length,p.run('data.nodes.length'));
    assert.equal((svg.match(/data-process="gnn"/g)||[]).length,2); // Header and main block.
    assert.equal((svg.match(/data-score=/g)||[]).length,5);
    assert.match(svg,/INPUT GRAPH/);assert.match(svg,/GRAPH NEURAL NETWORK/);assert.match(svg,/FINAL NODE SCORES/);
    assert.doesNotMatch(svg,/NaN|undefined/);
    for(const score of p.run('data.results.map(r=>r.score.toFixed(6))'))assert.ok(svg.includes(score));
  }
});

test('switching queries clears previous selection and explanation and restores the graph view',()=>{
  const p=page();p.run('selectScore(data.results[0].node_id)');assert.equal(p.element('#detail-title').textContent,'Cascade Range');
  p.run('zoom(2);selectQuery("query-2")');
  assert.equal(p.run('state.selected'),null);assert.equal(p.element('#inspector').hidden,true);
  assert.equal(p.element('#detail-content').innerHTML,'');assert.ok(p.run('state.scale')<2);
  p.run('selectScore(data.results[0].node_id)');assert.equal(p.element('#detail-title').textContent,'Rock music');
  assert.match(p.element('#detail-content').innerHTML,/0\.105725/);
  assert.doesNotMatch(p.element('#detail-content').innerHTML,/Cascade Range/);
});

test('nodes, relations, GNN operations and score readout have query-specific explanations',()=>{
  const p=page();
  for(const id of p.run('bundle.queries.map(q=>q.id)')){
    p.run(`selectQuery(${JSON.stringify(id)})`);
    for(const nodeId of p.run('data.nodes.map(n=>n.id)')){p.run(`selectNode(${JSON.stringify(nodeId)})`);assert.equal(p.element('#inspector').hidden,false);assert.ok(p.element('#detail-content').innerHTML);}
    for(const component of ['input','gnn','messages','updates','readout','scores']){p.run(`selectComponent(${JSON.stringify(component)})`);assert.ok(p.element('#detail-content').innerHTML.length>80);}
    p.run('selectComponent("input")');assert.ok(p.element('#detail-content').innerHTML.includes(p.run('data.nodes.length')+' nodes'));
    p.run('selectComponent("scores")');assert.ok(p.element('#detail-content').innerHTML.includes(p.run('data.provenance.full_candidate_count.toLocaleString("en-US")')));
    p.run('activate({closest:s=>s==="[data-edge]"?{dataset:{edge:"0"}}:null})');assert.match(p.element('#detail-content').innerHTML,/saved path/);
    p.element('#detail-close').handlers.click();assert.equal(p.element('#inspector').hidden,true);
  }
});

test('keyboard activation opens the selected operation without starting playback',()=>{
  const p=page();p.element('#viewport').handlers.keydown({key:'Enter',preventDefault(){},target:{closest:s=>s.includes('data-process')?{dataset:{process:'messages'}}:null}});
  assert.equal(p.run('state.selected'),'messages');assert.equal(p.element('#detail-title').textContent,'Pass messages');
  p.document.handlers.keydown({key:'Escape'});assert.equal(p.element('#inspector').hidden,true);
});

test('query selector retains the compact English interface without playback or dataset UI',()=>{
  for(const name of ['static/replay.html','static/replay.js','static/replay.css'])assert.doesNotMatch(read(name),/WebQSP|PubMedQA|QASPER|dataset-switch|sample-list|Dense|[\u4e00-\u9fff]/i);
  assert.doesNotMatch(read('static/replay.html'),/id="(?:play|previous|next|speed|reset|stages)"/);
  assert.doesNotMatch(read('static/replay.js'),/setTimeout|requestAnimationFrame|animateMotion|state\.phase/);
});

test('zoom and narrow-screen area focus stay finite for every query',()=>{
  const p=page(),old=p.run('state.scale');p.element('#zoom-in').handlers.click();assert.ok(p.run('state.scale')>old);
  p.element('#viewport').clientWidth=380;
  for(const id of p.run('bundle.queries.map(q=>q.id)'))for(const area of ['graph','gnn','scores']){
    p.run(`selectQuery(${JSON.stringify(id)});state.focus=${JSON.stringify(area)};fit()`);
    assert.equal(p.run('state.fittedFocus'),area);assert.ok(p.run('Number.isFinite(state.scale)&&state.scale>0'));
  }
});

test('the standalone artifact embeds its current assets and makes no backend requests',()=>{
  const html=read('rearev_replay.html');
  for(const name of ['static/replay.js','static/replay.css','static/replay-data.js'])assert.ok(html.includes(read(name)));
  assert.doesNotMatch(html,/<script\b[^>]*\bsrc\s*=|<link[^>]+rel="stylesheet"/i);
  assert.doesNotMatch(read('static/replay.js'),/\bfetch\s*\(|XMLHttpRequest|WebSocket|\/api\//);
});
