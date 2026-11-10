"use strict";
const $=s=>document.querySelector(s);
const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const bundle=globalThis.GNN_DEMO;
let data=bundle.queries[0];
const WIDTH=1420,HEIGHT=640;
const state={queryId:data.id,selected:null,scale:1,x:0,y:0,focus:'graph',fittedFocus:null};
const number=v=>v==null?'Not recorded':v>0&&v<.00001?Number(v).toExponential(4):Number(v).toFixed(6);
const short=(s,n=23)=>s.length>n?s.slice(0,n-1)+'…':s;

function layout(){
  const positions={},incoming=new Set(data.edges.map(e=>e.target)),roots=data.nodes.filter(n=>!incoming.has(n.id));
  roots.forEach((n,i)=>positions[n.id]={x:110,y:250+i*100});
  const direct=data.nodes.filter(n=>data.edges.some(e=>e.target===n.id&&roots.some(r=>r.id===e.source)));
  direct.forEach((n,i)=>positions[n.id]={x:390,y:115+i*Math.min(64,410/Math.max(1,direct.length-1))});
  data.nodes.filter(n=>!positions[n.id]).forEach((n,i)=>positions[n.id]={x:555,y:510-i*90});
  return positions;
}
function edgePath(edge,positions){
  const a=positions[edge.source],b=positions[edge.target],dx=b.x-a.x,dy=b.y-a.y,len=Math.hypot(dx,dy)||1;
  return `M${a.x+dx/len*24} ${a.y+dy/len*24} L${b.x-dx/len*28} ${b.y-dy/len*28}`;
}
function sectionHeading(label,x,component){return `<g class="section-link" data-process="${component}" role="button" tabindex="0" aria-label="Inspect ${label}"><rect x="${x-10}" y="44" width="265" height="42" rx="6" fill="transparent"/><text class="column-heading" x="${x}" y="70">${label}</text></g>`;}
function render(){
  const positions=layout(),max=Math.max(...data.results.map(r=>r.score));
  const graphEdges=data.edges.map((e,i)=>{const d=edgePath(e,positions);return `<g class="relation ${state.selected==='edge:'+i?'selected':''}"><path d="${d}" marker-end="url(#arrow)"/><path d="${d}" class="edge-hit" data-edge="${i}"/><title>${esc(e.relation)}</title></g>`;}).join('');
  const graphNodes=data.nodes.map(n=>{
    const p=positions[n.id],root=n.kind==='source',selected=state.selected===n.id,rank=data.results.find(r=>r.node_id===n.id)?.rank;
    const fill=n.score!=null?`hsl(135 23% ${95-21*n.score/max}%)`:root?'#e4eddc':'#f8faf4';
    return `<g class="node ${root?'source-node':''} ${selected?'selected':''}" transform="translate(${p.x},${p.y})" data-node="${n.id}" role="button" tabindex="0" aria-label="Inspect ${esc(n.label)}">${rank===1?'<circle class="best-ring" r="31"/>':''}<circle class="node-face" r="${root?25:22}" fill="${fill}"/><text class="node-id" text-anchor="middle" y="4">${root?'S':n.id.toUpperCase()}</text><text class="node-label" text-anchor="middle" y="42">${esc(short(n.label))}</text><title>${esc(n.label)}</title></g>`;
  }).join('');
  const operations=[['messages','Pass messages',300],['updates','Update nodes',345],['readout','Score nodes',390]];
  const gnn=`<g class="gnn-block ${['gnn','messages','updates','readout'].includes(state.selected)?'selected':''}"><g data-process="gnn" role="button" tabindex="0" aria-label="Inspect the schematic GNN process"><rect class="gnn-face" x="710" y="205" width="230" height="224" rx="17"/><text class="gnn-title" x="825" y="253" text-anchor="middle">GNN</text><path class="gnn-divider" d="M741 273 H909"/></g>${operations.map(([id,label,y])=>`<g class="gnn-operation ${state.selected===id?'selected':''}" data-process="${id}" role="button" tabindex="0" aria-label="Inspect ${label}"><rect x="737" y="${y-19}" width="176" height="29" rx="5"/><text class="gnn-step" x="825" y="${y}" text-anchor="middle">${label}</text></g>`).join('')}<text class="inner-arrow" x="825" y="321" text-anchor="middle">↓</text><text class="inner-arrow" x="825" y="366" text-anchor="middle">↓</text><text class="schematic-label" x="825" y="455" text-anchor="middle">Schematic</text></g>`;
  const scores=data.results.map((r,i)=>{
    const y=116+i*88,selected=state.selected===r.node_id;
    return `<g class="score-row ${selected?'selected':''}" data-score="${r.node_id}" role="button" tabindex="0" aria-label="Inspect ${esc(r.label)} final score"><rect class="score-hit" x="1015" y="${y-10}" width="364" height="73" rx="9"/><text class="score-node" x="1030" y="${y+11}">${r.node_id.toUpperCase()}</text><text class="score-label" x="1064" y="${y+11}">${esc(short(r.label,26))}</text><text class="score-value" x="1359" y="${y+37}" text-anchor="end">${number(r.score)}</text><rect class="bar-track" x="1030" y="${y+27}" width="235" height="6" rx="3"/><rect class="bar-fill" x="1030" y="${y+27}" width="${235*r.score/max}" height="6" rx="3"/><title>${esc(r.label)} · ${r.score}</title></g>`;
  }).join('');
  const arrows='<g class="pipeline-links"><path d="M623 317 H691" marker-end="url(#arrow)"/><path d="M957 317 H1004" marker-end="url(#arrow)"/></g>';
  $('#world').innerHTML=`<svg width="${WIDTH}" height="${HEIGHT}" viewBox="0 0 ${WIDTH} ${HEIGHT}" role="group" aria-label="Input graph through GNN to recorded node scores"><defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M1 1 L9 5 L1 9" fill="none" stroke="#9bae87" stroke-width="1.6"/></marker></defs><rect class="graph-background" x="22" y="35" width="592" height="565" rx="16"/>${sectionHeading('01 / INPUT GRAPH',49,'input')}${sectionHeading('02 / GRAPH NEURAL NETWORK',710,'gnn')}${sectionHeading('03 / FINAL NODE SCORES',1030,'scores')}${graphEdges}${graphNodes}${arrows}${gnn}${scores}</svg>`;
  $('#world').style.width=WIDTH+'px';$('#world').style.height=HEIGHT+'px';
  if($('#viewport').clientWidth<650&&state.fittedFocus!==state.focus)fit();else transform();renderDetail();
}
function selectQuery(id){
  const next=bundle.queries.find(q=>q.id===id);if(!next)return;
  data=next;state.queryId=id;state.focus='graph';state.fittedFocus=null;drag=null;suppressClick=false;
  closeDetail();$('#detail-title').textContent='';$('#detail-content').innerHTML='';$('#query-select').value=id;$('#query-select').title=data.question;render();fit();
}
function selectNode(id,focus='graph'){if(!data.nodes.some(n=>n.id===id))return;state.selected=id;state.focus=focus;$('#inspector').hidden=false;render();}
function selectScore(id){if(data.results.some(r=>r.node_id===id))selectNode(id,'scores');}
function selectComponent(id){if(!['input','gnn','messages','updates','readout','scores'].includes(id))return;state.selected=id;state.focus=id==='input'?'graph':id==='scores'?'scores':'gnn';$('#inspector').hidden=false;render();}
function closeDetail(){state.selected=null;$('#inspector').hidden=true;}
function renderDetail(){
  if(!state.selected)return;
  const components={
    input:['INPUT GRAPH','Entities and relationships',`The question is matched against connected candidate nodes. This local view has ${data.nodes.length} nodes and ${data.edges.length} saved relations. Select a node or edge to inspect it.`, 'This is a subset expanded from saved paths, not the complete model input.'],
    gnn:['SCHEMATIC PROCESS','Graph neural network','Combine neighbor messages and relation information under question guidance, update node representations, then read out a score for each candidate.','Intermediate tensors and message weights are unavailable. This block illustrates the process only.'],
    messages:['GNN OPERATION','Pass messages','Connected nodes supply information about their neighbors and the relations joining them. The question guides which signals are relevant.','No numerical messages or learned edge weights are available for this view.'],
    updates:['GNN OPERATION','Update nodes','Combine a node representation with information gathered from its neighbors. Repeated updates let connected evidence influence the final representation.','This describes the role of node updates, not an exact recovered layer architecture.'],
    readout:['GNN OPERATION','Score nodes','The final node representations are passed to the model readout to produce candidate scores. The results at right are copied from the saved model output.','No intermediate representations or readout calculations are simulated.'],
    scores:['RECORDED OUTPUT','Final node scores',`The five highest-ranked candidates are shown. Their probabilities were produced over ${data.provenance.full_candidate_count.toLocaleString('en-US')} original candidates, not just the visible nodes.`, 'Values are not renormalized here. Bar lengths are scaled to the largest visible score.']
  };
  if(components[state.selected]){const [kind,title,body,note]=components[state.selected];$('#detail-kind').textContent=kind;$('#detail-title').textContent=title;$('#detail-content').innerHTML=`<p>${esc(body)}</p><small>${esc(note)}</small>`;return;}
  if(state.selected.startsWith('edge:')){const e=data.edges[Number(state.selected.slice(5))];if(!e)return;const a=data.nodes.find(n=>n.id===e.source),b=data.nodes.find(n=>n.id===e.target);$('#detail-kind').textContent='SAVED RELATION';$('#detail-title').textContent=a.label+' → '+b.label;$('#detail-content').innerHTML=`<code>${esc(e.relation)}</code><p>This relation comes from a saved path between these nodes. A ~ prefix marks traversal opposite to the original relation direction.</p>`;return;}
  const n=data.nodes.find(n=>n.id===state.selected);if(!n)return;
  $('#detail-kind').textContent=n.kind==='source'?'SOURCE NODE':'CANDIDATE NODE';$('#detail-title').textContent=n.label;
  $('#detail-content').innerHTML=(n.score!=null?`<div class="detail-score"><span>Recorded final score</span><strong>${number(n.score)}</strong></div>`:'')+(n.text?`<p class="path-text">${esc(n.text)}</p>`:'<p>The starting entity of the visible graph. Its outgoing relations connect it to the candidate nodes.</p>');
}
function transform(){$('#world').style.transform=`translate(${state.x}px,${state.y}px) scale(${state.scale})`;$('#zoom-label').textContent=Math.round(state.scale*100)+'%';}
function fit(){const v=$('#viewport'),region=v.clientWidth<650?{graph:{x:22,y:35,w:592,h:565},gnn:{x:680,y:145,w:300,h:350},scores:{x:1004,y:70,w:390,h:510}}[state.focus]:{x:0,y:0,w:WIDTH,h:HEIGHT};state.scale=Math.min(1.1,(v.clientWidth-24)/region.w,(v.clientHeight-20)/region.h);state.x=(v.clientWidth-region.w*state.scale)/2-region.x*state.scale;state.y=(v.clientHeight-region.h*state.scale)/2-region.y*state.scale;state.fittedFocus=state.focus;transform();}
function zoom(value,point){const v=$('#viewport'),old=state.scale,next=Math.max(.15,Math.min(2.3,value)),p=point||{x:v.clientWidth/2,y:v.clientHeight/2};state.x=p.x-(p.x-state.x)*next/old;state.y=p.y-(p.y-state.y)*next/old;state.scale=next;transform();}

$('#query-select').innerHTML=bundle.queries.map(q=>`<option value="${q.id}">${esc(q.question)}</option>`).join('');
$('#query-select').addEventListener('change',e=>selectQuery(e.target.value));
$('#focus-controls').addEventListener('click',e=>{const b=e.target.closest('[data-focus]');if(b){state.focus=b.dataset.focus;fit();}});
$('#zoom-in').addEventListener('click',()=>zoom(state.scale*1.2));$('#zoom-out').addEventListener('click',()=>zoom(state.scale/1.2));$('#fit').addEventListener('click',fit);
$('#detail-close').addEventListener('click',()=>{closeDetail();render();});
$('#about-open').addEventListener('click',()=>$('#about').showModal());$('#about-close').addEventListener('click',()=>$('#about').close());
const viewport=$('#viewport');let drag=null,suppressClick=false;
viewport.addEventListener('pointerdown',e=>{if(e.target.closest('[data-node],[data-score],[data-edge],[data-process],#inspector'))return;drag={x:e.clientX,y:e.clientY,originX:state.x,originY:state.y,moved:false};viewport.setPointerCapture(e.pointerId);});
viewport.addEventListener('pointermove',e=>{if(!drag)return;if(Math.hypot(e.clientX-drag.x,e.clientY-drag.y)>4)drag.moved=true;state.x=drag.originX+e.clientX-drag.x;state.y=drag.originY+e.clientY-drag.y;transform();});
viewport.addEventListener('pointerup',()=>{suppressClick=Boolean(drag?.moved);drag=null;});viewport.addEventListener('pointercancel',()=>{drag=null;suppressClick=false;});
function activate(target){const process=target.closest('[data-process]');if(process){selectComponent(process.dataset.process);return;}const score=target.closest('[data-score]');if(score){selectScore(score.dataset.score);return;}const n=target.closest('[data-node]');if(n){selectNode(n.dataset.node);return;}const e=target.closest('[data-edge]');if(e){state.selected='edge:'+e.dataset.edge;state.focus='graph';$('#inspector').hidden=false;render();}}
viewport.addEventListener('click',e=>{if(suppressClick){suppressClick=false;return;}activate(e.target);});
viewport.addEventListener('keydown',e=>{if(['Enter',' '].includes(e.key)&&e.target.closest('[data-node],[data-score],[data-edge],[data-process]')){e.preventDefault();activate(e.target);}});
viewport.addEventListener('wheel',e=>{if(e.target.closest('#inspector'))return;e.preventDefault();const r=viewport.getBoundingClientRect();zoom(state.scale*(e.deltaY>0?.9:1.1),{x:e.clientX-r.left,y:e.clientY-r.top});},{passive:false});
document.addEventListener('keydown',e=>{if(e.key==='Escape'){closeDetail();render();}});window.addEventListener('resize',fit);
selectQuery(data.id);
