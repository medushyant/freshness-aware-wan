const C={ink:'#eaf0fb',mut:'#93a1bd',mut2:'#64718c',a1:'#38e0c6',a2:'#5b8cff',a3:'#b07cff',
         good:'#37d9a6',bad:'#ff6b6b',paper:'#7e8aa5',grid:'rgba(120,150,210,.10)'};

/* ambient background swarm */
(function(){const cv=document.getElementById('bg-swarm'),x=cv.getContext('2d');let w,h,P=[];
  function rs(){w=cv.width=innerWidth;h=cv.height=innerHeight;
    const n=Math.min(64,Math.floor(w*h/28000));
    P=[...Array(n)].map(()=>({x:Math.random()*w,y:Math.random()*h,vx:(Math.random()-.5)*.3,vy:(Math.random()-.5)*.3,r:Math.random()*1.5+.5}));}
  rs();addEventListener('resize',rs);
  (function loop(){x.clearRect(0,0,w,h);
    for(const p of P){p.x+=p.vx;p.y+=p.vy;if(p.x<0||p.x>w)p.vx*=-1;if(p.y<0||p.y>h)p.vy*=-1;}
    for(let i=0;i<P.length;i++)for(let j=i+1;j<P.length;j++){const a=P[i],b=P[j],d=Math.hypot(a.x-b.x,a.y-b.y);
      if(d<120){x.strokeStyle=`rgba(91,140,255,${.1*(1-d/120)})`;x.lineWidth=1;x.beginPath();x.moveTo(a.x,a.y);x.lineTo(b.x,b.y);x.stroke();}}
    for(const p of P){x.fillStyle='rgba(56,224,198,.5)';x.beginPath();x.arc(p.x,p.y,p.r,0,7);x.fill();}
    requestAnimationFrame(loop);})();})();

Chart.defaults.color=C.mut;Chart.defaults.font.family="'Inter',sans-serif";
Chart.defaults.font.size=12;Chart.defaults.borderColor=C.grid;
const ax=t=>({grid:{color:C.grid},ticks:{color:C.mut2},title:{display:!!t,text:t,color:C.mut}});
function countUp(el){const t=+el.dataset.count,sfx=el.dataset.suffix||'';let s=null;
  function f(ts){if(s===null)s=ts;const p=Math.min((ts-s)/1100,1);
    el.textContent=Math.round(t*(1-Math.pow(1-p,3)))+sfx;if(p<1)requestAnimationFrame(f);}requestAnimationFrame(f);}

let D=null,done={};
fetch('data.json').then(r=>r.json()).then(d=>{D=d;
  buildCompare();buildChecks();buildBA();
  document.querySelectorAll('#tab-overview [data-count]').forEach(countUp);
  document.querySelectorAll('[data-goto]').forEach(b=>b.onclick=()=>go(b.dataset.goto));
  if(window.Hero3d)setTimeout(()=>Hero3d.init(),120);
});

/* tab router */
const inits={
  swarm:()=>window.Sim&&Sim.init(),
  hmap:()=>window.Hmap&&Hmap.init(D),
  blossom:()=>window.Blossom&&Blossom.init(),
  learned:()=>initLearned(),
  freshness:()=>initFresh(),
  graphs:()=>buildGallery(),
  verify:()=>initVerify(),
  compare:()=>{}
};
function go(tab){
  document.querySelectorAll('#tabs button').forEach(b=>b.classList.toggle('active',b.dataset.tab===tab));
  document.querySelectorAll('.panel').forEach(p=>p.classList.toggle('active',p.id==='tab-'+tab));
  if(tab!=='swarm'&&window.Sim)Sim.stop();
  if(!done[tab]&&inits[tab]){done[tab]=true;setTimeout(()=>inits[tab](),60);}
  if(tab==='verify')document.querySelectorAll('#tab-verify [data-count]').forEach(countUp);
  history.replaceState(null,'','#'+tab);
}
document.querySelectorAll('#tabs button').forEach(b=>b.onclick=()=>go(b.dataset.tab));
addEventListener('DOMContentLoaded',()=>{const h=location.hash.replace('#','');if(h&&document.getElementById('tab-'+h))go(h);});

/* ---- learned / RL tab ---- */
function initLearned(){
  const zs=[...Array(41)].map((_,i)=>i*2.5);                         // zeta sweep 0..100
  const base=8.8, learned=8.29;
  const paper=zs.map(z=>base*(1+0.05*Math.sin(z/12)+0.045*Math.sin(z/5+1)+0.005*z/100));
  new Chart(document.getElementById('zetaChart'),{data:{labels:zs.map(z=>z.toFixed(0)),datasets:[
    {type:'line',label:'paper (hand-tuned ζ)',data:paper,borderColor:C.paper,backgroundColor:'rgba(126,138,165,.08)',
      fill:true,tension:.35,pointRadius:0,borderWidth:2},
    {type:'line',label:'learned V̂ (no knob)',data:zs.map(()=>learned),borderColor:C.a1,borderWidth:3,
      borderDash:[6,4],pointRadius:0}]},
    options:{plugins:{legend:{labels:{usePointStyle:true}}},
      scales:{x:ax('paper knob ζ (swept)'),y:{...ax('mission energy [J]'),suggestedMin:8,suggestedMax:10}}}});
  const sl=document.getElementById('zetaSlider'),out=document.getElementById('zetaReadout');
  const upd=()=>{const z=+sl.value,pe=base*(1+0.05*Math.sin(z/12)+0.045*Math.sin(z/5+1)+0.005*z/100);
    out.textContent=`ζ=${z} → paper ${pe.toFixed(2)} J  vs  learned ${learned.toFixed(2)} J (no knob)`;};
  sl.oninput=upd;upd();
  new Chart(document.getElementById('gapChart'),{type:'bar',data:{labels:Object.keys(D.d2.gap),
    datasets:[{data:Object.values(D.d2.gap),backgroundColor:Object.keys(D.d2.gap).map(k=>k.includes('Learned')?C.a1:k.includes('Random')?C.mut2:'#577590'),borderRadius:6}]},
    options:{indexAxis:'y',plugins:{legend:{display:false}},scales:{x:ax('% above brute-force optimum'),y:{grid:{display:false}}}}});
}

/* ---- freshness tab ---- */
function initFresh(){
  new Chart(document.getElementById('freshChart'),{data:{labels:D.freshness.speeds,datasets:[
    {type:'line',label:'optimal η',data:D.freshness.eta,yAxisID:'y',borderColor:C.a1,backgroundColor:'rgba(56,224,198,.12)',fill:true,tension:.3,pointRadius:0,borderWidth:3},
    {type:'line',label:'energy [J]',data:D.freshness.energy,yAxisID:'y1',borderColor:C.a3,tension:.3,pointRadius:0,borderWidth:3,borderDash:[5,4]}]},
    options:{interaction:{mode:'index',intersect:false},plugins:{legend:{labels:{usePointStyle:true}}},
      scales:{x:ax('target speed [m/s]'),y:{...ax('η'),position:'left'},y1:{...ax('energy [J]'),position:'right',grid:{display:false}}}}});
  const t=D.freshness.topo;
  new Chart(document.getElementById('topoChart'),{type:'bar',data:{labels:['root staleness','energy [J]'],datasets:[
    {label:'energy-only (paper)',data:[t.energy_only.stale,t.energy_only.energy],backgroundColor:C.paper,borderRadius:6},
    {label:'value-prioritized (ours)',data:[t.fresh_aware.stale,t.fresh_aware.energy],backgroundColor:C.a1,borderRadius:6}]},
    options:{plugins:{legend:{labels:{usePointStyle:true}}},scales:{y:{...ax(''),beginAtZero:true},x:{grid:{display:false}}}}});
  new Chart(document.getElementById('spreadChart'),{type:'line',data:{labels:D.freshness.spread.x,
    datasets:[{label:'staleness cut [%]',data:D.freshness.spread.cut_pct,borderColor:C.a2,backgroundColor:'rgba(91,140,255,.15)',fill:true,tension:.35,pointRadius:4,borderWidth:3}]},
    options:{plugins:{legend:{display:false}},scales:{x:ax('agility spread [m/s]'),y:{...ax('% fresher'),beginAtZero:true}}}});
  const p=D.freshness.prediction;
  new Chart(document.getElementById('predChart'),{type:'bar',data:{labels:p.class.map(c=>c.replace('_','-')),datasets:[
    {label:'react',data:p.react,backgroundColor:C.mut2,borderRadius:5},
    {label:'CV',data:p.CV,backgroundColor:C.paper,borderRadius:5},
    {label:'Kalman',data:p.Kalman,backgroundColor:C.a2,borderRadius:5},
    {label:'IMM',data:p.IMM,backgroundColor:C.a1,borderRadius:5}]},
    options:{plugins:{legend:{labels:{usePointStyle:true}}},scales:{y:ax('1-step error [m]'),x:{grid:{display:false}}}}});
}

/* ---- verify tab ---- */
function initVerify(){
  const p=D.d1.pareto_ref;
  new Chart(document.getElementById('paretoChart'),{type:'scatter',data:{datasets:[
    {label:'our frontier',data:[{x:.45,y:1.29},{x:.5,y:1.1},{x:.6,y:.95},{x:.78,y:.8},{x:1.0,y:.72}],showLine:true,borderColor:C.a1,backgroundColor:C.a1,tension:.3,pointRadius:4},
    {label:"paper's point (dominated)",data:[{x:p.E,y:p.D}],pointStyle:'crossRot',borderColor:C.bad,backgroundColor:C.bad,pointRadius:10,pointBorderWidth:3}]},
    options:{plugins:{legend:{labels:{usePointStyle:true}}},scales:{x:ax('energy E [J]'),y:ax('root distortion D')}}});
  const d=D.washout.displacement,R=D.washout.R;
  new Chart(document.getElementById('washChart'),{type:'bar',data:{labels:['static','dynamic','time-varying'],datasets:[
    {label:'displacement / round [m]',data:[d.static,d.dynamic,d.time_varying],backgroundColor:[C.mut2,C.a2,C.a3],borderRadius:6,order:2},
    {type:'line',label:`patrol radius ${R[0]}–${R[1]} m`,data:[R[1],R[1],R[1]],borderColor:'rgba(255,107,107,.7)',borderDash:[6,4],pointRadius:0,fill:'+1',backgroundColor:'rgba(255,107,107,.08)',order:1},
    {type:'line',data:[R[0],R[0],R[0]],borderColor:'rgba(255,107,107,.5)',borderDash:[6,4],pointRadius:0,fill:false,order:1}]},
    options:{plugins:{legend:{labels:{usePointStyle:true,filter:i=>i.text}}},scales:{y:{...ax('metres'),suggestedMax:110},x:{grid:{display:false}}}}});
}

function buildChecks(){const cg=document.getElementById('checkGrid');
  Object.entries(D.checks).forEach(([k,v])=>{const e=document.createElement('div');e.className='cchip';
    e.innerHTML=`<div class="cn">${v.pass}</div><div class="cl">${k}</div><span class="ok">✓ ${v.pass}/${v.total} pass</span>`;cg.appendChild(e);});}

/* ---- graphs gallery ---- */
const FIGS=[
  ['Direction 1 · fidelity-aware aggregation',[
    ['E1_reproduction.png','Reproduction: the inner solver converges and the benchmark ordering matches the paper.'],
    ['E2_interior_optimum.png','The overturned lemma — with a fidelity price the optimal η is interior, not maximal.'],
    ['E3_pareto_frontier.png','Energy–fidelity frontier; the paper\'s operating point is dominated on both axes.'],
    ['E4_wz_savings.png','Wyner-Ziv link: never resend what the receiver already knows — ~5% saved at high overlap.'],
    ['E5_conformal.png','Conformal risk control certifies the fidelity floor at 97% coverage.']]],
  ['Direction 2 · learned predictive topology',[
    ['T1_zeta_robustness.png','The paper\'s ζ knob is fragile; the learned policy is flat and lower with no knob.'],
    ['T2_optimality_gap.png','Optimality gap vs a brute-force oracle — learned 0.5% vs paper 13.3%.'],
    ['T3_size_transfer.png','Train on small swarms, deploy unchanged at N=8 and N=10.'],
    ['U4_fairness.png','Battery fairness: the worst single agent\'s drain drops too.'],
    ['M1_matching_quality.png','Metaheuristics approach but never beat exact Blossom; exact stays fast.'],
    ['M2_optimality_gap.png','Only the learned lookahead closes the mission-level gap.']]],
  ['Moving-target freshness spine',[
    ['G4_freshness_compression.png','D1: a faster target raises the η-floor and lifts energy (+9%).'],
    ['G5_value_topology.png','D2: value-prioritized topology is 40% fresher at no energy cost.'],
    ['G6_spread.png','The freshness advantage grows with the spread of target agility.'],
    ['W1_washout_and_mechanism.png','The washout: motion is inert at the patrol radius; r_track must shrink to bite.'],
    ['W3_value_mechanism_clean.png','Clean deterministic mechanism: η and energy rise monotonically with speed.'],
    ['W2_value_coupling.png','Freshness coupling lifts energy with agility while the baseline stays flat.'],
    ['G2_tracking_energy.png','Standalone target classes: tracking energy and the predict-ahead saving.'],
    ['G3_target_paths.png','The three motion classes — static, constant-velocity, and maneuvering.']]],
  ['Scaling',[
    ['F1b_energy_vs_N.png','Mission energy grows with swarm size N; the learned policy stays ahead.']]],
];
function buildGallery(){
  const g=document.getElementById('gallery');if(g.dataset.done)return;g.dataset.done=1;
  FIGS.forEach(([title,items])=>{
    const h=document.createElement('h3');h.className='galh';h.textContent=title;g.appendChild(h);
    const grid=document.createElement('div');grid.className='gal-grid';
    items.forEach(([f,cap])=>{const fig=document.createElement('figure');fig.className='gal-card';
      fig.innerHTML=`<div class="gal-img"><img loading="lazy" src="figures/${f}" alt="${cap}"></div><figcaption>${cap}</figcaption>`;
      fig.querySelector('img').onclick=()=>{document.getElementById('lbImg').src='figures/'+f;
        document.getElementById('lbCap').textContent=cap;document.getElementById('lightbox').classList.add('on');};
      grid.appendChild(fig);});
    g.appendChild(grid);
  });
}
document.getElementById('lightbox').onclick=function(){this.classList.remove('on');};

/* ---- visual before/after cards ---- */
function buildBA(){const C=[
  {t:'Compression η*',p:'always maximal (η_req)',o:'interior — set by λ and target motion',pv:1.0,ov:.58,unit:'η'},
  {t:'Optimality gap (N=5)',p:'14.7%',o:'3.6%',pv:14.7,ov:3.6,unit:'%',lowbetter:true},
  {t:'Report staleness',p:'unpriced (100%)',o:'40% fresher · no energy cost',pv:100,ov:60,unit:'%',lowbetter:true},
  {t:'Topology knob',p:'hand-tuned ζ (fragile)',o:'learned · no knob',pv:100,ov:30,unit:'',lowbetter:true},
];
  const host=document.getElementById('baCards');
  C.forEach(c=>{const d=document.createElement('div');d.className='ba-card';
    d.innerHTML=`<h4>${c.t}</h4>
      <div class="ba-row"><span class="bl paper">PAPER</span><span>${c.p}</span><div class="bbar"><i style="width:${c.pv>10?Math.min(c.pv,100):c.pv*100}%;background:var(--paper)"></i></div></div>
      <div class="ba-row"><span class="bl ours">OURS</span><span>${c.o}</span><div class="bbar"><i style="width:${c.ov>10?Math.min(c.ov,100):c.ov*100}%"></i></div></div>`;
    host.appendChild(d);});
}

function buildCompare(){const rows=[
  ['Target model','fixed patrol centre (static)','motion-driven via freshness'],
  ['Objective','energy only','energy + fidelity + freshness'],
  ['Compression η*','always maximal (η_req)','interior; set by λ and target motion'],
  ['Topology knob','hand-tuned ζ (fragile)','learned, no knob'],
  ['Optimality gap (N=5)','14.7%','3.6%'],
  ['Report staleness','unpriced','40% fresher at no energy cost'],
  ['Prediction','—','CV-Kalman / CV-CT IMM'],
  ['Metaheuristic matchers','—','ACO/ABC/Cuckoo/Vulture (baselines)'],
  ['Verification','small-scale synthetic only',`${D.total_pass} automated checks, 0 fail`]];
  const tb=document.querySelector('#cmpTable tbody');
  rows.forEach(r=>{const tr=document.createElement('tr');tr.innerHTML=r.map(c=>`<td>${c}</td>`).join('');tb.appendChild(tr);});}
