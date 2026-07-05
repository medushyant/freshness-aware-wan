/* H-MAP aggregation tournament + Blossom matching, animated (SVG + GSAP).
   H-MAP replays the REAL per-round pairing trace from data.json as a clean
   left-to-right bracket; payload rings and the quality meter illustrate the
   mechanics (compression, fusion, fidelity floor).
   UI v3: drone-chip nodes with rotor dots, gradient links, glowing packets. */
const NS='http://www.w3.org/2000/svg';
const Cc={ink:'#f2eefc',a1:'#3ce0c3',a2:'#7c93ff',a3:'#b78aff',mut:'#7a7099',
  bad:'#ff7385',good:'#43e5ae',paper:'#8f87a8',amber:'#ffb26b'};
function el(t,a){const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);return e;}

/* a small quad-rotor chip, centered at 0,0 (SVG group content) */
function droneChip(id,scale=1){
  const g=el('g',{});
  const arms=el('g',{stroke:'rgba(220,226,255,.8)','stroke-width':2*scale,'stroke-linecap':'round'});
  [[-10,-10],[10,-10],[-10,10],[10,10]].forEach(([dx,dy])=>{
    arms.appendChild(el('line',{x1:0,y1:0,x2:dx*scale,y2:dy*scale}));
    g.appendChild(el('circle',{cx:dx*scale,cy:dy*scale,r:3.6*scale,fill:'none',
      stroke:Cc.a1,'stroke-width':1.4,opacity:.95}));
    g.appendChild(el('circle',{cx:dx*scale,cy:dy*scale,r:1.4*scale,fill:'#100c1e'}));
  });
  g.insertBefore(arms,g.firstChild);
  g.appendChild(el('rect',{x:-8*scale,y:-6.5*scale,width:16*scale,height:13*scale,rx:4*scale,
    fill:'url(#bodyg)',stroke:'rgba(15,12,26,.9)','stroke-width':1.3}));
  const t=el('text',{y:4*scale,'text-anchor':'middle',fill:'#0b0716',
    'font-size':9.5*scale,'font-weight':'800','font-family':'JetBrains Mono'});
  t.textContent=id;
  g.appendChild(t);
  return g;
}

function sharedDefs(svg){
  const d=el('defs',{});
  const body=el('linearGradient',{id:'bodyg',x1:0,y1:0,x2:1,y2:1});
  body.append(el('stop',{offset:'0%','stop-color':'#c3cdff'}),el('stop',{offset:'100%','stop-color':'#7c93ff'}));
  const link=el('linearGradient',{id:'linkg',x1:0,y1:0,x2:1,y2:0});
  link.append(el('stop',{offset:'0%','stop-color':Cc.a1}),el('stop',{offset:'100%','stop-color':Cc.a3}));
  const f=el('filter',{id:'glow',x:'-60%',y:'-60%',width:'220%',height:'220%'});
  const fb=el('feGaussianBlur',{stdDeviation:3.2,result:'b'});
  const fm=el('feMerge',{});fm.append(el('feMergeNode',{in:'b'}),el('feMergeNode',{in:'SourceGraphic'}));
  f.append(fb,fm);
  const soft=el('filter',{id:'softsh',x:'-40%',y:'-40%',width:'180%',height:'180%'});
  soft.append(el('feDropShadow',{dx:0,dy:3,stdDeviation:4,'flood-color':'#05030e','flood-opacity':.55}));
  d.append(body,link,f,soft);svg.appendChild(d);
}

/* ---------------- H-MAP ---------------- */
window.Hmap=(function(){
  let svg,data,mode='paper',inited=false;
  let cols=[],steps=[],node={},edges=[],cur=0,playing=false;
  const VW=720,VH=440,R=21,COLS=[95,300,490,625];
  const ETA={paper:0.62,ours:0.55};

  function ring(frac){const c=2*Math.PI*(R+7);return `${Math.max(frac,0.02)*c} ${c}`;}

  function build(){
    svg.innerHTML='';node={};edges=[];steps=[];
    sharedDefs(svg);
    // round bands
    COLS.forEach((cx,i)=>{
      svg.appendChild(el('rect',{x:cx-56,y:16,width:112,height:VH-58,rx:18,
        fill:i%2?'rgba(124,147,255,.035)':'rgba(60,224,195,.03)',
        stroke:'rgba(168,160,220,.07)'}));
    });
    const tr=data.trace[mode].rounds;
    const pay=(data.payloads0||[18,20,20,17,24,17]).slice();
    const maxP=Math.max(...pay)*1.6;
    let prev={};[0,1,2,3,4,5].forEach((id,i)=>prev[id]={x:COLS[0],y:52+i*66,col:0,L:pay[id],id});
    cols=[Object.values(prev).map(p=>({...p}))];
    tr.forEach((rd,ri)=>{
      const surv=[];
      rd.pairs.forEach(([s,r])=>{
        if(r===null){surv.push({id:s,mid:prev[s].y,from:[s],L:prev[s].L});}
        else{surv.push({id:r,mid:(prev[s].y+prev[r].y)/2,from:[s,r],L:0});
          steps.push({s,r,round:rd.round,col:ri});}
      });
      surv.sort((a,b)=>a.mid-b.mid);
      const x=COLS[ri+1],n=surv.length,top=76,bot=VH-76,gap=n>1?(bot-top)/(n-1):0;
      const next={};
      surv.forEach((sv,k)=>{const y=n>1?top+k*gap:VH/2;next[sv.id]={x,y,col:ri+1,L:sv.L,from:sv.from};});
      steps.filter(st=>st.col===ri).forEach(st=>{st.tx=next[st.r].x;st.ty=next[st.r].y;});
      Object.entries(next).forEach(([id,p])=>prev[id]=p);
      cols[ri+1]=Object.entries(next).map(([id,p])=>({id:+id,...p}));
    });
    steps.forEach(st=>{
      const a=findPos(st.s,st.col),b=findPos(st.r,st.col),t={x:st.tx,y:st.ty};
      edges.push(curve(a,t,'rgba(168,160,220,.10)'));
      edges.push(curve(b,t,'rgba(168,160,220,.10)'));
    });
    cols.forEach((cl,ci)=>cl.forEach(p=>{
      const g=el('g',{transform:`translate(${p.x},${p.y})`,opacity:ci===0?1:.10});
      const halo=el('circle',{r:R+12,fill:'rgba(124,147,255,.09)'});
      const rg=el('circle',{r:R+7,fill:'none',stroke:'url(#linkg)','stroke-width':4.5,
        'stroke-dasharray':ring(Math.min((p.L||0)/maxP,1)),'stroke-linecap':'round',
        transform:'rotate(-90)',opacity:.9});
      const plate=el('circle',{r:R,fill:'rgba(20,16,36,.92)',stroke:'rgba(168,160,220,.35)',
        'stroke-width':1.2,filter:'url(#softsh)'});
      g.append(halo,rg,plate,droneChip(p.id+1,0.92));
      svg.appendChild(g);
      node[p.col+'_'+p.id]={g,rg,maxP,L:p.L||0};
    }));
    ['agents','round 1','round 2','root'].forEach((lab,i)=>{
      const t=el('text',{x:COLS[i],y:VH-22,'text-anchor':'middle',fill:Cc.mut,
        'font-size':11.5,'letter-spacing':'2','font-family':'JetBrains Mono'});
      t.textContent=lab.toUpperCase();svg.appendChild(t);
    });
    cur=0;updateMeters(0);
  }
  function findPos(id,col){const c=cols[col].find(p=>p.id===id);return c?{x:c.x,y:c.y}:{x:COLS[col],y:VH/2};}
  function curve(a,b,stroke){const mx=(a.x+b.x)/2;
    const p=el('path',{d:`M${a.x},${a.y} C${mx},${a.y} ${mx},${b.y} ${b.x},${b.y}`,fill:'none',
      stroke,'stroke-width':2,'stroke-dasharray':400,'stroke-dashoffset':400});
    svg.insertBefore(p,svg.firstChild.nextSibling);return p;}

  function updateMeters(frac){
    const E=data.trace[mode].E*frac;
    document.getElementById('energyBar').style.width=Math.min(frac*100,100)+'%';
    document.getElementById('energyVal').textContent=E.toFixed(2)+' J';
    const Dmax=1.2,D=(mode==='paper'?2.21:0.95)*frac,qb=document.getElementById('qualBar');
    qb.style.width=Math.min(D/2.5*100,100)+'%';
    qb.style.background=(D>Dmax)?'linear-gradient(90deg,#ff7385,#ffb26b)':'linear-gradient(90deg,#43e5ae,#3ce0c3)';
    document.getElementById('qualVal').textContent=frac>0?('D='+D.toFixed(2)+(D>Dmax?' ✗ over floor':' ✓ under floor')):'—';
  }

  function doStep(){
    if(cur>=steps.length)return false;
    const st=steps[cur];
    const sNode=node[st.col+'_'+st.s],rPos={x:st.tx,y:st.ty};
    const survNode=node[(st.col+1)+'_'+st.r];
    const a=findPos(st.s,st.col);
    const live=curve(a,rPos,'url(#linkg)');live.setAttribute('stroke-width',3);
    live.setAttribute('filter','url(#glow)');
    const packet=el('circle',{cx:a.x,cy:a.y,r:6.5,fill:Cc.a1,filter:'url(#glow)'});svg.appendChild(packet);
    const trailA=el('circle',{cx:a.x,cy:a.y,r:3.4,fill:'rgba(60,224,195,.5)'});svg.appendChild(trailA);
    const tl=gsap.timeline({onComplete:()=>{cur++;updateMeters(cur/steps.length);}});
    tl.to(sNode.g,{scale:1.14,transformOrigin:'center',duration:.2,svgOrigin:`${a.x} ${a.y}`},0)
      .to(live,{strokeDashoffset:0,duration:.5,ease:'power2.inOut'},0)
      .to(sNode.rg,{attr:{'stroke-dasharray':ring(Math.min(sNode.L*ETA[mode]/sNode.maxP,1))},duration:.35},.1)
      .to(packet,{attr:{cx:rPos.x,cy:rPos.y},duration:.55,ease:'power1.inOut'},.25)
      .to(trailA,{attr:{cx:rPos.x,cy:rPos.y},duration:.62,ease:'power1.inOut',opacity:0},.3)
      .to(sNode.g,{opacity:.10,duration:.35},.55)
      .to(survNode.g,{opacity:1,duration:.35},.5)
      .add(()=>{const L=(sNode.L*ETA[mode])+(survNode.L*ETA[mode]||sNode.L*ETA[mode]);
        survNode.L=Math.min(L,survNode.maxP);
        gsap.fromTo(survNode.g,{scale:.6,svgOrigin:`${rPos.x} ${rPos.y}`},{scale:1,duration:.45,ease:'back.out(2.2)'});
        gsap.to(survNode.rg,{attr:{'stroke-dasharray':ring(Math.min(survNode.L/survNode.maxP,1))},duration:.4});
        gsap.to(packet,{opacity:0,scale:2.4,duration:.32,svgOrigin:`${rPos.x} ${rPos.y}`,onComplete:()=>{packet.remove();trailA.remove();}});},.6);
    return true;
  }
  function play(){if(playing)return;playing=true;(function loop(){if(cur<steps.length){doStep();setTimeout(loop,1250);}else playing=false;})();}

  return{init(dd){data=dd;svg=document.getElementById('hmapSvg');if(!svg)return;
    svg.setAttribute('viewBox',`0 0 ${VW} ${VH}`);
    if(!inited){
      document.getElementById('hmapPlay').onclick=()=>play();
      document.getElementById('hmapStep').onclick=()=>{if(!playing)doStep();};
      document.getElementById('hmapReset').onclick=()=>{playing=false;build();};
      const mp=document.getElementById('modePaper'),mo=document.getElementById('modeOurs');
      mp.onclick=()=>{mode='paper';mp.classList.add('active');mo.classList.remove('active');playing=false;build();};
      mo.onclick=()=>{mode='ours';mo.classList.add('active');mp.classList.remove('active');playing=false;build();
        document.getElementById('hmapNote').textContent='Ours keeps compression interior so the report stays under the fidelity floor (green), and prioritizes fast-decaying sources — fresher at no energy cost.';};
      inited=true;
    }
    build();
  }};
})();

/* ---------------- Blossom matching ---------------- */
window.Blossom=(function(){
  let svg,n=6,P=[],W=[],best=null,inited=false,table;
  const R=152,cx=260,cy=205;
  function gen(){P=[];for(let i=0;i<n;i++){const a=-Math.PI/2+i/n*2*Math.PI;P.push({x:cx+R*Math.cos(a),y:cy+R*Math.sin(a)});}
    W=[...Array(n)].map(()=>Array(n).fill(0));
    for(let i=0;i<n;i++)for(let j=i+1;j<n;j++){const w=+(0.6+Math.random()*2.6).toFixed(2);W[i][j]=W[j][i]=w;}
    best=optimal();draw(null);table.innerHTML='';}
  function matchings(ids){if(ids.length===0)return[[]];const a=ids[0],out=[];
    for(let k=1;k<ids.length;k++){const b=ids[k],rest=ids.filter((_,i)=>i!==0&&i!==k);
      for(const m of matchings(rest))out.push([[a,b],...m]);}return out;}
  function cost(m){return m.reduce((s,[a,b])=>s+W[a][b],0);}
  function optimal(){let bm=null,bc=1e9;for(const m of matchings([...Array(n).keys()])){const c=cost(m);if(c<bc){bc=c;bm=m;}}return{m:bm,c:bc};}
  function greedy(){const e=[];for(let i=0;i<n;i++)for(let j=i+1;j<n;j++)e.push([i,j,W[i][j]]);e.sort((a,b)=>a[2]-b[2]);
    const u=new Set(),m=[];for(const[i,j]of e){if(!u.has(i)&&!u.has(j)){m.push([i,j]);u.add(i);u.add(j);}}return{m,c:cost(m)};}
  function randomM(){const ids=[...Array(n).keys()].sort(()=>Math.random()-.5),m=[];for(let i=0;i<n;i+=2)m.push([ids[i],ids[i+1]]);return{m,c:cost(m)};}
  function draw(sel,color){svg.innerHTML='';
    sharedDefs(svg);
    svg.appendChild(el('circle',{cx,cy,r:R,fill:'none',stroke:'rgba(168,160,220,.08)',
      'stroke-width':1,'stroke-dasharray':'3 8'}));
    for(let i=0;i<n;i++)for(let j=i+1;j<n;j++){
      const on=sel&&sel.some(([a,b])=>(a===i&&b===j)||(a===j&&b===i));
      const ln=el('line',{x1:P[i].x,y1:P[i].y,x2:P[j].x,y2:P[j].y,
        stroke:on?(color||'url(#linkg)'):'rgba(168,160,220,.10)','stroke-width':on?4.5:1,
        'stroke-linecap':'round'});
      if(on)ln.setAttribute('filter','url(#glow)');
      svg.appendChild(ln);
      if(on){const mx=(P[i].x+P[j].x)/2,my=(P[i].y+P[j].y)/2;
        const pill=el('rect',{x:mx-24,y:my-12,width:48,height:19,rx:9.5,
          fill:'rgba(20,16,36,.95)',stroke:'rgba(168,160,220,.35)','stroke-width':1});
        const t=el('text',{x:mx,y:my+2,'text-anchor':'middle',fill:color&&color!=='url(#linkg)'?color:Cc.a1,
          'font-size':10.5,'font-weight':'600','font-family':'JetBrains Mono'});
        t.textContent=W[i][j].toFixed(2)+' J';
        svg.appendChild(pill);svg.appendChild(t);
        gsap.from(ln,{attr:{'stroke-width':0},duration:.45,ease:'power2.out'});
        gsap.from([pill,t],{opacity:0,y:6,duration:.4,delay:.15,stagger:.02});}}
    for(let i=0;i<n;i++){
      const g=el('g',{transform:`translate(${P[i].x},${P[i].y})`});
      g.appendChild(el('circle',{r:22,fill:'rgba(124,147,255,.10)'}));
      g.appendChild(el('circle',{r:17,fill:'rgba(20,16,36,.94)',stroke:'rgba(168,160,220,.35)',
        'stroke-width':1.2,filter:'url(#softsh)'}));
      g.appendChild(droneChip(i+1,0.8));
      svg.appendChild(g);}}
  function row(name,c,win){const tr=document.createElement('tr');if(win)tr.className='win';
    tr.innerHTML=`<td>${name}</td><td>${c.toFixed(2)} J</td>`;table.appendChild(tr);}
  return{init(){svg=document.getElementById('blossomSvg');if(!svg)return;table=document.getElementById('blTable');
    if(!inited){document.getElementById('blNew').onclick=gen;
      document.getElementById('blShow').onclick=()=>{draw(best.m);table.innerHTML='';row('Blossom (exact)',best.c,true);};
      document.getElementById('blGreedy').onclick=()=>{const g=greedy();draw(g.m,Cc.paper);table.innerHTML='';row('greedy',g.c);row('Blossom (exact)',best.c,true);};
      document.getElementById('blRandom').onclick=()=>{const r=randomM();draw(r.m,Cc.bad);table.innerHTML='';row('random',r.c);row('Blossom (exact)',best.c,true);};
      inited=true;}
    gen();}};
})();
