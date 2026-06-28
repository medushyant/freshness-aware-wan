/* H-MAP aggregation tournament + Blossom matching, animated (SVG + GSAP).
   H-MAP replays the REAL per-round pairing trace from data.json as a clean
   left-to-right bracket; payload rings and the quality meter illustrate the
   mechanics (compression, fusion, fidelity floor). */
const NS='http://www.w3.org/2000/svg';
const Cc={ink:'#eaf0fb',a1:'#38e0c6',a2:'#5b8cff',a3:'#b07cff',mut:'#64718c',bad:'#ff6b6b',good:'#37d9a6',paper:'#7e8aa5'};
function el(t,a){const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);return e;}

/* ---------------- H-MAP ---------------- */
window.Hmap=(function(){
  let svg,data,mode='paper',inited=false;
  let cols=[],steps=[],node={},edges=[],cur=0,playing=false;
  const VW=720,VH=440,R=19,COLS=[95,300,490,625];
  const ETA={paper:0.62,ours:0.55};

  function ring(frac){const c=2*Math.PI*(R+6);return `${Math.max(frac,0.02)*c} ${c}`;}

  function build(){
    svg.innerHTML='';node={};edges=[];steps=[];
    const tr=data.trace[mode].rounds;
    const pay=(data.payloads0||[18,20,20,17,24,17]).slice();
    const maxP=Math.max(...pay)*1.6;
    // column 0
    let prev={};[0,1,2,3,4,5].forEach((id,i)=>prev[id]={x:COLS[0],y:46+i*68,col:0,L:pay[id],id});
    cols=[Object.values(prev).map(p=>({...p}))];
    // build survivor columns with even vertical spacing (no collisions)
    tr.forEach((rd,ri)=>{
      const surv=[];
      rd.pairs.forEach(([s,r])=>{
        if(r===null){surv.push({id:s,mid:prev[s].y,from:[s],L:prev[s].L});}
        else{surv.push({id:r,mid:(prev[s].y+prev[r].y)/2,from:[s,r],L:0});
          steps.push({s,r,round:rd.round,col:ri});}
      });
      surv.sort((a,b)=>a.mid-b.mid);
      const x=COLS[ri+1],n=surv.length,top=70,bot=VH-70,gap=n>1?(bot-top)/(n-1):0;
      const next={};
      surv.forEach((sv,k)=>{const y=n>1?top+k*gap:VH/2;next[sv.id]={x,y,col:ri+1,L:sv.L,from:sv.from};});
      // attach target coords to the steps of this round
      steps.filter(st=>st.col===ri).forEach(st=>{st.tx=next[st.r].x;st.ty=next[st.r].y;});
      cols.push(Object.values(next).map(p=>({id:Object.keys(next).find(k=>next[k]===p)})).map(()=>0)); // placeholder
      // store positions
      Object.entries(next).forEach(([id,p])=>prev[id]=p);
      cols[ri+1]=Object.entries(next).map(([id,p])=>({id:+id,...p}));
    });
    // faint full-tree connectors
    steps.forEach(st=>{
      const a=findPos(st.s,st.col),b=findPos(st.r,st.col),t={x:st.tx,y:st.ty};
      edges.push(curve(a,t,'rgba(120,150,210,.12)'));
      edges.push(curve(b,t,'rgba(120,150,210,.12)'));
    });
    // nodes for every column
    cols.forEach((cl,ci)=>cl.forEach(p=>{
      const g=el('g',{transform:`translate(${p.x},${p.y})`,opacity:ci===0?1:.12});
      const halo=el('circle',{r:R+9,fill:'rgba(91,140,255,.10)'});
      const rg=el('circle',{r:R+6,fill:'none',stroke:Cc.a1,'stroke-width':4,
        'stroke-dasharray':ring(Math.min((p.L||0)/maxP,1)),'stroke-linecap':'round',transform:'rotate(-90)',opacity:.85});
      const c=el('circle',{r:R,fill:'url(#nodeg)',stroke:Cc.a2,'stroke-width':2});
      const t=el('text',{y:5,'text-anchor':'middle',fill:Cc.ink,'font-size':13,'font-family':'JetBrains Mono'});
      t.textContent=(p.id+1);
      g.append(halo,rg,c,t);svg.appendChild(g);
      node[p.col+'_'+p.id]={g,rg,maxP,L:p.L||0};
    }));
    // round labels
    ['agents','round 1','round 2','root'].forEach((lab,i)=>{
      const t=el('text',{x:COLS[i],y:VH-22,'text-anchor':'middle',fill:Cc.mut,'font-size':12,'font-family':'JetBrains Mono'});
      t.textContent=lab;svg.appendChild(t);
    });
    cur=0;updateMeters(0);
  }
  function findPos(id,col){const c=cols[col].find(p=>p.id===id);return c?{x:c.x,y:c.y}:{x:COLS[col],y:VH/2};}
  function curve(a,b,stroke){const mx=(a.x+b.x)/2;
    const p=el('path',{d:`M${a.x},${a.y} C${mx},${a.y} ${mx},${b.y} ${b.x},${b.y}`,fill:'none',
      stroke,'stroke-width':2,'stroke-dasharray':400,'stroke-dashoffset':400});
    svg.insertBefore(p,svg.firstChild);return p;}

  function updateMeters(frac){
    const E=data.trace[mode].E*frac;
    document.getElementById('energyBar').style.width=Math.min(frac*100,100)+'%';
    document.getElementById('energyVal').textContent=E.toFixed(2)+' J';
    const Dmax=1.2,D=(mode==='paper'?2.21:0.95)*frac,qb=document.getElementById('qualBar');
    qb.style.width=Math.min(D/2.5*100,100)+'%';
    qb.style.background=(D>Dmax)?'linear-gradient(90deg,#ff6b6b,#ff9a6b)':'linear-gradient(90deg,#37d9a6,#38e0c6)';
    document.getElementById('qualVal').textContent=frac>0?('D='+D.toFixed(2)+(D>Dmax?' ✗ over floor':' ✓ under floor')):'—';
  }

  function doStep(){
    if(cur>=steps.length)return false;
    const st=steps[cur];
    const sNode=node[st.col+'_'+st.s],rPos={x:st.tx,y:st.ty};
    const survNode=node[(st.col+1)+'_'+st.r];
    const a=findPos(st.s,st.col);
    const live=curve(a,rPos,Cc.a1);live.setAttribute('stroke-width',2.5);
    const packet=el('circle',{cx:a.x,cy:a.y,r:6,fill:Cc.a1,filter:'url(#glow)'});svg.appendChild(packet);
    const tl=gsap.timeline({onComplete:()=>{cur++;updateMeters(cur/steps.length);}});
    tl.to(sNode.g,{scale:1.12,transformOrigin:'center',duration:.2,svgOrigin:`${a.x} ${a.y}`},0)
      .to(live,{strokeDashoffset:0,duration:.5,ease:'power2.inOut'},0)
      .to(sNode.rg,{attr:{'stroke-dasharray':ring(Math.min(sNode.L*ETA[mode]/sNode.maxP,1))},duration:.35},.1) // compress
      .to(packet,{attr:{cx:rPos.x,cy:rPos.y},duration:.55,ease:'power1.inOut'},.25)
      .to(sNode.g,{opacity:.12,duration:.35},.55)                                                              // retire
      .to(survNode.g,{opacity:1,duration:.35},.5)                                                             // survivor appears
      .add(()=>{const L=(sNode.L*ETA[mode])+(survNode.L*ETA[mode]||sNode.L*ETA[mode]);survNode.L=Math.min(L,survNode.maxP);
        gsap.fromTo(survNode.g,{scale:.6,svgOrigin:`${rPos.x} ${rPos.y}`},{scale:1,duration:.4,ease:'back.out(2)'});
        gsap.to(survNode.rg,{attr:{'stroke-dasharray':ring(Math.min(survNode.L/survNode.maxP,1))},duration:.4});
        gsap.to(packet,{opacity:0,scale:2,duration:.3,svgOrigin:`${rPos.x} ${rPos.y}`,onComplete:()=>packet.remove()});},.6);
    return true;
  }
  function play(){if(playing)return;playing=true;(function loop(){if(cur<steps.length){doStep();setTimeout(loop,1250);}else playing=false;})();}

  function defs(){
    const d=el('defs',{});
    const g=el('radialGradient',{id:'nodeg'});g.append(el('stop',{offset:'0%','stop-color':'#1b3a63'}),el('stop',{offset:'100%','stop-color':'#0c1830'}));
    const f=el('filter',{id:'glow'});const fb=el('feGaussianBlur',{stdDeviation:3,result:'b'});
    const fm=el('feMerge',{});fm.append(el('feMergeNode',{in:'b'}),el('feMergeNode',{in:'SourceGraphic'}));f.append(fb,fm);
    d.append(g,f);svg.appendChild(d);
  }
  return{init(dd){data=dd;svg=document.getElementById('hmapSvg');if(!svg)return;
    svg.setAttribute('viewBox',`0 0 ${VW} ${VH}`);
    if(!inited){
      document.getElementById('hmapPlay').onclick=()=>play();
      document.getElementById('hmapStep').onclick=()=>{if(!playing)doStep();};
      document.getElementById('hmapReset').onclick=()=>{playing=false;defs();build();};
      const mp=document.getElementById('modePaper'),mo=document.getElementById('modeOurs');
      mp.onclick=()=>{mode='paper';mp.classList.add('active');mo.classList.remove('active');playing=false;defs();build();};
      mo.onclick=()=>{mode='ours';mo.classList.add('active');mp.classList.remove('active');playing=false;defs();build();
        document.getElementById('hmapNote').textContent='Ours keeps compression interior so the report stays under the fidelity floor (green), and prioritizes fast-decaying sources — fresher at no energy cost.';};
      inited=true;
    }
    defs();build();
  }};
})();

/* ---------------- Blossom matching ---------------- */
window.Blossom=(function(){
  let svg,n=6,P=[],W=[],best=null,inited=false,table;
  const R=150,cx=260,cy=205;
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
    for(let i=0;i<n;i++)for(let j=i+1;j<n;j++){const on=sel&&sel.some(([a,b])=>(a===i&&b===j)||(a===j&&b===i));
      const ln=el('line',{x1:P[i].x,y1:P[i].y,x2:P[j].x,y2:P[j].y,stroke:on?(color||Cc.a1):'rgba(120,150,210,.12)','stroke-width':on?4:1});
      svg.appendChild(ln);
      if(on){const mx=(P[i].x+P[j].x)/2,my=(P[i].y+P[j].y)/2;
        const t=el('text',{x:mx,y:my-5,'text-anchor':'middle',fill:color||Cc.a1,'font-size':11,'font-family':'JetBrains Mono'});
        t.textContent=W[i][j].toFixed(2);svg.appendChild(t);gsap.from(ln,{attr:{'stroke-width':0},duration:.4});}}
    for(let i=0;i<n;i++){const c=el('circle',{cx:P[i].x,cy:P[i].y,r:17,fill:'rgba(91,140,255,.18)',stroke:Cc.a2,'stroke-width':2});
      const t=el('text',{x:P[i].x,y:P[i].y+4,'text-anchor':'middle',fill:Cc.ink,'font-size':12,'font-family':'JetBrains Mono'});t.textContent=i+1;svg.appendChild(c);svg.appendChild(t);}}
  function row(name,c,win){const tr=document.createElement('tr');if(win)tr.className='win';tr.innerHTML=`<td>${name}</td><td>${c.toFixed(2)} J</td>`;table.appendChild(tr);}
  return{init(){svg=document.getElementById('blossomSvg');if(!svg)return;table=document.getElementById('blTable');
    if(!inited){document.getElementById('blNew').onclick=gen;
      document.getElementById('blShow').onclick=()=>{draw(best.m,Cc.a1);table.innerHTML='';row('Blossom (exact)',best.c,true);};
      document.getElementById('blGreedy').onclick=()=>{const g=greedy();draw(g.m,Cc.paper);table.innerHTML='';row('greedy',g.c);row('Blossom (exact)',best.c,true);};
      document.getElementById('blRandom').onclick=()=>{const r=randomM();draw(r.m,Cc.bad);table.innerHTML='';row('random',r.c);row('Blossom (exact)',best.c,true);};
      inited=true;}
    gen();}};
})();
