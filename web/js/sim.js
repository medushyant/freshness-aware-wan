/* Live swarm simulation — drones patrolling and tracking moving targets.
   Pure canvas; the motion classes mirror wan/targets.py (static / constant-
   velocity / coordinated-turn). Illustrative of the scenario, not a data replay. */
window.Sim = (function(){
  const Col={ink:'#eaf0fb',a1:'#38e0c6',a2:'#5b8cff',a3:'#b07cff',mut:'#64718c'};
  let cv,x,W,H,raf=null,running=true,drones=[],targets=[],kind='dynamic',spd=5,N=6,t=0;

  function size(){const r=cv.parentElement.getBoundingClientRect();
    W=cv.width=Math.max(420,r.width-32);H=cv.height=420;}
  function rand(a,b){return a+Math.random()*(b-a);}

  function build(){
    drones=[];targets=[];
    for(let i=0;i<N;i++){
      const c={x:rand(.15*W,.85*W),y:rand(.18*H,.82*H)},R=rand(58,82);
      const ang=rand(0,7),sp=(kind==='static')?0:rand(.4,.8)*spd;
      const tg={x:c.x+rand(-20,20),y:c.y+rand(-20,20),
        vx:Math.cos(ang)*sp,vy:Math.sin(ang)*sp,mode:0,k:0};
      targets.push(tg);
      drones.push({c:{...c},R,x:c.x,y:c.y,id:i,tg});
    }
  }

  function stepTarget(tg){
    if(kind==='static')return;
    tg.k++;
    if(kind==='time_varying' && tg.k%26===0){const a=0.6;
      const c=Math.cos(a),s=Math.sin(a),vx=tg.vx,vy=tg.vy;tg.vx=c*vx-s*vy;tg.vy=s*vx+c*vy;}
    tg.vx+=rand(-.04,.04);tg.vy+=rand(-.04,.04);
    tg.x+=tg.vx;tg.y+=tg.vy;
    if(tg.x<14||tg.x>W-14)tg.vx*=-1;if(tg.y<14||tg.y>H-14)tg.vy*=-1;
    tg.x=Math.max(14,Math.min(W-14,tg.x));tg.y=Math.max(14,Math.min(H-14,tg.y));
  }

  function frame(){
    t++;x.clearRect(0,0,W,H);
    // re-anchor + move
    for(const d of drones){const tg=d.tg;stepTarget(tg);
      d.c.x+=(tg.x-d.c.x)*.012;d.c.y+=(tg.y-d.c.y)*.012;        // patrol centre tracks target
      let dx=tg.x-d.x,dy=tg.y-d.y;d.x+=dx*.05;d.y+=dy*.05;        // drone eases toward target
      const ox=d.x-d.c.x,oy=d.y-d.c.y,dist=Math.hypot(ox,oy);     // clamp inside patrol disk
      if(dist>d.R){d.x=d.c.x+ox/dist*d.R;d.y=d.c.y+oy/dist*d.R;}
    }
    // patrol disks
    for(const d of drones){x.beginPath();x.arc(d.c.x,d.c.y,d.R,0,7);
      x.strokeStyle='rgba(91,140,255,.16)';x.lineWidth=1;x.stroke();
      x.fillStyle='rgba(91,140,255,.03)';x.fill();}
    // comm links (drones within range)
    for(let i=0;i<drones.length;i++)for(let j=i+1;j<drones.length;j++){
      const a=drones[i],b=drones[j],dd=Math.hypot(a.x-b.x,a.y-b.y);
      if(dd<150){x.strokeStyle=`rgba(120,150,210,${.4*(1-dd/150)})`;x.lineWidth=1;
        x.beginPath();x.moveTo(a.x,a.y);x.lineTo(b.x,b.y);x.stroke();}}
    // sensing + targets
    for(const d of drones){const tg=d.tg,dd=Math.hypot(d.x-tg.x,d.y-tg.y);
      x.setLineDash([4,4]);x.strokeStyle=dd<60?'rgba(56,224,198,.7)':'rgba(56,224,198,.25)';
      x.beginPath();x.moveTo(d.x,d.y);x.lineTo(tg.x,tg.y);x.stroke();x.setLineDash([]);}
    for(const tg of targets){const pulse=3+Math.sin(t*.1)*1.2;
      x.fillStyle=Col.a3;x.save();x.translate(tg.x,tg.y);x.rotate(Math.PI/4);
      x.fillRect(-4,-4,8,8);x.restore();
      x.beginPath();x.arc(tg.x,tg.y,8+pulse,0,7);x.strokeStyle='rgba(176,124,255,.35)';x.lineWidth=1;x.stroke();}
    // drones (glow + glyph)
    for(const d of drones){x.beginPath();x.arc(d.x,d.y,11,0,7);
      x.fillStyle='rgba(91,140,255,.16)';x.fill();
      x.beginPath();x.arc(d.x,d.y,6,0,7);x.fillStyle=Col.a2;x.fill();
      x.fillStyle=Col.ink;x.font='9px JetBrains Mono';x.textAlign='center';x.textBaseline='middle';
      x.fillText(d.id+1,d.x,d.y);}
    if(running)raf=requestAnimationFrame(frame);
  }

  function start(){if(raf)cancelAnimationFrame(raf);running=true;frame();}
  let inited=false;
  return {
    init(){
      cv=document.getElementById('swarmCanvas');if(!cv)return;x=cv.getContext('2d');
      if(!inited){
        size();addEventListener('resize',()=>{if(document.getElementById('tab-swarm').classList.contains('active')){size();build();}});
        const K=document.getElementById('swarmKind'),Nn=document.getElementById('swarmN'),
          Nv=document.getElementById('swarmNv'),S=document.getElementById('swarmSpd'),
          P=document.getElementById('swarmPlay'),R=document.getElementById('swarmReset');
        K.onchange=e=>{kind=e.target.value;build();};
        Nn.oninput=e=>{N=+e.target.value;Nv.textContent=N;build();};
        S.oninput=e=>{spd=+e.target.value;build();};
        P.onclick=()=>{running=!running;P.textContent=running?'⏸ pause':'▶ play';if(running)start();};
        R.onclick=()=>{build();};
        inited=true;
      }
      size();build();start();
    },
    stop(){running=false;if(raf)cancelAnimationFrame(raf);}
  };
})();
