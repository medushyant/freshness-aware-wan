/* Live swarm simulation — drones patrolling and tracking moving targets.
   Pure canvas; the motion classes mirror wan/targets.py (static / constant-
   velocity / coordinated-turn). Illustrative of the scenario, not a data replay.
   UI v3: quad-rotor drone glyphs with spinning rotors and heading, radial
   patrol disks, sweeping sensing cones, flowing comm links, target trails. */
window.Sim = (function(){
  const Col={ink:'#f2eefc',a1:'#3ce0c3',a2:'#7c93ff',a3:'#b78aff',mut:'#7a7099',amber:'#ffb26b'};
  let cv,x,W,H,raf=null,running=true,drones=[],targets=[],kind='dynamic',spd=5,N=6,t=0;

  function size(){const r=cv.parentElement.getBoundingClientRect();
    W=cv.width=Math.max(420,r.width-32);H=cv.height=460;}
  function rand(a,b){return a+Math.random()*(b-a);}

  function build(){
    drones=[];targets=[];
    for(let i=0;i<N;i++){
      const c={x:rand(.15*W,.85*W),y:rand(.18*H,.82*H)},R=rand(60,86);
      const ang=rand(0,7),sp=(kind==='static')?0:rand(.4,.8)*spd;
      const tg={x:c.x+rand(-20,20),y:c.y+rand(-20,20),
        vx:Math.cos(ang)*sp,vy:Math.sin(ang)*sp,mode:0,k:0,trail:[]};
      targets.push(tg);
      drones.push({c:{...c},R,x:c.x,y:c.y,id:i,tg,head:rand(0,7)});
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
    tg.trail.push({x:tg.x,y:tg.y});if(tg.trail.length>26)tg.trail.shift();
  }

  function drone(d){
    const s=1.15, ang=d.head;
    x.save();x.translate(d.x,d.y);x.rotate(ang);
    // soft glow
    const gl=x.createRadialGradient(0,0,2,0,0,22);
    gl.addColorStop(0,'rgba(124,147,255,.35)');gl.addColorStop(1,'rgba(124,147,255,0)');
    x.fillStyle=gl;x.beginPath();x.arc(0,0,22,0,7);x.fill();
    // arms (X frame)
    x.strokeStyle='rgba(210,220,255,.85)';x.lineWidth=2.4*s;x.lineCap='round';
    for(const a of [Math.PI/4,3*Math.PI/4,5*Math.PI/4,7*Math.PI/4]){
      x.beginPath();x.moveTo(0,0);x.lineTo(Math.cos(a)*9.5*s,Math.sin(a)*9.5*s);x.stroke();}
    // rotors: spinning arcs
    const rs=t*.55+d.id;
    for(const a of [Math.PI/4,3*Math.PI/4,5*Math.PI/4,7*Math.PI/4]){
      const rx=Math.cos(a)*9.5*s,ry=Math.sin(a)*9.5*s;
      x.strokeStyle='rgba(60,224,195,.9)';x.lineWidth=1.6;
      x.beginPath();x.arc(rx,ry,4.6*s,rs,rs+4.4);x.stroke();
      x.fillStyle='rgba(15,12,26,.9)';x.beginPath();x.arc(rx,ry,1.8,0,7);x.fill();}
    // body
    const bg=x.createLinearGradient(-6,-6,7,8);
    bg.addColorStop(0,'#a9b8ff');bg.addColorStop(1,'#5f76e8');
    x.fillStyle=bg;x.strokeStyle='rgba(15,12,26,.85)';x.lineWidth=1.4;
    x.beginPath();x.roundRect(-6.4*s,-5*s,12.8*s,10*s,3.5);x.fill();x.stroke();
    // nose (heading)
    x.fillStyle=Col.amber;x.beginPath();x.moveTo(7.6*s,0);x.lineTo(3.4*s,-2.8);x.lineTo(3.4*s,2.8);x.closePath();x.fill();
    x.restore();
    // id badge
    x.fillStyle='rgba(15,12,26,.85)';x.beginPath();x.arc(d.x+11,d.y-11,7,0,7);x.fill();
    x.strokeStyle='rgba(124,147,255,.6)';x.lineWidth=1;x.stroke();
    x.fillStyle=Col.ink;x.font='700 8.5px JetBrains Mono';x.textAlign='center';x.textBaseline='middle';
    x.fillText(d.id+1,d.x+11,d.y-10.5);
  }

  function frame(){
    t++;x.clearRect(0,0,W,H);
    // subtle grid floor
    x.strokeStyle='rgba(168,160,220,.05)';x.lineWidth=1;
    for(let gx=0;gx<W;gx+=46){x.beginPath();x.moveTo(gx,0);x.lineTo(gx,H);x.stroke();}
    for(let gy=0;gy<H;gy+=46){x.beginPath();x.moveTo(0,gy);x.lineTo(W,gy);x.stroke();}
    // move
    for(const d of drones){const tg=d.tg;stepTarget(tg);
      d.c.x+=(tg.x-d.c.x)*.012;d.c.y+=(tg.y-d.c.y)*.012;
      let dx=tg.x-d.x,dy=tg.y-d.y;d.x+=dx*.05;d.y+=dy*.05;
      if(Math.hypot(dx,dy)>1.5)d.head+= (Math.atan2(dy,dx)-d.head)*.08;
      const ox=d.x-d.c.x,oy=d.y-d.c.y,dist=Math.hypot(ox,oy);
      if(dist>d.R){d.x=d.c.x+ox/dist*d.R;d.y=d.c.y+oy/dist*d.R;}}
    // patrol disks (soft radial)
    for(const d of drones){
      const rg=x.createRadialGradient(d.c.x,d.c.y,d.R*.4,d.c.x,d.c.y,d.R);
      rg.addColorStop(0,'rgba(124,147,255,.02)');rg.addColorStop(1,'rgba(124,147,255,.08)');
      x.fillStyle=rg;x.beginPath();x.arc(d.c.x,d.c.y,d.R,0,7);x.fill();
      x.setLineDash([5,7]);x.lineDashOffset=-t*.25;
      x.strokeStyle='rgba(124,147,255,.32)';x.lineWidth=1.2;x.stroke();x.setLineDash([]);
      x.fillStyle='rgba(124,147,255,.5)';x.beginPath();x.arc(d.c.x,d.c.y,2,0,7);x.fill();}
    // comm links: flowing dashes + moving packet dots
    for(let i=0;i<drones.length;i++)for(let j=i+1;j<drones.length;j++){
      const a=drones[i],b=drones[j],dd=Math.hypot(a.x-b.x,a.y-b.y);
      if(dd<170){const al=.55*(1-dd/170);
        x.setLineDash([2,9]);x.lineDashOffset=-t*.9;
        x.strokeStyle=`rgba(168,180,255,${al})`;x.lineWidth=1.6;
        x.beginPath();x.moveTo(a.x,a.y);x.lineTo(b.x,b.y);x.stroke();x.setLineDash([]);
        const ph=((t*.012)+(i*.37+j*.19))%1;
        const px=a.x+(b.x-a.x)*ph,py=a.y+(b.y-a.y)*ph;
        x.fillStyle=`rgba(60,224,195,${al+.25})`;x.beginPath();x.arc(px,py,2.2,0,7);x.fill();}}
    // target trails + sensing cones + targets
    for(const tg of targets){
      for(let k=1;k<tg.trail.length;k++){const p0=tg.trail[k-1],p1=tg.trail[k];
        x.strokeStyle=`rgba(183,138,255,${k/tg.trail.length*.3})`;x.lineWidth=1.6;
        x.beginPath();x.moveTo(p0.x,p0.y);x.lineTo(p1.x,p1.y);x.stroke();}}
    for(const d of drones){const tg=d.tg,dd=Math.hypot(d.x-tg.x,d.y-tg.y);
      const ang=Math.atan2(tg.y-d.y,tg.x-d.x),spread=.24;
      const grad=x.createLinearGradient(d.x,d.y,tg.x,tg.y);
      const hot=dd<70;
      grad.addColorStop(0,hot?'rgba(60,224,195,.30)':'rgba(60,224,195,.10)');
      grad.addColorStop(1,'rgba(60,224,195,0)');
      x.fillStyle=grad;x.beginPath();x.moveTo(d.x,d.y);
      x.arc(d.x,d.y,Math.min(dd,140),ang-spread,ang+spread);x.closePath();x.fill();
      if(hot){x.setLineDash([3,5]);x.lineDashOffset=-t*.6;
        x.strokeStyle='rgba(60,224,195,.75)';x.lineWidth=1.4;
        x.beginPath();x.moveTo(d.x,d.y);x.lineTo(tg.x,tg.y);x.stroke();x.setLineDash([]);}}
    for(const tg of targets){const pulse=3+Math.sin(t*.1)*1.4;
      const gl=x.createRadialGradient(tg.x,tg.y,1,tg.x,tg.y,14);
      gl.addColorStop(0,'rgba(183,138,255,.5)');gl.addColorStop(1,'rgba(183,138,255,0)');
      x.fillStyle=gl;x.beginPath();x.arc(tg.x,tg.y,14,0,7);x.fill();
      x.save();x.translate(tg.x,tg.y);x.rotate(Math.PI/4+t*.01);
      x.fillStyle=Col.a3;x.strokeStyle='rgba(15,12,26,.8)';x.lineWidth=1.2;
      x.beginPath();x.roundRect(-4.6,-4.6,9.2,9.2,2);x.fill();x.stroke();x.restore();
      x.beginPath();x.arc(tg.x,tg.y,9+pulse,0,7);x.strokeStyle='rgba(183,138,255,.4)';x.lineWidth=1.2;x.stroke();}
    // drones on top
    for(const d of drones)drone(d);
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
