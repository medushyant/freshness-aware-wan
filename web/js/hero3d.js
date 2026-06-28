/* 3D swarm network hero (Three.js). Glowing agents + comm links in 3D,
   slowly orbiting, drag to rotate. Decorative; the science is in the tabs. */
window.Hero3d=(function(){
  let scn,cam,rnd,grp,raf,host,W,H,drag=false,px=0,py=0,rx=.5,ry=0,vy=.0016,inited=false;
  const N=16,LINK=2.3;
  function init(){
    host=document.getElementById('hero3d');if(!host||!window.THREE||inited)return;inited=true;
    W=host.clientWidth;H=host.clientHeight||340;
    scn=new THREE.Scene();
    cam=new THREE.PerspectiveCamera(55,W/H,.1,100);cam.position.z=8;
    rnd=new THREE.WebGLRenderer({antialias:true,alpha:true});
    rnd.setSize(W,H);rnd.setPixelRatio(Math.min(devicePixelRatio,2));host.appendChild(rnd.domElement);
    grp=new THREE.Group();scn.add(grp);
    const pts=[];
    for(let i=0;i<N;i++){
      const v=new THREE.Vector3((Math.random()-.5)*6,(Math.random()-.5)*4.4,(Math.random()-.5)*6);
      pts.push(v);
      const fast=i<3;
      const m=new THREE.Mesh(new THREE.SphereGeometry(fast?.18:.13,18,18),
        new THREE.MeshBasicMaterial({color:fast?0xb07cff:0x38e0c6}));
      m.position.copy(v);grp.add(m);
      const halo=new THREE.Mesh(new THREE.SphereGeometry(fast?.34:.26,16,16),
        new THREE.MeshBasicMaterial({color:fast?0xb07cff:0x5b8cff,transparent:true,opacity:.16,blending:THREE.AdditiveBlending}));
      halo.position.copy(v);grp.add(halo);
    }
    const lp=[],lc=[];
    for(let i=0;i<N;i++)for(let j=i+1;j<N;j++){const d=pts[i].distanceTo(pts[j]);
      if(d<LINK){lp.push(pts[i].x,pts[i].y,pts[i].z,pts[j].x,pts[j].y,pts[j].z);
        const a=1-d/LINK;for(let k=0;k<2;k++)lc.push(.36*a,.55*a,1*a);}}
    const g=new THREE.BufferGeometry();
    g.setAttribute('position',new THREE.Float32BufferAttribute(lp,3));
    g.setAttribute('color',new THREE.Float32BufferAttribute(lc,3));
    grp.add(new THREE.LineSegments(g,new THREE.LineBasicMaterial({vertexColors:true,transparent:true,opacity:.6})));
    // interaction
    const el=rnd.domElement;
    el.addEventListener('pointerdown',e=>{drag=true;px=e.clientX;py=e.clientY;});
    addEventListener('pointerup',()=>drag=false);
    addEventListener('pointermove',e=>{if(!drag)return;ry+=(e.clientX-px)*.005;rx+=(e.clientY-py)*.005;px=e.clientX;py=e.clientY;});
    addEventListener('resize',()=>{if(!host.clientWidth)return;W=host.clientWidth;H=host.clientHeight||340;
      cam.aspect=W/H;cam.updateProjectionMatrix();rnd.setSize(W,H);});
    loop();
  }
  function loop(){raf=requestAnimationFrame(loop);if(!drag)ry+=vy;
    grp.rotation.y=ry;grp.rotation.x=rx*.6;rnd.render(scn,cam);}
  return{init};
})();
