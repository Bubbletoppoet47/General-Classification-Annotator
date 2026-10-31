"""The one web page the tablet gets (kept as a string so the packaged program needs no extra files)."""

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no,viewport-fit=cover">
<meta name="apple-mobile-web-app-capable" content="yes">
<title>Annotator</title>
<style>
:root{--bg:#16181d;--bar:#20242c;--fg:#eceef2;--mut:#98a0ad;--acc:#4ea1ff;--ok:#3ecf8e;--bad:#ff6b6b}
*{box-sizing:border-box;-webkit-user-select:none;user-select:none;-webkit-touch-callout:none}
html,body{margin:0;height:100%;background:var(--bg);color:var(--fg);font:16px system-ui,-apple-system,Segoe UI,sans-serif;overscroll-behavior:none}
#login{display:flex;flex-direction:column;gap:12px;max-width:340px;margin:12vh auto;padding:0 20px}
#login h1{font-size:22px;margin:0 0 4px}
input{font:inherit;padding:12px;border-radius:10px;border:1px solid #3a404c;background:#0f1115;color:var(--fg);-webkit-user-select:text;user-select:text}
button{font:inherit;color:var(--fg);background:#2d333e;border:0;border-radius:10px;padding:10px 14px;min-height:44px;min-width:44px}
button:active{background:#3a4250}button.p{background:var(--acc);color:#fff}button.ok{background:var(--ok);color:#06241a}
button:disabled{opacity:.4}
#err{color:var(--bad);min-height:1.2em}
#app{display:none;height:100%;flex-direction:column}
.bar{display:flex;align-items:center;gap:6px;padding:6px 8px;background:var(--bar);flex-wrap:nowrap;overflow-x:auto}
#classes button{border:3px solid transparent;font-weight:600}
#classes button.on{border-color:#fff}
#cv{flex:1;min-height:0;position:relative;touch-action:none;background:#000}
canvas{position:absolute;inset:0;width:100%;height:100%;touch-action:none}
#info{color:var(--mut);padding:0 6px;white-space:nowrap}
#toast{position:fixed;left:50%;bottom:24px;transform:translateX(-50%);background:#000c;padding:10px 16px;border-radius:10px;display:none}
.sp{flex:1}
</style></head><body>
<div id="login">
  <h1>Annotator</h1>
  <input id="name" placeholder="Your name" autocomplete="off" autocapitalize="words">
  <input id="code" placeholder="Code from the computer" inputmode="numeric" autocomplete="off">
  <button class="p" id="go">Open</button>
  <div id="err"></div>
</div>
<div id="app">
  <div class="bar" id="classes"></div>
  <div id="cv"><canvas id="c"></canvas></div>
  <div class="bar">
    <button id="prev" title="Previous">&#9664;</button>
    <span id="info"></span>
    <button id="next" title="Next">&#9654;</button>
    <button id="undo" title="Undo">&#8630; Undo</button>
    <button id="del" title="Delete box">&#128465; Delete</button>
    <button id="ai" title="AI boxes">AI</button>
    <button id="fit" title="Fit to screen">Fit</button>
    <span class="sp"></span>
    <button id="filter" title="Show">To do</button>
    <button id="pub" title="Publish on the computer">Publish</button>
    <button class="ok" id="conf" title="Confirm and go to next">&#10003; Confirm</button>
  </div>
</div>
<div id="toast"></div>
<script>
const $=id=>document.getElementById(id);
const COLORS=["#ff5d5d","#4ea1ff","#3ecf8e","#ffc94d","#c58bff","#ff8f4d","#4de0e0","#ff7ac8","#a3d95b","#9aa5ff"];
let S={classes:[],cls:0,items:[],idx:0,filter:"todo",boxes:[],undo:[],sel:-1,img:null,iw:1,ih:1,s:1,ox:0,oy:0,name:"",busy:false};
const cv=$("c"),ctx=cv.getContext("2d");
function toast(t,ms=2200){const e=$("toast");e.textContent=t;e.style.display="block";clearTimeout(toast.t);toast.t=setTimeout(()=>e.style.display="none",ms)}
async function api(path,body){
  const r=await fetch(path,body===undefined?{}:{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
  if(r.status===401){showLogin("");throw new Error("login")}
  const ct=r.headers.get("content-type")||"";
  const d=ct.includes("json")?await r.json():r;
  if(!r.ok)throw new Error((d&&d.error)||"Error");
  return d}
function showLogin(msg){$("app").style.display="none";$("login").style.display="flex";$("err").textContent=msg||""}
const q=new URLSearchParams(location.search);
$("code").value=q.get("c")||"";
try{$("name").value=localStorage.getItem("gca_name")||""}catch(e){}
$("go").onclick=async()=>{
  try{await api("/api/login",{name:$("name").value,code:$("code").value});
    try{localStorage.setItem("gca_name",$("name").value)}catch(e){}
    history.replaceState(null,"",location.pathname);start()}
  catch(e){$("err").textContent=e.message}};
async function start(){
  const st=await api("/api/state");S.name=st.me;S.classes=st.classes;S.hasAI=st.ai;
  $("login").style.display="none";$("app").style.display="flex";
  buildClasses();resize();await loadList(true)}
function buildClasses(){
  const el=$("classes");el.innerHTML="";
  S.classes.forEach((n,i)=>{const b=document.createElement("button");b.textContent=n;b.style.background=COLORS[i%COLORS.length]+"55";
    b.className=i===S.cls?"on":"";b.onclick=()=>{S.cls=i;if(S.sel>=0){push();S.boxes[S.sel][0]=i;save()}buildClasses();draw()};el.appendChild(b)})}
async function loadList(first){
  const d=await api("/api/list?f="+S.filter);S.items=d.items;
  if(!S.items.length){S.idx=0;S.img=null;S.boxes=[];$("info").textContent=S.filter==="todo"?"All done!":"No images";draw();return}
  if(S.idx>=S.items.length)S.idx=S.items.length-1;await openAt(S.idx)}
async function openAt(i){
  S.idx=i;const id=S.items[i][0];S.sel=-1;S.undo=[];
  const [b,im]=await Promise.all([api("/api/boxes/"+id),loadImg("/api/img/"+id)]);
  S.boxes=b.boxes.map(x=>x.slice());S.img=im;S.iw=im.naturalWidth;S.ih=im.naturalHeight;fit();
  $("info").textContent=(i+1)+" / "+S.items.length+(b.annotated?"  ✓":(b.ai?"  AI":""));draw()}
function loadImg(u){return new Promise((res,rej)=>{const im=new Image();im.onload=()=>res(im);im.onerror=()=>rej(new Error("Image failed"));im.src=u})}
function resize(){const r=$("cv").getBoundingClientRect(),d=window.devicePixelRatio||1;cv.width=r.width*d;cv.height=r.height*d;S.d=d;S.cw=r.width;S.ch=r.height;draw()}
function fit(){S.s=S.s0=Math.min(S.cw/S.iw,S.ch/S.ih);S.ox=(S.cw-S.iw*S.s)/2;S.oy=(S.ch-S.ih*S.s)/2}
addEventListener("resize",()=>{resize();if(S.img)fit();draw()});
const toImg=(x,y)=>[(x-S.ox)/S.s/S.iw,(y-S.oy)/S.s/S.ih];
const toScr=(u,v)=>[u*S.iw*S.s+S.ox,v*S.ih*S.s+S.oy];
function rect(b){const [x1,y1]=toScr(b[1]-b[3]/2,b[2]-b[4]/2),[x2,y2]=toScr(b[1]+b[3]/2,b[2]+b[4]/2);return[x1,y1,x2-x1,y2-y1]}
function draw(){
  ctx.setTransform(S.d,0,0,S.d,0,0);ctx.clearRect(0,0,S.cw,S.ch);
  if(!S.img){ctx.fillStyle="#98a0ad";ctx.font="18px system-ui";ctx.textAlign="center";ctx.fillText($("info").textContent||"Loading...",S.cw/2,S.ch/2);return}
  ctx.drawImage(S.img,S.ox,S.oy,S.iw*S.s,S.ih*S.s);
  S.boxes.concat(S.drag&&S.drag.mode==="new"?[S.drag.box]:[]).forEach((b,i)=>{
    const [x,y,w,h]=rect(b),c=COLORS[b[0]%COLORS.length],sel=i===S.sel;
    ctx.lineWidth=sel?4:2.5;ctx.strokeStyle=c;ctx.strokeRect(x,y,w,h);
    ctx.fillStyle=c+(sel?"40":"22");ctx.fillRect(x,y,w,h);
    ctx.font="600 14px system-ui";ctx.textAlign="left";const t=S.classes[b[0]]||"?";
    const tw=ctx.measureText(t).width+10;ctx.fillStyle=c;ctx.fillRect(x,Math.max(0,y-20),tw,20);ctx.fillStyle="#000";ctx.fillText(t,x+5,Math.max(14,y-5));
    if(sel){ctx.fillStyle="#fff";[[x,y],[x+w,y],[x,y+h],[x+w,y+h]].forEach(p=>ctx.fillRect(p[0]-7,p[1]-7,14,14))}})}
function hit(x,y){for(let i=S.boxes.length-1;i>=0;i--){const [bx,by,w,h]=rect(S.boxes[i]);if(x>=bx&&x<=bx+w&&y>=by&&y<=by+h)return i}return -1}
function corner(x,y){if(S.sel<0)return null;const [bx,by,w,h]=rect(S.boxes[S.sel]);
  for(const [cx,cy,k] of [[bx,by,"tl"],[bx+w,by,"tr"],[bx,by+h,"bl"],[bx+w,by+h,"br"]])if(Math.abs(x-cx)<22&&Math.abs(y-cy)<22)return k;return null}
function push(){S.undo.push(JSON.stringify(S.boxes));if(S.undo.length>50)S.undo.shift()}
let saveT=null;
function save(){clearTimeout(saveT);saveT=setTimeout(doSave,150)}
async function doSave(){
  if(!S.items.length)return;const id=S.items[S.idx][0];
  try{const d=await api("/api/boxes/"+id,{boxes:S.boxes});S.boxes=d.boxes.map(x=>x.slice());S.items[S.idx][1]=2;
    $("info").textContent=(S.idx+1)+" / "+S.items.length+"  ✓";draw()}catch(e){toast(e.message)}}
// ---- pointers: pen (or mouse) draws and edits; touch pans and zooms; never both at once
const P=new Map();let penDown=false,pinch=null;
cv.addEventListener("pointerdown",e=>{
  try{cv.setPointerCapture(e.pointerId)}catch(_){}const r=cv.getBoundingClientRect(),x=e.clientX-r.left,y=e.clientY-r.top;
  const pen=e.pointerType!=="touch";
  if(pen){penDown=true;penStart(x,y)}
  else{if(penDown)return;P.set(e.pointerId,{x,y,sx:x,sy:y,t:Date.now()});pinch=null}
  e.preventDefault()});
cv.addEventListener("pointermove",e=>{
  const r=cv.getBoundingClientRect(),x=e.clientX-r.left,y=e.clientY-r.top;
  if(e.pointerType!=="touch"){if(penDown)penMove(x,y);return}
  const p=P.get(e.pointerId);if(!p||penDown)return;
  const px=p.x,py=p.y;p.x=x;p.y=y;
  if(P.size===1){S.ox+=x-px;S.oy+=y-py}
  else if(P.size>=2){const [a,b]=[...P.values()],dist=Math.hypot(a.x-b.x,a.y-b.y),mx=(a.x+b.x)/2,my=(a.y+b.y)/2;
    if(pinch){const k=dist/pinch.dist,ns=Math.min(Math.max(S.s*k,S.s0*0.5),S.s0*12);
      const f=ns/S.s;S.ox=mx-(mx-S.ox)*f+(mx-pinch.mx);S.oy=my-(my-S.oy)*f+(my-pinch.my);S.s=ns}
    pinch={dist,mx,my}}
  draw();e.preventDefault()});
function endPtr(e){
  const r=cv.getBoundingClientRect(),x=e.clientX-r.left,y=e.clientY-r.top;
  if(e.pointerType!=="touch"){if(penDown){penEnd(x,y)}penDown=false;return}
  const p=P.get(e.pointerId);
  if(p&&P.size===1&&!pinch&&Math.hypot(x-p.sx,y-p.sy)<8&&Date.now()-p.t<500){S.sel=hit(x,y);draw()}   // a finger tap selects
  P.delete(e.pointerId);pinch=null}
cv.addEventListener("pointerup",endPtr);cv.addEventListener("pointercancel",e=>{P.delete(e.pointerId);penDown=false;S.drag=null;draw()});
cv.addEventListener("wheel",e=>{e.preventDefault();const r=cv.getBoundingClientRect(),x=e.clientX-r.left,y=e.clientY-r.top,
  ns=Math.min(Math.max(S.s*(e.deltaY<0?1.15:1/1.15),S.s0*0.5),S.s0*12),f=ns/S.s;S.ox=x-(x-S.ox)*f;S.oy=y-(y-S.oy)*f;S.s=ns;draw()},{passive:false});
function penStart(x,y){
  if(!S.img)return;S.moved=false;
  const k=corner(x,y);
  if(k){S.drag={mode:"resize",k,start:JSON.stringify(S.boxes)};return}
  const h=hit(x,y);
  if(h>=0&&h===S.sel){S.drag={mode:"move",px:x,py:y,start:JSON.stringify(S.boxes)};return}
  S.drag={mode:"new",x0:x,y0:y,box:[S.cls,0,0,0,0],tapHit:h}}
function penMove(x,y){
  const d=S.drag;if(!d)return;S.moved=true;
  if(d.mode==="new"){const [u0,v0]=toImg(d.x0,d.y0),[u1,v1]=toImg(x,y);
    d.box=[S.cls,(u0+u1)/2,(v0+v1)/2,Math.abs(u1-u0),Math.abs(v1-v0)]}
  else if(d.mode==="move"){const du=(x-d.px)/S.s/S.iw,dv=(y-d.py)/S.s/S.ih,b=S.boxes[S.sel];b[1]+=du;b[2]+=dv;d.px=x;d.py=y}
  else{const b=S.boxes[S.sel],[u,v]=toImg(x,y);let x1=b[1]-b[3]/2,x2=b[1]+b[3]/2,y1=b[2]-b[4]/2,y2=b[2]+b[4]/2;
    if(d.k[1]==="l")x1=u;else x2=u;if(d.k[0]==="t")y1=v;else y2=v;
    b[1]=(x1+x2)/2;b[2]=(y1+y2)/2;b[3]=Math.abs(x2-x1);b[4]=Math.abs(y2-y1)}
  draw()}
function penEnd(x,y){
  const d=S.drag;S.drag=null;if(!d)return;
  if(d.mode==="new"){
    const b=d.box;
    if(!S.moved||b[3]*S.iw*S.s<8||b[4]*S.ih*S.s<8){S.sel=d.tapHit;draw();return}      // a tap selects, it does not draw
    S.undo.push(JSON.stringify(S.boxes));S.boxes.push(b);S.sel=S.boxes.length-1;save()}
  else{if(S.moved){S.undo.push(d.start);save()}}
  draw()}
// ---- buttons
$("prev").onclick=()=>step(-1);$("next").onclick=()=>step(1);
async function step(n){const i=S.idx+n;if(i<0||i>=S.items.length)return toast(n>0?"That's the last one":"That's the first one");await openAt(i)}
$("undo").onclick=()=>{if(!S.undo.length)return toast("Nothing to undo");S.boxes=JSON.parse(S.undo.pop());S.sel=-1;save();draw()};
$("del").onclick=()=>{if(S.sel<0)return toast("Tap a box first");push();S.boxes.splice(S.sel,1);S.sel=-1;save();draw()};
$("fit").onclick=()=>{fit();draw()};
$("ai").onclick=async()=>{
  if(!S.items.length)return;$("ai").disabled=true;toast("AI is working...",8000);
  try{const d=await api("/api/ai/"+S.items[S.idx][0],{});if(!d.ai){toast("Already confirmed, so AI was skipped.",3000)}else{push();S.boxes=d.boxes.map(x=>x.slice());S.sel=-1;toast(d.boxes.length+" box(es)")}draw()}
  catch(e){toast(e.message,4000)}$("ai").disabled=false};
$("filter").onclick=async()=>{S.filter=S.filter==="todo"?"all":"todo";$("filter").textContent=S.filter==="todo"?"To do":"All";S.idx=0;await loadList()};
$("pub").onclick=async()=>{try{const d=await api("/api/publish",{});toast(d.msg,3000)}catch(e){toast(e.message,4000)}};
$("conf").onclick=async()=>{
  if(!S.items.length)return;clearTimeout(saveT);await doSave();
  if(S.filter==="todo"){S.items.splice(S.idx,1);if(!S.items.length)return loadList();if(S.idx>=S.items.length)S.idx=S.items.length-1;await openAt(S.idx)}
  else await step(1)};
new ResizeObserver(()=>{resize();if(S.img){fit();draw()}}).observe($("cv"));
api("/api/state").then(()=>start()).catch(()=>{});
</script></body></html>
"""
