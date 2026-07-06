"""Review and correct the model's 12 court keypoints, save as YOLO-pose labels.

Loads predictions written by scripts/roboflow_infer.py (labels/_keypoints.json), draws the
12 points on each image, and lets you DRAG each one to its true spot. Adjust, then Next —
the corrected points overwrite the YOLO-pose .txt so a model can be retrained on them.

Point order (model class ids, fixed):
  0 bg baseline left   1 bg baseline mid    2 bg baseline right
  3 bg kitchen right   4 bg kitchen mid     5 bg kitchen left
  6 fg kitchen left    7 fg kitchen mid     8 fg kitchen right
  9 fg baseline right 10 fg baseline mid   11 fg baseline left

Drag a dot to move it. Click a dot to select it; then:
  V = visible · X = not-visible (parked, excluded from label & bbox).
A missing prediction (v=0) starts parked at the left rail — drag it onto the court to use it.

Run:  python scripts/keypoint_review_app.py --dir dataset   ->  http://localhost:8079
Writes <dir>/labels/*.txt (YOLO-pose) and updates labels/_keypoints.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse

REPO = Path(__file__).resolve().parent.parent

KP_NAMES = [
    "bg_baseline_left", "bg_baseline_mid", "bg_baseline_right",
    "bg_kitchen_right", "bg_kitchen_mid", "bg_kitchen_left",
    "fg_kitchen_left", "fg_kitchen_mid", "fg_kitchen_right",
    "fg_baseline_right", "fg_baseline_mid", "fg_baseline_left",
]
N_KP = len(KP_NAMES)
IMG_EXTS = {".jpg", ".jpeg", ".png"}
BBOX_PAD_FRAC = 0.02

app = FastAPI(title="court keypoint review")
STATE: dict = {"dir": REPO / "dataset", "images": [], "labels_dir": None, "json_path": None}


def _images(d: Path) -> list[str]:
    if not d.is_dir():
        raise SystemExit(f"not a folder: {d}")
    imgs = sorted(p.name for p in d.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not imgs:
        raise SystemExit(f"no images in {d}")
    return imgs


def _load_json() -> dict:
    p = STATE["json_path"]
    return json.loads(p.read_text()) if p and p.exists() else {}


def _save_json(d: dict) -> None:
    STATE["json_path"].write_text(json.dumps(d, indent=1))


def _write_yolo(name: str, pts: list[dict], w: int, h: int) -> None:
    placed = [(p["x"], p["y"]) for p in pts if p["v"] > 0]
    txt_path = STATE["labels_dir"] / (Path(name).stem + ".txt")
    if not placed:
        txt_path.write_text("")
        return
    xs, ys = [p[0] for p in placed], [p[1] for p in placed]
    pad = BBOX_PAD_FRAC * max(w, h)
    x0, y0 = max(0, min(xs) - pad), max(0, min(ys) - pad)
    x1, y1 = min(w, max(xs) + pad), min(h, max(ys) + pad)
    cx, cy = (x0 + x1) / 2 / w, (y0 + y1) / 2 / h
    bw, bh = (x1 - x0) / w, (y1 - y0) / h
    fields = ["0", f"{cx:.6f}", f"{cy:.6f}", f"{bw:.6f}", f"{bh:.6f}"]
    for p in pts:
        if p["v"] > 0:
            fields += [f"{p['x']/w:.6f}", f"{p['y']/h:.6f}", str(int(p["v"]))]
        else:
            fields += ["0.000000", "0.000000", "0"]
    txt_path.write_text(" ".join(fields) + "\n")


@app.get("/api/state")
def api_state():
    data = _load_json()
    done = sum(1 for n in STATE["images"] if data.get(n, {}).get("reviewed"))
    has_pred = sum(1 for n in STATE["images"] if n in data)
    return {
        "images": STATE["images"],
        "labels": data,
        "kp_names": KP_NAMES,
        "n_done": done,
        "n_pred": has_pred,
        "total": len(STATE["images"]),
    }


@app.get("/img/{name}")
def img(name: str):
    p = STATE["dir"] / name
    if not p.exists():
        raise HTTPException(404, name)
    return FileResponse(p)


@app.post("/api/save/{name}")
async def save(name: str, req: Request):
    body = await req.json()
    pts = body.get("points", [])
    w, h = int(body["w"]), int(body["h"])
    if len(pts) != N_KP:
        raise HTTPException(400, f"expected {N_KP} points, got {len(pts)}")
    data = _load_json()
    prev = data.get(name, {})
    data[name] = {"points": pts, "w": w, "h": h,
                  "source": prev.get("source", "model"),
                  "reviewed": bool(body.get("reviewed", prev.get("reviewed", False)))}
    _save_json(data)
    _write_yolo(name, pts, w, h)
    return JSONResponse({"ok": True})


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=REPO / "dataset")
    ap.add_argument("--port", type=int, default=8079)
    args = ap.parse_args()
    STATE["dir"] = args.dir
    STATE["images"] = _images(args.dir)
    STATE["labels_dir"] = args.dir / "labels"
    STATE["labels_dir"].mkdir(parents=True, exist_ok=True)
    STATE["json_path"] = args.dir / "labels" / "_keypoints.json"
    if not STATE["json_path"].exists():
        print("  warning: no predictions found "
              f"({STATE['json_path']}). Run scripts/roboflow_infer.py first; "
              "points will start empty.")
    print(f"  {len(STATE['images'])} images in {args.dir}")
    print(f"  labels -> {STATE['labels_dir']}")
    print(f"  open http://localhost:{args.port}")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


HTML = r"""<!doctype html><html><head><meta charset=utf-8><title>court keypoint review</title>
<style>
:root{--bg:#0e1116;--card:#161b22;--edge:#2b3340;--txt:#e6edf3;--dim:#8b97a5;
--green:#3fb950;--amber:#d29922;--red:#f85149;--blue:#388bfd;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--txt);font:14px system-ui;-webkit-user-select:none;user-select:none}
.top{display:flex;align-items:center;gap:12px;padding:8px 12px;background:var(--card);border-bottom:1px solid var(--edge);flex-wrap:wrap}
button{background:#21304b;border:1px solid #3a4f78;color:var(--txt);border-radius:6px;padding:6px 10px;cursor:pointer}
button:hover{background:#2a3c5e}
button:disabled{opacity:.4;cursor:default}
.spacer{flex:1}
.wrap{position:relative;margin:10px auto;width:fit-content;max-width:98vw}
#stage{position:relative;line-height:0}
#img{max-width:98vw;max-height:80vh;display:block}
#ov{position:absolute;inset:0;cursor:crosshair}
.pill{padding:2px 8px;border-radius:10px;background:#1d2530;border:1px solid var(--edge);color:var(--dim)}
.legend{display:flex;gap:12px;font-size:12px;color:var(--dim);align-items:center}
.dot{width:10px;height:10px;border-radius:50%;display:inline-block;margin-right:3px;vertical-align:middle}
.side{position:fixed;right:8px;top:54px;width:188px;background:var(--card);border:1px solid var(--edge);border-radius:8px;padding:8px;font-size:12px;max-height:84vh;overflow:auto}
.side ol{margin:4px 0 0;padding-left:22px}
.side li{padding:1px 0;color:var(--dim);cursor:pointer}
.side li.cur{color:var(--txt);font-weight:700}
.side li.hidden{color:var(--red);text-decoration:line-through}
.side li .v{float:right;font-size:10px}
.hint{font-size:12px;color:var(--dim);padding:0 12px 8px}
.savemark{color:var(--green);font-size:12px;min-width:54px}
kbd{background:#1d2530;border:1px solid var(--edge);border-radius:4px;padding:0 5px;font-size:11px}
</style></head><body>
<div class=top>
  <button id=prev>◀ Prev</button>
  <span class=pill id=counter>—</span>
  <button id=next>Next ▶</button>
  <span class=pill>reviewed <b id=done>0</b>/<b id=total>0</b></span>
  <span class=pill>predicted <b id=pred>0</b></span>
  <span class=spacer></span>
  <span class=legend>
    <span><span class=dot style=background:var(--green)></span>visible</span>
    <span><span class=dot style=background:var(--red)></span>not-visible</span>
    <span>drag to move · <kbd>V</kbd>/<kbd>X</kbd> selected · <kbd>←</kbd><kbd>→</kbd> nav</span>
  </span>
  <button id=reset>Reset to model</button>
  <span class=savemark id=savemark></span>
</div>
<div class=hint id=hint></div>
<div class=wrap><div id=stage><img id=img><canvas id=ov></canvas></div></div>
<div class=side><b>Points</b> <span style="color:var(--dim)">(click to select)</span>
<ol id=list></ol></div>
<script>
let S={imgs:[],idx:0,names:[],pts:[],orig:[],w:0,h:0,scale:1,sel:-1,drag:-1,labels:{}};
const img=document.getElementById('img'),ov=document.getElementById('ov'),ctx=ov.getContext('2d');
const $=id=>document.getElementById(id);
const HIT=11;            // px radius for grabbing a dot (screen space)
const PARK_X=18;         // x for parked (not-visible) dots, screen space

async function boot(){
  const r=await fetch('/api/state').then(r=>r.json());
  S.imgs=r.images;S.names=r.kp_names;S.labels=r.labels||{};
  $('total').textContent=r.total;$('done').textContent=r.n_done;$('pred').textContent=r.n_pred;
  let i=S.imgs.findIndex(n=>!(S.labels[n]&&S.labels[n].reviewed));
  S.idx=i<0?0:i;
  load();
}
function freshPts(){return S.names.map(()=>({x:0,y:0,v:0}));}

function load(){
  const name=S.imgs[S.idx];
  $('counter').textContent=`${S.idx+1} / ${S.imgs.length}  ·  ${name}`;
  S.sel=-1;S.drag=-1;
  const saved=S.labels[name];
  S.pts = saved&&saved.points ? saved.points.map(p=>({x:+p.x||0,y:+p.y||0,v:+p.v||0}))
                              : freshPts();
  S.orig = S.pts.map(p=>({...p}));
  img.onload=()=>{
    S.w=img.naturalWidth;S.h=img.naturalHeight;
    ov.width=img.clientWidth;ov.height=img.clientHeight;
    S.scale=img.clientWidth/S.w;
    draw();renderList();updateHint();
  };
  img.src='/img/'+encodeURIComponent(name)+'?t='+Date.now();
  $('savemark').textContent='';
}

function updateHint(){
  const placed=S.pts.filter(p=>p.v>0).length;
  const src=(S.labels[S.imgs[S.idx]]||{}).source;
  const tag = src? `(${src} prediction)` : '(no prediction — drag points in)';
  $('hint').innerHTML=`${placed}/12 points visible ${tag}. `+
    `Drag a dot to correct it. Parked dots on the left are not-visible — drag onto the court to use.`;
}

// screen position of a point: visible -> its coords; not-visible -> parked rail by index
function screenPos(i){
  const p=S.pts[i];
  if(p.v>0)return{x:p.x*S.scale,y:p.y*S.scale};
  return{x:PARK_X,y:24+i*((ov.height-40)/Math.max(1,S.names.length-1))};
}
function pickPoint(sx,sy){
  let best=-1,bd=HIT*HIT;
  for(let i=0;i<S.names.length;i++){
    const q=screenPos(i);const dx=sx-q.x,dy=sy-q.y;const d=dx*dx+dy*dy;
    if(d<=bd){bd=d;best=i;}
  }
  return best;
}

ov.addEventListener('mousedown',e=>{
  e.preventDefault();
  const r=ov.getBoundingClientRect();
  const sx=e.clientX-r.left, sy=e.clientY-r.top;
  const i=pickPoint(sx,sy);
  if(i>=0){S.sel=i;S.drag=i;
    // grabbing a parked dot makes it visible at the cursor
    if(S.pts[i].v===0){S.pts[i]={x:sx/S.scale,y:sy/S.scale,v:2};}
    draw();renderList();
  } else {S.sel=-1;draw();renderList();}
});
window.addEventListener('mousemove',e=>{
  if(S.drag<0)return;
  const r=ov.getBoundingClientRect();
  let sx=e.clientX-r.left, sy=e.clientY-r.top;
  sx=Math.max(0,Math.min(ov.width,sx));sy=Math.max(0,Math.min(ov.height,sy));
  S.pts[S.drag]={x:sx/S.scale,y:sy/S.scale,v:S.pts[S.drag].v>0?S.pts[S.drag].v:2};
  draw();
});
window.addEventListener('mouseup',()=>{ if(S.drag>=0){S.drag=-1;renderList();autosave();} });

window.addEventListener('keydown',e=>{
  if(e.key==='ArrowRight'){e.preventDefault();save(true).then(()=>go(1));}
  else if(e.key==='ArrowLeft'){e.preventDefault();save(false).then(()=>go(-1));}
  else if((e.key==='v'||e.key==='V')&&S.sel>=0){const p=S.pts[S.sel];if(p.v===0){const q=screenPos(S.sel);S.pts[S.sel]={x:q.x/S.scale,y:q.y/S.scale,v:2};}else{p.v=2;}draw();renderList();autosave();}
  else if((e.key==='x'||e.key==='X')&&S.sel>=0){S.pts[S.sel].v=0;draw();renderList();autosave();}
});
$('reset').onclick=()=>{S.pts=S.orig.map(p=>({...p}));S.sel=-1;draw();renderList();autosave();};

function draw(){
  ctx.clearRect(0,0,ov.width,ov.height);
  const P=i=>screenPos(i);
  const vis=i=>S.pts[i].v>0;
  // court skeleton between visible points
  ctx.lineWidth=1.4;
  function seg(a,b,style){ if(vis(a)&&vis(b)){const pa=P(a),pb=P(b);ctx.strokeStyle=style;
    ctx.beginPath();ctx.moveTo(pa.x,pa.y);ctx.lineTo(pb.x,pb.y);ctx.stroke();}}
  // baselines (bg 0-1-2, fg 9-10-11), sidelines (0-11 left, 2-9 right), kitchen lines (5-4-3, 6-7-8)
  [[0,1],[1,2],[9,10],[10,11],[0,11],[2,9],[5,4],[4,3],[6,7],[7,8],[5,6],[3,8]].forEach(([a,b])=>seg(a,b,'#000'));
  // dots
  S.pts.forEach((p,i)=>{
    const q=P(i);const parked=p.v===0;
    ctx.beginPath();ctx.arc(q.x,q.y,parked?4:6,0,7);
    ctx.fillStyle=parked?'#f85149':(i===S.sel?'#388bfd':'#3fb950');ctx.fill();
    if(i===S.sel){ctx.lineWidth=2;ctx.strokeStyle='#fff';ctx.stroke();}
    ctx.fillStyle=parked?'#f85149':'#fff';ctx.font='bold 11px system-ui';
    ctx.fillText(i, q.x+8, q.y-6);
  });
}
function renderList(){
  const ol=$('list');ol.innerHTML='';
  S.names.forEach((nm,i)=>{
    const li=document.createElement('li');const p=S.pts[i];
    li.innerHTML=`${i}. ${nm}<span class=v>${p.v>0?'vis':'hidden'}</span>`;
    li.value=i;
    if(p.v===0)li.classList.add('hidden');
    if(i===S.sel)li.classList.add('cur');
    li.onclick=()=>{S.sel=i;draw();renderList();};
    ol.appendChild(li);
  });
}

let saveTimer=null;
function autosave(){ clearTimeout(saveTimer); saveTimer=setTimeout(()=>save(false),350); }

async function save(markReviewed){
  const name=S.imgs[S.idx];
  const pts=S.pts.map(p=>({x:Math.round(p.x),y:Math.round(p.y),v:p.v>0?p.v:0}));
  const wasReviewed=(S.labels[name]||{}).reviewed||false;
  const reviewed = markReviewed===true ? true : wasReviewed;
  const r=await fetch('/api/save/'+encodeURIComponent(name),{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({points:pts,w:S.w,h:S.h,reviewed:reviewed})});
  if(r.ok){
    S.labels[name]={...(S.labels[name]||{}),points:pts,reviewed:reviewed};
    $('savemark').textContent='saved ✓';
    $('done').textContent=S.imgs.filter(n=>S.labels[n]&&S.labels[n].reviewed).length;
  }
}
function go(d){S.idx=Math.max(0,Math.min(S.imgs.length-1,S.idx+d));load();}
$('prev').onclick=()=>save(false).then(()=>go(-1));
$('next').onclick=()=>save(true).then(()=>go(1));

boot();
</script></body></html>"""

if __name__ == "__main__":
    main()
