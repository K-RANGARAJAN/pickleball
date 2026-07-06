"""12-keypoint court labeling tool for YOLO-pose training.

Labels the FULL court (12 points) on each frame in dataset/. The POC goal is one-sided
calibration: a baseline camera should lock onto its NEAR half. So we label everything, but
the per-point VISIBILITY flag carries the weight — mark a far point (blurred through the net)
as partial/not-visible rather than guessing it visible. That asymmetry is what teaches the
model "near half solid, far half uncertain", which is what makes the near-half demo credible.

Click order (enforced):
  1-4  outer corners: TL, TR, BR, BL  -> thin quad drawn
  5    left  sideline @ far kitchen     6 right... no: see KP_NAMES below (matches the diagram)
  Actually points follow the user's diagram numbering (1-indexed):
    1 TL, 2 TR, 3 BR, 4 BL,
    5 left-sideline near-kitchen, 6 left-sideline far-kitchen,
    7 far baseline mid, 8 right-sideline far-kitchen, 9 right-sideline near-kitchen,
    10 near baseline mid, 11 far kitchen centerline, 12 near kitchen centerline.
  After 1-4: quad. After 5-10: lines 6->8 and 5->9 drawn. Then 11,12 on those lines.

Per point:  left-click = visible (v2) · right-click = partial (v1) · spacebar = not-visible (v0)
Output:  one YOLO-pose .txt per image in dataset/labels/  (class cx cy w h  x1 y1 v1 ... x12 y12 v12)
         plus labels/_keypoints.json (pixel coords + visibility) for inspection.

Run:  python scripts/keypoint_label_app.py [--dir dataset]  ->  http://localhost:8077
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse

REPO = Path(__file__).resolve().parent.parent

# 1-indexed point names matching the user's hand-drawn diagram, with which court HALF
# each belongs to. "near" = the half a baseline camera locks onto in the POC demo.
KP_NAMES = [
    ("top_left",            "far"),   # 1
    ("top_right",           "far"),   # 2
    ("bottom_right",        "near"),  # 3
    ("bottom_left",         "near"),  # 4
    ("left_near_kitchen",   "near"),  # 5
    ("left_far_kitchen",    "far"),   # 6
    ("far_baseline_mid",    "far"),   # 7
    ("right_far_kitchen",   "far"),   # 8
    ("right_near_kitchen",  "near"),  # 9
    ("near_baseline_mid",   "near"),  # 10
    ("far_kitchen_mid",     "far"),   # 11
    ("near_kitchen_mid",    "near"),  # 12
]
N_KP = len(KP_NAMES)
IMG_EXTS = {".jpg", ".jpeg", ".png"}

app = FastAPI(title="court keypoint labeler")
STATE: dict = {"dir": REPO / "dataset", "images": [], "labels_dir": None, "json_path": None}


def _images(d: Path) -> list[str]:
    if not d.is_dir():
        raise SystemExit(f"not a folder: {d}  (run scripts/extract_dataset.py first)")
    imgs = sorted(p.name for p in d.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not imgs:
        raise SystemExit(f"no images in {d}")
    return imgs


def _load_json() -> dict:
    p = STATE["json_path"]
    return json.loads(p.read_text()) if p.exists() else {}


def _save_json(d: dict) -> None:
    STATE["json_path"].write_text(json.dumps(d, indent=1))


def _write_yolo(name: str, pts: list[dict], w: int, h: int) -> None:
    """Write one YOLO-pose label line. pts: list of {x,y,v} (pixel coords, v in 0/1/2).
    Box = tight rect around all placed (v>0) points, with padding, normalized."""
    placed = [(p["x"], p["y"]) for p in pts if p["v"] > 0]
    txt_path = STATE["labels_dir"] / (Path(name).stem + ".txt")
    if not placed:
        txt_path.write_text("")          # nothing labeled yet
        return
    xs, ys = [p[0] for p in placed], [p[1] for p in placed]
    pad = 0.02 * max(w, h)
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
    done = sum(1 for n in STATE["images"] if data.get(n, {}).get("complete"))
    return {
        "images": STATE["images"],
        "labels": data,
        "kp_names": [n for n, _ in KP_NAMES],
        "kp_half": [h for _, h in KP_NAMES],
        "n_done": done,
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
    data[name] = {"points": pts, "w": w, "h": h, "complete": bool(body.get("complete"))}
    _save_json(data)
    _write_yolo(name, pts, w, h)
    return JSONResponse({"ok": True})


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=REPO / "dataset")
    ap.add_argument("--port", type=int, default=8077)
    args = ap.parse_args()
    STATE["dir"] = args.dir
    STATE["images"] = _images(args.dir)
    STATE["labels_dir"] = args.dir / "labels"
    STATE["labels_dir"].mkdir(parents=True, exist_ok=True)
    STATE["json_path"] = args.dir / "labels" / "_keypoints.json"
    print(f"  {len(STATE['images'])} images in {args.dir}")
    print(f"  labels -> {STATE['labels_dir']}")
    print(f"  open http://localhost:{args.port}")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


HTML = r"""<!doctype html><html><head><meta charset=utf-8><title>court keypoint labeler</title>
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
#img{max-width:98vw;max-height:78vh;display:block}
#ov{position:absolute;inset:0;cursor:crosshair}
.pill{padding:2px 8px;border-radius:10px;background:#1d2530;border:1px solid var(--edge);color:var(--dim)}
.next{color:var(--txt);font-weight:600}
.legend{display:flex;gap:10px;font-size:12px;color:var(--dim);align-items:center}
.dot{width:10px;height:10px;border-radius:50%;display:inline-block;margin-right:3px;vertical-align:middle}
.side{position:fixed;right:8px;top:54px;width:172px;background:var(--card);border:1px solid var(--edge);border-radius:8px;padding:8px;font-size:12px}
.side ol{margin:4px 0 0;padding-left:20px}
.side li{padding:1px 0;color:var(--dim)}
.side li.done{color:var(--green)}
.side li.cur{color:var(--txt);font-weight:700}
.side li .v{float:right;font-size:10px}
.hint{font-size:12px;color:var(--dim);padding:0 12px 8px}
.savemark{color:var(--green);font-size:12px}
</style></head><body>
<div class=top>
  <button id=prev>◀ Prev</button>
  <span class=pill id=counter>—</span>
  <button id=next>Next ▶</button>
  <span class=pill>done <b id=done>0</b>/<b id=total>0</b></span>
  <span class=spacer></span>
  <span class=legend>
    <span><span class=dot style=background:var(--green)></span>visible (L-click)</span>
    <span><span class=dot style=background:var(--amber)></span>partial (R-click)</span>
    <span><span class=dot style=background:var(--red)></span>not-visible (space)</span>
  </span>
  <button id=undo>Undo (Z)</button>
  <button id=clear>Clear</button>
  <span class=savemark id=savemark></span>
</div>
<div class=hint id=hint></div>
<div class=wrap><div id=stage><img id=img><canvas id=ov></canvas></div></div>
<div class=side><b>Points</b><ol id=list></ol>
<div style="margin-top:6px;color:var(--dim);font-size:11px">near-half points are the POC demo target</div></div>
<script>
let S={imgs:[],idx:0,names:[],half:[],pts:[],step:0,w:0,h:0,scale:1};
const img=document.getElementById('img'),ov=document.getElementById('ov'),ctx=ov.getContext('2d');
const $=id=>document.getElementById(id);

// click order = the diagram order 1..12 (indices 0..11)
const ORDER=[0,1,2,3,4,5,6,7,8,9,10,11];

async function boot(){
  const r=await fetch('/api/state').then(r=>r.json());
  S.imgs=r.images;S.names=r.kp_names;S.half=r.kp_half;
  $('total').textContent=r.total;$('done').textContent=r.n_done;
  S.labels=r.labels||{};
  S.idx=S.imgs.findIndex(n=>!(S.labels[n]&&S.labels[n].complete));
  if(S.idx<0)S.idx=0;
  load();
}
function freshPts(){return S.names.map(()=>({x:0,y:0,v:0,set:false}));}

function load(){
  const name=S.imgs[S.idx];
  $('counter').textContent=`${S.idx+1} / ${S.imgs.length}  ·  ${name}`;
  S.step=0;
  const saved=S.labels[name];
  S.pts = saved&&saved.points ? saved.points.map(p=>({...p,set:p.v>0||p.placed})) : freshPts();
  if(saved&&saved.points){S.step=S.pts.filter(p=>p.set).length;}
  img.onload=()=>{
    S.w=img.naturalWidth;S.h=img.naturalHeight;
    ov.width=img.clientWidth;ov.height=img.clientHeight;
    S.scale=img.clientWidth/S.w;
    draw();updateHint();renderList();
  };
  img.src='/img/'+encodeURIComponent(name)+'?t='+Date.now();
  $('savemark').textContent='';
}

function nextStep(){return ORDER[S.step];} // point index to place next, or undefined
function updateHint(){
  const i=nextStep();
  if(i===undefined){$('hint').innerHTML='<span class=next>All 12 placed.</span> Adjust if needed, then Next ▶ (auto-saves).';return;}
  $('hint').innerHTML=`Place point <span class=next>${i+1} — ${S.names[i]}</span> (${S.half[i]} half). `+
    `Left=visible · Right=partial · Space=not-visible`;
}

function place(i,x,y,v){
  S.pts[i]={x:Math.round(x),y:Math.round(y),v:v,set:true,placed:true};
}
function advance(){ if(S.step<ORDER.length)S.step++; updateHint(); }

ov.addEventListener('mousedown',e=>{
  const i=nextStep(); if(i===undefined)return;
  e.preventDefault();
  const r=ov.getBoundingClientRect();
  const x=(e.clientX-r.left)/S.scale, y=(e.clientY-r.top)/S.scale;
  if(e.button===0)place(i,x,y,2);
  else if(e.button===2)place(i,x,y,1);
  else return;
  advance();draw();renderList();
});
ov.addEventListener('contextmenu',e=>e.preventDefault());
window.addEventListener('keydown',e=>{
  if(e.code==='Space'){const i=nextStep();if(i!==undefined){e.preventDefault();place(i,0,0,0);advance();draw();renderList();}}
  else if(e.key==='z'||e.key==='Z'){undo();}
  else if(e.key==='ArrowRight'){save(true).then(go.bind(null,1));}
  else if(e.key==='ArrowLeft'){save(false).then(go.bind(null,-1));}
});
$('undo').onclick=undo;
function undo(){ if(S.step>0){S.step--;const i=ORDER[S.step];S.pts[i]={x:0,y:0,v:0,set:false};draw();updateHint();renderList();} }
$('clear').onclick=()=>{S.pts=freshPts();S.step=0;draw();updateHint();renderList();};

function lineBetween(a,b){
  const pa=S.pts[a],pb=S.pts[b];
  if(pa&&pa.set&&pa.v>0&&pb&&pb.set&&pb.v>0){
    ctx.beginPath();ctx.moveTo(pa.x*S.scale,pa.y*S.scale);ctx.lineTo(pb.x*S.scale,pb.y*S.scale);ctx.stroke();
  }
}
function draw(){
  ctx.clearRect(0,0,ov.width,ov.height);
  // thin black court lines once enough points exist
  ctx.lineWidth=1.4;ctx.strokeStyle='#000';
  // outer quad 1-2-3-4 (idx 0-1-2-3)
  [[0,1],[1,2],[2,3],[3,0]].forEach(([a,b])=>lineBetween(a,b));
  // kitchen lines: 6->8 (idx5->7) and 5->9 (idx4->8)
  lineBetween(5,7);lineBetween(4,8);
  // centerline 7->10 (idx6->9) faint to help place 11,12
  ctx.strokeStyle='#444';lineBetween(6,9);
  // points
  const col={2:'#3fb950',1:'#d29922',0:'#f85149'};
  S.pts.forEach((p,i)=>{
    if(!p.set)return;
    const px=p.v>0?p.x*S.scale:14+ (i*0); // not-visible: park label at corner
    if(p.v>0){
      ctx.fillStyle=col[p.v];ctx.beginPath();ctx.arc(p.x*S.scale,p.y*S.scale,5,0,7);ctx.fill();
      ctx.fillStyle='#fff';ctx.font='bold 11px system-ui';ctx.fillText(i+1,p.x*S.scale+7,p.y*S.scale-6);
    }
  });
}
function renderList(){
  const ol=$('list');ol.innerHTML='';
  S.names.forEach((nm,i)=>{
    const li=document.createElement('li');
    const p=S.pts[i];
    const vtxt=p&&p.set?({2:'vis',1:'part',0:'skip'})[p.v]:'';
    li.innerHTML=`${i+1}. ${nm}<span class=v>${vtxt}</span>`;
    if(p&&p.set)li.className='done';
    if(ORDER[S.step]===i)li.className='cur';
    if(S.half[i]==='near')li.style.borderLeft='2px solid var(--blue)',li.style.paddingLeft='4px';
    ol.appendChild(li);
  });
}

async function save(markComplete){
  const name=S.imgs[S.idx];
  const pts=S.pts.map(p=>({x:p.set?p.x:0,y:p.set?p.y:0,v:p.set?p.v:0,placed:!!p.set}));
  const complete=S.pts.every(p=>p.set);
  const r=await fetch('/api/save/'+encodeURIComponent(name),{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({points:pts,w:S.w,h:S.h,complete:complete&&markComplete!==false?true:complete})});
  if(r.ok){S.labels[name]={points:pts,complete:complete};$('savemark').textContent='saved ✓';
    $('done').textContent=S.imgs.filter(n=>S.labels[n]&&S.labels[n].complete).length;}
}
function go(d){S.idx=Math.max(0,Math.min(S.imgs.length-1,S.idx+d));load();}
$('prev').onclick=()=>save(false).then(()=>go(-1));
$('next').onclick=()=>save(true).then(()=>go(1));

boot();
</script></body></html>"""

if __name__ == "__main__":
    main()
