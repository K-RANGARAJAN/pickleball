"""Line-drag court calibration UI.

Drag the 4 court edges in a loop (A->B left sideline, B->C net line, C->D right sideline,
D->A near baseline); each edge is fit by total-least-squares and the 4 corners are the edge
intersections. Solves the pose immediately and shows reprojection error. Trace the same
physical corner / direction across cameras.

Run:  python scripts/corner_click_app.py   ->  http://localhost:8078
Writes configs/camera_corners.json.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import HTMLResponse

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from linecall.calibration.pose import solve_camera_pose_auto  # noqa: E402
from linecall.calibration.line_fit import corners_from_edges    # noqa: E402

CLIPS = {"cam1": REPO / "CAMERA 1.mp4", "cam2": REPO / "camera2_muted.mp4",
         "metric": REPO / "sideline.mp4"}
EDGE_KEYS = ["AB", "BC", "CD", "DA"]   # loop order
OUT_JSON = REPO / "configs" / "camera_corners.json"

app = FastAPI(title="line-drag calibration")
_caps: dict[str, cv2.VideoCapture] = {}
_meta: dict[str, dict] = {}


def _cap(cam: str) -> cv2.VideoCapture:
    if cam not in CLIPS:
        raise HTTPException(404, cam)
    if cam not in _caps:
        c = cv2.VideoCapture(str(CLIPS[cam]))
        if not c.isOpened():
            raise HTTPException(500, f"cannot open {CLIPS[cam]}")
        _caps[cam] = c
        _meta[cam] = {"fps": c.get(cv2.CAP_PROP_FPS) or 60.0,
                      "frames": int(c.get(cv2.CAP_PROP_FRAME_COUNT)),
                      "w": int(c.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      "h": int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    return _caps[cam]


def _clap_frame(cam: str) -> int:
    """Default calibration frame: ~10 s after the clap (empty-court seconds)."""
    try:
        out = subprocess.run(["ffmpeg", "-v", "error", "-t", "30", "-i", str(CLIPS[cam]),
                              "-vn", "-ac", "1", "-ar", "8000", "-f", "f32le", "pipe:1"],
                             capture_output=True).stdout
        sig = np.frombuffer(out, dtype=np.float32)
        if sig.size == 0:
            return int(10.0 * _meta[cam]["fps"])
        env = np.convolve(sig ** 2, np.ones(40) / 40, mode="same")
        clap_s = float(np.argmax(env)) / 8000.0
        return int((clap_s + 10.0) * _meta[cam]["fps"])
    except Exception:
        return int(10.0 * _meta[cam]["fps"])


@app.get("/api/info")
def info():
    out = {}
    for cam in CLIPS:
        _cap(cam)
        out[cam] = {**_meta[cam], "default_frame": _clap_frame(cam)}
    saved = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {}
    return {"cams": out, "saved": saved}


@app.get("/frame/{cam}/{idx}")
def frame(cam: str, idx: int):
    c = _cap(cam)
    c.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(idx, _meta[cam]["frames"] - 1)))
    ok, f = c.read()
    if not ok:
        raise HTTPException(404, f"frame {idx}")
    ok, jpg = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return Response(jpg.tobytes(), media_type="image/jpeg")


@app.post("/api/solve/{cam}")
def solve(cam: str, payload: dict):
    """payload: {frame_idx, edges: {AB:[[x,y],...], BC:[...], CD:[...], DA:[...]}}
    -> fit edges, intersect into 4 corners, solve pose, save."""
    raw = payload.get("edges", {})
    edges = {k: [tuple(p) for p in raw.get(k, [])] for k in EDGE_KEYS}
    thin = [k for k in EDGE_KEYS if len(edges[k]) < 2]
    if thin:
        raise HTTPException(400, f"edges need >=2 points each; too few on: {thin}")
    try:
        corners, rms = corners_from_edges(edges)
    except ValueError as e:
        raise HTTPException(400, str(e))

    m = _meta[cam]
    # HONEST QUALITY CHECK + AUTO-RELABEL. The 4 traced corners must form a real 20x44 court
    # rectangle in 3D; we measure the REPROJECTION ERROR as the GOOD/BAD signal. But the
    # naive named solve assumes the human's far/near + left/right labels match the world
    # frame, which FAILS for an opposite-facing camera (cam2 sees the court from the other
    # end, so its "far-left" pixel is physically a different world corner). So we solve via
    # solve_camera_pose_auto, which tries all 8 cyclic relabelings of the traced quad and
    # keeps the physically-valid pose (camera above ground, outside court) with the lowest
    # reprojection. A correct human trace then always lands a few px regardless of which
    # corner they started from or which way the camera faces.
    loop = [corners["far_left"], corners["far_right"],
            corners["near_right"], corners["near_left"]]
    pose, named = None, None
    for hint in ("near", "far"):
        try:
            p, n = solve_camera_pose_auto(cam, loop, m["w"], m["h"], camera_behind=hint)
        except Exception:
            continue
        if pose is None or p.reproj_error_px < pose.reproj_error_px:
            pose, named = p, n

    reproj = pose.reproj_error_px if pose else 9999.0
    cz = pose.camera_position_ft[2] if pose else -1.0
    good = pose is not None and reproj <= 8.0 and cz > 0

    # persist the corners under their TRUE world identity (the relabeling the solve found),
    # so the downstream ground homography maps pixels to the correct court positions.
    world_corners = named if (good and named) else corners

    result = {
        "reproj_px": round(reproj, 2),
        "well_calibrated": bool(good),
        "camera_position_ft": [round(v, 1) for v in pose.camera_position_ft] if pose else None,
        "plausible": bool(good),
        "baseline": pose.baseline if pose else "",
        "edge_rms_px": rms,
        "corners_px": {k: [round(v[0], 1), round(v[1], 1)] for k, v in world_corners.items()},
    }
    saved = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {}
    saved[cam] = {"frame_idx": payload["frame_idx"],
                  "image_w": m["w"], "image_h": m["h"],
                  "corners": {k: list(map(float, v)) for k, v in world_corners.items()},
                  "calibrated_via": "line_drag_auto",
                  **result}
    OUT_JSON.parent.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps(saved, indent=1))
    return result


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>line-drag calibration</title>
<style>
 body{margin:0;background:#0e1116;color:#e6edf3;font:14px system-ui;padding:16px}
 h1{font-size:16px;margin:0 0 4px} .sub{color:#8b97a5;font-size:12.5px;margin-bottom:12px;max-width:1100px}
 .cams{display:grid;grid-template-columns:1fr 1fr;gap:14px}
 .cam{background:#161b22;border:1px solid #2b3340;border-radius:10px;padding:10px}
 .cam h2{font-size:14px;margin:0 0 6px}
 .wrap{position:relative} canvas{width:100%;border-radius:6px;display:block;cursor:crosshair;touch-action:none}
 .bar{display:flex;gap:8px;align-items:center;margin-top:8px;flex-wrap:wrap;font-size:12.5px}
 button{background:#21304b;color:#e6edf3;border:1px solid #3a4f78;border-radius:6px;padding:4px 10px;cursor:pointer}
 .next{font-weight:700} .res{margin-top:6px;font-size:12.5px;white-space:pre-line}
 .ok{color:#3fb950}.bad{color:#f85149}
 #diagram{background:#161b22;border:1px solid #2b3340;border-radius:10px;
   padding:10px;margin-bottom:14px;display:flex;gap:18px;align-items:center}
 svg text{font:bold 13px system-ui}
 .legend{font-size:12.5px;color:#8b97a5;line-height:1.6}
 .e0{color:#f85149}.e1{color:#d29922}.e2{color:#3fb950}.e3{color:#58a6ff}
 #zoom{position:fixed;border:2px solid #3a4f78;border-radius:8px;background:#000;
   width:200px;height:200px;pointer-events:none;display:none;z-index:10}
</style></head><body>
<h1>line-drag court calibration (diagonal-safe)</h1>
<div class="sub">Lay a <b>straight line</b> along each of the four court edges, in loop order starting at
corner <b>A</b>: press and drag to draw a line (anchor → release), then nudge either
endpoint to sit it exactly on the edge. Order is <b>A→B→C→D</b> then the baseline back
<b>D→A</b>. Corners are computed where your lines meet — more accurate than clicking. Use the
<b>same physical loop, same direction</b> in both cameras; frame slider if a player covers a line.</div>

<div id="diagram">
  <svg viewBox="-3 -3 26 50" width="130" height="240">
    <rect x="0" y="0" width="20" height="22" fill="#173043" stroke="#8b97a5" stroke-width=".3"/>
    <line x1="0" y1="22" x2="20" y2="22" stroke="#3fb950" stroke-width=".7"/>
    <line x1="0" y1="0" x2="0" y2="22" stroke="#f85149" stroke-width=".7"/>
    <line x1="20" y1="22" x2="20" y2="0" stroke="#58a6ff" stroke-width=".7"/>
    <line x1="20" y1="0" x2="0" y2="0" stroke="#d29922" stroke-width=".7"/>
    <circle cx="0" cy="0" r="1.6" fill="#fff"/><text x="-3" y="-0.2" fill="#fff">A</text>
    <circle cx="0" cy="22" r="1.4" fill="#3fb950"/><text x="1.5" y="25" fill="#3fb950">B</text>
    <circle cx="20" cy="22" r="1.4" fill="#58a6ff"/><text x="15" y="25" fill="#58a6ff">C</text>
    <circle cx="20" cy="0" r="1.4" fill="#d29922"/><text x="15" y="-0.2" fill="#d29922">D</text>
  </svg>
  <div class="legend">
    Trace the FULL OUTER COURT (the whole 20×44 rectangle) — both cameras now calibrate to
    the SAME court, so there is no half-court seam and no false OUTs in the gap:<br>
    <span class="e0">edge A→B = FAR baseline</span> (far-left corner → far-right corner)<br>
    <span class="e2">edge B→C = RIGHT sideline</span> (far-right corner → near-right corner)<br>
    <span class="e3">edge C→D = NEAR baseline</span> (near-right corner → near-left corner)<br>
    <span class="e1">edge D→A = LEFT sideline</span> (near-left corner → far-left corner)<br>
    A = your FAR-LEFT baseline corner; keep "left" and "far" the same physical sides in both
    cameras, and trace the loop the same rotational direction in both.<br>
    Each edge is a solid straight line — drop it, then drag its endpoints onto the court line.
    After all 4 lines the camera auto-solves: <b>green ✓ &lt; 8 px</b> reprojection; a wrong
    loop or swapped corner shows a huge error — clear &amp; redo.
  </div>
</div>

<div class="cams" id="cams"></div>
<canvas id="zoom" width="200" height="200"></canvas>

<script>
"use strict";
const EDGES=["AB","BC","CD","DA"];
const ELABEL=["A→B FAR baseline","B→C RIGHT sideline","C→D NEAR baseline","D→A LEFT sideline"];
const COLORS=["#f85149","#3fb950","#58a6ff","#d29922"];
const state={};
const zoom=document.getElementById('zoom'), zctx=zoom.getContext('2d');

async function boot(){
  const info=await (await fetch('/api/info')).json();
  for(const cam of Object.keys(info.cams)){
    const m=info.cams[cam];
    state[cam]={frame:m.default_frame, frames:m.frames,
                // each edge is a straight line: {a:[x,y], b:[x,y]} or null until placed
                edges:{AB:null,BC:null,CD:null,DA:null}, cur:0,
                mode:null, grab:null,   // mode: 'place' (new line) | 'handle' (move endpoint)
                img:new Image(), saved:info.saved[cam]||null};
    const div=document.createElement('div');
    div.className='cam';
    div.innerHTML=`<h2>${cam}</h2>
      <div class="wrap"><canvas id="cv_${cam}"></canvas></div>
      <div class="bar">
        <span>frame <input type="range" id="sl_${cam}" min="0" max="${m.frames-1}" value="${m.default_frame}" style="width:170px">
        <span id="fl_${cam}">${m.default_frame}</span></span>
        <button onclick="step('${cam}',-30)">-30</button><button onclick="step('${cam}',30)">+30</button>
        <button onclick="undoEdge('${cam}')">undo edge</button>
        <button onclick="clearAll('${cam}')">clear</button>
        <span class="next" id="nx_${cam}"></span>
      </div><div class="res" id="res_${cam}"></div>`;
    document.getElementById('cams').appendChild(div);
    const cv=document.getElementById('cv_'+cam);
    cv.onpointerdown=e=>down(cam,e);
    cv.onpointermove=e=>{move(cam,e); mag(cam,e);};
    cv.onpointerup=e=>up(cam,e);
    cv.onpointerleave=()=>{zoom.style.display='none';};
    const sl=document.getElementById('sl_'+cam);
    sl.oninput=()=>{state[cam].frame=+sl.value; document.getElementById('fl_'+cam).textContent=sl.value;};
    sl.onchange=()=>loadFrame(cam);
    if(state[cam].saved && state[cam].saved.corners){
      // rebuild the 4 straight edges from the saved corners so they can be re-tuned
      const C=state[cam].saved.corners;
      if(C.far_left&&C.far_right&&C.near_right&&C.near_left){
        state[cam].edges={AB:{a:C.far_left,b:C.far_right}, BC:{a:C.far_right,b:C.near_right},
                          CD:{a:C.near_right,b:C.near_left}, DA:{a:C.near_left,b:C.far_left}};
        state[cam].cur=4;
        if(state[cam].saved.reproj_px!==undefined) showResult(cam, state[cam].saved);
      }
    }
    loadFrame(cam);
  }
}
function step(cam,d){const s=state[cam]; s.frame=Math.max(0,Math.min(s.frames-1,s.frame+d));
  document.getElementById('sl_'+cam).value=s.frame;
  document.getElementById('fl_'+cam).textContent=s.frame; loadFrame(cam);}
function loadFrame(cam){
  const s=state[cam]; s.img=new Image(); s.img.onload=()=>draw(cam);
  s.img.src=`/frame/${cam}/${s.frame}?t=${Date.now()}`; updateNext(cam);
}
function pos(cam,e){
  const cv=document.getElementById('cv_'+cam), r=cv.getBoundingClientRect();
  return [(e.clientX-r.left)*cv.width/r.width, (e.clientY-r.top)*cv.height/r.height];
}
const HANDLE_R=16;   // px radius to "grab" an endpoint for fine-tuning
function dist(p,q){return Math.hypot(p[0]-q[0],p[1]-q[1]);}
// find an existing endpoint near point p -> {key, end:'a'|'b'} or null
function nearHandle(s,p){
  for(const k of EDGES){
    const ln=s.edges[k]; if(!ln) continue;
    if(dist(p,ln.a)<=HANDLE_R) return {key:k,end:'a'};
    if(dist(p,ln.b)<=HANDLE_R) return {key:k,end:'b'};
  }
  return null;
}
function down(cam,e){
  const s=state[cam]; const p=pos(cam,e);
  // 1) if pressing on an existing endpoint, grab it to adjust (works for any placed edge)
  const h=nearHandle(s,p);
  if(h){ s.mode='handle'; s.grab=h; draw(cam); return; }
  // 2) otherwise, if there's still an edge to place, start a new straight line
  if(s.cur<4){ s.mode='place'; s.edges[EDGES[s.cur]]={a:p,b:p}; draw(cam); }
}
function move(cam,e){
  const s=state[cam]; if(!s.mode) return;
  const p=pos(cam,e);
  if(s.mode==='place'){ s.edges[EDGES[s.cur]].b=p; }
  else if(s.mode==='handle'){ s.edges[s.grab.key][s.grab.end]=p; }
  draw(cam);
}
function up(cam,e){
  const s=state[cam]; if(!s.mode) return;
  if(s.mode==='place'){
    const k=EDGES[s.cur], ln=s.edges[k];
    if(!ln || dist(ln.a,ln.b)<HANDLE_R){ s.edges[k]=null; s.mode=null; updateNext(cam); draw(cam); return; } // too short, redo
    s.cur++; updateNext(cam);
  }
  // adjusting a handle: re-solve once all 4 are present (handled below)
  s.mode=null; s.grab=null; draw(cam);
  if(s.cur===4) solve(cam);
}
function draw(cam){
  const s=state[cam], cv=document.getElementById('cv_'+cam), ctx=cv.getContext('2d');
  if(!s.img.width) return;
  cv.width=s.img.width; cv.height=s.img.height; ctx.drawImage(s.img,0,0);
  EDGES.forEach((k,i)=>{
    const ln=s.edges[k]; if(!ln) return;
    // solid straight line
    ctx.strokeStyle=COLORS[i]; ctx.lineWidth=4;
    ctx.beginPath(); ctx.moveTo(ln.a[0],ln.a[1]); ctx.lineTo(ln.b[0],ln.b[1]); ctx.stroke();
    // endpoint handles (draggable)
    ctx.fillStyle=COLORS[i]; ctx.strokeStyle='#fff'; ctx.lineWidth=2;
    [ln.a,ln.b].forEach(pt=>{ ctx.beginPath(); ctx.arc(pt[0],pt[1],8,0,7); ctx.fill(); ctx.stroke(); });
  });
  if(s.solved && s.solved.corners_px){
    const C=s.solved.corners_px;
    ctx.fillStyle='#fff'; ctx.strokeStyle='#111'; ctx.lineWidth=2;
    for(const n in C){ ctx.beginPath(); ctx.arc(C[n][0],C[n][1],6,0,7); ctx.fill(); ctx.stroke(); }
  }
}
// turn a 2-endpoint line into N points along it, for the line-fit backend
function densify(ln,n=80){
  const out=[]; for(let i=0;i<n;i++){const t=i/(n-1);
    out.push([ln.a[0]+t*(ln.b[0]-ln.a[0]), ln.a[1]+t*(ln.b[1]-ln.a[1])]);}
  return out;
}
function updateNext(cam){
  const s=state[cam];
  document.getElementById('nx_'+cam).innerHTML =
    s.cur<4 ? `draw <span style="color:${COLORS[s.cur]}">${ELABEL[s.cur]}</span> — press & drag a straight line, then nudge the endpoints`
            : 'all 4 lines set — drag any endpoint to fine-tune (auto re-solves)';
}
function undoEdge(cam){const s=state[cam]; if(s.cur>0){s.cur--; s.edges[EDGES[s.cur]]=null;}
  s.solved=null; document.getElementById('res_'+cam).textContent=''; updateNext(cam); draw(cam);}
function clearAll(cam){const s=state[cam]; s.cur=0; EDGES.forEach(k=>s.edges[k]=null); s.solved=null;
  document.getElementById('res_'+cam).textContent=''; updateNext(cam); draw(cam);}
async function solve(cam){
  const s=state[cam];
  if(EDGES.some(k=>!s.edges[k])) return;       // not all 4 lines placed yet
  const payload={}; EDGES.forEach(k=>payload[k]=densify(s.edges[k]));
  const r=await fetch('/api/solve/'+cam,{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({frame_idx:s.frame, edges:payload})});
  const j=await r.json();
  if(!r.ok){document.getElementById('res_'+cam).textContent='solve failed: '+(j.detail||r.status);return;}
  s.solved=j; showResult(cam,j); draw(cam);
}
function showResult(cam,j){
  const el=document.getElementById('res_'+cam);
  const good=j.well_calibrated;
  el.className='res '+(good?'ok':'bad');
  const reproj=(j.reproj_px!==undefined&&j.reproj_px!==null)?j.reproj_px.toFixed(1)+'px':'n/a';
  const pos=j.camera_position_ft?` · camera at (${j.camera_position_ft.join(', ')}) ft`:'';
  if(good){
    el.textContent=`✓ GOOD CALIBRATION — corners reproject to ${reproj} (tight)${pos} · saved`;
  }else{
    el.textContent=`✗ BAD CALIBRATION — corners reproject to ${reproj} (should be ≤8px). The 4 corners don't form a real court rectangle — re-trace them on the actual baseline/sideline lines, in order, and re-solve.`;
  }
}
function mag(cam,e){
  const s=state[cam]; if(!s.img.width) return;
  const [x,y]=pos(cam,e);
  zoom.style.display='block';
  zoom.style.left=(e.clientX+24)+'px'; zoom.style.top=(e.clientY-100)+'px';
  zctx.imageSmoothingEnabled=false; zctx.clearRect(0,0,200,200);
  zctx.drawImage(s.img, x-25, y-25, 50, 50, 0, 0, 200, 200);
  zctx.strokeStyle='#f85149'; zctx.beginPath();
  zctx.moveTo(100,85);zctx.lineTo(100,115);zctx.moveTo(85,100);zctx.lineTo(115,100);zctx.stroke();
}
boot();
</script></body></html>"""


if __name__ == "__main__":
    print("line-drag calibration -> http://localhost:8078  (Ctrl-C to stop)")
    uvicorn.run(app, host="127.0.0.1", port=8078, log_level="warning")
