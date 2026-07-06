"""Two-angle recorded demo: synced cam1/cam2 playback + cam1 ball trail + IN/OUT beeps.

Reads runs/pipeline/<tag>/{calls.json, track.csv}; on first load it cuts both clips to the
slice with ffmpeg (so they share one timeline) and transcodes to H.264. Calls come from cam1;
cam2 is a synced second view. Beeps via WebAudio (IN 880Hz single / OUT 300Hz double).

  python scripts/demo_app.py --tag final   ->  http://localhost:8080
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

REPO = Path(__file__).resolve().parent.parent
CLIPS = {"cam1": REPO / "CAMERA 1.mp4", "cam2": REPO / "camera2_muted.mp4"}
CHUNK = 1024 * 1024

app = FastAPI(title="linecall-demo")
RUN_DIR: Path = None  # set in main()


def _calls() -> dict:
    return json.loads((RUN_DIR / "calls.json").read_text())


def _ensure_demo_clips() -> None:
    """Cut+transcode both cams to the slice, baking in the sync offset."""
    c = _calls()
    t0, t1 = c["slice"]["t0"], c["slice"]["t1"]
    for cam, src in CLIPS.items():
        out = RUN_DIR / f"demo_{cam}.mp4"
        if out.exists():
            continue
        start_cam = t0 + c.get("clip_offsets", {}).get(cam, 0.0)   # ref time -> cam clock
        dur = t1 - t0
        print(f"[transcode] {cam}: cutting {dur:.0f}s from {start_cam:.2f}s "
              f"(one-time, ~a minute)…")
        cmd = ["ffmpeg", "-y", "-v", "error",
               "-ss", f"{start_cam:.3f}", "-i", str(src), "-t", f"{dur:.3f}",
               "-vf", "fps=60", "-c:v", "libx264", "-preset", "veryfast",
               "-crf", "21", "-pix_fmt", "yuv420p",
               "-c:a", "aac", "-movflags", "+faststart", str(out)]
        r = subprocess.run(cmd, capture_output=True)
        if r.returncode != 0:
            sys.exit(f"ffmpeg failed: {r.stderr.decode(errors='replace')[:300]}")


@app.get("/api/data")
def data():
    c = _calls()
    t0 = c["slice"]["t0"]
    fps_ref = c["slice"]["fps_ref"]
    def _f(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    def _pair(a, b):
        fa, fb = _f(a), _f(b)
        return [fa, fb] if (fa is not None and fb is not None) else None

    track = []
    with open(RUN_DIR / "track.csv") as f:
        for row in csv.DictReader(f):
            c1 = _pair(row["cam1_px_x"], row["cam1_px_y"])
            if c1 is None:
                continue  # cam1 didn't see the ball this frame — overlay gap
            track.append({
                "t": round(float(row["ref_idx"]) / fps_ref - t0, 4),  # demo-clip time
                "c1": c1,
            })
    # cam1 calls, each with demo-clip time
    cam_calls = {}
    for cam, lst in c.get("cam_calls", {}).items():
        cam_calls[cam] = [{**b, "t": round(b["t_ref"] - t0, 3)} for b in lst]

    all_calls = sorted(cam_calls.get("cam1", []), key=lambda b: b["t"])
    return JSONResponse({"slice": c["slice"], "sync": c["sync"],
                         "calibration": c["calibration"],
                         "detection": c["detection"],
                         "track": track,
                         "cam_calls": cam_calls,
                         "disagreements": [],
                         "bounces": all_calls})


def _parse_range(header: str, size: int):
    try:
        unit, _, rng = header.partition("=")
        if unit.strip().lower() != "bytes":
            return None
        s, _, e = rng.split(",")[0].partition("-")
        if s.strip() == "":
            ln = int(e)
            return (max(0, size - ln), size - 1) if ln > 0 else None
        start = int(s)
        end = min(int(e) if e.strip() else size - 1, size - 1)
        return (start, end) if start <= end and start < size else None
    except ValueError:
        return None


def _iter(path: Path, start: int, end: int):
    with path.open("rb") as f:
        f.seek(start)
        rem = end - start + 1
        while rem > 0:
            d = f.read(min(CHUNK, rem))
            if not d:
                break
            rem -= len(d)
            yield d


@app.get("/video/{cam}")
def video(cam: str, request: Request):
    path = RUN_DIR / f"demo_{cam}.mp4"
    if not path.exists():
        raise HTTPException(404)
    size = path.stat().st_size
    base = {"Accept-Ranges": "bytes"}
    rh = request.headers.get("range")
    if rh:
        rng = _parse_range(rh, size)
        if rng is None:
            return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})
        s, e = rng
        return StreamingResponse(_iter(path, s, e), status_code=206, media_type="video/mp4",
                                 headers={**base, "Content-Range": f"bytes {s}-{e}/{size}",
                                          "Content-Length": str(e - s + 1)})
    return StreamingResponse(_iter(path, 0, size - 1), media_type="video/mp4",
                             headers={**base, "Content-Length": str(size)})


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>LINECALL — Pickleball Line Caller</title>
<style>
 :root{--bg:#0b0e14;--card:#141a23;--card2:#1b2330;--edge:#28303d;--txt:#e8eef6;
   --dim:#8b97a5;--accent:#4f9cff;--green:#3fb950;--amber:#d29922;--red:#f85149}
 *{box-sizing:border-box}
 body{margin:0;padding:0;background:
   radial-gradient(1200px 500px at 50% -200px,#15203400,#0b0e14),var(--bg);
   color:var(--txt);font:14px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
 .wrap{max-width:1320px;margin:0 auto;padding:18px 22px 40px}
 header{display:flex;align-items:center;gap:14px;margin:6px 0 18px}
 .logo{width:42px;height:42px;border-radius:11px;display:flex;align-items:center;
   justify-content:center;font-size:22px;
   background:linear-gradient(135deg,#1f6feb,#3fb950);box-shadow:0 4px 18px #1f6feb44}
 header h1{font-size:21px;font-weight:700;margin:0;letter-spacing:.2px}
 header .sub{font-size:12.5px;color:var(--dim);margin-top:1px}
 header .pill{margin-left:auto;font-size:12px;color:var(--dim);background:var(--card);
   border:1px solid var(--edge);border-radius:20px;padding:6px 13px}
 .top{display:grid;grid-template-columns:1fr 1fr;gap:14px;max-width:1320px;margin:0 auto}
 .panel{position:relative;background:#000;border:1px solid var(--edge);border-radius:12px;
   overflow:hidden;aspect-ratio:16/9;box-shadow:0 6px 24px #00000040}
 .panel video{display:block;width:100%;height:100%;object-fit:contain}
 .panel canvas{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
 .tag{position:absolute;top:10px;left:10px;z-index:2;background:rgba(0,0,0,.6);
   backdrop-filter:blur(4px);padding:4px 11px;border-radius:7px;font-size:12px;
   font-weight:600;letter-spacing:.3px}
 #verdict{margin:16px 0;padding:18px;border-radius:14px;border:1px solid var(--edge);
   background:var(--card);font-size:34px;font-weight:800;text-align:center;min-height:46px;
   letter-spacing:1px;transition:background .15s,color .15s}
 #verdict.IN{background:#0f3a1d;color:var(--green);border-color:#1d6b34}
 #verdict.OUT{background:#43161a;color:var(--red);border-color:#7a2a2a}
 #verdict.CLOSE_IN{background:#3a2c10;color:var(--amber);border-color:#7a5a1a}
 .controls{display:flex;gap:9px;align-items:center;flex-wrap:wrap}
 button{background:var(--card2);color:var(--txt);border:1px solid #34465f;border-radius:9px;
   padding:9px 15px;font-size:13.5px;font-weight:500;cursor:pointer;transition:.12s}
 button:hover{background:#243349;border-color:var(--accent)}
 button.primary{background:linear-gradient(135deg,#1f6feb,#388bfd);border-color:#388bfd}
 button.primary:hover{filter:brightness(1.1)}
 .clock{margin-left:auto;color:var(--dim);font-variant-numeric:tabular-nums;font-size:13px}
 .stats{display:flex;gap:10px;margin:16px 0 10px;flex-wrap:wrap}
 .stat{flex:1;min-width:120px;background:var(--card);border:1px solid var(--edge);
   border-radius:11px;padding:12px 15px}
 .stat .k{font-size:11.5px;color:var(--dim);text-transform:uppercase;letter-spacing:.5px}
 .stat .v{font-size:22px;font-weight:700;margin-top:3px}
 .records{background:var(--card);border:1px solid var(--edge);border-radius:12px;
   overflow:hidden;margin-top:6px}
 .records summary{cursor:pointer;list-style:none;padding:13px 16px;font-weight:600;
   font-size:14px;display:flex;align-items:center;gap:8px;user-select:none}
 .records summary::-webkit-details-marker{display:none}
 .records summary .chev{transition:.2s;color:var(--dim)}
 .records[open] summary .chev{transform:rotate(90deg)}
 .records summary .count{margin-left:auto;color:var(--dim);font-weight:500;font-size:12.5px}
 .tablewrap{max-height:340px;overflow-y:auto;border-top:1px solid var(--edge)}
 table{width:100%;border-collapse:collapse;font-size:13px}
 thead th{position:sticky;top:0;background:var(--card2);text-align:left;
   padding:9px 16px;font-size:11.5px;color:var(--dim);text-transform:uppercase;
   letter-spacing:.5px;font-weight:600}
 tbody td{padding:9px 16px;border-top:1px solid #1f2733}
 tbody tr{cursor:pointer;transition:.1s}
 tbody tr:hover{background:#1a2230}
 .badge{display:inline-block;padding:2px 10px;border-radius:20px;font-size:11.5px;
   font-weight:700;letter-spacing:.3px}
 .badge.IN{background:#0f3a1d;color:var(--green)}
 .badge.OUT{background:#43161a;color:var(--red)}
 .tables{display:grid;grid-template-columns:1fr;gap:14px;max-width:900px;margin:0 auto}
 tbody tr.disagree{background:#3a2c10}
 tbody tr.disagree:hover{background:#46350f}
 tbody tr.disagree td:first-child::before{content:"⚠ ";color:var(--amber)}
 #disagree_banner{margin:0 0 14px}
 .dbar{background:#3a2c10;border:1px solid #7a5a1a;border-radius:11px;padding:11px 16px;
   color:var(--amber);font-size:13.5px;font-weight:500}
 .dbar b{color:#f0c060}
 .legend{font-size:12px;color:var(--dim);margin-top:14px;line-height:1.6}
</style></head><body>
<div class="wrap">
 <header>
   <div class="logo">🎾</div>
   <div>
     <h1>LINECALL</h1>
     <div class="sub">Multi-Camera 3D Pickleball Line Caller</div>
   </div>
   <div class="pill" id="pill">recorded demo</div>
 </header>

 <div class="top">
  <div class="panel"><span class="tag">CAM 1 — calls</span><video id="v1" muted playsinline></video><canvas id="o1"></canvas></div>
  <div class="panel"><span class="tag">CAM 2 — view</span><video id="v2" muted playsinline></video><canvas id="o2"></canvas></div>
 </div>

 <div id="verdict">—</div>

 <div class="controls">
  <button id="play" class="primary">▶ Play</button>
  <button id="restart">⟲ Restart</button>
  <button id="prevB">◀ Prev bounce</button>
  <button id="nextB">Next bounce ▶</button>
  <button id="slow">0.5×</button>
  <span class="clock" id="clock"></span>
 </div>

 <div class="stats" id="stats"></div>

 <div class="tables">
   <details class="records" open>
     <summary><span class="chev">▶</span> Camera 1 — Calls <span class="count" id="bcount1"></span></summary>
     <div class="tablewrap">
       <table>
         <thead><tr><th>#</th><th>Time</th><th>Call</th><th>Conf</th><th>Court (x, y)</th></tr></thead>
         <tbody id="brows1"></tbody>
       </table>
     </div>
   </details>
 </div>

 <div class="legend">Ball tracking, calibration and IN/OUT calls all from Camera 1. Click any row to jump to that call.</div>
</div>
<svg id="court" style="display:none"><g id="dots"></g><circle id="ballDot" visibility="hidden"/></svg>

<script>
"use strict";
const $=id=>document.getElementById(id);
const v1=$("v1"), v2=$("v2");
let D=null, lastBeeped=-1, slow=false;

// --- audio: same verdict scheme as linecall/bounce/sound.py, via WebAudio ---
const AC=new (window.AudioContext||window.webkitAudioContext)();
function tone(f,dur,at){const o=AC.createOscillator(),g=AC.createGain();
  o.frequency.value=f;o.connect(g);g.connect(AC.destination);
  g.gain.setValueAtTime(0,at);g.gain.linearRampToValueAtTime(.5,at+.008);
  g.gain.setValueAtTime(.5,at+dur-.012);g.gain.linearRampToValueAtTime(0,at+dur);
  o.start(at);o.stop(at+dur+.01);}
function beep(verdict){const t=AC.currentTime+.02;
  if(verdict==="OUT"){tone(300,.12,t);tone(300,.12,t+.16);}
  else {tone(880,.18,t);}}   // IN = single tone, OUT = low double

function rowsHTML(list){
  return list.map((b,i)=>
    `<tr class="${b.disagree?'disagree':''}" onclick="seek(${b.t})">`+
      `<td>${i+1}</td>`+
      `<td>${b.t.toFixed(2)}s</td>`+
      `<td><span class="badge ${b.verdict}">${b.verdict}</span></td>`+
      `<td>${b.confidence!=null?Math.round(b.confidence*100)+'%':'—'}</td>`+
      `<td>(${b.x_ft.toFixed(1)}, ${b.y_ft.toFixed(1)})</td></tr>`).join('');
}

async function boot(){
  D=await (await fetch('/api/data')).json();
  v1.src='/video/cam1';
  v2.src='/video/cam2';
  v2.onerror=()=>{v2.closest('.panel').style.display='none';
    document.querySelector('.top').style.gridTemplateColumns='1fr';};
  const cal=D.calibration;
  const cc=D.cam_calls||{cam1:[]};
  const c1=cc.cam1||[];

  // summary stat cards
  const ins=c1.filter(b=>b.verdict==='IN').length, outs=c1.length-ins;
  $("stats").innerHTML=[
    ['Calls', c1.length],
    ['IN', ins, 'var(--green)'],
    ['OUT', outs, 'var(--red)'],
    ['Calibration', `${cal.cam1.reproj_px?cal.cam1.reproj_px.toFixed(1):'—'} px`],
  ].map(([k,v,col])=>`<div class="stat"><div class="k">${k}</div>`+
    `<div class="v" style="color:${col||'var(--txt)'}">${v}</div></div>`).join('');

  $("bcount1").textContent=`${c1.length} calls`;
  $("brows1").innerHTML=rowsHTML(c1);

  // hidden court dots (overlay JS still references these)
  const dots=$("dots");
  D.bounces.forEach(b=>{
    const c=document.createElementNS('http://www.w3.org/2000/svg','circle');
    c.setAttribute('cx',b.x_ft);c.setAttribute('cy',b.y_ft);c.id='dot'+(b.ref_idx||b.t);
    dots.appendChild(c);
  });
}

// --- playback (single camera) ---
$("play").onclick=async()=>{AC.resume();
  if(!v1.paused){v1.pause();v2.pause();return;}
  await v1.play().catch(()=>{});v2.play().catch(()=>{});};
$("restart").onclick=()=>{v1.currentTime=0;v2.currentTime=0;lastBeeped=-1;};
$("slow").onclick=()=>{slow=!slow;v1.playbackRate=v2.playbackRate=slow?0.5:1;
  $("slow").textContent=slow?'1×':'0.5×';};
function seek(t){v1.currentTime=Math.max(0,t-1.5);lastBeeped=-1;}
window.seek=seek;
$("nextB").onclick=()=>{const b=D.bounces.find(b=>b.t>v1.currentTime+0.2);if(b)seek(b.t);};
$("prevB").onclick=()=>{const bs=D.bounces.filter(b=>b.t<v1.currentTime-2);if(bs.length)seek(bs[bs.length-1].t);};

// --- overlays ---
function trackAt(t){ // binary search nearest track point within 80ms
  const tr=D.track; let lo=0,hi=tr.length-1;
  while(lo<hi){const m=(lo+hi)>>1; tr[m].t<t?lo=m+1:hi=m;}
  const c=[tr[lo],tr[lo-1]].filter(Boolean)
    .sort((a,b)=>Math.abs(a.t-t)-Math.abs(b.t-t))[0];
  return c&&Math.abs(c.t-t)<0.08?c:null;
}
(function draw(){
  requestAnimationFrame(draw);
  if(!D)return;
  const t=v1.currentTime;
  // keep cam2 following cam1 (nudge, don't reseek every frame)
  if(v2.readyState>1){
    if(Math.abs(v2.currentTime-t)>0.15) v2.currentTime=t;
    if(!v1.paused&&v2.paused) v2.play().catch(()=>{});
  }
  $("clock").textContent=t.toFixed(2)+'s / '+(v1.duration||0).toFixed(0)+'s';
  {
    const v=v1, cv=$("o1"), key='c1';
    const ctx=cv.getContext('2d');
    if(cv.width!==v.clientWidth||cv.height!==v.clientHeight){cv.width=v.clientWidth||1;cv.height=v.clientHeight||1;}
    ctx.clearRect(0,0,cv.width,cv.height);
    if(v.videoWidth){
    const sx=cv.width/v.videoWidth, sy=cv.height/v.videoHeight;
    // particle trail — small uniform dots behind the ball (no connecting line).
    // Every dot is the same small size; only opacity fades with age.
    const TRAIL=0.5;  // seconds of trail
    for(const p of D.track){
      if(p.t>t)break; const age=t-p.t; if(age>TRAIL)continue;
      if(!p[key])continue;
      const [px,py]=p[key];
      const f=1-age/TRAIL;                     // 1 = now, 0 = oldest
      const a=(0.12+0.78*f).toFixed(3);        // opacity fades; radius stays fixed
      ctx.beginPath();ctx.arc(px*sx,py*sy,2.5,0,7);
      ctx.fillStyle='rgba(248,225,108,'+a+')';ctx.fill();
    }
    // bounce ring flash (0.6s after each bounce)
    for(const b of D.bounces){
      if(t<b.t||t>b.t+0.6)continue;
      const p=trackAt(b.t); if(!p||!p[key])continue;
      const [px,py]=p[key], k=(t-b.t)/0.6;
      ctx.beginPath();ctx.arc(px*sx,py*sy,10+k*30,0,7);
      ctx.strokeStyle=b.verdict==='IN'?'rgba(63,185,80,'+(1-k)+')':
        b.verdict==='OUT'?'rgba(248,81,73,'+(1-k)+')':'rgba(210,153,34,'+(1-k)+')';
      ctx.lineWidth=4;ctx.stroke();}
    }
  }
  // verdict + beeps. (No single minimap ball: the cameras are independent, so there is no
  // one shared 3D court position — each panel shows its own ball trail instead.)
  const bd=$("ballDot");
  if(bd)bd.setAttribute('visibility','hidden');
  let active=null;
  for(const b of D.bounces){
    const key=(b.camera_id||'')+'@'+b.t.toFixed(3);
    if(t>=b.t && t<b.t+2.2) active=b;
    if(!v1.paused&&t>=b.t&&t-b.t<0.25&&lastBeeped!==key){
      lastBeeped=key;beep(b.verdict);}
  }
  const vd=$("verdict");
  if(active){vd.textContent=active.confidence!=null?
      `${active.verdict} · ${Math.round(active.confidence*100)}% sure`:active.verdict;
    vd.className=active.verdict;}
  else{vd.textContent='—';vd.className='';}
})();
boot();
</script></body></html>"""


def main() -> int:
    global RUN_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None, help="runs/pipeline/<tag>; default = newest")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()
    base = REPO / "runs" / "pipeline"
    if args.tag:
        RUN_DIR = base / args.tag
    else:
        runs = sorted((d for d in base.glob("*") if (d / "calls.json").exists()),
                      key=lambda d: d.stat().st_mtime)
        if not runs:
            sys.exit("no pipeline runs found — run scripts/run_gridtracknet.py first")
        RUN_DIR = runs[-1]
    print(f"demo run: {RUN_DIR}")
    _ensure_demo_clips()
    print(f"demo -> http://localhost:{args.port}")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
