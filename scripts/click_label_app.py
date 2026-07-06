"""Click-to-label ball centers over a clip (manual point labeling).

Step through frames (A/D, slider, jump) and click the ball center; N marks no-ball. Saves
continuously to labels/click_labels.json as {cam: {frame: [x,y] or null}}. Serves frames
straight from the video, no pre-extraction.

Run:  python scripts/click_label_app.py [--cam cam1] [--labels-file ...]  ->  http://localhost:8079
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, Response

REPO = Path(__file__).resolve().parent.parent
CLIPS = {"cam1": REPO / "CAMERA 1.mp4", "cam2": REPO / "camera2_muted.mp4"}
LABELS_PATH = REPO / "labels" / "click_labels.json"
LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
BOX_PX = 18  # fixed box size (px) drawn around each clicked center for YOLO export

app = FastAPI()
_caps: dict[str, cv2.VideoCapture] = {}
_meta: dict[str, dict] = {}


def cap(cam: str) -> cv2.VideoCapture:
    if cam not in _caps:
        c = cv2.VideoCapture(str(CLIPS[cam]))
        _caps[cam] = c
        _meta[cam] = {"n": int(c.get(cv2.CAP_PROP_FRAME_COUNT)),
                      "w": int(c.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      "h": int(c.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                      "fps": c.get(cv2.CAP_PROP_FPS) or 60.0}
    return _caps[cam]


def load_labels() -> dict:
    if LABELS_PATH.exists():
        return json.loads(LABELS_PATH.read_text())
    return {}


def save_labels(d: dict) -> None:
    LABELS_PATH.write_text(json.dumps(d))


def _nav_windows() -> list:
    """cam2 bounce-window starts (mapped from cam1 labels) so the user jumps straight to
    the same bounce moments instead of hunting across 19k frames."""
    p = REPO / "labels" / "cam2_nav_windows.json"
    if not p.exists():
        return []
    data = json.loads(p.read_text())
    return [w["cam2"][0] for w in data.get("windows", [])]


@app.get("/api/meta")
def meta(cam: str = "cam1"):
    cap(cam)
    lab = load_labels().get(cam, {})
    nav = _nav_windows() if cam == "cam2" else []
    return {"cam": cam, **_meta[cam], "labeled": len(lab),
            "cams": list(CLIPS), "nav": nav}


@app.get("/frame/{cam}/{idx}.jpg")
def frame(cam: str, idx: int):
    c = cap(cam)
    c.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, fr = c.read()
    if not ok:
        return Response(status_code=404)
    ok, buf = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return Response(content=buf.tobytes(), media_type="image/jpeg")


@app.get("/api/label")
def get_label(cam: str, idx: int):
    lab = load_labels().get(cam, {})
    v = lab.get(str(idx), "unset")
    return {"cam": cam, "idx": idx, "point": v}


@app.post("/api/label")
async def set_label(req: Request):
    p = await req.json()
    cam, idx = p["cam"], str(p["idx"])
    d = load_labels()
    d.setdefault(cam, {})
    d[cam][idx] = p.get("point")  # [x,y] or None (no ball)
    save_labels(d)
    return {"ok": True, "labeled": len([1 for v in d[cam].values() if v])}


@app.post("/api/export")
def export():
    d = load_labels()
    ds = REPO / "data" / "dataset_click"
    import shutil
    if ds.exists():
        shutil.rmtree(ds)
    n_train = n_val = boxes = 0
    for cam in CLIPS:
        c = cap(cam)
        W, H = _meta[cam]["w"], _meta[cam]["h"]
        items = sorted(((int(f), v) for f, v in d.get(cam, {}).items()),
                       key=lambda x: x[0])
        if not items:
            continue
        cut = items[int(len(items) * 0.85)][0] if len(items) > 6 else 10 ** 9
        for f, pt in items:
            split = "val" if f >= cut else "train"
            (ds / "images" / split).mkdir(parents=True, exist_ok=True)
            (ds / "labels" / split).mkdir(parents=True, exist_ok=True)
            c.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, fr = c.read()
            if not ok:
                continue
            stem = f"{cam}_{f:06d}"
            cv2.imwrite(str(ds / "images" / split / f"{stem}.jpg"), fr)
            lbl = ds / "labels" / split / f"{stem}.txt"
            if pt:  # ball present -> one box centered on the click
                x, y = pt
                bw = bh = BOX_PX
                lbl.write_text(f"0 {x/W:.6f} {y/H:.6f} {bw/W:.6f} {bh/H:.6f}\n")
                boxes += 1
            else:   # no ball -> empty label (negative example)
                lbl.write_text("")
            if split == "val":
                n_val += 1
            else:
                n_train += 1
    (ds / "data.yaml").write_text(
        f"path: {ds}\ntrain: images/train\nval: images/val\nnames:\n  0: ball\n")
    return {"train": n_train, "val": n_val, "boxes": boxes, "dataset": str(ds)}


PAGE = """
<!doctype html><html><head><meta charset=utf-8><title>click-label</title>
<style>
 body{background:#0e1726;color:#e8eef7;font-family:system-ui;margin:0;padding:12px}
 #wrap{position:relative;display:inline-block;border:1px solid #2a3a55}
 #img{display:block;max-width:96vw;max-height:78vh}
 #dot{position:absolute;width:14px;height:14px;border:2px solid #ff3;border-radius:50%;
   transform:translate(-50%,-50%);pointer-events:none;box-shadow:0 0 0 1px #000;display:none}
 .bar{margin:10px 0;display:flex;gap:8px;align-items:center;flex-wrap:wrap}
 button,select,input{background:#1b2740;color:#e8eef7;border:1px solid #3a4f78;
   border-radius:7px;padding:7px 12px;font-size:14px;cursor:pointer}
 input[type=range]{width:60vw}
 .k{color:#8aa0c0;font-size:13px}
 #stat{font-size:13px;color:#9fd0a0}
 .noball{background:#3d1518;border-color:#7a2;}
</style></head><body>
<div class=bar>
 <select id=cam><option>cam1</option><option>cam2</option></select>
 <button id=prev>◀ prev (A)</button>
 <button id=next>next (D) ▶</button>
 <button id=back30>-30</button><button id=fwd30>+30</button>
 <button id=prevW>◀◀ prev bounce</button>
 <button id=nextW>next bounce ▶▶</button>
 <input id=jump type=number style=width:110px placeholder="frame #">
 <button id=go>go</button>
 <button id=noball class=noball>N = no ball</button>
 <button id=export>Export dataset</button>
 <span id=stat></span>
</div>
<input id=slider type=range min=0 value=0>
<div class=bar><span class=k>click the ball center to mark it • click again to move • marks save instantly</span></div>
<div id=wrap><img id=img><div id=dot></div></div>
<script>
let cam="__INITCAM__", idx=0, N=0, W=0, H=0, point=null, nav=[];
const $=id=>document.getElementById(id);
document.getElementById("cam").value=cam;
async function meta(){const m=await(await fetch(`/api/meta?cam=${cam}`)).json();
  N=m.n;W=m.w;H=m.h;nav=m.nav||[];$("slider").max=N-1;
  const hasNav=nav.length>0;
  $("prevW").style.display=hasNav?"":"none";$("nextW").style.display=hasNav?"":"none";
  $("stat").textContent=`${m.labeled} labeled • ${N} frames`+(hasNav?` • ${nav.length} bounce windows`:"");}
async function load(){
  $("img").src=`/frame/${cam}/${idx}.jpg`;
  $("jump").value=idx;$("slider").value=idx;
  const r=await(await fetch(`/api/label?cam=${cam}&idx=${idx}`)).json();
  point=(r.point&&r.point.length===2)?r.point:null;
  draw();
  $("stat").textContent=`frame ${idx}/${N-1}` + (point?` • ball @ ${point[0].toFixed(0)},${point[1].toFixed(0)}`:(r.point===null?" • NO BALL":" • unlabeled"));
}
function draw(){const img=$("img"),dot=$("dot");
  if(!point){dot.style.display="none";return;}
  const sx=img.clientWidth/W, sy=img.clientHeight/H;
  dot.style.left=(point[0]*sx)+"px";dot.style.top=(point[1]*sy)+"px";dot.style.display="block";}
async function setPoint(p){point=p;
  await fetch("/api/label",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({cam,idx,point:p})});
  draw();meta();}
$("img").onclick=e=>{const img=$("img"),r=img.getBoundingClientRect();
  const x=(e.clientX-r.left)/img.clientWidth*W, y=(e.clientY-r.top)/img.clientHeight*H;
  setPoint([x,y]);};
$("next").onclick=()=>{if(idx<N-1){idx++;load();}};
$("prev").onclick=()=>{if(idx>0){idx--;load();}};
$("fwd30").onclick=()=>{idx=Math.min(N-1,idx+30);load();};
$("back30").onclick=()=>{idx=Math.max(0,idx-30);load();};
$("go").onclick=()=>{idx=Math.max(0,Math.min(N-1,+$("jump").value||0));load();};
$("nextW").onclick=()=>{const w=nav.find(f=>f>idx+3);if(w!=null){idx=w;load();}};
$("prevW").onclick=()=>{const ws=nav.filter(f=>f<idx-3);if(ws.length){idx=ws[ws.length-1];load();}};
$("slider").oninput=e=>{idx=+e.target.value;load();};
$("noball").onclick=()=>setPoint(null);
$("cam").onchange=async e=>{cam=e.target.value;idx=0;await meta();load();};
$("export").onclick=async()=>{$("stat").textContent="exporting…";
  const j=await(await fetch("/api/export",{method:"POST"})).json();
  alert(`exported ${j.train} train / ${j.val} val images, ${j.boxes} ball boxes\n${j.dataset}`);};
window.onkeydown=e=>{if(e.target.tagName==="INPUT")return;
  if(e.key==="d"||e.key==="ArrowRight")$("next").click();
  else if(e.key==="a"||e.key==="ArrowLeft")$("prev").click();
  else if(e.key==="n"||e.key==="N")$("noball").click();};
(async()=>{await meta();load();})();
</script></body></html>
"""


INITIAL_CAM = "cam1"


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE.replace("__INITCAM__", INITIAL_CAM)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-file", default=str(LABELS_PATH),
                    help="where clicks are saved. Use labels/bounce_labels.json to mark "
                         "ONLY bounce-contact points without touching the trail labels.")
    ap.add_argument("--cam", default="cam1", choices=["cam1", "cam2"],
                    help="camera the UI opens on")
    ap.add_argument("--port", type=int, default=8079)
    args = ap.parse_args()
    LABELS_PATH = Path(args.labels_file)
    LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    INITIAL_CAM = args.cam
    print(f"click-label ({LABELS_PATH.name}, {INITIAL_CAM}) -> http://localhost:{args.port}")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
