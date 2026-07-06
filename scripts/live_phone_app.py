#!/usr/bin/env python3
"""
Live two-phone camera feed over local WiFi — real-time dual-camera viewer.

Two phones each open /phone (pick CAM A / CAM B), grant camera access, and stream JPEG
frames to the laptop in real time. The laptop opens /view to see both angles side-by-side,
live, with per-camera connection + fps status cards.

IMPORTANT — browsers only allow camera access over a SECURE context. Plain http://<ip>
will NOT give camera permission on a phone. So this runs HTTPS with a self-signed cert
(auto-generated). Phones show a "not private" warning first time — tap Advanced -> proceed.

Run:  python scripts/live_phone_app.py
  -> prints the exact https://<laptop-ip>:8443 URLs for each phone + the laptop.
"""
from __future__ import annotations

import socket
import subprocess
import time
from collections import deque
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response

REPO = Path(__file__).resolve().parent.parent
CERT = REPO / "configs" / "live_cert.pem"
KEY = REPO / "configs" / "live_key.pem"
PORT = 8443

app = FastAPI()
FRAMES: dict[str, bytes] = {"cam1": b"", "cam2": b""}
STAMPS: dict[str, deque] = {"cam1": deque(maxlen=60), "cam2": deque(maxlen=60)}


def _lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def _ensure_cert() -> None:
    if CERT.exists() and KEY.exists():
        return
    CERT.parent.mkdir(parents=True, exist_ok=True)
    print("[cert] generating self-signed certificate (one-time)…")
    subprocess.run(
        ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
         "-keyout", str(KEY), "-out", str(CERT), "-days", "365",
         "-subj", "/CN=linecall-live"], check=True, capture_output=True)


@app.post("/upload/{cam}")
async def upload(cam: str, request: Request):
    if cam in FRAMES:
        FRAMES[cam] = await request.body()
        STAMPS[cam].append(time.time())
    return {"ok": True}


@app.get("/frame/{cam}")
def frame(cam: str):
    data = FRAMES.get(cam, b"")
    if not data:
        return Response(status_code=204)
    return Response(content=data, media_type="image/jpeg",
                    headers={"Cache-Control": "no-store"})


@app.get("/status")
def status():
    now = time.time()
    out = {}
    for cam, st in STAMPS.items():
        recent = [t for t in st if now - t <= 1.0]
        last = st[-1] if st else 0
        out[cam] = {"live": bool(st) and (now - last) < 2.0,
                    "fps": len(recent), "age_ms": int((now - last) * 1000) if st else None}
    return JSONResponse(out)


PHONE = """<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>LINECALL phone</title>
<style>
 body{margin:0;background:#0b0e14;color:#e8eef6;font:16px system-ui;text-align:center}
 video{width:100%;max-height:60vh;background:#000;object-fit:contain}
 .bar{padding:12px;display:flex;gap:10px;justify-content:center;flex-wrap:wrap}
 button{background:#1b2740;color:#e8eef6;border:1px solid #3a4f78;border-radius:9px;
   padding:13px 20px;font-size:16px}
 button.sel{background:#1d6b34;border-color:#3fb950}
 button.go{background:#214a8c;border-color:#4f9cff}
 #status{padding:12px;color:#8aa0c0;font-size:14px;min-height:20px}
</style></head><body>
<div class=bar>
 <button id=c1 class=sel>CAM A</button>
 <button id=c2>CAM B</button>
 <button id=go class=go>● Start streaming</button>
</div>
<video id=v autoplay playsinline muted></video>
<div id=status>1) pick CAM A or CAM B &nbsp; 2) Start &nbsp; 3) allow camera</div>
<canvas id=cv style=display:none></canvas>
<script>
let cam="cam1", timer=null;
const $=i=>document.getElementById(i);
$("c1").onclick=()=>{cam="cam1";$("c1").classList.add("sel");$("c2").classList.remove("sel");};
$("c2").onclick=()=>{cam="cam2";$("c2").classList.add("sel");$("c1").classList.remove("sel");};
$("go").onclick=async()=>{
  try{
    const stream=await navigator.mediaDevices.getUserMedia(
      {video:{facingMode:"environment",width:{ideal:1280}},audio:false});
    $("v").srcObject=stream;
    $("status").textContent="● LIVE — streaming "+(cam==="cam1"?"CAM A":"CAM B")+". keep screen on.";
    const cv=$("cv"), ctx=cv.getContext("2d"), v=$("v");
    if(timer)clearInterval(timer);
    timer=setInterval(()=>{
      if(!v.videoWidth)return;
      cv.width=640; cv.height=Math.round(640*v.videoHeight/v.videoWidth);
      ctx.drawImage(v,0,0,cv.width,cv.height);
      cv.toBlob(b=>{if(b)fetch("/upload/"+cam,{method:"POST",body:b}).catch(()=>{});},
                "image/jpeg",0.6);
    },90);
  }catch(e){$("status").textContent="camera blocked: "+e.message+
     " — did you accept the certificate warning? (must be https)";}
};
</script></body></html>"""

VIEW = """<!doctype html><html><head><meta charset=utf-8><title>LINECALL — live dual-camera</title>
<style>
 :root{--bg:#0b0e14;--card:#141a23;--edge:#28303d;--txt:#e8eef6;--dim:#8aa0c0;
   --green:#3fb950;--red:#f85149;--accent:#4f9cff}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--txt);font:14px system-ui;padding:20px}
 h1{font-size:21px;font-weight:700;margin:0;letter-spacing:.2px}
 .sub{color:var(--dim);font-size:12.5px;margin:2px 0 16px}
 .cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;max-width:1320px;
   margin:0 auto 16px}
 .card{background:var(--card);border:1px solid var(--edge);border-radius:11px;padding:13px 15px}
 .card .k{font-size:11px;letter-spacing:.6px;color:var(--dim);text-transform:uppercase}
 .card .v{font-size:22px;font-weight:700;margin-top:5px}
 .grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;max-width:1320px;margin:0 auto}
 .panel{position:relative;background:#000;border:1px solid var(--edge);border-radius:12px;
   overflow:hidden;aspect-ratio:16/10}
 .panel img{width:100%;height:100%;object-fit:contain;display:block}
 .tag{position:absolute;top:9px;left:9px;background:#000a;padding:3px 10px;border-radius:6px;
   font-size:12px;letter-spacing:.5px}
 .wait{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
   color:var(--dim);font-size:14px}
 .dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px;
   vertical-align:middle;background:var(--red)}
 .dot.on{background:var(--green);box-shadow:0 0 8px var(--green)}
</style></head><body>
<h1>LINECALL — live dual-camera</h1>
<div class=sub>two phones streaming over local WiFi · real-time · detection-seam ready</div>
<div class=cards>
 <div class=card><div class=k>Camera A</div><div class=v id=sa><span class="dot" id=da></span>waiting</div></div>
 <div class=card><div class=k>Camera A · fps</div><div class=v id=fa>—</div></div>
 <div class=card><div class=k>Camera B</div><div class=v id=sb><span class="dot" id=db></span>waiting</div></div>
 <div class=card><div class=k>Camera B · fps</div><div class=v id=fb>—</div></div>
</div>
<div class=grid>
 <div class=panel><span class=tag>CAM A</span><img id=cam1><div class=wait id=wa>waiting for phone A…</div></div>
 <div class=panel><span class=tag>CAM B</span><img id=cam2><div class=wait id=wb>waiting for phone B…</div></div>
</div>
<script>
const $=i=>document.getElementById(i);
function feed(cam,img,wait){
  const el=$(img);
  setInterval(()=>{el.src="/frame/"+cam+"?t="+Date.now();},90);
  el.onload=()=>{$(wait).style.display="none";};
}
feed("cam1","cam1","wa"); feed("cam2","cam2","wb");
async function poll(){
  try{
    const s=await (await fetch("/status")).json();
    for(const [cam,a,b,d] of [["cam1","sa","fa","da"],["cam2","sb","fb","db"]]){
      const st=s[cam];
      $(d).className="dot"+(st.live?" on":"");
      $(a).innerHTML=`<span class="dot${st.live?' on':''}"></span>`+(st.live?"LIVE":"waiting");
      $(b).textContent=st.live?st.fps+" fps":"—";
    }
  }catch(e){}
  setTimeout(poll,700);
}
poll();
</script></body></html>"""


@app.get("/phone", response_class=HTMLResponse)
def phone():
    return PHONE


@app.get("/view", response_class=HTMLResponse)
def view():
    return VIEW


@app.get("/", response_class=HTMLResponse)
def index():
    ip = _lan_ip()
    return f"""<!doctype html><html><head><meta charset=utf-8><title>LINECALL live setup</title>
<style>body{{background:#0b0e14;color:#e8eef6;font:16px system-ui;padding:24px;line-height:1.7}}
a{{color:#4f9cff}} code{{background:#141a23;padding:2px 7px;border-radius:5px}}</style></head><body>
<h2>LINECALL — live two-phone setup</h2>
<ol>
<li><b>Phone A:</b> open <code>https://{ip}:{PORT}/phone</code> → CAM A → Start</li>
<li><b>Phone B:</b> open <code>https://{ip}:{PORT}/phone</code> → CAM B → Start</li>
<li><b>Laptop:</b> open <a href="/view">the live viewer →</a></li>
</ol>
<p>Accept the "not private" certificate warning on each phone (self-signed, safe on your LAN).</p>
</body></html>"""


if __name__ == "__main__":
    _ensure_cert()
    ip = _lan_ip()
    print("\n=== LINECALL live dual-camera ===")
    print(f"  Phone A:  https://{ip}:{PORT}/phone   (pick CAM A)")
    print(f"  Phone B:  https://{ip}:{PORT}/phone   (pick CAM B)")
    print(f"  Laptop :  https://{ip}:{PORT}/view")
    print("  (accept the certificate warning on each phone)\n")
    uvicorn.run(app, host="0.0.0.0", port=PORT,
                ssl_keyfile=str(KEY), ssl_certfile=str(CERT), log_level="warning")
