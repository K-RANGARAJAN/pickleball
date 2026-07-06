#!/usr/bin/env python3
"""
Metric page — compare the model's IN/OUT calls against your manual ground-truth CSV.

Shows the model's predicted call per throw (from runs/metric/<session>/predictions.csv),
lets you upload your manual CSV, aligns them by throw order, and reports match % +
a confusion matrix. Sideline and baseline scored separately.

Your CSV: one row per throw in throw order, columns `throw,call` (call = IN | OUT | ON_LINE).

Run:  python scripts/metric_app.py     -> http://localhost:8081
"""
from __future__ import annotations

import csv
import io
from pathlib import Path

import uvicorn
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse

REPO = Path(__file__).resolve().parent.parent
METRIC = REPO / "runs" / "pipeline"  # not used; predictions live under runs/metric
PRED_DIR = REPO / "runs" / "metric"
SESSIONS = ["sideline", "baseline"]

app = FastAPI()


def load_predictions(session: str) -> list:
    p = PRED_DIR / session / "predictions.csv"
    if not p.exists():
        return []
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def _parse_time(s: str):
    """'mm:ss' or 'm:ss' or seconds -> float seconds; None if unparseable."""
    s = (s or "").strip()
    if not s:
        return None
    try:
        if ":" in s:
            parts = [float(p) for p in s.split(":")]
            return parts[-1] + 60 * parts[-2] + (3600 * parts[-3] if len(parts) > 2 else 0)
        return float(s)
    except (ValueError, IndexError):
        return None


def parse_truth(text: str) -> list:
    """Parse ground-truth CSV -> [{call, t}] in order. call uppercased ('ON LINE'->'ON_LINE');
    t is seconds from the timestamp column if present, else None."""
    rows = list(csv.DictReader(io.StringIO(text)))
    out = []
    for r in rows:
        c = (r.get("call") or r.get("Call") or r.get("verdict") or "").strip().upper().replace(" ", "_")
        if not c:
            continue
        t = _parse_time(r.get("Approx time") or r.get("approx time") or r.get("time") or r.get("t") or "")
        out.append({"call": c, "t": t})
    return out


def score(pred_rows: list, truth: list, tol_s: float = 3.0) -> dict:
    """Match each ground-truth throw to the model bounce NEAREST IN TIME (within tol_s),
    since the model over/under-detects so row-order alignment is wrong. Reports detection
    recall (did the model fire near the throw), IN/OUT accuracy on detected throws
    (CLOSE_IN counts as IN), extra detections, and a confusion matrix. Falls back to
    order-alignment only if no timestamps are present."""
    pred = [{"t": float(r.get("t_s") or 0), "call": (r.get("call") or "").strip().upper()}
            for r in pred_rows]
    have_time = any(x["t"] is not None for x in truth)
    cats_truth = ["IN", "OUT", "ON_LINE"]
    cats_pred = ["IN", "OUT", "CLOSE_IN", "UNKNOWN", "MISSED"]
    cm = {t: {p: 0 for p in cats_pred} for t in cats_truth}

    used = set()
    pairs = []  # (truth_call, pred_call_or_MISSED)
    if have_time:
        for tr in truth:
            tt = tr["t"]
            best = None
            for i, pc in enumerate(pred):
                if i in used or tt is None:
                    continue
                d = abs(pc["t"] - tt)
                if d <= tol_s and (best is None or d < best[1]):
                    best = (i, d)
            if best is not None:
                used.add(best[0])
                pairs.append((tr["call"], pred[best[0]]["call"]))
            else:
                pairs.append((tr["call"], "MISSED"))
    else:
        for i in range(min(len(pred), len(truth))):
            used.add(i)
            pairs.append((truth[i]["call"], pred[i]["call"]))

    detected = matched = considered = 0
    for tcall, pcall in pairs:
        if tcall in cm and pcall in cm[tcall]:
            cm[tcall][pcall] += 1
        if pcall != "MISSED":
            detected += 1
        if tcall in ("IN", "OUT") and pcall != "MISSED":
            considered += 1
            pp = "IN" if pcall == "CLOSE_IN" else pcall
            if pp == tcall:
                matched += 1

    return {
        "n_pred": len(pred), "n_truth": len(truth),
        "detected": detected, "missed": len(truth) - detected,
        "extra": len(pred) - len(used),
        "recall_pct": round(100 * detected / len(truth), 1) if truth else None,
        "matched": matched, "considered": considered,
        "match_pct": round(100 * matched / considered, 1) if considered else None,
        "by_time": have_time, "tol_s": tol_s,
        "confusion": cm, "cats_pred": cats_pred, "cats_truth": cats_truth,
    }


@app.get("/api/predictions")
def api_predictions():
    return JSONResponse({s: load_predictions(s) for s in SESSIONS})


@app.post("/api/score")
async def api_score(session: str = Form(...), file: UploadFile = None):
    text = (await file.read()).decode("utf-8", "replace")
    truth = parse_truth(text)
    return JSONResponse(score(load_predictions(session), truth))


PAGE = """<!doctype html><html><head><meta charset=utf-8><title>LINECALL — metric</title>
<style>
 :root{--bg:#0e1116;--card:#161b22;--edge:#2b3340;--txt:#e6edf3;--dim:#8b97a5;
   --green:#3fb950;--amber:#d29922;--red:#f85149;--accent:#4f9cff}
 body{margin:0;background:var(--bg);color:var(--txt);font:14px system-ui;padding:22px}
 h1{font-size:20px;margin:0 0 4px} .sub{color:var(--dim);font-size:13px;margin-bottom:18px}
 .grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}
 .card{background:var(--card);border:1px solid var(--edge);border-radius:12px;padding:16px}
 h2{font-size:15px;margin:0 0 10px}
 table{border-collapse:collapse;width:100%;font-size:12.5px} th,td{padding:5px 8px;text-align:left}
 thead th{color:var(--dim);border-bottom:1px solid var(--edge)}
 tbody tr:nth-child(even){background:#0f1722}
 .badge{padding:1px 7px;border-radius:6px;font-weight:600;font-size:11px}
 .IN{background:#0f3a1d;color:var(--green)} .OUT{background:#43161a;color:var(--red)}
 .CLOSE_IN{background:#3a2c10;color:var(--amber)} .UNKNOWN{background:#222;color:var(--dim)}
 .ON_LINE{background:#10283a;color:var(--accent)}
 .big{font-size:34px;font-weight:800;margin:6px 0} .pct{color:var(--green)}
 input,button{background:#1b2740;color:var(--txt);border:1px solid #3a4f78;border-radius:7px;
   padding:7px 11px;font-size:13px}
 button{cursor:pointer} .scroll{max-height:340px;overflow:auto}
 .cm td,.cm th{text-align:center} .warn{color:var(--amber);font-size:12.5px;margin-top:8px}
 .note{color:var(--dim);font-size:12px;margin-top:8px;line-height:1.5}
</style></head><body>
<h1>Line-call accuracy — model vs. your manual calls</h1>
<div class=sub>First bounce of each throw. Upload your CSV (columns <b>throw,call</b>; call = IN / OUT / ON_LINE) per session.</div>
<div class=grid id=grid></div>
<script>
const $=s=>document.querySelector(s);
let DATA={};
function badge(c){return `<span class="badge ${c||'UNKNOWN'}">${c||'—'}</span>`;}
function predTable(rows){
  if(!rows.length) return '<div class=note>No predictions yet. Run scripts/run_metric.py for this session.</div>';
  return `<div class=scroll><table><thead><tr><th>#</th><th>t (s)</th><th>court (x,y)</th><th>call</th></tr></thead><tbody>`+
    rows.map(r=>`<tr><td>${r.throw}</td><td>${r.t_s}</td><td>(${r.court_x||'—'}, ${r.court_y||'—'})</td><td>${badge(r.call)}</td></tr>`).join('')+
    `</tbody></table></div>`;
}
function cmTable(s){
  const rt=s.cats_truth, cp=s.cats_pred;
  let h='<table class=cm><thead><tr><th>truth ↓ / pred →</th>'+cp.map(c=>`<th>${badge(c)}</th>`).join('')+'</tr></thead><tbody>';
  rt.forEach(t=>{h+=`<tr><th>${badge(t)}</th>`+cp.map(p=>`<td>${s.confusion[t][p]||''}</td>`).join('')+'</tr>';});
  return h+'</tbody></table>';
}
function render(){
  $('#grid').innerHTML=SESSIONS.map(sess=>`
    <div class=card>
      <h2>${sess.toUpperCase()}</h2>
      <div id=pred_${sess}>${predTable(DATA[sess]||[])}</div>
      <div style="margin:14px 0 6px">
        <input type=file id=file_${sess} accept=".csv">
        <button onclick="scoreSess('${sess}')">Compare</button>
      </div>
      <div id=res_${sess}></div>
    </div>`).join('');
}
async function scoreSess(sess){
  const f=$('#file_'+sess).files[0]; if(!f){alert('choose your CSV first');return;}
  const fd=new FormData(); fd.append('session',sess); fd.append('file',f);
  const s=await (await fetch('/api/score',{method:'POST',body:fd})).json();
  const mode = s.by_time ? `matched by time (±${s.tol_s}s)` : 'matched by row order (no timestamps)';
  $('#res_'+sess).innerHTML=
    `<div class=big><span class=pct>${s.match_pct==null?'—':s.match_pct+'%'}</span>`+
    ` <span style="font-size:14px;color:var(--dim)">IN/OUT correct on ${s.considered} detected throws (${s.matched} right)</span></div>`+
    `<div class=note><b>Detection:</b> model found ${s.detected}/${s.n_truth} of your throws `+
    `(${s.recall_pct}% recall), missed ${s.missed}, plus ${s.extra} extra detections not matching any throw. ${mode}.</div>`+
    cmTable(s)+
    `<div class=note>CLOSE_IN counts as IN. MISSED = no model bounce within ±${s.tol_s}s of your throw. `+
    `ON_LINE rows show how the model handled the literal edge cases.</div>`;
}
const SESSIONS=["sideline","baseline"];
(async()=>{DATA=await (await fetch('/api/predictions')).json(); render();})();
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


if __name__ == "__main__":
    print("metric -> http://localhost:8081")
    uvicorn.run(app, host="127.0.0.1", port=8081, log_level="warning")
