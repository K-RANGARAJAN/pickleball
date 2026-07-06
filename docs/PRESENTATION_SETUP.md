# Presentation setup — 4 pages

Run each in its **own terminal** (venv active). Each prints its URL. Open in the browser.

| # | Page | Command | URL |
|---|------|---------|-----|
| 1 | **Calibration** | `python scripts/corner_click_app.py` | http://localhost:8078 |
| 2 | **Metric (accuracy)** | `python scripts/metric_app.py` | http://localhost:8081 |
| 3 | **Recorded demo (2-cam)** | `python scripts/demo_app.py --tag final` | http://localhost:8080 |
| 4 | **Live two-phone feed** | `python scripts/live_phone_app.py` | https://<laptop-ip>:8443/view |

Before launching, if a port says "address already in use":
`pkill -f corner_click_app` / `demo_app` / `metric_app` / `live_phone_app`.

---

## 1. Calibration page (:8078)
Pick a camera from the dropdown, drag the 4 court edges, watch the reproj error, save.
Talking point: "we click the court corners once → a homography maps pixels to court feet."

## 2. Metric page (:8081)
Shows the model's per-throw predictions. Upload `sideline.csv` (under SIDELINE) and
`baseline.csv` (under BASELINE) → match % + detection recall + confusion matrix.
Numbers to call out: **96.2% recall (sideline), 97.8% (baseline)**, matched by timestamp.
If upload errors with `python-multipart`: `pip install python-multipart` in this venv.

## 3. Recorded demo (:8080)
Two camera angles of real play; bounce calls from cam1 (IN/OUT + beeps + ball trail).
`--tag final` = clean hand-labeled calls (the version that works). Hit ▶ Play.
Backup copy of this exact run is in `runs/pipeline/final_BACKUP/` if anything is touched.

## 4. Live two-phone feed (:8443, HTTPS) — the stretch page
Two phones stream their cameras to the laptop over local WiFi; laptop shows both live.

**Setup order (all 3 devices on the SAME WiFi):**
1. Start the app on the laptop. It prints `https://<ip>:8443/...` (e.g. `192.168.1.4`).
2. **Phone 1:** open `https://<ip>:8443/phone` → accept the certificate warning
   (Advanced → Proceed) → pick **CAM 1** → **Start streaming** → allow camera.
3. **Phone 2:** same URL → pick **CAM 2** → Start.
4. **Laptop:** open `https://<ip>:8443/view` → both feeds appear side-by-side.
Keep each phone's screen ON and on that tab while streaming.

**Two things that can break it (know these):**
- **Camera needs HTTPS** — that's why it's a self-signed cert; you MUST accept the warning
  on each phone or the camera won't turn on. (iOS Safari is the fussiest; if it refuses,
  use Android Chrome, or trust the cert in iOS settings.)
- **WiFi client isolation** — some venue/guest networks block phone↔laptop. If `/view`
  stays black, switch all 3 devices to a **phone hotspot** or your own router.

**Fallback:** if the live page won't cooperate in the room, the **recorded 2-cam demo
(page 3) shows the exact same two-angle idea reliably** — lead with that, present live as
"and here it is running off two phones in real time" only once you've confirmed it connects.

---

## One-line port summary
8078 calibration · 8080 recorded demo · 8081 metric · 8443 live (https)
