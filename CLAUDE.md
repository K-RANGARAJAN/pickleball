# Pickleball LINECALL

> Coding rules live in `.claude/rules/` — read them before writing code, especially
> `code-style.md` (write FEW comments) and `web-ui.md` (single-file FastAPI + inline HTML/JS).

## 1. What this is

A phone camera watches a pickleball court. We track the ball (GridTrackNet), map each
detection to court feet through a **ground homography** from 4 clicked court corners, find
the **bounce** (the ball's lowest point on screen = ground contact), and call **IN / OUT**
against the court lines. The system is deliberately mostly **geometry**, not ML — only the
ball tracker is a neural net, because geometry is exact and debuggable.

## 2. The 4 working flows (each a single-file FastAPI page)

| Flow | Script | Port | What it does |
|---|---|---|---|
| Calibration | `corner_click_app.py` | 8078 | drag the 4 court edges → `configs/camera_corners.json` |
| Metric/accuracy | `metric_app.py` + `run_metric.py` + `track_to_csv.py` | 8081 | model calls vs a manual ground-truth CSV, matched by timestamp → recall + IN/OUT % + confusion matrix |
| Recorded 2-cam demo | `demo_app.py` + `run_from_labels.py` / `run_two_cam_labels.py` | 8080 | two synced angles, cam1 ball trail + IN/OUT + beeps |
| Live two-phone | `live_phone_app.py` | 8443 (https) | two phones stream cameras to the laptop over local WiFi, shown live |

Producers: `run_gridtracknet.py` (automated cam1 calls), `click_label_app.py` (hand labels).
Full run/runbook: `docs/PRESENTATION_SETUP.md`. Pipeline detail: `docs/PIPELINE.md`.

## 3. The pipeline (per camera)

```
clip / phone frames
  A. DETECTION   GridTrackNet (ONNX, 5-frame temporal) -> ball pixel per frame
  B. CALIBRATION 4 clicked court corners -> ground homography (pixel -> court feet)
  C. BOUNCE      pixel-y local maximum (lowest on screen = ground contact)
  D. IN/OUT      bounce (x,y) in feet vs court lines -> IN / OUT / CLOSE_IN (+ confidence)
```

The friend's pure-numpy bounce detector (`bounce_detection_handoff/bounce_detector.py`) is
used by the metric flow; our `linecall/bounce/bounce_2d.py` is used by the demo flows.

## 4. Court coordinate system (world frame, feet)

Ground z = 0, height +z. Court 20 ft wide (x: 0→20) × 44 ft long (y: 0 far → 44 near).
Net at y = 22. A ball on or inside a line is IN. Definitions: `configs/court_geometry.json`.

## 5. Repository map

```
linecall/
  detection/base.py                  Detection dataclass + BallDetector ABC
  detection/gridtracknet_detector.py track_clip(): wraps ball_tracker_handoff/ (ONNX)
  calibration/pose.py                4 corners -> CameraPose (solvePnP) + ALL_WORLD_POINTS
  calibration/line_fit.py            fit court edges, corners = edge intersections
  bounce/bounce.py                   court geometry + classify_bounce + 3D detector
  bounce/bounce_2d.py                2D bounce detection, homography, IN/OUT, confidence, merge
scripts/                            the 10 entry points (4 flows + producers, see table)
ball_tracker_handoff/               GridTrackNet ONNX model + wrapper (external, do NOT edit)
bounce_detection_handoff/           friend's bounce_detector.py (external, do NOT edit)
configs/   camera_corners.json, court_geometry.json, live_cert/key.pem
labels/    click_labels.json (trail), bounce_labels.json (hand-marked bounces)
runs/      outputs (gitignored): runs/pipeline/<tag>/, runs/metric/<session>/
tests/     4 suites (bounce_2d, calibration, calibration_named, line_fit) — all pass
docs/      PRESENTATION_SETUP.md (runbook), PIPELINE.md
```

## 6. Key conventions

- Pixel vs world space is explicit in names: `px_x/px_y` vs `court_x/court_y` (or `*_ft`).
- A `Detection` = `(camera_id, frame_idx, px_x, px_y, confidence, source)`; an empty one =
  source `"none"`, NaN coords.
- `solve_camera_pose_named` picks the physically valid pose (camera above ground, z > 0).
- Clips: `CAMERA 1.mp4` + `camera2_muted.mp4` (demo), `sideline.mp4` + `baseline.mp4` (metric);
  all `*.mp4` are gitignored.
- Run UI servers in a real terminal (anything launched from an agent session gets reaped).

## 7. Honest status

- Detection + bounce on this court: **~97% bounce-detection recall** measured against 337
  hand-thrown, hand-labeled bounces (sideline 96.2%, baseline 97.8%).
- IN/OUT accuracy: gated on calibrating the metric camera (homography), then it's geometry.
- Two-camera 3D triangulation (true height, kills the ground-vs-air ambiguity) is the future
  scope; the original 3D/YOLO scaffolding was removed to keep the repo to what works today.
