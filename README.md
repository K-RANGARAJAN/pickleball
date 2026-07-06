# Pickleball LINECALL

Automatic pickleball line-calling from phone video. The ball is tracked with **GridTrackNet**
(a temporal ONNX tracker), each detection is mapped to court feet through a **ground
homography** from 4 clicked court corners, the **bounce** is the ball's lowest point on
screen, and the bounce position vs. the court lines gives the **IN / OUT** call.

Only the ball tracker is a neural net — everything else is exact geometry.

## The 4 pages

| Page | Command | Open |
|---|---|---|
| Calibration | `python scripts/corner_click_app.py` | http://localhost:8078 |
| Metric / accuracy | `python scripts/metric_app.py` | http://localhost:8081 |
| Recorded 2-cam demo | `python scripts/demo_app.py --tag final` | http://localhost:8080 |
| Live two-phone | `python scripts/live_phone_app.py` | https://&lt;laptop-ip&gt;:8443/view |

Full setup + phone/WiFi notes: **`docs/PRESENTATION_SETUP.md`**. Pipeline detail: `docs/PIPELINE.md`.

## Quick start

```bash
pip install -r requirements.txt
python -m pytest -q          # or: for t in tests/test_*.py; do python "$t"; done

# Calibrate, then run the recorded demo:
python scripts/corner_click_app.py     # drag the 4 court edges, save
python scripts/demo_app.py --tag final # -> http://localhost:8080
```

## Accuracy (measured on this court)

Against 337 hand-thrown, hand-labeled bounces (timestamp-matched): **~97% bounce-detection
recall** (sideline 96.2%, baseline 97.8%). IN/OUT accuracy follows from calibrating the
camera. Upload your ground-truth CSV in the metric page to reproduce.

## Layout

```
linecall/      detection (GridTrackNet wrapper), calibration (pose + homography), bounce (2D + IN/OUT)
scripts/       the 10 entry points (4 flows + producers)
ball_tracker_handoff/      GridTrackNet ONNX model + wrapper (external)
bounce_detection_handoff/  friend's bounce detector (external)
configs/ labels/ runs/ tests/ docs/
```

## Next step

Two-camera 3D triangulation — recover the ball's true height to resolve the single-camera
ground-vs-air ambiguity on the closest calls.
