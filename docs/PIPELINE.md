# Current Pipeline — Single-Camera GridTrackNet (cam1 only)

This document describes the active pipeline as of June 2026. The earlier multicam/3D
architecture is preserved in `docs/archive/`; it is no longer the live path.

## What the system does

One phone films the pickleball court. GridTrackNet (ONNX temporal tracker) finds the ball
in each frame, a ground homography maps pixels to court feet, the bounce detector finds
ground contacts, and each bounce is called IN / OUT / CLOSE_IN with a beep.

## Entrypoints

```
# One-time calibration (click 4 court corners)
python scripts/corner_click_app.py        # -> localhost:8078 -> configs/camera_corners.json

# Detection + bounce calls
python scripts/run_gridtracknet.py --provider coreml --tag gridtracknet
#   (provider: cpu | gpu for CUDA on the 4060 | coreml for Mac GPU/Neural Engine)
#   writes runs/pipeline/gridtracknet/{calls.json, track.csv}

# Playback UI
python scripts/demo_app.py --tag gridtracknet   # -> localhost:8080
```

## Pipeline stages

```
CAMERA 1.mp4
  A. CALIBRATION   scripts/corner_click_app.py (localhost:8078)
       Human drags 4 lines onto outer court -> corners_from_edges intersects them ->
       solve_camera_pose_auto tries all 8 corner relabelings, keeps physically valid pose
       (camera above ground, reproj <= 8px). Saves configs/camera_corners.json.

  B. DETECTION     linecall/detection/gridtracknet_detector.py :: track_clip()
       Wraps ball_tracker_handoff/ball_tracker.py (ONNX GridTrackNet, 5-frame temporal).
       Provider: cpu | gpu (CUDA, RTX 4060) | coreml (Mac GPU/Neural Engine).
       Returns list[Detection(camera_id, frame_idx, px_x, px_y, conf, source)].

  C. BOUNCE + CALL linecall/bounce/bounce_2d.py :: detect_bounces_2d()
       Pixel -> court feet via ground homography from the 4 calibrated corners.
       Bounce = ground contact (direction reversal in the homography-mapped trajectory).
       IN / OUT / CLOSE_IN vs court lines; per-side suppression (one call per rally side;
       net dribbles ignored).

  D. UI            scripts/demo_app.py (localhost:8080)
       Single video panel + calls table + particle-dot ball trail + WebAudio beeps
       (IN = 880 Hz single, OUT = 300 Hz double).
       Reads runs/pipeline/<tag>/{calls.json, track.csv}.
```

## Key files

| Role | File |
|---|---|
| Calibration | `linecall/calibration/pose.py`, `linecall/calibration/line_fit.py` |
| Detection | `linecall/detection/gridtracknet_detector.py`, `linecall/detection/base.py` |
| ONNX tracker | `ball_tracker_handoff/` (do not edit) |
| Bounce / calls | `linecall/bounce/bounce_2d.py`, `linecall/bounce/bounce.py` |
| Sound | `linecall/bounce/sound.py` |
| Configs | `configs/camera_corners.json`, `configs/court_geometry.json` |

## Live runner (secondary, not yet updated to GridTrackNet)

`scripts/live_runner.py` + `linecall/live/online_pipeline.py` implement real-time
streaming but still use the OLDER stack (YOLO detector + triangulation + 3D Kalman).
They are kept intact as priority #2. Migrating the live runner to the GridTrackNet 2D
path is the logical next step once detection quality is validated.

## Status

| Stage | State |
|---|---|
| Calibration (4 corners -> pose, solvePnP) | **WORKS** — cam1 reproj ~4.9 px |
| GridTrackNet detection on cam1 | **WORKS on recorded clip** — live not yet validated |
| 2D bounce + IN/OUT (homography path) | **WORKS** |
| Distinct beeps per verdict | **WORKS** |
| Live detection on this court in real time | **NOT DONE** — active goal |

Live detection is the remaining gap: GridTrackNet has not been validated on live footage
from this court. Do not mark this "working" until it has been tested end-to-end live.

---

## Two pages (June 17 2026): two-angle demo + controlled-throw accuracy metric

The system now has two browser layouts, run independently.

### 1. Two-angle demo — `scripts/demo_app.py` (port 8080)
Recorded play, both camera angles shown side-by-side; **ball tracking + bounce calls come
from cam1 only** (cam2 is a synced second view). Calls + trail read from a
`runs/pipeline/<tag>/` run (e.g. `--tag final` = hand-labeled cam1 bounces, `--tag twocam`
= two-camera merge). cam2 video is transcoded at the manual start offset (0) and nudged to
follow cam1. Both `demo_cam1.mp4`/`demo_cam2.mp4` are cached after first transcode.

### 2. Controlled-throw metric — `scripts/metric_app.py` (port 8081)
Measures IN/OUT accuracy against a manual ground-truth CSV on the throw-test clips
(`sideline.mp4`, `baseline.mp4`, one fixed camera, 1080p60). Pipeline:

```
sideline.mp4 / baseline.mp4
  A. scripts/track_to_csv.py    GridTrackNet -> runs/metric/<session>/ball.csv (Frame,Visibility,X,Y)
  B. scripts/run_metric.py      friend's bounce_detection_handoff/bounce_detector.detect_bounces
                                 (pixel-y maxima) + OUR ground homography -> IN/OUT/CLOSE_IN;
                                 then cluster bounces into THROWS by time gap, keep the FIRST
                                 bounce of each throw -> runs/metric/<session>/predictions.csv
  C. scripts/metric_app.py      show predictions; upload your manual CSV (throw,call with
                                 call = IN|OUT|ON_LINE); align by throw order; match % +
                                 confusion matrix. CLOSE_IN counts as IN; ON_LINE kept separate.
```

**Calibration is the one human step** for trustworthy IN/OUT: run `scripts/corner_click_app.py`,
pick cam `metric`, trace the 4 court edges (writes `configs/camera_corners.json["metric"]`),
then re-run `run_metric.py`. Until calibrated, calls are `UNKNOWN` (pipeline still runs).
The friend's detector is pure-numpy and only consumes the tracker's `(x,y)`-per-frame output.

Status (June 17 2026): tracker run on both throw clips (sideline 11k, baseline 18k ball
frames); first-bounce-per-throw gives 216 / 264 throws; demo + metric pages serve and the
scorer is validated. Real IN/OUT match % is pending the one-time metric-camera calibration.
