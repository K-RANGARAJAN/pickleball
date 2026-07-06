# GridTrackNet Detector Upgrade

Replaces the YOLO ball detector (which locked onto stationary objects on our night footage)
with **GridTrackNet** — a temporal 5-frame ONNX tracker that traces the small, fast ball
cleanly. Everything downstream (sync, calibration, triangulation, bounce, IN/OUT) is unchanged.

## Why it's better

The old YOLO saw each frame in isolation and often fixated on a fixed object (a spare ball /
mark). GridTrackNet looks at **5 consecutive frames** and uses motion, so it follows the
moving ball. Verified on our real cam1 footage (frames 3000–3044): it returned a smooth,
physical trajectory with a clearly visible bounce (pixel-y rises then falls), e.g.
`(607,600) → (615,605) → (625,582)` — a ground contact at the pixel-y peak. That bounce maps
to a sensible court position and an IN call.

## What was added

| File | Role |
|---|---|
| `ball_tracker_handoff/` | The handoff bundle: `model_weights.onnx`, `ball_tracker.py`, README. |
| `linecall/detection/gridtracknet_detector.py` | `track_clip(...)` — runs GridTrackNet over a clip, returns one `Detection` per original frame. |
| `scripts/run_gridtracknet.py` | Full pipeline: GridTrackNet → per-camera 2D bounces (+3D where both co-detect) → IN/OUT/CLOSE_IN → demo format. |

## Key facts / constraints (from the model's README)

- **5-frame temporal window**: frames are processed in chunks of 5, in order. The wrapper
  handles chunking.
- **30 fps spacing**: the model was trained for ~30 fps. Our clips are 60 fps, so we feed
  **every other frame** (`fps_skip=2`) and map coordinates back to the original frame indices,
  keeping detections on the same timeline as sync/calibration.
- **(0,0) = no ball** → emitted as an empty `Detection` (the Kalman bridges gaps).
- **Single ball** (argmax over a 48×27 grid). Coordinates returned in original pixel space.
- **Do not modify** the two `cv2.cvtColor` calls in `ball_tracker.py` — that exact
  preprocessing matches how the model was trained.
- **Tennis-trained**: works on pickleball as a strong baseline (verified), but accuracy isn't
  guaranteed; fine-tune on labelled pickleball data later if needed.

## Performance note

CPU inference is ~170 ms/frame — fine for an offline recorded run, too slow for live. On the
**RTX 4060 with `--provider gpu`** (needs `onnxruntime-gpu` + CUDA 12 / cuDNN 9) it is far
faster and is the path to the real-time goal.

## How to run

```bash
cd Pickleball_LINECALL_model
pip install -r ball_tracker_handoff/requirements.txt   # onnxruntime(-gpu), numpy, opencv

# CPU (works anywhere; slow on the full clip — use a slice first):
python scripts/run_gridtracknet.py --provider cpu --start 3000 --stop 6000 --tag gridtracknet

# GPU (the 4060) — whole clip, fast:
python scripts/run_gridtracknet.py --provider gpu --tag gridtracknet

# then watch it:
python scripts/demo_app.py --tag gridtracknet     # http://localhost:8080
```

`--provider gpu` requires `CUDAExecutionProvider` to be available:
```bash
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```
If `CUDAExecutionProvider` isn't listed, install `onnxruntime-gpu` with CUDA 12 / cuDNN 9, or
fall back to `--provider cpu`.

## Honest status

- Wrapper + pipeline: built and verified — GridTrackNet produces clean ball trajectories and
  the bounce→IN/OUT chain produces correct calls on the real trajectory shape.
- Not yet run on the full multi-minute clip end-to-end (needs your machine / GPU; the dev
  sandbox can't run the model at length). That run is the verification step below.
