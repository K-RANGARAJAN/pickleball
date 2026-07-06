# Reusing GridTrackNet ball tracking in another project (ONNX / Route B)

This bundle lets you take **only the ball-tracking part** of GridTrackNet and run it
inside another repo (e.g. your pickleball project) with **no TensorFlow and no GridTrackNet
source code** — just an ONNX model file plus one small Python module.

### Files in this bundle
| File | Role |
|------|------|
| `ball_tracker.py` | Self-contained `BallTracker` class (onnxruntime + numpy + opencv only). |
| `example_usage.py` | Shows how to call it over a video and where to fuse with your detector. |
| `requirements.txt` | The only dependencies the other repo needs. |
| `README_INTEGRATION.md` | This guide. |

> ⚠️ **Read this first — tennis vs. pickleball.** The shipped weights were trained on
> **tennis** footage. A pickleball is a similar small, fast object, so the tracker often
> works out-of-the-box, but accuracy is **not guaranteed** (different ball colour/size,
> court, occlusions). Treat the transferred model as a strong baseline; validate on your
> clips and plan to **fine-tune/retrain on labelled pickleball data** if you need
> production accuracy. The architecture and the whole training pipeline ship with the
> GridTrackNet repo (`Train.py`, `DataGen.py`, `LabellingTool.py`).

---

## Step 1 — Export the model to ONNX (do this once, in the GridTrackNet repo)

The `.h5` is TensorFlow weights-only — not portable on its own. Convert it to a
self-contained `.onnx` (architecture **+** weights in one file):

```powershell
cd D:\GridTracknet_trial\GridTrackNet
uv run export_onnx.py --model_path model_weights.h5 --output_path model_weights.onnx
```

This produces **`model_weights.onnx`**. Its I/O contract (good to know, but
`ball_tracker.py` handles it for you):

- **Input**: name `input`, shape `(N, 15, 432, 768)`, `float32`, values scaled to `0..1`.
  Channels-first; the 15 channels are 5 frames x 3 (R,G,B-ish) planes.
- **Output**: shape `(N, 15, 27, 48)` — a 48x27 grid x 15 channels = 5 frames x
  (confidence, x-offset, y-offset).
- **Opset** 17, batch dimension dynamic.

---

## Step 2 — What to actually send to the other repo

Only **two** things are required:

1. **`model_weights.onnx`** (from Step 1) — the model.
2. **`ball_tracker.py`** (this bundle) — the loader + preprocessing + grid decode.

You do **NOT** need: `model_weights.h5`, `GridTrackNet.py`, TensorFlow, Keras, or anything
else from the original repo. `example_usage.py` and `requirements.txt` are helpers.

Suggested layout in your pickleball repo:
```
your-pickleball-repo/
|-- models/
|   |-- model_weights.onnx        <- copied from Step 1
|-- ball_tracker.py               <- copied from this bundle
|-- your_pipeline.py              <- your code, imports BallTracker
```

### Transferring the .onnx file
The ONNX file is a binary of roughly the same size as the `.h5` (tens of MB). Options:
- **Just copy it** into `models/` (simplest).
- **Commit to git**: fine if it's under ~50 MB; for anything larger use **Git LFS**
  (`git lfs track "*.onnx"`) so you don't bloat normal git history. Many teams keep model
  binaries **out** of git entirely and fetch them from a release asset / cloud bucket.
- **Share via cloud** (Drive, S3, release attachment) and download at setup time.

---

## Step 3 — Install dependencies in the other repo

```bash
pip install -r requirements.txt
```
That installs `onnxruntime-gpu` (for your RTX 4060), `numpy`, `opencv-python`. For GPU you
also need the **CUDA 12 runtime + cuDNN 9** available on PATH (same requirement as the ONNX
run in the GridTrackNet repo). No CUDA? Switch `requirements.txt` to plain `onnxruntime` and
construct the tracker with `provider="cpu"`.

Verify the GPU is visible:
```bash
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
# want: CUDAExecutionProvider in the list
```

---

## Step 4 — Integrate (the actual code)

```python
import cv2
from ball_tracker import BallTracker

tracker = BallTracker("models/model_weights.onnx", provider="gpu")  # or "cpu"

cap = cv2.VideoCapture("clip.mp4")
buffer = []
while True:
    ok, frame = cap.read()
    if not ok:
        break
    buffer.append(frame)
    if len(buffer) == 5:                      # model works in groups of 5 frames
        coords = tracker.predict(buffer)      # -> [(x, y), ...] one per frame; (0,0)=none
        for f, (x, y) in zip(buffer, coords):
            if (x, y) != (0, 0):
                cv2.circle(f, (x, y), 6, (0, 0, 255), -1)
            # fuse (x, y) with your pickleball detection here
        buffer = []
cap.release()
```

`predict()` contract:
- **Input**: a list of consecutive frames whose length is a multiple of 5 (a trailing
  remainder < 5 is ignored). Pass OpenCV (BGR) frames with the default `is_bgr=True`.
- **Output**: a list of `(x, y)` tuples, one per frame, in the **original frame's pixel
  coordinates** (so you can overlay directly). `(0, 0)` = no ball detected.

See `example_usage.py` for a complete runnable version.

---

## Input requirements & gotchas (carry over from GridTrackNet)

- **Process in chunks of 5 frames**, in order. Temporal order matters — the model uses
  motion across the 5 frames. Don't shuffle.
- **Frame rate**: the model was trained for ~30 FPS spacing. For **60 FPS** input, feed
  every other frame (skip 1) to match, exactly as `Predict.py` does; otherwise motion cues
  are too small. 30 FPS: feed as-is.
- **Resolution**: frames are resized to 768x432 internally, so any size works, but
  >= 1280x720 source gives best results. Output coordinates are rescaled back to your
  original frame size automatically.
- **Colour**: pass BGR frames (OpenCV default) with `is_bgr=True`. The two-step colour
  conversion in `predict()` is intentional and matches the trained model — don't "simplify"
  it or results will drift.
- **One ball only**: GridTrackNet returns a single best location per frame (argmax over the
  grid). It is not multi-object; for multiple balls you'd need a different approach.

## Verifying you got identical predictions
Run the GridTrackNet repo's `inference_onnx.py` on a clip and keep its `ball.csv`. Run this
module on the same clip and compare the `(x, y)` per frame — they should match. (Use the
same FPS handling on both sides.)

## Performance
On the RTX 4060 with `CUDAExecutionProvider`, inference is far faster than the ~6.5 FPS
you saw with CPU TensorFlow. The first `predict()` call is slower (CUDA/session warm-up);
steady-state is what matters.
