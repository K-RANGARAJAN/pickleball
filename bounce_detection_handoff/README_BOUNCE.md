# Bounce detection from ball-tracking output

This module turns the per-frame ball positions from GridTrackNet into **bounce events**
— the moments the ball hits the ground — and (optionally) classifies each as
**IN / OUT / CLOSE_IN** in real court coordinates.

It is **pure numpy** (no OpenCV, ONNX, or TensorFlow) and plugs onto the *output* of the
tracker, so it doesn't touch the model. Drop the folder into your project and import it.

### Files
| File | Role |
|------|------|
| `bounce_detector.py` | The whole pipeline (functions + `detect_bounces`). |
| `example_usage.py` | End-to-end example: `ball.csv` -> bounces -> `bounces.csv`. |
| `requirements.txt` | Just `numpy`. |
| `README_BOUNCE.md` | This file. |

---

## How it connects to ball tracking
The tracker gives you one `(x, y)` per frame, where `(0, 0)` means "no ball this frame".
Feed that straight in — either the list from `BallTracker.predict(frames)` or a tracker
`ball.csv` (`Frame, Visibility, X, Y`):

```python
from bounce_detector import detect_bounces, compute_homography, load_track_csv, PICKLEBALL

coords  = load_track_csv("ball.csv")               # or: coords = ball_tracker.predict(frames)
H       = compute_homography(image_corners, PICKLEBALL)
bounces = detect_bounces(coords, homography=H, court=PICKLEBALL)
```

A bounce happens when the ball's **pixel-y reaches a local maximum** (y grows downward in
images, so the on-screen "lowest point") before rising again.

---

## The pipeline (what each step does)

1. **Interpolate gaps** — `interpolate_gaps`. The tracker only sees the ball in a fraction
   of frames, so we linearly fill `(x, y)` through gaps of **<= `max_gap` (20)** frames.
   A bounce that lands inside a short gap can then still be found. Gaps longer than
   `max_gap` are left empty (and become trajectory-segment boundaries).
2. **Smooth y** — `_smooth`. Box-filter the y-values with a width-**5** kernel using
   **edge padding** (avoids false peaks at the very start/end). The smoothed y is used
   *only* to find peaks; the **original** `(x, y)` are what get stored in each bounce.
3. **Find local y-maxima** — in `detect_bounces`. A point `i` is a candidate bounce when:
   - `sy[i] >= sy[i-1]` and `sy[i] >= sy[i+1]` (local max = lowest point on screen);
   - the frame gap on each side is `<= max_gap` (no giant jump bridging two segments);
   - the drop is real: `sy[i] - min(sy[i-3], sy[i+3]) >= min_drop_px` (6 px), filtering jitter.
4. **Lockout** — after an accepted bounce, the next **`lockout_frames` (15)** frames are
   ignored so one bounce can't fire twice.
5. **Map to court feet** — `pixel_to_court`. The bounce pixel is passed through a 3x3
   **ground homography** (from 4 clicked court corners) to get `(court_x, court_y)` in feet.
6. **IN / OUT / CLOSE_IN** — `classify_bounce`. Inside the court rectangle -> `IN`; outside
   -> `OUT`; inside but within **`close_margin_ft` (0.5)** of a line -> `CLOSE_IN`.
7. **Per-side suppression** — `_suppress_same_side`. In a real rally the ball alternates
   sides of the net. Two consecutive IN bounces on the **same** side (dribble / jitter) ->
   the second is marked `suppressed`. An `OUT` resets the alternation.

> **Known limitation (step 1):** interpolation only recovers a bounce if its gap is
> `<= max_gap` frames. With low detection density, bounces buried in long gaps are never
> seen. Raising `max_gap` recovers more but risks interpolating through genuinely missing
> arcs — tune against your footage.

---

## Setting up the homography (the one thing you must configure)
`compute_homography(image_corners, court)` needs the 4 court corners **clicked in the image**,
in this exact order so they line up with `CourtModel.corners_ft`:

```
near-left, near-right, far-right, far-left
(near = baseline closest to the camera)
```

Pass the matching court via a preset (`PICKLEBALL`, `TENNIS_SINGLES`, `TENNIS_DOUBLES`) or a
custom `CourtModel(width_ft, length_ft)`. If your real-world corner coordinates differ from
the default rectangle, pass them explicitly with `court_corners=[...]`. The homography
assumes the bounce point lies on the **ground plane** (true for bounces, which is the point).

Without a homography you can still run `detect_bounces(coords)` for **pixel-space** bounces
(`classification` stays `UNKNOWN`) — handy before the court is calibrated.

---

## Output
`detect_bounces(...)` returns a list of `Bounce` objects (chronological):

| field | meaning |
|-------|---------|
| `frame` | frame index of the bounce |
| `px_x, px_y` | bounce location in original video pixels |
| `court_x, court_y` | bounce location in court feet (`None` if no homography) |
| `classification` | `IN` / `OUT` / `CLOSE_IN` / `UNKNOWN` |
| `side` | `near` / `far` (which half of the net) |
| `suppressed` | `True` if dropped by step 7 |

`save_bounces_csv(bounces, "bounces.csv")` writes them out (suppressed rows skipped by default).

---

## Tuning (all on `BounceParams`)
| param | default | effect |
|-------|---------|--------|
| `max_gap` | 20 | interpolation / max temporal jump across a peak (frames) |
| `smooth_kernel` | 5 | y smoothing width (odd number) |
| `min_drop_px` | 6 | min y drop vs `i±3` to count as a real bounce |
| `drop_lookback` | 3 | the `i±N` used for the drop test |
| `lockout_frames` | 15 | dead time after a bounce |
| `close_margin_ft` | 0.5 | inside-but-this-close -> `CLOSE_IN` |
| `suppress_same_side` | True | enable step 7 |

**Frame-rate note:** these frame counts assume ~30 fps tracker output (GridTrackNet's working
rate, after its 60->30 down-sampling). If your effective rate differs, scale `max_gap`,
`lockout_frames`, and `smooth_kernel` proportionally.

---

## Handoff checklist
Send your friend just this folder. They need to:
1. `pip install -r requirements.txt` (numpy).
2. Feed in ball-tracking `(x, y)` (their tracker output or your `ball.csv`).
3. Click the 4 court corners once and build the homography (`compute_homography`).
4. Call `detect_bounces(...)` and consume the `Bounce` list / `bounces.csv`.

Nothing here depends on the tracker internals — only on the `(x, y)`-per-frame contract.

---

## Hardening against occlusion / camera issues (optional)

Two failure modes on real footage: a bounce fired inside a long detection gap (the ball was
occluded and interpolation drew a straight line that looked like a turn), and false peaks
high in the frame (player/net contact). Both are addressed with opt-in knobs on
`BounceParams` (defaults preserve the simple behaviour):

| param | default | effect |
|-------|---------|--------|
| `split_gap` | = `max_gap` | the trajectory is split wherever valid frames are more than this apart; each segment is smoothed + scanned on its own, so the smoother never bridges an occlusion and a peak at a gap edge (ball reappearing) is excluded. Lower it (e.g. 8–12) to harden. |
| `surface_min_py` | None | drop bounces whose pixel-y is above this row (too high in frame = likely player/net/occlusion). Eyeball the court-surface rows in one frame. |
| `max_out_of_bounds_ft` | None | with a homography, drop bounces that map more than this many feet outside the court (occlusion artifacts fly off-court). |

From the video tool:
```
uv run predict_with_bounces.py --video_path "my_videos\CAMERA 1.mp4" --model_path model_weights.onnx \
    --provider gpu --split_gap 10 --surface_min_py 500
```

## Live / streaming: `StreamingBounceDetector` (causal Kalman)

The offline `detect_bounces` uses centered smoothing (it looks at future frames), so it
can't run truly live. For a stream, use the **causal** detector: a constant-velocity Kalman
filter tracks `y` and `vy` from past frames only, **coasts through missing detections with
predict-only steps** (no interpolation, so no interpolation-induced false bounce), and fires
when `vy` flips from descending to ascending at a y-maximum.

```python
from bounce_detector import StreamingBounceDetector, BounceParams

det = StreamingBounceDetector(params=BounceParams(), homography=H, court=PICKLEBALL)
for frame_idx, (x, y) in enumerate(live_ball_positions):     # x,y per frame; (0,0)=miss
    bounce = det.update(frame_idx, x, y, detected=(x, y) != (0, 0))
    if bounce:
        handle(bounce)        # fires ~1-3 frames after the real contact (causal lag)
```

It resets (splits the trajectory) after `max_misses` consecutive gaps, applies the same
`lockout_frames` debounce, and honours the surface / out-of-bounds checks above. Tune
`process_var` (higher = trust motion model less, follow measurements more) and `meas_var`
(higher = smoother, more lag) for your tracker's noise. `detect_bounces_streaming(coords)`
runs it over a list for offline comparison.

**Reality check:** these are robustness improvements, not a substitute for calibration. The
thresholds (`min_drop_px`, `lockout_frames`, KF variances) still need tuning against labeled
pickleball bounces before you can trust precision/recall numbers.
