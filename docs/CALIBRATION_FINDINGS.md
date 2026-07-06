# Calibration & Geometry Findings — 2026-06-15

Produced while validating the Phase-5 bounce / IN-OUT layer
(`scripts/validate_bounce_inout.py`, `tests/test_bounce_inout_pipeline.py`).

## TL;DR — a calibration bug was masking good geometry (now fixed)

The cameras were filmed correctly — **opposite diagonal corners** (cam1 at corner **B**,
cam2 at corner **C**), exactly per Phase-0 spec #7. But the calibration corners were clicked
in each camera's **local** "bottom-left = near_left" convention. Because cam1 and cam2 look
from opposite ends, the same local clicks map to **opposite world corners**, so both cameras
solved to nearly the same world position — a degenerate 3.7° triangulation angle.

**Fix:** cam1's four corners were remapped from camera-local names to the shared world frame
(local `near_left/near_right` → world `far_right/far_left`, and `net_left/net_right` swapped).
No reshoot, no new labels. After the fix the cameras sit at opposite corners with a **163.9°**
intersection angle and an **79.3 ft** baseline.

- Backup of the buggy config: `configs/camera_corners.BROKEN_backup.json`
- Corrected config (now live): `configs/camera_corners.json`

## Impact — same detector, same footage, just correct calibration

| Detector localization error | BROKEN calib (3.7°) | FIXED calib (163.9°) |
|---|---|---|
| | median / p90 / verdict | median / p90 / verdict |
| 0.5 px | 3.1 in / 8.0 in / 90% | **1.6 in / 3.1 in / 98%** |
| 1.0 px | 4.8 in / 16.9 in / 85% | **1.7 in / 3.2 in / 97%** |
| 1.5 px | 6.9 in / 25.6 in / 83% | **1.9 in / 3.3 in / 98%** |
| 2.0 px | 9.4 in / 33.4 in / 80% | **2.1 in / 3.6 in / 96%** |
| 3.0 px | 13.1 in / 47.7 in / 77% | **2.5 in / 4.6 in / 94%** |

Conclusions:

1. **In/out error dropped ~7 in → ~1.9 in**, verdict accuracy **83% → 98%**, and worst-case
   (p90) **25.6 in → 3.3 in** at realistic 1.5 px detector noise.
2. **The fixed geometry barely degrades with detector noise** (2.5 in even at 3 px) — a wide
   baseline absorbs pixel jitter. This is the regime the whole project was designed for.
3. A "too close to call" band is still appropriate, but it can now be much tighter (a couple
   of inches) instead of ~½ ft.

Frame-rate note: 60→120 fps barely changes spatial in/out error but halves bounce **timing**
error (~17→8 ms). fps is the timing dial, not the spatial-accuracy dial.

## Process lesson

The bug was invisible from the numbers alone: triangulation round-trips to 0.00 ft because
the collapsed calibration is internally *self-consistent*. It only showed up by (a) computing
the physical camera positions and the ray-intersection angle, and (b) looking at the actual
calibration frames, whose **different backgrounds** (cam1: blue container/shed; cam2: metal
warehouse) proved the cameras faced different directions. Always sanity-check camera geometry
against the real frames, not just reprojection error.

## New: line-drag calibration is now the standard

`scripts/corner_click_app.py` was replaced with a **line-drag** tool (old version kept at
`scripts/corner_click_app.BACKUP.py`). Instead of clicking 4 corners, you DRAG along the
four court edges as a loop (A→B→C→D→A); each edge is fit by total-least-squares over every
dragged pixel, and the 4 calibration corners are the **intersections of adjacent edges**.

Why: an intersection of two well-supported lines beats a single hand click, because each
line is backed by hundreds of pixels and per-click jitter averages out. Measured on this
footage, drag-derived corners are **~3× more accurate than clicked corners** at every hand-
noise level (e.g. at 3 px jitter: clicking ≈ 3.5 px corner error, dragging ≈ 1.15 px). The
new tool writes the same `configs/camera_corners.json` via the same `solve_camera_pose_named`,
so it is a drop-in front-end — the pose solve is unchanged. Math lives in
`linecall/calibration/line_fit.py` (pure numpy, unit-tested in `tests/test_line_fit.py`).

Note it does NOT remove the need to state corner identity: you still trace the same physical
loop, same rotational direction, in both cameras — that identity step is what the earlier bug
got wrong, so the tool makes it explicit (start at A, follow the on-screen edge order).

## What to do next

- **Detector precision now matters less than it did** — the geometry carries the accuracy.
  Aim detection effort at recall/robustness rather than sub-pixel centroids.
- The cam2 manual labeling effort is **no longer chasing a ceiling** — with correct geometry,
  those labels feed a reconstruction that can actually resolve line calls.
- Re-run any prior 3D/triangulation results: they were computed against the broken calibration
  and will improve substantially under the fixed one.

## How to reproduce

```bash
python scripts/validate_bounce_inout.py --figure   # numbers + 3-panel PNG + geometry readout
python tests/test_bounce_inout_pipeline.py         # end-to-end regression asserts

# Compare broken vs fixed directly:
python scripts/validate_bounce_inout.py --figure \
    # (broken config preserved at configs/camera_corners.BROKEN_backup.json)
```

Outputs land in `runs/validation/` (`bounce_inout_report.json`,
`bounce_inout_validation.png`, and the two calibration frames).

## Update 2 — calibration was STILL wrong; now fixed against real data (2026-06-15)

The earlier "fix" validated against synthetic data derived from the same corners, so it
confirmed self-consistency, not correctness. Testing triangulation on the REAL hand-labeled
ball points (cam1 + the 69 cam2 labels) exposed a ~20px residual floor and a flat ~5ft
height — both calibration symptoms.

Root cause was THREE things that had to be right at once, and no prior attempt had all three:
  1. cam1 corner world-identities (the B/C remap) — lived in camera_corners_fixed.json but a
     later focal-tuning save had reverted the live config to the broken local naming.
  2. focal length — default_intrinsics' 60-deg-FOV guess (~1663px) was wrong; the true value
     recovered from real correspondences is ~1900px.
  3. sync offset — 15 frames (cam2 = cam1 + 15).

With all three: median triangulation residual on real GT points = **1.9px**, and GT balls
trace a physical arc (x steady, y crossing toward the net, z descending smoothly), cameras at
opposite corners (162.5deg). configs/camera_corners.json now stores focal_px per camera; the
validation harness reads it.

Lesson: validate calibration against INDEPENDENT real correspondences (hand-labeled points in
both views), never against data synthesized from the calibration being tested.

## Update 3 — detection status (honest)

Classical color detector finds the ball in ~97% (cam1) / ~93% (cam2) of frames in its top-K
candidates, but per-camera it can't reliably pick THE ball as #1 (best-by-color ~38%, ~51%
with motion gating). Cross-camera triangulation with the corrected calibration produces a
physical 3D track but currently lands ~48-56% of frames per-rally — clutter pairs sometimes
triangulate cleanly, and some frames miss the ball in one view.

So: calibration is no longer the blocker (solid at 1.9px). Remaining gap is detection
robustness. Options to close it: (a) train YOLO on the existing 2,127 cam1 labels for clean
single detections, (b) tighten the multiview gate (epipolar constraint between cameras, not
just reprojection), (c) stronger temporal tracking to bridge single-view misses.
