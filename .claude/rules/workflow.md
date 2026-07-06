# Workflow Rules

## Validate on real data, not just synthetic

- Synthetic/known-answer tests prove the *math*. They do NOT prove it works on the real
  night footage. Before claiming a stage works, run it on the real clips
  (`CAMERA 1.mp4` + `camera2_muted.mp4`) and inspect the actual output.
- When something fails, find the root cause with data (print the real track, the detection
  positions, the z-trajectory) before changing code. Several bugs here looked like one layer
  but were caused by another (e.g. "no bounces" was actually the detector locking onto a
  stationary object, not a bounce-logic problem).

## Be honest about status

- Don't call something "working" until it's verified on real footage. Distinguish clearly:
  math verified vs. works on this court's real data.
- State limitations plainly. The detector does not reliably see the ball on night footage;
  the recorded demo runs off hand-labels, not live detection. Don't paper over this.

## The two data paths (don't confuse them)

- **Recorded demo:** hand-labels in `labels/click_labels.json` ARE the detections →
  `run_from_labels.py`. No detector needed. This is what currently works.
- **Live (the goal):** a real YOLO detector runs per frame → the same 3D pipeline. This is
  the unsolved part and needs a detector retrained on this court.

## Detection lessons (so they aren't relearned)

- The ball is small, fast, and dark on night footage. A model trained on public/daylight
  data fails here. Train on OUR labeled frames for OUR court.
- The detector keeps locking onto stationary objects (a spare ball on court, a mark). Motion
  gating (`motion_gate.py`) and "keep all candidates, pick the moving one" mitigate this.
- For tiny objects, tiled (SAHI) inference helps recall but is slower — fine for recorded,
  watch the budget for live.

## Calibration

- Calibrate from the 4 corners the user clicks (`corner_click_app.py`). The far baseline is
  often blocked by the net → use the two net-post bases (y=22) plus the two near corners.
- Always sanity-check a solve: camera must be above ground (z > 0) and outside the court. A
  below-ground camera means a corner identity / mirror problem — reject and retry.

## Files & outputs

- Final outputs go under `runs/` (gitignored). Big files (videos, weights) are gitignored.
- Don't commit `venv/`, `.venv/`, `runs/`, `data/weights/*.pt`, `*.mp4`.
- Keep `scripts/*.BACKUP.py` out of the way; delete stale backups when confident.
