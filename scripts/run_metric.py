#!/usr/bin/env python3
"""
Metric pipeline for the controlled throw test.

tracker CSV (Frame,Visibility,X,Y)
  -> friend's bounce_detector.detect_bounces  (pixel-y local maxima -> IN/OUT/CLOSE_IN)
  -> cluster bounces into THROWS by time gap, keep the FIRST bounce of each throw
  -> predictions.csv  (throw, frame, t_s, court_x, court_y, call)

Only the first bounce of each throw is the call (the rest are dribbles). Throws are separated
by multi-second gaps (retrieve + re-throw); dribbles cluster within ~1-2s of the first bounce.

IN/OUT needs a ground homography. We use the calibrated corners under the given key in
configs/camera_corners.json (run scripts/corner_click_app.py, cam = "metric"). Without it,
calls come out UNKNOWN (positions still computed in pixels) so the pipeline still runs.

Usage:
  python scripts/run_metric.py --ball-csv runs/metric/sideline/ball.csv --session sideline \
      --out runs/metric/sideline/predictions.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "bounce_detection_handoff"))

from bounce_detector import detect_bounces, BounceParams, PICKLEBALL   # noqa: E402
from linecall.bounce.bounce_2d import ground_homography                # noqa: E402

CORNERS = REPO / "configs" / "camera_corners.json"


def load_coords(ball_csv: Path) -> list:
    coords = []
    with open(ball_csv, newline="") as f:
        for row in csv.DictReader(f):
            vis = int(float(row.get("Visibility", 1) or 0))
            if vis and (row["X"] or row["Y"]):
                coords.append((float(row["X"]), float(row["Y"])))
            else:
                coords.append((0.0, 0.0))
    return coords


def first_bounce_per_throw(bounces: list, fps: float, throw_gap_s: float) -> list:
    """Keep the first bounce of each throw. New throw when the time gap to the previous
    bounce exceeds throw_gap_s (dribbles are <~1-2s apart; throws are several s apart)."""
    bs = sorted((b for b in bounces if not b.suppressed), key=lambda b: b.frame)
    throws, last = [], None
    for b in bs:
        if last is None or (b.frame - last) / fps > throw_gap_s:
            throws.append(b)
        last = b.frame
    return throws


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ball-csv", required=True)
    ap.add_argument("--session", required=True, help="sideline | baseline (label only)")
    ap.add_argument("--corners-key", default="metric",
                    help="key in camera_corners.json for this camera's calibration")
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--throw-gap-s", type=float, default=1.5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    coords = load_coords(Path(args.ball_csv))
    detected = sum(1 for x, y in coords if (x, y) != (0.0, 0.0))

    H = None
    if CORNERS.exists():
        cc = json.loads(CORNERS.read_text())
        if args.corners_key in cc and "corners" in cc[args.corners_key]:
            H = ground_homography(cc[args.corners_key]["corners"])

    # tracker CSV is at native 60fps; friend's defaults assume ~30fps -> scale the
    # frame-based params x2 (gap/lockout/smoothing), keep px-based min_drop the same.
    params = BounceParams(max_gap=40, smooth_kernel=9, drop_lookback=6, min_drop_px=6.0,
                          lockout_frames=20, split_gap=40)
    bounces = detect_bounces(coords, homography=H, court=PICKLEBALL, params=params)
    throws = first_bounce_per_throw(bounces, args.fps, args.throw_gap_s)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["throw", "frame", "t_s", "court_x", "court_y", "call"])
        for i, b in enumerate(throws, 1):
            w.writerow([i, b.frame, round(b.frame / args.fps, 2),
                        "" if b.court_x is None else round(b.court_x, 2),
                        "" if b.court_y is None else round(b.court_y, 2),
                        b.classification])

    print(f"[metric] {args.session}: {detected}/{len(coords)} ball frames")
    print(f"[metric] {len(bounces)} raw bounces -> {len(throws)} throws (first bounce each)")
    print(f"[metric] calibration: "
          + ("YES" if H is not None
             else f"NONE -> calls UNKNOWN. Calibrate: python scripts/corner_click_app.py "
                  f"(cam '{args.corners_key}'), then re-run."))
    print(f"[metric] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
