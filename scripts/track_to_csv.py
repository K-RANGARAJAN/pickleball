#!/usr/bin/env python3
"""
Run GridTrackNet over a clip and write a ball-track CSV (Frame,Visibility,X,Y).

This is the tracker step for the metric pipeline: the CSV is exactly what the friend's
bounce_detection_handoff/bounce_detector.py consumes (load_track_csv). One row per ORIGINAL
video frame; Visibility=0 (X=Y=0) on frames with no ball.

Usage:
  python scripts/track_to_csv.py --clip sideline.mp4 --out runs/metric/sideline/ball.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(REPO))

from linecall.detection.gridtracknet_detector import track_clip   # noqa: E402

MODEL = REPO / "ball_tracker_handoff" / "model_weights.onnx"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--provider", default="coreml", choices=["cpu", "gpu", "coreml"])
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=None)
    args = ap.parse_args()

    clip = REPO / args.clip if not Path(args.clip).is_absolute() else Path(args.clip)
    dets = track_clip(str(clip), "cam", str(MODEL), provider=args.provider,
                      start_frame=args.start, stop_frame=args.stop)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out, "w") as f:
        f.write("Frame,Visibility,X,Y\n")
        for d in dets:
            if d.is_detected:
                f.write(f"{d.frame_idx},1,{d.px_x:.1f},{d.px_y:.1f}\n")
                n += 1
            else:
                f.write(f"{d.frame_idx},0,0,0\n")
    print(f"[track] {clip.name}: {n}/{len(dets)} frames with ball -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
