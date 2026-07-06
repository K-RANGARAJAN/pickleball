#!/usr/bin/env python3
"""Automated cam1 pipeline: GridTrackNet -> 2D bounces (homography) -> IN/OUT.

Writes runs/pipeline/<tag>/{calls.json, track.csv} for demo_app.

  python scripts/run_gridtracknet.py --provider coreml --tag gridtracknet
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(REPO))

from linecall.detection.gridtracknet_detector import track_clip            # noqa: E402
from linecall.calibration.pose import solve_camera_pose_named             # noqa: E402
from linecall.bounce.bounce_2d import detect_bounces_2d                    # noqa: E402
CLIPS = {"cam1": "CAMERA 1.mp4", "cam2": "camera2_muted.mp4"}
CORNERS = REPO / "configs" / "camera_corners.json"
MODEL = REPO / "ball_tracker_handoff" / "model_weights.onnx"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="cpu", choices=["cpu", "gpu", "coreml"],
                    help="cpu | gpu (CUDA, the 4060) | coreml (Mac GPU/Neural Engine)")
    ap.add_argument("--start", type=int, default=0, help="start frame (cam1 timeline)")
    ap.add_argument("--stop", type=int, default=None, help="stop frame (cam1 timeline)")
    ap.add_argument("--tag", default="gridtracknet")
    args = ap.parse_args()

    fps = 60.0
    corners = json.loads(CORNERS.read_text())
    pose = solve_camera_pose_named(
        "cam1", {k: tuple(v) for k, v in corners["cam1"]["corners"].items()},
        corners["cam1"]["image_w"], corners["cam1"]["image_h"])
    print(f"[calib] cam1 reproj {pose.reproj_error_px:.2f}px")

    # --- detection: two passes (even + odd frames) for full 60fps coverage --
    # GridTrackNet needs ~30fps spacing; our clip is 60fps so fps_skip=2.
    # A bounce peak can land on the skipped frame — running a second pass on
    # the offset-1 frames doubles coverage without violating the model's
    # temporal contract (each pass still sees 30fps spacing).
    print("[detect] running GridTrackNet on cam1 (even frames) …")
    d_even = track_clip(str(REPO / CLIPS["cam1"]), "cam1", str(MODEL),
                        provider=args.provider, start_frame=args.start, stop_frame=args.stop)
    print("[detect] running GridTrackNet on cam1 (odd frames) …")
    d_odd = track_clip(str(REPO / CLIPS["cam1"]), "cam1", str(MODEL),
                       provider=args.provider,
                       start_frame=(args.start or 0) + 1, stop_frame=args.stop)

    det: dict = {d.frame_idx: d for d in d_even}
    for d in d_odd:
        if d.is_detected and (d.frame_idx not in det or not det[d.frame_idx].is_detected):
            det[d.frame_idx] = d

    # --- 2D bounces from cam1 alone (homography to court) ------------------
    from collections import Counter
    hits = [d for d in det.values() if d.is_detected]
    bounces = detect_bounces_2d(hits, "cam1", corners["cam1"]["corners"])
    vc = dict(Counter(b.verdict for b in bounces))
    print(f"[2d] cam1: {len(hits)} detections -> {len(bounces)} bounces {vc}")

    # --- cam1 ball pixels per frame, for the overlay trail ----------------
    f0 = args.start
    f1 = (args.stop if args.stop is not None else max(det)) - 1
    rows = []
    for f in range(f0, f1 + 1):
        p1 = det.get(f)
        rows.append((f, (p1.px_x, p1.px_y) if p1 and p1.is_detected else None))

    out = REPO / "runs" / "pipeline" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    t0_s = f0 / fps

    with open(out / "track.csv", "w") as fc:
        fc.write("ref_idx,t_ref,cam1_px_x,cam1_px_y,cam1_frame\n")
        for (f, p1) in rows:
            c1 = (round(p1[0], 1), round(p1[1], 1)) if p1 else ("", "")
            fc.write(f"{f},{round(f/fps,4)},{c1[0]},{c1[1]},{f}\n")

    n1 = sum(1 for d in det.values() if d.is_detected)
    calls = {
        "slice": {"ref_cam": "cam1", "t0": t0_s, "t1": f1 / fps, "fps_ref": fps},
        "sync": [{"cam": "cam1", "offset_s": 0.0, "confidence": 1.0, "reference": True}],
        "calibration": {"cam1": {"reproj_px": pose.reproj_error_px,
                                 "camera_ft": list(pose.camera_position_ft)}},
        "clip_offsets": {"cam1": 0.0},
        "clip_fps": {"cam1": fps},
        "detection": {"cam1": {"frames": len(det), "detected": n1,
                               "density": round(n1 / max(len(det), 1), 3)}},
        "cam_calls": {
            "cam1": [{"ref_idx": b.frame_idx, "t_ref": b.frame_idx / fps,
                      "x_ft": round(b.court_x, 1), "y_ft": round(b.court_y, 1),
                      "verdict": b.verdict, "camera_id": "cam1",
                      "cam1_frame": b.frame_idx}
                     for b in bounces]
        },
    }
    (out / "calls.json").write_text(json.dumps(calls, indent=1))
    print(f"\n[done] cam1={len(bounces)} calls -> {out}")
    print(f"next: python scripts/demo_app.py --tag {args.tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
