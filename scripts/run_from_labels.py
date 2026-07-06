#!/usr/bin/env python3
"""
Single-camera pipeline driven by hand-labels instead of the live detector.

The labels in labels/click_labels.json ARE the detections: each cam1 frame's clicked
ball-center pixel is fed straight into the same 2D bounce pipeline run_gridtracknet.py uses
(homography to court feet -> ground contact -> IN/OUT). Writes the identical output format
so demo_app.py reads it unchanged.

This is the guaranteed-clean demo path: when the detector is too sparse, accurate labels
give a flawless bounce track for THIS clip.

Usage:
  python scripts/run_from_labels.py --tag from_labels
  python scripts/demo_app.py --tag from_labels
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(REPO))

from linecall.detection.base import Detection                              # noqa: E402
from linecall.calibration.pose import solve_camera_pose_named              # noqa: E402
from linecall.bounce.bounce_2d import detect_bounces_2d, bounces_from_points  # noqa: E402

CLIP = REPO / "CAMERA 1.mp4"
CORNERS = REPO / "configs" / "camera_corners.json"
LABELS = REPO / "labels" / "click_labels.json"
BOUNCE_LABELS = REPO / "labels" / "bounce_labels.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="from_labels")
    ap.add_argument("--labels", default=str(LABELS),
                    help="trail labels (ball pixel per frame) -> the visual ball trail")
    ap.add_argument("--bounce-labels", default=None,
                    help="hand-marked bounce-contact points; each becomes a call directly. "
                         f"Use {BOUNCE_LABELS} to enable the bulletproof direct-bounce path.")
    ap.add_argument("--fps", type=float, default=60.0)
    args = ap.parse_args()

    fps = args.fps
    corners = json.loads(CORNERS.read_text())
    pose = solve_camera_pose_named(
        "cam1", {k: tuple(v) for k, v in corners["cam1"]["corners"].items()},
        corners["cam1"]["image_w"], corners["cam1"]["image_h"])
    print(f"[calib] cam1 reproj {pose.reproj_error_px:.2f}px")

    labels = json.loads(Path(args.labels).read_text())
    cam1 = labels.get("cam1", {})
    det = {}
    for k, v in cam1.items():
        f = int(k)
        if v and len(v) == 2:
            det[f] = Detection("cam1", f, float(v[0]), float(v[1]), 1.0, "label")
        else:
            det[f] = Detection.empty("cam1", f)

    hits = [d for d in det.values() if d.is_detected]
    from collections import Counter
    if args.bounce_labels:
        bl = json.loads(Path(args.bounce_labels).read_text()).get("cam1", {})
        pts = [(int(k), float(v[0]), float(v[1])) for k, v in bl.items()
               if v and len(v) == 2]
        bounces = bounces_from_points(pts, "cam1", corners["cam1"]["corners"])
        vc = dict(Counter(b.verdict for b in bounces))
        print(f"[bounce] {len(pts)} hand-marked points -> {len(bounces)} calls {vc} "
              f"(direct map, no trajectory search)")
    else:
        bounces = detect_bounces_2d(hits, "cam1", corners["cam1"]["corners"])
        vc = dict(Counter(b.verdict for b in bounces))
        print(f"[2d] cam1: {len(hits)} labeled points -> {len(bounces)} bounces {vc}")

    f0 = min(det) if det else 0
    f1 = (max(det) if det else 0)
    rows = []
    for f in range(f0, f1 + 1):
        p1 = det.get(f)
        rows.append((f, (p1.px_x, p1.px_y) if p1 and p1.is_detected else None))

    out = REPO / "runs" / "pipeline" / args.tag
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "track.csv", "w") as fc:
        fc.write("ref_idx,t_ref,cam1_px_x,cam1_px_y,cam1_frame\n")
        for (f, p1) in rows:
            c1 = (round(p1[0], 1), round(p1[1], 1)) if p1 else ("", "")
            fc.write(f"{f},{round(f/fps,4)},{c1[0]},{c1[1]},{f}\n")

    calls = {
        "slice": {"ref_cam": "cam1", "t0": f0 / fps, "t1": f1 / fps, "fps_ref": fps},
        "sync": [{"cam": "cam1", "offset_s": 0.0, "confidence": 1.0, "reference": True}],
        "calibration": {"cam1": {"reproj_px": pose.reproj_error_px,
                                 "camera_ft": list(pose.camera_position_ft)}},
        "clip_offsets": {"cam1": 0.0},
        "clip_fps": {"cam1": fps},
        "detection": {"cam1": {"frames": len(det), "detected": len(hits),
                               "density": round(len(hits) / max(len(det), 1), 3)}},
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
