#!/usr/bin/env python3
"""
Two-camera bounce calls from hand-marked contact points.

Each camera reliably calls bounces on the half it sits behind; neither covers the full
court alone. This merges both cameras' hand-marked bounces: every bounce is placed by
whichever camera saw it from its NEAR (reliable) side (closer-camera-wins). Produces:
  - runs/pipeline/<tag>/calls.json + track.csv   (demo_app, cam1 video + merged calls)
  - runs/pipeline/<tag>/topdown.png              (full court, every bounce by verdict+cam)

Inputs: labels/bounce_labels.json with cam1 and/or cam2 ground-contact clicks.

Usage:
  python scripts/run_two_cam_labels.py --tag twocam
  python scripts/demo_app.py --tag twocam
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(REPO))

from linecall.calibration.pose import solve_camera_pose_named                       # noqa: E402
from linecall.bounce.bounce_2d import bounces_from_points, merge_bounces_union       # noqa: E402
from linecall.bounce.bounce import IN, OUT, CLOSE_IN                                 # noqa: E402

CORNERS = REPO / "configs" / "camera_corners.json"
BOUNCE_LABELS = REPO / "labels" / "bounce_labels.json"
TRAIL_LABELS = REPO / "labels" / "click_labels.json"


def draw_topdown(merged, path: Path, shared_frames=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mp
    shared = set(shared_frames or [])
    fig, ax = plt.subplots(figsize=(5, 9))
    ax.add_patch(mp.Rectangle((0, 0), 20, 44, fill=False, lw=2, ec="#222"))
    for y in (15, 22, 37):
        ax.plot([0, 20], [y, y], "-" if y == 22 else "--",
                color="#222" if y == 22 else "#999", lw=2 if y == 22 else 1)
    ax.plot([10, 10], [0, 15], "--", color="#ccc", lw=1)
    ax.plot([10, 10], [37, 44], "--", color="#ccc", lw=1)
    col = {IN: "#2a9d8f", OUT: "#e76f51", CLOSE_IN: "#e9c46a"}
    marker = {"cam1": "o", "cam2": "s"}
    seen = set()
    for b in merged:
        key = (b.verdict, b.camera_id)
        lab = f"{b.verdict} ({b.camera_id})" if key not in seen else None
        seen.add(key)
        if b.frame_idx in shared:
            ax.scatter(b.court_x, b.court_y, s=210, facecolors="none",
                       edgecolors="#000", linewidths=1.6, zorder=4)
        ax.scatter(b.court_x, b.court_y, s=90, c=col.get(b.verdict, "#888"),
                   marker=marker.get(b.camera_id, "o"),
                   edgecolors="#111", linewidths=0.9, label=lab, zorder=5)
    ax.scatter([], [], s=210, facecolors="none", edgecolors="#000", linewidths=1.6,
               label=f"both cams ({len(shared)})")
    ax.set_xlim(-3, 23); ax.set_ylim(-3, 47); ax.set_aspect("equal"); ax.invert_yaxis()
    ax.set_xlabel("x (ft)"); ax.set_ylabel("y (ft)  [0=far baseline, 44=near]")
    ax.set_title(f"Full-court bounces — {len(merged)} calls (○ cam1, □ cam2)")
    if seen:
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="twocam")
    ap.add_argument("--offset-frames", type=int, default=None,
                    help="cam2 frame = cam1 frame + OFFSET. Omit to auto-estimate from bounces.")
    ap.add_argument("--fps", type=float, default=60.0)
    args = ap.parse_args()

    corners = json.loads(CORNERS.read_text())
    poses, cam_y = {}, {}
    for cam in ("cam1", "cam2"):
        b = corners[cam]
        p = solve_camera_pose_named(cam, {k: tuple(v) for k, v in b["corners"].items()},
                                    b["image_w"], b["image_h"])
        poses[cam] = p
        cam_y[cam] = p.camera_position_ft[1]
        print(f"[calib] {cam} reproj {p.reproj_error_px:.2f}px, y={cam_y[cam]:.1f}ft")

    bl = json.loads(BOUNCE_LABELS.read_text())
    per_cam = {}
    for cam in ("cam1", "cam2"):
        pts = [(int(k), float(v[0]), float(v[1]))
               for k, v in bl.get(cam, {}).items() if v and len(v) == 2]
        calls = bounces_from_points(pts, cam, corners[cam]["corners"],
                                    reproj_px=poses[cam].reproj_error_px)
        per_cam[cam] = calls
        print(f"[{cam}] {len(pts)} marked -> {len(calls)} on-court calls "
              f"({len(pts) - len(calls)} dropped off-court)")

    merged, rep = merge_bounces_union(per_cam["cam1"], per_cam["cam2"], args.fps, cam_y,
                                      offset_frames=args.offset_frames)
    from collections import Counter
    by_verdict = Counter(b.verdict for b in merged)
    print(f"[offset] cam2 = cam1 {rep['offset_frames']:+d} frames "
          f"({rep['offset_frames']/args.fps:+.2f}s)"
          + ("  (given)" if args.offset_frames is not None else "  (auto-estimated)"))
    print(f"[merge] {rep['total']} unique bounces = {rep['shared']} shared "
          f"+ {rep['cam1_only']} cam1-only + {rep['cam2_only']} cam2-only")
    print(f"[merge] verdicts {dict(by_verdict)}")

    out = REPO / "runs" / "pipeline" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    draw_topdown(merged, out / "topdown.png", rep["shared_frames"])
    print(f"[topdown] wrote {out/'topdown.png'}")

    # cam1 trail for the demo overlay (from the GridTrackNet-seeded trail labels)
    trail = json.loads(TRAIL_LABELS.read_text()).get("cam1", {}) if TRAIL_LABELS.exists() else {}
    tr = {int(k): v for k, v in trail.items() if v and len(v) == 2}
    f0 = min(tr) if tr else 0
    f1 = max(tr) if tr else (max((b.frame_idx for b in merged), default=0))
    with open(out / "track.csv", "w") as fc:
        fc.write("ref_idx,t_ref,cam1_px_x,cam1_px_y,cam1_frame\n")
        for f in range(f0, f1 + 1):
            p = tr.get(f)
            c1 = (round(p[0], 1), round(p[1], 1)) if p else ("", "")
            fc.write(f"{f},{round(f/args.fps,4)},{c1[0]},{c1[1]},{f}\n")

    calls = {
        "slice": {"ref_cam": "cam1", "t0": f0 / args.fps, "t1": f1 / args.fps, "fps_ref": args.fps},
        "sync": [{"cam": "cam1", "offset_s": 0.0, "confidence": 1.0, "reference": True},
                 {"cam": "cam2", "offset_s": rep["offset_frames"] / args.fps, "confidence": 1.0}],
        "calibration": {c: {"reproj_px": poses[c].reproj_error_px,
                            "camera_ft": list(poses[c].camera_position_ft)} for c in poses},
        "clip_offsets": {"cam1": 0.0},
        "clip_fps": {"cam1": args.fps},
        "detection": {"cam1": {"frames": len(tr), "detected": len(tr),
                               "density": 1.0 if tr else 0.0}},
        "cam_calls": {
            "cam1": [{"ref_idx": b.frame_idx, "t_ref": b.frame_idx / args.fps,
                      "x_ft": round(b.court_x, 1), "y_ft": round(b.court_y, 1),
                      "verdict": b.verdict, "camera_id": b.camera_id,
                      "confidence": round(b.confidence, 2),
                      "cam1_frame": b.frame_idx}
                     for b in merged]
        },
    }
    (out / "calls.json").write_text(json.dumps(calls, indent=1))
    print(f"\n[done] {len(merged)} merged calls -> {out}")
    print(f"next: python scripts/demo_app.py --tag {args.tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
