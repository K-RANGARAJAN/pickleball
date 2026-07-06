"""
Tests for solve_camera_pose_named — the diagonal-camera calibration path.

Mirrors the handoff test style: synthesize a camera at a known pose, project the
known world corners through it, feed those pixels back with EXPLICIT identity, and
require the recovered pose to match. The key case the original mapping could not
handle: a camera at a diagonal corner (not behind a baseline).

Run: python tests/test_calibration_named.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

from linecall.calibration.pose import (WORLD_CORNERS, default_intrinsics,  # noqa: E402
                                       solve_camera_pose_named)


def _synth_camera(cam_pos, look_at, image_w=1920, image_h=1080):
    """Build K, R, t for a camera at cam_pos looking at look_at (world feet, z up)."""
    K = default_intrinsics(image_w, image_h)
    fwd = np.array(look_at, float) - np.array(cam_pos, float)
    fwd /= np.linalg.norm(fwd)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, world_up); right /= np.linalg.norm(right)
    down = np.cross(fwd, right)              # camera y points down in image
    R = np.stack([right, down, fwd])          # world -> camera rotation
    t = -R @ np.array(cam_pos, float)
    return K, R, t


def _project(K, R, t, pts_world):
    rvec, _ = cv2.Rodrigues(R)
    proj, _ = cv2.projectPoints(np.asarray(pts_world, float), rvec, t.reshape(3, 1),
                                K, np.zeros((4, 1)))
    return proj.reshape(-1, 2)


def test_diagonal_corner_camera_recovered():
    """Camera clamped at a DIAGONAL corner (the real rig) — pose must be recovered
    from explicitly-named corners with tiny reprojection error."""
    for cam_pos in [(-6.0, -8.0, 9.0),    # behind far-left corner, 9 ft up
                    (26.0, 52.0, 9.0)]:   # behind near-right corner (opposite diagonal)
        K, R, t = _synth_camera(cam_pos, look_at=(10.0, 22.0, 0.0))
        names = list(WORLD_CORNERS.keys())
        px = _project(K, R, t, [WORLD_CORNERS[n] for n in names])
        named = {n: tuple(px[i]) for i, n in enumerate(names)}
        pose = solve_camera_pose_named("diag", named, 1920, 1080, K=K)
        err_pos = np.linalg.norm(np.array(pose.camera_position_ft) - np.array(cam_pos))
        assert pose.reproj_error_px < 1.0, f"reproj {pose.reproj_error_px:.2f}px"
        assert err_pos < 1.0, f"camera position off by {err_pos:.2f} ft"
        assert pose.camera_position_ft[2] > 0, "must pick the above-ground solution"
    print("ok: diagonal-corner cameras recovered from named corners (<1 ft, <1 px)")


def test_identity_actually_matters():
    """Swapping two corner identities must degrade the solve (i.e., identity is real
    information, not something reprojection can silently fix)."""
    cam_pos = (-6.0, -8.0, 9.0)
    K, R, t = _synth_camera(cam_pos, look_at=(10.0, 22.0, 0.0))
    names = list(WORLD_CORNERS.keys())
    px = _project(K, R, t, [WORLD_CORNERS[n] for n in names])
    named = {n: tuple(px[i]) for i, n in enumerate(names)}
    swapped = dict(named)
    swapped["far_left"], swapped["far_right"] = swapped["far_right"], swapped["far_left"]
    good = solve_camera_pose_named("good", named, 1920, 1080, K=K)
    bad = solve_camera_pose_named("bad", swapped, 1920, 1080, K=K)
    assert bad.reproj_error_px > 10 * max(good.reproj_error_px, 0.05), (
        f"swap not caught: good={good.reproj_error_px:.3f} bad={bad.reproj_error_px:.3f}")
    print(f"ok: corner-identity swap detected (reproj {good.reproj_error_px:.2f}px -> "
          f"{bad.reproj_error_px:.1f}px) — the UI surfaces this")


if __name__ == "__main__":
    test_diagonal_corner_camera_recovered()
    test_identity_actually_matters()
    print("\nALL NAMED-CALIBRATION TESTS PASSED")
