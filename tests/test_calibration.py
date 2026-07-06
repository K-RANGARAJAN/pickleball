"""
Calibration tests — recover known camera poses from 4 court corners.

Builds synthetic cameras behind each baseline with KNOWN poses, projects the 4 court
corners to get the pixel corners the court model would output, then checks solve_camera_pose
recovers the correct pose, the correct baseline, a physically-valid camera (z>0), and low
reprojection error under corner-detection noise.

Guards against the planar-PnP mirror ambiguity bug (camera came out upside-down) and the
image-corner -> world-corner mapping.

Run:  python tests/test_calibration.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import cv2

from linecall.calibration.pose import solve_camera_pose, infer_corner_label

CW = {"far_left": (0, 0, 0), "far_right": (20, 0, 0),
      "near_right": (20, 44, 0), "near_left": (0, 44, 0)}


def _make_cam(cam_pos, look_at, w=1920, h=1080, fov=60):
    cam_pos = np.array(cam_pos, float); look_at = np.array(look_at, float)
    fwd = look_at - cam_pos; fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.vstack([right, down, fwd]); t = (-R @ cam_pos).reshape(3, 1)
    f = (w / 2) / np.tan(np.radians(fov / 2))
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
    return K, R, t


def _project(K, R, t, X):
    Xc = R @ np.array(X, float) + t.reshape(3)
    x = K @ Xc
    return x[:2] / x[2]


def _label_corners(pts):
    """Sort the 4 projected corners into TL,TR,BR,BL image order (as the model would)."""
    names = list(CW); P = np.array([pts[k] for k in names])
    tl = names[np.argmin(P[:, 0] + P[:, 1])]
    tr = names[np.argmax(P[:, 0] - P[:, 1])]
    br = names[np.argmax(P[:, 0] + P[:, 1])]
    bl = names[np.argmin(P[:, 0] - P[:, 1])]
    return np.array([pts[tl], pts[tr], pts[br], pts[bl]])


def _run_camera(cam_pos, truth_baseline, noise, seed=0):
    K, R, t = _make_cam(cam_pos, [10, 22, 0])
    pts = {k: _project(K, R, t, v) for k, v in CW.items()}
    ic = _label_corners(pts)
    if noise:
        ic = ic + np.random.default_rng(seed).normal(0, noise, ic.shape)
    return solve_camera_pose("cam", ic, 1920, 1080, camera_behind=truth_baseline)


def test_far_camera_noiseless():
    p = _run_camera([10, -8, 10], "far", 0.0)
    assert p.baseline == "far"
    assert p.camera_position_ft[2] > 0
    assert abs(p.camera_position_ft[1] - (-8)) < 1.0
    assert p.reproj_error_px < 0.5
    print("ok: far camera recovered exactly (noiseless)")


def test_near_camera_noiseless():
    p = _run_camera([10, 52, 10], "near", 0.0)
    assert p.baseline == "near"
    assert p.camera_position_ft[2] > 0
    assert abs(p.camera_position_ft[1] - 52) < 1.0
    assert p.reproj_error_px < 0.5
    print("ok: near camera recovered exactly (noiseless)")


def test_pose_is_right_side_up_under_noise():
    # The mirror-ambiguity bug produced z<0 (upside-down). Ensure z>0 holds with noise.
    for cam_pos, base in [([10, -8, 10], "far"), ([10, 52, 10], "near")]:
        for noise in [2.0, 4.0]:
            p = _run_camera(cam_pos, base, noise)
            assert p.camera_position_ft[2] > 0, f"{base} z<=0 at noise {noise}"
            assert p.baseline == base
            assert p.reproj_error_px < noise * 3 + 2
    print("ok: pose stays physically valid (z>0) and correct baseline under noise")


def test_corner_label_matches_baseline():
    assert infer_corner_label(_run_camera([10, -8, 10], "far", 0.0)) == "far_baseline"
    assert infer_corner_label(_run_camera([10, 52, 10], "near", 0.0)) == "near_baseline"
    print("ok: corner-label (know-your-corner) matches baseline")


def run_all():
    test_far_camera_noiseless()
    test_near_camera_noiseless()
    test_pose_is_right_side_up_under_noise()
    test_corner_label_matches_baseline()
    print("\nALL CALIBRATION TESTS PASSED")


if __name__ == "__main__":
    run_all()
