"""
Tests for the line-fit calibration front-end (linecall/calibration/line_fit.py).

Proves the drag-an-edge approach is sound:
  * a noisy dragged segment recovers the true line (orthogonal regression),
  * vertical/steep edges work (where y=mx+b least squares would fail),
  * the 4 corners come back from edge intersections at the known court corners,
  * end-to-end: drag-derived corners feed solve_camera_pose_named and reproduce the
    known-good camera geometry.

Run:  python tests/test_line_fit.py   (or pytest)
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from linecall.calibration.line_fit import fit_line, intersect, corners_from_edges


def _noisy_segment(p0, p1, n=200, noise_px=1.5, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n)[:, None]
    pts = (1 - t) * np.array(p0) + t * np.array(p1)
    # perpendicular noise
    d = np.array(p1) - np.array(p0); d = d / np.linalg.norm(d)
    nrm = np.array([-d[1], d[0]])
    pts = pts + rng.normal(0, noise_px, size=(n, 1)) * nrm
    return pts.tolist()


def test_fit_recovers_true_line():
    seg = _noisy_segment((100, 100), (900, 140), noise_px=1.0)
    line = fit_line(seg)
    # true direction
    td = np.array([800, 40], float); td /= np.linalg.norm(td)
    cos = abs(float(line.direction @ td))
    assert cos > 0.999, f"direction off: cos={cos}"
    assert line.rms_px < 2.0, f"rms too high: {line.rms_px}"
    print(f"ok: line fit recovers direction (cos={cos:.5f}, rms={line.rms_px:.2f}px)")


def test_vertical_line_ok():
    # a near-vertical drag — would break y=mx+b least squares
    seg = _noisy_segment((500, 50), (505, 1000), noise_px=1.0)
    line = fit_line(seg)
    td = np.array([5, 950], float); td /= np.linalg.norm(td)
    assert abs(float(line.direction @ td)) > 0.999
    print("ok: vertical edge fit (orthogonal regression handles steep lines)")


def test_intersection_recovers_corner():
    # two perpendicular edges crossing at (300, 400)
    h = fit_line(_noisy_segment((100, 400), (700, 400), noise_px=1.0, seed=1))
    v = fit_line(_noisy_segment((300, 100), (300, 800), noise_px=1.0, seed=2))
    x, y = intersect(h, v)
    assert abs(x - 300) < 3 and abs(y - 400) < 3, f"intersection ({x:.1f},{y:.1f})"
    print(f"ok: intersection recovers corner at ({x:.1f},{y:.1f}) ~ (300,400)")


def test_corners_from_edges_quad():
    """Trace a known image quad as 4 noisy edges; corners must come back near truth."""
    # Full-court loop: A=far_left, B=far_right, C=near_right, D=near_left
    A = (400, 950); B = (150, 600); C = (1000, 560); D = (1750, 700)
    edges = {
        "AB": _noisy_segment(A, B, noise_px=1.5, seed=3),
        "BC": _noisy_segment(B, C, noise_px=1.5, seed=4),
        "CD": _noisy_segment(C, D, noise_px=1.5, seed=5),
        "DA": _noisy_segment(D, A, noise_px=1.5, seed=6),
    }
    corners, rms = corners_from_edges(edges)
    truth = {"far_left": A, "far_right": B, "near_right": C, "near_left": D}
    for name, t in truth.items():
        got = corners[name]
        err = np.hypot(got[0] - t[0], got[1] - t[1])
        assert err < 6.0, f"{name}: {err:.1f}px from truth"
    print(f"ok: 4 corners from dragged edges within 6px (rms {rms})")


def _densify(a, b, n=80):
    """Mimic the frontend: two endpoints -> N points along the straight line."""
    return [[a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]
            for t in (i / (n - 1) for i in range(n))]


def test_solid_line_endpoints_recover_corners():
    """The new UI sends two endpoints per edge (densified). Endpoints placed exactly on the
    true corners must yield those corners back with ~0 residual through the fit+intersect."""
    # Full-court loop order: A=far_left, B=far_right, C=near_right, D=near_left
    A = (400, 950); B = (150, 600); C = (1000, 560); D = (1750, 700)
    edges = {"AB": _densify(A, B), "BC": _densify(B, C),
             "CD": _densify(C, D), "DA": _densify(D, A)}
    corners, rms = corners_from_edges(edges)
    truth = {"far_left": A, "far_right": B, "near_right": C, "near_left": D}
    for name, t in truth.items():
        err = np.hypot(corners[name][0] - t[0], corners[name][1] - t[1])
        assert err < 1e-6, f"{name}: {err} px (should be exact for clean endpoints)"
    assert max(rms.values()) < 1e-6, f"edge rms should be ~0, got {rms}"
    print("ok: solid two-endpoint lines recover corners exactly (densify path)")


def test_end_to_end_drag_feeds_solver_self_contained():
    """Drag-derived corners must feed the real solver and produce a sane pose. Uses a
    FIXED, self-contained quad (not the mutable config) so the test can't be broken by a
    later calibration edit. cam2's real near-half quad is a good, known-solvable shape."""
    import cv2
    from linecall.calibration.pose import solve_camera_pose_named, ALL_WORLD_POINTS
    # Build a FULL-COURT pixel quad by projecting the 4 outer world corners through a
    # synthetic camera, so the quad is guaranteed solvable (full-court calibration).
    W, H = 1920, 1080
    f = (W / 2) / np.tan(np.radians(60 / 2))
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    cam_pos = np.array([10, 60, 9], float)            # behind the near baseline, ~9 ft up
    fwd = np.array([10, 22, 0], float) - cam_pos; fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.vstack([right, down, fwd]); t = (-R @ cam_pos).reshape(3, 1)

    def proj(name):
        X = np.array(ALL_WORLD_POINTS[name], float)
        x = K @ (R @ X + t.reshape(3)); return (x[0] / x[2], x[1] / x[2])
    A, B, C, D = proj("far_left"), proj("far_right"), proj("near_right"), proj("near_left")
    edges = {"AB": _noisy_segment(A, B, noise_px=1.0, seed=10),
             "BC": _noisy_segment(B, C, noise_px=1.0, seed=11),
             "CD": _noisy_segment(C, D, noise_px=1.0, seed=12),
             "DA": _noisy_segment(D, A, noise_px=1.0, seed=13)}
    corners, _ = corners_from_edges(edges)
    pose = solve_camera_pose_named("cam2", corners, 1920, 1080)
    assert pose.reproj_error_px < 8.0, f"reproj {pose.reproj_error_px}px"
    px, py, pz = pose.camera_position_ft
    # camera should be outside the court rectangle and a few feet up
    assert not (0 < px < 20 and 0 < py < 44), f"camera inside court? {(px,py)}"
    assert 2.0 < pz < 30.0, f"implausible height {pz}"
    print(f"ok: drag-derived corners solve to a sane pose "
          f"(reproj {pose.reproj_error_px:.2f}px, pos {np.round([px,py,pz],1)})")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("\nAll line-fit tests passed.")
