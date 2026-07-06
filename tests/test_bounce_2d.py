"""
Per-camera 2D bounce detection tests.

Verifies the ground homography maps pixels to court feet, and that a synthetic
descending-then-ascending pixel trajectory (a bounce in image space) is detected and
classified. This is the high-coverage path used when the two cameras rarely co-detect.

Run:  python tests/test_bounce_2d.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from dataclasses import dataclass

from linecall.bounce.bounce_2d import (
    detect_bounces_2d, ground_homography, _px_to_court, merge_camera_bounces,
)
from linecall.bounce.bounce import IN, OUT


@dataclass
class D:
    frame_idx: int
    px_x: float
    px_y: float
    is_detected: bool = True


# A fake camera looking at the court: 4 named corners at known pixels.
# Use a simple perspective-ish layout (doesn't need to be physically exact for the test).
CORNERS = {
    "near_left":  (300.0, 900.0),
    "near_right": (1600.0, 900.0),
    "net_left":   (500.0, 400.0),
    "net_right":  (1400.0, 400.0),
}


def test_homography_maps_corners_to_court():
    H = ground_homography(CORNERS)
    assert H is not None
    # the near_left pixel should map near world (0,44)
    cx, cy = _px_to_court(H, *CORNERS["near_left"])
    assert abs(cx - 0) < 1.0 and abs(cy - 44) < 1.0, (cx, cy)
    print("ok: ground homography maps corners to court feet")


def test_detects_a_bounce_trajectory():
    # Build a pixel trajectory that descends (y increases) then ascends — a bounce in image.
    pts = []
    f = 100
    for k, py in enumerate([300, 400, 520, 640, 700, 720, 690, 600, 480, 360]):
        px = 800 + k * 5  # drifting horizontally
        pts.append(D(f + k, float(px), float(py)))
    calls = detect_bounces_2d(pts, "cam1", CORNERS, min_drop_px=8.0)
    assert len(calls) == 1, f"expected 1 bounce, got {len(calls)}"
    b = calls[0]
    # the peak (max pixel-y = 720) is at k=5
    assert b.px_y == 720.0
    assert 0 <= b.court_x <= 20 and 0 <= b.court_y <= 44
    print(f"ok: detected bounce at court ({b.court_x:.1f},{b.court_y:.1f}) -> {b.verdict}")


def test_no_bounce_on_monotonic():
    # ball only rising (y decreasing) — no ground contact, no bounce
    pts = [D(100 + k, 800.0, float(700 - k * 20)) for k in range(10)]
    calls = detect_bounces_2d(pts, "cam1", CORNERS)
    assert len(calls) == 0
    print("ok: no bounce on a monotonic (non-bouncing) trajectory")


def test_merge_dedups_simultaneous_bounces():
    # same bounce seen by both cams at ~same ref time -> 1 merged call
    from linecall.bounce.bounce_2d import Bounce2D
    b1 = Bounce2D("cam1", 600, 800, 700, 10.0, 22.0, IN, 8.0)
    b2 = Bounce2D("cam2", 600, 700, 600, 10.2, 22.1, IN, 7.8)
    merged = merge_camera_bounces({"cam1": [b1], "cam2": [b2]},
                                  {"cam1": 60.0, "cam2": 60.0},
                                  {"cam1": 0.0, "cam2": 0.0}, "cam1")
    assert len(merged) == 1, f"expected 1 merged bounce, got {len(merged)}"
    print("ok: simultaneous bounces from both cams de-duplicated to one call")


def run_all():
    test_homography_maps_corners_to_court()
    test_detects_a_bounce_trajectory()
    test_no_bounce_on_monotonic()
    test_merge_dedups_simultaneous_bounces()
    print("\nALL 2D-BOUNCE TESTS PASSED")


if __name__ == "__main__":
    run_all()
