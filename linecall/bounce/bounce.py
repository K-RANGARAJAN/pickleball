"""Court geometry + IN/OUT classification, and a 3D z-crossing bounce detector.

classify_bounce(x, y) and the IN/OUT/CLOSE_IN constants are the pieces used by the live
pipeline (bounce_2d.py); detect_bounces() is the 3D-track detector kept for reference.
Court is x in [0, 20], y in [0, 44] feet; a ball on or inside the line is IN.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

IN = "IN"
OUT = "OUT"
CLOSE_IN = "CLOSE_IN"

COURT_X_MIN, COURT_X_MAX = 0.0, 20.0     # world feet (configs/court_geometry.json)
COURT_Y_MIN, COURT_Y_MAX = 0.0, 44.0
CLOSE_BAND_FT = 0.5
Z_GROUND_FT = 0.4        # a local-z-min counts as a bounce only at/below this height


@dataclass
class BounceCall:
    frame_idx: int
    x: float                 # world feet
    y: float
    z: float                 # height at the detected bounce (~0)
    verdict: str
    margin_ft: float         # signed distance to nearest line (+ inside, - outside)


def _dist_inside(x: float, y: float) -> float:
    """Signed distance to the court boundary: positive if inside, negative if outside.
    Magnitude is the distance to the nearest boundary edge."""
    # distance to each edge (positive means inside that edge)
    dx_min = x - COURT_X_MIN
    dx_max = COURT_X_MAX - x
    dy_min = y - COURT_Y_MIN
    dy_max = COURT_Y_MAX - y
    inside = (dx_min >= 0 and dx_max >= 0 and dy_min >= 0 and dy_max >= 0)
    nearest = min(abs(dx_min), abs(dx_max), abs(dy_min), abs(dy_max))
    if inside:
        return nearest          # how far inside the nearest line
    # outside: distance to the rectangle (negative)
    ox = max(COURT_X_MIN - x, 0.0, x - COURT_X_MAX)
    oy = max(COURT_Y_MIN - y, 0.0, y - COURT_Y_MAX)
    return -float(np.hypot(ox, oy))


def classify_bounce(x: float, y: float) -> tuple[str, float]:
    """Return (verdict, signed_margin_ft) for a bounce at ground (x, y).

    IN/OUT only — a ball on or inside the line is IN, anything outside is OUT. (No CLOSE_IN:
    the demo reports a clean IN/OUT and surfaces camera DISAGREEMENT separately.)"""
    margin = _dist_inside(x, y)
    return (OUT, margin) if margin < 0 else (IN, margin)


def _on_court(s, margin_ft: float = 6.0, max_height_ft: float = 30.0) -> bool:
    """Reject triangulation outliers: a real bounce can't be far outside the court."""
    return (COURT_X_MIN - margin_ft <= s.x <= COURT_X_MAX + margin_ft and
            COURT_Y_MIN - margin_ft <= s.y <= COURT_Y_MAX + margin_ft and
            s.z <= max_height_ft)


def detect_bounces(track: List[Optional["object"]],
                   z_ground_ft: float = Z_GROUND_FT,
                   min_drop_ft: float = 0.15,
                   lockout_frames: int = 12,
                   court_margin_ft: float = 6.0) -> List[BounceCall]:
    """Find bounces in a smoothed 3D track (states with .frame_idx,.x,.y,.z,.vz; None gaps).

    A bounce fires on either a local-z-minimum near the ground (vz flips or a real prior
    descent) or a descending ground-crossing (z: + -> -, taken at the z=0 interpolation).
    Off-court outliers are rejected; repeats within lockout_frames are suppressed.
    """
    calls: List[BounceCall] = []
    states = [s for s in track if s is not None and _on_court(s, court_margin_ft)]
    if len(states) < 3:
        return calls

    last_bounce_frame = -10 ** 9

    def emit(frame_idx, x, y, z):
        nonlocal last_bounce_frame
        if frame_idx - last_bounce_frame < lockout_frames:
            return
        verdict, margin = classify_bounce(x, y)
        calls.append(BounceCall(frame_idx=frame_idx, x=x, y=y, z=z,
                                verdict=verdict, margin_ft=margin))
        last_bounce_frame = frame_idx

    for i in range(1, len(states) - 1):
        prev, cur, nxt = states[i - 1], states[i], states[i + 1]

        # --- Signature 1: clean local minimum near ground ---
        is_local_min = cur.z <= prev.z and cur.z <= nxt.z
        near_ground = cur.z <= z_ground_ft
        vz_flip = getattr(prev, "vz", 0.0) < 0 <= getattr(nxt, "vz", 0.0)
        dropped = (max(s.z for s in states[max(0, i - 6):i + 1]) - cur.z) >= min_drop_ft
        if is_local_min and near_ground and (vz_flip or dropped):
            emit(cur.frame_idx, cur.x, cur.y, cur.z)
            continue

        # --- Signature 2: descending ground-crossing (z: + -> -) ---
        # The ball reached the surface; take (x,y) at the z=0 crossing by linear interp.
        if prev.z >= 0.0 > cur.z and (prev.z - cur.z) > 0:
            span = prev.z - cur.z
            f = prev.z / span if span > 1e-6 else 0.5      # fraction from prev to cur at z=0
            bx = prev.x + f * (cur.x - prev.x)
            by = prev.y + f * (cur.y - prev.y)
            emit(prev.frame_idx + int(round(f * (cur.frame_idx - prev.frame_idx))), bx, by, 0.0)

    return calls
