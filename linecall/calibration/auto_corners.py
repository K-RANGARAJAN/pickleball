"""Automatic court-corner detection from a single frame — no trained model.

Classical CV: detect line segments, cluster them into the two near-parallel families that
make up a rectangular court (sidelines vs baselines), take the outermost line of each family
as the court boundary, and intersect them into the 4 outer corners. The caller verifies the
quad with the ground-homography GOOD/BAD check (solve_camera_pose_auto), so a wrong guess is
rejected rather than trusted — the live app then falls back to manual drag.

Works best on a clean, well-lit court where the painted outer lines are the strongest long
segments. Clutter (players, shadows, the net, kitchen lines) is the failure mode; that's why
detection is always verified, never trusted blind.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def _segments(gray: "np.ndarray") -> "np.ndarray":
    import cv2
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    lines = lsd.detect(gray)[0]
    if lines is None:
        return np.empty((0, 4))
    return lines.reshape(-1, 4)


def _long_enough(seg: "np.ndarray", min_len: float) -> "np.ndarray":
    d = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
    return seg[d >= min_len]


def _angle_deg(seg: "np.ndarray") -> "np.ndarray":
    a = np.degrees(np.arctan2(seg[:, 3] - seg[:, 1], seg[:, 2] - seg[:, 0]))
    return (a + 180.0) % 180.0


def _extreme_line(group: "np.ndarray", axis: int, take_min: bool) -> Optional["np.ndarray"]:
    """Outermost segment of a family by its midpoint along `axis` (0=x, 1=y)."""
    if len(group) == 0:
        return None
    mids = (group[:, axis] + group[:, axis + 2]) / 2.0
    idx = int(np.argmin(mids) if take_min else np.argmax(mids))
    return group[idx]


def _line_from_seg(seg: "np.ndarray") -> Tuple[float, float, float]:
    """Segment endpoints -> line ax+by+c=0, normalized."""
    x1, y1, x2, y2 = seg
    a, b = y2 - y1, x1 - x2
    c = -(a * x1 + b * y1)
    n = np.hypot(a, b) or 1.0
    return a / n, b / n, c / n


def _intersect(l1, l2) -> Optional[Tuple[float, float]]:
    a1, b1, c1 = l1
    a2, b2, c2 = l2
    d = a1 * b2 - a2 * b1
    if abs(d) < 1e-6:
        return None
    x = (b1 * c2 - b2 * c1) / d
    y = (c1 * a2 - c2 * a1) / d
    return float(x), float(y)


def _split_two_directions(seg: "np.ndarray") -> Optional[Tuple["np.ndarray", "np.ndarray"]]:
    """Cluster segments into the two dominant orientation families (a court seen from any
    angle has two: the baselines and the sidelines), via 1-D k-means on angle. Returns the
    two groups, or None if they aren't well separated."""
    ang = _angle_deg(seg)
    a = np.sort(ang)
    gaps = np.diff(np.concatenate([a, [a[0] + 180.0]]))
    split = (a[np.argmax(gaps)] + gaps.max() / 2.0) % 180.0
    c1, c2 = split, (split + 90.0) % 180.0
    for _ in range(8):
        d1 = np.minimum(np.abs(ang - c1), 180 - np.abs(ang - c1))
        d2 = np.minimum(np.abs(ang - c2), 180 - np.abs(ang - c2))
        m = d1 < d2
        if m.sum() < 2 or (~m).sum() < 2:
            return None
        c1, c2 = ang[m].mean(), ang[~m].mean()
    sep = min(abs(c1 - c2), 180 - abs(c1 - c2))
    if sep < 20.0:
        return None
    return seg[m], seg[~m]


def detect_court_corners(frame_bgr: "np.ndarray") -> Optional[dict]:
    """Best-effort 4 outer court corners from one BGR frame, or None if no plausible quad.

    Angle-agnostic: clusters segments into the court's two line families (whatever their
    image orientation), takes the outer line of each family, intersects into 4 corners.
    Returns {"far_left","far_right","near_right","near_left": (x,y)} in pixels — ordered by
    image position; the caller's solve_camera_pose_auto relabels to true world identity.
    """
    h, w = frame_bgr.shape[:2]
    import cv2
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    seg = _long_enough(_segments(gray), min_len=0.12 * min(h, w))
    if len(seg) < 4:
        return None
    fam = _split_two_directions(seg)
    if fam is None:
        return None
    fam_a, fam_b = fam

    # For each family, the court boundary is the pair of OUTERMOST lines. Order each family
    # along the perpendicular direction and take the two extremes.
    def outer_pair(group):
        mids = np.stack([(group[:, 0] + group[:, 2]) / 2, (group[:, 1] + group[:, 3]) / 2], 1)
        d = group[:, 2:4] - group[:, 0:2]
        normal = np.array([-d.mean(0)[1], d.mean(0)[0]])
        normal = normal / (np.hypot(*normal) or 1.0)
        proj = mids @ normal
        return group[int(np.argmin(proj))], group[int(np.argmax(proj))]

    a0, a1 = outer_pair(fam_a)
    b0, b1 = outer_pair(fam_b)
    la0, la1 = _line_from_seg(a0), _line_from_seg(a1)
    lb0, lb1 = _line_from_seg(b0), _line_from_seg(b1)

    quad = [_intersect(la0, lb0), _intersect(la0, lb1),
            _intersect(la1, lb1), _intersect(la1, lb0)]
    if any(p is None for p in quad):
        return None
    margin = 0.10
    for (x, y) in quad:
        if x < -margin * w or x > (1 + margin) * w or y < -margin * h or y > (1 + margin) * h:
            return None

    # order the 4 quad points by image position into FL/FR/NR/NL (top-left CW)
    pts = np.array(quad)
    cx, cy = pts.mean(0)
    def label(p):
        return ("far" if p[1] < cy else "near") + ("_left" if p[0] < cx else "_right")
    out = {}
    for p in pts:
        out[label(p)] = (float(p[0]), float(p[1]))
    if set(out) != {"far_left", "far_right", "near_right", "near_left"}:
        return None
    return out
