"""Per-camera 2D bounce detection + IN/OUT.

A bounce = the ball's pixel-y reaches a local maximum (lowest point on screen, since image
y grows downward) after a descent and before an ascent. The bounce pixel maps to court feet
via a ground homography (valid because a bounce is on the z=0 plane) and is classified
IN / OUT / CLOSE_IN against the court lines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from .bounce import classify_bounce, IN
from ..calibration.pose import ALL_WORLD_POINTS


@dataclass
class Bounce2D:
    camera_id: str
    frame_idx: int
    px_x: float
    px_y: float
    court_x: float
    court_y: float
    verdict: str
    margin_ft: float
    confidence: float = 1.0   # 0..1, how sure the bounce is on the called side of the line


def ground_homography(named_corners_px: dict) -> Optional[np.ndarray]:
    """Homography mapping image pixels -> court ground coords (x,y) in feet.

    Built from the same named court points used for calibration. Valid only for points ON
    the ground plane (z=0) — which is exactly where a bounce is, so it's the right tool."""
    if cv2 is None:
        return None
    names = [n for n in named_corners_px if n in ALL_WORLD_POINTS]
    if len(names) < 4:
        return None
    img = np.array([named_corners_px[n] for n in names], np.float64)
    wld = np.array([[ALL_WORLD_POINTS[n][0], ALL_WORLD_POINTS[n][1]] for n in names],
                   np.float64)
    H, _ = cv2.findHomography(img, wld)
    return H


def _px_to_court(H: np.ndarray, x: float, y: float) -> Tuple[float, float]:
    p = H @ np.array([x, y, 1.0])
    return float(p[0] / p[2]), float(p[1] / p[2])


def _ft_per_px(H: np.ndarray, x: float, y: float) -> float:
    """Court-feet moved per 1px at this image point (large = foreshortened far court)."""
    p0 = _px_to_court(H, x, y)
    px = _px_to_court(H, x + 1.0, y)
    py = _px_to_court(H, x, y + 1.0)
    d1 = ((px[0] - p0[0]) ** 2 + (px[1] - p0[1]) ** 2) ** 0.5
    d2 = ((py[0] - p0[0]) ** 2 + (py[1] - p0[1]) ** 2) ** 0.5
    return 0.5 * (d1 + d2)


def call_confidence(H: np.ndarray, px_x: float, px_y: float, margin_ft: float,
                    reproj_px: float, label_px: float = 2.0) -> float:
    """Confidence (0..1) that a bounce is on the CALLED side of the nearest line.

    Positional uncertainty = (calibration reproj + click error) in px, mapped to feet AT this
    point (so foreshortened far-court positions and a looser-calibrated camera get a wider
    sigma). Confidence = Phi(margin / sigma): deep in/out -> ~1, on the line -> ~0.5.
    """
    import math
    sigma_ft = (reproj_px + label_px) * _ft_per_px(H, px_x, px_y)
    if sigma_ft <= 1e-9:
        return 1.0
    return 0.5 * (1.0 + math.erf((abs(margin_ft) / sigma_ft) / (2 ** 0.5)))


def bounces_from_points(points: List[Tuple[int, float, float]], camera_id: str,
                        named_corners_px: dict,
                        court_margin_ft: float = 4.0,
                        reproj_px: float = 0.0) -> List[Bounce2D]:
    """Map hand-marked bounce-contact pixels directly to IN/OUT calls (no trajectory search).

    Each (frame, px_x, px_y) -> court feet via homography -> classify. Off-court points are
    dropped. reproj_px > 0 attaches a confidence to each call.
    """
    H = ground_homography(named_corners_px)
    if H is None:
        return []
    calls: List[Bounce2D] = []
    for f, px, py in sorted(points):
        cx, cy = _px_to_court(H, px, py)
        if not (-court_margin_ft <= cx <= 20 + court_margin_ft and
                -court_margin_ft <= cy <= 44 + court_margin_ft):
            continue
        verdict, margin = classify_bounce(cx, cy)
        conf = call_confidence(H, px, py, margin, reproj_px) if reproj_px > 0 else 1.0
        calls.append(Bounce2D(camera_id, int(f), float(px), float(py),
                              cx, cy, verdict, margin, conf))
    return calls


def _fill_gaps(pts: list, max_interp_gap: int) -> list:
    """Linearly interpolate (x, y) through gaps <= max_interp_gap frames.

    Gaps larger than max_interp_gap are left as breaks — the bounce detector's
    frame-gap check will reject peaks that straddle them.
    """
    if len(pts) < 2:
        return pts
    out: list = [pts[0]]
    for i in range(1, len(pts)):
        fi_prev, xi_prev, yi_prev = out[-1]
        fi_curr, xi_curr, yi_curr = pts[i]
        gap = fi_curr - fi_prev
        if 1 < gap <= max_interp_gap:
            for g in range(1, gap):
                t = g / gap
                out.append((fi_prev + g,
                             xi_prev + t * (xi_curr - xi_prev),
                             yi_prev + t * (yi_curr - yi_prev)))
        out.append((fi_curr, xi_curr, yi_curr))
    return out


def _remove_angle_outliers(pts: list, max_angle_deg: float = 170.0) -> list:
    """Drop detected points whose entry and exit vectors nearly reverse direction.

    A near-180° reversal (≥170°) means the tracker briefly locked onto the wrong object
    and snapped back. Real bounces reverse y but keep x-direction, giving ~90-160° turns
    well below this threshold.
    """
    if len(pts) < 3:
        return pts
    cos_thresh = np.cos(np.radians(max_angle_deg))
    clean = [pts[0]]
    i = 1
    while i < len(pts):
        if i + 1 < len(pts):
            a = clean[-1]
            b = pts[i]
            c = pts[i + 1]
            vx1, vy1 = b[1] - a[1], b[2] - a[2]
            vx2, vy2 = c[1] - b[1], c[2] - b[2]
            n1 = (vx1 ** 2 + vy1 ** 2) ** 0.5
            n2 = (vx2 ** 2 + vy2 ** 2) ** 0.5
            if n1 > 0 and n2 > 0 and (vx1 * vx2 + vy1 * vy2) / (n1 * n2) < cos_thresh:
                i += 1  # b is a spike — skip it, re-evaluate with next point as middle
                continue
        clean.append(pts[i])
        i += 1
    return clean


def detect_bounces_2d(detections: List, camera_id: str,
                      named_corners_px: dict,
                      min_drop_px: float = 6.0,
                      max_frame_gap: int = 20,
                      lockout_frames: int = 20,
                      court_margin_ft: float = 4.0,
                      interp_gap: int = 20,
                      smooth_k: int = 5,
                      max_angle_deg: float = 170.0) -> List[Bounce2D]:
    """Find bounces from ONE camera's 2D ball trajectory (pixel-y local maxima).

    detections: objects with .frame_idx, .px_x, .px_y, .is_detected.
    Gaps <= interp_gap are linearly filled; y is box-smoothed (smooth_k) before the local-max
    search; peaks are mapped to court feet and classified, off-court ones rejected.
    """
    H = ground_homography(named_corners_px)
    if H is None:
        return []

    pts = [(d.frame_idx, d.px_x, d.px_y) for d in detections
           if getattr(d, "is_detected", True)]
    pts.sort()
    if len(pts) < 5:
        return []

    pts = _remove_angle_outliers(pts, max_angle_deg)
    pts = _fill_gaps(pts, max_interp_gap=interp_gap)

    fr = np.array([p[0] for p in pts])
    xs = np.array([p[1] for p in pts])
    ys_orig = np.array([p[2] for p in pts])   # interpolated, pre-smooth — used for storage
    if smooth_k > 1 and len(ys_orig) >= smooth_k:
        pad = smooth_k // 2
        kernel = np.ones(smooth_k) / smooth_k
        ys = np.convolve(np.pad(ys_orig, pad, mode="edge"), kernel, mode="valid")
        ys = ys[:len(ys_orig)]  # guard against off-by-one from odd kernels
    else:
        ys = ys_orig

    NET_Y = 22.0   # net line splits the court into far (y<22) and near (y>22) sides

    calls: List[Bounce2D] = []
    last_bounce = -10 ** 9
    last_in_side = None    # side ('far'/'near') of the previous COUNTED in-bounds bounce
    for i in range(2, len(pts) - 2):
        # require local continuity around the candidate peak
        if fr[i] - fr[i - 1] > max_frame_gap or fr[i + 1] - fr[i] > max_frame_gap:
            continue
        # pixel-y local maximum (lowest point in image = ground contact)
        if not (ys[i] >= ys[i - 1] and ys[i] >= ys[i + 1]):
            continue
        # genuine descent into the peak and ascent out of it (filters jitter)
        lo = i - 3 if i - 3 >= 0 else 0
        hi = i + 3 if i + 3 < len(ys) else len(ys) - 1
        if (ys[i] - min(ys[lo], ys[hi])) < min_drop_px:
            continue
        if fr[i] - last_bounce < lockout_frames:
            continue

        cx, cy = _px_to_court(H, xs[i], ys_orig[i])
        # reject homography blow-ups far off court
        if not (-court_margin_ft <= cx <= 20 + court_margin_ft and
                -court_margin_ft <= cy <= 44 + court_margin_ft):
            continue
        verdict, margin = classify_bounce(cx, cy)

        # Per-side suppression: ignore repeat IN bounces on the same net side (dribble/jitter)
        # until the ball bounces on the other side; an OUT resets the tracker.
        side = "far" if cy < NET_Y else "near"
        if verdict == IN:
            if side == last_in_side:
                continue                    # suppressed same-side repeat
            last_in_side = side
        else:
            last_in_side = None             # OUT resets — next IN on either side counts

        calls.append(Bounce2D(camera_id, int(fr[i]), float(xs[i]), float(ys_orig[i]),
                              cx, cy, verdict, margin))
        last_bounce = fr[i]
    return calls


def estimate_offset_frames(cam1: List[Bounce2D], cam2: List[Bounce2D], fps: float,
                           pos_tol_ft: float = 4.0, time_tol_s: float = 0.4,
                           search: int = 180) -> Tuple[int, int]:
    """Estimate the cam1<->cam2 frame offset from the bounces themselves.

    A bounce both cameras saw lands at the SAME court (x,y) at the SAME real time. We try
    each candidate offset D (cam2_frame = cam1_frame + D) and count cam1/cam2 bounce pairs
    that agree in BOTH court position and time; the D with the most agreements is the offset.
    Self-calibrating — no clap/audio sync needed. Returns (best_offset_frames, n_matched).
    """
    c1 = [(b.frame_idx, b.court_x, b.court_y) for b in cam1]
    c2 = [(b.frame_idx, b.court_x, b.court_y) for b in cam2]
    best = (0, -1)
    for D in range(-search, search + 1):
        matched, used = 0, set()
        for f1, x1, y1 in c1:
            for j, (f2, x2, y2) in enumerate(c2):
                if j in used:
                    continue
                if abs(f1 - (f2 - D)) / fps < time_tol_s and \
                        ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5 < pos_tol_ft:
                    matched += 1
                    used.add(j)
                    break
        if matched > best[1]:
            best = (D, matched)
    return best


def merge_bounces_union(cam1: List[Bounce2D], cam2: List[Bounce2D], fps: float,
                        cam_y_ft: Dict[str, float],
                        offset_frames: Optional[int] = None,
                        pos_tol_ft: float = 4.0, time_tol_s: float = 0.4
                        ) -> Tuple[List[Bounce2D], dict]:
    """Union of both cameras' bounces on cam1's timeline, keeping single-camera bounces.

    A bounce both cameras caught (same court position AND time, after aligning by
    offset_frames) is merged to one call placed by whichever camera is closer to it; cam1-only
    and cam2-only bounces are kept. offset_frames is estimated from the data if None.
    Returns (merged_sorted_by_frame, report).
    """
    if offset_frames is None:
        offset_frames, _ = estimate_offset_frames(cam1, cam2, fps, pos_tol_ft, time_tol_s)

    def closeness(b: "Bounce2D") -> float:
        cy = cam_y_ft.get(b.camera_id)
        return abs(cy - b.court_y) if cy is not None else 1e9

    # cam2 bounces, restamped onto cam1's timeline
    c2_ref = [(b.frame_idx - offset_frames, b) for b in cam2]
    matched2 = set()
    merged: List[Bounce2D] = []
    shared_frames: List[int] = []
    n_shared = 0
    for b1 in cam1:
        partner = None
        for k, (rf2, b2) in enumerate(c2_ref):
            if k in matched2:
                continue
            if abs(b1.frame_idx - rf2) / fps < time_tol_s and \
                    ((b1.court_x - b2.court_x) ** 2 +
                     (b1.court_y - b2.court_y) ** 2) ** 0.5 < pos_tol_ft:
                partner = (k, rf2, b2)
                break
        if partner is None:
            merged.append(b1)
            continue
        k, rf2, b2 = partner
        matched2.add(k)
        n_shared += 1
        winner = b2 if closeness(b2) < closeness(b1) else b1
        winner.frame_idx = b1.frame_idx          # place on cam1 timeline
        shared_frames.append(b1.frame_idx)
        merged.append(winner)
    n_cam2_only = 0
    for k, (rf2, b2) in enumerate(c2_ref):
        if k in matched2:
            continue
        b2.frame_idx = int(rf2)
        merged.append(b2)
        n_cam2_only += 1

    merged.sort(key=lambda b: b.frame_idx)
    report = {"offset_frames": offset_frames, "shared": n_shared,
              "cam1_only": len(cam1) - n_shared, "cam2_only": n_cam2_only,
              "total": len(merged), "shared_frames": shared_frames}
    return merged, report


def merge_camera_bounces(per_cam: Dict[str, List[Bounce2D]],
                         fps: Dict[str, float],
                         offset_s: Dict[str, float],
                         ref_cam: str,
                         dedup_window_s: float = 0.25,
                         cam_y_ft: Optional[Dict[str, float]] = None) -> List[Bounce2D]:
    """Combine both cameras' bounces on one timeline, de-duplicating within dedup_window_s
    and keeping the call from whichever camera is closer to the bounce (cam_y_ft). Older
    time-only merge; merge_bounces_union is the position+time version. Sorted by ref time."""
    fps_ref = fps[ref_cam]
    cam_y_ft = cam_y_ft or {}

    def closeness(b: "Bounce2D") -> float:
        cy = cam_y_ft.get(b.camera_id)
        return abs(cy - b.court_y) if cy is not None else 1e9  # smaller = closer = better

    events = []
    for cam, bounces in per_cam.items():
        for b in bounces:
            t_ref = b.frame_idx / fps[cam] - offset_s.get(cam, 0.0)
            events.append((t_ref, b))
    events.sort(key=lambda e: e[0])

    merged: List[Bounce2D] = []
    last_t = -10 ** 9
    for t_ref, b in events:
        if t_ref - last_t < dedup_window_s and merged:
            # same physical bounce — keep whichever camera is closer to it (sees it best)
            if closeness(b) < closeness(merged[-1]):
                b.frame_idx = int(round(t_ref * fps_ref))
                merged[-1] = b
            continue
        b.frame_idx = int(round(t_ref * fps_ref))
        merged.append(b)
        last_t = t_ref
    return merged
