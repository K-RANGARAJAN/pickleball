"""
bounce_detector.py - bounce detection from GridTrackNet ball-tracking output.

Pure-numpy. Two detectors:
  * detect_bounces(...)            - OFFLINE, centered smoothing over the whole clip.
  * StreamingBounceDetector(...)   - CAUSAL (one-sided Kalman) for live / streaming.

A ball bounces when its pixel-y reaches a LOCAL MAXIMUM (y grows downward, so the
on-screen "lowest point") before rising again.

Offline pipeline:
  1. interpolate_gaps  - linearly fill detection gaps up to max_gap frames
  2. SEGMENT on gaps   - the trajectory is split wherever consecutive valid frames are
                         more than `split_gap` apart, and each segment is smoothed and
                         scanned independently. This stops the smoother + peak scan from
                         bridging an occlusion (the #1 false-bounce source), and a peak
                         landing right at a gap edge (a ball reappearing) is excluded.
  3. _smooth + peak    - box-filter y, local maxima with drop-magnitude guard
  4. lockout           - no new bounce within `lockout_frames` of an accepted one
  5/6. pixel_to_court / classify_bounce - court feet + IN/OUT/CLOSE_IN (needs homography)
  7. _suppress_same_side
  + SURFACE CHECK      - optional: drop bounces above `surface_min_py` (pixel) or more than
                         `max_out_of_bounds_ft` outside the court (occlusion / player FPs).

Defaults preserve the simple behaviour (segmentation is behaviour-neutral on clean data;
surface checks are off until you set them).
"""
from __future__ import annotations

import csv
from dataclasses import dataclass

import numpy as np


@dataclass
class CourtModel:
    width_ft: float
    length_ft: float
    name: str = "court"

    @property
    def corners_ft(self):
        # Order: near-left, near-right, far-right, far-left (clockwise from camera).
        return [(0.0, 0.0), (self.width_ft, 0.0),
                (self.width_ft, self.length_ft), (0.0, self.length_ft)]


PICKLEBALL = CourtModel(20.0, 44.0, "pickleball")
TENNIS_SINGLES = CourtModel(27.0, 78.0, "tennis_singles")
TENNIS_DOUBLES = CourtModel(36.0, 78.0, "tennis_doubles")


@dataclass
class BounceParams:
    max_gap: int = 20            # interpolation gap (frames)
    smooth_kernel: int = 5       # box-filter width (odd)
    drop_lookback: int = 3       # compare against i-3 and i+3
    min_drop_px: float = 6.0     # min y drop vs those neighbours
    lockout_frames: int = 15     # no new bounce within this many frames
    close_margin_ft: float = 0.5
    suppress_same_side: bool = True
    # --- hardening (opt-in) ---
    split_gap: int | None = None         # split trajectory on gaps > this (default = max_gap)
    surface_min_py: float | None = None  # pixel: drop bounces with px_y < this (too high in frame)
    max_out_of_bounds_ft: float | None = None  # court: drop bounces this far outside the court


@dataclass
class Bounce:
    frame: int
    px_x: float
    px_y: float
    court_x: float | None = None
    court_y: float | None = None
    classification: str = "UNKNOWN"
    side: str | None = None
    suppressed: bool = False


def interpolate_gaps(coords, max_gap: int = 20):
    coords = np.asarray(coords, dtype=float)
    xs, ys = coords[:, 0].copy(), coords[:, 1].copy()
    detected = ~((coords[:, 0] == 0) & (coords[:, 1] == 0))
    valid = detected.copy()
    det_idx = np.where(detected)[0]
    for a, b in zip(det_idx[:-1], det_idx[1:]):
        gap = b - a
        if gap <= 1 or gap > max_gap:
            continue
        for f in range(a + 1, b):
            t = (f - a) / gap
            xs[f] = xs[a] + t * (xs[b] - xs[a])
            ys[f] = ys[a] + t * (ys[b] - ys[a])
            valid[f] = True
    return xs, ys, valid


def _smooth(y, kernel: int = 5):
    y = np.asarray(y, dtype=float)
    if kernel <= 1 or y.size < 2:
        return y
    pad = kernel // 2
    padded = np.pad(y, pad, mode="edge")
    return np.convolve(padded, np.ones(kernel) / kernel, mode="valid")


def _segments(frames, split_gap):
    """Yield (start, end) index ranges of runs where consecutive frame gaps <= split_gap."""
    if frames.size == 0:
        return []
    segs, s = [], 0
    for k in range(1, frames.size):
        if frames[k] - frames[k - 1] > split_gap:
            segs.append((s, k)); s = k
    segs.append((s, frames.size))
    return segs


def pixel_to_court(homography, px, py):
    v = np.asarray(homography, dtype=float) @ np.array([px, py, 1.0])
    return float(v[0] / v[2]), float(v[1] / v[2])


def classify_bounce(court_x, court_y, court: CourtModel, close_margin: float = 0.5):
    inside = (0.0 <= court_x <= court.width_ft) and (0.0 <= court_y <= court.length_ft)
    if not inside:
        return "OUT"
    margin = min(court_x, court.width_ft - court_x,
                 court_y, court.length_ft - court_y)
    return "CLOSE_IN" if margin < close_margin else "IN"


def _out_of_bounds_ft(cx, cy, court):
    ox = max(0.0, -cx, cx - court.width_ft)
    oy = max(0.0, -cy, cy - court.length_ft)
    return max(ox, oy)


def compute_homography(image_corners, court: CourtModel | None = None, court_corners=None):
    """4-point ground homography pixels -> court feet (numpy DLT).
    image_corners order: near-left, near-right, far-right, far-left."""
    if court_corners is None:
        if court is None:
            raise ValueError("Provide either court or court_corners.")
        court_corners = court.corners_ft
    src = np.asarray(image_corners, dtype=float)
    dst = np.asarray(court_corners, dtype=float)
    if src.shape != (4, 2) or dst.shape != (4, 2):
        raise ValueError("Need exactly 4 image corners and 4 court corners.")
    A, b = [], []
    for (x, y), (u, v) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y]); b.append(u)
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y]); b.append(v)
    h = np.linalg.solve(np.asarray(A, dtype=float), np.asarray(b, dtype=float))
    return np.array([[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1.0]])


def _suppress_same_side(bounces):
    last_side = None
    for b in bounces:
        if b.classification == "OUT":
            last_side = None
            continue
        if last_side is not None and b.side == last_side:
            b.suppressed = True
        else:
            b.suppressed = False
            last_side = b.side


def _finalize(raw, homography, court, p):
    """raw: list of (frame, px_x, px_y) -> map, surface-check, lockout, suppress."""
    mapped = []
    for fr, px, py in raw:
        b = Bounce(frame=fr, px_x=px, px_y=py)
        if homography is not None and court is not None:
            b.court_x, b.court_y = pixel_to_court(homography, px, py)
            if p.max_out_of_bounds_ft is not None and \
                    _out_of_bounds_ft(b.court_x, b.court_y, court) > p.max_out_of_bounds_ft:
                continue
            b.classification = classify_bounce(b.court_x, b.court_y, court, p.close_margin_ft)
            b.side = "near" if b.court_y < court.length_ft / 2 else "far"
        mapped.append(b)
    kept, last = [], -(10 ** 9)
    for b in mapped:
        if b.frame - last <= p.lockout_frames:
            continue
        last = b.frame
        kept.append(b)
    if homography is not None and court is not None and p.suppress_same_side:
        _suppress_same_side(kept)
    return kept


def detect_bounces(coords, homography=None, court: CourtModel | None = None,
                   params: BounceParams | None = None):
    """OFFLINE ground-bounce detection. Returns list[Bounce] in chronological order."""
    p = params or BounceParams()
    xs, ys, valid = interpolate_gaps(coords, p.max_gap)
    frames = np.where(valid)[0]
    lb = p.drop_lookback
    if frames.size < 2 * lb + 1:
        return []
    cx, cy = xs[frames], ys[frames]
    split_gap = p.split_gap if p.split_gap is not None else p.max_gap

    raw = []
    for s, e in _segments(frames, split_gap):
        if e - s < 2 * lb + 1:
            continue
        sf, scx, scy = frames[s:e], cx[s:e], cy[s:e]
        sy = _smooth(scy, p.smooth_kernel)
        for j in range(lb, (e - s) - lb):
            if not (sy[j] >= sy[j - 1] and sy[j] >= sy[j + 1]):
                continue
            if sy[j] - min(sy[j - lb], sy[j + lb]) < p.min_drop_px:
                continue
            if p.surface_min_py is not None and scy[j] < p.surface_min_py:
                continue
            raw.append((int(sf[j]), float(scx[j]), float(scy[j])))
    return _finalize(raw, homography, court, p)


class StreamingBounceDetector:
    """CAUSAL bounce detector for live / streaming.

    A constant-velocity Kalman filter tracks vertical position y and velocity vy from
    ONLY past frames. Missing detections are handled by predict-only steps (the filter
    coasts on its velocity estimate; no linear interpolation, so no interpolation-induced
    false bounce). A bounce fires when the estimated vy flips from descending (+) to
    ascending (-) at a y-maximum, with a prominence (min_drop), debounce (lockout) and
    optional surface checks. After `max_misses` consecutive gaps the filter resets
    (trajectory split). Feed frames in order with update().
    """

    def __init__(self, params: BounceParams | None = None, homography=None,
                 court: CourtModel | None = None, process_var: float = 3.0,
                 meas_var: float = 4.0, max_misses: int | None = None,
                 min_descent_vy: float = 0.5):
        self.p = params or BounceParams()
        self.homography = homography
        self.court = court
        self.q = float(process_var)
        self.r = float(meas_var)
        self.max_misses = self.p.max_gap if max_misses is None else max_misses
        self.min_descent_vy = float(min_descent_vy)
        self.reset()

    def reset(self):
        self.s = None            # [y, vy]
        self.P = None
        self.prev_vy = None
        self.apex_y = None
        self.last_x = None
        self.last_bounce_frame = -(10 ** 9)
        self.misses = 0

    def _init(self, y):
        self.s = np.array([y, 0.0])
        self.P = np.array([[self.r, 0.0], [0.0, 100.0]])
        self.prev_vy = 0.0
        self.apex_y = y

    def update(self, frame, x, y, detected=True):
        """Feed one frame. Returns a Bounce when one fires this frame, else None."""
        measured = detected and not (x == 0 and y == 0)
        if measured:
            self.last_x = x
            self.misses = 0
            if self.s is None:
                self._init(y)
                return None
        else:
            self.misses += 1
            if self.s is None:
                return None
            if self.misses > self.max_misses:
                self.reset()
                return None

        # predict (dt = 1, constant velocity)
        F = np.array([[1.0, 1.0], [0.0, 1.0]])
        Q = self.q * np.array([[0.25, 0.5], [0.5, 1.0]])
        self.s = F @ self.s
        self.P = F @ self.P @ F.T + Q

        # update
        if measured:
            H = np.array([[1.0, 0.0]])
            innov = y - (H @ self.s)[0]
            S = (H @ self.P @ H.T)[0, 0] + self.r
            K = (self.P @ H.T).flatten() / S
            self.s = self.s + K * innov
            self.P = (np.eye(2) - np.outer(K, H[0])) @ self.P

        fy, vy = self.s[0], self.s[1]
        bounce = None
        if self.prev_vy is not None:
            if self.prev_vy < 0 and vy >= 0:                 # apex (y minimum)
                self.apex_y = fy
            elif self.prev_vy >= self.min_descent_vy and vy < 0:   # bounce (y maximum)
                if self.apex_y is None:
                    self.apex_y = fy
                prominence = fy - self.apex_y
                if prominence >= self.p.min_drop_px and \
                        frame - self.last_bounce_frame > self.p.lockout_frames:
                    bounce = self._emit(frame, fy)
        self.prev_vy = vy
        return bounce

    def _emit(self, frame, fy):
        px = self.last_x if self.last_x is not None else 0.0
        if self.p.surface_min_py is not None and fy < self.p.surface_min_py:
            return None
        b = Bounce(frame=int(frame), px_x=float(px), px_y=float(fy))
        if self.homography is not None and self.court is not None:
            b.court_x, b.court_y = pixel_to_court(self.homography, px, fy)
            if self.p.max_out_of_bounds_ft is not None and \
                    _out_of_bounds_ft(b.court_x, b.court_y, self.court) > self.p.max_out_of_bounds_ft:
                return None
            b.classification = classify_bounce(b.court_x, b.court_y, self.court, self.p.close_margin_ft)
            b.side = "near" if b.court_y < self.court.length_ft / 2 else "far"
        self.last_bounce_frame = frame
        return b


def detect_bounces_streaming(coords, homography=None, court=None, params=None, **kwargs):
    """Run the causal StreamingBounceDetector over a coords list (offline convenience)."""
    det = StreamingBounceDetector(params=params, homography=homography, court=court, **kwargs)
    out = []
    for i, (x, y) in enumerate(coords):
        b = det.update(i, x, y, detected=not (x == 0 and y == 0))
        if b is not None:
            out.append(b)
    return out


def load_track_csv(path):
    coords = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            x, y = float(row.get("X", 0) or 0), float(row.get("Y", 0) or 0)
            vis = row.get("Visibility")
            if vis is not None and vis != "" and int(float(vis)) == 0:
                x, y = 0.0, 0.0
            coords.append((x, y))
    return coords


def save_bounces_csv(bounces, path, include_suppressed=False):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "px_x", "px_y", "court_x", "court_y",
                    "classification", "side", "suppressed"])
        for b in bounces:
            if b.suppressed and not include_suppressed:
                continue
            w.writerow([b.frame, round(b.px_x, 1), round(b.px_y, 1),
                        None if b.court_x is None else round(b.court_x, 2),
                        None if b.court_y is None else round(b.court_y, 2),
                        b.classification, b.side, b.suppressed])
