"""Per-camera calibration: 4 clicked court corners -> camera pose via cv2.solvePnP.

Output is a CameraPose (K, R, t); ALL_WORLD_POINTS feeds the ground homography in bounce_2d.
World frame (configs/court_geometry.json): x 0..20 ft (sidelines), y 0..44 ft (baselines),
z up, ground z=0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

# The four OUTER court corners in world feet (the model detects these 4).
# Order here is far-left, far-right, near-right, near-left (a consistent CW loop in world).
WORLD_CORNERS = {
    "far_left":  (0.0,  0.0,  0.0),
    "far_right": (20.0, 0.0,  0.0),
    "near_right": (20.0, 44.0, 0.0),
    "near_left": (0.0,  44.0, 0.0),
}

# Net-post ground positions (where each post meets the court surface), at the
# net line y=22 (the court midline). ADDED for diagonal rigs where the far
# baseline (y=0) is out of frame — e.g. a camera that only sees its own half
# of the court plus an unrelated adjacent court beyond the net. These two
# points plus the two NEAR baseline corners give 4 known, visible, coplanar
# (z=0) ground points — solvePnP doesn't require the 4 points to be the
# court's own outer corners, just 4 known non-collinear ground points.
NET_POSTS = {
    "net_left":  (0.0,  22.0, 0.0),
    "net_right": (20.0, 22.0, 0.0),
}

# All known world ground-points a camera might be calibrated from.
# solve_camera_pose_named accepts any 4-key subset of this dict.
ALL_WORLD_POINTS = {**WORLD_CORNERS, **NET_POSTS}


@dataclass
class CameraPose:
    """A camera's calibrated pose in the shared world frame."""
    camera_id: str
    K: np.ndarray            # 3x3 intrinsics
    R: np.ndarray            # 3x3 rotation (world -> camera)
    t: np.ndarray            # 3x1 translation (world -> camera)
    reproj_error_px: float   # mean reprojection error of the 4 court corners (validation)
    baseline: str = ""       # "near" or "far" — which baseline the camera sits behind
    camera_position_ft: Optional[Tuple[float, float, float]] = None  # camera center in world

    @property
    def projection_matrix(self) -> np.ndarray:
        """3x4 P = K [R | t] — used directly by cv2.triangulatePoints."""
        return self.K @ np.hstack([self.R, self.t.reshape(3, 1)])

    @property
    def well_calibrated(self) -> bool:
        # A few px of corner reprojection error is fine; large error => bad detection/pose.
        return self.reproj_error_px <= 8.0


def default_intrinsics(image_w: int, image_h: int,
                       fov_deg: float = 60.0) -> np.ndarray:
    """A reasonable phone-camera intrinsics guess when we don't have a calibration file.

    solvePnP needs K. With only 4 coplanar points we can't also solve focal length
    reliably, so we assume a sensible focal from a typical horizontal FOV and the image
    size. This is good enough for amateur-grade calibration; a one-time checkerboard
    calibration per phone model would tighten it later.
    """
    f = (image_w / 2.0) / np.tan(np.radians(fov_deg / 2.0))
    return np.array([[f, 0, image_w / 2.0],
                     [0, f, image_h / 2.0],
                     [0, 0, 1.0]], dtype=np.float64)


def _image_corners_to_world(corners_tl_tr_br_bl: np.ndarray,
                            camera_behind: str) -> np.ndarray:
    """Map the model's TL,TR,BR,BL image corners to world corners.

    A camera mounted BEHIND a baseline looks ACROSS the court, so the *opposite* baseline
    appears at the top of its image. The mapping below is verified empirically by projecting
    the known world corners through synthetic cameras at each baseline:

      - camera behind the FAR baseline (looking toward near): the NEAR baseline is at the
        top of the image. image TL=near_left, TR=near_right, BR=far_right, BL=far_left.
      - camera behind the NEAR baseline (looking toward far): the FAR baseline is at the
        top, and left/right are mirrored. image TL=far_right, TR=far_left, BR=near_left,
        BL=near_right.

    camera_behind: "far" or "near".
    """
    if camera_behind == "far":
        order = ["near_left", "near_right", "far_right", "far_left"]
    elif camera_behind == "near":
        order = ["far_right", "far_left", "near_left", "near_right"]
    else:
        raise ValueError("camera_behind must be 'far' or 'near'")
    return np.array([WORLD_CORNERS[k] for k in order], dtype=np.float64)


def _reproj_err(world_corners, image_corners, rvec, tvec, K, dist):
    proj, _ = cv2.projectPoints(world_corners, rvec, tvec, K, dist)
    return float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - image_corners, axis=1)))


def _solve_candidates(image_corners, world_corners, K):
    """All candidate (R, t, err) poses for a fixed world<->image correspondence."""
    dist = np.zeros((4, 1))
    cands = []
    ok, rvec, tvec = cv2.solvePnP(world_corners, image_corners, K, dist,
                                  flags=cv2.SOLVEPNP_ITERATIVE)
    if ok:
        cands.append((rvec, tvec))
    try:
        _, rvecs, tvecs, _ = cv2.solvePnPGeneric(
            world_corners, image_corners, K, dist, flags=cv2.SOLVEPNP_IPPE)
        cands.extend(zip(rvecs, tvecs))
    except cv2.error:
        pass
    out = []
    for rvec, tvec in cands:
        R, _ = cv2.Rodrigues(rvec)
        err = _reproj_err(world_corners, image_corners, rvec, tvec, K, dist)
        out.append((R, tvec, err))
    return out


def _solve_once(image_corners: np.ndarray, world_corners: np.ndarray,
                K: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Solve camera pose from 4 coplanar court points, returning a PHYSICALLY VALID pose
    (camera above the ground plane, z>0).

    Two ambiguities are handled:
      1. Planar 2-fold (IPPE mirror): solvePnPGeneric returns both; we prefer z>0.
      2. World-winding flip: for some real corner layouts BOTH IPPE solutions come out
         below ground (verified on real diagonal-corner footage — perfect 1.4px reproj
         but camera at z=-5.4 ft). That means the world frame's in-plane handedness is
         flipped relative to the image. We detect "no above-ground solution" and retry
         with the world X-axis mirrored about the court centre (x -> 20 - x), which
         produces the correct above-ground pose at the same reprojection error; the
         recovered R,t are then mapped back into the canonical (un-mirrored) world frame
         so all downstream geometry stays in the standard court coordinates.

    Returns (R, t, mean_reproj_error_px).
    """
    def best_above_ground(cands):
        ag = [(R, t, e) for (R, t, e) in cands if (-R.T @ t).reshape(3)[2] > 0]
        return min(ag, key=lambda c: c[2]) if ag else None

    # Attempt 1: as given.
    cands = _solve_candidates(image_corners, world_corners, K)
    win = best_above_ground(cands)
    if win is not None:
        return win

    # Attempt 2: mirror the world X about the court centre (x -> 20 - x). Reflection M.
    M = np.diag([-1.0, 1.0, 1.0])
    offset = np.array([20.0, 0.0, 0.0])
    world_m = (world_corners @ M.T) + offset      # reflect each world point
    cands_m = _solve_candidates(image_corners, world_m, K)
    win_m = best_above_ground(cands_m)
    if win_m is not None:
        R_m, t_m, err_m = win_m
        # Map pose back to the canonical frame: a world point Xc relates to the mirrored
        # one by Xm = M @ Xc + offset. The camera eqn R_m*Xm + t_m must equal R*Xc + t,
        # so R = R_m @ M and t = R_m @ offset + t_m.
        R = R_m @ M
        t = (R_m @ offset).reshape(3, 1) + t_m.reshape(3, 1)
        return R, t, err_m

    # Fallback: neither orientation gave an above-ground camera — return the lowest-error
    # solution we have so the caller can flag it (rare; means bad/degenerate clicks).
    allc = cands + cands_m
    return min(allc, key=lambda c: c[2])


def solve_camera_pose(camera_id: str,
                      corners_tl_tr_br_bl: np.ndarray,
                      image_w: int,
                      image_h: int,
                      K: Optional[np.ndarray] = None,
                      camera_behind: Optional[str] = None) -> CameraPose:
    """Calibrate one camera from its 4 detected court corners.

    corners_tl_tr_br_bl: (4,2) pixel array, ordered TL,TR,BR,BL (the model's convention).
    camera_behind: "far"/"near" if known. If None, we try both and keep the lower-error
                   solution — which also tells us which baseline the camera is at.

    Returns a CameraPose with the recovered pose, reprojection error, and inferred baseline.
    """
    if cv2 is None:
        raise RuntimeError("cv2 required for calibration")
    image_corners = np.asarray(corners_tl_tr_br_bl, dtype=np.float64).reshape(4, 2)
    if K is None:
        K = default_intrinsics(image_w, image_h)

    candidates = [camera_behind] if camera_behind in ("far", "near") else ["far", "near"]
    solutions = []
    for cb in candidates:
        world = _image_corners_to_world(image_corners, cb)
        R, t, err = _solve_once(image_corners, world, K)
        cam_center = (-R.T @ t).reshape(3)
        solutions.append((err, R, t, cb, cam_center))

    # Picking the orientation is NOT decidable by reprojection error alone — a 180-degree
    # world flip reprojects equally well. Disambiguate by physical plausibility of the
    # recovered camera: it must sit ABOVE the ground (z>0) and OUTSIDE the court on the
    # side of the baseline it claims (far => y<0, near => y>44). Score each solution by
    # how well it matches that, then break ties with reprojection error.
    def plausibility(sol):
        err, R, t, cb, c = sol
        cx, cy, cz = c
        score = 0.0
        if cz > 0:                       # camera above ground
            score += 1.0
        if cb == "far" and cy < 0:       # far cam behind far baseline
            score += 1.0
        if cb == "near" and cy > 44:     # near cam behind near baseline
            score += 1.0
        # small bonus for being roughly centered in x and at a sane height
        if 2 < cz < 20:
            score += 0.5
        return score

    if len(solutions) == 1:
        err, R, t, cb, cam_center = solutions[0]
    else:
        solutions.sort(key=lambda s: (-plausibility(s), s[0]))  # max plausibility, then min err
        err, R, t, cb, cam_center = solutions[0]
    return CameraPose(
        camera_id=camera_id, K=K, R=R, t=t.reshape(3, 1),
        reproj_error_px=err, baseline=cb,
        camera_position_ft=(float(cam_center[0]), float(cam_center[1]), float(cam_center[2])),
    )


def infer_corner_label(pose: CameraPose) -> str:
    """The director's 'each camera knows which corner it is' — derived from the pose.
    For 2-cam (behind baselines) this returns which baseline; extendable to 4-cam corners."""
    cx, cy, _ = pose.camera_position_ft or (10, 22, 0)
    side = "left" if cx < 10 else "right"
    end = "near" if cy > 22 else "far"
    return f"{end}_baseline" if abs(cx - 10) < 5 else f"{end}_{side}_corner"


def estimate_focal_from_corners(named_corners_px: dict, image_w: int,
                                image_h: int) -> Optional[float]:
    """Estimate the focal length from the court homography itself (Zhang's
    single-plane self-calibration; square pixels + centred principal point).

    ADDED after real footage: the 60-degree-FOV guess in default_intrinsics
    was far off for the actual phones (estimated ~2500 px vs the ~1660 px
    guess), which made solvePnP unable to fit the coplanar corners at all
    (thousands of px reprojection). With H = [h1 h2 h3] mapping court-plane
    (x,y) to pixels, r1 = K^-1 h1 and r2 = K^-1 h2 must be orthonormal; the
    two constraints solve for f in closed form. Exact on synthetic tests.
    Returns None when degenerate (e.g. fronto-parallel view)."""
    if cv2 is None:
        return None
    names = [n for n in named_corners_px if n in ALL_WORLD_POINTS]
    if len(names) < 4:
        return None
    wpts = np.array([[ALL_WORLD_POINTS[n][0], ALL_WORLD_POINTS[n][1]] for n in names],
                    np.float64)
    ipts = np.array([named_corners_px[n] for n in names], np.float64)
    H, _ = cv2.findHomography(wpts, ipts)
    if H is None:
        return None
    cx, cy = image_w / 2.0, image_h / 2.0
    h1, h2 = H[:, 0], H[:, 1]

    def g(a, b):
        return (a[0] * b[0] + a[1] * b[1]
                - cx * (a[0] * b[2] + a[2] * b[0])
                - cy * (a[1] * b[2] + a[2] * b[1])
                + (cx * cx + cy * cy) * a[2] * b[2])

    def s(a, b):
        return a[2] * b[2]

    A = np.array([g(h1, h2), g(h1, h1) - g(h2, h2)])
    b = -np.array([s(h1, h2), s(h1, h1) - s(h2, h2)])
    denom = float(A @ A)
    if denom < 1e-12:
        return None
    u = float(A @ b) / denom            # u = 1/f^2
    if u <= 0:
        return None
    f = 1.0 / np.sqrt(u)
    return float(f) if 200.0 < f < 20000.0 else None


def solve_camera_pose_named(camera_id: str,
                            named_corners_px: dict,
                            image_w: int,
                            image_h: int,
                            K: Optional[np.ndarray] = None) -> CameraPose:
    """Calibrate from corners with EXPLICIT world identity — the diagonal-camera path.

    ADDED for diagonal corner mounts (unchanged solve_camera_pose remains the
    behind-baseline path). The original TL/TR/BR/BL -> world mapping was verified for
    cameras behind a baseline; from a diagonal corner that image-order heuristic can
    swap corner identities. Here the HUMAN states identity directly (click/confirm in
    the UI), so there is nothing to guess: each pixel is tied to its named world point
    and the pose is solved with the same verified _solve_once (z>0 disambiguation).

    named_corners_px: {name: (px,py), ...} — exactly 4 entries, each name a key of
    ALL_WORLD_POINTS (the 4 outer court corners "far_left"/"far_right"/"near_right"/
    "near_left", and/or the net-post ground points "net_left"/"net_right" for rigs
    where the far baseline isn't in frame). The 4 points must not be collinear.
    """
    if cv2 is None:
        raise RuntimeError("cv2 required for calibration")
    names = list(named_corners_px.keys())
    if len(names) != 4:
        raise ValueError(f"expected exactly 4 named corners, got {len(names)}: {names}")
    unknown = [n for n in names if n not in ALL_WORLD_POINTS]
    if unknown:
        raise ValueError(f"unknown world point name(s): {unknown}; "
                         f"valid names: {sorted(ALL_WORLD_POINTS)}")
    image_corners = np.array([named_corners_px[n] for n in names], dtype=np.float64)
    world_corners = np.array([ALL_WORLD_POINTS[n] for n in names], dtype=np.float64)
    if K is None:
        # Self-calibrate the focal from the court homography (real phones are
        # far from the 60-degree default); fall back to the guess if degenerate.
        f = estimate_focal_from_corners(named_corners_px, image_w, image_h)
        if f is not None:
            K = np.array([[f, 0, image_w / 2.0],
                          [0, f, image_h / 2.0],
                          [0, 0, 1.0]], dtype=np.float64)
        else:
            K = default_intrinsics(image_w, image_h)
    R, t, err = _solve_once(image_corners, world_corners, K)
    cam_center = (-R.T @ t).reshape(3)
    baseline = "near" if cam_center[1] > 22 else "far"
    return CameraPose(
        camera_id=camera_id, K=K, R=R, t=t.reshape(3, 1),
        reproj_error_px=err, baseline=baseline,
        camera_position_ft=(float(cam_center[0]), float(cam_center[1]),
                            float(cam_center[2])),
    )


# Outer-court corners in the canonical CW loop. Any traced quad maps to one of the 8
# (4 rotations x 2 directions) cyclic relabelings of this loop.
_OUTER_LOOP = ["far_left", "far_right", "near_right", "near_left"]


def solve_camera_pose_auto(camera_id: str,
                           corner_px_loop: list,
                           image_w: int,
                           image_h: int,
                           camera_behind: str = "near") -> CameraPose:
    """Calibrate from 4 traced corners WITHOUT the user getting the labels right.

    The user traces the outer court as a quad (any starting corner, either rotational
    direction) and states ONE bit: which baseline this camera sits behind ("near" => the
    camera is at y>44 side; "far" => y<0 side). That single hint resolves the court's
    front/back symmetry, which geometry alone cannot (both solve equally well). With it
    fixed, this tries all 8 cyclic relabelings (4 rotations x 2 directions) and keeps the
    physically-valid one (camera above ground, outside court, on the stated side) with the
    lowest reprojection error.

    Why the hint is needed: the two cameras face opposite ways, so "far-left" by eye is
    ambiguous and the court is symmetric front-to-back. The auto-relabel removes the
    rotation/reflection guesswork; the near/far bit removes the remaining mirror.

    corner_px_loop: list of 4 (px,py) in the order traced around the court.
    camera_behind: "near" or "far".
    """
    if cv2 is None:
        raise RuntimeError("cv2 required for calibration")
    pts = [tuple(p) for p in corner_px_loop]
    if len(pts) != 4:
        raise ValueError(f"need exactly 4 corners, got {len(pts)}")

    want_far_side = camera_behind == "far"   # far cam sits at y<0; near cam at y>44

    candidates = []
    for direction in (pts, pts[::-1]):
        for r in range(4):
            rotated = direction[r:] + direction[:r]
            named = {_OUTER_LOOP[i]: rotated[i] for i in range(4)}
            try:
                pose = solve_camera_pose_named(camera_id, named, image_w, image_h)
            except Exception:
                continue
            cx, cy, cz = pose.camera_position_ft
            above = cz > 0
            sane = 2.0 < cz < 30.0
            on_side = (cy < 0) if want_far_side else (cy > 44)   # the stated baseline
            score = (0 if (above and sane and on_side) else 1, pose.reproj_error_px)
            candidates.append((score, pose, named))

    if not candidates:
        raise ValueError("no solvable corner labeling found — check the traced quad")
    candidates.sort(key=lambda c: c[0])
    _, best_pose, best_named = candidates[0]
    return best_pose, best_named
