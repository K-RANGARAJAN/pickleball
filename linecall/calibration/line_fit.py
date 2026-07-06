"""Line-fit calibration math: fit court edges by total-least-squares over a dragged segment,
then take the 4 calibration corners as intersections of adjacent edges (more accurate than
hand-placed clicks). Pure numpy. The drag loop A->B->C->D maps to the near-half corners:

    A->B left sideline, B->C net line, C->D right sideline, D->A near baseline
    near_left=(D->A)∩(A->B), net_left=(A->B)∩(B->C), net_right=(B->C)∩(C->D), near_right=(C->D)∩(D->A)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

Point = Tuple[float, float]

# The four world points this front-end produces, in the loop order A,B,C,D.
# FULL-COURT calibration: trace the outer court rectangle, so every camera is calibrated
# to the SAME 20x44 court (no half-court seam at the net, which caused false OUTs in the gap).
CORNER_LOOP = ["far_left", "far_right", "near_right", "near_left"]  # A B C D

# Each corner is the intersection of the two edges adjacent to it in the loop.
# Edges are indexed by their start corner: edge i goes CORNER_LOOP[i] -> CORNER_LOOP[i+1].
#   corner i = edge (i-1)  ∩  edge (i)
# i.e. near_left = edge3(D->A) ∩ edge0(A->B), etc.


@dataclass
class Line2D:
    """A 2D line as a point + unit direction, plus the implicit form a*x + b*y + c = 0
    where (a, b) is the unit normal. Built by total-least-squares over dragged pixels."""
    point: np.ndarray        # (2,) a point on the line (the centroid of the drag)
    direction: np.ndarray    # (2,) unit direction
    normal: np.ndarray       # (2,) unit normal
    c: float                 # offset so a*x + b*y + c = 0
    rms_px: float            # RMS orthogonal distance of the fitted points (fit quality)
    n_points: int

    def implicit(self) -> Tuple[float, float, float]:
        return float(self.normal[0]), float(self.normal[1]), float(self.c)


def fit_line(points: Sequence[Point]) -> Line2D:
    """Total-least-squares fit of a line to 2D points (orthogonal regression via PCA).

    Unlike y = m*x + b least squares, this handles vertical/steep edges correctly and
    minimizes perpendicular distance — the right error model for a dragged court line.
    """
    P = np.asarray(points, dtype=float)
    if P.ndim != 2 or P.shape[0] < 2 or P.shape[1] != 2:
        raise ValueError("need >=2 points of shape (N,2)")
    centroid = P.mean(axis=0)
    Q = P - centroid
    # principal direction = eigenvector of largest singular value
    _, _, Vt = np.linalg.svd(Q, full_matrices=False)
    direction = Vt[0] / np.linalg.norm(Vt[0])
    normal = np.array([-direction[1], direction[0]])
    c = -float(normal @ centroid)
    # RMS orthogonal residual
    resid = Q @ normal
    rms = float(np.sqrt(np.mean(resid ** 2)))
    return Line2D(point=centroid, direction=direction, normal=normal, c=c,
                  rms_px=rms, n_points=int(P.shape[0]))


def intersect(l1: Line2D, l2: Line2D) -> Point:
    """Intersection of two lines in implicit form. Raises if near-parallel."""
    a1, b1, c1 = l1.implicit()
    a2, b2, c2 = l2.implicit()
    A = np.array([[a1, b1], [a2, b2]])
    det = np.linalg.det(A)
    if abs(det) < 1e-9:
        raise ValueError("edges are parallel — cannot intersect (did two drags overlap?)")
    rhs = np.array([-c1, -c2])
    xy = np.linalg.solve(A, rhs)
    return float(xy[0]), float(xy[1])


def corners_from_edges(edges: Dict[str, Sequence[Point]]) -> Tuple[Dict[str, Point],
                                                                    Dict[str, float]]:
    """Given the 4 dragged edges, return the 4 named corners (from intersections) and a
    per-edge RMS-fit dict for UI feedback.

    edges: {"AB": [...pts...], "BC": [...], "CD": [...], "DA": [...]}
           FULL OUTER COURT loop:
           AB = far baseline, BC = right sideline, CD = near baseline, DA = left sideline.
    """
    need = {"AB", "BC", "CD", "DA"}
    missing = need - set(edges)
    if missing:
        raise ValueError(f"missing edges: {sorted(missing)} (need {sorted(need)})")
    fits = {k: fit_line(edges[k]) for k in ("AB", "BC", "CD", "DA")}
    corners = {
        "far_left":   intersect(fits["DA"], fits["AB"]),   # A
        "far_right":  intersect(fits["AB"], fits["BC"]),   # B
        "near_right": intersect(fits["BC"], fits["CD"]),   # C
        "near_left":  intersect(fits["CD"], fits["DA"]),   # D
    }
    rms = {k: round(v.rms_px, 2) for k, v in fits.items()}
    return corners, rms
