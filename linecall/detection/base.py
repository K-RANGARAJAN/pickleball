"""Per-frame ball-detection interface: a Detection dataclass + BallDetector ABC."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Detection:
    """A single 2D ball observation in one camera's image plane.

    source:
      'yolo'     - detected by the YOLO primary detector
      'tracknet' - detected by the TrackNet fallback
      'none'     - no detection this frame (px_x/px_y are NaN, confidence 0)
    """
    camera_id: str
    frame_idx: int
    px_x: float
    px_y: float
    confidence: float
    source: str = "none"

    @property
    def is_detected(self) -> bool:
        return self.source != "none" and np.isfinite(self.px_x) and np.isfinite(self.px_y)

    @classmethod
    def empty(cls, camera_id: str, frame_idx: int) -> "Detection":
        return cls(camera_id, frame_idx, float("nan"), float("nan"), 0.0, "none")

    def as_row(self) -> dict:
        """For writing per-camera detection CSVs."""
        return {
            "camera_id": self.camera_id,
            "frame_idx": self.frame_idx,
            "px_x": self.px_x,
            "px_y": self.px_y,
            "confidence": self.confidence,
            "source": self.source,
        }


class BallDetector(ABC):
    """Stateless-per-frame 2D ball detector for a single camera view."""

    @abstractmethod
    def detect(self, frame_bgr: np.ndarray, frame_idx: int, camera_id: str,
               court_mask: Optional[np.ndarray] = None) -> Detection:
        """Return a Detection for this frame. Never raises on a normal miss -
        returns Detection.empty(...) instead."""
        ...

    def warmup(self) -> None:
        """Optional: load weights / run a dummy inference so the first real frame
        isn't slow. Default no-op."""
        return None
