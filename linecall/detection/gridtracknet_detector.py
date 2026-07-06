"""GridTrackNet (ONNX) ball tracker wrapped as a per-camera detection source.

Temporal tracker: consumes 5 consecutive frames, returns one (x, y) per frame. Trained for
~30fps, so on 60fps clips we feed every other frame and map results back to original frame
indices. Returns a list of Detection; (0,0) from the model -> an empty Detection.
"""

from __future__ import annotations

import os
import sys
from typing import List

import numpy as np

from .base import Detection

# the wrapper ships in ball_tracker_handoff/ at the repo root
_HANDOFF = os.path.join(os.path.dirname(__file__), "..", "..", "ball_tracker_handoff")


def _load_tracker(model_path: str, provider: str):
    """Load the GridTrackNet BallTracker.

    The handoff BallTracker only knows 'gpu' (CUDA) and 'cpu'. To use the Mac GPU we add a
    'coreml' path here WITHOUT editing the handoff file: build the tracker on CPU, then swap
    its ONNX session for one running CoreMLExecutionProvider. Everything else (preprocessing,
    grid decode) is untouched, so predictions are identical — only the execution backend changes.
    """
    handoff = os.path.abspath(_HANDOFF)
    if handoff not in sys.path:
        sys.path.insert(0, handoff)
    from ball_tracker import BallTracker

    if provider == "coreml":
        import onnxruntime as ort
        avail = ort.get_available_providers()
        if "CoreMLExecutionProvider" not in avail:
            print("[gridtracknet] CoreML not available; falling back to CPU")
            return BallTracker(model_path, provider="cpu")
        tracker = BallTracker(model_path, provider="cpu")   # safe construction
        tracker.session = ort.InferenceSession(
            model_path,
            providers=["CoreMLExecutionProvider", "CPUExecutionProvider"])
        tracker.input_name = tracker.session.get_inputs()[0].name
        tracker.active_providers = tracker.session.get_providers()
        return tracker

    return BallTracker(model_path, provider=provider)


def track_clip(video_path: str, camera_id: str,
               model_path: str, provider: str = "cpu",
               fps_skip: int = 2, start_frame: int = 0,
               stop_frame: int | None = None,
               batch_frames: int = 200,
               progress_every: int = 2000) -> List[Detection]:
    """Run GridTrackNet over one clip, returning one Detection per ORIGINAL frame index.

    fps_skip: feed every Nth frame (2 for 60fps -> 30fps spacing). Frames between fed
    samples get a 'none' Detection (the 3D Kalman bridges those).

    MEMORY-SAFE: processes in batches of ~batch_frames FED frames (a multiple of 5) so the
    whole clip is never held in RAM — only one batch of frames at a time. The earlier
    "load all frames then predict once" version exhausted memory on a 5-minute 1080p clip.
    """
    import cv2

    tracker = _load_tracker(model_path, provider)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"cannot open {video_path}")
    if start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    batch = max(5, (batch_frames // 5) * 5)     # multiple of 5
    coord_by_index: dict = {}
    buf_frames: list = []
    buf_indices: list = []

    def flush():
        if len(buf_frames) < 5:
            buf_frames.clear(); buf_indices.clear(); return
        usable = (len(buf_frames) // 5) * 5
        coords = tracker.predict(buf_frames[:usable], is_bgr=True)
        for k, (x, y) in enumerate(coords):
            coord_by_index[buf_indices[k]] = (x, y)
        buf_frames.clear(); buf_indices.clear()

    idx = start_frame
    while True:
        if stop_frame is not None and idx >= stop_frame:
            break
        ok, frame = cap.read()
        if not ok:
            break
        if (idx - start_frame) % fps_skip == 0:
            buf_frames.append(frame)
            buf_indices.append(idx)
            if len(buf_frames) >= batch:
                flush()
        idx += 1
        if progress_every and idx % progress_every == 0:
            print(f"[{camera_id}] processed {idx - start_frame} frames")
    flush()
    cap.release()
    last_idx = idx

    dets: List[Detection] = []
    for fi in range(start_frame, last_idx):
        if fi in coord_by_index:
            x, y = coord_by_index[fi]
            if (x, y) == (0, 0):
                dets.append(Detection.empty(camera_id, fi))
            else:
                dets.append(Detection(camera_id, fi, float(x), float(y), 1.0,
                                      "gridtracknet"))
        else:
            dets.append(Detection.empty(camera_id, fi))

    got = sum(1 for d in dets if d.is_detected)
    print(f"[{camera_id}] GridTrackNet: {got}/{len(dets)} frames with a ball "
          f"({100*got/max(len(dets),1):.0f}%)")
    return dets
