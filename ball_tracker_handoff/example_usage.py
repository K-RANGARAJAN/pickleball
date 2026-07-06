"""
example_usage.py - using BallTracker inside your own (pickleball) pipeline.

Two examples:
  1) quickstart      - track a whole video file and report stats
  2) fusion sketch   - where to merge ball tracking with your own detector

Run:  python example_usage.py path/to/clip.mp4
"""

import sys
import cv2

from ball_tracker import BallTracker

MODEL_PATH = "models/model_weights.onnx"   # <-- adjust to where you copied the .onnx
CHUNK = 5                                    # must be a multiple of 5


def quickstart(video_path: str):
    tracker = BallTracker(MODEL_PATH, provider="gpu")   # use provider="cpu" if no CUDA
    print("ONNX providers in use:", tracker.active_providers)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")

    buffer, all_coords = [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        buffer.append(frame)

        if len(buffer) == CHUNK:
            # one (x, y) per frame, in the original frame's pixel space
            coords = tracker.predict(buffer, is_bgr=True)

            for frame_img, (x, y) in zip(buffer, coords):
                if (x, y) != (0, 0):
                    cv2.circle(frame_img, (x, y), 6, (0, 0, 255), -1)   # draw the ball

                # ----------------------------------------------------------------
                # >>> FUSION POINT: run YOUR pickleball logic on `frame_img` here,
                #     e.g. player/court detection, then combine with the ball (x, y):
                #         players = my_pickleball_detector(frame_img)
                #         annotate(frame_img, players, ball=(x, y))
                # ----------------------------------------------------------------

                all_coords.append((x, y))

            buffer = []   # reset for the next group of 5

    cap.release()
    visible = sum(1 for c in all_coords if c != (0, 0))
    print(f"Tracked {len(all_coords)} frames; {visible} had a detected ball "
          f"({len(all_coords) - visible} empty).")
    return all_coords


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "clip.mp4"
    quickstart(path)
