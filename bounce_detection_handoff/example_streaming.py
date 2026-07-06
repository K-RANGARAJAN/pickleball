"""
example_streaming.py - causal (live) bounce detection with StreamingBounceDetector.

Unlike example_usage.py (offline, whole clip at once), this processes frames ONE AT A TIME,
the way a live stream arrives. Bounces are emitted as soon as they're confirmed, a few
frames after the real contact (causal lag). Missing detections are coasted through by the
Kalman filter, so brief occlusions don't create false bounces.

Run:  python example_streaming.py ball.csv
"""
import sys

from bounce_detector import StreamingBounceDetector, BounceParams, load_track_csv
# from bounce_detector import compute_homography, PICKLEBALL   # for IN/OUT


def main(csv_path):
    coords = load_track_csv(csv_path)     # in a real stream, (x, y) arrives frame by frame

    # For IN/OUT add: homography=compute_homography(image_corners, PICKLEBALL), court=PICKLEBALL
    det = StreamingBounceDetector(params=BounceParams())

    bounces = []
    for frame_idx, (x, y) in enumerate(coords):
        bounce = det.update(frame_idx, x, y, detected=(x != 0 or y != 0))
        if bounce:
            bounces.append(bounce)
            print(f"BOUNCE @ frame {bounce.frame}: pixel ({bounce.px_x:.0f}, {bounce.px_y:.0f})")

    print(f"\n{len(bounces)} bounces (causal). Each fires ~1-3 frames after the true contact.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ball.csv")
