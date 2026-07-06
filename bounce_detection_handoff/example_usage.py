"""
example_usage.py - detect bounces from ball-tracking output.

Run:  python example_usage.py ball.csv
"""
import sys

from bounce_detector import (
    detect_bounces, compute_homography, load_track_csv, save_bounces_csv,
    BounceParams, PICKLEBALL,
)


def main(csv_path):
    # 1) Ball-tracking output: per-frame (x, y); (0, 0) == no detection.
    #    Either load a tracker ball.csv ...
    coords = load_track_csv(csv_path)
    #    ... or pass the list straight from the tracker:
    #        coords = ball_tracker.predict(frames)

    # 2) Ground homography from 4 court corners clicked in the image.
    #    CLICK ORDER MUST MATCH CourtModel.corners_ft:
    #        near-left, near-right, far-right, far-left
    #    >>> replace these placeholder pixels with your real clicked corners <<<
    image_corners = [
        (350, 980),    # near-left   (baseline nearest camera, left sideline)
        (1570, 980),   # near-right
        (1180, 360),   # far-right   (baseline across the net)
        (740, 360),    # far-left
    ]
    H = compute_homography(image_corners, PICKLEBALL)

    # 3) Detect + classify.
    bounces = detect_bounces(coords, homography=H, court=PICKLEBALL,
                             params=BounceParams())

    for b in bounces:
        flag = "  (suppressed)" if b.suppressed else ""
        print(f"frame {b.frame:5d}  px=({b.px_x:4.0f},{b.px_y:4.0f})  "
              f"court=({b.court_x:5.1f},{b.court_y:5.1f}) ft  "
              f"{b.classification:8s} [{b.side}]{flag}")

    kept = [b for b in bounces if not b.suppressed]
    save_bounces_csv(bounces, "bounces.csv")
    print(f"\n{len(kept)} bounces kept ({len(bounces) - len(kept)} suppressed) "
          f"-> bounces.csv")


def pixel_only_demo(csv_path):
    """No court set up yet? You still get bounce frames + pixels (UNKNOWN class)."""
    coords = load_track_csv(csv_path)
    for b in detect_bounces(coords):
        print(f"bounce at frame {b.frame}: pixel ({b.px_x:.0f}, {b.px_y:.0f})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ball.csv")
