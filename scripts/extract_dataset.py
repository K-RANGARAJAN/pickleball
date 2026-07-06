"""Extract every Nth frame from baseline clips into a flat dataset/ folder for labeling.

Reads the source clips read-only (originals untouched). Optionally duplicates the
clips into a working folder first (--copy), though copying is not needed just to
extract frames. Frames are named <clip-stem>_f<original-frame-index>.jpg so each
frame is traceable back to its source clip and exact position.

Usage:
    python scripts/extract_dataset.py
    python scripts/extract_dataset.py --src "/path/to/Online clips/baseline" --every 7
    python scripts/extract_dataset.py --copy        # also duplicate clips to a new folder

Requires ffmpeg + ffprobe on PATH (already used elsewhere in this repo).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

VIDEO_EXTS = {".mov", ".mp4", ".mkv", ".avi", ".webm"}

DEFAULT_SRC = Path.home() / "Pictures" / "Online clips" / "baseline"
REPO = Path(__file__).resolve().parent.parent


def find_clips(src: Path) -> list[Path]:
    clips = sorted(p for p in src.iterdir() if p.suffix.lower() in VIDEO_EXTS)
    if not clips:
        raise SystemExit(f"no video clips found in {src}")
    return clips


def copy_clips(clips: list[Path], dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for i, c in enumerate(clips, 1):
        target = dst / c.name
        if target.exists() and target.stat().st_size == c.stat().st_size:
            print(f"  [{i}/{len(clips)}] skip (already copied) {c.name}")
            continue
        print(f"  [{i}/{len(clips)}] copy {c.name}")
        shutil.copy2(c, target)


def extract_frames(clip: Path, out_dir: Path, every: int, jpeg_q: int = 2) -> int:
    """Save every Nth frame of one clip. Returns the number of frames written.

    Uses ffmpeg's select filter so decoding happens once in C, not frame-by-frame
    in Python. -vsync vfr keeps only the selected frames. jpeg_q: 2 = high quality.
    """
    stem = clip.stem.replace(" ", "_")
    pattern = str(out_dir / f"{stem}_f%06d.jpg")
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(clip),
        "-vf", f"select=not(mod(n\\,{every}))",
        "-vsync", "vfr", "-q:v", str(jpeg_q),
        "-frame_pts", "1",            # encode original frame index into %06d
        pattern, "-y",
    ]
    subprocess.run(cmd, check=True)
    return len(list(out_dir.glob(f"{stem}_f*.jpg")))


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract every Nth frame into dataset/")
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC,
                    help=f"folder of source clips (default: {DEFAULT_SRC})")
    ap.add_argument("--out", type=Path, default=REPO / "dataset",
                    help="output frames folder (default: <repo>/dataset)")
    ap.add_argument("--every", type=int, default=7, help="keep every Nth frame (default 7)")
    ap.add_argument("--copy", action="store_true",
                    help="also duplicate clips into <repo>/baseline_clips_combined first")
    args = ap.parse_args()

    clips = find_clips(args.src)
    print(f"found {len(clips)} clips in {args.src}")

    if args.copy:
        combined = REPO / "baseline_clips_combined"
        print(f"copying clips -> {combined}")
        copy_clips(clips, combined)
        clips = find_clips(combined)

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"extracting every {args.every}th frame -> {args.out}")

    total = 0
    for i, clip in enumerate(clips, 1):
        try:
            written = extract_frames(clip, args.out, args.every)
        except subprocess.CalledProcessError:
            print(f"  [{i}/{len(clips)}] FAILED {clip.name} (skipped)")
            continue
        total = len(list(args.out.glob("*.jpg")))
        print(f"  [{i}/{len(clips)}] {clip.name} -> {written} frames (dataset total {total})")

    print(f"\nDone. {total} frames in {args.out}")


if __name__ == "__main__":
    main()
