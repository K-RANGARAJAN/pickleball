"""Resize all dataset images to a uniform 1920x1080, preserving court geometry.

The dataset mixes resolutions/aspect ratios (your ~3:2 frames + a friend's 16:9 frames).
A plain stretch to 16:9 would distort the 3:2 courts (skewed shape), which is harmful for a
keypoint model that learns court geometry. So the default is LETTERBOX: scale to fit inside
1920x1080 keeping aspect ratio, pad the remainder with black. 16:9 inputs get no bars; other
ratios get symmetric bars. Run this BEFORE labeling so keypoints are clicked on final frames.

Edits in place by default (overwrites each image at 1920x1080). Use --out to write to a new
folder instead and keep originals.

Usage:
    python scripts/resize_dataset.py
    python scripts/resize_dataset.py --dir dataset --out dataset_1080
    python scripts/resize_dataset.py --mode stretch     # force exact size (distorts non-16:9)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parent.parent
IMG_EXTS = {".jpg", ".jpeg", ".png"}
TARGET_W, TARGET_H = 1920, 1080


def letterbox(img, tw=TARGET_W, th=TARGET_H):
    h, w = img.shape[:2]
    scale = min(tw / w, th / h)
    nw, nh = round(w * scale), round(h * scale)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(img, (nw, nh), interpolation=interp)
    canvas = cv2.copyMakeBorder(
        resized,
        (th - nh) // 2, th - nh - (th - nh) // 2,
        (tw - nw) // 2, tw - nw - (tw - nw) // 2,
        cv2.BORDER_CONSTANT, value=(0, 0, 0),
    )
    return canvas


def stretch(img, tw=TARGET_W, th=TARGET_H):
    return cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=REPO / "dataset")
    ap.add_argument("--out", type=Path, default=None, help="write here instead of in place")
    ap.add_argument("--mode", choices=["letterbox", "stretch"], default="letterbox")
    args = ap.parse_args()

    if not args.dir.is_dir():
        raise SystemExit(f"not a folder: {args.dir}")
    out_dir = args.out or args.dir
    out_dir.mkdir(parents=True, exist_ok=True)

    imgs = sorted(p for p in args.dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not imgs:
        raise SystemExit(f"no images in {args.dir}")

    fn = letterbox if args.mode == "letterbox" else stretch
    sizes_seen, changed, skipped = {}, 0, 0
    for i, p in enumerate(imgs, 1):
        img = cv2.imread(str(p))
        if img is None:
            print(f"  [{i}/{len(imgs)}] UNREADABLE {p.name} (skipped)")
            skipped += 1
            continue
        h, w = img.shape[:2]
        sizes_seen[(w, h)] = sizes_seen.get((w, h), 0) + 1
        if (w, h) == (TARGET_W, TARGET_H):
            if args.out:                      # still copy through to out dir
                cv2.imwrite(str(out_dir / p.name), img)
            continue
        out = fn(img)
        cv2.imwrite(str(out_dir / p.name), out)
        changed += 1
        if i % 200 == 0 or i == len(imgs):
            print(f"  [{i}/{len(imgs)}] processed")

    print(f"\nDone. {len(imgs)} images -> {out_dir} ({TARGET_W}x{TARGET_H}, {args.mode})")
    print(f"  resized: {changed}, already-target/skipped: {len(imgs)-changed}")
    print("  source sizes found:")
    for (w, h), n in sorted(sizes_seen.items(), key=lambda kv: -kv[1]):
        print(f"    {w}x{h}: {n} images")


if __name__ == "__main__":
    main()
