"""Rename all images in dataset/ to img_1.jpg, img_2.jpg, ... in sorted order.

Writes rename_map.csv (new_name,original_name) alongside the images so the link
back to the source clip + frame index is NOT lost — you need that to split
train/test BY CLIP later (random frame splits leak near-duplicates and inflate
your score). Idempotent-ish: skips if files already look renamed unless --force.

Usage:
    python scripts/rename_dataset.py
    python scripts/rename_dataset.py --dir dataset --prefix img --start 1
    python scripts/rename_dataset.py --dry-run        # show what would happen
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png"}
REPO = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(description="Rename dataset images to img_N")
    ap.add_argument("--dir", type=Path, default=REPO / "dataset", help="folder of images")
    ap.add_argument("--prefix", default="img", help="name prefix (default 'img')")
    ap.add_argument("--start", type=int, default=1, help="first index (default 1)")
    ap.add_argument("--dry-run", action="store_true", help="print, don't rename")
    ap.add_argument("--force", action="store_true", help="rename even if already img_*")
    args = ap.parse_args()

    if not args.dir.is_dir():
        raise SystemExit(f"not a folder: {args.dir}")

    imgs = sorted(p for p in args.dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not imgs:
        raise SystemExit(f"no images in {args.dir}")

    already = all(p.stem.startswith(args.prefix + "_") for p in imgs)
    if already and not args.force:
        print(f"images already look like {args.prefix}_*; nothing to do (use --force to redo)")
        return

    # Two-phase rename through temp names so we never overwrite an existing target
    # (e.g. renaming img_2 -> img_1 when img_1 still exists).
    pad = len(str(args.start + len(imgs) - 1))
    plan = []
    for i, src in enumerate(imgs):
        new_name = f"{args.prefix}_{args.start + i}{src.suffix.lower()}"
        plan.append((src, args.dir / new_name))

    print(f"{len(plan)} images in {args.dir}")
    for src, dst in plan[:3]:
        print(f"  {src.name}  ->  {dst.name}")
    if len(plan) > 3:
        print(f"  ... ({len(plan) - 3} more)")

    if args.dry_run:
        print("dry-run: no files changed")
        return

    # phase 1: to temp
    tmp_map = []
    for idx, (src, dst) in enumerate(plan):
        tmp = args.dir / f".__tmp_rename_{idx}{src.suffix.lower()}"
        src.rename(tmp)
        tmp_map.append((tmp, dst, src.name))
    # phase 2: temp -> final, recording the map
    rows = []
    for tmp, dst, original in tmp_map:
        tmp.rename(dst)
        rows.append((dst.name, original))

    map_path = args.dir / "rename_map.csv"
    with map_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["new_name", "original_name"])
        w.writerows(rows)

    print(f"\nRenamed {len(rows)} images. Mapping saved to {map_path}")
    print("Keep rename_map.csv — it's how you recover which clip each frame came from.")


if __name__ == "__main__":
    main()
