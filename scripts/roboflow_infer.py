"""Run the Roboflow court-keypoint model over a folder of images and save YOLO-pose labels.

Runs the hosted model `pb-9bsin/4` on every image in --dir and writes one YOLO-pose .txt
per image into <dir>/labels/, plus labels/_keypoints.json (pixel coords + visibility) that
the editor (keypoint_review_app.py) reads to pre-fill draggable points.

The model returns 12 court keypoints. We map them into THIS fixed order (model class ids):
  0 bg baseline left   1 bg baseline mid    2 bg baseline right
  3 bg kitchen right   4 bg kitchen mid     5 bg kitchen left
  6 fg kitchen left    7 fg kitchen mid     8 fg kitchen right
  9 fg baseline right 10 fg baseline mid   11 fg baseline left

This must run on a machine with open internet — the Roboflow API is the network dependency.
Calls the hosted REST endpoint directly with `requests` (no inference-sdk), so it works on
any Python 3.10+.

Run:  python scripts/roboflow_infer.py --dir dataset
      python scripts/roboflow_infer.py --dir dataset --overwrite   # redo already-done images
"""
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

MODEL_ID = "pb-9bsin/4"
API_URL = "https://serverless.roboflow.com"
API_KEY = os.environ.get("ROBOFLOW_API_KEY", "GD2Vh60x6e9XGMNStpkD")

KP_NAMES = [
    "bg_baseline_left", "bg_baseline_mid", "bg_baseline_right",
    "bg_kitchen_right", "bg_kitchen_mid", "bg_kitchen_left",
    "fg_kitchen_left", "fg_kitchen_mid", "fg_kitchen_right",
    "fg_baseline_right", "fg_baseline_mid", "fg_baseline_left",
]
N_KP = len(KP_NAMES)
IMG_EXTS = {".jpg", ".jpeg", ".png"}
BBOX_PAD_FRAC = 0.02


def list_images(d: Path) -> list[str]:
    if not d.is_dir():
        raise SystemExit(f"not a folder: {d}")
    imgs = sorted(p.name for p in d.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not imgs:
        raise SystemExit(f"no images in {d}")
    return imgs


def extract_keypoints(result: dict) -> list[dict] | None:
    """Pull 12 keypoints from a Roboflow keypoint-detection response into fixed class order.

    Returns a list of {x,y,v,conf} length 12 (v=2 found, v=0 missing), or None if the
    response has no usable keypoints. Maps by class_id when present, else falls back to the
    order the model returned them in.
    """
    preds = result.get("predictions")
    if isinstance(preds, dict):
        preds = preds.get("predictions", [])
    if not preds:
        return None

    # Keypoint models nest a `keypoints` list under the (single) detected object; pick the
    # detection that carries the most keypoints.
    best = max(preds, key=lambda p: len(p.get("keypoints", []) or []), default=None)
    kps = (best or {}).get("keypoints") or []
    if not kps:
        return None

    out = [{"x": 0.0, "y": 0.0, "v": 0, "conf": 0.0} for _ in range(N_KP)]
    by_id_used = False
    for kp in kps:
        cid = kp.get("class_id")
        if isinstance(cid, int) and 0 <= cid < N_KP:
            out[cid] = {"x": float(kp.get("x", 0)), "y": float(kp.get("y", 0)),
                        "v": 2, "conf": float(kp.get("confidence", 0) or 0)}
            by_id_used = True
    if not by_id_used:
        for i, kp in enumerate(kps[:N_KP]):
            out[i] = {"x": float(kp.get("x", 0)), "y": float(kp.get("y", 0)),
                      "v": 2, "conf": float(kp.get("confidence", 0) or 0)}
    return out


def yolo_pose_line(pts: list[dict], w: int, h: int) -> str:
    """One YOLO-pose line: class cx cy bw bh  x1 y1 v1 ... x12 y12 v12 (all normalized).
    Box tightly encloses placed (v>0) points with small padding."""
    placed = [(p["x"], p["y"]) for p in pts if p["v"] > 0]
    if not placed:
        return ""
    xs, ys = [p[0] for p in placed], [p[1] for p in placed]
    pad = BBOX_PAD_FRAC * max(w, h)
    x0, y0 = max(0, min(xs) - pad), max(0, min(ys) - pad)
    x1, y1 = min(w, max(xs) + pad), min(h, max(ys) + pad)
    cx, cy = (x0 + x1) / 2 / w, (y0 + y1) / 2 / h
    bw, bh = (x1 - x0) / w, (y1 - y0) / h
    fields = ["0", f"{cx:.6f}", f"{cy:.6f}", f"{bw:.6f}", f"{bh:.6f}"]
    for p in pts:
        if p["v"] > 0:
            fields += [f"{p['x']/w:.6f}", f"{p['y']/h:.6f}", str(int(p["v"]))]
        else:
            fields += ["0.000000", "0.000000", "0"]
    return " ".join(fields) + "\n"


def infer_image(session, path: Path) -> dict:
    """POST one image (base64) to the hosted model and return the parsed JSON response.
    Raises on non-200 so the caller can count it as an error."""
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    url = f"{API_URL}/{MODEL_ID}"
    resp = session.post(url, params={"api_key": API_KEY}, data=b64,
                        headers={"Content-Type": "application/x-www-form-urlencoded"},
                        timeout=60)
    resp.raise_for_status()
    return resp.json()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=REPO / "dataset")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-run images already present in _keypoints.json")
    ap.add_argument("--limit", type=int, default=0, help="stop after N images (0 = all)")
    args = ap.parse_args()

    import requests  # network dep, import lazily
    import cv2

    session = requests.Session()

    images = list_images(args.dir)
    if args.limit:
        images = images[: args.limit]
    labels_dir = args.dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    json_path = labels_dir / "_keypoints.json"
    store: dict = json.loads(json_path.read_text()) if json_path.exists() else {}

    n_ok = n_skip = n_empty = n_err = 0
    for i, name in enumerate(images, 1):
        if not args.overwrite and store.get(name, {}).get("source") == "model":
            n_skip += 1
            continue
        path = args.dir / name
        img = cv2.imread(str(path))
        if img is None:
            print(f"  [{i}/{len(images)}] {name}: cannot read, skipped")
            n_err += 1
            continue
        h, w = img.shape[:2]
        try:
            result = infer_image(session, path)
        except Exception as e:  # noqa: BLE001 - one image failing must not stop the batch
            print(f"  [{i}/{len(images)}] {name}: inference error: {e}")
            n_err += 1
            continue

        pts = extract_keypoints(result)
        if pts is None:
            pts = [{"x": 0.0, "y": 0.0, "v": 0, "conf": 0.0} for _ in range(N_KP)]
            n_empty += 1
            tag = "no keypoints"
        else:
            found = sum(1 for p in pts if p["v"] > 0)
            n_ok += 1
            tag = f"{found}/12 pts"

        store[name] = {"points": pts, "w": w, "h": h, "source": "model",
                       "complete": False}
        (labels_dir / (Path(name).stem + ".txt")).write_text(yolo_pose_line(pts, w, h))

        if i % 25 == 0 or i == len(images):
            json_path.write_text(json.dumps(store, indent=1))
        if i % 20 == 0 or i <= 3:
            print(f"  [{i}/{len(images)}] {name}: {tag}")

    json_path.write_text(json.dumps(store, indent=1))
    print(f"\ndone. predicted {n_ok}, empty {n_empty}, skipped(existing) {n_skip}, errors {n_err}")
    print(f"labels -> {labels_dir}")
    print(f"next:  python scripts/keypoint_review_app.py --dir {args.dir}")


if __name__ == "__main__":
    main()
