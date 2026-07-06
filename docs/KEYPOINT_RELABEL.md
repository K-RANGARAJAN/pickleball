# Re-label court keypoints with the Roboflow model

Goal: run `pb-9bsin/4` over a folder of images, then drag-correct its 12 court keypoints and
save them as YOLO-pose labels for retraining.

## Point order (model class ids — fixed everywhere)

```
0 bg baseline left    1 bg baseline mid     2 bg baseline right
3 bg kitchen right    4 bg kitchen mid      5 bg kitchen left
6 fg kitchen left     7 fg kitchen mid      8 fg kitchen right
9 fg baseline right  10 fg baseline mid    11 fg baseline left
```
`bg` = background (far) half, `fg` = foreground (near) half.

## 1. Predict (needs internet — run on your laptop, not in a sandbox)

```bash
pip install requests opencv-python      # once
python scripts/roboflow_infer.py --dir dataset
```

Calls the Roboflow hosted REST endpoint directly with `requests` (works on any Python
3.10+; no `inference-sdk`, which doesn't support Python 3.13).

Writes `dataset/labels/<name>.txt` (YOLO-pose) for every image plus
`dataset/labels/_keypoints.json` (pixel coords + visibility) that the editor reads.

- Already-predicted images are skipped on re-run; add `--overwrite` to redo them.
- `--limit N` predicts only the first N (handy for a quick test).
- The API key is read from `$ROBOFLOW_API_KEY` if set, else falls back to the embedded key.

## 2. Correct the points

```bash
python scripts/keypoint_review_app.py --dir dataset      # http://localhost:8079
```

Each image opens with the 12 predicted dots drawn on it.

- **Drag** a dot to its true spot. It auto-saves on release.
- **Click** a dot (or its row in the side list) to select it, then **V** = visible, **X** =
  not-visible.
- **Not-visible** dots park on the left rail and are excluded from the label and the bbox.
  Drag one onto the court to bring it back (e.g. a far point the model missed).
- **Reset to model** restores that image's original predictions.
- **← / →** (or Prev / Next) navigate; Next also marks the image *reviewed*.

The editor opens on the first not-yet-reviewed image, so you can stop and resume.

## 3. Output (ready for YOLO-pose training)

Each `dataset/labels/<name>.txt` is one line:

```
0  cx cy w h  x1 y1 v1  x2 y2 v2  ...  x12 y12 v12        (all normalized 0..1, v in {0,2})
```

class `0` = court, one box auto-fit around the visible keypoints, then the 12 keypoints in the
order above. A `data.yaml` for training should set `kpt_shape: [12, 3]` and `flip_idx`
appropriately for these 12 named points.

## Notes

- Both scripts use the same point order and the same YOLO writer, so predictions and
  corrections stay consistent.
- The existing `scripts/keypoint_label_app.py` (click-to-place from scratch, different point
  order) is unchanged — this is a separate flow that starts from model predictions.
