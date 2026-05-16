# models/mask_metrics.py
import math
import numpy as np
from PIL import Image


def compute_mask_metrics(mask_path: str) -> dict | None:
    img = Image.open(mask_path).convert("L")
    m = np.array(img)
    bin_m = (m > 127).astype(np.uint8)

    ys, xs = np.where(bin_m == 1)
    if xs.size == 0:
        return None

    area_pixels = int(xs.size)
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())

    cx = float(xs.mean())
    cy = float(ys.mean())

    max_diameter_pixels = float(math.hypot(x_max - x_min, y_max - y_min))

    w = bin_m.shape[1]
    mid = (w - 1) / 2.0
    if abs(cx - mid) <= 0.05 * w:
        laterality = "midline"
    elif cx < mid:
        laterality = "left"
    else:
        laterality = "right"

    return {
        "area_pixels": area_pixels,
        "bbox": [x_min, y_min, x_max, y_max],
        "centroid": [round(cx, 2), round(cy, 2)],
        "max_diameter_pixels": round(max_diameter_pixels, 2),
        "laterality": laterality,
        "note": "All measurements are pixel-based (2D image; no DICOM/NIfTI spacing available).",
    }
