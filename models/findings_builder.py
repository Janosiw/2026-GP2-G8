def build_findings_text(
    tumor_type: str,
    confidence: float | None,
    mask_metrics: dict | None,
    volume_3d: dict | None = None,
) -> str:
    tumor_str = str(tumor_type).strip() if tumor_type else "unknown"

    try:
        conf_val = float(confidence)
        conf_str = f"{conf_val:.2f}"
    except Exception:
        conf_str = ""

    area_pixels = ""
    max_d = ""
    laterality = ""
    x_min = y_min = x_max = y_max = ""
    cx = cy = ""

    if mask_metrics:
        area_pixels = mask_metrics.get("area_pixels", "")
        max_d = mask_metrics.get("max_diameter_pixels", "")
        laterality = mask_metrics.get("laterality", "")

        bbox = mask_metrics.get("bbox")
        if isinstance(bbox, dict):
            x_min = bbox.get("x_min", "")
            y_min = bbox.get("y_min", "")
            x_max = bbox.get("x_max", "")
            y_max = bbox.get("y_max", "")
        elif isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            x_min, y_min, x_max, y_max = bbox

        centroid = mask_metrics.get("centroid")
        if isinstance(centroid, dict):
            cx = centroid.get("x", "")
            cy = centroid.get("y", "")
        elif isinstance(centroid, (list, tuple)) and len(centroid) == 2:
            cx, cy = centroid

    # ── Base structured block ─────────────────────────────────────────────────
    findings_text = f"""BRAINALYZE_STRUCTURED_FINDINGS:
classification={tumor_str}
confidence={conf_str}
tumor_mask={"present" if mask_metrics else "not_detected"}
laterality={laterality}
area_pixels={area_pixels}
max_diameter_pixels={max_d}
bbox_pixels=x_min:{x_min},y_min:{y_min},x_max:{x_max},y_max:{y_max}
centroid_pixels=x:{cx},y:{cy}""".strip()

    # ── 3-D volume block (appended when available) ────────────────────────────
    if volume_3d:
        total   = volume_3d.get("total_volume_cm3", "")
        ncr     = volume_3d.get("ncr_volume_cm3",   "")
        ed      = volume_3d.get("ed_volume_cm3",    "")
        et      = volume_3d.get("et_volume_cm3",    "")
        loc     = volume_3d.get("location_text",    "")

        findings_text += f"""
volume_3d_total_cm3={total}
volume_3d_ncr_cm3={ncr}
volume_3d_edema_cm3={ed}
volume_3d_et_cm3={et}
anatomical_location={loc}""".strip()

    return findings_text
