import re
from datetime import datetime
from flask import render_template, redirect, url_for, request, jsonify

import shared
from utils import _get_logged_doctor, compute_age_from_dob


def _parse_area(findings_bullets):
    """Extract tumor segmentation area (pixels) from findings bullets."""
    for b in (findings_bullets or []):
        m = re.search(r'area[:\s]+(\d+)\s*pixels?', b, re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None


def _safe_age(age_val):
    """Return integer age or None."""
    try:
        return int(str(age_val).strip())
    except Exception:
        return None


@shared.app.route("/reports")
def reports():
    doctor = _get_logged_doctor()
    if not doctor:
        return redirect(url_for("register_login"))

    doctor_id = doctor["id"]

    # ── 1. Patients belonging to this doctor
    patients_docs = list(
        shared.db.collection("Patients")
        .where("CreatedBy", "==", f"/Radiologists/{doctor_id}")
        .stream()
    )
    patient_map = {}
    for p in patients_docs:
        pd = p.to_dict() or {}
        pd["id"] = p.id
        patient_map[p.id] = pd

    # ── 2. Cases → TreatmentPlan (for all patients of this doctor)
    #    Cases have PatientID = "/Patients/<pid>"
    patient_ids = list(patient_map.keys())
    case_map = {}   # case_id → treatment_plan string
    for i in range(0, len(patient_ids), 30):
        batch = patient_ids[i:i+30]
        batch_refs = [f"/Patients/{pid}" for pid in batch]
        if not batch_refs:
            continue
        cases_batch = shared.db.collection("Cases").where("PatientID", "in", batch_refs).stream()
        for cdoc in cases_batch:
            cd = cdoc.to_dict() or {}
            tp = (cd.get("TreatmentPlan") or "").strip() or "—"
            case_map[cdoc.id] = tp

    # ── 3. MRI_Scans for these patients (capture CaseID too)
    scan_map = {}
    for i in range(0, len(patient_ids), 30):
        batch_refs = [f"/Patients/{pid}" for pid in patient_ids[i:i+30]]
        if not batch_refs:
            continue
        scans_batch = shared.db.collection("MRI_Scans").where("PatientID", "in", batch_refs).stream()
        for s in scans_batch:
            sd = s.to_dict() or {}
            sd["id"] = s.id
            pat_ref = (sd.get("PatientID") or "")
            pid = pat_ref.split("/")[-1] if "/" in pat_ref else ""
            if pid in patient_map:
                sd["patient"] = patient_map[pid]
                # resolve CaseID
                case_ref = sd.get("CaseID") or ""
                if hasattr(case_ref, "id"):
                    case_ref = case_ref.id
                elif isinstance(case_ref, str):
                    case_ref = case_ref.split("/")[-1]
                sd["_case_id"] = case_ref
                scan_map[s.id] = sd

    # ── 4. Build reports list
    all_reports_docs = shared.db.collection("Reports").stream()
    reports_list = []
    for rdoc in all_reports_docs:
        rdata = rdoc.to_dict() or {}
        rdata["id"] = rdoc.id

        scan_ref = rdata.get("ScanID")
        scan_id = None
        if hasattr(scan_ref, "id"):
            scan_id = scan_ref.id
        elif isinstance(scan_ref, str):
            scan_id = scan_ref.split("/")[-1]

        if scan_id and scan_id in scan_map:
            scan_info    = scan_map[scan_id]
            patient_info = scan_map[scan_id]["patient"]

            content            = rdata.get("Content") or {}
            findings_bullets   = content.get("findings_bullets") or []
            impression_bullets = content.get("impression_bullets") or []
            created_at         = rdata.get("CreatedAt")

            # Treatment plan via CaseID
            case_id       = scan_info.get("_case_id", "")
            treatment_plan = case_map.get(case_id, "—")

            is_3d          = bool(scan_info.get("Is3D", False))
            volume_metrics = scan_info.get("VolumeMetrics") or {}

            clean = {
                "id": rdoc.id,
                "GradCAMPath":          rdata.get("GradCAMPath") or "",
                "MRIFilePath":          rdata.get("MRIFilePath") or "",
                "SegmentationMaskPath": rdata.get("SegmentationMaskPath") or "",
                "findings_bullets":     findings_bullets,
                "impression_bullets":   impression_bullets,
                "summary":              findings_bullets[0] if findings_bullets else "No findings",
                "created_str":          created_at.strftime("%d %b %Y %H:%M") if isinstance(created_at, datetime) else "—",
                "_sort_ts":             created_at.timestamp() if isinstance(created_at, datetime) else 0,
                "_area_px":             _parse_area(findings_bullets),
                "is_3d":                is_3d,
                "volume_metrics":       volume_metrics,
                "scan": {
                    "id":                   scan_info.get("id", ""),
                    "ClassificationResult": scan_info.get("ClassificationResult") or "Unknown",
                    "ConfidenceScore":      scan_info.get("ConfidenceScore") or 0,
                    "MRIFilePath":          scan_info.get("MRIFilePath") or "",
                    "GradCAMPath":          scan_info.get("GradCAMPath") or "",
                    "SegModelUsed":         scan_info.get("SegModelUsed") or "",
                },
                "patient": {
                    "id":          patient_info.get("id", ""),
                    "FullName":    patient_info.get("FullName") or "Unknown Patient",
                    "Gender":      patient_info.get("Gender") or "",
                    "DateOfBirth": patient_info.get("DateOfBirth") or "",
                    "PhoneNumber": patient_info.get("ContactNumber") or patient_info.get("PhoneNumber") or patient_info.get("Phone") or "",
                    "Age":         compute_age_from_dob(patient_info.get("DateOfBirth") or ""),
                },
                "treatment_plan": treatment_plan,
            }
            reports_list.append(clean)

    reports_list.sort(key=lambda r: r.get("_sort_ts") or 0, reverse=True)

    return render_template("reports.html", doctor=doctor, reports=reports_list)


@shared.app.route("/update_report", methods=["POST"])
def update_report():
    try:
        data = request.get_json(force=True) or {}

        report_id         = (data.get("report_id") or "").strip()
        findings_bullets  = data.get("findings_bullets", [])
        impression_bullets= data.get("impression_bullets", [])

        if not report_id:
            return jsonify({"status": "error", "message": "Missing report_id"}), 400

        if not isinstance(findings_bullets, list) or not isinstance(impression_bullets, list):
            return jsonify({"status": "error", "message": "Invalid report content"}), 400

        findings_bullets  = [str(x).strip() for x in findings_bullets  if str(x).strip()]
        impression_bullets= [str(x).strip() for x in impression_bullets if str(x).strip()]

        report_ref = shared.db.collection("Reports").document(report_id)
        snap       = report_ref.get()

        if not snap.exists:
            return jsonify({"status": "error", "message": "Report not found"}), 404

        old_data    = snap.to_dict() or {}
        old_content = old_data.get("Content", {})

        if isinstance(old_content, dict):
            updated_content = {
                **old_content,
                "findings_bullets":  findings_bullets,
                "impression_bullets": impression_bullets,
                "impression_text":   " ".join(impression_bullets)
            }
        else:
            updated_content = {
                "findings_bullets":  findings_bullets,
                "impression_bullets": impression_bullets,
                "impression_text":   " ".join(impression_bullets)
            }

        report_ref.update({
            "Content":   updated_content,
            "CreatedAt": shared.now_sa()
        })

        return jsonify({
            "status":             "success",
            "report_id":          report_id,
            "findings_bullets":   findings_bullets,
            "impression_bullets": impression_bullets
        }), 200

    except Exception as e:
        print("Error in /update_report:", e)
        return jsonify({"status": "error", "message": str(e)}), 500
