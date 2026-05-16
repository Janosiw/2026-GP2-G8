import os
import re
import numpy as np
from datetime import datetime
from flask import render_template, redirect, url_for, request, jsonify

import shared
from utils import _get_logged_doctor, compute_age_from_dob, _parse_yyyy_mm_dd, call_hf_report_api, save_report_document
from models.segmentation_model import load_segmentation_model, segment_image
from models.classification_model import load_classifier_model, classify_image, generate_gradcam
from models.mask_metrics import compute_mask_metrics
from models.findings_builder import build_findings_text
from models.segmentation_3d import (
    load_nifti_volume, load_3d_model,
    run_3d_segmentation, run_3d_segmentation_multichannel,
    save_3d_results, load_zip_nifti, build_3d_plotly_figure,
    get_best_slice_for_2d_seg, extract_slice_as_image_file,
    load_image_as_pseudo_3d, compute_3d_tumor_metrics,
    render_mpr_slice_b64
)


def _volume_descriptor(vol_cm3):
    """Return a size adjective for tumour volume (input in cm³)."""
    if vol_cm3 is None:
        return ""
    if vol_cm3 < 5:
        return "small"
    if vol_cm3 < 30:
        return "moderate"
    if vol_cm3 < 100:
        return "large"
    return "extensive"


def _fallback_report(findings_text: str, tumor_type: str, metrics: dict) -> dict:
    """Build a clinically meaningful report when the LLM is unavailable."""
    m     = metrics or {}
    tumor = (tumor_type or "unknown").strip().lower()

    # ── 2-D metrics ──────────────────────────────────────────────────────────
    area = m.get("area_pixels")
    lat  = m.get("laterality", "")
    lat_str = f"{lat}-sided " if lat and lat.lower() not in ("midline", "unknown", "") else ""

    # ── 3-D metrics ──────────────────────────────────────────────────────────
    vol_total    = m.get("total_volume_cm3")
    vol_ncr      = m.get("ncr_volume_cm3")
    vol_ed       = m.get("ed_volume_cm3")
    vol_et       = m.get("et_volume_cm3")
    location     = m.get("location_text", "")
    location_list = m.get("location_list", [location] if location else [])
    n_foci       = m.get("n_foci", 1)
    multifocal   = n_foci > 1

    # ── Findings bullets ─────────────────────────────────────────────────────
    findings_bullets = [f"Classification: {tumor.capitalize()}"]

    if multifocal:
        findings_bullets.append(f"Distribution: Multifocal ({n_foci} distinct lesion sites)")
        for i, loc_i in enumerate(location_list, 1):
            findings_bullets.append(f"Focus {i}: {loc_i}")
    elif location:
        findings_bullets.append(f"Anatomical location: {location}")

    findings_bullets += [b for b in [
        f"Laterality: {lat.capitalize()}" if lat else None,
        f"Total tumour volume: {vol_total} cm³" if vol_total is not None else None,
        f"Necrotic core (NCR): {vol_ncr} cm³" if vol_ncr is not None else None,
        f"Peritumoral edema (ED): {vol_ed} cm³" if vol_ed is not None else None,
        f"Enhancing tumour (ET): {vol_et} cm³" if vol_et is not None else None,
        f"Segmentation area: {area} pixels" if area is not None else None,
    ] if b]

    # ── Clinical impression ───────────────────────────────────────────────────
    size_adj   = _volume_descriptor(vol_total)
    vol_phrase = f"{vol_total} cm³" if vol_total is not None else "undetermined"

    # Opening sentence — unifocal vs multifocal
    if multifocal:
        foci_str = "; ".join(location_list)
        imp_lines = [
            f"MRI findings are consistent with {size_adj} multifocal intra-axial lesions "
            f"with imaging characteristics suggestive of {tumor}, "
            f"involving {n_foci} distinct sites: {foci_str}."
        ]
    else:
        loc_phrase = f"in the {location}" if location else ""
        imp_lines = [
            f"MRI findings are consistent with a {size_adj} {lat_str}intra-axial mass lesion "
            f"{loc_phrase}, with imaging characteristics suggestive of {tumor}."
        ]

    # Volumetric breakdown (only for 3-D segmentation)
    if vol_total is not None:
        imp_lines.append(
            f"Volumetric analysis demonstrates a total lesion volume of {vol_phrase}, "
            f"comprising a necrotic core of {vol_ncr} cm³, "
            f"peritumoral edema of {vol_ed} cm³, "
            f"and an enhancing tumour component of {vol_et} cm³."
        )

        # Edema dominance comment
        if vol_ed is not None and vol_total and vol_ed / vol_total > 0.6:
            imp_lines.append(
                "The lesion is characterised by a dominant peritumoral edema component, "
                "indicating significant mass effect and possible midline shift."
            )

        # High-grade comment if ET present (> 5 cm³)
        if vol_et is not None and vol_et > 5:
            imp_lines.append(
                "The presence of a significant enhancing tumour component is associated with "
                "high-grade behaviour; correlation with histopathological grading is recommended."
            )

        # Multifocal-specific clinical note
        if multifocal:
            imp_lines.append(
                "The multifocal distribution pattern suggests either widespread gliomatous infiltration "
                "or leptomeningeal spread; comprehensive staging and neurosurgical evaluation are essential."
            )

    imp_lines.append(
        "Clinical correlation and multidisciplinary neurosurgical review are advised."
    )

    impression = " ".join(imp_lines)

    return {
        "findings_bullets":   findings_bullets,
        "impression_bullets": imp_lines,
        "impression":         impression,
    }

seg_model = load_segmentation_model()
cls_model = load_classifier_model()


@shared.app.route("/scans")
def scans():
    doctor = _get_logged_doctor()
    if not doctor:
        return redirect(url_for("register_login"))

    patient_id = request.args.get("patient_id")
    case_id = request.args.get("case_id")
    first_image = request.args.get("first_image")
    scan_date = request.args.get("scan_date", "").strip()

    patient_name = ""
    patient_phone = ""
    patient_age = ""
    patient_gender = ""
    if patient_id:
        p_doc = shared.db.collection("Patients").document(patient_id).get()
        if p_doc.exists:
            pd = p_doc.to_dict()
            patient_name = pd.get("FullName", "")
            patient_phone = pd.get("ContactNumber", pd.get("PhoneNumber", pd.get("Phone", "")))
            dob = pd.get("DateOfBirth", "")
            patient_gender = pd.get("Gender", "")
            if dob:
                try:
                    from datetime import date as _date
                    born = datetime.strptime(dob, "%Y-%m-%d").date()
                    today = _date.today()
                    patient_age = str(today.year - born.year - ((today.month, today.day) < (born.month, born.day)))
                except Exception:
                    patient_age = ""

    # Detect scan types already in this case — block mixing 2D and 3D
    has_2d_scan = False
    has_3d_scan = False
    if case_id:
        try:
            existing = (
                shared.db.collection("MRI_Scans")
                .where("CaseID", "==", f"/Cases/{case_id}")
                .limit(10)
                .stream()
            )
            for s in existing:
                sd = s.to_dict() or {}
                if sd.get("InputType", "") == "3d" or sd.get("Is3D"):
                    has_3d_scan = True
                else:
                    has_2d_scan = True
                if has_2d_scan and has_3d_scan:
                    break
        except Exception:
            pass

    return render_template(
        "scans.html",
        doctor=doctor,
        patient_id=patient_id,
        case_id=case_id,
        scan_date=scan_date,
        first_image=first_image,
        patient_name=patient_name,
        patient_phone=patient_phone,
        patient_age=patient_age,
        patient_gender=patient_gender,
        has_2d_scan=has_2d_scan,
        has_3d_scan=has_3d_scan,
    )


@shared.app.route("/analyze_mri", methods=["POST"])
def analyze_mri():
    try:
        file = request.files.get("file")
        patient_id = request.form.get("patient_id", "")
        case_id = request.form.get("case_id", "")
        selected_scan_date = request.form.get("scan_date", "").strip()
        doctor = _get_logged_doctor()

        if not file or not patient_id:
            return jsonify({"status": "error", "message": "Missing file or patient_id"}), 400

        # Block 2D upload if this case already has a 3D scan
        if case_id:
            try:
                existing = (
                    shared.db.collection("MRI_Scans")
                    .where("CaseID", "==", f"/Cases/{case_id}")
                    .limit(10)
                    .stream()
                )
                for s in existing:
                    sd = s.to_dict() or {}
                    if sd.get("InputType", "") == "3d" or sd.get("Is3D"):
                        return jsonify({
                            "status": "error",
                            "message": "This case already has a 3D MRI scan. Please create a new case to add a 2D scan."
                        }), 400
            except Exception:
                pass

        # Reserve a Firestore doc ID (not written yet)
        scan_id = shared.db.collection("MRI_Scans").document().id

        filename = f"{shared.now_sa().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        save_path = os.path.join(shared.app.config["UPLOAD_FOLDER"], filename)
        file.save(save_path)

        rel_original = "/" + save_path.replace("\\", "/")

        now = shared.now_sa()
        parsed_scan_date = _parse_yyyy_mm_dd(selected_scan_date)
        if parsed_scan_date:
            upload_dt = datetime.combine(parsed_scan_date, now.time())
            last_mri_date_str = parsed_scan_date.strftime("%Y-%m-%d")
        else:
            upload_dt = now
            last_mri_date_str = now.strftime("%Y-%m-%d")

        tumor_type, confidence = classify_image(cls_model, save_path)
        gradcam_path, pred_idx = generate_gradcam(cls_model, save_path, save_name=f"gradcam_{scan_id}.png")
        rel_gradcam = "/" + gradcam_path.replace("\\", "/")

        # Store in memory — Firestore write happens only on /finish_scan
        shared._pending_scans[scan_id] = {
            "patient_id": patient_id,
            "case_id": case_id,
            "doctor_id": doctor["id"] if doctor else "",
            "upload_dt": upload_dt,
            "last_mri_date_str": last_mri_date_str,
            "mri_path": rel_original,
            "mri_fs_path": save_path,
            "tumor_type": tumor_type,
            "confidence": confidence,
            "gradcam_path": rel_gradcam,
            "mask_path": None,
            "findings_text": "",
            "findings_bullets": [],
            "impression_bullets": [],
            "impression": "",
            "mask_metrics": None,
            "is_3d": False,
        }

        return jsonify({
            "status": "success",
            "scan_id": scan_id,
            "original": rel_original,
            "mask": None,
            "gradcam": rel_gradcam,
            "tumor_type": tumor_type,
            "confidence": confidence,
            "description": f"Detected {tumor_type} with {confidence:.1f}% confidence"
        }), 200

    except Exception as e:
        print("Error in /analyze_mri:", e)
        return jsonify({"status": "error", "message": str(e)}), 500


@shared.app.route("/segment_only", methods=["POST"])
def segment_only():
    try:
        scan_id = request.form.get("scan_id", "").strip()
        if not scan_id:
            return jsonify({"status": "error", "message": "Missing scan_id"}), 400

        # Try pending dict first; fall back to Firestore for older scans
        pending = shared._pending_scans.get(scan_id)
        if pending:
            mri_fs_path = pending.get("mri_fs_path", "")
            tumor_type  = pending.get("tumor_type")
            confidence  = pending.get("confidence")
        else:
            snap = shared.db.collection("MRI_Scans").document(scan_id).get()
            if not snap.exists:
                return jsonify({"status": "error", "message": "Scan not found"}), 404
            data = snap.to_dict() or {}
            mri_path = data.get("MRIFilePath")
            if not mri_path:
                return jsonify({"status": "error", "message": "Missing MRIFilePath"}), 400
            mri_fs_path = mri_path.lstrip("/")
            tumor_type  = data.get("ClassificationResult")
            confidence  = data.get("ConfidenceScore")

        mask_path = segment_image(seg_model, mri_fs_path, scan_id=scan_id)
        rel_mask = "/" + mask_path.replace("\\", "/")

        metrics = compute_mask_metrics(mask_path)

        findings_text = build_findings_text(
            tumor_type=tumor_type,
            confidence=confidence,
            mask_metrics=metrics
        )

        try:
            report = call_hf_report_api(findings_text)
            findings_bullets  = report.get("findings_bullets", [])
            impression_bullets = report.get("impression_bullets", [])
            impression = report.get("impression") or " ".join(impression_bullets)
        except Exception as llm_err:
            print(f"LLM unavailable, using fallback: {llm_err}")
            fallback = _fallback_report(findings_text, tumor_type, metrics)
            findings_bullets  = fallback["findings_bullets"]
            impression_bullets = fallback["impression_bullets"]
            impression = fallback["impression"]
        # Update pending dict (Firestore write happens on /finish_scan)
        if pending:
            pending["mask_path"]         = rel_mask
            pending["mask_metrics"]      = metrics
            pending["findings_text"]     = findings_text
            pending["findings_bullets"]  = findings_bullets
            pending["impression_bullets"] = impression_bullets
            pending["impression"]        = impression
        else:
            # Fallback: update Firestore directly for legacy saved scans
            shared.db.collection("MRI_Scans").document(scan_id).update({
                "SegmentationMaskPath": rel_mask,
                "MaskMetrics": metrics,
                "FindingsText": findings_text,
                "FindingsBullets": findings_bullets,
                "ImpressionBullets": impression_bullets,
                "ImpressionText": impression,
                "LastUpdate": shared.now_sa()
            })

        return jsonify({
            "status": "success",
            "mask": rel_mask,
            "mask_metrics": metrics,
            "findings_text": findings_text,
            "findings_bullets": findings_bullets,
            "impression_bullets": impression_bullets,
            "impression": impression,
            "report_id": ""
        }), 200

    except Exception as e:
        print("Error in /segment_only:", e)
        return jsonify({"status": "error", "message": str(e)}), 500


@shared.app.route("/finish_scan", methods=["POST"])
def finish_scan():
    """Save a pending scan to Firestore. Called when the doctor clicks Finish Analysis."""
    try:
        scan_id = request.form.get("scan_id", "").strip()
        if not scan_id:
            return jsonify({"status": "ok", "message": "No scan_id provided"}), 200

        pending = shared._pending_scans.pop(scan_id, None)
        if not pending:
            # Already saved or never existed — silently succeed
            return jsonify({"status": "ok"}), 200

        now          = shared.now_sa()
        patient_id   = pending["patient_id"]
        case_id      = pending.get("case_id", "")
        upload_dt    = pending["upload_dt"]
        last_mri_str = pending["last_mri_date_str"]
        is_3d        = pending.get("is_3d", False)
        conf         = pending.get("confidence") or 0.0
        ttype        = pending.get("tumor_type") or ""

        scan_data = {
            "ScanID":                scan_id,
            "PatientID":             f"/Patients/{patient_id}",
            "CaseID":                f"/Cases/{case_id}",
            "MRIFilePath":           pending.get("mri_path", ""),
            "SegmentationMaskPath":  pending.get("mask_path"),
            "GradCAMPath":           pending.get("gradcam_path"),
            "ClassificationResult":  ttype,
            "ConfidenceScore":       conf,
            "QuickDescription":      f"Detected {ttype} with {conf:.1f}% confidence",
            # UploadDate = actual time doctor clicked Finish Analysis (UTC for Firestore)
            "UploadDate":            now,
            "UploadDateStr":         shared.fmt_sa_verbose(now),
            # ScanAcquisitionDate = the date the MRI scan was physically taken (from picker)
            "ScanAcquisitionDate":   last_mri_str,
            "FindingsText":          pending.get("findings_text", ""),
            "FindingsBullets":       pending.get("findings_bullets", []),
            "ImpressionBullets":     pending.get("impression_bullets", []),
            "ImpressionText":        pending.get("impression", ""),
            "MaskMetrics":           pending.get("mask_metrics"),
            "LastUpdate":            now,
        }

        if is_3d:
            scan_data.update({
                "InputType":       "3d",
                "Is3D":            True,
                "SegModelUsed":    pending.get("seg_model_used", "3d_glioma"),
                "VolumeMetrics":   pending.get("volume_metrics"),
                "Slices3D":        pending.get("slices_3d", []),
                "Plot3DPath":      pending.get("plot3d_path", ""),
            })

        # Include doctor feedback if submitted before Finish was clicked
        if pending.get("doctor_approval"):
            scan_data["DoctorApproval"] = pending["doctor_approval"]

        shared.db.collection("MRI_Scans").document(scan_id).set(scan_data)

        shared.db.collection("Patients").document(patient_id).update({
            "LastMRIDate":        last_mri_str,
            "LastScanAt":         now,
            "LastScanId":         scan_id,
            "LastScanTumor":      ttype,
            "LastScanConfidence": conf,
            # Display strings stored in SA time (+3h)
            "LastScanUploadStr":  shared.fmt_sa(now),
            "LastActivityAt":     now,
            "LastVisitedAt":      now,
        })

        shared.clear_dash_cache(pending.get("doctor_id"))
        return jsonify({"status": "success"}), 200

    except Exception as e:
        print("Error in /finish_scan:", e)
        return jsonify({"status": "error", "message": str(e)}), 500


@shared.app.route("/view_scan")
def view_scan():
    doctor = _get_logged_doctor()
    if not doctor:
        return redirect(url_for("register_login"))

    scan_id = request.args.get("scan_id")
    scan_number = request.args.get("scan_number")
    case_id = request.args.get("case_id")
    case_number = request.args.get("case_number")

    if not scan_id:
        return "Missing scan_id", 400

    snap = shared.db.collection("MRI_Scans").document(scan_id).get()
    if not snap.exists:
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return jsonify({"status": "error", "message": "Scan not found"}), 404
        return "Scan not found", 404

    d = snap.to_dict() or {}

    patient_ref = d.get("PatientID")
    patient_id = None
    if isinstance(patient_ref, str):
        patient_id = patient_ref.split("/")[-1]

    if not case_id:
        case_ref = d.get("CaseID")
        if isinstance(case_ref, str):
            case_id = case_ref.split("/")[-1]

    if case_id and not case_number:
        all_cases = list(
            shared.db.collection("Cases")
            .where("PatientID", "==", f"/Patients/{patient_id}")
            .stream()
        )
        sorted_cases = sorted(all_cases, key=lambda x: x.to_dict().get("StartDate") or "")
        for idx, c in enumerate(sorted_cases, start=1):
            if c.id == case_id:
                case_number = idx
                break

    patient = None
    is_owner = False
    if patient_id:
        p_doc = shared.db.collection("Patients").document(patient_id).get()
        if p_doc.exists:
            pdata = p_doc.to_dict() or {}
            patient = {
                "patient_id": patient_id,
                "name": pdata.get("FullName", "")
            }
            is_owner = (pdata.get("CreatedBy", "") == f"/Radiologists/{doctor['id']}")

    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        scan_ref_doc = shared.db.document(f"MRI_Scans/{scan_id}")
        report_docs = list(shared.db.collection("Reports").where("ScanID", "==", scan_ref_doc).stream())
        report_id = ""
        report_findings = []
        report_impression = []
        report_mask = ""
        report_mri = ""
        report_date = ""
        patient_phone = ""
        patient_age = ""
        patient_gender = ""
        patient_name_full = ""
        if report_docs:
            rdoc = report_docs[0]
            rdata = rdoc.to_dict() or {}
            content = rdata.get("Content") or {}
            report_id = rdoc.id
            report_findings = content.get("findings_bullets") or []
            report_impression = content.get("impression_bullets") or []
            report_mask = rdata.get("SegmentationMaskPath") or ""
            report_mri = rdata.get("MRIFilePath") or d.get("MRIFilePath") or ""
            created_at = rdata.get("CreatedAt")
            report_date = created_at.strftime("%d %b %Y") if isinstance(created_at, datetime) else ""
        if patient_id:
            p_doc2 = shared.db.collection("Patients").document(patient_id).get()
            if p_doc2.exists:
                pd2 = p_doc2.to_dict()
                patient_name_full = pd2.get("FullName", "")
                patient_phone = pd2.get("ContactNumber", pd2.get("PhoneNumber", pd2.get("Phone", "")))
                patient_gender = pd2.get("Gender", "")
                patient_age = compute_age_from_dob(pd2.get("DateOfBirth", ""))
        # ── 3D-specific fields from MRI_Scans ────────────────────────────────
        volume_metrics = d.get("VolumeMetrics") or {}
        slices_3d      = d.get("Slices3D") or []
        seg_model      = d.get("SegModelUsed", "")
        # Robust 3D check: any of these signals confirm a 3D scan
        is_3d = bool(
            d.get("Is3D")
            or seg_model == "3d_glioma"
            or slices_3d
            or volume_metrics
        )

        # ── Reconstruct slices from disk for older scans ──────────────────
        if is_3d and not slices_3d:
            overlays_dir = os.path.join(
                shared.app.root_path, "static", "uploads", "overlays_3d"
            )
            if os.path.isdir(overlays_dir):
                import glob as _glob
                pattern = os.path.join(overlays_dir, f"overlay3d_{scan_id}_*.png")
                found = sorted(_glob.glob(pattern))
                for fp in found:
                    base = os.path.basename(fp)           # overlay3d_SCANID_N.png
                    try:
                        z_idx = int(base.rsplit("_", 1)[1].replace(".png", ""))
                    except Exception:
                        z_idx = 0
                    slices_3d.append({
                        "overlay": f"/static/uploads/overlays_3d/{base}",
                        "slice_idx": z_idx,
                    })

        plot3d_path    = d.get("Plot3DPath", "")
        has_plot3d     = bool(plot3d_path and os.path.exists(
            os.path.join(shared.app.root_path,
                         plot3d_path.lstrip("/").replace("static/", "static/", 1))
        ))

        raw_approval = d.get("DoctorApproval") or {}
        doctor_approval = None
        if raw_approval.get("decision"):
            saved_ts = raw_approval.get("timestamp")
            saved_at_str = ""
            if isinstance(saved_ts, datetime):
                saved_at_str = saved_ts.strftime("%d %b %Y, %H:%M")
            doctor_approval = {
                "decision": raw_approval.get("decision", ""),
                "reason":   raw_approval.get("reason", ""),
                "saved_at": saved_at_str,
            }

        return jsonify({
            "status": "success",
            "tumor_type": d.get("ClassificationResult"),
            "confidence": d.get("ConfidenceScore"),
            "original": d.get("MRIFilePath"),
            "gradcam": d.get("GradCAMPath"),
            "mask": d.get("SegmentationMaskPath"),
            "description": d.get("QuickDescription"),
            "report_id": report_id,
            "report_findings": report_findings,
            "report_impression": report_impression,
            "report_mask": report_mask,
            "report_mri": report_mri,
            "report_date": report_date,
            "patient_name": patient_name_full,
            "patient_phone": patient_phone,
            "patient_age": patient_age,
            "patient_gender": patient_gender,
            "doctor_approval": doctor_approval,
            # 3D fields
            "is_3d":                   is_3d,
            "volume_metrics":          volume_metrics,
            "slices_3d":               slices_3d,
            "segmentation_model_used": seg_model,
            "plot3d_path":             plot3d_path,
            "has_plot3d":              has_plot3d,
        })

    return render_template(
        "scan_view.html",
        doctor=doctor,
        scan_id=scan_id,
        scan_number=scan_number,
        case_id=case_id,
        case_number=case_number,
        patient=patient,
        is_owner=is_owner
    )


@shared.app.route("/api/scan_approval", methods=["POST"])
def api_scan_approval():
    doctor = _get_logged_doctor()
    if not doctor:
        return jsonify({"status": "error", "message": "Not authenticated"}), 401

    data = request.get_json(silent=True) or {}
    scan_id  = (data.get("scan_id") or "").strip()
    decision = (data.get("decision") or "").strip()
    reason   = (data.get("reason") or "").strip()

    if not scan_id or decision not in ("approved", "rejected"):
        return jsonify({"status": "error", "message": "Invalid data"}), 400

    if decision == "rejected" and not reason:
        return jsonify({"status": "error", "message": "Reason required for rejection"}), 400

    now = datetime.utcnow()
    approval_payload = {
        "decision":  decision,
        "reason":    reason,
        "timestamp": now,
        "doctor_id": doctor["id"],
    }

    # Cache in memory so it's included when Finish saves the full scan
    if scan_id in shared._pending_scans:
        shared._pending_scans[scan_id]["doctor_approval"] = approval_payload

    # Save directly to Firestore immediately (merge so we don't overwrite existing fields)
    shared.db.collection("MRI_Scans").document(scan_id).set(
        {"DoctorApproval": approval_payload},
        merge=True
    )

    return jsonify({
        "status": "success",
        "saved_at": now.strftime("%d %b %Y, %H:%M"),
    })


@shared.app.route("/load_plot3d")
def load_plot3d():
    """Return the saved Plotly 3D JSON for a scan so scan_view can render it.
    Applies a live opacity boost to the brain mesh so older plots look correct too.
    """
    import json as _json
    scan_id = request.args.get("scan_id", "").strip()
    if not scan_id:
        return jsonify({"status": "error", "message": "Missing scan_id"}), 400

    snap = shared.db.collection("MRI_Scans").document(scan_id).get()
    if not snap.exists:
        return jsonify({"status": "error", "message": "Scan not found"}), 404

    plot3d_path = (snap.to_dict() or {}).get("Plot3DPath", "")
    if not plot3d_path:
        return jsonify({"status": "error", "message": "No 3D plot for this scan"}), 404

    rel = plot3d_path.lstrip("/")
    abs_path = os.path.join(shared.app.root_path, rel)

    if not os.path.exists(abs_path):
        return jsonify({"status": "error", "message": "Plot file missing on disk"}), 404

    with open(abs_path, "r") as f:
        fig_spec = _json.load(f)

    # ── Live-patch brain opacity so old & new plots both look correct ──────
    for trace in fig_spec.get("data", []):
        if trace.get("name") == "Brain":
            trace["opacity"] = 0.40
            trace["color"]   = "rgb(200, 215, 245)"

    from flask import Response
    return Response(_json.dumps(fig_spec), mimetype="application/json")


@shared.app.route("/download_slices_3d")
def download_slices_3d():
    """
    Build a ZIP of the 3D segmentation overlay slices for a scan and return it
    as a file download.  Slices are ordered by slice_idx (ascending).
    """
    import io, zipfile
    from flask import send_file

    doctor = _get_logged_doctor()
    if not doctor:
        return "Not authenticated", 401

    scan_id = request.args.get("scan_id", "").strip()
    if not scan_id:
        return "Missing scan_id", 400

    snap = shared.db.collection("MRI_Scans").document(scan_id).get()
    if not snap.exists:
        return "Scan not found", 404

    data      = snap.to_dict() or {}
    slices_3d = data.get("Slices3D", [])

    if not slices_3d:
        return "No 3D slices found for this scan", 404

    # Sort by slice_idx
    slices_sorted = sorted(slices_3d, key=lambda s: int(s.get("slice_idx", 0)))

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, sl in enumerate(slices_sorted, start=1):
            overlay_url  = sl.get("overlay", "")
            slice_idx    = sl.get("slice_idx", "?")
            rel_path     = overlay_url.lstrip("/")
            abs_path     = os.path.join(shared.app.root_path, rel_path)
            if os.path.exists(abs_path):
                arcname = f"slice_{i:02d}_axial_{slice_idx}.png"
                zf.write(abs_path, arcname)

    buf.seek(0)
    zip_name = f"brainalyze_3d_slices_{scan_id[:8]}.zip"
    return send_file(buf, mimetype="application/zip",
                     as_attachment=True, download_name=zip_name)


@shared.app.route("/load_more_scans")
def load_more_scans():
    offset = int(request.args.get("offset", 0))

    scans_query = shared.db.collection("MRI_Scans").stream()

    scans_list = []
    for s in scans_query:
        sd = s.to_dict()
        upload_date = sd.get("UploadDate")

        patient_ref = sd.get("PatientID")
        if isinstance(patient_ref, str):
            patient_id = patient_ref.split("/")[-1]
        elif hasattr(patient_ref, "id"):
            patient_id = patient_ref.id
        else:
            patient_id = None

        pdata = {}
        if patient_id:
            patient_doc = shared.db.collection("Patients").document(patient_id).get()
            pdata = patient_doc.to_dict() if patient_doc.exists else {}

        scans_list.append({
            "id": s.id,
            "FullName": pdata.get("FullName", ""),
            "UploadDate": upload_date.strftime("%Y-%m-%d %H:%M") if hasattr(upload_date, "strftime") else "",
        })

    scans_list = sorted(scans_list, key=lambda x: x["UploadDate"], reverse=True)
    more = scans_list[offset: offset + 5]

    return jsonify({"scans": more, "count": len(more)})


# ─── 3D Segmentation on 2D Image (pseudo-3D) ────────────────────────────────

@shared.app.route("/segment_only_3d", methods=["POST"])
def segment_only_3d():
    """
    Accepts:
      - scan_id  (form field)       — Firestore scan document ID
      - file     (multipart/form)   — single .nii/.nii.gz OR .zip containing .nii
    Runs the 3D glioma segmentation model and returns coloured overlays.
    """
    try:
        scan_id   = request.form.get("scan_id", "").strip()
        nifti_file = request.files.get("file")

        if not scan_id:
            return jsonify({"status": "error", "message": "Missing scan_id"}), 400
        if not nifti_file:
            return jsonify({"status": "error", "message": "Missing NIfTI/ZIP file"}), 400

        scan_ref = shared.db.collection("MRI_Scans").document(scan_id)
        snap     = scan_ref.get()
        if not snap.exists:
            return jsonify({"status": "error", "message": "Scan not found"}), 404
        data = snap.to_dict() or {}

        # Save the uploaded file temporarily
        orig_name  = nifti_file.filename or "upload"
        ext        = ".zip" if orig_name.lower().endswith(".zip") else ".nii.gz"
        save_path  = os.path.join(shared.app.config["UPLOAD_FOLDER"], f"seg3d_{scan_id}{ext}")
        nifti_file.save(save_path)

        # Load volume(s):
        # ZIP with FLAIR/T1/T1CE/T2 → full 4-channel multichannel inference (proper BraTS)
        # Single .nii/.nii.gz        → single-channel pseudo-3D fallback
        model_3d = load_3d_model()
        if save_path.endswith(".zip"):
            # load_zip_nifti reads each modality by filename and returns them in BraTS order
            channels, voxel_spacing, affine = load_zip_nifti(save_path)
            flair_vol     = channels[0]   # FLAIR for overlay rendering & visualisation
            pred_label, tumor_mask = run_3d_segmentation_multichannel(model_3d, channels)
        else:
            flair_vol     = load_nifti_volume(save_path)
            voxel_spacing = (1.0, 1.0, 1.0)
            affine        = None
            pred_label, tumor_mask = run_3d_segmentation(model_3d, flair_vol)

        slices_info = save_3d_results(flair_vol, pred_label, scan_id)

        # Save pred_label + flair_vol so overlays can be regenerated later
        try:
            npz_dir  = os.path.join(shared.app.config["UPLOAD_FOLDER"], "seg3d_cache")
            os.makedirs(npz_dir, exist_ok=True)
            npz_path = os.path.join(npz_dir, f"pred_{scan_id}.npz")
            np.savez_compressed(npz_path, pred_label=pred_label, flair_vol=flair_vol)
        except Exception as npz_err:
            print(f"[3D] Could not cache pred_label: {npz_err}")

        _cache_mpr_preview(scan_ref, flair_vol, pred_label)

        # Compute accurate 3-D tumour volume (cm³) and anatomical location
        # voxel_spacing/affine not available for single-channel input — use defaults
        vol_metrics = compute_3d_tumor_metrics(pred_label, voxel_spacing, affine)

        # Build interactive 3D Plotly figure — brain shell + tumour regions
        plot3d_json = build_3d_plotly_figure(pred_label, flair_vol=flair_vol)

        tumor_type = data.get("ClassificationResult")
        confidence = data.get("ConfidenceScore")

        findings_text = build_findings_text(
            tumor_type=tumor_type,
            confidence=confidence,
            mask_metrics=None,
            volume_3d=vol_metrics,
        )

        try:
            report            = call_hf_report_api(findings_text)
            findings_bullets  = report.get("findings_bullets", [])
            impression_bullets = report.get("impression_bullets", [])
            impression        = report.get("impression") or " ".join(impression_bullets)
        except Exception as llm_err:
            print(f"LLM unavailable, using fallback: {llm_err}")
            fallback           = _fallback_report(findings_text, tumor_type, vol_metrics)
            findings_bullets   = fallback["findings_bullets"]
            impression_bullets = fallback["impression_bullets"]
            impression         = fallback["impression"]

        report_content = {
            "findings_text":      findings_text,
            "findings_bullets":   findings_bullets,
            "impression_bullets": impression_bullets,
            "impression_text":    impression
        }

        primary_overlay = slices_info[len(slices_info) // 2]["overlay"] if slices_info else None

        # Sanitize vol_metrics for Firestore (no numpy/tuple types)
        vm_store = {
            "total_volume_cm3": float(vol_metrics.get("total_volume_cm3") or 0),
            "ncr_volume_cm3":   float(vol_metrics.get("ncr_volume_cm3")   or 0),
            "ed_volume_cm3":    float(vol_metrics.get("ed_volume_cm3")    or 0),
            "et_volume_cm3":    float(vol_metrics.get("et_volume_cm3")    or 0),
            "location_text":    str(vol_metrics.get("location_text")      or ""),
            "location_list":    [str(l) for l in (vol_metrics.get("location_list") or [])],
            "n_foci":           int(vol_metrics.get("n_foci")             or 1),
            "centroid_voxel":   list(vol_metrics.get("centroid_voxel")    or []),
        }

        # Save slice paths (not base64) for later retrieval
        slices_store = [
            {"overlay": s["overlay"], "slice_idx": int(s["slice_idx"])}
            for s in slices_info
        ]

        # ── Persist plot3d_json to disk so it can be reloaded later ─────────
        plot3d_path = ""
        if plot3d_json:
            plot3d_filename = f"plot3d_{scan_id}.json"
            plot3d_filepath = os.path.join(shared.app.config["UPLOAD_FOLDER"], plot3d_filename)
            try:
                with open(plot3d_filepath, "w") as pf:
                    pf.write(plot3d_json)
                plot3d_path = f"/static/uploads/{plot3d_filename}"
            except Exception as pe:
                print(f"Warning: could not save plot3d_json: {pe}")

        scan_ref.update({
            "ClassificationResult": "Glioma",
            "ConfidenceScore":      100.0,
            "SegmentationMaskPath": primary_overlay,
            "FindingsText":         findings_text,
            "FindingsBullets":      findings_bullets,
            "ImpressionBullets":    impression_bullets,
            "ImpressionText":       impression,
            "SegModelUsed":         "3d_glioma",
            "Is3D":                 True,
            "InputType":            "3d",
            "VolumeMetrics":        vm_store,
            "Slices3D":             slices_store,
            "Plot3DPath":           plot3d_path,
            "LastUpdate":           shared.now_sa()
        })

        # Clean up uploaded file
        try:
            os.remove(save_path)
        except Exception:
            pass

        return jsonify({
            "status":                   "success",
            "slices":                   slices_info,
            "plot3d_url":               plot3d_path,
            "segmentation_model_used":  "3d_glioma",
            "findings_bullets":         findings_bullets,
            "impression_bullets":       impression_bullets,
            "impression":               impression,
            "report_id":                "",
            "volume_metrics":           vol_metrics,
        }), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


# ─── 3D NIfTI MRI Pipeline ───────────────────────────────────────────────────

@shared.app.route("/analyze_mri_3d", methods=["POST"])
def analyze_mri_3d():
    try:
        file = request.files.get("file")
        patient_id = request.form.get("patient_id", "")
        case_id = request.form.get("case_id", "")
        selected_scan_date = request.form.get("scan_date", "").strip()

        if not file or not patient_id:
            return jsonify({"status": "error", "message": "Missing file or patient_id"}), 400

        filename = f"{shared.now_sa().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        save_path = os.path.join(shared.app.config["UPLOAD_FOLDER"], filename)
        file.save(save_path)

        now = shared.now_sa()
        parsed_scan_date = _parse_yyyy_mm_dd(selected_scan_date)
        if parsed_scan_date:
            upload_dt = datetime.combine(parsed_scan_date, now.time())
            last_mri_date_str = parsed_scan_date.strftime("%Y-%m-%d")
        else:
            upload_dt = now
            last_mri_date_str = now.strftime("%Y-%m-%d")

        scan_id = shared.db.collection("MRI_Scans").document().id

        # Step 1: Load the NIfTI volume(s)
        # ZIP with FLAIR/T1/T1CE/T2 → full 4-channel multichannel inference (proper BraTS)
        # Single .nii/.nii.gz        → single-channel pseudo-3D fallback
        fname_lower = save_path.lower()
        model_3d    = load_3d_model()
        if fname_lower.endswith('.zip'):
            # load_zip_nifti detects FLAIR/T1/T1CE/T2 by filename → correct BraTS order
            channels, voxel_spacing, affine = load_zip_nifti(save_path)
            volume        = channels[0]   # FLAIR for overlay rendering & visualisation
            pred_label, _ = run_3d_segmentation_multichannel(model_3d, channels)
        else:
            volume        = load_nifti_volume(save_path)
            voxel_spacing = (1.0, 1.0, 1.0)
            affine        = None
            pred_label, _ = run_3d_segmentation(model_3d, volume)

        # Step 2: 3D NIfTI → always Glioma (2D classifier not reliable on NIfTI slices)
        tumor_type = "Glioma"
        confidence = 100.0
        prob_dict  = {"Glioma": 100.0, "Meningioma": 0.0, "Pituitary": 0.0, "No Tumor": 0.0}

        # Step 3: Save overlays and select primary mask
        slices_info    = save_3d_results(volume, pred_label, scan_id)
        seg_model_used = "3d_glioma"
        primary_mask   = slices_info[len(slices_info) // 2]["overlay"] if slices_info else None

        # Cache pred_label + flair_vol for MPR viewer
        try:
            npz_dir  = os.path.join(shared.app.config["UPLOAD_FOLDER"], "seg3d_cache")
            os.makedirs(npz_dir, exist_ok=True)
            npz_path = os.path.join(npz_dir, f"pred_{scan_id}.npz")
            np.savez_compressed(npz_path, pred_label=pred_label, flair_vol=volume)
        except Exception as npz_err:
            print(f"[3D] Could not cache pred_label: {npz_err}")

        # MPR preview is cached from NPZ on first /api/mpr_preview request (no Firestore write needed yet)

        # Step 4: Compute volume metrics + interactive 3D Plotly figure
        vol_metrics  = compute_3d_tumor_metrics(pred_label, voxel_spacing, affine)
        plot3d_json  = build_3d_plotly_figure(pred_label, flair_vol=volume)

        vm_store = {
            "total_volume_cm3": float(vol_metrics.get("total_volume_cm3") or 0),
            "ncr_volume_cm3":   float(vol_metrics.get("ncr_volume_cm3")   or 0),
            "ed_volume_cm3":    float(vol_metrics.get("ed_volume_cm3")    or 0),
            "et_volume_cm3":    float(vol_metrics.get("et_volume_cm3")    or 0),
            "location_text":    str(vol_metrics.get("location_text")      or ""),
            "location_list":    [str(l) for l in (vol_metrics.get("location_list") or [])],
            "n_foci":           int(vol_metrics.get("n_foci")             or 1),
            "centroid_voxel":   list(vol_metrics.get("centroid_voxel")    or []),
        }

        slices_store = [
            {"overlay": s["overlay"], "slice_idx": int(s["slice_idx"])}
            for s in slices_info
        ]

        # Save plot3d JSON to disk
        plot3d_path = ""
        if plot3d_json:
            plot3d_filename = f"plot3d_{scan_id}.json"
            plot3d_filepath = os.path.join(shared.app.config["UPLOAD_FOLDER"], plot3d_filename)
            try:
                with open(plot3d_filepath, "w") as pf:
                    pf.write(plot3d_json)
                plot3d_path = f"/static/uploads/{plot3d_filename}"
            except Exception as pe:
                print(f"Warning: could not save plot3d_json: {pe}")

        # Step 5: Generate AI radiology report (same as segment_only_3d)
        findings_text = build_findings_text(
            tumor_type=tumor_type,
            confidence=confidence,
            mask_metrics=None,
            volume_3d=vol_metrics,
        )
        try:
            report            = call_hf_report_api(findings_text)
            findings_bullets  = report.get("findings_bullets", [])
            impression_bullets = report.get("impression_bullets", [])
            impression        = report.get("impression") or " ".join(impression_bullets)
        except Exception as llm_err:
            print(f"LLM unavailable, using fallback: {llm_err}")
            fallback           = _fallback_report(findings_text, tumor_type, vol_metrics)
            findings_bullets   = fallback["findings_bullets"]
            impression_bullets = fallback["impression_bullets"]
            impression         = fallback["impression"]

        report_content = {
            "findings_text":      findings_text,
            "findings_bullets":   findings_bullets,
            "impression_bullets": impression_bullets,
            "impression_text":    impression
        }

        # Store in memory — Firestore write happens only on /finish_scan
        doctor = _get_logged_doctor()
        shared._pending_scans[scan_id] = {
            "patient_id":      patient_id,
            "case_id":         case_id,
            "doctor_id":       doctor["id"] if doctor else "",
            "upload_dt":       upload_dt,
            "last_mri_date_str": last_mri_date_str,
            "mri_path":        "/" + save_path.replace("\\", "/"),
            "mri_fs_path":     save_path,
            "tumor_type":      tumor_type,
            "confidence":      confidence,
            "gradcam_path":    None,
            "mask_path":       primary_mask,
            "findings_text":   findings_text,
            "findings_bullets": findings_bullets,
            "impression_bullets": impression_bullets,
            "impression":      impression,
            "mask_metrics":    None,
            "is_3d":           True,
            "seg_model_used":  seg_model_used,
            "volume_metrics":  vm_store,
            "slices_3d":       slices_store,
            "plot3d_path":     plot3d_path,
        }

        return jsonify({
            "status": "success",
            "input_type": "3d",
            "scan_id": scan_id,
            "tumor_type": tumor_type,
            "confidence": confidence,
            "class_probabilities": prob_dict,
            "segmentation_model_used": seg_model_used,
            "slices": slices_info,
            "plot3d_url": plot3d_path,
            "volume_metrics": vm_store,
            "findings_bullets": findings_bullets,
            "impression_bullets": impression_bullets,
            "report_id": "",
            "description": f"3D MRI: {tumor_type} detected with {confidence:.1f}% confidence"
        }), 200

    except RuntimeError as e:
        import traceback; traceback.print_exc()
        msg = str(e)
        print("RuntimeError in /analyze_mri_3d:", msg)
        return jsonify({"status": "error", "message": msg}), 500
    except Exception as e:
        import traceback; traceback.print_exc()
        msg = str(e)
        print("Error in /analyze_mri_3d:", msg)
        # Give a friendlier message for common failures
        if "nibabel" in msg or "nib" in msg.lower():
            msg = "Could not read the NIfTI file. Make sure it is a valid .nii or .nii.gz file."
        elif "download" in msg.lower() or "gdown" in msg.lower():
            msg = "Failed to download the 3D model. Check your internet connection and try again."
        elif "memory" in msg.lower() or "oom" in msg.lower():
            msg = "Not enough memory to run 3D analysis. Try closing other applications."
        elif "dynamic_network" in msg.lower() or "PlainConvUNet" in msg:
            msg = "Missing required package: dynamic-network-architectures. Run: pip install dynamic-network-architectures"
        return jsonify({"status": "error", "message": msg}), 500


@shared.app.route("/save_report_for_scan", methods=["POST"])
def save_report_for_scan():
    """Save a report for a scan — called only when doctor explicitly clicks Generate Report."""
    doctor = _get_logged_doctor()
    if not doctor:
        return jsonify({"status": "error", "message": "Not logged in"}), 403

    scan_id = request.form.get("scan_id", "").strip()
    if not scan_id:
        return jsonify({"status": "error", "message": "Missing scan_id"}), 400

    snap = shared.db.collection("MRI_Scans").document(scan_id).get()

    # Auto-commit pending scan if not yet saved to Firestore
    if not snap.exists:
        pending = shared._pending_scans.get(scan_id)
        if not pending:
            return jsonify({"status": "error", "message": "Scan not found"}), 404

        now          = shared.now_sa()
        patient_id   = pending["patient_id"]
        case_id      = pending.get("case_id", "")
        upload_dt    = pending["upload_dt"]
        last_mri_str = pending["last_mri_date_str"]
        is_3d_p      = pending.get("is_3d", False)
        conf         = pending.get("confidence") or 0.0
        ttype        = pending.get("tumor_type") or ""

        scan_data = {
            "ScanID":                scan_id,
            "PatientID":             f"/Patients/{patient_id}",
            "CaseID":                f"/Cases/{case_id}",
            "MRIFilePath":           pending.get("mri_path", ""),
            "SegmentationMaskPath":  pending.get("mask_path"),
            "GradCAMPath":           pending.get("gradcam_path"),
            "ClassificationResult":  ttype,
            "ConfidenceScore":       conf,
            "QuickDescription":      f"Detected {ttype} with {conf:.1f}% confidence",
            "UploadDate":            now,
            "UploadDateStr":         shared.fmt_sa_verbose(now),
            "ScanAcquisitionDate":   last_mri_str,
            "FindingsText":          pending.get("findings_text", ""),
            "FindingsBullets":       pending.get("findings_bullets", []),
            "ImpressionBullets":     pending.get("impression_bullets", []),
            "ImpressionText":        pending.get("impression", ""),
            "MaskMetrics":           pending.get("mask_metrics"),
            "LastUpdate":            now,
        }
        if is_3d_p:
            scan_data.update({
                "InputType":     "3d",
                "Is3D":          True,
                "SegModelUsed":  pending.get("seg_model_used", "3d_glioma"),
                "VolumeMetrics": pending.get("volume_metrics"),
                "Slices3D":      pending.get("slices_3d", []),
                "Plot3DPath":    pending.get("plot3d_path", ""),
            })

        shared.db.collection("MRI_Scans").document(scan_id).set(scan_data)
        shared.db.collection("Patients").document(patient_id).update({
            "LastMRIDate":        last_mri_str,
            "LastScanAt":         now,
            "LastScanId":         scan_id,
            "LastScanTumor":      ttype,
            "LastScanConfidence": conf,
            "LastScanUploadStr":  shared.fmt_sa(now),
            "LastActivityAt":     now,
            "LastVisitedAt":      now,
        })
        shared.clear_dash_cache(pending.get("doctor_id"))
        # Remove from pending (it's now committed)
        shared._pending_scans.pop(scan_id, None)
        snap = shared.db.collection("MRI_Scans").document(scan_id).get()

    d = snap.to_dict() or {}

    # Check if a report already exists for this scan
    existing = list(
        shared.db.collection("Reports")
        .where("ScanID", "==", shared.db.document(f"MRI_Scans/{scan_id}"))
        .limit(1).stream()
    )
    if existing:
        return jsonify({"status": "success", "report_id": existing[0].id}), 200

    findings_bullets   = d.get("FindingsBullets")   or []
    impression_bullets = d.get("ImpressionBullets") or []
    findings_text      = d.get("FindingsText")      or ""
    impression_text    = d.get("ImpressionText")    or ""

    # If no findings data stored yet, generate it now
    if not findings_bullets:
        try:
            from utils import call_hf_report_api, build_findings_text
            vm = d.get("VolumeMetrics") or {}
            ft = findings_text or build_findings_text(
                tumor_type=d.get("ClassificationResult"),
                confidence=d.get("ConfidenceScore"),
                mask_metrics=d.get("MaskMetrics"),
                volume_3d=vm if d.get("Is3D") else None,
            )
            report = call_hf_report_api(ft)
            findings_bullets   = report.get("findings_bullets", [])
            impression_bullets = report.get("impression_bullets", [])
            impression_text    = report.get("impression") or " ".join(impression_bullets)
        except Exception as e:
            fallback = _fallback_report(
                findings_text or "",
                d.get("ClassificationResult", ""),
                d.get("VolumeMetrics") or d.get("MaskMetrics") or {}
            )
            findings_bullets   = fallback["findings_bullets"]
            impression_bullets = fallback["impression_bullets"]
            impression_text    = fallback["impression"]

    report_content = {
        "findings_text":      findings_text,
        "findings_bullets":   findings_bullets,
        "impression_bullets": impression_bullets,
        "impression_text":    impression_text,
    }

    is_3d = d.get("Is3D") or d.get("InputType") == "3d"
    vm    = d.get("VolumeMetrics") or {}

    report_id = save_report_document(
        scan_id=scan_id,
        content=report_content,
        created_at=shared.now_sa(),
        gradcam_path=d.get("GradCAMPath") or "",
        mri_file_path=d.get("MRIFilePath") or "",
        pdf_path="",
        segmentation_mask_path=d.get("SegmentationMaskPath") or "",
        is_3d=bool(is_3d),
        volume_metrics=vm,
    )

    return jsonify({"status": "success", "report_id": report_id}), 200


def _cache_mpr_preview(scan_ref, flair_vol, pred_label):
    """Compute 3 middle-slice MPR images and cache as base64 in Firestore."""
    try:
        D, H, W = flair_vol.shape
        preview = {}
        for plane, idx in [("axial", D // 2), ("coronal", H // 2), ("sagittal", W // 2)]:
            b64 = render_mpr_slice_b64(flair_vol, pred_label, plane, idx)
            if b64:
                preview[plane] = b64
        if preview:
            scan_ref.update({"MprPreview": preview})
    except Exception as e:
        print(f"[MPR preview cache] {e}")


@shared.app.route("/api/mpr_preview/<scan_id>")
def api_mpr_preview(scan_id):
    """Return cached MPR preview slices (3 middle slices) from Firestore.
    Falls back to computing from NPZ if Firestore cache is missing."""
    doctor = _get_logged_doctor()
    if not doctor:
        return jsonify({"error": "Not authenticated"}), 401

    snap = shared.db.collection("MRI_Scans").document(scan_id).get()
    if not snap.exists:
        return jsonify({"error": "Scan not found"}), 404

    data    = snap.to_dict() or {}
    preview = data.get("MprPreview", {})

    if not preview:
        # Try to generate from NPZ on disk
        npz_dir  = os.path.join(shared.app.config["UPLOAD_FOLDER"], "seg3d_cache")
        npz_path = os.path.join(npz_dir, f"pred_{scan_id}.npz")
        if os.path.exists(npz_path):
            try:
                npz_data   = np.load(npz_path)
                flair_vol  = npz_data["flair_vol"]
                pred_label = npz_data["pred_label"]
                D, H, W    = flair_vol.shape
                preview    = {}
                for plane, idx in [("axial", D // 2), ("coronal", H // 2), ("sagittal", W // 2)]:
                    b64 = render_mpr_slice_b64(flair_vol, pred_label, plane, idx)
                    if b64:
                        preview[plane] = b64
                if preview:
                    shared.db.collection("MRI_Scans").document(scan_id).update({"MprPreview": preview})
            except Exception as e:
                print(f"[mpr_preview fallback] {e}")

    if not preview:
        return jsonify({"error": "No MPR preview available — run 3D Analysis first"}), 404

    return jsonify({"preview": preview})


# ─── Regenerate 3D overlay images from cached pred_label ─────────────────────
@shared.app.route("/regen_3d_overlays", methods=["POST"])
def regen_3d_overlays():
    """
    Re-generate the coloured overlay slice images for an existing 3D scan
    using the cached pred_label / flair_vol (.npz) file saved during analysis.
    """
    doctor = _get_logged_doctor()
    if not doctor:
        return jsonify({"status": "error", "message": "Not authenticated"}), 401

    scan_id = (request.json or {}).get("scan_id", "")
    if not scan_id:
        return jsonify({"status": "error", "message": "scan_id required"}), 400

    npz_dir  = os.path.join(shared.app.config["UPLOAD_FOLDER"], "seg3d_cache")
    npz_path = os.path.join(npz_dir, f"pred_{scan_id}.npz")

    if not os.path.exists(npz_path):
        return jsonify({"status": "error",
                        "message": "No cached data found for this scan. "
                                   "Please re-run 3D Analysis to get improved images."}), 404

    try:
        data      = np.load(npz_path)
        pred_label = data["pred_label"]
        flair_vol  = data["flair_vol"]

        slices_info = save_3d_results(flair_vol, pred_label, scan_id)

        slices_store = [
            {"overlay": s["overlay"], "slice_idx": int(s["slice_idx"])}
            for s in slices_info
        ]
        primary_overlay = slices_info[len(slices_info) // 2]["overlay"] if slices_info else None

        scan_ref = shared.db.collection("MRI_Scans").document(scan_id)
        scan_ref.update({
            "Slices3D":             slices_store,
            "SegmentationMaskPath": primary_overlay,
            "LastUpdate":           shared.now_sa()
        })

        return jsonify({
            "status":  "success",
            "slices":  slices_info,
            "primary": primary_overlay
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@shared.app.route("/api/mpr_slice/<scan_id>")
def api_mpr_slice(scan_id):
    """
    Return a single rendered MPR slice as a base64 data-URI.
    Query params:
      plane : axial | coronal | sagittal   (default: axial)
      idx   : slice index (int)            (default: middle slice)
    Response JSON: { image, total, idx }
    """
    doctor = _get_logged_doctor()
    if not doctor:
        return jsonify({"error": "Not authenticated"}), 401

    plane = request.args.get("plane", "axial")
    if plane not in ("axial", "coronal", "sagittal"):
        plane = "axial"

    npz_dir  = os.path.join(shared.app.config["UPLOAD_FOLDER"], "seg3d_cache")
    npz_path = os.path.join(npz_dir, f"pred_{scan_id}.npz")

    if not os.path.exists(npz_path):
        return jsonify({"error": "No cached data — please re-run 3D Analysis"}), 404

    try:
        data       = np.load(npz_path)
        flair_vol  = data["flair_vol"]
        pred_label = data["pred_label"]

        D, H, W = flair_vol.shape
        sizes   = {"axial": D, "coronal": H, "sagittal": W}
        total   = sizes[plane]

        raw_idx = request.args.get("idx")
        idx = int(raw_idx) if raw_idx is not None else total // 2
        idx = max(0, min(idx, total - 1))

        b64 = render_mpr_slice_b64(flair_vol, pred_label, plane, idx)
        if b64 is None:
            return jsonify({"error": "Render failed"}), 500

        return jsonify({"image": b64, "total": total, "idx": idx})

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@shared.app.route("/api/nifti_preview", methods=["POST"])
def api_nifti_preview():
    """
    Accept a NIfTI (.nii / .nii.gz) or ZIP file, extract the middle
    Axial / Coronal / Sagittal slices (with CLAHE) and return all three
    as base64 JPEGs for the instant upload preview.
    """
    import base64, zipfile, tempfile
    import nibabel as nib
    import numpy as np
    import cv2

    doctor = _get_logged_doctor()
    if not doctor:
        return jsonify({"error": "Not authenticated"}), 401

    f = request.files.get("file")
    if not f:
        return jsonify({"error": "No file"}), 400

    name = f.filename.lower()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            raw_path = os.path.join(tmp, f.filename)
            f.save(raw_path)

            # Resolve the NIfTI file path
            nii_path = None
            if name.endswith(".zip"):
                with zipfile.ZipFile(raw_path) as zf:
                    for m in zf.namelist():
                        if m.endswith(".nii") or m.endswith(".nii.gz"):
                            zf.extract(m, tmp)
                            nii_path = os.path.join(tmp, m)
                            break
            else:
                nii_path = raw_path

            if not nii_path or not os.path.exists(nii_path):
                return jsonify({"error": "No NIfTI found in upload"}), 400

            img  = nib.load(nii_path)
            data = np.asarray(img.dataobj, dtype=np.float32)

            # Handle 4-D (pick first volume), then orient as (D,H,W)
            if data.ndim == 4:
                data = data[..., 0]
            if data.ndim == 3:
                data = np.transpose(data, (2, 0, 1))   # (D, H, W)

            D, H, W = data.shape
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

            def _encode_slice(sl_raw, flip_lr=False):
                sl = np.flipud(sl_raw)
                if flip_lr:
                    sl = np.fliplr(sl)
                mn, mx = sl.min(), sl.max()
                if mx > mn:
                    sl_u8 = ((sl - mn) / (mx - mn) * 255).astype(np.uint8)
                else:
                    sl_u8 = np.zeros_like(sl, dtype=np.uint8)
                sl_u8 = clahe.apply(sl_u8)
                ok, buf = cv2.imencode(".jpg", sl_u8,
                                       [cv2.IMWRITE_JPEG_QUALITY, 88])
                if not ok:
                    return None
                return "data:image/jpeg;base64," + \
                       base64.b64encode(buf.tobytes()).decode("ascii")

            axial    = _encode_slice(data[D // 2, :, :])
            coronal  = _encode_slice(data[:, H // 2, :])
            sagittal = _encode_slice(data[:, :, W // 2], flip_lr=True)

            return jsonify({
                "axial":    axial,
                "coronal":  coronal,
                "sagittal": sagittal,
                "depth": D, "height": H, "width": W
            })

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@shared.app.route("/mpr_thumbnail/<scan_id>")
def mpr_thumbnail(scan_id):
    """
    Return a red-only tumor overlay thumbnail for a 3D scan.
    Priority:
      1. MprPreview.axial from Firestore  (already red, best quality)
      2. overlay3d_SCANID_48.png on disk  (multi-color → recolored to red on-the-fly)
    """
    import base64, re, io
    import numpy as np
    import cv2
    from flask import Response

    snap = shared.db.collection("MRI_Scans").document(scan_id).get()
    if not snap.exists:
        return "", 404

    data    = snap.to_dict() or {}
    preview = data.get("MprPreview", {})
    b64_uri = preview.get("axial", "")

    # ── Path 1: stored MPR preview (already red) ────────────────────────────
    if b64_uri:
        m = re.match(r"data:image/(\w+);base64,(.+)", b64_uri, re.DOTALL)
        if m:
            raw = base64.b64decode(m.group(2))
            return Response(raw, mimetype=f"image/{m.group(1)}",
                            headers={"Cache-Control": "public, max-age=86400"})

    # ── Path 2: generate from .npz cache using render_mpr_slice_b64 ────────
    npz_path   = os.path.join(shared.app.root_path, "static", "uploads",
                              "seg3d_cache", f"pred_{scan_id}.npz")
    thumb_dir  = os.path.join(shared.app.root_path, "static", "uploads", "thumbnails")
    thumb_path = os.path.join(thumb_dir, f"thumb_{scan_id}.png")

    if os.path.exists(npz_path):
        try:
            import cv2 as _cv2
            npz        = np.load(npz_path, allow_pickle=False)
            pred_label = npz["pred_label"]
            flair_vol  = npz["flair_vol"]
            D          = flair_vol.shape[0]

            tumor_per_slice = [(pred_label[z] > 0).sum() for z in range(D)]
            best_z = int(np.argmax(tumor_per_slice)) if any(tumor_per_slice) else D // 2

            b64_uri = render_mpr_slice_b64(flair_vol, pred_label, "axial", best_z)
            if b64_uri:
                m = re.match(r"data:image/(\w+);base64,(.+)", b64_uri, re.DOTALL)
                if m:
                    raw = base64.b64decode(m.group(2))
                    # Save to disk so patients.py serves it as a static file next time
                    try:
                        os.makedirs(thumb_dir, exist_ok=True)
                        arr = np.frombuffer(raw, dtype=np.uint8)
                        img = _cv2.imdecode(arr, _cv2.IMREAD_COLOR)
                        _cv2.imwrite(thumb_path, img)
                    except Exception:
                        pass
                    return Response(raw, mimetype=f"image/{m.group(1)}",
                                    headers={"Cache-Control": "no-cache, no-store, must-revalidate"})
        except Exception as npz_err:
            print(f"[mpr_thumbnail] npz load failed for {scan_id}: {npz_err}")

    # ── Path 3 (last resort): recolor overlay3d PNG to red ─────────────────
    overlay_fname = f"overlay3d_{scan_id}_48.png"
    overlay_abs   = os.path.join(shared.app.root_path,
                                 "static", "uploads", "overlays_3d", overlay_fname)
    if not os.path.exists(overlay_abs):
        return "", 404

    bgr = cv2.imread(overlay_abs)
    if bgr is None:
        return "", 500

    # Blank out the burned-in legend (bottom-right corner)
    bgr[-80:, -95:] = 0

    # Detect colored (non-gray) tumor pixels via HSV saturation
    hsv        = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    tumor_mask = hsv[:, :, 1] > 40
    tumor_mask[-80:, -95:] = False

    gray  = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    rgb   = np.stack([gray, gray, gray], axis=-1).astype(np.float32)
    alpha = 0.55
    TUMOR_R, TUMOR_G, TUMOR_B = 220, 50, 50
    rgb[tumor_mask, 0] = (1 - alpha) * rgb[tumor_mask, 0] + alpha * TUMOR_R
    rgb[tumor_mask, 1] = (1 - alpha) * rgb[tumor_mask, 1] + alpha * TUMOR_G
    rgb[tumor_mask, 2] = (1 - alpha) * rgb[tumor_mask, 2] + alpha * TUMOR_B
    result = np.clip(rgb, 0, 255).astype(np.uint8)

    binary = tumor_mask.astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (255, 210, 210), 2)
    cv2.drawContours(result, contours, -1, (220,  50,  50), 1)

    out_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", out_bgr)
    if not ok:
        return "", 500

    return Response(buf.tobytes(), mimetype="image/png",
                    headers={"Cache-Control": "public, max-age=86400"})


@shared.app.route("/update_report_content", methods=["POST"])
def update_report_content():
    doctor = _get_logged_doctor()
    if not doctor:
        return jsonify({"error": "Not authenticated"}), 401

    data = request.get_json(silent=True) or {}
    report_id = data.get("report_id", "").strip()
    if not report_id:
        return jsonify({"error": "report_id required"}), 400

    try:
        report_ref = shared.db.collection("Reports").document(report_id)
        report_doc = report_ref.get()
        if not report_doc.exists:
            return jsonify({"error": "Report not found"}), 404

        update_payload = {
            "DoctorName":    data.get("doctor_name",    ""),
            "DoctorEmail":   data.get("doctor_email",   ""),
            "PatientName":   data.get("patient_name",   ""),
            "PatientPhone":  data.get("patient_phone",  ""),
            "PatientAge":    data.get("patient_age",    ""),
            "PatientGender": data.get("patient_gender", ""),
        }
        findings   = data.get("findings",   [])
        impression = data.get("impression", [])
        if isinstance(findings,   list): update_payload["findings_bullets"]   = findings
        if isinstance(impression, list): update_payload["impression_bullets"] = impression

        report_ref.update(update_payload)
        return jsonify({"ok": True})

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500
