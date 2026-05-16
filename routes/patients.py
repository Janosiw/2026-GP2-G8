import os
import re
import secrets
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, date
from flask import render_template, redirect, url_for, request, jsonify, flash, session
from firebase_admin import firestore, storage

import shared
from utils import (
    _get_logged_doctor, compute_age_from_dob, _mask_identifier,
    _serialize_patient_for_search, _filter_and_suggest_patients,
    _apply_patient_filters, _normalize_tumor_type, _parse_yyyy_mm_dd,
    _batch_case_status, send_invite_email, send_join_email, send_comment_notification
)


def _safe_int(v):
    try:
        return int(str(v).strip())
    except Exception:
        return None


def _find_similar_cases(doctor_id, current_patient_id, current_tumor_type,
                        current_patient_age, current_patient_gender, current_first_scan_area):
    """Return up to 5 similar RECOVERED cases from the same doctor (different patient)."""
    cur_tumor = (current_tumor_type or "").strip().lower()
    if not cur_tumor or cur_tumor in {"notumor", "no tumor", "none", "unknown", ""}:
        return []

    patients_docs = list(
        shared.db.collection("Patients")
        .where("CreatedBy", "==", f"/Radiologists/{doctor_id}")
        .stream()
    )

    cur_age_int = _safe_int(current_patient_age)
    cur_gender  = (current_patient_gender or "").strip().lower()

    results = []

    for pdoc in patients_docs:
        if pdoc.id == current_patient_id:
            continue

        pd = pdoc.to_dict() or {}
        comp_age    = compute_age_from_dob(pd.get("DateOfBirth", ""))
        comp_gender = (pd.get("Gender") or "").strip()
        comp_age_int = _safe_int(comp_age)

        recovered_cases = list(
            shared.db.collection("Cases")
            .where("PatientID", "==", f"/Patients/{pdoc.id}")
            .where("Status", "==", "Recovered")
            .stream()
        )

        for cdoc in recovered_cases:
            cd = cdoc.to_dict() or {}

            scans = list(
                shared.db.collection("MRI_Scans")
                .where("CaseID", "==", f"/Cases/{cdoc.id}")
                .stream()
            )

            scan_list = []
            for sdoc in scans:
                sd = sdoc.to_dict() or {}
                mm = sd.get("MaskMetrics") or {}
                dt = sd.get("UploadDate")
                is_dt = isinstance(dt, datetime)
                scan_list.append({
                    "tumor_type":  sd.get("ClassificationResult") or "—",
                    "confidence":  sd.get("ConfidenceScore") or 0,
                    "area":        mm.get("area_pixels"),
                    "date":        dt.strftime("%Y-%m-%d") if is_dt else "—",
                    "_dt":         dt if is_dt else None,
                    "path":        sd.get("MRIFilePath") or "",
                })
            # Sort ascending by real datetime; scans without a date go last
            _epoch = datetime(1970, 1, 1)
            def _scan_sort_key(x):
                if x["_dt"] is None:
                    return (1, _epoch)
                try:
                    return (0, x["_dt"].replace(tzinfo=None))
                except Exception:
                    return (0, x["_dt"])
            scan_list.sort(key=_scan_sort_key)
            for s in scan_list:
                del s["_dt"]

            # Determine tumor type from Case Diagnosis or first scan
            case_tumor = (cd.get("Diagnosis") or "").strip().lower()
            if not case_tumor and scan_list:
                case_tumor = (scan_list[0]["tumor_type"] or "").strip().lower()

            if case_tumor != cur_tumor:
                continue

            # Use only the first scan (baseline) for size comparison
            comp_first_area = scan_list[0]["area"] if scan_list else None

            # Build match criteria & score
            # Priority order: tumor type (required) > tumor size baseline (+3) > gender (+1) > age (+1)
            match_criteria = [{
                "label": "Tumor type",
                "value": case_tumor.title(),
                "matched": True,
            }]
            score = 2  # base for required tumor type match

            if current_first_scan_area is not None and comp_first_area is not None:
                # Percentage-based threshold: within ±25% of the current baseline area
                pct_diff = abs(current_first_scan_area - comp_first_area) / current_first_scan_area
                matched  = pct_diff <= 0.25
                match_criteria.append({
                    "label": "Tumor size (baseline)",
                    "value": f"{comp_first_area} px — ±25%",
                    "matched": matched,
                })
                if matched:
                    score += 3  # highest secondary weight

            if cur_gender and comp_gender:
                matched = cur_gender == comp_gender.lower()
                match_criteria.append({
                    "label": "Gender",
                    "value": comp_gender,
                    "matched": matched,
                })
                if matched:
                    score += 1

            if cur_age_int is not None and comp_age_int is not None:
                age_diff = abs(cur_age_int - comp_age_int)
                matched  = age_diff <= 5
                match_criteria.append({
                    "label": "Age",
                    "value": f"{comp_age_int} yrs — ±5 yrs",
                    "matched": matched,
                })
                if matched:
                    score += 1

            # Require: tumor type (always) + tumor size match — gender/age are bonus only
            size_criterion = next((c for c in match_criteria if c["label"] == "Tumor size (baseline)"), None)
            size_matched = size_criterion is not None and size_criterion["matched"]
            if not size_matched:
                continue  # tumor size match is required

            results.append({
                "patient_name":    pd.get("FullName") or "Unknown",
                "age":             comp_age or "—",
                "gender":          comp_gender or "—",
                "diagnosis":       case_tumor.title(),
                "treatment_plan":  (cd.get("TreatmentPlan") or "").strip() or "—",
                "start_date":      cd.get("StartDate") or "—",
                "end_date":        cd.get("EndDate") or "—",
                "scan_count":      len(scan_list),
                "scans":           scan_list,
                "match_criteria":  match_criteria,
                "_score":          score,
            })

    results.sort(key=lambda x: x["_score"], reverse=True)
    for r in results:
        r.pop("_score", None)
    return results[:5]


@shared.app.route("/patients", methods=["GET", "POST"])
def patients():
    doctor = _get_logged_doctor()
    if not doctor:
        return redirect(url_for("register_login"))

    if request.method == "POST":
        full_name = request.form.get("FullName", "").strip()
        dob = request.form.get("DateOfBirth", "").strip()
        gender = request.form.get("Gender", "").strip()
        tumor_type = _normalize_tumor_type(request.form.get("TumorType", ""))
        medical_notes = request.form.get("MedicalNotes", "").strip()
        contact_number = request.form.get("ContactNumber", "").strip()
        if contact_number and not contact_number.startswith("+966"):
            contact_number = "+966" + contact_number

        if full_name and dob and gender:
            now = shared.now_sa()
            new_patient = {
                "FullName": full_name,
                "ContactNumber": contact_number,
                "DateOfBirth": dob,
                "Gender": gender,
                "TumorType": tumor_type or "",
                "MedicalNotes": medical_notes,
                "CreatedBy": f"/Radiologists/{doctor['id']}",
                "CreatedAt": now,
                "LastActivityAt": now,
                "LastVisitedAt": shared.EPOCH,
                "LastScanAt": shared.EPOCH,
                "LastScanId": "",
                "LastScanTumor": "",
                "LastScanConfidence": None,
                "LastScanUploadStr": ""
            }
            shared.db.collection("Patients").add(new_patient)
            flash("Patient added successfully.", "success")
            return redirect(url_for("patients"))
        else:
            return redirect(url_for("patients"))

    q_raw = request.args.get("q", "").strip()
    tumor = _normalize_tumor_type(request.args.get("tumor", ""))
    date_from = request.args.get("from", "").strip()
    date_to = request.args.get("to", "").strip()
    case_status = request.args.get("case_status", "").strip()
    no_scan = request.args.get("no_scan", "") == "1"

    patients_ref = shared.db.collection("Patients").where("CreatedBy", "==", f"/Radiologists/{doctor['id']}")
    all_patients = []
    for p in patients_ref.stream():
        all_patients.append(_serialize_patient_for_search(p))

    case_status_map = _batch_case_status([p["id"] for p in all_patients])
    for p in all_patients:
        p["CaseStatus"] = case_status_map.get(p["id"], "")

    filtered_patients = _apply_patient_filters(
        all_patients, tumor=tumor, date_from=date_from, date_to=date_to, case_status=case_status, no_scan=no_scan
    )
    patients_list, suggestions = _filter_and_suggest_patients(q_raw, filtered_patients)
    has_filters = bool(tumor or date_from or date_to or case_status or no_scan)

    return render_template(
        "patients.html",
        doctor=doctor,
        patients=patients_list,
        suggestions=suggestions,
        has_filters=has_filters,
        tumor_types=shared.TUMOR_TYPE_OPTIONS,
        q=q_raw,
        tumor=tumor,
        date_from=date_from,
        date_to=date_to,
        case_status=case_status,
        no_scan=no_scan
    )


@shared.app.route("/patients/search_api", methods=["GET"])
def patients_search_api():
    doctor = _get_logged_doctor()
    if not doctor:
        return jsonify({"status": "error", "message": "Not logged in"}), 403

    q_raw = request.args.get("q", "").strip()
    tumor = _normalize_tumor_type(request.args.get("tumor", ""))
    date_from = request.args.get("from", "").strip()
    date_to = request.args.get("to", "").strip()
    case_status = request.args.get("case_status", "").strip()
    patients_ref = shared.db.collection("Patients").where("CreatedBy", "==", f"/Radiologists/{doctor['id']}")

    all_patients = [_serialize_patient_for_search(p) for p in patients_ref.stream()]

    if case_status:
        case_status_map2 = _batch_case_status([p["id"] for p in all_patients])
        for p in all_patients:
            p["CaseStatus"] = case_status_map2.get(p["id"], "")

    filtered_patients = _apply_patient_filters(
        all_patients, tumor=tumor, date_from=date_from, date_to=date_to, case_status=case_status
    )
    matches, suggestions = _filter_and_suggest_patients(q_raw, filtered_patients)

    return jsonify({
        "status": "success",
        "query": q_raw,
        "filters": {"tumor": tumor, "from": date_from, "to": date_to, "case_status": case_status},
        "patients": matches,
        "suggestions": suggestions
    })


@shared.app.route("/add_patient", methods=["POST"])
def add_patient():
    if "radiologist_id" not in session:
        return jsonify({"status": "error", "message": "Not logged in"})

    data = request.json
    name = data.get("FullName")
    dob = data.get("DateOfBirth")
    gender = data.get("Gender")
    notes = data.get("MedicalNotes", "")

    if not all([name, dob, gender]):
        return jsonify({"status": "error", "message": "Missing fields"})

    rid = session["radiologist_id"]
    now = shared.now_sa()
    new_doc = shared.db.collection("Patients").document()
    new_doc.set({
        "FullName": name,
        "DateOfBirth": dob,
        "Gender": gender,
        "MedicalNotes": notes,
        "CreatedBy": f"/Radiologists/{rid}",
        "CreatedAt": now,
        "LastActivityAt": now,
        "LastVisitedAt": shared.EPOCH,
        "LastScanAt": shared.EPOCH,
        "LastScanId": "",
        "LastScanTumor": "",
        "LastScanConfidence": None,
        "LastScanUploadStr": ""
    })
    shared.clear_dash_cache(rid)

    return jsonify({"status": "success", "message": "Patient added successfully"})


@shared.app.route("/patients/<patient_id>/profile", methods=["GET", "POST"], endpoint="patient_profile")
def patient_profile(patient_id):
    doctor = _get_logged_doctor()
    if not doctor:
        return redirect(url_for("register_login"))

    p_ref = shared.db.collection("Patients").document(patient_id)
    snap = p_ref.get()
    if not snap.exists:
        return "Patient not found", 404

    p = snap.to_dict() or {}
    if p.get("CreatedBy") != f"/Radiologists/{doctor['id']}":
        return "Unauthorized", 403

    now = shared.now_sa()
    # Fire-and-forget: don't block page load waiting for this write
    def _update_visit():
        try:
            p_ref.update({"LastVisitedAt": now, "LastActivityAt": now})
        except Exception:
            pass
    import threading
    threading.Thread(target=_update_visit, daemon=True).start()

    if request.method == "POST":
        full_name = request.form.get("name", "").strip()
        dob = request.form.get("dob", "").strip()
        gender = request.form.get("gender", "").strip()
        phone = request.form.get("phone", "").strip()
        notes = request.form.get("notes", "").strip()

        updated = {}
        if full_name:
            updated["FullName"] = full_name
        if dob:
            updated["DateOfBirth"] = dob
        if gender:
            updated["Gender"] = gender
        if phone:
            if not phone.startswith("+966"):
                phone = "+966" + phone
            updated["ContactNumber"] = phone
        updated["MedicalNotes"] = notes

        file = request.files.get("profile_pic")
        if file and file.filename.strip():
            ext = os.path.splitext(file.filename)[1] or ".jpg"
            folder = os.path.join(shared.app.config["UPLOAD_FOLDER"], "patients")
            os.makedirs(folder, exist_ok=True)
            filename = f"{patient_id}{ext}"
            path = os.path.join(folder, filename)
            file.save(path)
            updated["ProfilePicture"] = f"/static/uploads/patients/{filename}"
        else:
            updated["ProfilePicture"] = p.get("ProfilePicture", "/static/images/user.png")

        if updated:
            p_ref.update(updated)

        return redirect(url_for("patient_profile", patient_id=patient_id))

    patient_ctx = {
        "patient_id": patient_id,
        "masked_id": _mask_identifier(patient_id),
        "name": p.get("FullName", ""),
        "dob": p.get("DateOfBirth", ""),
        "age": compute_age_from_dob(p.get("DateOfBirth")),
        "gender": p.get("Gender", ""),
        "phone": p.get("ContactNumber", ""),
        "ProfilePicture": p.get("ProfilePicture", "/static/images/user.png"),
        "MedicalNotes": p.get("MedicalNotes", ""),
        "LastScanDate": p.get("LastMRIDate", ""),
    }

    # ── Fetch cases and ALL patient scans in parallel (2 queries instead of N+1) ──
    def _fetch_cases():
        return list(
            shared.db.collection("Cases")
            .where("PatientID", "==", f"/Patients/{patient_id}")
            .order_by("CreatedAt", direction=firestore.Query.ASCENDING)
            .stream()
        )

    def _fetch_scans():
        return list(
            shared.db.collection("MRI_Scans")
            .where("PatientID", "==", f"/Patients/{patient_id}")
            .order_by("UploadDate", direction=firestore.Query.ASCENDING)
            .stream()
        )

    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_cases = ex.submit(_fetch_cases)
        fut_scans = ex.submit(_fetch_scans)
        cases_docs  = fut_cases.result()
        scans_docs  = fut_scans.result()

    # Group scan dates by case_id for last_update computation (no extra queries)
    case_scan_dates: dict = {}
    scans = []
    for s in scans_docs:
        sd = s.to_dict() or {}
        dt = sd.get("UploadDate")
        dt_clean = shared.fmt_sa(dt) if isinstance(dt, datetime) else "—"
        _is_3d = sd.get("InputType", "") == "3d"
        _seg   = sd.get("SegmentationMaskPath") or ""
        _mri   = sd.get("MRIFilePath", "")
        _thumb = _seg if (_is_3d and _seg) else _mri
        cid_key = sd.get("CaseID", "")
        scans.append({
            "id": s.id,
            "path": _thumb,
            "date": dt_clean,
            "case_id": cid_key
        })
        if isinstance(dt, datetime) and cid_key:
            case_scan_dates.setdefault(cid_key, []).append(dt)

    cases = []
    for c in cases_docs:
        cd = c.to_dict() or {}
        cid_key = f"/Cases/{c.id}"
        dates_for_case = case_scan_dates.get(cid_key, [])
        if dates_for_case:
            last_update_clean = shared.fmt_sa(max(dates_for_case))
        else:
            raw = cd.get("LastUpdate")
            last_update_clean = shared.fmt_sa(raw) if isinstance(raw, datetime) else "—"

        cases.append({
            "id": c.id,
            "diagnosis": cd.get("Diagnosis", "—"),
            "treatment_plan": cd.get("TreatmentPlan", "—"),
            "status": cd.get("Status", "—"),
            "start_date": cd.get("StartDate", "—"),
            "end_date": cd.get("EndDate", None),
            "last_update": last_update_clean
        })

    cases_sorted = sorted(cases, key=lambda x: x["start_date"] if x["start_date"] != "—" else "")
    for idx, c in enumerate(cases_sorted, start=1):
        c["display_id"] = idx

    has_scans = len(scans) > 0

    return render_template(
        "patient_profile.html",
        doctor=doctor,
        patient=patient_ctx,
        cases=cases_sorted,
        scans=scans,
        has_scans=has_scans,
        current_date_iso=date.today().isoformat()
    )


@shared.app.route("/patients/<patient_id>/update", methods=["POST"])
def update_patient(patient_id):
    return redirect(url_for("patient_profile", patient_id=patient_id))


@shared.app.route("/patients/<patient_id>/create_case", methods=["POST"])
def create_case(patient_id):
    doctor = _get_logged_doctor()
    if not doctor:
        return redirect(url_for("register_login"))

    patient_ref = shared.db.collection("Patients").document(patient_id)
    snap = patient_ref.get()
    if not snap.exists:
        return "Patient not found", 404

    treatment_plan = request.form.get("treatment_plan", "").strip()
    selected_scan_date = request.form.get("scan_date", "").strip()
    parsed_scan_date = _parse_yyyy_mm_dd(selected_scan_date)
    case_start_date = parsed_scan_date.strftime("%Y-%m-%d") if parsed_scan_date else shared.now_sa().strftime("%Y-%m-%d")

    first_scan = request.files.get("mri_file")
    if not first_scan:
        return "Missing MRI scan", 400

    now = shared.now_sa()
    case_ref = shared.db.collection("Cases").document()
    case_id = case_ref.id

    case_ref.set({
        "PatientID": f"/Patients/{patient_id}",
        "Diagnosis": "",
        "TreatmentPlan": treatment_plan,
        "Status": "Active",
        "StartDate": case_start_date,
        "EndDate": None,
        "Notes": "",
        "CreatedAt": now,
        "LastUpdate": now,
        "FirstScanID": None
    })
    shared.clear_dash_cache(doctor["id"])

    filename = f"{case_id}_{first_scan.filename}"
    save_path = os.path.join(shared.app.config["UPLOAD_FOLDER"], filename)
    first_scan.save(save_path)

    rel_path = "/" + save_path.replace("\\", "/")

    return redirect(url_for(
        "scans",
        patient_id=patient_id,
        case_id=case_id,
        scan_date=case_start_date,
        first_image=rel_path
    ))


@shared.app.route("/patients/<patient_id>/cases/<case_id>")
def view_case(patient_id, case_id):
    doctor = _get_logged_doctor()
    if not doctor:
        return redirect(url_for("register_login"))

    case_doc = shared.db.collection("Cases").document(case_id).get()
    if not case_doc.exists:
        return "Case not found", 404

    case = case_doc.to_dict() or {}

    # Determine if current doctor is invited (has InvitedDoctors entry)
    invited_doctors = case.get("InvitedDoctors", [])
    is_invited = doctor["id"] in invited_doctors

    all_cases = list(
        shared.db.collection("Cases")
        .where("PatientID", "==", f"/Patients/{patient_id}")
        .order_by("CreatedAt", direction=firestore.Query.ASCENDING)
        .stream()
    )

    sorted_cases = sorted(
        all_cases,
        key=lambda x: (x.to_dict().get("StartDate") or "")
    )

    display_number = 1
    for idx, cc in enumerate(sorted_cases, start=1):
        if cc.id == case_id:
            display_number = idx
            break

    case["DisplayID"] = display_number

    scans = list(
        shared.db.collection("MRI_Scans")
        .where("CaseID", "==", f"/Cases/{case_id}")
        .stream()
    )

    scans_list = []
    for s in scans:
        d = s.to_dict()
        dt = d.get("UploadDate")
        clean_date = shared.fmt_sa(dt) if isinstance(dt, datetime) else "—"

        mask_metrics   = d.get("MaskMetrics")   or {}
        volume_metrics = d.get("VolumeMetrics") or {}
        area_pixels    = mask_metrics.get("area_pixels")
        volume_cm3     = volume_metrics.get("total_volume_cm3")
        is_3d          = d.get("InputType", "") == "3d"

        # Thumbnail: for 3D use overlay slice; for 2D use mask if available, else original
        import os as _os
        mri_path = d.get("MRIFilePath", "")
        seg_mask = d.get("SegmentationMaskPath") or ""
        if is_3d:
            # Prefer pre-generated static thumbnail (no cache issues)
            _thumb_rel  = _os.path.join("static", "uploads", "thumbnails", f"thumb_{s.id}.png")
            _thumb_abs  = _os.path.join(shared.app.root_path, _thumb_rel)
            if _os.path.exists(_thumb_abs):
                thumbnail = f"/static/uploads/thumbnails/thumb_{s.id}.png"
            else:
                # Dynamic endpoint generates + saves thumbnail on first request
                thumbnail = f"/mpr_thumbnail/{s.id}"
        else:
            thumbnail = seg_mask if seg_mask else mri_path

        # Tumor label for the thumbnail badge (one label only)
        _cls = (d.get("ClassificationResult") or "").strip()
        if is_3d:
            tumor_label = "Glioma"
            tumor_color = "#1d4ed8"
        elif _cls.lower() in ("", "none", "notumor", "no_tumor", "no tumor"):
            tumor_label = ""
            tumor_color = ""
        else:
            _color_map = {
                "meningioma": "#7c3aed",
                "glioma":     "#dc2626",
                "pituitary":  "#d97706",
            }
            tumor_label = _cls.title()
            tumor_color = _color_map.get(_cls.lower(), "#374151")

        scans_list.append({
            "id": s.id,
            "MRIFilePath": thumbnail,
            "UploadDate": clean_date,
            "ClassificationResult": _cls,
            "tumor_label": tumor_label,
            "tumor_color": tumor_color,
            "area_pixels": area_pixels,
            "volume_cm3": volume_cm3,
            "is_3d": is_3d,
            "size_change": None,
            "size_trend": None,
            "is_healthy": False,
            "is_baseline": False,
        })

    scans_list.sort(key=lambda x: x["UploadDate"])

    if scans_list:
        case["LastUpdateFormatted"] = scans_list[-1]["UploadDate"]
    else:
        case["LastUpdateFormatted"] = "—"

    NO_TUMOR_TOKENS = {"notumor", "none", "unknown", ""}

    # Flag scans with no detected tumor
    for scan in scans_list:
        cr = scan.get("ClassificationResult", "") or ""
        normalized = cr.strip().lower().replace(" ", "").replace("_", "")
        if normalized in NO_TUMOR_TOKENS:
            scan["is_healthy"] = True

    # ── Baseline determination ──────────────────────────────────────────────────
    # Prefer 3D volume_cm3, fall back to 2D area_pixels.
    # Baseline = first scan (by date) with a valid measurement.
    def _metric(scan):
        """Return (value, type) where type is '3d' or '2d'."""
        if scan.get("is_3d") and scan.get("volume_cm3") is not None:
            return scan["volume_cm3"], "3d"
        if scan.get("area_pixels") is not None:
            return scan["area_pixels"], "2d"
        return None, None

    # Two separate baselines — first valid 2D, first valid 3D
    baseline_2d = next(
        (s for s in scans_list if not s.get("is_3d") and s.get("area_pixels") is not None and s["area_pixels"] > 0),
        None
    )
    baseline_3d = next(
        (s for s in scans_list if s.get("is_3d") and s.get("volume_cm3") is not None and s["volume_cm3"] > 0),
        None
    )

    # Mark baselines — each type gets its own "Baseline" card
    baseline_scan = baseline_2d or baseline_3d or (scans_list[0] if scans_list else None)
    if baseline_scan:
        baseline_scan["is_baseline"] = True
    # If 3D baseline is a different scan from the main baseline, mark it too
    if baseline_3d and baseline_3d is not baseline_scan:
        baseline_3d["is_baseline"] = True

    # Each scan is shown as % of its own-type baseline
    STABLE_THRESHOLD = 5.0

    for scan in scans_list:
        if scan.get("is_baseline"):
            continue
        curr_val, curr_type = _metric(scan)
        if curr_val is None or curr_val == 0:
            continue

        # Pick the matching baseline by type
        if curr_type == "3d" and baseline_3d and baseline_3d.get("volume_cm3"):
            ref_val = baseline_3d["volume_cm3"]
        elif curr_type == "2d" and baseline_2d and baseline_2d.get("area_pixels"):
            ref_val = baseline_2d["area_pixels"]
        else:
            continue

        if ref_val and ref_val > 0:
            pct_of_baseline = round((curr_val / ref_val) * 100, 1)
            scan["size_change"] = pct_of_baseline
            if pct_of_baseline < (100 - STABLE_THRESHOLD):
                scan["size_trend"] = "improved"
            elif pct_of_baseline > (100 + STABLE_THRESHOLD):
                scan["size_trend"] = "worsened"
            else:
                scan["size_trend"] = "stable"

    first_scan_diagnosis = None
    if scans_list:
        first_scan_diagnosis = scans_list[0].get("ClassificationResult", "").strip()

    if not case.get("Diagnosis") or case["Diagnosis"].strip() == "":
        if first_scan_diagnosis:
            case["Diagnosis"] = first_scan_diagnosis
            shared.db.collection("Cases").document(case_id).update({"Diagnosis": first_scan_diagnosis})
        else:
            case["Diagnosis"] = "Pending Diagnosis"

    p_doc = shared.db.collection("Patients").document(patient_id).get()
    patient_name = ""
    patient_phone = ""
    patient_age = ""
    patient_gender = ""
    is_owner = False
    if p_doc.exists:
        pd = p_doc.to_dict()
        patient_name = pd.get("FullName", "")
        patient_phone = pd.get("PhoneNumber", pd.get("Phone", ""))
        patient_gender = pd.get("Gender", "")
        patient_age = compute_age_from_dob(pd.get("DateOfBirth", ""))
        owner_ref = pd.get("CreatedBy", "")
        is_owner = (owner_ref == f"/Radiologists/{doctor['id']}")

    # DEMO: ?_demo_role=invited forces the invited-consultant view
    if request.args.get("_demo_role") == "invited":
        is_invited = True
        is_owner   = False

    if not is_owner and not is_invited:
        return "Unauthorized: You do not have access to this case.", 403

    for scan in scans_list:
        scan_ref = shared.db.document(f"MRI_Scans/{scan['id']}")
        report_docs = list(
            shared.db.collection("Reports").where("ScanID", "==", scan_ref).stream()
        )
        if report_docs:
            rdoc = report_docs[0]
            rdata = rdoc.to_dict() or {}
            content = rdata.get("Content") or {}
            scan["report_id"] = rdoc.id
            scan["report_findings"] = content.get("findings_bullets") or []
            scan["report_impression"] = content.get("impression_bullets") or []
            scan["report_mask"] = rdata.get("SegmentationMaskPath") or ""
            scan["report_mri"] = rdata.get("MRIFilePath") or scan.get("MRIFilePath") or ""
            created_at = rdata.get("CreatedAt")
            scan["report_date"] = created_at.strftime("%d %b %Y") if isinstance(created_at, datetime) else ""
        else:
            scan["report_id"] = ""

    return render_template(
        "view_case.html",
        doctor=doctor,
        case_id=case_id,
        case_number=display_number,
        patient_id=patient_id,
        patient_name=patient_name,
        patient_phone=patient_phone,
        patient_age=patient_age,
        patient_gender=patient_gender,
        case=case,
        scans=scans_list,
        is_owner=is_owner,
        is_invited=is_invited,
    )


@shared.app.route("/api/shared_cases")
def api_shared_cases():
    doctor = _get_logged_doctor()
    if not doctor:
        return jsonify({"status": "error"}), 401

    doctor_id  = doctor["id"]
    doctor_ref = f"/Radiologists/{doctor_id}"
    patient_cache = {}
    seen_case_ids = set()
    result = []

    def _patient_data(pid):
        if pid not in patient_cache:
            pdoc = shared.db.collection("Patients").document(pid).get()
            patient_cache[pid] = (pdoc.to_dict() or {}) if pdoc.exists else {}
        return patient_cache[pid]

    def _append(c, data, role):
        pid = (data.get("PatientID", "") or "").split("/")[-1]
        pdata = _patient_data(pid)
        start = data.get("StartDate")
        result.append({
            "case_id":      c.id,
            "patient_id":   pid,
            "patient_name": pdata.get("FullName", "Unknown"),
            "diagnosis":    data.get("Diagnosis") or data.get("TumorType") or "Pending",
            "status":       data.get("Status", "Active"),
            "start_date":   start.strftime("%d %b %Y") if hasattr(start, "strftime") else (start or ""),
            "role":         role,
        })

    # ── 1. Cases shared WITH me (efficient array_contains query) ──────────
    for c in shared.db.collection("Cases").where(
            "InvitedDoctors", "array_contains", doctor_id).stream():
        seen_case_ids.add(c.id)
        _append(c, c.to_dict() or {}, "invited")

    # ── 2. Cases I shared with others (query my patients, then their cases) ──
    my_patients = list(
        shared.db.collection("Patients")
        .where("CreatedBy", "==", doctor_ref).stream()
    )
    my_patient_refs = [f"/Patients/{p.id}" for p in my_patients]

    BATCH = 30
    for i in range(0, len(my_patient_refs), BATCH):
        batch_refs = my_patient_refs[i:i + BATCH]
        for c in shared.db.collection("Cases").where(
                "PatientID", "in", batch_refs).stream():
            if c.id in seen_case_ids:
                continue
            data = c.to_dict() or {}
            if data.get("InvitedDoctors"):
                seen_case_ids.add(c.id)
                _append(c, data, "shared_by_me")

    return jsonify({"status": "ok", "cases": result})


@shared.app.route("/api/similar_cases/<patient_id>/<case_id>")
def api_similar_cases(patient_id, case_id):
    """Lazy-load similar recovered cases — only called when the user requests them."""
    doctor = _get_logged_doctor()
    if not doctor:
        return jsonify({"status": "error", "message": "Unauthorized"}), 401

    case_doc = shared.db.collection("Cases").document(case_id).get()
    if not case_doc.exists:
        return jsonify({"status": "error", "message": "Case not found"}), 404
    c = case_doc.to_dict() or {}

    p_doc = shared.db.collection("Patients").document(patient_id).get()
    if not p_doc.exists:
        return jsonify({"status": "error", "message": "Patient not found"}), 404
    pd_data = p_doc.to_dict() or {}

    patient_age    = compute_age_from_dob(pd_data.get("DateOfBirth", ""))
    patient_gender = pd_data.get("Gender", "")

    scan_docs = list(
        shared.db.collection("MRI_Scans")
        .where("CaseID", "==", f"/Cases/{case_id}")
        .stream()
    )

    # Build scan list sorted by date to identify the baseline (first) scan
    raw_scans = []
    for s in scan_docs:
        sd = s.to_dict() or {}
        mm = sd.get("MaskMetrics") or {}
        dt = sd.get("UploadDate")
        is_dt = isinstance(dt, datetime)
        raw_scans.append({
            "cr":   sd.get("ClassificationResult", ""),
            "area": mm.get("area_pixels"),
            "_dt":  dt if is_dt else None,
        })

    def _caller_sort(x):
        if x["_dt"] is None:
            return (1, datetime(1970, 1, 1))
        try:
            return (0, x["_dt"].replace(tzinfo=None))
        except Exception:
            return (0, x["_dt"])

    raw_scans.sort(key=_caller_sort)

    first_tumor          = raw_scans[0]["cr"]  if raw_scans else ""
    current_first_area   = raw_scans[0]["area"] if raw_scans else None
    current_tumor        = c.get("Diagnosis") or first_tumor

    similar = _find_similar_cases(
        doctor_id=doctor["id"],
        current_patient_id=patient_id,
        current_tumor_type=current_tumor,
        current_patient_age=patient_age,
        current_patient_gender=patient_gender,
        current_first_scan_area=current_first_area,
    )

    return jsonify({"status": "success", "cases": similar, "count": len(similar)})


@shared.app.route("/api/delete_scan/<scan_id>", methods=["POST"])
def delete_scan(scan_id):
    doctor = _get_logged_doctor()
    if not doctor:
        return jsonify({"status": "error", "message": "Unauthorized"}), 401
    try:
        scan_ref = shared.db.document(f"MRI_Scans/{scan_id}")
        reports = shared.db.collection("Reports").where("ScanID", "==", scan_ref).stream()
        for r in reports:
            r.reference.delete()
        shared.db.collection("MRI_Scans").document(scan_id).delete()
        return jsonify({"status": "success"}), 200
    except Exception as e:
        print("Error deleting scan:", e)
        return jsonify({"status": "error", "message": str(e)}), 500


@shared.app.route("/patients/<patient_id>/cases/<case_id>/update_treatment", methods=["POST"])
def update_treatment_plan(patient_id, case_id):
    doctor = _get_logged_doctor()
    if not doctor:
        return redirect(url_for("register_login"))

    new_plan = request.form.get("treatment_plan", "").strip()
    shared.db.collection("Cases").document(case_id).update({
        "TreatmentPlan": new_plan,
        "LastUpdate": shared.now_sa()
    })
    return redirect(url_for("view_case", patient_id=patient_id, case_id=case_id))


@shared.app.route("/patients/<patient_id>/cases/<case_id>/mark_recovered", methods=["POST"])
def mark_case_recovered(patient_id, case_id):
    doctor = _get_logged_doctor()
    if not doctor:
        return jsonify({"status": "error", "message": "Unauthorized"}), 401
    try:
        now = shared.now_sa()
        shared.db.collection("Cases").document(case_id).update({
            "Status": "Recovered",
            "EndDate": now.strftime("%Y-%m-%d"),
            "LastUpdate": now
        })
        shared.clear_dash_cache(doctor["id"])
        return jsonify({"status": "success"}), 200
    except Exception as e:
        print("Error marking case recovered:", e)
        return jsonify({"status": "error", "message": str(e)}), 500


def _collect_refs_for_patient(pid, doc_ref_str):
    """Return list of Firestore DocumentReferences to delete for one patient (or None if unauthorized)."""
    try:
        p_snap = shared.db.collection("Patients").document(pid).get()
        if not p_snap.exists:
            return None
        if (p_snap.to_dict() or {}).get("CreatedBy") != doc_ref_str:
            return None
        refs = [shared.db.collection("Patients").document(pid)]
        for case in shared.db.collection("Cases").where("PatientID", "==", f"/Patients/{pid}").stream():
            refs.append(shared.db.collection("Cases").document(case.id))
            for scan in shared.db.collection("MRI_Scans").where("CaseID", "==", case.id).stream():
                refs.append(scan.reference)
        return refs
    except Exception:
        return None


@shared.app.route("/patients/bulk_delete", methods=["POST"])
def bulk_delete_patients():
    doctor = _get_logged_doctor()
    if not doctor:
        return jsonify({"status": "error", "message": "Unauthorized"}), 401

    data = request.get_json() or {}
    ids = data.get("ids", [])
    if not isinstance(ids, list) or not ids:
        return jsonify({"status": "error", "message": "No IDs provided"}), 400

    doc_ref_str = f"/Radiologists/{doctor['id']}"

    # Collect all refs to delete in parallel
    all_refs = []
    deleted = 0
    with ThreadPoolExecutor(max_workers=min(20, len(ids))) as ex:
        futures = {ex.submit(_collect_refs_for_patient, pid, doc_ref_str): pid for pid in ids}
        for fut in as_completed(futures):
            refs = fut.result()
            if refs:
                all_refs.extend(refs)
                deleted += 1

    # Commit deletes in Firestore batches (max 500 per batch)
    BATCH_SIZE = 450
    for i in range(0, len(all_refs), BATCH_SIZE):
        batch = shared.db.batch()
        for ref in all_refs[i:i + BATCH_SIZE]:
            batch.delete(ref)
        batch.commit()

    shared.clear_dash_cache(doctor["id"])
    return jsonify({"status": "success", "deleted": deleted}), 200


@shared.app.route("/delete_patient/<patient_id>", methods=["POST"])
def delete_patient(patient_id):
    doctor = _get_logged_doctor()
    if not doctor:
        return redirect(url_for("register_login"))

    p_snap = shared.db.collection("Patients").document(patient_id).get()
    if not p_snap.exists:
        return redirect(url_for("patients"))
    if (p_snap.to_dict() or {}).get("CreatedBy") != f"/Radiologists/{doctor['id']}":
        return "Unauthorized: you do not own this patient.", 403

    patient_ref = shared.db.collection("Patients").document(patient_id)
    patient_ref.delete()

    cases_ref = shared.db.collection("Cases").where("PatientID", "==", f"/Patients/{patient_id}").stream()

    for case in cases_ref:
        case_id = case.id
        scans_ref = shared.db.collection("MRI_Scans").where("CaseID", "==", case_id).stream()

        for scan in scans_ref:
            scan_data = scan.to_dict()
            paths = [
                scan_data.get("MRIFilePath"),
                scan_data.get("GradCAMPath"),
                scan_data.get("SegmentationMaskPath"),
            ]
            for p in paths:
                if p:
                    try:
                        bucket = storage.bucket()
                        blob = bucket.blob(p.replace("/storage/", ""))
                        blob.delete()
                    except Exception:
                        pass

            shared.db.collection("MRI_Scans").document(scan.id).delete()

        shared.db.collection("Cases").document(case_id).delete()

    doctor = _get_logged_doctor()
    if doctor:
        shared.clear_dash_cache(doctor["id"])
    flash("Patient deleted successfully.", "success")
    return redirect(url_for("patients"))


def _can_access_case_comments(doctor, case_id):
    """Return True if doctor is the case owner or an invited doctor."""
    case_doc = shared.db.collection("Cases").document(case_id).get()
    if not case_doc.exists:
        return False
    case = case_doc.to_dict() or {}
    patient_id = case.get("PatientID", "").split("/")[-1]
    if patient_id:
        p = shared.db.collection("Patients").document(patient_id).get()
        if p.exists:
            owner_ref = p.to_dict().get("CreatedBy", "")
            if owner_ref == f"/Radiologists/{doctor['id']}":
                return True
    return doctor["id"] in case.get("InvitedDoctors", [])


@shared.app.route("/api/case_comments/<case_id>", methods=["GET"])
def get_case_comments(case_id):
    doctor = _get_logged_doctor()
    if not doctor:
        return jsonify({"status": "error"}), 401
    if not _can_access_case_comments(doctor, case_id):
        return jsonify({"status": "error", "message": "Unauthorized"}), 403
    try:
        docs = (
            shared.db.collection("Cases").document(case_id)
            .collection("Comments")
            .order_by("created_at")
            .stream()
        )
        comments = []
        for d in docs:
            data = d.to_dict()
            ts = data.get("created_at")
            comments.append({
                "id":           d.id,
                "doctor_name":  data.get("doctor_name", "Unknown"),
                "doctor_id":    data.get("doctor_id", ""),
                "text":         data.get("text", ""),
                "created_at":   ts.strftime("%d %b %Y, %H:%M") if ts else "",
            })
        return jsonify({"status": "success", "comments": comments})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@shared.app.route("/api/case_comments/<case_id>", methods=["POST"])
def add_case_comment(case_id):
    doctor = _get_logged_doctor()
    if not doctor:
        return jsonify({"status": "error"}), 401
    if not _can_access_case_comments(doctor, case_id):
        return jsonify({"status": "error", "message": "Unauthorized"}), 403
    text = (request.json or {}).get("text", "").strip()
    if not text:
        return jsonify({"status": "error", "message": "Empty comment"}), 400
    try:
        shared.db.collection("Cases").document(case_id).collection("Comments").add({
            "doctor_id":   doctor["id"],
            "doctor_name": doctor.get("name", "Dr."),
            "text":        text,
            "created_at":  shared.now_sa(),
        })

        # ── Notify all OTHER doctors who have access to this case ──
        _notify_case_doctors_of_comment(case_id, doctor, text, request.host_url)

        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


def _notify_case_doctors_of_comment(case_id, commenter, comment_text, host_url):
    """Send email notification to all case doctors (owner + invited) except the commenter."""
    try:
        case_doc = shared.db.collection("Cases").document(case_id).get()
        if not case_doc.exists:
            return
        case_data = case_doc.to_dict() or {}

        notify_ids = set()

        # Owner
        patient_ref = case_data.get("PatientID", "")
        patient_id  = patient_ref.split("/")[-1] if "/" in patient_ref else patient_ref
        if patient_id:
            p = shared.db.collection("Patients").document(patient_id).get()
            if p.exists:
                owner_ref = (p.to_dict() or {}).get("CreatedBy", "")
                owner_id  = owner_ref.split("/")[-1] if "/" in owner_ref else owner_ref
                if owner_id:
                    notify_ids.add(owner_id)

        # Invited doctors
        for inv_id in case_data.get("InvitedDoctors", []):
            notify_ids.add(inv_id)

        # Exclude the commenter
        notify_ids.discard(commenter["id"])

        base_url = host_url.rstrip("/")
        case_url = f"{base_url}/patients"

        for doc_id in notify_ids:
            doc_snap = shared.db.collection("Radiologists").document(doc_id).get()
            if not doc_snap.exists:
                continue
            doc_email = (doc_snap.to_dict() or {}).get("Email", "")
            if not doc_email:
                continue
            try:
                send_comment_notification(
                    to_email       = doc_email,
                    commenter_name = commenter.get("name", "A doctor"),
                    comment_text   = comment_text,
                    case_url       = case_url,
                )
            except Exception as mail_err:
                print(f"Comment notification failed for {doc_email}:", mail_err)
    except Exception as e:
        print("_notify_case_doctors_of_comment error:", e)


@shared.app.route("/api/case_comments/<case_id>/<comment_id>", methods=["DELETE"])
def delete_case_comment(case_id, comment_id):
    doctor = _get_logged_doctor()
    if not doctor:
        return jsonify({"status": "error"}), 401
    if not _can_access_case_comments(doctor, case_id):
        return jsonify({"status": "error", "message": "Unauthorized"}), 403
    # Invited doctors cannot delete any comment
    case_doc = shared.db.collection("Cases").document(case_id).get()
    if case_doc.exists:
        invited = case_doc.to_dict().get("InvitedDoctors", [])
        if doctor["id"] in invited:
            return jsonify({"status": "error", "message": "Invited consultants cannot delete comments"}), 403
    try:
        ref = (
            shared.db.collection("Cases").document(case_id)
            .collection("Comments").document(comment_id)
        )
        snap = ref.get()
        if not snap.exists:
            return jsonify({"status": "error", "message": "Not found"}), 404
        if snap.to_dict().get("doctor_id") != doctor["id"]:
            return jsonify({"status": "error", "message": "Unauthorized"}), 403
        ref.delete()
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@shared.app.route("/api/invite_case/<case_id>", methods=["POST"])
def invite_case(case_id):
    doctor = _get_logged_doctor()
    if not doctor:
        return jsonify({"status": "error", "message": "Unauthorized"}), 401

    email_raw = (request.json or {}).get("email", "").strip()
    email = email_raw.lower()
    if not email:
        return jsonify({"status": "error", "message": "Email required"}), 400

    case_doc = shared.db.collection("Cases").document(case_id).get()
    if not case_doc.exists:
        return jsonify({"status": "error", "message": "Case not found"}), 404
    case_data = case_doc.to_dict() or {}

    patient_ref_str = case_data.get("PatientID", "")
    p_id = patient_ref_str.replace("/Patients/", "")
    p_snap = shared.db.collection("Patients").document(p_id).get()
    if not p_snap.exists:
        return jsonify({"status": "error", "message": "Patient not found"}), 404
    p_data = p_snap.to_dict() or {}

    owner_ref = p_data.get("CreatedBy", "")
    if owner_ref != f"/Radiologists/{doctor['id']}":
        return jsonify({"status": "error", "message": "Only the case owner can send invites"}), 403

    # Try to find existing Brainalyze account (case-insensitive)
    invited_docs = list(
        shared.db.collection("Radiologists").where("Email", "==", email_raw).limit(1).stream()
    )
    if not invited_docs and email_raw != email:
        invited_docs = list(
            shared.db.collection("Radiologists").where("Email", "==", email).limit(1).stream()
        )

    invited_id = invited_docs[0].id if invited_docs else None

    if invited_id and invited_id == doctor["id"]:
        return jsonify({"status": "error", "message": "You cannot invite yourself"}), 400

    already = case_data.get("InvitedDoctors", [])
    if invited_id and invited_id in already:
        return jsonify({"status": "error", "message": "This doctor is already invited"}), 400

    token = secrets.token_urlsafe(32)
    shared.db.collection("CaseInvites").document(token).set({
        "token":              token,
        "case_id":            case_id,
        "patient_id":         p_id,
        "inviter_id":         doctor["id"],
        "inviter_name":       doctor["name"],
        "invited_doctor_id":  invited_id or "",
        "invited_email":      email_raw,
        "status":             "pending",
        "created_at":         shared.now_sa(),
    })

    accept_url  = f"{request.host_url}accept_invite/{token}"
    decline_url = f"{request.host_url}decline_invite/{token}"
    try:
        if invited_id:
            # Registered doctor → full invite email with Accept & Decline
            send_invite_email(
                to_email=email_raw,
                inviter_name=doctor["name"],
                patient_name=p_data.get("FullName", "Patient"),
                diagnosis=case_data.get("Diagnosis", ""),
                accept_url=accept_url,
                decline_url=decline_url,
            )
        else:
            # Unregistered → Join Brainalyze email with Create Account button
            send_join_email(
                to_email=email_raw,
                inviter_name=doctor["name"],
                patient_name=p_data.get("FullName", "Patient"),
                diagnosis=case_data.get("Diagnosis", ""),
                accept_url=accept_url,
            )
    except Exception as e:
        print("Invite email error:", e)
        return jsonify({"status": "error", "message": "Failed to send email"}), 500

    return jsonify({"status": "success"})



@shared.app.route("/switch_account_for_invite/<token>")
def switch_account_for_invite(token):
    session.pop("radiologist_id", None)
    session["_pending_invite_token"] = token
    return redirect(url_for("register_login") + "#login")


@shared.app.route("/decline_invite/<token>")
def decline_invite(token):
    invite_ref = shared.db.collection("CaseInvites").document(token)
    invite_snap = invite_ref.get()
    if not invite_snap.exists:
        return """<!DOCTYPE html><html><body style="font-family:Poppins,Arial,sans-serif;
                  display:flex;align-items:center;justify-content:center;min-height:100vh;
                  background:#f7faff;margin:0;">
                  <div style="text-align:center;padding:40px;">
                    <div style="font-size:48px;">❌</div>
                    <h2 style="color:#1e2a47;">Invalid Link</h2>
                    <p style="color:#6a7393;">This invite link is invalid or has already been used.</p>
                  </div></body></html>""", 404

    invite = invite_snap.to_dict() or {}
    if invite.get("status") in ("accepted", "declined"):
        msg = "already accepted" if invite["status"] == "accepted" else "already declined"
        return f"""<!DOCTYPE html><html><body style="font-family:Poppins,Arial,sans-serif;
                   display:flex;align-items:center;justify-content:center;min-height:100vh;
                   background:#f7faff;margin:0;">
                   <div style="text-align:center;padding:40px;">
                     <div style="font-size:48px;">ℹ️</div>
                     <h2 style="color:#1e2a47;">Invite {msg.title()}</h2>
                     <p style="color:#6a7393;">This invitation has been {msg}.</p>
                   </div></body></html>"""

    invite_ref.update({"status": "declined"})

    # Remove doctor from InvitedDoctors if they were already added (edge-case safety)
    invited_doctor_id = invite.get("invited_doctor_id", "")
    if invited_doctor_id:
        try:
            from google.cloud.firestore_v1 import ArrayRemove
            shared.db.collection("Cases").document(invite.get("case_id", "")).update({
                "InvitedDoctors": ArrayRemove([invited_doctor_id])
            })
        except Exception:
            pass

    inviter_name = invite.get("inviter_name", "The doctor")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Invitation Declined — Brainalyze</title>
  <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap" rel="stylesheet">
</head>
<body style="margin:0;background:#f7faff;font-family:'Poppins',Arial,sans-serif;
             display:flex;align-items:center;justify-content:center;min-height:100vh;">
  <div style="background:#fff;border-radius:20px;padding:48px 40px;max-width:440px;width:100%;
              text-align:center;box-shadow:0 10px 40px rgba(0,0,0,0.08);margin:16px;">
    <div style="font-size:52px;margin-bottom:16px;">🚫</div>
    <h2 style="font-size:22px;font-weight:700;color:#1e2a47;margin:0 0 10px 0;">
      Invitation Declined
    </h2>
    <p style="font-size:14px;color:#6a7393;line-height:1.7;margin:0 0 28px 0;">
      You have declined the case consultation invitation from
      <strong>Dr. {inviter_name}</strong>.<br>
      They will be notified that you are not available.
    </p>
    <a href="/home" style="background:#506DCA;color:white;padding:12px 30px;border-radius:12px;
                           font-size:14px;font-weight:600;text-decoration:none;display:inline-block;">
      Go to Brainalyze
    </a>
  </div>
</body>
</html>"""


@shared.app.route("/accept_invite/<token>")
def accept_invite(token):
    invite_ref = shared.db.collection("CaseInvites").document(token)
    invite_snap = invite_ref.get()
    if not invite_snap.exists:
        return "Invalid or expired invite link.", 404

    invite = invite_snap.to_dict() or {}

    doctor = _get_logged_doctor()
    if not doctor:
        session["_pending_invite_token"] = token
        return redirect(url_for("register_login"))

    stored_id    = invite.get("invited_doctor_id", "")
    invited_email = invite.get("invited_email", "").lower()
    doctor_email  = (doctor.get("email") or "").lower()

    mismatch = False
    if stored_id:
        if stored_id != doctor["id"]:
            mismatch = True
    else:
        if doctor_email != invited_email:
            mismatch = True
        else:
            invite_ref.update({"invited_doctor_id": doctor["id"]})

    if mismatch:
        switch_url = f"/switch_account_for_invite/{token}"
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Wrong Account — Brainalyze</title>
  <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap" rel="stylesheet">
</head>
<body style="margin:0;background:#f7faff;font-family:'Poppins',Arial,sans-serif;
             display:flex;align-items:center;justify-content:center;min-height:100vh;">
  <div style="background:#fff;border-radius:20px;padding:48px 40px;max-width:440px;width:100%;
              text-align:center;box-shadow:0 10px 40px rgba(0,0,0,0.08);margin:16px;">
    <div style="font-size:52px;margin-bottom:16px;">🔐</div>
    <h2 style="font-size:22px;font-weight:700;color:#1e2a47;margin:0 0 10px 0;">
      Wrong Account
    </h2>
    <p style="font-size:14px;color:#6a7393;line-height:1.7;margin:0 0 28px 0;">
      This invitation was sent to <strong>{invited_email}</strong>.<br>
      You are currently logged in with a different account.<br><br>
      Please switch to the correct account to accept this invitation.
    </p>
    <a href="{switch_url}"
       style="background:#506DCA;color:white;padding:13px 30px;border-radius:12px;
              font-size:14px;font-weight:600;text-decoration:none;display:inline-block;margin-bottom:12px;">
      Switch Account &amp; Accept
    </a>
    <br>
    <a href="/home"
       style="font-size:13px;color:#9ca3af;text-decoration:none;">
      Back to Brainalyze
    </a>
  </div>
</body>
</html>"""

    if invite.get("status") != "accepted":
        invite_ref.update({"status": "accepted"})
        shared.db.collection("Cases").document(invite["case_id"]).update({
            "InvitedDoctors": firestore.ArrayUnion([doctor["id"]])
        })

    return redirect(url_for(
        "view_case",
        patient_id=invite["patient_id"],
        case_id=invite["case_id"]
    ))


@shared.app.route("/patients/<patient_id>/cases/<case_id>/delete", methods=["POST"])
def delete_case(patient_id, case_id):
    doctor = _get_logged_doctor()
    if not doctor:
        return redirect(url_for("register_login"))

    case_ref = shared.db.collection("Cases").document(case_id)
    case_snap = case_ref.get()

    if not case_snap.exists:
        return "Case not found", 404

    case_data = case_snap.to_dict()
    if case_data.get("PatientID") != f"/Patients/{patient_id}":
        return "Unauthorized", 403

    scans_to_delete = shared.db.collection("MRI_Scans").where(
        "CaseID", "==", f"/Cases/{case_id}"
    ).stream()

    for s in scans_to_delete:
        s.reference.delete()

    case_ref.delete()
    shared.clear_dash_cache(doctor["id"])
    return redirect(url_for("patient_profile", patient_id=patient_id))
