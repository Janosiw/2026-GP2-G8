import time
from datetime import datetime
from flask import render_template, redirect, url_for
from firebase_admin import firestore

import shared
from utils import _get_logged_doctor, compute_age_from_dob, _normalize_tumor_type

def _get_dash_cache(doctor_id):
    entry = shared._dashboard_cache.get(doctor_id)
    if entry and (time.time() - entry[0]) < shared._DASH_TTL:
        return entry[1]
    return None

def _set_dash_cache(doctor_id, data):
    shared._dashboard_cache[doctor_id] = (time.time(), data)


@shared.app.route("/home")
def home():
    cases = []
    doctor = _get_logged_doctor()
    if not doctor:
        return redirect(url_for("register_login"))

    today = shared.now_sa().date()
    now = shared.now_sa()

    patients_ref = shared.db.collection("Patients").where(
        "CreatedBy", "==", f"/Radiologists/{doctor['id']}"
    )
    patients_docs = list(patients_ref.stream())

    today_patients = 0
    for p in patients_docs:
        pdata = p.to_dict() or {}
        created_at = pdata.get("CreatedAt")
        if isinstance(created_at, datetime) and created_at.date() == today:
            today_patients += 1

    start_of_day = datetime(now.year, now.month, now.day, 0, 0, 0)
    doctor_patient_refs = {f"/Patients/{p.id}" for p in patients_docs}
    scans_today = shared.db.collection("MRI_Scans").where("UploadDate", ">=", start_of_day).stream()
    today_completed = sum(
        1 for s in scans_today
        if (s.to_dict() or {}).get("PatientID", "") in doctor_patient_refs
    )

    visited_docs = (
        patients_ref
        .where("LastVisitedAt", ">", shared.EPOCH)
        .order_by("LastVisitedAt", direction=firestore.Query.DESCENDING)
        .stream()
    )

    visited_list = []
    for d in visited_docs:
        x = d.to_dict() or {}
        t = x.get("LastVisitedAt")
        visited_list.append({
            "patient_id": d.id,
            "name": x.get("FullName", "—"),
            "time": shared.fmt_sa(t) if isinstance(t, datetime) else "—"
        })

    # Build patient_map for name lookup
    patient_map = {}
    for p in patients_docs:
        x = p.to_dict() or {}
        patient_map[f"/Patients/{p.id}"] = {"id": p.id, "name": x.get("FullName", "—")}

    # Query MRI_Scans directly — one entry per scan, not per patient
    all_recent_scans = []
    doctor_patient_refs_list = list(doctor_patient_refs)
    for i in range(0, len(doctor_patient_refs_list), 30):
        batch = doctor_patient_refs_list[i:i+30]
        if not batch:
            continue
        batch_docs = (
            shared.db.collection("MRI_Scans")
            .where("PatientID", "in", batch)
            .stream()
        )
        for s in batch_docs:
            sd = s.to_dict() or {}
            sd["_scan_id"] = s.id
            all_recent_scans.append(sd)

    # Sort across all batches by date descending
    all_recent_scans.sort(
        key=lambda x: x.get("UploadDate") or shared.EPOCH,
        reverse=True
    )

    scan_list_raw = []
    for sd in all_recent_scans:
        pat_ref  = sd.get("PatientID", "")
        pat_info = patient_map.get(pat_ref, {})
        tumor    = _normalize_tumor_type(sd.get("ClassificationResult") or "")
        upload_dt = sd.get("UploadDate")
        time_str  = shared.fmt_sa(upload_dt) if isinstance(upload_dt, datetime) else "—"
        scan_list_raw.append({
            "patient_id": pat_info.get("id", ""),
            "scan_id":    sd["_scan_id"],
            "name":       pat_info.get("name", "—"),
            "time":       time_str,
            "tumor":      tumor,
        })

    has_more_scans    = len(scan_list_raw) > 5
    has_more_visited  = len(visited_list) > 5

    return render_template(
        "home.html",
        doctor=doctor,
        total_patients=today_patients,
        completed_scans=today_completed,
        pending_reports=0,
        visited_list=visited_list,
        scan_list=scan_list_raw,
        has_more_scans=has_more_scans,
        has_more_visited=has_more_visited,
        cases=[]
    )


@shared.app.route("/dashboard")
def dashboard():
    doctor = _get_logged_doctor()
    if not doctor:
        return redirect(url_for("register_login"))

    doctor_id = doctor["id"]

    cached = _get_dash_cache(doctor_id)
    if cached:
        return render_template("dashboard.html", doctor=doctor, **cached)

    patients_docs = list(
        shared.db.collection("Patients")
        .where("CreatedBy", "==", f"/Radiologists/{doctor_id}")
        .stream()
    )

    total_patients = len(patients_docs)
    age_groups = {"0-20": 0, "21-40": 0, "41-60": 0, "61+": 0}
    gender_counts = {"Male": 0, "Female": 0}
    monthly_trend = {}
    yearly_trend = {}
    patient_ids = []

    for pdoc in patients_docs:
        pdata = pdoc.to_dict() or {}
        patient_ids.append(pdoc.id)

        age_str = compute_age_from_dob(pdata.get("DateOfBirth", ""))
        if age_str:
            try:
                age = int(age_str)
                if age <= 20:
                    age_groups["0-20"] += 1
                elif age <= 40:
                    age_groups["21-40"] += 1
                elif age <= 60:
                    age_groups["41-60"] += 1
                else:
                    age_groups["61+"] += 1
            except Exception:
                pass

        gender = (pdata.get("Gender") or "").strip().lower()
        if gender in ("male", "m"):
            gender_counts["Male"] += 1
        elif gender in ("female", "f"):
            gender_counts["Female"] += 1
        # Other genders are excluded from chart

        created_at = pdata.get("CreatedAt")
        if isinstance(created_at, datetime):
            mk = created_at.strftime("%Y-%m")
            yk = created_at.strftime("%Y")
            monthly_trend[mk] = monthly_trend.get(mk, 0) + 1
            yearly_trend[yk] = yearly_trend.get(yk, 0) + 1

    # Count patients without any scan uploaded
    patients_no_scan = sum(
        1 for pdoc in patients_docs
        if not (pdoc.to_dict() or {}).get("LastScanId", "")
    )

    # Build tumor-by-status + avg recovery time from Cases
    TUMOR_TYPES = ["Glioma", "Meningioma", "Pituitary", "Unknown"]
    tumor_by_status = {t: {"Active": 0, "Recovered": 0} for t in TUMOR_TYPES}
    recovery_counts  = {"Active": 0, "Recovered": 0}
    recovery_days    = {t: [] for t in TUMOR_TYPES}   # days list per tumor type

    for i in range(0, len(patient_ids), 30):
        batch_refs = [f"/Patients/{pid}" for pid in patient_ids[i:i+30]]
        if not batch_refs:
            continue
        cases_batch = shared.db.collection("Cases").where("PatientID", "in", batch_refs).stream()
        for case in cases_batch:
            cdata = case.to_dict() or {}
            st = (cdata.get("Status") or "Active").strip().lower()
            status_key = "Recovered" if st == "recovered" else "Active"

            tumor = _normalize_tumor_type(cdata.get("Diagnosis") or "")
            if tumor not in TUMOR_TYPES:
                tumor = "Unknown"
            tumor_by_status[tumor][status_key] += 1
            recovery_counts[status_key] += 1

            # Compute recovery duration for Recovered cases with both dates
            if status_key == "Recovered":
                try:
                    start = datetime.strptime(cdata["StartDate"], "%Y-%m-%d")
                    end   = datetime.strptime(cdata["EndDate"],   "%Y-%m-%d")
                    days  = abs((end - start).days)
                    if days > 0:
                        recovery_days[tumor].append(days)
                except Exception:
                    pass

    # Average recovery time per tumor type (None if no data)
    avg_recovery_days = {
        t: round(sum(v) / len(v)) if v else None
        for t, v in recovery_days.items()
    }

    # Recovery rate % per tumor type
    recovery_rate = {}
    for t in TUMOR_TYPES:
        total = tumor_by_status[t]["Active"] + tumor_by_status[t]["Recovered"]
        recovery_rate[t] = round(tumor_by_status[t]["Recovered"] / total * 100) if total else None

    sorted_monthly = sorted(monthly_trend.items())
    sorted_yearly = sorted(yearly_trend.items())

    stats = dict(
        total_patients=total_patients,
        age_groups=age_groups,
        gender_counts=gender_counts,
        tumor_by_status=tumor_by_status,
        recovery_counts=recovery_counts,
        avg_recovery_days=avg_recovery_days,
        recovery_rate=recovery_rate,
        patients_no_scan=patients_no_scan,
        monthly_labels=[m for m, _ in sorted_monthly],
        monthly_data=[c for _, c in sorted_monthly],
        yearly_labels=[y for y, _ in sorted_yearly],
        yearly_data=[c for _, c in sorted_yearly],
    )
    _set_dash_cache(doctor_id, stats)

    return render_template("dashboard.html", doctor=doctor, **stats)
