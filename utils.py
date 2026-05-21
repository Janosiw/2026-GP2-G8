import re
import os
import random
import smtplib
import requests
from datetime import datetime, date
from difflib import SequenceMatcher
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import shared
from smtp_config import SMTP_HOST, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD, FROM_EMAIL


def send_otp_email(to_email, otp_code):
    msg = MIMEMultipart("alternative")
    msg["Subject"] = "Your Brainalyze Login Code"
    msg["From"] = FROM_EMAIL
    msg["To"] = to_email
    html = f"""
<!DOCTYPE html>
<html lang="en">
<body style="margin:0;padding:30px;background:#e9ecff;font-family:Poppins,Arial,sans-serif;">
  <table width="100%" align="center" style="max-width:520px;margin:auto;background:#fff;border-radius:16px;padding:35px;">
    <tr><td style="text-align:center;font-size:26px;font-weight:700;color:#506DCA;padding-bottom:25px;">Brainalyze – Login Verification</td></tr>
    <tr><td style="font-size:16px;color:#222;line-height:1.7;">Your one-time login code is:</td></tr>
    <tr><td style="text-align:center;padding:30px 0;">
      <span style="background:#506DCA;color:white;padding:14px 32px;border-radius:12px;font-size:28px;font-weight:700;letter-spacing:8px;">{otp_code}</span>
    </td></tr>
    <tr><td style="font-size:13px;color:#777;padding-top:25px;">This code expires in 10 minutes. If you did not request this, please ignore it.</td></tr>
  </table>
</body>
</html>"""
    msg.attach(MIMEText(html, "html"))
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.sendmail(FROM_EMAIL, to_email, msg.as_string())


def call_hf_report_api(findings_text: str) -> dict:
    import os
    import requests

    base = os.getenv("REPORT_LLM_URL", "").rstrip("/")

    if not base:
        raise RuntimeError("REPORT_LLM_URL is not set")

    url = f"{base}/generate_impression"

    headers = {
        "Content-Type": "application/json"
    }

    token = os.getenv("REPORT_LLM_TOKEN", "").strip()

    if token:
        headers["Authorization"] = f"Bearer {token}"

    timeout = int(os.getenv("REPORT_LLM_TIMEOUT", "60"))

    r = requests.post(
        url,
        json={"findings_text": findings_text},
        headers=headers,
        timeout=timeout
    )

    print("LLM status:", r.status_code)
    print("LLM response:", r.text[:500])

    r.raise_for_status()

    return r.json()


def _get_logged_doctor():
    from flask import session
    rid = session.get("radiologist_id")
    if not rid:
        return None

    cached = session.get("_doctor_cache")
    if cached and cached.get("id") == rid:
        return cached

    doc = shared.db.collection("Radiologists").document(rid).get()
    if not doc.exists:
        return None
    d = doc.to_dict() or {}
    doctor = {
        "id": rid,
        "name": d.get("FullName", "Radiologist"),
        "email": d.get("Email", "N/A"),
        "ProfilePicture": d.get("ProfilePicture", None)
    }
    session["_doctor_cache"] = doctor
    return doctor


def _invalidate_doctor_cache():
    from flask import session
    session.pop("_doctor_cache", None)


def _batch_case_status(patient_ids):
    """Returns {patient_id: 'Active'|'Recovered'|''} using batch Firestore queries."""
    status_map = {}
    ids = list(patient_ids)
    for i in range(0, len(ids), 30):
        batch_refs = [f"/Patients/{pid}" for pid in ids[i:i+30]]
        if not batch_refs:
            continue
        cases = shared.db.collection("Cases").where("PatientID", "in", batch_refs).stream()
        for case in cases:
            cdata = case.to_dict() or {}
            pid_ref = (cdata.get("PatientID") or "")
            pid = pid_ref.split("/")[-1] if "/" in pid_ref else pid_ref
            if cdata.get("Status", "Active").strip().lower() == "active":
                status_map[pid] = "Active"
            elif pid not in status_map:
                status_map[pid] = "Recovered"
    return status_map


def compute_initials(full_name: str) -> str:
    name = (full_name or "").strip()
    if not name:
        return ""
    parts = name.split()
    if len(parts) == 1:
        return parts[0][0].upper()
    return (parts[0][0] + parts[1][0]).upper()


def compute_age_from_dob(dob_value) -> str:
    if not dob_value:
        return ""
    dob_date = None
    if isinstance(dob_value, datetime):
        dob_date = dob_value.date()
    elif isinstance(dob_value, date):
        dob_date = dob_value
    elif isinstance(dob_value, str):
        try:
            dob_date = datetime.strptime(dob_value, "%Y-%m-%d").date()
        except ValueError:
            return ""
    else:
        return ""
    today = date.today()
    if dob_date > today:
        return ""
    age = today.year - dob_date.year - ((today.month, today.day) < (dob_date.month, dob_date.day))
    if age < 0 or age > 130:
        return ""
    return str(age)


def _mask_identifier(value: str, prefix: int = 6, mask_len: int = 4, mask_char: str = "*") -> str:
    raw = (value or "").strip()
    if not raw:
        return ""
    if len(raw) <= prefix:
        return raw
    return f"{raw[:prefix]}{mask_char * mask_len}"


def _normalize_tumor_type(value: str) -> str:
    return shared.TUMOR_TYPE_CANONICAL.get(_canonical_tumor_type(value), "")


def _canonical_tumor_type(value: str) -> str:
    token = re.sub(r"[^a-z]", "", (value or "").strip().lower())
    if token in ("notumor", "none", "unknown", ""):
        return ""
    return token if token in shared.TUMOR_TYPE_CANONICAL else ""


def _serialize_patient_for_search(patient_doc):
    pdata = patient_doc.to_dict() or {}
    full_name = pdata.get("FullName", "")
    gender = pdata.get("Gender", "")
    tumor_type = (
        _normalize_tumor_type(pdata.get("TumorType", ""))
        or _normalize_tumor_type(pdata.get("LastScanTumor", ""))
        or pdata.get("TumorType", "")
        or pdata.get("LastScanTumor", "")
    )
    last_mri_date = pdata.get("LastMRIDate", "")
    dob = pdata.get("DateOfBirth", "")
    return {
        "id": patient_doc.id,
        "MaskedId": _mask_identifier(patient_doc.id),
        "FullName": full_name,
        "Gender": gender,
        "TumorType": tumor_type or "",
        "LastMRIDate": last_mri_date,
        "LastScanId": pdata.get("LastScanId", ""),
        "Initials": compute_initials(full_name),
        "Age": compute_age_from_dob(dob)
    }


def _filter_and_suggest_patients(query_text, all_patients):
    q = (query_text or "").strip().lower()
    if not q:
        return list(all_patients), []

    matches = []
    for p in all_patients:
        name_l = (p.get("FullName") or "").lower()
        pid_l = (p.get("id") or "").lower()
        if q in name_l or q in pid_l:
            matches.append(p)

    if matches:
        return matches, []

    scored = []
    for p in all_patients:
        name_l = (p.get("FullName") or "").lower()
        pid_l = (p.get("id") or "").lower()
        name_tokens = [t for t in name_l.split() if t]
        score = max(
            SequenceMatcher(None, q, name_l).ratio() if name_l else 0.0,
            SequenceMatcher(None, q, pid_l).ratio() if pid_l else 0.0,
            max((SequenceMatcher(None, q, tok).ratio() for tok in name_tokens), default=0.0)
        )
        if score >= 0.55:
            scored.append((score, p))

    scored.sort(key=lambda x: x[0], reverse=True)
    suggestions = []
    used_ids = set()
    for _, p in scored:
        pid = p.get("id")
        if pid in used_ids:
            continue
        used_ids.add(pid)
        suggestions.append(p)
        if len(suggestions) == 5:
            break

    return [], suggestions


def _parse_yyyy_mm_dd(value):
    try:
        return datetime.strptime((value or "").strip(), "%Y-%m-%d").date()
    except ValueError:
        return None


def _apply_patient_filters(all_patients, tumor="", date_from="", date_to="", case_status="", no_scan=False):
    tumor_norm = _canonical_tumor_type(tumor)
    from_date = _parse_yyyy_mm_dd(date_from)
    to_date = _parse_yyyy_mm_dd(date_to)
    status_norm = (case_status or "").strip().lower()

    filtered = []
    for p in all_patients:
        if no_scan and p.get("LastScanId", ""):
            continue
        patient_tumor_norm = _canonical_tumor_type(p.get("TumorType", ""))
        if tumor_norm and patient_tumor_norm != tumor_norm:
            continue
        if from_date or to_date:
            visit_date = _parse_yyyy_mm_dd(p.get("LastMRIDate", ""))
            if not visit_date:
                continue
            if from_date and visit_date < from_date:
                continue
            if to_date and visit_date > to_date:
                continue
        if status_norm:
            p_status = (p.get("CaseStatus") or "").strip().lower()
            if p_status != status_norm:
                continue
        filtered.append(p)

    return filtered


def send_join_email(to_email, inviter_name, patient_name, diagnosis, accept_url):
    """Email sent to doctors who don't have a Brainalyze account yet."""
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"Dr. {inviter_name} invited you to review a case on Brainalyze"
    msg["From"] = FROM_EMAIL
    msg["To"] = to_email
    html = f"""
<!DOCTYPE html>
<html lang="en">
<body style="margin:0;padding:30px;background:#e9ecff;font-family:Poppins,Arial,sans-serif;">
  <table width="100%" align="center" style="max-width:520px;margin:auto;background:#fff;border-radius:16px;padding:35px;">
    <tr><td style="text-align:center;font-size:24px;font-weight:700;color:#506DCA;padding-bottom:20px;">
      Join Brainalyze
    </td></tr>
    <tr><td style="font-size:15px;color:#222;line-height:1.8;">
      Hello,<br><br>
      <b>Dr. {inviter_name}</b> has invited you to consult on a medical case:
    </td></tr>
    <tr><td style="background:#f7faff;border-radius:12px;padding:16px 20px;margin:16px 0;
                   font-size:14px;color:#374151;line-height:1.9;display:block;">
      <b>Patient:</b> {patient_name}<br>
      <b>Diagnosis:</b> {diagnosis or "Pending"}
    </td></tr>
    <tr><td style="font-size:14px;color:#374151;line-height:1.8;padding-bottom:10px;">
      You have been invited to join Brainalyze, a platform for managing and sharing medical cases.
    </td></tr>
    <tr><td style="text-align:center;padding:20px 0;">
      <a href="{accept_url}"
         style="background:#506DCA;color:white;padding:14px 36px;border-radius:12px;
                font-size:16px;font-weight:700;text-decoration:none;display:inline-block;">
        Create Account &amp; View Case
      </a>
    </td></tr>
    <tr><td style="font-size:12px;color:#9ca3af;padding-top:6px;text-align:center;">
      Once registered, you will be able to view and add clinical comments on this case.<br>
      Please register using this exact email address: <b>{to_email}</b>
    </td></tr>
  </table>
</body>
</html>"""
    msg.attach(MIMEText(html, "html"))
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.sendmail(FROM_EMAIL, to_email, msg.as_string())


def send_invite_email(to_email, inviter_name, patient_name, diagnosis, accept_url, decline_url=""):
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"Dr. {inviter_name} invited you to review a case on Brainalyze"
    msg["From"] = FROM_EMAIL
    msg["To"] = to_email
    decline_btn = f"""
      <a href="{decline_url}"
         style="background:#fff;color:#6b7280;padding:13px 28px;border-radius:12px;
                font-size:15px;font-weight:600;text-decoration:none;display:inline-block;
                border:1.5px solid #d1d5db;margin-left:12px;">
        Decline
      </a>""" if decline_url else ""
    html = f"""
<!DOCTYPE html>
<html lang="en">
<body style="margin:0;padding:30px;background:#e9ecff;font-family:Poppins,Arial,sans-serif;">
  <table width="100%" align="center" style="max-width:520px;margin:auto;background:#fff;border-radius:16px;padding:35px;">
    <tr><td style="text-align:center;font-size:24px;font-weight:700;color:#506DCA;padding-bottom:20px;">
      Brainalyze &mdash; Case Invitation
    </td></tr>
    <tr><td style="font-size:15px;color:#222;line-height:1.8;">
      <b>Dr. {inviter_name}</b> has invited you to review a case as a consultant:
    </td></tr>
    <tr><td style="background:#f7faff;border-radius:12px;padding:16px 20px;margin:16px 0;
                   font-size:14px;color:#374151;line-height:1.9;display:block;">
      <b>Patient:</b> {patient_name}<br>
      <b>Diagnosis:</b> {diagnosis or "Pending"}
    </td></tr>
    <tr><td style="text-align:center;padding:28px 0;">
      <a href="{accept_url}"
         style="background:#506DCA;color:white;padding:14px 36px;border-radius:12px;
                font-size:16px;font-weight:700;text-decoration:none;display:inline-block;">
        Accept &amp; View Case
      </a>
      {decline_btn}
    </td></tr>
    <tr><td style="font-size:12px;color:#9ca3af;padding-top:6px;text-align:center;">
      You can add clinical comments on this case. This invite link is unique to you.<br>
      Make sure you are logged in to your Brainalyze account before clicking.
    </td></tr>
  </table>
</body>
</html>"""
    msg.attach(MIMEText(html, "html"))
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.sendmail(FROM_EMAIL, to_email, msg.as_string())


def save_report_document(scan_id, content, created_at, gradcam_path="",
                         mri_file_path="", pdf_path="", segmentation_mask_path="",
                         is_3d=False, volume_metrics=None):
    report_ref  = shared.db.collection("Reports").document()
    report_data = {
        "ReportID":             report_ref.id,
        "Content":              content,
        "CreatedAt":            created_at,
        "GradCAMPath":          gradcam_path,
        "MRIFilePath":          mri_file_path,
        "PDFPath":              pdf_path,
        "ScanID":               shared.db.document(f"MRI_Scans/{scan_id}"),
        "SegmentationMaskPath": segmentation_mask_path,
        "Is3D":                 is_3d,
        "VolumeMetrics":        volume_metrics or {},
    }
    report_ref.set(report_data)
    return report_ref.id


def send_comment_notification(to_email: str, commenter_name: str, comment_text: str, case_url: str) -> None:
    """Notify a doctor that a new comment was posted on a case they have access to."""
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"New comment by Dr. {commenter_name} — Brainalyze"
    msg["From"]    = FROM_EMAIL
    msg["To"]      = to_email
    html = f"""
<!DOCTYPE html>
<html lang="en">
<body style="margin:0;padding:30px;background:#e9ecff;font-family:Poppins,Arial,sans-serif;">
  <table width="100%" align="center" style="max-width:520px;margin:auto;background:#fff;border-radius:16px;padding:35px;">
    <tr><td style="text-align:center;font-size:22px;font-weight:700;color:#1e2a47;padding-bottom:18px;">
      New Comment on a Shared Case
    </td></tr>
    <tr><td style="font-size:15px;color:#222;line-height:1.8;">
      Dr. <b>{commenter_name}</b> left a new comment on a case you have access to:
    </td></tr>
    <tr><td style="background:#f5f8ff;border-radius:12px;padding:16px 20px;margin:16px 0;
                   font-size:14px;color:#374151;line-height:1.7;display:block;
                   border-left:4px solid #3d6fa8;">
      &ldquo;{comment_text}&rdquo;
    </td></tr>
    <tr><td style="text-align:center;padding:24px 0 8px;">
      <a href="{case_url}"
         style="background:#3d6fa8;color:white;padding:13px 30px;border-radius:12px;
                font-size:15px;font-weight:600;text-decoration:none;display:inline-block;">
        Open Brainalyze
      </a>
    </td></tr>
    <tr><td style="font-size:12px;color:#9ca3af;padding-top:10px;text-align:center;">
      Log in to view the full case and reply.
    </td></tr>
  </table>
</body>
</html>"""
    msg.attach(MIMEText(html, "html"))
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.sendmail(FROM_EMAIL, to_email, msg.as_string())
