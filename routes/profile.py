import os
import requests as _req
from flask import render_template, redirect, url_for, request, jsonify, flash
from firebase_admin import auth

import shared
from utils import _get_logged_doctor, _invalidate_doctor_cache


@shared.app.route("/profile", methods=["GET", "POST"])
def profile():
    doctor = _get_logged_doctor()
    if not doctor:
        return redirect(url_for("register_login"))

    doc_ref = shared.db.collection("Radiologists").document(doctor["id"])
    snap = doc_ref.get()
    data = snap.to_dict() or {}

    if request.method == "POST":
        updated = {
            "FullName": request.form.get("name", "").strip(),
            "Email": request.form.get("email", "").strip(),
            "ContactNumber": request.form.get("phone", "").strip(),
            "Specialty": request.form.get("specialty", "").strip(),
        }

        file = request.files.get("profile_pic")
        if file and file.filename.strip():
            filename = f"{doctor['id']}.jpg"
            path = os.path.join(shared.app.config["UPLOAD_FOLDER"], filename)
            file.save(path)
            updated["ProfilePicture"] = f"/static/uploads/{filename}"
        else:
            updated["ProfilePicture"] = data.get("ProfilePicture", "/static/images/user.png")

        doc_ref.update(updated)
        data.update(updated)
        _invalidate_doctor_cache()

        try:
            phone = updated["ContactNumber"]
            if phone and phone.startswith("+"):
                auth.update_user(doctor["id"], phone_number=phone)
        except Exception as e:
            print("Auth phone update Failed:", e)

        flash("Profile updated successfully!", "success")
        return redirect(url_for("profile"))

    doctor_ctx = {
        "name": data.get("FullName", ""),
        "email": data.get("Email", ""),
        "phone": data.get("ContactNumber", ""),
        "specialty": data.get("Specialty", ""),
        "ProfilePicture": data.get("ProfilePicture", "/static/images/user.png"),
    }

    return render_template("profile.html", doctor=doctor_ctx)


@shared.app.route("/profile/update_ajax", methods=["POST"])
def update_profile_ajax():
    doctor = _get_logged_doctor()
    if not doctor:
        return jsonify({"status": "error", "message": "Not logged in"}), 403

    doc_ref = shared.db.collection("Radiologists").document(doctor["id"])
    old_data = doc_ref.get().to_dict() or {}

    name = request.form.get("name", "").strip()
    email = request.form.get("email", "").strip()
    phone = request.form.get("phone", "").strip()
    specialty = request.form.get("specialty", "").strip()

    updated = {
        "FullName": name,
        "Email": email,
        "ContactNumber": phone,
        "Specialty": specialty
    }

    file = request.files.get("profile_pic")
    if file and file.filename.strip():
        filename = f"{doctor['id']}.jpg"
        upload_path = os.path.join(shared.app.config["UPLOAD_FOLDER"], filename)
        file.save(upload_path)
        updated["ProfilePicture"] = f"/static/uploads/{filename}"
    else:
        updated["ProfilePicture"] = old_data.get("ProfilePicture", "/static/images/user.png")

    doc_ref.update(updated)
    _invalidate_doctor_cache()

    return jsonify({
        "status": "success",
        "message": "Profile updated successfully!",
        "updated_data": updated
    })


@shared.app.route("/profile/change_password_admin", methods=["POST"])
def change_password_admin():
    data = request.get_json(force=True) or {}

    id_token = (data.get("idToken") or "").strip()
    new_password = (data.get("new_password") or "").strip()

    if not id_token or not new_password:
        return jsonify({"status": "error", "message": "Missing required fields."}), 400

    if len(new_password) < 8:
        return jsonify({"status": "error", "message": "New password must be at least 8 characters."}), 400

    try:
        decoded = auth.verify_id_token(id_token)
        uid = decoded.get("uid")

        return jsonify({"status": "success", "message": "Password updated successfully!"}), 200
        session_uid = session.get("radiologist_id")
        if session_uid and session_uid != uid:
            return jsonify({"status": "error", "message": "Unauthorized."}), 403

    except auth.ExpiredIdTokenError:
        return jsonify({"status": "error", "message": "Session expired. Please login again."}), 401
    except auth.InvalidIdTokenError:
        return jsonify({"status": "error", "message": "Invalid token. Please login again."}), 401
    except Exception as e:
        print("Error in change_password_admin:", e)
        return jsonify({"status": "error", "message": "Failed to update password."}), 500

    doctor = _get_logged_doctor()
    if not doctor:
        return jsonify({"status": "error", "message": "Not logged in"}), 403

    data2 = request.get_json(force=True) or {}
    current_password = (data2.get("current_password") or "").strip()
    new_password2 = (data2.get("new_password") or "").strip()

    if len(new_password2) < 8:
        return jsonify({"status": "error", "message": "New password must be at least 8 characters."}), 400

    email = doctor.get("email")
    if not email:
        return jsonify({"status": "error", "message": "Missing email."}), 400

    sign_in_url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={shared.FIREBASE_API_KEY}"
    r1 = _req.post(sign_in_url, json={
        "email": email,
        "password": current_password,
        "returnSecureToken": True
    })

    if r1.status_code != 200:
        return jsonify({"status": "error", "message": "Current password is incorrect."}), 400

    id_token2 = r1.json().get("idToken")
    if not id_token2:
        return jsonify({"status": "error", "message": "Failed to get token."}), 500

    update_url = f"https://identitytoolkit.googleapis.com/v1/accounts:update?key={shared.FIREBASE_API_KEY}"
    r2 = _req.post(update_url, json={
        "idToken": id_token2,
        "password": new_password2,
        "returnSecureToken": True
    })

    if r2.status_code != 200:
        return jsonify({"status": "error", "message": "Failed to update password."}), 500

    return jsonify({"status": "success", "message": "Password updated successfully!"})
