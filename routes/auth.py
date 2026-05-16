import random
from datetime import datetime
from flask import render_template, session, redirect, url_for, request, jsonify
from firebase_admin import auth
from send_verification_email import send_verification_email

import shared
from utils import send_otp_email, _get_logged_doctor


@shared.app.route("/")
def index():
    return render_template("index.html")


@shared.app.route("/register_login")
def register_login():
    return render_template("register_login.html")


@shared.app.route("/api/signup", methods=["POST"])
def api_signup():
    import hashlib
    try:
        data     = request.get_json(force=True) or {}
        uid      = data.get("uid", "").strip()
        name     = data.get("FullName", "").strip()
        email    = data.get("Email", "").strip()
        password = data.get("Password", "").strip()
        phone    = data.get("ContactNumber", "").strip()
        pic      = data.get("ProfilePicture", "")

        if not uid or not email or not name:
            return jsonify({"error": "Missing required fields"}), 400

        hashed_pw = hashlib.sha256(password.encode()).hexdigest() if password else ""

        shared.db.collection("Radiologists").document(uid).set({
            "UID":              uid,
            "FullName":         name,
            "Email":            email,
            "ContactNumber":    phone,
            "PasswordHash":     hashed_pw,
            "ProfilePicture":   pic,
            "Specialty":        "",
            "Status":           "Active",
            "CreatedAt":        datetime.utcnow().isoformat(),
            "TwoFactorEnabled": False,
        })

        return jsonify({"status": "ok", "uid": uid}), 201

    except Exception as e:
        print("api_signup error:", e)
        return jsonify({"error": str(e)}), 500


@shared.app.route("/api/login", methods=["POST"])
def api_login():
    import requests as _req
    data = request.get_json(force=True) or {}
    email = (data.get("email") or "").strip()
    password = (data.get("password") or "").strip()

    if not email or not password:
        return jsonify({"status": "error", "message": "Email and password are required."}), 400

    sign_in_url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={shared.FIREBASE_API_KEY}"
    r = _req.post(sign_in_url, json={"email": email, "password": password, "returnSecureToken": True})

    if r.status_code != 200:
        err = r.json().get("error", {}).get("message", "")
        if "INVALID_PASSWORD" in err or "EMAIL_NOT_FOUND" in err or "INVALID_LOGIN_CREDENTIALS" in err:
            return jsonify({"status": "error", "message": "Incorrect email or password."}), 401
        return jsonify({"status": "error", "message": "Login failed. Please try again."}), 401

    firebase_user = r.json()
    uid = firebase_user.get("localId")

    # Always fetch live emailVerified status from Admin SDK (REST API response can be stale)
    try:
        admin_user = auth.get_user(uid)
        email_verified = admin_user.email_verified
    except Exception:
        email_verified = firebase_user.get("emailVerified", False)

    if not email_verified:
        return jsonify({"status": "error", "message": "Please verify your email first.", "redirect": "/verify"}), 403

    otp_code = str(random.randint(100000, 999999))
    session.clear()
    session["pending_uid"] = uid
    session["pending_email"] = email
    session["otp_code"] = otp_code

    try:
        send_otp_email(email, otp_code)
    except Exception as e:
        print("OTP email error:", e)
        return jsonify({"status": "error", "message": "Failed to send verification email. Please try again."}), 500

    return jsonify({"status": "ok", "message": "Verification code sent to your email."})


@shared.app.route("/api/verify-otp", methods=["POST"])
def api_verify_otp():
    data = request.get_json(force=True) or {}
    entered_otp = (data.get("otp") or "").strip()

    stored_otp = session.get("otp_code")
    uid = session.get("pending_uid")

    if not stored_otp or not uid:
        return jsonify({"status": "error", "message": "Session expired. Please login again."}), 400

    if entered_otp != stored_otp:
        return jsonify({"status": "error", "message": "Invalid or expired code. Try again."}), 400

    session.pop("otp_code", None)
    session.pop("pending_uid", None)
    session.pop("pending_email", None)
    session["radiologist_id"] = uid

    pending_invite = session.pop("_pending_invite_token", None)
    redirect_url = f"/accept_invite/{pending_invite}" if pending_invite else "/home"
    return jsonify({"status": "ok", "redirect": redirect_url})


@shared.app.route("/api/resend-otp", methods=["POST"])
def api_resend_otp():
    email = session.get("pending_email")
    if not email:
        return jsonify({"status": "error", "message": "Session expired. Please login again."}), 400

    otp_code = str(random.randint(100000, 999999))
    session["otp_code"] = otp_code

    try:
        send_otp_email(email, otp_code)
    except Exception as e:
        print("OTP resend error:", e)
        return jsonify({"status": "error", "message": "Failed to resend code."}), 500

    return jsonify({"status": "ok", "message": "New code sent to your email."})


@shared.app.route("/verify")
def verify():
    mode = request.args.get("mode")
    oob_code = request.args.get("oobCode")

    if mode == "verifyEmail":
        return render_template("verify.html")
    elif mode == "resetPassword":
        return render_template("reset_password.html", oob_code=oob_code)
    else:
        return render_template("verify.html")


@shared.app.route("/forget")
def forget():
    return render_template("forget.html")


@shared.app.route("/check_email")
def check_email():
    return render_template("check_email.html")


@shared.app.route("/login_from_firebase")
def login_from_firebase():
    uid = request.args.get("uid")
    if not uid:
        return "Missing UID", 400
    try:
        auth.get_user(uid)
    except Exception as e:
        return f"Invalid Firebase user: {str(e)}", 403

    session["radiologist_id"] = uid

    pending_invite = session.pop("_pending_invite_token", None)
    if pending_invite:
        return redirect(f"/accept_invite/{pending_invite}")
    return redirect(url_for("home"))


@shared.app.route("/dev-login")
def dev_login():
    docs = list(shared.db.collection("Radiologists").limit(1).stream())
    if not docs:
        return "No radiologist accounts found in database.", 404
    uid = docs[0].id
    session["radiologist_id"] = uid
    return redirect(url_for("home"))


@shared.app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))


@shared.app.route("/2FA_Prosses")
def twofa_prosses():
    return render_template("2FA_Prosses.html")


def get_verify_link(email):
    try:
        link = auth.generate_email_verification_link(email)
        return link
    except Exception as e:
        print("Error generating verify link:", e)
        raise Exception("Failed to generate Firebase verification link.")


@shared.app.route("/send_verification_email", methods=["POST"])
def send_verification_email_route():
    data = request.json
    email = data.get("email")
    name = data.get("name")

    firebase_link = auth.generate_email_verification_link(email)
    continue_url = "https://brainalyze.vercel.app/login_from_firebase"
    final_link = firebase_link + f"&continueUrl={continue_url}"

    send_verification_email(email, name, final_link)
    return {"status": "ok"}
