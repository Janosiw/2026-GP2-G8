from flask import Blueprint, request, jsonify
from shared.firebase_config import db
import hashlib
from firebase_admin import auth  # إدارة مستخدمي Firebase
import requests  # للتحقق من تسجيل الدخول من Firebase

# إنشاء Blueprint خاص بمسارات التسجيل والدخول
auth_bp = Blueprint("auth_bp", __name__)

# ==========================================================
# 🔹 التسجيل (Sign Up)
# ==========================================================
@auth_bp.route("/api/signup", methods=["POST"])
def signup():
    try:
        data = request.get_json()

        full_name = data.get("FullName")
        email = data.get("Email")
        password = data.get("Password")
        contact = data.get("ContactNumber", "")
        specialty = data.get("Specialty", "")
        profile_pic = data.get("ProfilePicture", "")
        status = data.get("Status", "Active")
        two_factor = data.get("TwoFactorEnabled", False)

        if not full_name or not email or not password:
            return jsonify({"error": "Missing required fields"}), 400

        doctors_ref = db.collection("Radiologists")

        # ✅ التحقق من البريد الإلكتروني المكرر داخل Firestore
        existing = doctors_ref.where("Email", "==", email).get()
        if existing:
            return jsonify({"error": "Email already registered"}), 400

        # ✅ إنشاء المستخدم داخل Firebase Authentication
        try:
            user_record = auth.create_user(
                email=email,
                password=password,
                display_name=full_name,
                photo_url=profile_pic if profile_pic else None,
                disabled=False
            )
        except Exception as e:
            return jsonify({"error": f"Firebase Auth error: {str(e)}"}), 400

        # ✅ تشفير كلمة المرور
        hashed_pw = hashlib.sha256(password.encode()).hexdigest()

        # ✅ تخزين بيانات المستخدم في Firestore
        user_data = {
            "UID": user_record.uid,
            "FullName": full_name,
            "Email": email,
            "ContactNumber": contact,
            "Password": hashed_pw,
            "Specialty": specialty,
            "ProfilePicture": profile_pic,
            "Status": status,
            "TwoFactorEnabled": two_factor
        }

        doctors_ref.document(user_record.uid).set(user_data)

        return jsonify({
            "message": "Radiologist registered successfully ✅",
            "uid": user_record.uid
        }), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==========================================================
# 🔹 تسجيل الدخول (Log In)
# ==========================================================
@auth_bp.route("/api/login", methods=["POST"])
def login():
    try:
        data = request.get_json()
        email = data.get("email")
        password = data.get("password")

        if not email or not password:
            return jsonify({"error": "Missing email or password"}), 400

        # ✅ تحقق فعلي من الباسوورد في Firebase Authentication عبر REST API
        firebase_api_key = "AIzaSyC5bb6M-sEVu9JL7mkVLFvkv44k8JIG9Es"  # مفتاح مشروعك
        url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={firebase_api_key}"

        payload = {
            "email": email,
            "password": password,
            "returnSecureToken": True
        }

        response = requests.post(url, json=payload)
        result = response.json()

        # 🔸 التحقق من نتيجة تسجيل الدخول
        if "error" in result:
            message = result["error"]["message"]
            if message == "EMAIL_NOT_FOUND":
                return jsonify({"error": "Email not found"}), 404
            elif message == "INVALID_PASSWORD":
                return jsonify({"error": "Incorrect password"}), 401
            else:
                return jsonify({"error": message}), 400

        # ✅ تسجيل الدخول ناجح من Firebase
        doctors_ref = db.collection("Radiologists")
        users = doctors_ref.where("Email", "==", email).get()

        hashed_pw = hashlib.sha256(password.encode()).hexdigest()

        # ✅ تحديث كلمة المرور في Firestore إن تغيّرت بعد Reset
        for user in users:
            user_data = user.to_dict()
            if user_data.get("Password") != hashed_pw:
                doctors_ref.document(user.id).update({"Password": hashed_pw})

            return jsonify({
                "message": f"Welcome back {user_data['FullName']} 👋",
                "uid": user_data.get("UID"),
                "email": email
            }), 200

        # في حال المستخدم موجود في Auth لكن مو مضاف في Firestore
        return jsonify({
            "message": "Account exists in Firebase but not in Firestore.",
            "email": email
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==========================================================
# 🔹 تحديث كلمة المرور (Update Password)
# ==========================================================
@auth_bp.route("/update-password", methods=["POST"])
def update_password():
    try:
        data = request.get_json()
        email = data.get("email")
        new_password = data.get("new_password")

        if not email or not new_password:
            return jsonify({"error": "Missing email or new password"}), 400

        doctors_ref = db.collection("Radiologists")
        query = doctors_ref.where("Email", "==", email).get()

        if not query:
            return jsonify({"error": "Doctor not found in database"}), 404

        hashed_pw = hashlib.sha256(new_password.encode()).hexdigest()

        for doc in query:
            uid = doc.to_dict().get("UID")

            # ✅ تحديث كلمة المرور في Firestore
            doctors_ref.document(doc.id).update({"Password": hashed_pw})

            # ✅ تحديث كلمة المرور في Firebase Auth
            if uid:
                auth.update_user(uid, password=new_password)

        return jsonify({"message": "Password updated successfully ✅"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
