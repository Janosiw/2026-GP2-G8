from flask import Flask, jsonify
from shared.firebase_config import db
from flask_cors import CORS

# إنشاء تطبيق Flask
app = Flask(__name__)
CORS(app)  # للسماح للواجهة بالاتصال مع الباك اند

# 🔹 اختبار الاتصال بـ Firebase Firestore
@app.route("/")
def home():
    try:
        # تجربة بسيطة للوصول إلى مجموعة Radiologists
        test_ref = db.collection("Radiologists").limit(1).get()
        return jsonify({
            "message": "Firebase connected successfully ✅",
            "records_found": len(test_ref)
        }), 200
    except Exception as e:
        return jsonify({
            "error": f"Failed to connect to Firebase ❌: {str(e)}"
        }), 500


# 🔹 تشغيل السيرفر
if __name__ == "__main__":
    app.run(debug=False, port=5000)
