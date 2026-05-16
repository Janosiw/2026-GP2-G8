from flask import Flask
from firebase_admin import firestore, credentials
import firebase_admin
import os
from datetime import datetime, timedelta

# Saudi Arabia is UTC+3
_SA_OFFSET = timedelta(hours=3)

def now_sa() -> datetime:
    """Return current UTC time for storage in Firestore (naive datetime)."""
    return datetime.utcnow()

def fmt_sa(dt) -> str:
    """Format a datetime for display in Saudi Arabia time (UTC+3) → 'YYYY-MM-DD HH:MM'."""
    if dt is None:
        return "—"
    return (dt + _SA_OFFSET).strftime("%Y-%m-%d %H:%M")

def fmt_sa_verbose(dt) -> str:
    """Format a datetime for display in Saudi Arabia time (UTC+3) → 'DD Mon YYYY HH:MM'."""
    if dt is None:
        return "—"
    return (dt + _SA_OFFSET).strftime("%d %b %Y %H:%M")

app = Flask(__name__)
app.secret_key = "brainalyze-secret"

cred = credentials.Certificate("brainalyze-admin.json")
try:
    firebase_admin.get_app()
except ValueError:
    firebase_admin.initialize_app(cred)

db = firestore.client()

FIREBASE_API_KEY = "AIzaSyC5bb6M-sEVu9JL7mkVLFvkv44k8JIG9Es"
EPOCH = datetime(1970, 1, 1)

TUMOR_TYPE_OPTIONS = ["Glioma", "Meningioma", "Pituitary"]
TUMOR_TYPE_CANONICAL = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "pituitary": "Pituitary"
}

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ── Dashboard cache ──────────────────────────────────────────
_dashboard_cache: dict = {}
_DASH_TTL = 300  # seconds

# ── Pending scans (not yet committed to Firestore) ────────────
# Scans are stored here after analysis and only saved to Firestore
# when the doctor clicks "Finish Analysis".
_pending_scans: dict = {}

def clear_dash_cache(doctor_id: str | None = None):
    if doctor_id:
        _dashboard_cache.pop(doctor_id, None)
    else:
        _dashboard_cache.clear()
