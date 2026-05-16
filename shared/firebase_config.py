import firebase_admin
from firebase_admin import credentials, firestore, auth

cred = credentials.Certificate("brainalyze-admin.json")
firebase_api_key = "AIzaSyC5bb6M-sEVu9JL7mkVLFvkv44k8JIG9Es"

try:
    firebase_admin.get_app()
except ValueError:
    firebase_admin.initialize_app(cred)

db = firestore.client()


