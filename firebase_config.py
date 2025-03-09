import firebase_admin
from firebase_admin import credentials, auth, firestore

# Firebase Config - Replace with your Firebase Service Account JSON
firebase_config = {
    apiKey: "AIzaSyB-gvbrjxROoeR1hkQP1pyynAOs1A9d0mo",
  authDomain: "textsummarizer-fdb75.firebaseapp.com",
  projectId: "textsummarizer-fdb75",
  storageBucket: "textsummarizer-fdb75.firebasestorage.app",
  messagingSenderId: "547653980696",
  appId: "1:547653980696:web:5a8d43fdefe0688a88696d",
  measurementId: "G-3HKV85MM0H"
}

# Initialize Firebase
if not firebase_admin._apps:
    cred = credentials.Certificate(firebase_config)
    firebase_admin.initialize_app(cred)

# Firestore Database
db = firestore.client()

# Function to save summary
def save_summary(user_email, summary_text):
    doc_ref = db.collection("summaries").document(user_email)
    doc_ref.set({"summary": summary_text}, merge=True)

# Function to retrieve summaries
def get_summaries(user_email):
    doc_ref = db.collection("summaries").document(user_email)
    doc = doc_ref.get()
    return doc.to_dict().get("summary", "No summary found") if doc.exists else "No data found."
