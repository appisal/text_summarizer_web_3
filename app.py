import streamlit as st
from transformers import pipeline
import torch
import time
import firebase_config  # Import Firebase Module
import streamlit_authenticator as stauth
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from docx import Document
from gtts import gTTS
from keybert import KeyBERT
from bs4 import BeautifulSoup
import requests
from zipfile import ZipFile

# Check for GPU availability
device = 0 if torch.cuda.is_available() else -1

# Load summarizer and sentiment analysis pipelines
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn", device=device)

@st.cache_resource
def load_sentiment_analyzer():
    return pipeline("sentiment-analysis", device=device)

summarizer = load_summarizer()
sentiment_analyzer = load_sentiment_analyzer()
keyword_extractor = KeyBERT()

# Initialize session state for history
if "summary_history" not in st.session_state:
    st.session_state.summary_history = []

# Function to summarize text with progress indicator
def summarize_text(text, max_length, min_length):
    if len(text.split()) < min_length:
        return "Input text is too short to summarize."

    progress_bar = st.progress(0)
    for percent in range(1, 101, 10):
        time.sleep(0.05)
        progress_bar.progress(percent)

    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    progress_bar.empty()
    
    # Save to history
    st.session_state.summary_history.append(summary[0]["summary_text"])
    
    return summary[0]["summary_text"]

# Function to extract text from a URL
def extract_text_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        paragraphs = soup.find_all("p")
        return " ".join([p.get_text() for p in paragraphs]) if paragraphs else "No readable content found."
    except requests.exceptions.RequestException as e:
        return f"Error fetching URL: {e}"

# Functions to create downloadable files
def create_pdf(summary):
    buffer = BytesIO()
    pdf_canvas = canvas.Canvas(buffer, pagesize=letter)
    pdf_canvas.setFont("Helvetica", 12)
    pdf_canvas.drawString(30, 750, "Summary:")
    text_obj = pdf_canvas.beginText(30, 730)
    text_obj.setFont("Helvetica", 10)
    for line in summary.split("\n"):
        text_obj.textLine(line)
    pdf_canvas.drawText(text_obj)
    pdf_canvas.save()
    buffer.seek(0)
    return buffer

def create_txt(summary):
    return BytesIO(summary.encode())

def create_docx(summary):
    doc = Document()
    doc.add_heading("Summary", level=1)
    doc.add_paragraph(summary)
    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer

def text_to_speech(summary):
    tts = gTTS(summary, lang="en")
    buffer = BytesIO()
    tts.write_to_fp(buffer)
    buffer.seek(0)
    return buffer

# Streamlit App
st.title("Text Summarizer - Enhanced 🚀")

# Authentication
users = {"user1@example.com": {"name": "User One", "password": "password123"}}

authenticator = stauth.Authenticate(users, "app_cookie", "random_key", cookie_expiry_days=1)

name, authentication_status, username = authenticator.login("Login", "main")

if authentication_status:
    st.sidebar.success(f"Welcome, {name}!")
    if st.button("Logout"):
        authenticator.logout("Logout", "sidebar")

elif authentication_status is False:
    st.error("Username/password incorrect")

elif authentication_status is None:
    st.warning("Please enter your credentials")

# Sidebar
st.sidebar.title("Features")
option = st.sidebar.radio(
    "Choose an option:",
    ["Single File", "Multiple Files", "URL", "Compare Texts", "Summary History"]
)

# Single File Summarization
if option == "Single File" and authentication_status:
    st.write("Upload a text file or paste text below to summarize it.")

    uploaded_file = st.file_uploader("Choose a .txt file", type=["txt"])
    if uploaded_file:
        text = uploaded_file.read().decode("utf-8")
    else:
        text = st.text_area("Paste your text here:", height=200)

    if text.strip():
        st.write(f"📝 Word Count: {len(text.split())}")
        st.write(f"🔢 Character Count: {len(text)}")

        max_length = st.slider("Max summary length (words):", 50, 500, 200)
        min_length = st.slider("Min summary length (words):", 10, 100, 50)

        if st.button("Summarize ✨"):
            summary = summarize_text(text, max_length, min_length)
            st.subheader("📌 Summary:")
            st.write(summary)

            sentiment = sentiment_analyzer(summary)[0]
            st.write(f"📊 Sentiment: **{sentiment['label']}** (Confidence: {sentiment['score']:.2f})")

            keywords = keyword_extractor.extract_keywords(summary, top_n=5)
            st.write("🔑 Keywords:", ", ".join([word for word, _ in keywords]))

            # Save summary in Firestore
            firebase_config.save_summary(username, summary)

            # Retrieve previous summaries
            st.subheader("📜 Your Previous Summaries:")
            st.write(firebase_config.get_summaries(username))

# Summary History
elif option == "Summary History" and authentication_status:
    st.subheader("📜 Summary History")
    if st.session_state.summary_history:
        for i, summary in enumerate(reversed(st.session_state.summary_history)):
            with st.expander(f"Summary {len(st.session_state.summary_history) - i}"):
                st.write(summary)
    else:
        st.write("No previous summaries found.")

