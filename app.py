import streamlit as st
from transformers import pipeline
import torch
from io import BytesIO
from gtts import gTTS
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from keybert import KeyBERT
import urllib.parse  
from reportlab.pdfgen import canvas
from docx import Document
import pdfplumber
from langdetect import detect
from googletrans import Translator
import time

# GPU Check
device = 0 if torch.cuda.is_available() else -1

# Load Models
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn", device=device)

@st.cache_resource
def load_sentiment_analyzer():
    return pipeline("sentiment-analysis", device=device)

summarizer = load_summarizer()
sentiment_analyzer = load_sentiment_analyzer()
keyword_extractor = KeyBERT()
translator = Translator()

# Session state for history
if "summary_history" not in st.session_state:
    st.session_state.summary_history = []

# Function to detect language
def detect_language(text):
    return detect(text)

# Multi-Language Summarization
def summarize_multilang_text(text, max_length, min_length):
    lang = detect_language(text)
    if lang != "en":
        text = translator.translate(text, src=lang, dest="en").text
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]["summary_text"]
    if lang != "en":
        summary = translator.translate(summary, src="en", dest=lang).text
    st.session_state.summary_history.append(summary)
    return summary

# PDF & DOCX Upload Support
def extract_text_from_pdf(uploaded_file):
    with pdfplumber.open(uploaded_file) as pdf:
        return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

def extract_text_from_docx(uploaded_file):
    doc = Document(uploaded_file)
    return "\n".join([para.text for para in doc.paragraphs])

# Function to download summary as PDF
def download_pdf(summary):
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer)
    pdf.drawString(100, 750, "Summary Report")
    pdf.drawString(100, 730, summary)
    pdf.save()
    buffer.seek(0)
    return buffer

# Function to generate sharing links
def generate_share_links(summary):
    encoded_summary = urllib.parse.quote(summary)
    return {
        "WhatsApp": f"https://wa.me/?text={encoded_summary}",
        "Twitter": f"https://twitter.com/intent/tweet?text={encoded_summary}",
        "Email": f"mailto:?subject=Summary&body={encoded_summary}",
        "LinkedIn": f"https://www.linkedin.com/sharing/share-offsite/?url={encoded_summary}"
    }

# Function to create animated share buttons
def create_share_buttons(summary):
    share_links = generate_share_links(summary)
    share_html = f"""
    <style>
        .share-btns {{
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-top: 20px;
        }}
        .share-btns a img {{
            width: 50px;
            height: 50px;
            transition: transform 0.3s ease-in-out;
        }}
        .share-btns a img:hover {{
            transform: scale(1.2) rotate(10deg);
        }}
    </style>
    <div class="share-btns">
        <a href="{share_links['WhatsApp']}" target="_blank">
            <img src="https://upload.wikimedia.org/wikipedia/commons/6/6b/WhatsApp.svg">
        </a>
        <a href="{share_links['Twitter']}" target="_blank">
            <img src="https://upload.wikimedia.org/wikipedia/en/6/60/Twitter_Logo_as_of_2021.svg">
        </a>
        <a href="{share_links['Email']}" target="_blank">
            <img src="https://upload.wikimedia.org/wikipedia/commons/4/4e/Mail_%28iOS%29.svg">
        </a>
        <a href="{share_links['LinkedIn']}" target="_blank">
            <img src="https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png">
        </a>
    </div>
    """
    st.markdown(share_html, unsafe_allow_html=True)

# UI Setup with Animation
st.markdown("""
    <style>
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        h1 {
            text-align: center;
            animation: fadeIn 1s ease-in-out;
        }
        .stButton>button {
            transition: transform 0.3s ease-in-out;
        }
        .stButton>button:hover {
            transform: scale(1.1);
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>🚀 AI-Powered Text Summarizer</h1>", unsafe_allow_html=True)

st.sidebar.title("⚡ Features")
option = st.sidebar.radio("Choose an option:", ["Single File", "Bulk File Processing", "Summary History"])

if option == "Single File":
    st.markdown("<h3>📂 Upload a file to summarize.</h3>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx"])
    text = ""
    if uploaded_file:
        text = extract_text_from_pdf(uploaded_file) if uploaded_file.type == "application/pdf" else extract_text_from_docx(uploaded_file)
    if text.strip():
        max_length = st.slider("📏 Max summary length (words):", 50, 500, 200)
        min_length = st.slider("📏 Min summary length (words):", 10, 100, 50)
        if st.button("✨ Summarize", use_container_width=True):
            summary = summarize_multilang_text(text, max_length, min_length)
            st.success(summary)
            create_share_buttons(summary)

st.markdown("<hr><p style='text-align: center;'>🔗 AI-Powered Summarizer</p>", unsafe_allow_html=True)
