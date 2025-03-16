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

# Function to download summary as Word
def download_word(summary):
    doc = Document()
    doc.add_paragraph(summary)
    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer

# Function to download summary as audio
def download_audio(summary):
    tts = gTTS(summary, lang="en")
    buffer = BytesIO()
    tts.write_to_fp(buffer)
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

# Function to create share buttons with icons
def create_share_buttons(summary):
    share_links = generate_share_links(summary)
    share_html = f"""
    <style>
        .share-btns {{
            display: flex;
            justify-content: center;
            gap: 15px;
            margin-top: 20px;
        }}
        .share-btns a img {{
            width: 50px;
            height: 50px;
            transition: transform 0.3s ease-in-out;
        }}
        .share-btns a img:hover {{
            transform: scale(1.2);
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

# UI Setup
st.markdown("<h1 style='text-align: center;'>🚀 AI-Powered Text Summarizer</h1>", unsafe_allow_html=True)

st.sidebar.title("⚡ Features")
option = st.sidebar.radio("Choose an option:", ["Single File", "Bulk File Processing", "Summary History"])

if option == "Single File":
    st.markdown("<h3>📂 Upload a file or paste text to summarize.</h3>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx"])
    text = ""

    if uploaded_file:
        if uploaded_file.type == "application/pdf":
            text = extract_text_from_pdf(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text = extract_text_from_docx(uploaded_file)
    else:
    # Initialize session state for text if not present
    if "input_text" not in st.session_state:
        st.session_state.input_text = ""

    # Text Area with session state
    text = st.text_area("✍️ Paste your text here:", value=st.session_state.input_text, height=200, key="input_text")

    # Clear Text Button
    if st.button("❌ Clear Text"):
        st.session_state.input_text = ""  # Reset text
        st.rerun()  # Refresh the app



    if text.strip():
        max_length = st.slider("📏 Max summary length (words):", 50, 500, 200)
        min_length = st.slider("📏 Min summary length (words):", 10, 100, 50)

        if st.button("✨ Summarize", use_container_width=True):
            summary = summarize_multilang_text(text, max_length, min_length)
            st.markdown("<h3>📌 Summary:</h3>", unsafe_allow_html=True)
            st.success(summary)

            # Download Buttons
            col1, col2, col3 = st.columns(3)
            with col1:
                st.download_button("📄 PDF", download_pdf(summary), file_name="summary.pdf", mime="application/pdf")
            with col2:
                st.download_button("📝 Word", download_word(summary), file_name="summary.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
            with col3:
                st.download_button("🔊 Audio", download_audio(summary), file_name="summary.mp3", mime="audio/mp3")

            # Share Buttons
            st.markdown("<h3 style='text-align: center;'>📢 Share this Summary</h3>", unsafe_allow_html=True)
            create_share_buttons(summary)

elif option == "Bulk File Processing":
    uploaded_files = st.file_uploader("Upload multiple files", type=["pdf", "docx"], accept_multiple_files=True)
    if uploaded_files:
        for file in uploaded_files:
            text = extract_text_from_pdf(file) if file.type == "application/pdf" else extract_text_from_docx(file)
            summary = summarize_multilang_text(text, 200, 50)
            st.markdown(f"### 📜 Summary for {file.name}")
            st.success(summary)

elif option == "Summary History":
    st.subheader("📜 Summary History")
    for i, summary in enumerate(reversed(st.session_state.summary_history)):
        with st.expander(f"📄 Summary {len(st.session_state.summary_history) - i}"):
            st.write(summary)
            create_share_buttons(summary)

st.markdown("<hr><p style='text-align: center;'>🔗 AI-Powered Summarizer</p>", unsafe_allow_html=True)
