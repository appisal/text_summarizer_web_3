import streamlit as st
from transformers import pipeline
import torch
from io import BytesIO
from gtts import gTTS
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from keybert import KeyBERT
import time
import urllib.parse  
from reportlab.pdfgen import canvas
from docx import Document
from googletrans import Translator
import pdfplumber
from langdetect import detect
from docx import Document
from PIL import Image
import base64

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

# Function to create icon buttons
def create_icon_button(icon_url, link, tooltip):
    encoded_url = base64.b64encode(icon_url.getvalue()).decode()
    html_code = f'<a href="{link}" target="_blank"><img src="data:image/png;base64,{encoded_url}" width="40" title="{tooltip}" style="margin: 5px; cursor: pointer;"></a>'
    st.markdown(html_code, unsafe_allow_html=True)

# UI Setup
st.sidebar.title("⚡ Features")
option = st.sidebar.radio("Choose an option:", ["Single File", "Bulk File Processing", "Summary History"])

st.markdown("<h2 style='text-align: center;'>📖 Text Summarization & Analysis</h2>", unsafe_allow_html=True)

if option == "Single File":
    st.markdown("<h3>📂 Upload a file to summarize.</h3>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx"])
    text = ""
    
    if uploaded_file:
        if uploaded_file.type == "application/pdf":
            text = extract_text_from_pdf(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text = extract_text_from_docx(uploaded_file)
    
    if text.strip():
        max_length = st.slider("📏 Max summary length (words):", 50, 500, 200)
        min_length = st.slider("📏 Min summary length (words):", 10, 100, 50)
        
        if st.button("✨ Summarize", use_container_width=True):
            summary = summarize_multilang_text(text, max_length, min_length)
            st.markdown("<h3>📌 Summary:</h3>", unsafe_allow_html=True)
            st.success(summary)
            
            # Share Buttons - Icons as buttons
            st.markdown("<h3 style='text-align: center;'>🔗 Share Summary</h3>", unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                whatsapp_icon = open("whatsapp.png", "rb")
                create_icon_button(whatsapp_icon, f"https://api.whatsapp.com/send?text={urllib.parse.quote(summary)}", "Share on WhatsApp")
            with col2:
                twitter_icon = open("twitter.png", "rb")
                create_icon_button(twitter_icon, f"https://twitter.com/intent/tweet?text={urllib.parse.quote(summary)}", "Share on Twitter")
            with col3:
                linkedin_icon = open("linkedin.png", "rb")
                create_icon_button(linkedin_icon, f"https://www.linkedin.com/sharing/share-offsite/?url={urllib.parse.quote(summary)}", "Share on LinkedIn")

            # Download Buttons
            st.markdown("<h3 style='text-align: center;'>⬇️ Download Summary</h3>", unsafe_allow_html=True)
            
            col4, col5 = st.columns([1, 1])
            
            with col4:
                if st.button("📄 Download as PDF", use_container_width=True):
                    pdf_buffer = BytesIO()
                    c = canvas.Canvas(pdf_buffer)
                    c.drawString(100, 750, summary)
                    c.save()
                    st.download_button("Download PDF", pdf_buffer, file_name="summary.pdf", mime="application/pdf")
            
            with col5:
                if st.button("🔊 Download as Audio", use_container_width=True):
                    tts = gTTS(summary, lang='en')
                    audio_buffer = BytesIO()
                    tts.write_to_fp(audio_buffer)
                    st.download_button("Download Audio", audio_buffer, file_name="summary.mp3", mime="audio/mpeg")

# Bulk Processing & History remain unchanged

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
