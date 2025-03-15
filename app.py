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
import pdfplumber
from langdetect import detect

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

# Session state for history
if "summary_history" not in st.session_state:
    st.session_state.summary_history = []

# Function to detect language
def detect_language(text):
    return detect(text)

# Summarization
def summarize_text(text, max_length, min_length):
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]["summary_text"]
    st.session_state.summary_history.append(summary)
    return summary

# PDF & DOCX Upload Support
def extract_text_from_pdf(uploaded_file):
    with pdfplumber.open(uploaded_file) as pdf:
        return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

def extract_text_from_docx(uploaded_file):
    doc = Document(uploaded_file)
    return "\n".join([para.text for para in doc.paragraphs])

# UI Setup
st.sidebar.title("⚡ Features")
option = st.sidebar.radio("Choose an option:", ["Single File", "Bulk File Processing", "Summary History"])

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
            summary = summarize_text(text, max_length, min_length)
            st.markdown("<h3>📌 Summary:</h3>", unsafe_allow_html=True)
            st.success(summary)
            
            # Download as PDF
            pdf_buffer = BytesIO()
            pdf = canvas.Canvas(pdf_buffer)
            pdf.drawString(100, 750, "Summary:")
            pdf.drawString(100, 730, summary)
            pdf.save()
            pdf_buffer.seek(0)
            st.download_button(label="📥 Download as PDF", data=pdf_buffer, file_name="summary.pdf", mime="application/pdf")
            
            # Download as Word
            doc = Document()
            doc.add_paragraph(summary)
            doc_buffer = BytesIO()
            doc.save(doc_buffer)
            doc_buffer.seek(0)
            st.download_button(label="📄 Download as Word", data=doc_buffer, file_name="summary.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
            
            # Convert to Audio
            tts = gTTS(summary)
            audio_buffer = BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            st.download_button(label="🔊 Download Audio", data=audio_buffer, file_name="summary.mp3", mime="audio/mpeg")
            
            # Share options
            encoded_summary = urllib.parse.quote(summary)
            whatsapp_url = f"https://api.whatsapp.com/send?text={encoded_summary}"
            twitter_url = f"https://twitter.com/intent/tweet?text={encoded_summary}"
            facebook_url = f"https://www.facebook.com/sharer/sharer.php?u={encoded_summary}"
            linkedin_url = f"https://www.linkedin.com/shareArticle?mini=true&url={encoded_summary}"
            
            st.markdown("<h3 style='text-align: center;'>📤 Share Summary:</h3>", unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"<a href='{whatsapp_url}' target='_blank'><img src='https://upload.wikimedia.org/wikipedia/commons/6/6b/WhatsApp.svg' width='40'></a>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"<a href='{twitter_url}' target='_blank'><img src='https://upload.wikimedia.org/wikipedia/en/6/60/Twitter_Logo_as_of_2021.svg' width='40'></a>", unsafe_allow_html=True)
            with col3:
                st.markdown(f"<a href='{facebook_url}' target='_blank'><img src='https://upload.wikimedia.org/wikipedia/commons/5/51/Facebook_f_logo_%282019%29.svg' width='40'></a>", unsafe_allow_html=True)
            with col4:
                st.markdown(f"<a href='{linkedin_url}' target='_blank'><img src='https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png' width='40'></a>", unsafe_allow_html=True)

# Bulk Processing & History remain unchanged
