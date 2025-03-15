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

# GPU Check
device = 0 if torch.cuda.is_available() else -1

# Load Models
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn", device=device)

summarizer = load_summarizer()
keyword_extractor = KeyBERT()

# Session state for history
if "summary_history" not in st.session_state:
    st.session_state.summary_history = []

# Summarization Function
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

# File Download Function
def download_summary_as_pdf(summary):
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer)
    pdf.drawString(100, 750, "Summarized Text")
    pdf.drawString(100, 730, summary)
    pdf.save()
    buffer.seek(0)
    return buffer

def download_summary_as_docx(summary):
    buffer = BytesIO()
    doc = Document()
    doc.add_paragraph(summary)
    doc.save(buffer)
    buffer.seek(0)
    return buffer

def download_summary_as_audio(summary):
    tts = gTTS(summary, lang='en')
    buffer = BytesIO()
    tts.write_to_fp(buffer)
    buffer.seek(0)
    return buffer

# UI Setup
st.sidebar.title("⚡ Features")
option = st.sidebar.radio("Choose an option:", ["Single File", "Summary History"])

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

            # Download Buttons
            col1, col2, col3 = st.columns(3)
            with col1:
                st.download_button("📄 Download PDF", data=download_summary_as_pdf(summary), file_name="summary.pdf", mime="application/pdf")
            with col2:
                st.download_button("📃 Download DOCX", data=download_summary_as_docx(summary), file_name="summary.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
            with col3:
                st.download_button("🔊 Download Audio", data=download_summary_as_audio(summary), file_name="summary.mp3", mime="audio/mp3")

            # Share on Social Media
            st.markdown("<h3>📢 Share Summary:</h3>", unsafe_allow_html=True)
            share_text = urllib.parse.quote(summary)
            whatsapp_url = f"https://api.whatsapp.com/send?text={share_text}"
            twitter_url = f"https://twitter.com/intent/tweet?text={share_text}"
            linkedin_url = f"https://www.linkedin.com/sharing/share-offsite/?url={share_text}"
            facebook_url = f"https://www.facebook.com/sharer/sharer.php?u={share_text}"
            email_url = f"mailto:?subject=Shared Summary&body={share_text}"

            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.markdown(f'<a href="{whatsapp_url}" target="_blank"><img src="https://upload.wikimedia.org/wikipedia/commons/6/6b/WhatsApp.svg" width="40"></a>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<a href="{twitter_url}" target="_blank"><img src="https://upload.wikimedia.org/wikipedia/en/6/60/Twitter_Logo_as_of_2021.svg" width="40"></a>', unsafe_allow_html=True)
            with col3:
                st.markdown(f'<a href="{linkedin_url}" target="_blank"><img src="https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png" width="40"></a>', unsafe_allow_html=True)
            with col4:
                st.markdown(f'<a href="{facebook_url}" target="_blank"><img src="https://upload.wikimedia.org/wikipedia/commons/1/1b/Facebook_icon.svg" width="40"></a>', unsafe_allow_html=True)
            with col5:
                st.markdown(f'<a href="{email_url}" target="_blank"><img src="https://upload.wikimedia.org/wikipedia/commons/4/4e/Mail_%28iOS%29.svg" width="40"></a>', unsafe_allow_html=True)

# Summary History
elif option == "Summary History":
    st.markdown("<h3>📜 Summary History:</h3>", unsafe_allow_html=True)
    for i, summary in enumerate(st.session_state.summary_history[::-1]):
        st.markdown(f"**{i+1}.** {summary}")
