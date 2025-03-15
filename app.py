import streamlit as st
from transformers import pipeline
import torch
import pdfplumber
from docx import Document
from gtts import gTTS
import base64
import tempfile
import os
from reportlab.pdfgen import canvas
from urllib.parse import quote

# GPU Check
device = 0 if torch.cuda.is_available() else -1

# Load Summarization Model
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn", device=device)

summarizer = load_summarizer()

# Session state for history
if "summary_history" not in st.session_state:
    st.session_state.summary_history = []

# Extract text from PDF
def extract_text_from_pdf(uploaded_file):
    with pdfplumber.open(uploaded_file) as pdf:
        return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

# Extract text from DOCX
def extract_text_from_docx(uploaded_file):
    doc = Document(uploaded_file)
    return "\n".join([para.text for para in doc.paragraphs])

# Summarization Function
def summarize_text(text, max_length, min_length):
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]["summary_text"]
    st.session_state.summary_history.append(summary)
    return summary

# Convert text to audio
def text_to_audio(summary):
    tts = gTTS(summary, lang="en")
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_audio.name)
    return temp_audio.name

# Convert text to PDF
def text_to_pdf(summary):
    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    c = canvas.Canvas(temp_pdf.name)
    c.drawString(100, 750, "Summary:")
    c.drawString(100, 730, summary)
    c.save()
    return temp_pdf.name

# Convert text to DOCX
def text_to_docx(summary):
    temp_docx = tempfile.NamedTemporaryFile(delete=False, suffix=".docx")
    doc = Document()
    doc.add_paragraph(summary)
    doc.save(temp_docx.name)
    return temp_docx.name

# UI Setup
st.sidebar.title("⚡ Features")
option = st.sidebar.radio("Choose an option:", ["Single File", "Bulk File Processing", "Summary History"])

if option == "Single File":
    st.markdown("<h3>📂 Upload a PDF or DOCX file to summarize.</h3>", unsafe_allow_html=True)
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

            # Download Options
            col1, col2, col3 = st.columns(3)
            with col1:
                pdf_path = text_to_pdf(summary)
                with open(pdf_path, "rb") as f:
                    st.download_button("📄 Download PDF", f, file_name="summary.pdf", mime="application/pdf")

            with col2:
                docx_path = text_to_docx(summary)
                with open(docx_path, "rb") as f:
                    st.download_button("📝 Download DOCX", f, file_name="summary.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")

            with col3:
                audio_path = text_to_audio(summary)
                with open(audio_path, "rb") as f:
                    st.download_button("🔊 Download MP3", f, file_name="summary.mp3", mime="audio/mpeg")

            # WhatsApp Share Link
            whatsapp_message = f"Here's the summary: {summary}"
            whatsapp_url = f"https://wa.me/?text={quote(whatsapp_message)}"
            st.markdown(f"[📤 Share on WhatsApp]({whatsapp_url})")

elif option == "Summary History":
    st.markdown("<h3>📜 Summary History:</h3>", unsafe_allow_html=True)
    for i, hist_summary in enumerate(st.session_state.summary_history):
        with st.expander(f"Summary {i+1}"):
            st.write(hist_summary)

elif option == "Bulk File Processing":
    st.warning("⚡ Bulk processing is under development.")
