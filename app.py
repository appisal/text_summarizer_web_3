import streamlit as st
from transformers import pipeline
import torch
from io import BytesIO
from gtts import gTTS
import pdfplumber
from docx import Document
from reportlab.pdfgen import canvas
import base64
import urllib.parse

# GPU Check
device = 0 if torch.cuda.is_available() else -1

# Load Summarization Model
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn", device=device)

summarizer = load_summarizer()

# Extract text from PDF
def extract_text_from_pdf(uploaded_file):
    with pdfplumber.open(uploaded_file) as pdf:
        return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

# Extract text from DOCX
def extract_text_from_docx(uploaded_file):
    doc = Document(uploaded_file)
    return "\n".join([para.text for para in doc.paragraphs])

# Generate downloadable link
def get_binary_file_downloader_link(bin_data, file_label, file_ext):
    b64 = base64.b64encode(bin_data).decode()
    href = f'<a href="data:file/{file_ext};base64,{b64}" download="summary.{file_ext}">📥 Download {file_label}</a>'
    return href

# Generate social media sharing buttons
def social_share_buttons(summary):
    encoded_summary = urllib.parse.quote(summary)
    facebook_url = f"https://www.facebook.com/sharer/sharer.php?u={encoded_summary}"
    twitter_url = f"https://twitter.com/intent/tweet?text={encoded_summary}"
    linkedin_url = f"https://www.linkedin.com/shareArticle?mini=true&url={encoded_summary}"
    email_url = f"mailto:?subject=Summarized Text&body={encoded_summary}"
    
    st.markdown(f"""
        <div style='text-align: center;'>
            <a href='{facebook_url}' target='_blank'><img src='https://img.icons8.com/fluency/48/facebook.png' alt='Facebook'></a>
            <a href='{twitter_url}' target='_blank'><img src='https://img.icons8.com/color/48/twitter--v1.png' alt='Twitter'></a>
            <a href='{linkedin_url}' target='_blank'><img src='https://img.icons8.com/color/48/linkedin.png' alt='LinkedIn'></a>
            <a href='{email_url}' target='_blank'><img src='https://img.icons8.com/color/48/gmail-new.png' alt='Email'></a>
        </div>
    """, unsafe_allow_html=True)

# Streamlit UI
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
            summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]["summary_text"]
            st.markdown("<h3>📌 Summary:</h3>", unsafe_allow_html=True)
            st.success(summary)
            
            # Download Options
            pdf_buffer = BytesIO()
            pdf = canvas.Canvas(pdf_buffer)
            pdf.drawString(100, 750, summary)
            pdf.save()
            st.markdown(get_binary_file_downloader_link(pdf_buffer.getvalue(), "as PDF", "pdf"), unsafe_allow_html=True)
            
            docx_buffer = BytesIO()
            doc = Document()
            doc.add_paragraph(summary)
            doc.save(docx_buffer)
            st.markdown(get_binary_file_downloader_link(docx_buffer.getvalue(), "as DOCX", "docx"), unsafe_allow_html=True)
            
            # Text-to-Speech Download
            tts = gTTS(summary)
            audio_buffer = BytesIO()
            tts.save(audio_buffer)
            st.markdown(get_binary_file_downloader_link(audio_buffer.getvalue(), "as MP3", "mp3"), unsafe_allow_html=True)
            
            # Social Media Share Buttons
            social_share_buttons(summary)
            
if option == "Summary History":
    st.markdown("<h3>📜 Summary History</h3>", unsafe_allow_html=True)
    if "summary_history" in st.session_state and st.session_state.summary_history:
        for idx, summary in enumerate(st.session_state.summary_history[::-1]):
            with st.expander(f"Summary {idx+1}"):
                st.write(summary)
    else:
        st.info("No summaries yet!")
