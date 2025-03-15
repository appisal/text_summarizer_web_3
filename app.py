import streamlit as st
from transformers import pipeline
import torch
from io import BytesIO
from gtts import gTTS
import pdfplumber
from docx import Document
from reportlab.pdfgen import canvas
import base64
import webbrowser

# GPU Check
device = 0 if torch.cuda.is_available() else -1

# Load Summarizer Model
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

# Generate PDF
def generate_pdf(summary):
    pdf_buffer = BytesIO()
    c = canvas.Canvas(pdf_buffer)
    c.drawString(100, 750, "Summary:")
    c.drawString(100, 730, summary)
    c.save()
    pdf_buffer.seek(0)
    return pdf_buffer

# Generate DOCX
def generate_docx(summary):
    doc_buffer = BytesIO()
    doc = Document()
    doc.add_heading("Summary", level=1)
    doc.add_paragraph(summary)
    doc.save(doc_buffer)
    doc_buffer.seek(0)
    return doc_buffer

# Convert text to speech
def generate_audio(summary):
    tts = gTTS(text=summary, lang="en")
    audio_buffer = BytesIO()
    tts.save(audio_buffer)
    audio_buffer.seek(0)
    return audio_buffer

# Create shareable links
def get_share_link(text, platform):
    encoded_text = base64.urlsafe_b64encode(text.encode()).decode()
    links = {
        "Facebook": f"https://www.facebook.com/sharer/sharer.php?u={encoded_text}",
        "Twitter": f"https://twitter.com/intent/tweet?text={encoded_text}",
        "LinkedIn": f"https://www.linkedin.com/shareArticle?mini=true&url={encoded_text}",
        "Email": f"mailto:?subject=Summary&body={encoded_text}"
    }
    return links.get(platform, "#")

# UI Setup
st.sidebar.title("⚡ Features")
option = st.sidebar.radio("Choose an option:", ["Single File", "Bulk File Processing", "Summary History"])

if "summary_history" not in st.session_state:
    st.session_state.summary_history = []

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
            st.session_state.summary_history.append(summary)
            
            st.markdown("<h3>📌 Summary:</h3>", unsafe_allow_html=True)
            st.success(summary)
            
            # Download options
            pdf_file = generate_pdf(summary)
            docx_file = generate_docx(summary)
            audio_file = generate_audio(summary)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.download_button("📄 Download PDF", pdf_file, "summary.pdf", "application/pdf")
            with col2:
                st.download_button("📝 Download DOCX", docx_file, "summary.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
            with col3:
                st.download_button("🔊 Download Audio", audio_file, "summary.mp3", "audio/mpeg")
            
            # Share options
            st.markdown("<h3>🔗 Share:</h3>", unsafe_allow_html=True)
            share_col1, share_col2, share_col3, share_col4 = st.columns(4)
            
            with share_col1:
                if st.button("📘 Facebook"):
                    webbrowser.open(get_share_link(summary, "Facebook"))
            with share_col2:
                if st.button("🐦 Twitter"):
                    webbrowser.open(get_share_link(summary, "Twitter"))
            with share_col3:
                if st.button("💼 LinkedIn"):
                    webbrowser.open(get_share_link(summary, "LinkedIn"))
            with share_col4:
                if st.button("📧 Email"):
                    webbrowser.open(get_share_link(summary, "Email"))

elif option == "Summary History":
    st.markdown("<h3>📜 Summary History</h3>", unsafe_allow_html=True)
    for idx, summary in enumerate(st.session_state.summary_history):
        st.write(f"**{idx+1}.** {summary}")

elif option == "Bulk File Processing":
    st.markdown("<h3>⚙️ Bulk File Processing (Coming Soon)</h3>", unsafe_allow_html=True)
