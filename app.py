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

# GPU Check
device = 0 if torch.cuda.is_available() else -1

# Load models
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

# Dark Mode Toggle
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

# Function to summarize text
def summarize_text(text, max_length, min_length):
    if len(text.split()) < min_length:
        return "Input text is too short to summarize."
    
    with st.spinner("🔄 Summarizing... Please wait."):
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)

    st.session_state.summary_history.append(summary[0]["summary_text"])
    return summary[0]["summary_text"]

# Function to download all summaries as PDF
def download_all_summaries_pdf():
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer)
    pdf.drawString(100, 800, "Summary History Report")
    y = 780
    for summary in st.session_state.summary_history:
        pdf.drawString(100, y, summary)
        y -= 20
    pdf.save()
    buffer.seek(0)
    return buffer

# Function to download all summaries as Word
def download_all_summaries_word():
    doc = Document()
    doc.add_heading("Summary History Report", level=1)
    for summary in st.session_state.summary_history:
        doc.add_paragraph(summary)
    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer

# Streamlit App Header
st.markdown("""
    <h1 style='text-align: center; color: #FF4B4B; font-size: 3rem;'>
        🚀 AI Text Summarizer
    </h1>
    <hr style='border-top: 3px solid #FF4B4B;'>
""", unsafe_allow_html=True)

# Sidebar with Logo & Dark Mode Toggle
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/a/a7/React-icon.svg", width=100)
st.sidebar.title("⚡ Features")

# Dark Mode Toggle
if st.sidebar.checkbox("🌙 Dark Mode", st.session_state.dark_mode):
    st.session_state.dark_mode = True
    st.markdown("<style>body { background-color: #333; color: white; }</style>", unsafe_allow_html=True)
else:
    st.session_state.dark_mode = False

option = st.sidebar.radio("Choose an option:", ["Single File", "Bulk Upload", "Summary History"])

# Single File Summarization
if option == "Single File":
    st.markdown("<h3 style='color: #333;'>📂 Upload a text file or paste text below to summarize it.</h3>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a .txt file", type=["txt"])
    text = uploaded_file.read().decode("utf-8") if uploaded_file else st.text_area("✍️ Paste your text here:", height=200)

    if text.strip():
        max_length = st.slider("📏 Max summary length (words):", 50, 500, 200)
        min_length = st.slider("📏 Min summary length (words):", 10, 100, 50)

        if st.button("✨ Summarize", use_container_width=True):
            summary = summarize_text(text, max_length, min_length)
            if summary:
                st.markdown("<h3 style='color: #FF4B4B;'>📌 Summary:</h3>", unsafe_allow_html=True)
                st.success(summary)

# Bulk File Summarization
elif option == "Bulk Upload":
    st.subheader("📂 Upload multiple text files for summarization")
    uploaded_files = st.file_uploader("Choose .txt files", type=["txt"], accept_multiple_files=True)
    
    if uploaded_files and st.button("✨ Summarize All"):
        summaries = []
        for file in uploaded_files:
            text = file.read().decode("utf-8")
            summary = summarize_text(text, 200, 50)
            summaries.append(summary)
        
        st.subheader("📌 Summaries:")
        for i, summary in enumerate(summaries):
            st.write(f"### 📄 File {i+1} Summary:")
            st.success(summary)

# Summary History & Export Options
elif option == "Summary History":
    st.subheader("📜 Summary History")
    if st.session_state.summary_history:
        for i, summary in enumerate(reversed(st.session_state.summary_history)):
            with st.expander(f"📄 Summary {len(st.session_state.summary_history) - i}"):
                st.write(summary)

        # Download & Clear History Buttons
        st.markdown("### ⬇️ Export Options:")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button("📄 Download All as PDF", download_all_summaries_pdf(), file_name="summaries.pdf", mime="application/pdf")
        with col2:
            st.download_button("📝 Download All as Word", download_all_summaries_word(), file_name="summaries.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        with col3:
            if st.button("🗑️ Clear History"):
                st.session_state.summary_history = []
                st.experimental_rerun()
    else:
        st.write("🔍 No previous summaries found.")

# Footer
st.markdown("""
    <hr>
    <p style='text-align: center; font-size: 14px; color: #555;'>
        © 2025 Text Summarizer AI | Built with ❤️ using Streamlit
    </p>
""", unsafe_allow_html=True)
