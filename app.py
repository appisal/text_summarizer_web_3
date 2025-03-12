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

# Custom CSS (Navbar + Sidebar Styling)
st.markdown("""
    <style>
        .navbar {
            display: flex;
            justify-content: center;
            gap: 30px;
            background: #ff7eb3;
            padding: 15px;
            border-radius: 15px;
            box-shadow: 0px 5px 15px rgba(0, 0, 0, 0.2);
            margin-bottom: 20px;
        }
        .navbar a {
            text-decoration: none;
            font-size: 18px;
            font-weight: bold;
            color: white;
            cursor: pointer;
        }
        .navbar a:hover {
            color: #ffebf0;
            transform: scale(1.05);
        }
    </style>
""", unsafe_allow_html=True)

# Render the Navbar
st.markdown("""
    <div class="navbar">
        <a onclick="window.location.href='#features'">⚡ Features</a>
    </div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("⚡ Features")
option = st.sidebar.radio("Choose an option:", ["Single File", "Bulk File Processing", "Summary History"])

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

if "summary_history" not in st.session_state:
    st.session_state.summary_history = []

def summarize_text(text, max_length, min_length):
    if len(text.split()) < min_length:
        return "Input text is too short to summarize."
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    st.session_state.summary_history.append(summary[0]["summary_text"])
    return summary[0]["summary_text"]

def generate_share_links(summary):
    encoded_summary = urllib.parse.quote(summary)
    return {
        "WhatsApp": f"https://wa.me/?text={encoded_summary}",
        "Twitter": f"https://twitter.com/intent/tweet?text={encoded_summary}",
        "Email": f"mailto:?subject=Summary&body={encoded_summary}",
        "LinkedIn": f"https://www.linkedin.com/sharing/share-offsite/?url={encoded_summary}"
    }

def create_share_buttons(summary):
    share_links = generate_share_links(summary)
    st.markdown(f"""
    <div>
        <a href="{share_links['WhatsApp']}" target="_blank">📲 WhatsApp</a>
        <a href="{share_links['Twitter']}" target="_blank">🐦 Twitter</a>
        <a href="{share_links['Email']}" target="_blank">✉️ Email</a>
        <a href="{share_links['LinkedIn']}" target="_blank">💼 LinkedIn</a>
    </div>
    """, unsafe_allow_html=True)

def download_pdf(summary):
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer)
    pdf.drawString(100, 750, "Summary Report")
    pdf.drawString(100, 730, summary)
    pdf.save()
    buffer.seek(0)
    return buffer

def download_word(summary):
    doc = Document()
    doc.add_paragraph(summary)
    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer

def download_audio(summary):
    tts = gTTS(summary, lang="en")
    buffer = BytesIO()
    tts.write_to_fp(buffer)
    buffer.seek(0)
    return buffer

if option == "Single File":
    uploaded_file = st.file_uploader("📂 Drag & Drop or Browse a .txt file", type=["txt"])
    text = uploaded_file.read().decode("utf-8") if uploaded_file else st.text_area("✍️ Paste your text here:", height=200)
    if text.strip():
        max_length = st.slider("📏 Max summary length (words):", 50, 500, 200)
        min_length = st.slider("📏 Min summary length (words):", 10, 100, 50)
        if st.button("✨ Summarize"):
            summary = summarize_text(text, max_length, min_length)
            if summary:
                st.success(summary)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.download_button("📄 PDF", download_pdf(summary), file_name="summary.pdf", mime="application/pdf")
                with col2:
                    st.download_button("📝 Word", download_word(summary), file_name="summary.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
                with col3:
                    st.download_button("🔊 Audio", download_audio(summary), file_name="summary.mp3", mime="audio/mp3")
                create_share_buttons(summary)

elif option == "Bulk File Processing":
    uploaded_files = st.file_uploader("Upload multiple .txt files", type=["txt"], accept_multiple_files=True)
    if uploaded_files:
        for file in uploaded_files:
            text = file.read().decode("utf-8")
            summary = summarize_text(text, 200, 50)
            st.success(summary)
            st.download_button("📄 PDF", download_pdf(summary), file_name=f"summary_{file.name}.pdf", mime="application/pdf")
            st.download_button("📝 Word", download_word(summary), file_name=f"summary_{file.name}.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
            st.download_button("🔊 Audio", download_audio(summary), file_name=f"summary_{file.name}.mp3", mime="audio/mp3")

elif option == "Summary History":
    st.subheader("📜 Summary History")
    for i, summary in enumerate(reversed(st.session_state.summary_history)):
        with st.expander(f"📄 Summary {len(st.session_state.summary_history) - i}"):
            st.write(summary)
            create_share_buttons(summary)

st.markdown("<hr><p style='text-align: center;'> Text Summarizer</p>", unsafe_allow_html=True)
