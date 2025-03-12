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

st.set_page_config(page_title="AI Text Summarizer", layout="wide")

# Styling and UI Enhancements
st.markdown("""
    <style>
        body { font-family: 'Poppins', sans-serif; }
        .title-container { text-align: center; font-size: 2.5rem; font-weight: bold; }
        .stButton > button { transition: 0.3s; border-radius: 10px; }
        .stButton > button:hover { transform: scale(1.05); }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='title-container'>🚀 AI Text Summarizer</div>", unsafe_allow_html=True)

# Sidebar
dark_mode = st.sidebar.toggle("🌗 Dark Mode")
if dark_mode:
    st.markdown("<style>body { background-color: #1E1E1E; color: white; }</style>", unsafe_allow_html=True)

st.sidebar.title("⚡ Features")
option = st.sidebar.radio("Choose an option:", ["Single Text", "Bulk Processing", "History"])

# Function to summarize text
def summarize_text(text, max_length, min_length):
    if len(text.split()) < min_length:
        return "Input text is too short to summarize."
    with st.spinner("🔄 Summarizing..."):
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    st.session_state.summary_history.append(summary[0]["summary_text"])
    return summary[0]["summary_text"]

# Function to analyze sentiment
def analyze_sentiment(text):
    sentiment = sentiment_analyzer(text)[0]
    return sentiment["label"], sentiment["score"]

# Function to generate word cloud
def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

# Function to export summary
def export_summary(summary, format):
    buffer = BytesIO()
    if format == "pdf":
        pdf = canvas.Canvas(buffer)
        pdf.drawString(100, 750, "Summary Report")
        pdf.drawString(100, 730, summary)
        pdf.save()
    elif format == "word":
        doc = Document()
        doc.add_paragraph(summary)
        doc.save(buffer)
    elif format == "audio":
        tts = gTTS(summary, lang="en")
        tts.write_to_fp(buffer)
    buffer.seek(0)
    return buffer

# Summary History
if "summary_history" not in st.session_state:
    st.session_state.summary_history = []

if option == "Single Text":
    text = st.text_area("✍️ Enter Text:", height=200)
    max_length = st.slider("📏 Max summary length:", 50, 500, 200)
    min_length = st.slider("📏 Min summary length:", 10, 100, 50)

    if st.button("✨ Summarize", use_container_width=True):
        summary = summarize_text(text, max_length, min_length)
        sentiment, score = analyze_sentiment(summary)
        
        st.success(summary)
        st.write(f"**Sentiment:** {sentiment} ({score:.2f})")
        generate_wordcloud(summary)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button("📄 PDF", export_summary(summary, "pdf"), "summary.pdf", "application/pdf")
        with col2:
            st.download_button("📝 Word", export_summary(summary, "word"), "summary.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        with col3:
            st.download_button("🔊 Audio", export_summary(summary, "audio"), "summary.mp3", "audio/mp3")

elif option == "Bulk Processing":
    uploaded_files = st.file_uploader("Upload multiple .txt files", type=["txt"], accept_multiple_files=True)
    if uploaded_files:
        for file in uploaded_files:
            text = file.read().decode("utf-8")
            summary = summarize_text(text, 200, 50)
            st.markdown(f"### 📜 Summary for {file.name}")
            st.success(summary)
            st.download_button("📄 PDF", export_summary(summary, "pdf"), f"summary_{file.name}.pdf", "application/pdf")
            st.download_button("📝 Word", export_summary(summary, "word"), f"summary_{file.name}.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
            st.download_button("🔊 Audio", export_summary(summary, "audio"), f"summary_{file.name}.mp3", "audio/mp3")

elif option == "History":
    st.subheader("📜 Summary History")
    if st.session_state.summary_history:
        for i, summary in enumerate(reversed(st.session_state.summary_history)):
            with st.expander(f"📄 Summary {len(st.session_state.summary_history) - i}"):
                st.write(summary)
                generate_wordcloud(summary)

st.markdown("<hr><p style='text-align: center;'>AI Text Summarizer</p>", unsafe_allow_html=True)
