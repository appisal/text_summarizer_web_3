import streamlit as st
from transformers import pipeline
import torch
from io import BytesIO
from gtts import gTTS
from keybert import KeyBERT
import urllib.parse  

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

# Streamlit Taskbar (Fixed at Top)
st.markdown("""
    <style>
        .taskbar {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            background-color: #FF4B4B;
            padding: 10px;
            text-align: center;
            font-size: 20px;
            font-weight: bold;
            color: white;
            z-index: 1000;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        }
        .content { margin-top: 60px; }
    </style>
    <div class="taskbar">🚀 AI Text Summarizer | 📄 Upload | 📜 History | ℹ️ About</div>
    <div class="content">
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/a/a7/React-icon.svg", width=100)
st.sidebar.title("⚡ Features")
option = st.sidebar.radio("Choose an option:", ["Single File", "Summary History"])

# Single File Summarization
if option == "Single File":
    st.markdown("<h3>📂 Upload a text file or paste text below to summarize it.</h3>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a .txt file", type=["txt"])
    text = uploaded_file.read().decode("utf-8") if uploaded_file else st.text_area("✍️ Paste your text here:", height=200)

    if text.strip():
        max_length = st.slider("📏 Max summary length (words):", 50, 500, 200)
        min_length = st.slider("📏 Min summary length (words):", 10, 100, 50)

        if st.button("✨ Summarize", use_container_width=True):
            summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]["summary_text"]
            st.session_state.summary_history.append(summary)

            st.markdown("<h3>📌 Summary:</h3>", unsafe_allow_html=True)
            st.success(summary)

# Summary History
elif option == "Summary History":
    st.subheader("📜 Summary History")
    if st.session_state.summary_history:
        for i, summary in enumerate(reversed(st.session_state.summary_history)):
            with st.expander(f"📄 Summary {len(st.session_state.summary_history) - i}"):
                st.write(summary)
    else:
        st.write("🔍 No previous summaries found.")

# Footer
st.markdown("""
    <hr>
    <p style='text-align: center; font-size: 14px; color: #555;'>
        © 2025 Text Summarizer AI | Built with ❤️ using Streamlit
    </p>
""", unsafe_allow_html=True)
