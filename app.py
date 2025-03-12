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
# Stylish Animated Title
# Apply global CSS animations and styles
# Enhanced Premium UI Styling
st.markdown("""
    <style>
        /* Global Styling */
        body {
            font-family: 'Poppins', sans-serif;
            background: #eef1f6;
            color: #333;
        }
        /* Smooth Fade-in Animation */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        /* Glossy Glassmorphism Card */
        .glass-card {
            background: rgba(255, 255, 255, 0.3);
            backdrop-filter: blur(12px);
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0px 8px 20px rgba(0, 0, 0, 0.15);
            animation: fadeIn 1s ease-in-out;
        }
        
        /* Premium Gradient Buttons */
        .stButton > button {
            transition: all 0.3s ease-in-out;
            border-radius: 12px !important;
            background: linear-gradient(135deg, #ff7eb3, #ff758c);
            color: white !important;
            font-weight: bold;
            padding: 14px;
            border: none;
            box-shadow: 0px 4px 12px rgba(255, 118, 136, 0.3);
        }
        .stButton > button:hover {
            background: linear-gradient(135deg, #ff758c, #ff7eb3);
            transform: scale(1.07);
        }
        
        /* Input Field Styling */
        .stTextArea, .stFileUploader {
            border-radius: 12px;
            padding: 12px;
            box-shadow: 0px 5px 12px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease-in-out;
            border: 1px solid rgba(255, 118, 136, 0.4);
        }
        .stTextArea:hover, .stFileUploader:hover {
            box-shadow: 0px 7px 18px rgba(0, 0, 0, 0.2);
        }
        
        /* Premium Header Styling */
        .title-container {
            text-align: center;
            animation: fadeIn 1.5s ease-in-out;
            font-size: 3rem;
            font-weight: bold;
            background: linear-gradient(45deg, #ff7eb3, #ff758c);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            padding: 12px;
            text-shadow: 2px 4px 10px rgba(255, 118, 136, 0.4);
        }

        /* Professional Navigation Bar */
        .navbar {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            background: linear-gradient(135deg, #ff7eb3, #ff758c);
            padding: 15px 20px;
            z-index: 1000;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
        }

        /* Navbar Left (Logo) */
        .navbar .logo {
            font-size: 22px;
            font-weight: bold;
            color: white;
            text-decoration: none;
            margin-left: 20px;
        }

        /* Navbar Links */
        .navbar .nav-links {
            list-style: none;
            display: flex;
            gap: 20px;
            margin-right: 20px;
        }

        .navbar .nav-links li {
            display: inline;
        }

        .navbar .nav-links a {
            color: white;
            text-decoration: none;
            font-size: 16px;
            font-weight: 500;
            padding: 8px 12px;
            border-radius: 5px;
            transition: background 0.3s ease;
        }

        .navbar .nav-links a:hover {
            background: rgba(255, 255, 255, 0.2);
        }

        /* Adjust main content to avoid overlap with navbar */
        .content {
            margin-top: 80px;
        }
        
        /* Dark Mode */
        .dark-mode body {
            background: #1e1e1e;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Add Navigation Bar
st.markdown("""
    <div class="navbar">
        <a href="#" class="logo">🚀 Premium Summarizer</a>
        <ul class="nav-links">
            <li><a href="#home">Home</a></li>
            <li><a href="#features">Features</a></li>
            <li><a href="#about">About</a></li>
            <li><a href="#contact">Contact</a></li>
        </ul>
    </div>
""", unsafe_allow_html=True)

# Add Title and Padding to Avoid Navbar Overlap
st.markdown("<div class='content'></div>", unsafe_allow_html=True)
st.markdown("<div class='title-container'>🚀 Premium Text Summarizer</div>", unsafe_allow_html=True)



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

# Function to summarize text
def summarize_text(text, max_length, min_length):
    if len(text.split()) < min_length:
        return "Input text is too short to summarize."
    
    with st.spinner("🔄 Summarizing... Please wait."):
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)

    st.session_state.summary_history.append(summary[0]["summary_text"])
    return summary[0]["summary_text"]

# Function to generate sharing links
def generate_share_links(summary):
    encoded_summary = urllib.parse.quote(summary)
    return {
        "WhatsApp": f"https://wa.me/?text={encoded_summary}",
        "Twitter": f"https://twitter.com/intent/tweet?text={encoded_summary}",
        "Email": f"mailto:?subject=Summary&body={encoded_summary}",
        "LinkedIn": f"https://www.linkedin.com/sharing/share-offsite/?url={encoded_summary}"
    }

# Function to create share buttons with icons
def create_share_buttons(summary):
    share_links = generate_share_links(summary)
    share_html = f"""
    <style>
        .share-btns {{
            display: flex;
            gap: 10px;
            margin-top: 10px;
        }}
        .share-btns a img {{
            width: 40px;
            height: 40px;
        }}
    </style>
    <div class="share-btns">
        <a href="{share_links['WhatsApp']}" target="_blank">
            <img src="https://upload.wikimedia.org/wikipedia/commons/6/6b/WhatsApp.svg">
        </a>
        <a href="{share_links['Twitter']}" target="_blank">
            <img src="https://upload.wikimedia.org/wikipedia/en/6/60/Twitter_Logo_as_of_2021.svg">
        </a>
        <a href="{share_links['Email']}" target="_blank">
            <img src="https://upload.wikimedia.org/wikipedia/commons/4/4e/Mail_%28iOS%29.svg">
        </a>
        <a href="{share_links['LinkedIn']}" target="_blank">
            <img src="https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png">
        </a>
    </div>
    """
    st.markdown(share_html, unsafe_allow_html=True)

# Function to download summary as PDF
def download_pdf(summary):
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer)
    pdf.drawString(100, 750, "Summary Report")
    pdf.drawString(100, 730, summary)
    pdf.save()
    buffer.seek(0)
    return buffer

# Function to download summary as Word
def download_word(summary):
    doc = Document()
    doc.add_paragraph(summary)
    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer

# Function to download summary as audio
def download_audio(summary):
    tts = gTTS(summary, lang="en")
    buffer = BytesIO()
    tts.write_to_fp(buffer)
    buffer.seek(0)
    return buffer

# Dark Mode Toggle
st.sidebar.markdown("🌗 **Theme:**")
dark_mode = st.sidebar.toggle("Enable Dark Mode")
if dark_mode:
    st.markdown("<style>body { background-color: #1E1E1E; color: white; }</style>", unsafe_allow_html=True)

# Sidebar with Logo
#st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/a/a7/React-icon.svg", width=100)  
st.sidebar.title("⚡ Features")
option = st.sidebar.radio("Choose an option:", ["Single File", "Bulk File Processing", "Summary History"])

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

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.download_button("📄 PDF", download_pdf(summary), file_name="summary.pdf", mime="application/pdf")
                with col2:
                    st.download_button("📝 Word", download_word(summary), file_name="summary.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
                with col3:
                    st.download_button("🔊 Audio", download_audio(summary), file_name="summary.mp3", mime="audio/mp3")
                
                # Display Share Buttons
                st.markdown("### 📢 Share this Summary:")
                create_share_buttons(summary)

# Bulk File Processing
elif option == "Bulk File Processing":
    uploaded_files = st.file_uploader("Upload multiple .txt files", type=["txt"], accept_multiple_files=True)
    if uploaded_files:
        for file in uploaded_files:
            text = file.read().decode("utf-8")
            summary = summarize_text(text, 200, 50)
            st.markdown(f"### 📜 Summary for {file.name}")
            st.success(summary)
            st.download_button("📄 PDF", download_pdf(summary), file_name=f"summary_{file.name}.pdf", mime="application/pdf")
            st.download_button("📝 Word", download_word(summary), file_name=f"summary_{file.name}.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
            st.download_button("🔊 Audio", download_audio(summary), file_name=f"summary_{file.name}.mp3", mime="audio/mp3")

# Summary History
elif option == "Summary History":
    st.subheader("📜 Summary History")
    if st.session_state.summary_history:
        for i, summary in enumerate(reversed(st.session_state.summary_history)):
            with st.expander(f"📄 Summary {len(st.session_state.summary_history) - i}"):
                st.write(summary)
                create_share_buttons(summary)

st.markdown("<hr><p style='text-align: center;'> Text Summarizer</p>", unsafe_allow_html=True)
