import streamlit as st
from transformers import pipeline
import torch
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from docx import Document
from gtts import gTTS
from keybert import KeyBERT
from bs4 import BeautifulSoup
import requests
from zipfile import ZipFile
import os
import json
from datetime import datetime

# Custom CSS for professional styling
st.markdown("""
    <style>
    .stApp {
        background-color: #f5f5f5;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stTextArea>textarea {
        border-radius: 5px;
        border: 1px solid #ccc;
        padding: 10px;
    }
    .stSlider>div {
        color: #4CAF50;
    }
    .stHeader {
        color: #4CAF50;
    }
    .stDownloadButton>button {
        background-color: #008CBA;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stDownloadButton>button:hover {
        background-color: #007B9E;
    }
    .dark-mode {
        background-color: #1e1e1e;
        color: #ffffff;
    }
    </style>
    """, unsafe_allow_html=True)

# Check for GPU availability
device = 0 if torch.cuda.is_available() else -1

# Load summarizer and sentiment analysis pipelines
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn", device=device)

@st.cache_resource
def load_sentiment_analyzer():
    return pipeline("sentiment-analysis", device=device)

summarizer = load_summarizer()
sentiment_analyzer = load_sentiment_analyzer()
keyword_extractor = KeyBERT()

# Function to summarize text
def summarize_text(text, max_length, min_length, language="en"):
    if len(text.split()) < min_length:
        return "Input text is too short to summarize."
    summary = summarizer(
        text,
        max_length=max_length,
        min_length=min_length,
        do_sample=False,
    )
    return summary[0]["summary_text"]

# Function to extract text from URL
def extract_text_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses
        soup = BeautifulSoup(response.content, "html.parser")
        paragraphs = soup.find_all("p")
        return " ".join([p.text for p in paragraphs])
    except Exception as e:
        return f"Error extracting text from URL: {e}"

# Function to create files (PDF, TXT, DOCX)
def create_pdf(summary):
    buffer = BytesIO()
    pdf_canvas = canvas.Canvas(buffer, pagesize=letter)
    pdf_canvas.setFont("Helvetica", 12)
    pdf_canvas.drawString(30, 750, "Summary:")
    text_obj = pdf_canvas.beginText(30, 730)
    text_obj.setFont("Helvetica", 10)
    for line in summary.split("\n"):
        text_obj.textLine(line)
    pdf_canvas.drawText(text_obj)
    pdf_canvas.save()
    buffer.seek(0)
    return buffer

def create_txt(summary):
    return BytesIO(summary.encode())

def create_docx(summary):
    doc = Document()
    doc.add_heading("Summary", level=1)
    doc.add_paragraph(summary)
    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer

# Function to create audio
def text_to_speech(summary, language="en"):
    tts = gTTS(summary, lang=language)
    buffer = BytesIO()
    tts.write_to_fp(buffer)
    buffer.seek(0)
    return buffer

# Streamlit App
st.set_page_config(page_title="Text Summarizer", layout="wide", page_icon="📝")
st.title("📝 Text Summarizer - Enhanced")

# Dark Mode Toggle
dark_mode = st.checkbox("Dark Mode")
if dark_mode:
    st.markdown("<style>.stApp {background-color: #1e1e1e; color: #ffffff;}</style>", unsafe_allow_html=True)

# User Authentication
st.sidebar.title("User Authentication")
username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")
if st.sidebar.button("Login"):
    if username == "admin" and password == "password":
        st.sidebar.success("Logged in as admin")
    else:
        st.sidebar.error("Invalid credentials")

# Features
st.sidebar.title("Features")
st.sidebar.markdown("### Choose an option:")
option = st.sidebar.radio("", ["Single File", "Multiple Files", "URL", "Compare Texts", "API Integration"])

# Single File Summarization
if option == "Single File":
    st.write("Upload a text file or paste text below to summarize it.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Choose a file", type=["txt", "pdf", "docx"])
        if uploaded_file:
            if uploaded_file.type == "application/pdf":
                text = extract_text_from_pdf(uploaded_file)
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                text = extract_text_from_docx(uploaded_file)
            else:
                text = uploaded_file.read().decode("utf-8")
        else:
            text = st.text_area("Paste your text here:", height=300)

    with col2:
        if text.strip():
            st.write(f"**Word Count:** {len(text.split())}")
            st.write(f"**Character Count:** {len(text)}")

            max_length = st.slider("Max summary length (words):", 50, 500, 200)
            min_length = st.slider("Min summary length (words):", 10, 100, 50)
            language = st.selectbox("Select Language", ["en", "es", "fr", "de"])

            if st.button("Summarize"):
                with st.spinner("Generating summary..."):
                    summary = summarize_text(text, max_length, min_length, language)
                    st.subheader("Summary:")
                    st.write(summary)

                    sentiment = sentiment_analyzer(summary)[0]
                    st.write(f"**Sentiment:** {sentiment['label']} (Confidence: {sentiment['score']:.2f})")

                    keywords = keyword_extractor.extract_keywords(summary, top_n=5)
                    st.write("**Keywords:**", ", ".join([word for word, _ in keywords]))

                    # File download buttons
                    st.markdown("### Download Options:")
                    col3, col4, col5, col6 = st.columns(4)
                    with col3:
                        pdf_data = create_pdf(summary)
                        st.download_button("📄 Download PDF", pdf_data, "summary.pdf", "application/pdf")
                    with col4:
                        txt_data = create_txt(summary)
                        st.download_button("📝 Download TXT", txt_data, "summary.txt", "text/plain")
                    with col5:
                        docx_data = create_docx(summary)
                        st.download_button("📑 Download DOCX", docx_data, "summary.docx",
                                           "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
                    with col6:
                        audio_data = text_to_speech(summary, language)
                        st.audio(audio_data, format="audio/mp3")
                        st.download_button("🎧 Download Audio", audio_data, "summary.mp3", "audio/mpeg")

                    # Word cloud visualization
                    st.markdown("### Word Cloud:")
                    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(summary)
                    fig, ax = plt.subplots()
                    ax.imshow(wordcloud, interpolation="bilinear")
                    ax.axis("off")
                    st.pyplot(fig)

# Multiple File Summarization
elif option == "Multiple Files":
    st.write("Upload multiple text files to summarize them.")
    uploaded_files = st.file_uploader("Choose files", type=["txt", "pdf", "docx"], accept_multiple_files=True)
    if uploaded_files:
        with st.spinner("Generating summaries..."):
            zip_buffer = BytesIO()
            with ZipFile(zip_buffer, "w") as zip_file:
                for file in uploaded_files:
                    if file.type == "application/pdf":
                        text = extract_text_from_pdf(file)
                    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                        text = extract_text_from_docx(file)
                    else:
                        text = file.read().decode("utf-8")
                    summary = summarize_text(text, 200, 50)
                    zip_file.writestr(f"{file.name}_summary.txt", summary)
            zip_buffer.seek(0)
            st.download_button("📦 Download All Summaries as ZIP", zip_buffer, "summaries.zip", "application/zip")

# URL Summarization
elif option == "URL":
    st.write("Enter a URL to extract and summarize text.")
    url = st.text_input("Enter URL:")
    if st.button("Extract and Summarize"):
        with st.spinner("Extracting and summarizing..."):
            text = extract_text_from_url(url)
            if "Error" in text:
                st.error(text)
            else:
                summary = summarize_text(text, 200, 50)
                st.subheader("Summary:")
                st.write(summary)

# Compare Texts
elif option == "Compare Texts":
    st.write("Enter two texts to compare their summaries.")
    col1, col2 = st.columns(2)
    
    with col1:
        text1 = st.text_area("Text 1:", height=300)
    
    with col2:
        text2 = st.text_area("Text 2:", height=300)

    if st.button("Compare Summaries"):
        with st.spinner("Generating summaries..."):
            summary1 = summarize_text(text1, 200, 50)
            summary2 = summarize_text(text2, 200, 50)
            st.write("**Summary 1:**")
            st.write(summary1)
            st.write("**Summary 2:**")
            st.write(summary2)
            st.write("**Are the summaries identical?**", "Yes" if summary1 == summary2 else "No")

# API Integration
elif option == "API Integration":
    st.write("Fetch data from an API and summarize it.")
    api_url = st.text_input("Enter API URL:")
    if st.button("Fetch and Summarize"):
        with st.spinner("Fetching and summarizing..."):
            try:
                response = requests.get(api_url)
                response.raise_for_status()
                data = response.json()
                text = json.dumps(data)
                summary = summarize_text(text, 200, 50)
                st.subheader("Summary:")
                st.write(summary)
            except Exception as e:
                st.error(f"Error fetching data from API: {e}")
