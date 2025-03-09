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
import time

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

# Initialize session state for history if not present
if "summary_history" not in st.session_state:
    st.session_state.summary_history = []

# Function to summarize text with a progress indicator
def summarize_text(text, max_length, min_length):
    if len(text.split()) < min_length:
        return "Input text is too short to summarize."

    progress_bar = st.progress(0)
    for percent in range(1, 101, 10):
        time.sleep(0.05)  # Simulate processing delay
        progress_bar.progress(percent)

    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    progress_bar.empty()

    # Store summary in session history
    st.session_state.summary_history.append(summary[0]["summary_text"])
    return summary[0]["summary_text"]

# Function to extract text from URL
def extract_text_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        paragraphs = soup.find_all("p")
        return " ".join([p.get_text() for p in paragraphs]) if paragraphs else "No readable content found."
    except requests.exceptions.RequestException as e:
        return f"Error fetching URL: {e}"

# Streamlit App
st.title("Text Summarizer - Enhanced 🚀")

st.sidebar.title("Features")
option = st.sidebar.radio("Choose an option:", ["Single File", "Multiple Files", "URL", "Compare Texts", "Summary History"])

# Summary History Feature
if option == "Summary History":
    st.subheader("📜 Summary History")
    if st.session_state.summary_history:
        for i, summary in enumerate(reversed(st.session_state.summary_history)):
            with st.expander(f"Summary {len(st.session_state.summary_history) - i}"):
                st.write(summary)
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Reuse Summary", key=f"reuse_{i}"):
                        st.session_state.summary_history.append(summary)
                        st.success("Summary added back for reanalysis!")
                with col2:
                    if st.button("Delete", key=f"delete_{i}"):
                        st.session_state.summary_history.pop(i)
                        st.experimental_rerun()
    else:
        st.write("No previous summaries found.")

# Single File Summarization
elif option == "Single File":
    st.write("Upload a text file or paste text below to summarize it.")
    
    uploaded_file = st.file_uploader("Choose a .txt file", type=["txt"])
    if uploaded_file:
        text = uploaded_file.read().decode("utf-8")
    else:
        text = st.text_area("Paste your text here:", height=200)

    if text.strip():
        max_length = st.slider("Max summary length (words):", 50, 500, 200)
        min_length = st.slider("Min summary length (words):", 10, 100, 50)

        if st.button("Summarize ✨"):
            summary = summarize_text(text, max_length, min_length)
            st.subheader("📌 Summary:")
            st.write(summary)

# Multiple File Summarization
elif option == "Multiple Files":
    uploaded_files = st.file_uploader("Choose .txt files", type=["txt"], accept_multiple_files=True)
    if uploaded_files:
        zip_buffer = BytesIO()
        with ZipFile(zip_buffer, "w") as zip_file:
            for file in uploaded_files:
                text = file.read().decode("utf-8")
                summary = summarize_text(text, 200, 50)
                zip_file.writestr(f"{file.name}_summary.txt", summary)
        zip_buffer.seek(0)
        st.download_button("📦 Download All Summaries as ZIP", zip_buffer, "summaries.zip", "application/zip")

# URL Summarization
elif option == "URL":
    url = st.text_input("Enter URL:")
    if st.button("Extract and Summarize 🌐"):
        text = extract_text_from_url(url)
        if text.startswith("Error"):
            st.error(text)
        else:
            summary = summarize_text(text, 200, 50)
            st.subheader("📌 Summary:")
            st.write(summary)

# Compare Texts
elif option == "Compare Texts":
    text1 = st.text_area("Text 1:", height=200)
    text2 = st.text_area("Text 2:", height=200)
    if st.button("Compare Summaries ⚖️"):
        summary1 = summarize_text(text1, 200, 50)
        summary2 = summarize_text(text2, 200, 50)
        st.write("📌 **Summary 1:**")
        st.write(summary1)
        st.write("📌 **Summary 2:**")
        st.write(summary2)
        st.write("✅ Are the summaries identical?", "Yes" if summary1 == summary2 else "No")
