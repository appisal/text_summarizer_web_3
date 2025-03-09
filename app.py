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

# Streamlit App
st.title("Text Summarizer - Enhanced 🚀")

st.sidebar.title("Features")
option = st.sidebar.radio("Choose an option:", ["Summarize Text", "Summary History"])

if option == "Summarize Text":
    text = st.text_area("Paste your text here:", height=200)
    if text.strip():
        max_length = st.slider("Max summary length (words):", 50, 500, 200)
        min_length = st.slider("Min summary length (words):", 10, 100, 50)
        
        if st.button("Summarize ✨"):
            summary = summarize_text(text, max_length, min_length)
            st.subheader("📌 Summary:")
            st.write(summary)

elif option == "Summary History":
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
