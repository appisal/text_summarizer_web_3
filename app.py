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
from sentence_transformers import SentenceTransformer, util
from langdetect import detect
from bertopic import BERTopic

# Check for GPU availability
device = 0 if torch.cuda.is_available() else -1

# Load NLP Models
@st.cache_resource
def load_models():
    return {
        "summarizer": pipeline("summarization", model="facebook/bart-large-cnn", device=device),
        "sentiment": pipeline("sentiment-analysis", device=device),
        "embedding": SentenceTransformer("all-MiniLM-L6-v2"),
        "topic_model": BERTopic()
    }

models = load_models()
summarizer = models["summarizer"]
sentiment_analyzer = models["sentiment"]
embedding_model = models["embedding"]
topic_model = models["topic_model"]
keyword_extractor = KeyBERT()

# Function to summarize text with progress indicator
def summarize_text(text, max_length, min_length):
    if len(text.split()) < min_length:
        return "Input text is too short to summarize."
    progress_bar = st.progress(0)
    for percent in range(1, 101, 10):
        time.sleep(0.05)
        progress_bar.progress(percent)
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    progress_bar.empty()
    return summary[0]["summary_text"]

# Function to detect language
def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

# Function to check text similarity
def compare_texts(text1, text2):
    emb1 = embedding_model.encode(text1, convert_to_tensor=True)
    emb2 = embedding_model.encode(text2, convert_to_tensor=True)
    similarity_score = util.pytorch_cos_sim(emb1, emb2).item()
    return similarity_score

# Function to detect topics
def detect_topics(text):
    topics, _ = topic_model.fit_transform([text])
    return topics[0] if topics else "No clear topic detected"

# Streamlit App
st.title("🚀 AI-Powered Text Summarizer")

st.sidebar.title("🔹 Features")
option = st.sidebar.radio("Choose an option:", ["Single File", "Multiple Files", "URL", "Compare Texts"])

if option == "Single File":
    st.write("Upload a text file or paste text below to summarize it.")
    uploaded_file = st.file_uploader("Choose a .txt file", type=["txt"])
    text = uploaded_file.read().decode("utf-8") if uploaded_file else st.text_area("Paste your text here:", height=200)
    
    if text.strip():
        st.write(f"📝 Word Count: {len(text.split())}")
        lang = detect_language(text)
        st.write(f"🌍 Language Detected: {lang}")
        
        max_length = st.slider("Max summary length (words):", 50, 500, 200)
        min_length = st.slider("Min summary length (words):", 10, 100, 50)
        
        if st.button("Summarize ✨"):
            summary = summarize_text(text, max_length, min_length)
            st.subheader("📌 Summary:")
            st.write(summary)
            
            sentiment = sentiment_analyzer(summary)[0]
            st.write(f"📊 Sentiment: **{sentiment['label']}** (Confidence: {sentiment['score']:.2f})")
            
            keywords = keyword_extractor.extract_keywords(summary, top_n=5)
            st.write("🔑 Keywords:", ", ".join([word for word, _ in keywords]))
            
            topic = detect_topics(summary)
            st.write(f"📖 Detected Topic: {topic}")

if option == "Compare Texts":
    text1 = st.text_area("Text 1:", height=200)
    text2 = st.text_area("Text 2:", height=200)
    
    if st.button("Compare Summaries ⚖️"):
        summary1 = summarize_text(text1, 200, 50)
        summary2 = summarize_text(text2, 200, 50)
        
        st.write("📌 **Summary 1:**")
        st.write(summary1)
        st.write("📌 **Summary 2:**")
        st.write(summary2)
        
        similarity = compare_texts(summary1, summary2)
        st.write(f"🔍 Similarity Score: {similarity:.2f}")
        st.write("✅ Are the summaries identical?", "Yes" if similarity > 0.9 else "No")
