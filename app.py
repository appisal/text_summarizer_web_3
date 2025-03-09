import streamlit as st

# Set page config must be the first Streamlit command
st.set_page_config(page_title="Text Summarizer Dashboard", layout="wide")

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

# Rest of your Streamlit code...

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

# Streamlit UI Configuration
st.set_page_config(page_title="Text Summarizer Dashboard", layout="wide")
st.markdown("""
    <style>
        .stApp { background-color: #f8f9fa; }
        .main { padding: 2rem; }
        h1 { color: #2c3e50; text-align: center; }
    </style>
""", unsafe_allow_html=True)

st.title("📄 AI-Powered Text Summarizer Dashboard")
st.sidebar.title("🔧 Features")
option = st.sidebar.radio("Select an option:", ["Summarize Text", "Summarize from URL", "Compare Texts"])

# Function to summarize text
def summarize_text(text, max_length, min_length):
    if len(text.split()) < min_length:
        return "⚠️ Input text is too short to summarize."
    summary = summarizer(
        text, max_length=max_length, min_length=min_length, do_sample=False
    )
    return summary[0]["summary_text"]

# Function to extract text from URL
def extract_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    paragraphs = soup.find_all("p")
    return " ".join([p.text for p in paragraphs])

# Layout
col1, col2 = st.columns(2)

if option == "Summarize Text":
    with col1:
        st.subheader("📝 Enter Your Text")
        text = st.text_area("Paste your text here:", height=250)
    with col2:
        if text.strip():
            st.subheader("📊 Summary & Insights")
            max_length = st.slider("Max summary length:", 50, 500, 200)
            min_length = st.slider("Min summary length:", 10, 100, 50)
            
            if st.button("Summarize 🔍"):
                summary = summarize_text(text, max_length, min_length)
                st.success(summary)
                sentiment = sentiment_analyzer(summary)[0]
                st.write(f"**Sentiment:** {sentiment['label']} (Confidence: {sentiment['score']:.2f})")
                
                keywords = keyword_extractor.extract_keywords(summary, top_n=5)
                st.write("**Keywords:**", ", ".join([word for word, _ in keywords]))
                
                # Word Cloud
                wordcloud = WordCloud(width=800, height=400, background_color="white").generate(summary)
                fig, ax = plt.subplots()
                ax.imshow(wordcloud, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig)

elif option == "Summarize from URL":
    url = st.text_input("🔗 Enter URL:")
    if st.button("Extract & Summarize"):
        text = extract_text_from_url(url)
        summary = summarize_text(text, 200, 50)
        st.success(summary)

elif option == "Compare Texts":
    text1 = st.text_area("Text 1", height=150)
    text2 = st.text_area("Text 2", height=150)
    if st.button("Compare Summaries"):
        summary1 = summarize_text(text1, 200, 50)
        summary2 = summarize_text(text2, 200, 50)
        st.write("**Summary 1:**")
        st.success(summary1)
        st.write("**Summary 2:**")
        st.success(summary2)
        st.write("Are the summaries identical?", "✅ Yes" if summary1 == summary2 else "❌ No")
