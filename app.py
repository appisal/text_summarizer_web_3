import streamlit as st
from transformers import pipeline
import torch
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from docx import Document
from gtts import gTTS
from keybert import KeyBERT
from bs4 import BeautifulSoup
import requests
from zipfile import ZipFile
import time
import urllib.parse  # For sharing URLs

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

# Initialize session state for history
if "summary_history" not in st.session_state:
    st.session_state.summary_history = []

# Function to summarize text with progress indicator
def summarize_text(text, max_length, min_length):
    if len(text.split()) < min_length:
        return "Input text is too short to summarize."

    progress_bar = st.progress(0)
    for percent in range(1, 101, 10):
        time.sleep(0.05)  # Simulate processing delay
        progress_bar.progress(percent)

    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    progress_bar.empty()

    # Save to history
    st.session_state.summary_history.append(summary[0]["summary_text"])

    return summary[0]["summary_text"]

# Function to generate a shareable summary link
def generate_share_links(summary):
    encoded_summary = urllib.parse.quote(summary)
    email_link = f"mailto:?subject=Check%20this%20Summary&body={encoded_summary}"
    whatsapp_link = f"https://wa.me/?text={encoded_summary}"
    twitter_link = f"https://twitter.com/intent/tweet?text={encoded_summary}"
    linkedin_link = f"https://www.linkedin.com/sharing/share-offsite/?url={encoded_summary}"
    
    return email_link, whatsapp_link, twitter_link, linkedin_link

# Streamlit App Header
st.markdown(
    """
    <h1 style='text-align: center; color: #FF4B4B;'>🚀 Text Summarizer - Enhanced</h1>
    <p style='text-align: center; color: #555;'>AI-powered summarization for quick insights</p>
    <hr style="border:1px solid #FF4B4B;">
    """,
    unsafe_allow_html=True
)

st.sidebar.title("⚡ Features")
option = st.sidebar.radio(
    "Choose an option:",
    ["Single File", "Multiple Files", "URL", "Compare Texts", "Summary History"]
)

# Single File Summarization
if option == "Single File":
    st.write("📂 Upload a text file or paste text below to summarize it.")

    uploaded_file = st.file_uploader("Choose a .txt file", type=["txt"])
    if uploaded_file:
        text = uploaded_file.read().decode("utf-8")
    else:
        text = st.text_area("✍️ Paste your text here:", height=200)

    if text.strip():
        original_word_count = len(text.split())
        st.write(f"📝 **Word Count:** {original_word_count}")
        st.write(f"🔢 **Character Count:** {len(text)}")

        max_length = st.slider("📏 Max summary length (words):", 50, 500, 200)
        min_length = st.slider("📏 Min summary length (words):", 10, 100, 50)

        if st.button("✨ Summarize"):
            summary = summarize_text(text, max_length, min_length)
            summarized_word_count = len(summary.split())
            reduction_percentage = (1 - (summarized_word_count / original_word_count)) * 100

            st.subheader("📌 Summary:")
            st.write(summary)

            # Display Summary Reduction
            st.write(f"📉 **Reduction:** {reduction_percentage:.2f}% fewer words!")

            # Word Count Comparison Chart
            chart_data = {"Original": original_word_count, "Summarized": summarized_word_count}
            fig, ax = plt.subplots()
            sns.barplot(x=list(chart_data.keys()), y=list(chart_data.values()), palette=["#FF4B4B", "#4BFFB3"])
            plt.ylabel("Word Count")
            plt.title("📊 Word Reduction Chart")
            st.pyplot(fig)

            # Sentiment Analysis
            sentiment = sentiment_analyzer(summary)[0]
            st.write(f"📊 **Sentiment:** **{sentiment['label']}** (Confidence: {sentiment['score']:.2f})")

            # Keywords Extraction
            keywords = keyword_extractor.extract_keywords(summary, top_n=5)
            st.write("🔑 **Keywords:**", ", ".join([word for word, _ in keywords]))

            # Generate Share Links
            email_link, whatsapp_link, twitter_link, linkedin_link = generate_share_links(summary)

            # Share Buttons
            st.markdown("### 📢 Share Summary:")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f'<a href="{email_link}" target="_blank">📧 Email</a>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<a href="{whatsapp_link}" target="_blank">📱 WhatsApp</a>', unsafe_allow_html=True)
            with col3:
                st.markdown(f'<a href="{twitter_link}" target="_blank">🐦 Twitter</a>', unsafe_allow_html=True)
            with col4:
                st.markdown(f'<a href="{linkedin_link}" target="_blank">🔗 LinkedIn</a>', unsafe_allow_html=True)

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
st.markdown("<br><hr><p style='text-align: center;'>© 2025 Text Summarizer AI | Built with ❤️ by AI</p>", unsafe_allow_html=True)
