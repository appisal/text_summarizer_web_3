import streamlit as st
from transformers import pipeline
import torch
from io import BytesIO
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from keybert import KeyBERT
import time
import urllib.parse  # For sharing URLs
from gtts import gTTS  # For text-to-speech
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from docx import Document  # For Word documents

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

# Function to summarize text
def summarize_text(text, max_length, min_length):
    if len(text.split()) < min_length:
        return "Input text is too short to summarize."

    with st.spinner("🔄 Summarizing... Please wait."):
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)

    # Save to history
    st.session_state.summary_history.append(summary[0]["summary_text"])
    return summary[0]["summary_text"]

# Function to generate PDFs
def generate_pdf(summary):
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    pdf.setFont("Helvetica", 12)
    pdf.drawString(100, 750, "Summary:")
    text_lines = summary.split("\n")

    y_position = 730
    for line in text_lines:
        pdf.drawString(100, y_position, line)
        y_position -= 20

    pdf.save()
    buffer.seek(0)
    return buffer

# Function to generate Word document
def generate_word(summary):
    buffer = BytesIO()
    doc = Document()
    doc.add_heading("Summary", level=1)
    doc.add_paragraph(summary)
    doc.save(buffer)
    buffer.seek(0)
    return buffer

# Function to generate audio (MP3)
def generate_audio(summary):
    buffer = BytesIO()
    tts = gTTS(text=summary, lang="en")
    tts.save(buffer)
    buffer.seek(0)
    return buffer

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
option = st.sidebar.radio("Choose an option:", ["Single File", "Summary History"])

# Single File Summarization
if option == "Single File":
    st.write("📂 Upload a text file or paste text below to summarize it.")
    uploaded_file = st.file_uploader("Choose a .txt file", type=["txt"])
    text = uploaded_file.read().decode("utf-8") if uploaded_file else st.text_area("✍️ Paste your text here:", height=200)

    if text.strip():
        original_word_count = len(text.split())
        st.write(f"📝 **Word Count:** {original_word_count}")
        st.write(f"🔢 **Character Count:** {len(text)}")

        max_length = st.slider("📏 Max summary length (words):", 50, 500, 200)
        min_length = st.slider("📏 Min summary length (words):", 10, 100, 50)

        if st.button("✨ Summarize", use_container_width=True):
            summary = summarize_text(text, max_length, min_length)
            summarized_word_count = len(summary.split())
            reduction_percentage = (1 - (summarized_word_count / original_word_count)) * 100

            st.subheader("📌 Summary:")
            st.success(summary)

            # Display Reduction
            st.write(f"📉 **Reduction:** {reduction_percentage:.2f}% fewer words!")

            # Word Count Chart
            fig, ax = plt.subplots()
            sns.barplot(x=["Original", "Summarized"], y=[original_word_count, summarized_word_count], palette=["#FF4B4B", "#4BFFB3"])
            plt.ylabel("Word Count")
            plt.title("📊 Word Reduction Chart")
            st.pyplot(fig)

            # Sentiment Analysis
            sentiment = sentiment_analyzer(summary)[0]
            st.write(f"📊 **Sentiment:** **{sentiment['label']}** (Confidence: {sentiment['score']:.2f})")

            # Keywords Extraction
            keywords = keyword_extractor.extract_keywords(summary, top_n=5)
            st.write("🔑 **Keywords:**", ", ".join([word for word, _ in keywords]))

            # **Download Options**
            st.markdown("### 📥 Download Summary:")
            col1, col2, col3, col4 = st.columns(4)

            # Download as Text
            text_buffer = BytesIO(summary.encode('utf-8'))
            col1.download_button(
                label="📄 TXT",
                data=text_buffer,
                file_name="summary.txt",
                mime="text/plain"
            )

            # Download as PDF
            pdf_buffer = generate_pdf(summary)
            col2.download_button(
                label="📑 PDF",
                data=pdf_buffer,
                file_name="summary.pdf",
                mime="application/pdf"
            )

            # Download as Word
            word_buffer = generate_word(summary)
            col3.download_button(
                label="📝 Word",
                data=word_buffer,
                file_name="summary.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )

            # Download as MP3
            audio_buffer = generate_audio(summary)
            col4.download_button(
                label="🔊 Audio",
                data=audio_buffer,
                file_name="summary.mp3",
                mime="audio/mpeg"
            )

            # Share Summary
            st.markdown("### 📢 Share Summary:")
            cols = st.columns(4)
            for col, (label, link) in zip(cols, generate_share_links(summary).items()):
                col.markdown(f'<a href="{link}" target="_blank">{label}</a>', unsafe_allow_html=True)

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
st.markdown("<br><hr><p style='text-align: center;'>© 2025 Text Summarizer AI </p>", unsafe_allow_html=True)
