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
def summarize_text(text, max_length, min_length):
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
def text_to_speech(summary):
    tts = gTTS(summary, lang="en")
    buffer = BytesIO()
    tts.write_to_fp(buffer)
    buffer.seek(0)
    return buffer

# Streamlit App
st.title("Text Summarizer - Enhanced")

st.sidebar.title("Features")

# Ensure username is stored in session state
if "username" not in st.session_state:
    st.session_state["username"] = None  # Initialize it

if st.session_state["username"]:
    st.sidebar.success(f"Logged in as {st.session_state['username']}")
else:
    st.sidebar.error("User  not logged in")

option = st.sidebar.radio(
    "Choose an option:",
    ["Single File", "Multiple Files", "URL", "Compare Texts"]
)

# Single File Summarization
if option == "Single File":
    st.write("Upload a text file or paste text below to summarize it.")

    uploaded_file = st.file_uploader("Choose a .txt file", type=["txt"])
    if uploaded_file:
        text = uploaded_file.read().decode("utf-8")
    else:
        text = st.text_area("Paste your text here:", height=200)

    if text.strip():
        st.write(f"Word Count: {len(text.split())}")
        st.write(f"Character Count: {len(text)}")

        max_length = st.slider("Max summary length (words):", 50, 500, 200)
        min_length = st.slider("Min summary length (words):", 10, 100, 50)

        if st.button("Summarize"):
            summary = summarize_text(text, max_length, min_length)
            st.subheader("Summary:")
            st.write(summary)

            sentiment = sentiment_analyzer(summary)[0]
            st.write(f"Sentiment: {sentiment['label']} (Confidence: {sentiment['score']:.2f})")

            keywords = keyword_extractor.extract_keywords(summary, top_n=5)
            st.write("Keywords:", ", ".join([word for word, _ in keywords]))

            pdf_data = create_pdf(summary)
            st.download_button("Download PDF", pdf_data, "summary.pdf", "application/pdf")

            txt_data = create_txt(summary)
            st.download_button("Download TXT", txt_data, "summary.txt", "text/plain")

            docx_data = create_docx(summary)
            st.download_button("Download DOCX", docx_data, "summary.docx",
                               "application/vnd.openxmlformats-officedocument.wordprocessingml.document")

            audio_data = text_to_speech(summary)
            st.audio(audio_data, format="audio/mp3")
            st.download_button("Download Audio", audio_data, "summary.mp3", "audio/mpeg")

            wordcloud = WordCloud(width=800, height=400, background_color="white").generate(summary)
            fig, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)

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
        st.download_button("Download All Summaries as ZIP", zip_buffer, "summaries.zip", "application/zip")

# URL Summarization
elif option == "URL":
    url = st.text_input("Enter URL:")
    if st.button("Extract and Summarize"):
        text = extract_text_from_url(url)
        if "Error" in text:
            st.error(text)
        else:
            summary = summarize_text(text, 200, 50)
            st.subheader("Summary:")
            st.write(summary)

# Compare Texts
elif option == "Compare Texts":
    text1 = st.text_area("Text 1:", height=200)
    text2 = st.text_area("Text 2:", height=200)
    if st.button("Compare Summaries"):
        summary1 = summarize_text(text1, 200, 50)
        summary2 = summarize_text(text2, 200, 50)
        st.write("Summary 1:")
        st.write(summary1)
        st.write("Summary 2:")
        st.write(summary2)
        st.write("Are the summaries identical?", "Yes" if summary1 == summary2 else "No")
