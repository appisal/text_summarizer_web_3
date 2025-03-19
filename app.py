import streamlit as st
from io import BytesIO
from gtts import gTTS
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pdfplumber
from docx import Document
from reportlab.pdfgen import canvas
import urllib.parse

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    with pdfplumber.open(uploaded_file) as pdf:
        return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

# Function to extract text from DOCX
def extract_text_from_docx(uploaded_file):
    doc = Document(uploaded_file)
    return "\n".join([para.text for para in doc.paragraphs])

# Function to download text as PDF
def download_pdf(text):
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer)
    pdf.drawString(100, 750, "Processed Text")
    pdf.drawString(100, 730, text)
    pdf.save()
    buffer.seek(0)
    return buffer

# Function to download text as DOCX
def download_word(text):
    doc = Document()
    doc.add_paragraph(text)
    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer

# Function to download text as Audio
def download_audio(text):
    tts = gTTS(text, lang="en")
    buffer = BytesIO()
    tts.write_to_fp(buffer)
    buffer.seek(0)
    return buffer

# Function to generate share links
def generate_share_links(text):
    encoded_text = urllib.parse.quote(text)
    return {
        "WhatsApp": f"https://wa.me/?text={encoded_text}",
        "Twitter": f"https://twitter.com/intent/tweet?text={encoded_text}",
        "Email": f"mailto:?subject=Text&body={encoded_text}",
        "LinkedIn": f"https://www.linkedin.com/sharing/share-offsite/?url={encoded_text}"
    }

# UI Setup
st.markdown("<h1 style='text-align: center;'>🚀 QuickText: Text Processor</h1>", unsafe_allow_html=True)

st.sidebar.title("⚡ Features")
option = st.sidebar.radio("Choose an option:", ["Process Single File", "History"])

if option == "Process Single File":
    st.markdown("<h3>📂 Upload a file or paste text.</h3>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx"])
    text = ""

    if uploaded_file:
        if uploaded_file.type == "application/pdf":
            text = extract_text_from_pdf(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text = extract_text_from_docx(uploaded_file)
    else:
        text = st.text_area("✍️ Paste your text here:", height=200)

    if text.strip():
        st.markdown("<h3>📌 Processed Text:</h3>", unsafe_allow_html=True)
        st.success(text)

        # Download Buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button("📄 PDF", download_pdf(text), file_name="processed_text.pdf", mime="application/pdf")
        with col2:
            st.download_button("📝 Word", download_word(text), file_name="processed_text.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        with col3:
            st.download_button("🔊 Audio", download_audio(text), file_name="processed_text.mp3", mime="audio/mp3")

        # Share Buttons
        st.markdown("<h3 style='text-align: center;'>📢 Share</h3>", unsafe_allow_html=True)
        share_links = generate_share_links(text)
        st.markdown(f"[WhatsApp]({share_links['WhatsApp']}) | [Twitter]({share_links['Twitter']}) | [Email]({share_links['Email']}) | [LinkedIn]({share_links['LinkedIn']})")

elif option == "History":
    st.subheader("📜 Text Processing History")
    st.write("(Feature coming soon!)")

st.markdown("<hr><p style='text-align: center;'>🔗 QuickText</p>", unsafe_allow_html=True)
