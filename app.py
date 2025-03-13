import streamlit as st
from transformers import pipeline
import torch
from googletrans import Translator, LANGUAGES
import pdfplumber
from docx import Document
import os

# Load Summarization Model
summarizer = pipeline("summarization")

# Function to detect language and translate text
def translate_text(text, target_lang="en"):
    translator = Translator()
    detected_lang = translator.detect(text).lang

    if detected_lang != target_lang:
        translated_text = translator.translate(text, src=detected_lang, dest=target_lang).text
        return translated_text, detected_lang
    return text, target_lang

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text.strip()

# Function to extract text from DOCX
def extract_text_from_docx(docx_file):
    doc = Document(docx_file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text.strip()

# Streamlit UI
st.title("🌍 Multi-Language Text Summarizer 📄")

st.subheader("Upload a PDF or DOCX file")
uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx"])

st.subheader("Or enter/paste text below")
user_input = st.text_area("Enter your text here:")

if st.button("Summarize"):
    if uploaded_file:
        file_type = uploaded_file.type

        if file_type == "application/pdf":
            text = extract_text_from_pdf(uploaded_file)
        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text = extract_text_from_docx(uploaded_file)
        else:
            st.error("Unsupported file format")
            st.stop()
    else:
        text = user_input

    if text:
        # Translate if needed
        translated_text, original_lang = translate_text(text, "en")

        # Summarize the translated text
        summary = summarizer(translated_text, max_length=150, min_length=50, do_sample=False)[0]['summary_text']

        # Translate back if necessary
        if original_lang != "en":
            summary = translate_text(summary, original_lang)[0]

        st.subheader("Summary:")
        st.write(summary)
    else:
        st.warning("Please enter text or upload a file.")
