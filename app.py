import streamlit as st
import PyPDF2
import docx
import speech_recognition as sr
import os
import tempfile
from transformers import pipeline
import gensim
from gensim.summarization import summarize

def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

def extract_text_from_audio(file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(file) as source:
        audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
    return text

def summarize_text(text, max_length, min_length):
    summarizer = pipeline("summarization")
    return summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']

def main():
    st.title("Text Summarizer")
    uploaded_file = st.file_uploader("Upload a file (PDF, DOCX, TXT, MP3, WAV)", type=["pdf", "docx", "txt", "mp3", "wav"])
    text = ""
    
    if uploaded_file:
        file_extension = uploaded_file.name.split(".")[-1]
        if file_extension == "pdf":
            text = extract_text_from_pdf(uploaded_file)
        elif file_extension == "docx":
            text = extract_text_from_docx(uploaded_file)
        elif file_extension in ["mp3", "wav"]:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_audio:
                temp_audio.write(uploaded_file.read())
                text = extract_text_from_audio(temp_audio.name)
                os.remove(temp_audio.name)
        else:
            text = uploaded_file.read().decode("utf-8")
    
    user_input = st.text_area("Paste your text here:", text)
    max_length = st.slider("Max summary length (words):", 50, 500, 200)
    min_length = st.slider("Min summary length (words):", 10, 100, 50)
    
    if st.button("Summarize"):
        if user_input:
            summarized_text = summarize_text(user_input, max_length, min_length)
            st.text_area("Summarized Text:", summarized_text, height=200)
            
            # Download options
            st.download_button("Download as TXT", summarized_text, file_name="summary.txt")
            doc = docx.Document()
            doc.add_paragraph(summarized_text)
            temp_docx = tempfile.NamedTemporaryFile(delete=False, suffix=".docx")
            doc.save(temp_docx.name)
            st.download_button("Download as DOCX", temp_docx.name, file_name="summary.docx")
            
            with open("summary.pdf", "wb") as f:
                pdf_writer = PyPDF2.PdfWriter()
                pdf_writer.add_page(PyPDF2.pdf.PageObject.create_blank_page(None, 600, 800))
                pdf_writer.write(f)
            st.download_button("Download as PDF", "summary.pdf", file_name="summary.pdf")
        else:
            st.warning("Please provide some text to summarize.")

if __name__ == "__main__":
    main()
