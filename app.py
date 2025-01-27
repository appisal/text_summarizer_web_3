import streamlit as st
from transformers import pipeline
import torch
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from docx import Document
import pandas as pd
from collections import Counter

# Check for GPU availability
device = 0 if torch.cuda.is_available() else -1

# Initialize the summarization pipeline
@st.cache_resource  # Cache the model to reduce loading time
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn", device=device)

summarizer = load_summarizer()

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

# Function to create a PDF
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

# Function to create a TXT file
def create_txt(summary):
    return BytesIO(summary.encode())

# Function to create a DOCX file
def create_docx(summary):
    doc = Document()
    doc.add_heading("Summary", level=1)
    doc.add_paragraph(summary)
    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer

# Function to create a CSV file
def create_csv(summary):
    buffer = BytesIO()
    df = pd.DataFrame({"Summary": [summary]})
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    return buffer

# Function to get advanced text statistics
def get_text_statistics(text):
    word_count = len(text.split())
    unique_words = len(set(text.split()))
    most_common_words = Counter(text.split()).most_common(5)
    return {
        "Word Count": word_count,
        "Unique Words": unique_words,
        "Most Common Words": most_common_words,
    }

# Streamlit App
st.set_page_config(page_title="Text Summarizer", page_icon="📄", layout="wide")

st.title("Text Summarizer")
st.write("Upload a text file or paste text below to summarize it.")

# Add Dark/Light Theme Toggle
theme = st.radio("Select Theme:", ["Light", "Dark"])
if theme == "Dark":
    st.markdown(
        """
        <style>
        body {
            background-color: #1e1e1e;
            color: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# File Upload Section
uploaded_file = st.file_uploader("Choose a .txt file", type=["txt"])
if uploaded_file:
    text = uploaded_file.read().decode("utf-8")
    st.write("Uploaded Text:")
    st.text_area("Original Text", value=text, height=200)

    # Text Statistics
    stats = get_text_statistics(text)
    st.write("Advanced Text Statistics:")
    st.write(f"Word Count: {stats['Word Count']}")
    st.write(f"Unique Words: {stats['Unique Words']}")
    st.write("Most Common Words:")
    for word, count in stats["Most Common Words"]:
        st.write(f"{word}: {count} times")

    # Summary Length Sliders
    max_length = st.slider("Max summary length (words):", 50, 500, 200)
    min_length = st.slider("Min summary length (words):", 10, 100, 50)

    if max_length <= min_length:
        st.error("Max length must be greater than Min length.")
    else:
        # Generate Summary
        if st.button("Summarize"):
            try:
                summary = summarize_text(text, max_length, min_length)
                st.subheader("Summary:")
                st.write(summary)

                # Language Translation
                lang = st.selectbox("Translate Summary to:", ["None", "Spanish", "French", "German", "Chinese"])
                if lang != "None":
                    from googletrans import Translator
                    translator = Translator()
                    translated_summary = translator.translate(summary, dest=lang[:2].lower()).text
                    st.write(f"Translated Summary ({lang}):")
                    st.write(translated_summary)

                # Download PDF Button
                pdf_data = create_pdf(summary)
                st.download_button(
                    label="Download Summary as PDF",
                    data=pdf_data,
                    file_name="summary.pdf",
                    mime="application/pdf",
                )

                # Download TXT Button
                txt_data = create_txt(summary)
                st.download_button(
                    label="Download Summary as TXT",
                    data=txt_data,
                    file_name="summary.txt",
                    mime="text/plain",
                )

                # Download DOCX Button
                docx_data = create_docx(summary)
                st.download_button(
                    label="Download Summary as DOCX",
                    data=docx_data,
                    file_name="summary.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )

                # Download CSV Button
                csv_data = create_csv(summary)
                st.download_button(
                    label="Download Summary as CSV",
                    data=csv_data,
                    file_name="summary.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# Text Input Section
else:
    text = st.text_area("Paste your text here:", height=200)

    # Word and Character Count
    if text.strip():
        st.write(f"Word Count: {len(text.split())}")
        st.write(f"Character Count: {len(text)}")

    max_length = st.slider("Max summary length (words):", 50, 500, 200)
    min_length = st.slider("Min summary length (words):", 10, 100, 50)

    if max_length <= min_length:
        st.error("Max length must be greater than Min length.")
    elif not text.strip():
        st.warning("Please provide some text to summarize.")
    else:
        # Generate Summary
        if st.button("Summarize"):
            try:
                summary = summarize_text(text, max_length, min_length)
                st.subheader("Summary:")
                st.write(summary)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
