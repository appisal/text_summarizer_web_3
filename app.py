import streamlit as st
from transformers import pipeline
import torch
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

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

# Streamlit App
st.title("Text Summarizer")

st.write("Upload a text file or paste text below to summarize it.")

# File Upload Section
uploaded_file = st.file_uploader("Choose a .txt file", type=["txt"])
if uploaded_file:
    text = uploaded_file.read().decode("utf-8")
    st.write("Uploaded Text:")
    st.text_area("Original Text", value=text, height=200)

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

                # Download PDF Button
                pdf_data = create_pdf(summary)
                st.download_button(
                    label="Download Summary as PDF",
                    data=pdf_data,
                    file_name="summary.pdf",
                    mime="application/pdf",
                )
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# Text Input Section
else:
    text = st.text_area("Paste your text here:", height=200)
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

                # Download PDF Button
                pdf_data = create_pdf(summary)
                st.download_button(
                    label="Download Summary as PDF",
                    data=pdf_data,
                    file_name="summary.pdf",
                    mime="application/pdf",
                )
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
