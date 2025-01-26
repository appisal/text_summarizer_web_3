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

# Check for GPU availability
device = 0 if torch.cuda.is_available() else -1

# Initialize the summarization pipeline
@st.cache_resource  # Cache the model to reduce loading time
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn", device=device)

summarizer = load_summarizer()

# Initialize KeyBERT for keyword extraction
@st.cache_resource
def load_keybert():
    return KeyBERT()

keybert_model = load_keybert()

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

# Function to extract keywords
def extract_keywords(text, top_n=5):
    keywords = keybert_model.extract_keywords(text, top_n=top_n)
    return [kw[0] for kw in keywords]

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

# Streamlit App
st.title("Text Summarizer")

st.write("Upload a text file or paste text below to summarize it.")

# File Upload Section
uploaded_file = st.file_uploader("Choose a .txt file", type=["txt"])
if uploaded_file:
    text = uploaded_file.read().decode("utf-8")
    st.write("Uploaded Text:")
    st.text_area("Original Text", value=text, height=200)

    # Word and Character Count
    st.write(f"Word Count: {len(text.split())}")
    st.write(f"Character Count: {len(text)}")

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

                # Extract Keywords
                st.subheader("Extracted Keywords:")
                keywords = extract_keywords(summary)
                st.write(", ".join(keywords))

                # Generate Word Cloud
                wordcloud = WordCloud(width=800, height=400, background_color="white").generate(summary)
                fig, ax = plt.subplots()
                ax.imshow(wordcloud, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig)

                # Download Options
                st.subheader("Download Files:")
                col1, col2, col3 = st.columns(3)

                # PDF Download Button
                with col1:
                    pdf_data = create_pdf(summary)
                    st.download_button(
                        label="Download PDF",
                        data=pdf_data,
                        file_name="summary.pdf",
                        mime="application/pdf",
                    )

                # DOCX Download Button
                with col2:
                    docx_data = create_docx(summary)
                    st.download_button(
                        label="Download DOCX",
                        data=docx_data,
                        file_name="summary.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    )

                # TXT Download Button
                with col3:
                    txt_data = create_txt(summary)
                    st.download_button(
                        label="Download TXT",
                        data=txt_data,
                        file_name="summary.txt",
                        mime="text/plain",
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

                # Extract Keywords
                st.subheader("Extracted Keywords:")
                keywords = extract_keywords(summary)
                st.write(", ".join(keywords))

                # Generate Word Cloud
                wordcloud = WordCloud(width=800, height=400, background_color="white").generate(summary)
                fig, ax = plt.subplots()
                ax.imshow(wordcloud, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig)

                # Download Options
                st.subheader("Download Files:")
                col1, col2, col3 = st.columns(3)

                # PDF Download Button
                with col1:
                    pdf_data = create_pdf(summary)
                    st.download_button(
                        label="Download PDF",
                        data=pdf_data,
                        file_name="summary.pdf",
                        mime="application/pdf",
                    )

                # DOCX Download Button
                with col2:
                    docx_data = create_docx(summary)
                    st.download_button(
                        label="Download DOCX",
                        data=docx_data,
                        file_name="summary.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    )

                # TXT Download Button
                with col3:
                    txt_data = create_txt(summary)
                    st.download_button(
                        label="Download TXT",
                        data=txt_data,
                        file_name="summary.txt",
                        mime="text/plain",
                    )

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
