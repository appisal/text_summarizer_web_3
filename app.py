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
import pandas as pd
import re

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

# Function to convert summary to audio
def create_audio(summary, language="en"):
    tts = gTTS(summary, lang=language)
    buffer = BytesIO()
    tts.write_to_fp(buffer)
    buffer.seek(0)
    return buffer

# Spell-checker
def suggest_corrections(text):
    words = text.split()
    corrections = [word for word in words if not re.match(r"^[a-zA-Z0-9]+$", word)]
    return corrections

# Streamlit App
st.title("Enhanced Text Summarizer")

st.write("Upload a text file or paste text below to summarize it. Supports `.txt` and `.docx` files.")

# File Upload Section
uploaded_file = st.file_uploader("Choose a .txt or .docx file", type=["txt", "docx"])
if uploaded_file:
    if uploaded_file.name.endswith(".txt"):
        text = uploaded_file.read().decode("utf-8")
    elif uploaded_file.name.endswith(".docx"):
        doc = Document(uploaded_file)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    st.write("Uploaded Text:")
    st.text_area("Original Text", value=text, height=200)

    # Word and Character Count
    st.write(f"Word Count: {len(text.split())}")
    st.write(f"Character Count: {len(text)}")

    # Summary Length Sliders
    max_length = st.slider("Max summary length (words):", 50, 500, 200)
    min_length = st.slider("Min summary length (words):", 10, 100, 50)

    # Language Selection for Audio
    language = st.selectbox("Select language for audio output:", options=["en", "es", "fr", "de", "it"])

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

                # Keyword Visualization
                keyword_counts = pd.DataFrame(keywords, columns=["Keyword"])
                keyword_counts["Count"] = keyword_counts["Keyword"].value_counts().values
                st.bar_chart(keyword_counts.set_index("Keyword"))

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

                # Thematic Word Cloud
                wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(keywords))
                fig, ax = plt.subplots()
                ax.imshow(wordcloud, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig)

                # Audio Output
                audio_data = create_audio(summary, language)
                st.audio(audio_data, format="audio/mp3", start_time=0)

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# Text Input Section
else:
    text = st.text_area("Paste your text here:", height=200)

    if text.strip():
        st.write(f"Word Count: {len(text.split())}")
        st.write(f"Character Count: {len(text)}")

        # Spell-check suggestion
        corrections = suggest_corrections(text)
        if corrections:
            st.warning(f"Possible issues found: {', '.join(corrections)}")

        max_length = st.slider("Max summary length (words):", 50, 500, 200)
        min_length = st.slider("Min summary length (words):", 10, 100, 50)

        language = st.selectbox("Select language for audio output:", options=["en", "es", "fr", "de", "it"])

        if max_length <= min_length:
            st.error("Max length must be greater than Min length.")
        else:
            if st.button("Summarize"):
                try:
                    summary = summarize_text(text, max_length, min_length)
                    st.subheader("Summary:")
                    st.write(summary)

                    # Extract Keywords
                    st.subheader("Extracted Keywords:")
                    keywords = extract_keywords(summary)
                    st.write(", ".join(keywords))

                    # Keyword Visualization
                    keyword_counts = pd.DataFrame(keywords, columns=["Keyword"])
                    keyword_counts["Count"] = keyword_counts["Keyword"].value_counts().values
                    st.bar_chart(keyword_counts.set_index("Keyword"))

                    # Download PDF Button
                    pdf_data = create_pdf(summary)
                    st.download_button(
                        label="Download Summary as PDF",
                        data=pdf_data,
                        file_name="summary.pdf",
                        mime="application/pdf",
                    )

                    # Thematic Word Cloud
                    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(keywords))
                    fig, ax = plt.subplots()
                    ax.imshow(wordcloud, interpolation="bilinear")
                    ax.axis("off")
                    st.pyplot(fig)

                    # Audio Output
                    audio_data = create_audio(summary, language)
                    st.audio(audio_data, format="audio/mp3", start_time=0)

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
