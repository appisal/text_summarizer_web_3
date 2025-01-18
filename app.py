import streamlit as st
from transformers import pipeline
import torch

# Check for GPU availability
device = 0 if torch.cuda.is_available() else -1

# Initialize the summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)

# Function to summarize text
def summarize_text(text, max_length, min_length):
    summary = summarizer(
        text,
        max_length=max_length,
        min_length=min_length,
        do_sample=False,
    )
    return summary[0]["summary_text"]

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

    # Generate Summary
    if st.button("Summarize"):
        try:
            summary = summarize_text(text, max_length, min_length)
            st.subheader("Summary:")
            st.write(summary)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Text Input Section
else:
    text = st.text_area("Paste your text here:", height=200)
    max_length = st.slider("Max summary length (words):", 50, 500, 200)
    min_length = st.slider("Min summary length (words):", 10, 100, 50)

    # Generate Summary
    if st.button("Summarize"):
        try:
            summary = summarize_text(text, max_length, min_length)
            st.subheader("Summary:")
            st.write(summary)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
