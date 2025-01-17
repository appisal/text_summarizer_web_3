import streamlit as st
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import nltk

# Download punkt tokenizer
nltk.download('punkt')

# Add punkt tokenizer to the path if needed
nltk.data.path.append('./nltk_data')

# Function to summarize text
def summarize_text(text, num_sentences):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, num_sentences)
    summarized_text = " ".join(str(sentence) for sentence in summary)
    return summarized_text

# Streamlit App
st.title("Text Summarizer")

st.write("Upload a text file or paste text below to summarize it.")

# File Upload Section
uploaded_file = st.file_uploader("Choose a .txt file", type=["txt"])
if uploaded_file:
    text = uploaded_file.read().decode("utf-8")
    st.write("Uploaded Text:")
    st.text_area("Original Text", value=text, height=200)

    # Summary Slider
    num_sentences = st.slider("Number of sentences for the summary:", 1, 10, 3)

    # Generate Summary
    if st.button("Summarize"):
        summary = summarize_text(text, num_sentences)
        st.subheader("Summary:")
        st.write(summary)

# Text Input Section
else:
    text = st.text_area("Paste your text here:", height=200)
    num_sentences = st.slider("Number of sentences for the summary:", 1, 10, 3)

    # Generate Summary
    if st.button("Summarize"):
        summary = summarize_text(text, num_sentences)
        st.subheader("Summary:")
        st.write(summary)
