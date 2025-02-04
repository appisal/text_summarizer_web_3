import streamlit as st
from transformers import pipeline
import torch
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from docx import Document
from gtts import gTTS
from keybert import KeyBERT
from bs4 import BeautifulSoup
import requests
import spacy
from textstat import flesch_reading_ease

# Check for GPU availability
device = 0 if torch.cuda.is_available() else -1

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn", device=device)

@st.cache_resource
def load_sentiment_analyzer():
    return pipeline("sentiment-analysis", device=device)

nlp = spacy.load("en_core_web_sm")
summarizer = load_summarizer()
sentiment_analyzer = load_sentiment_analyzer()
keyword_extractor = KeyBERT()

def summarize_text(text, max_length, min_length, style="concise"):
    if len(text.split()) < min_length:
        return "Input text is too short to summarize."
    
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]["summary_text"]
    
    if style == "bullet":
        return "\n".join(["- " + sentence for sentence in summary.split('. ')])
    elif style == "detailed":
        return summary + "\n\n(This is a detailed summary.)"
    return summary

def extract_named_entities(text):
    doc = nlp(text)
    entities = {"People": [], "Organizations": [], "Locations": []}
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            entities["People"].append(ent.text)
        elif ent.label_ == "ORG":
            entities["Organizations"].append(ent.text)
        elif ent.label_ in ["GPE", "LOC"]:
            entities["Locations"].append(ent.text)
    return entities

def readability_score(text):
    return flesch_reading_ease(text)

def interactive_summary_expansion(summary):
    return "\n".join([f"- {sentence} \[Click to expand\]" for sentence in summary.split('. ')])

st.title("Text Summarizer - Enhanced")

st.sidebar.title("Features")
option = st.sidebar.radio(
    "Choose an option:",
    ["Single File", "Multiple Files", "URL", "Compare Texts"]
)

if option == "Single File":
    uploaded_file = st.file_uploader("Choose a .txt file", type=["txt"])
    if uploaded_file:
        text = uploaded_file.read().decode("utf-8")
    else:
        text = st.text_area("Paste your text here:", height=200)
    
    if text.strip():
        max_length = st.slider("Max summary length (words):", 50, 500, 200)
        min_length = st.slider("Min summary length (words):", 10, 100, 50)
        summary_style = st.selectbox("Summary Style", ["concise", "bullet", "detailed"])
        
        if st.button("Summarize"):
            summary = summarize_text(text, max_length, min_length, summary_style)
            st.subheader("Summary:")
            st.write(summary)
            
            sentiment = sentiment_analyzer(summary)[0]
            st.write(f"Sentiment: {sentiment['label']} (Confidence: {sentiment['score']:.2f})")
            
            entities = extract_named_entities(summary)
            st.write("Named Entities:", entities)
            
            score = readability_score(summary)
            st.write(f"Readability Score: {score:.2f} (Higher is easier to read)")
            
            interactive_summary = interactive_summary_expansion(summary)
            st.write("### Interactive Summary Expansion")
            st.write(interactive_summary)
            
            explanation = "AI summarizes by focusing on main ideas and discarding redundant information."
            st.write("### AI Explanation")
            st.write(explanation)
