import streamlit as st
from transformers import pipeline
import torch
from io import BytesIO
from gtts import gTTS
from keybert import KeyBERT
import urllib.parse  

# GPU Check
device = 0 if torch.cuda.is_available() else -1

# Load models
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn", device=device)

@st.cache_resource
def load_sentiment_analyzer():
    return pipeline("sentiment-analysis", device=device)

summarizer = load_summarizer()
sentiment_analyzer = load_sentiment_analyzer()
keyword_extractor = KeyBERT()

# Session state for history
if "summary_history" not in st.session_state:
    st.session_state.summary_history = []

# Function to summarize text
def summarize_text(text, max_length, min_length):
    if len(text.split()) < min_length:
        return "Input text is too short to summarize."
    
    with st.spinner("🔄 Summarizing... Please wait."):
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)

    st.session_state.summary_history.append(summary[0]["summary_text"])
    return summary[0]["summary_text"]

# Function to generate sharing links
def generate_share_links(summary):
    encoded_summary = urllib.parse.quote(summary)
    return {
        "📧 Email": f"mailto:?subject=Summary&body={encoded_summary}",
        "📱 WhatsApp": f"https://wa.me/?text={encoded_summary}",
        "🐦 Twitter": f"https://twitter.com/intent/tweet?text={encoded_summary}",
        "🔗 LinkedIn": f"https://www.linkedin.com/sharing/share-offsite/?url={encoded_summary}"
    }

# Streamlit App Header with Logo
st.image("https://in.images.search.yahoo.com/search/images;_ylt=Awrx.olk6c5nzJQyJ4O9HAx.;_ylu=c2VjA3NlYXJjaARzbGsDYnV0dG9u;_ylc=X1MDMjExNDcyMzAwNQRfcgMyBGZyA21jYWZlZQRmcjIDcDpzLHY6aSxtOnNiLXRvcARncHJpZANRWWhKX2tDeVF5eXplTnBZSi55NFpBBG5fcnNsdAMwBG5fc3VnZwMwBG9yaWdpbgNpbi5pbWFnZXMuc2VhcmNoLnlhaG9vLmNvbQRwb3MDMARwcXN0cgMEcHFzdHJsAzAEcXN0cmwDMzAEcXVlcnkDdGV4dCUyMHN1bW1hcml6ZXJsJTIwbG9nbyUyMHBuZyUyMGZyZWUEdF9zdG1wAzE3NDE2MTM0MTc-?p=text+summarizerl+logo+png+free&fr=mcafee&fr2=p%3As%2Cv%3Ai%2Cm%3Asb-top&ei=UTF-8&x=wrt&type=E210IN885G0#id=7&iurl=https%3A%2F%2Fsummarizer-ai.com%2F_next%2Fstatic%2Fmedia%2Flogo-main.eda27e57.png&action=click", width=100)  # Replace with your logo URL
st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>🚀 Text Summarizer</h1>", unsafe_allow_html=True)

st.sidebar.title("⚡ Features")
option = st.sidebar.radio("Choose an option:", ["Single File", "Summary History"])

# Single File Summarization
if option == "Single File":
    st.write("📂 Upload a text file or paste text below to summarize it.")
    uploaded_file = st.file_uploader("Choose a .txt file", type=["txt"])
    text = uploaded_file.read().decode("utf-8") if uploaded_file else st.text_area("✍️ Paste your text here:", height=200)

    if text.strip():
        max_length = st.slider("📏 Max summary length (words):", 50, 500, 200)
        min_length = st.slider("📏 Min summary length (words):", 10, 100, 50)

        if st.button("✨ Summarize", use_container_width=True):
            summary = summarize_text(text, max_length, min_length)

            if summary:
                st.subheader("📌 Summary:")
                st.success(summary)

                # Share Summary with Icons
                st.markdown("### 📢 Share Summary:")
                cols = st.columns(4)
                for col, (label, link) in zip(cols, generate_share_links(summary).items()):
                    if "WhatsApp" in label:
                        col.markdown(f'<a href="{link}" target="_blank"><img src="https://img.icons8.com/color/48/whatsapp.png"/></a>', unsafe_allow_html=True)
                    elif "Twitter" in label:
                        col.markdown(f'<a href="{link}" target="_blank"><img src="https://img.icons8.com/color/48/twitter.png"/></a>', unsafe_allow_html=True)
                    elif "Email" in label:
                        col.markdown(f'<a href="{link}" target="_blank"><img src="https://img.icons8.com/color/48/new-post.png"/></a>', unsafe_allow_html=True)
                    elif "LinkedIn" in label:
                        col.markdown(f'<a href="{link}" target="_blank"><img src="https://img.icons8.com/color/48/linkedin.png"/></a>', unsafe_allow_html=True)

# Summary History
elif option == "Summary History":
    st.subheader("📜 Summary History")
    if st.session_state.summary_history:
        for i, summary in enumerate(reversed(st.session_state.summary_history)):
            with st.expander(f"📄 Summary {len(st.session_state.summary_history) - i}"):
                st.write(summary)
    else:
        st.write("🔍 No previous summaries found.")

# Footer
st.markdown("<br><hr><p style='text-align: center;'>© 2025 Text Summarizer AI</p>", unsafe_allow_html=True)
