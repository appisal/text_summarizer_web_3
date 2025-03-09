import streamlit as st
import PyPDF2
import docx
import os
from gtts import gTTS
from io import BytesIO
import speech_recognition as sr
from pydub import AudioSegment
import tempfile

# ✅ Set Page Configuration (Must be the first command)
st.set_page_config(page_title="Text Summarizer", layout="wide")

# ✅ Page Title
st.title("📄 Text Summarizer")
st.write("Upload a text file, audio file, or paste text below to summarize it.")

# ✅ Organize layout using columns
col1, col2 = st.columns([2, 3])

# ✅ File upload section (Text, PDF, Word, Audio)
with col1:
    st.subheader("Choose a file to upload:")
    uploaded_file = st.file_uploader("Drag & drop or browse", type=["txt", "pdf", "docx", "mp3", "wav"])

    # Read text from file
    def extract_text_from_file(file):
        text = ""
        if file.name.endswith(".txt"):
            text = file.read().decode("utf-8")
        elif file.name.endswith(".pdf"):
            pdf_reader = PyPDF2.PdfReader(file)
            text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
        elif file.name.endswith(".docx"):
            doc = docx.Document(file)
            text = "\n".join([para.text for para in doc.paragraphs])
        return text

    # Read text from audio file
    def extract_text_from_audio(file):
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, file.name)
        with open(temp_path, "wb") as f:
            f.write(file.read())
        
        audio = AudioSegment.from_file(temp_path)
        audio.export(temp_path, format="wav")

        recognizer = sr.Recognizer()
        with sr.AudioFile(temp_path) as source:
            audio_data = recognizer.record(source)
            try:
                return recognizer.recognize_google(audio_data)
            except sr.UnknownValueError:
                return "Could not understand audio"
            except sr.RequestError:
                return "Speech recognition service unavailable"

    if uploaded_file:
        if uploaded_file.name.endswith((".txt", ".pdf", ".docx")):
            extracted_text = extract_text_from_file(uploaded_file)
        elif uploaded_file.name.endswith((".mp3", ".wav")):
            extracted_text = extract_text_from_audio(uploaded_file)
        else:
            extracted_text = ""
    else:
        extracted_text = ""

# ✅ Text Input Section
with col2:
    st.subheader("Paste your text here:")
    text_input = st.text_area("Enter text:", extracted_text, height=200)

# ✅ Summary Length Controls
st.subheader("Summary Length:")
col1, col2 = st.columns(2)
with col1:
    min_words = st.slider("Min words", 10, 100, 50)
with col2:
    max_words = st.slider("Max words", 50, 500, 200)

# ✅ Summarization Button
if st.button("Summarize", use_container_width=True):
    if not text_input:
        st.warning("⚠️ Please provide text to summarize.")
    else:
        summary = f"📝 [Summarized Text] (Mock Summary - Replace with actual logic)\n\n{text_input[:min_words]}..."
        st.success("✅ Summary Generated!")
        st.text_area("Summary:", summary, height=150)

        # ✅ Download as PDF
        pdf_bytes = BytesIO()
        pdf = docx.Document()
        pdf.add_paragraph(summary)
        pdf.save(pdf_bytes)
        pdf_bytes.seek(0)
        st.download_button("📥 Download as PDF", pdf_bytes, "summary.pdf", "application/pdf")

        # ✅ Download as Word
        doc_bytes = BytesIO()
        doc = docx.Document()
        doc.add_paragraph(summary)
        doc.save(doc_bytes)
        doc_bytes.seek(0)
        st.download_button("📥 Download as Word", doc_bytes, "summary.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")

        # ✅ Download as Audio (MP3)
        tts = gTTS(text=summary, lang="en")
        audio_bytes = BytesIO()
        tts.save(audio_bytes)
        audio_bytes.seek(0)
        st.download_button("🔊 Download as MP3", audio_bytes, "summary.mp3", "audio/mpeg")

