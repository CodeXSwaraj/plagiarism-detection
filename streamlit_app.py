import streamlit as st
import os
from docx import Document
from pdfminer.high_level import extract_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

UPLOAD_FOLDER = "uploads"
DATABASE_FOLDER = "database"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATABASE_FOLDER, exist_ok=True)


def extract_text_from_file(file_path):
    """Extracts text from PDF or DOCX files."""
    if file_path.endswith(".pdf"):
        return extract_text(file_path)
    elif file_path.endswith((".docx", ".doc")):
        doc = Document(file_path)
        text = [paragraph.text for paragraph in doc.paragraphs]
        return "\n".join(text)
    else:
        return None


def calculate_similarity(text1, text2):
    """Calculates cosine similarity and highlights similar sentences."""

    # 1. Tokenize into Sentences:
    text1_sentences = nltk.sent_tokenize(text1)
    text2_sentences = nltk.sent_tokenize(text2)

    # 2. Calculate Similarity (using sentences):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(text1_sentences + text2_sentences)
    similarity_matrix = cosine_similarity(
        tfidf_matrix[: len(text1_sentences)], tfidf_matrix[len(text1_sentences) :]
    )

    # 3.  Highlight Similar Sentences:
    highlighted_text1 = []
    highlighted_text2 = []
    for i, sentence1 in enumerate(text1_sentences):
        match_found = False
        for j, sentence2 in enumerate(text2_sentences):
            similarity = similarity_matrix[i][j]
            if similarity > 0.8:  # You can adjust the threshold
                highlighted_text1.append(f"<mark>{sentence1}</mark>")
                highlighted_text2.append(f"<mark>{sentence2}</mark>")
                match_found = True
                break
        if not match_found:
            highlighted_text1.append(sentence1)

    return (
        similarity_matrix.max(),
        " ".join(highlighted_text1),
        " ".join(highlighted_text2),
    )


# --- Streamlit UI ---
st.title("Plagiarism Detection App")

# Database Management Section: 
st.header("Manage Database PDFs")

database_files = os.listdir(DATABASE_FOLDER)
st.write(f"**Existing Database Files:** {', '.join(database_files) or 'None'}")

db_file = st.file_uploader("Add a PDF to the database", type=["pdf"])
if db_file is not None:
    # ... (The database file upload and deletion logic remains the same) 

st.markdown("---") # Visual separator

# Plagiarism Detection Section: 

st.header("Plagiarism Detection")

uploaded_file = st.file_uploader(
    "Upload your document (PDF or DOCX)", type=["pdf", "docx"]
)

if uploaded_file is not None:
    # Save the uploaded file temporarily
    file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    input_text = extract_text_from_file(file_path)

    if input_text:
        similarity_results = []
        for filename in os.listdir(DATABASE_FOLDER):
            database_file_path = os.path.join(DATABASE_FOLDER, filename)
            database_text = extract_text_from_file(database_file_path)
            if database_text:
                (
                    similarity,
                    highlighted_input,
                    highlighted_database,
                ) = calculate_similarity(input_text, database_text)
                similarity_results.append(
                    {
                        "document": filename,
                        "similarity": round(similarity * 100, 2),
                        "highlighted_input": highlighted_input,
                        "highlighted_database": highlighted_database,
                    }
                )

        st.subheader("Plagiarism Results:")
        for result in similarity_results:
            st.write(f"**Document:** {result['document']}")
            st.write(f"**Similarity:** {result['similarity']}%")
            st.markdown(
                f"**Highlighted Input Text:** {result['highlighted_input']}",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"**Highlighted Database Text:** {result['highlighted_database']}",
                unsafe_allow_html=True,
            )
            st.markdown("---")

    else:
        st.error("Error: Could not extract text from the uploaded file.")