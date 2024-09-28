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

    # 3. Find matching sentence indexes:
    matching_indexes = []
    for i in range(len(text1_sentences)):
        for j in range(len(text2_sentences)):
            if similarity_matrix[i][j] > 0.8:  # You can adjust this threshold
                matching_indexes.append((i, j))

    # 4. Highlight matching sentences:
    highlighted_text1 = []
    highlighted_text2 = []
    for i, sentence in enumerate(text1_sentences):
        if any(i == index1 for index1, _ in matching_indexes):
            highlighted_text1.append(f"<mark>{sentence}</mark>")
        else:
            highlighted_text1.append(sentence)

    for j, sentence in enumerate(text2_sentences):
        if any(j == index2 for _, index2 in matching_indexes):
            highlighted_text2.append(f"<mark>{sentence}</mark>")
        else:
            highlighted_text2.append(sentence)

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
    file_path = os.path.join(DATABASE_FOLDER, db_file.name)
    with open(file_path, "wb") as f:
        f.write(db_file.getbuffer())
    st.success(f"File '{db_file.name}' added to database.")
    st.experimental_rerun() 

# Delete PDF from the database
file_to_delete = st.selectbox("Select a PDF to delete", database_files)
if st.button("Delete PDF"):
    file_path = os.path.join(DATABASE_FOLDER, file_to_delete)
    if os.path.exists(file_path):
        os.remove(file_path)
        st.success(f"File '{file_to_delete}' deleted from database.")
        st.experimental_rerun()
    else:
        st.error(f"File '{file_to_delete}' not found in the database.")

    st.markdown("---")  # Visual separator (Correct indentation)

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

    
