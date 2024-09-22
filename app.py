from flask import Flask, render_template, request, jsonify
import os
from docx import Document
from pdfminer.high_level import extract_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
import nltk

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
DATABASE_FOLDER = 'database'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DATABASE_FOLDER'] = DATABASE_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATABASE_FOLDER, exist_ok=True)

def extract_text_from_file(file_path):
    """Extracts text from PDF or DOCX files."""
    if file_path.endswith('.pdf'):
        return extract_text(file_path)
    elif file_path.endswith(('.docx', '.doc')):
        doc = Document(file_path)
        text = [paragraph.text for paragraph in doc.paragraphs]
        return '\n'.join(text)
    else:
        return None

def calculate_similarity(text1, text2):
    """Calculates cosine similarity and highlights similar words/phrases."""

    # 1. Tokenize into Words:
    text1_words = nltk.word_tokenize(text1)
    text2_words = nltk.word_tokenize(text2)

    # 2. Calculate Similarity (using words)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([' '.join(text1_words), ' '.join(text2_words)])
    similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]

    # 3. Improved Highlighting (using SequenceMatcher on words):
    matcher = SequenceMatcher(None, text1_words, text2_words)

    highlighted_text1 = []
    highlighted_text2 = []
    for tag, a0, a1, b0, b1 in matcher.get_opcodes():
        if tag == 'equal':
            highlighted_text1.append(f"<mark>{' '.join(text1_words[a0:a1])}</mark>")
            highlighted_text2.append(f"<mark>{' '.join(text2_words[b0:b1])}</mark>")
        else:
            highlighted_text1.append(' '.join(text1_words[a0:a1]))
            highlighted_text2.append(' '.join(text2_words[b0:b1]))

    return similarity, ' '.join(highlighted_text1), ' '.join(highlighted_text2)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files['file']
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            input_text = extract_text_from_file(file_path)
            if input_text:
                similarity_results = []
                for filename in os.listdir(app.config['DATABASE_FOLDER']):
                    database_file_path = os.path.join(app.config['DATABASE_FOLDER'], filename)
                    database_text = extract_text_from_file(database_file_path)
                    if database_text:
                        similarity, highlighted_input, highlighted_database = calculate_similarity(input_text, database_text)
                        similarity_results.append({
                            "document": filename,
                            "similarity": round(similarity * 100, 2),
                            "highlighted_input": highlighted_input,
                            "highlighted_database": highlighted_database
                        })

                return render_template("results.html", results=similarity_results, input_text=input_text)
            else:
                return "Error: Could not extract text from the uploaded file."
        else:
            return "No file uploaded."
    else:
        return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)