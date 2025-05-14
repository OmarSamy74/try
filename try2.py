# -*- coding: utf-8 -*-
import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from scipy import spatial
import numpy as np

# Download required NLTK data
nltk.download('punkt_tab')
nltk.download('stopwords')


# Load and extract text from TXT
@st.cache_data
def load_txt_text(txt_path="constitution.txt"):
    try:
        with open(txt_path, 'r', encoding='utf-8') as file:
            text = file.read()
        if not text.strip():
            st.error("Error: constitution.txt is empty.")
            return ""
        return text
    except FileNotFoundError:
        st.error("Error: constitution.txt not found. Please ensure the file is in the same directory as this script.")
        return ""
    except Exception as e:
        st.error(f"Error loading text file: {str(e)}")
        return ""


# Preprocess and index the Constitution
@st.cache_data
def load_and_preprocess_constitution(txt_path="constitution.txt"):
    constitution_text = load_txt_text(txt_path)
    if not constitution_text:
        return pd.DataFrame(columns=['docno', 'text', 'processed_text'])

    # Split into sections (e.g., by double newlines)
    sections = [s.strip() for s in constitution_text.split('\n\n') if s.strip()]
    if not sections:
        sections = [constitution_text.strip()]  # Fallback to whole text as one section

    df = pd.DataFrame({
        'docno': range(len(sections)),
        'text': sections
    })

    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    def preprocess_text(text):
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        words = word_tokenize(text)
        words = [stemmer.stem(word) for word in words if word not in stop_words]
        return ' '.join(words)

    df['processed_text'] = df['text'].apply(preprocess_text)
    return df


# Initialize TF-IDF vectorizer
@st.cache_resource
def initialize_vectorizer(df):
    if df.empty:
        return None, None
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['processed_text'])
    return vectorizer, tfidf_matrix


# Train Word2Vec model
@st.cache_resource
def train_word2vec(df):
    if df.empty:
        return None
    sentences = [word_tokenize(text.lower()) for text in df['text']]
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
    return model


# ELMo embeddings
@st.cache_resource
def load_elmo():
    return hub.load("https://tfhub.dev/google/elmo/3")


def elmo_embed(text, elmo):
    emb = elmo.signatures["default"](tf.constant([text]))["elmo"]
    return emb.numpy()[0]


# Search function
def search_constitution(query):
    df = load_and_preprocess_constitution()
    if df.empty:
        return {"query": query, "processed_query": "", "results": []}

    vectorizer, tfidf_matrix = initialize_vectorizer(df)
    if vectorizer is None or tfidf_matrix is None:
        return {"query": query, "processed_query": "", "results": []}

    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    query = re.sub(r'[^a-zA-Z\s]', '', query.lower())
    query_words = word_tokenize(query)
    query_processed = ' '.join([stemmer.stem(word) for word in query_words if word not in stop_words])

    query_vec = vectorizer.transform([query_processed])
    scores = tfidf_matrix.dot(query_vec.T).toarray().flatten()
    top_indices = np.argsort(scores)[::-1][:5]
    results = df.iloc[top_indices][['docno', 'text']].copy()
    results['score'] = scores[top_indices]

    return {
        "query": query,
        "processed_query": query_processed,
        "results": results.to_dict('records')
    }


# Term comparison
def compare_terms(term1, term2):
    df = load_and_preprocess_constitution()
    w2v_model = train_word2vec(df)
    if w2v_model is None:
        return "Error: No valid Constitution data available for term comparison."
    try:
        vec1 = w2v_model.wv[term1.lower()]
        vec2 = w2v_model.wv[term2.lower()]
        similarity = 1 - spatial.distance.cosine(vec1, vec2)
        return f"Similarity between '{term1}' and '{term2}': {similarity:.4f}"
    except KeyError:
        return "One or both terms not in Constitution vocabulary"


# Text processing steps
def compare_processing(text):
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    cleaned = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    tokens = word_tokenize(cleaned)
    no_stopwords = [word for word in tokens if word not in stop_words]
    stemmed = [stemmer.stem(word) for word in no_stopwords]

    return {
        "original": text,
        "cleaned": cleaned,
        "tokenized": tokens,
        "stopwords_removed": no_stopwords,
        "stemmed": stemmed,
        "final": ' '.join(stemmed)
    }


# Streamlit UI
def main():
    st.title("📜 US Constitution Search Engine")
    st.write("Explore the US Constitution with search, term comparison, and text processing tools.")

    # File uploader for TXT
    st.header("Upload Constitution Text File (Optional)")
    uploaded_file = st.file_uploader("Upload constitution.txt (or use default)", type="txt")
    txt_path = "constitution.txt"

    if uploaded_file:
        with open(txt_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("Text file uploaded successfully!")

    # Search Section
    st.header("Search the Constitution")
    query = st.text_input("Search Query (e.g., 'freedom speech')", key="search_query")
    if st.button("Search"):
        if query:
            with st.spinner("Searching..."):
                results = search_constitution(query)
                st.subheader("Search Results")
                if results["results"]:
                    # Display query and processed query
                    st.markdown(f"**Original Query**: {results['query']}")
                    st.markdown(f"**Processed Query**: {results['processed_query']}")
                    # Format results as a table
                    results_df = pd.DataFrame(results["results"])
                    results_df = results_df.rename(columns={
                        "docno": "Document ID",
                        "text": "Constitution Text",
                        "score": "Relevance Score"
                    })
                    # Truncate text for display (optional, to avoid long outputs)
                    results_df["Constitution Text"] = results_df["Constitution Text"].apply(
                        lambda x: x[:200] + "..." if len(x) > 200 else x
                    )
                    st.dataframe(results_df, use_container_width=True)
                else:
                    st.warning("No results found. Ensure the text file is valid and contains text.")
        else:
            st.warning("Please enter a search query.")

    # Term Comparison Section
    st.header("Compare Constitution Terms")
    term1 = st.text_input("First Term (e.g., 'amendment')", key="term1")
    term2 = st.text_input("Second Term (e.g., 'article')", key="term2")
    if st.button("Compare"):
        if term1 and term2:
            result = compare_terms(term1, term2)
            st.write(result)
        else:
            st.warning("Please enter both terms.")

    # Text Processing Section
    st.header("Text Processing Steps")
    text_input = st.text_input("Input Text",
                               value="Congress shall make no law respecting an establishment of religion",
                               key="text_input")
    if st.button("Process Text"):
        results = compare_processing(text_input)
        st.json(results)

    # ELMo Embeddings Section
    st.header("ELMo Contextual Embeddings")
    elmo_text = st.text_input("Constitutional Text",
                              value="The right to bear arms shall not be infringed",
                              key="elmo_text")
    if st.button("Generate Embedding"):
        with st.spinner("Generating embedding..."):
            elmo = load_elmo()
            shape = str(elmo_embed(elmo_text, elmo).shape)
            st.write(f"ELMo Embedding Shape: {shape}")

    # RNN Concepts Section
    st.header("RNN Concepts")
    st.markdown("""
    **RNN Key Characteristics:**
    - Processes text sequentially
    - Maintains hidden state
    - Captures temporal dependencies
    - Used in this project for:
      * Sequence prediction (Article numbers)
      * Text generation
    """)


if __name__ == "__main__":
    main()