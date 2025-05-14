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
from rank_bm25 import BM25Okapi

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
        'docno': [f"doc{i}" for i in range(len(sections))],
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
def initialize_tfidf(df):
    if df.empty:
        return None, None
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['processed_text'])
    return vectorizer, tfidf_matrix

# Initialize BM25
@st.cache_resource
def initialize_bm25(df):
    if df.empty:
        return None
    tokenized_corpus = [word_tokenize(text.lower()) for text in df['processed_text']]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25

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

# Preprocess query for search
def preprocess(query):
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    query = re.sub(r'[^a-zA-Z\s]', '', query.lower())
    words = word_tokenize(query)
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Search function with model selection
def search_constitution(query, model, df, tfidf_vectorizer, tfidf_matrix, bm25):
    processed = preprocess(query)
    if df.empty or (model == "TF-IDF" and (tfidf_vectorizer is None or tfidf_matrix is None)) or (model == "BM25" and bm25 is None):
        return {"query": query, "processed": processed, "results": []}

    if model == "TF-IDF":
        query_vec = tfidf_vectorizer.transform([processed])
        scores = tfidf_matrix.dot(query_vec.T).toarray().flatten()
        top_indices = np.argsort(scores)[::-1][:5]
        results = df.iloc[top_indices][['docno', 'text']].copy()
        results['score'] = scores[top_indices]
    else:  # BM25
        tokenized_query = word_tokenize(processed)
        scores = bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:5]
        results = df.iloc[top_indices][['docno', 'text']].copy()
        results['score'] = scores[top_indices]

    return {
        "query": query,
        "processed": processed,
        "results": results[['docno', 'score']].to_dict('records')
    }

# Term comparison across two search methods
def compare_terms(term1, term2, model1, model2, df, tfidf_vectorizer, tfidf_matrix, bm25, w2v_model):
    results = {}
    
    # Word2Vec similarity (original comparison)
    w2v_result = "Word2Vec similarity not available"
    if w2v_model is not None:
        try:
            vec1 = w2v_model.wv[term1.lower()]
            vec2 = w2v_model.wv[term2.lower()]
            similarity = 1 - spatial.distance.cosine(vec1, vec2)
            w2v_result = f"Word2Vec Similarity between '{term1}' and '{term2}': {similarity:.4f}"
        except KeyError:
            w2v_result = "One or both terms not in Constitution vocabulary (Word2Vec)"

    # Search-based comparison
    results["Word2Vec"] = w2v_result
    for term in [term1, term2]:
        results[term] = {}
        for model in [model1, model2]:
            search_results = search_constitution(term, model, df, tfidf_vectorizer, tfidf_matrix, bm25)
            results[term][model] = {
                "processed": search_results["processed"],
                "results": search_results["results"]
            }
    
    return results

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

    # Initialize data and search models
    df = load_and_preprocess_constitution(txt_path)
    tfidf_vectorizer, tfidf_matrix = initialize_tfidf(df)
    bm25 = initialize_bm25(df)
    w2v_model = train_word2vec(df)

    # Search Section
    st.header("Search the Constitution")
    query = st.text_input("Search Query (e.g., 'freedom speech')", key="search_query")
    search_model = st.radio(
        "Select Search Model",
        ["TF-IDF", "BM25"],
        key="search_model"
    )
    if st.button("Search"):
        if query:
            with st.spinner("Searching..."):
                results = search_constitution(query, search_model, df, tfidf_vectorizer, tfidf_matrix, bm25)
                st.subheader("Search Results")
                if results["results"]:
                    # Display query and processed query
                    st.markdown(f"**Original Query**: {results['query']}")
                    st.markdown(f"**Processed Query**: {results['processed']}")
                    # Format results as a table
                    results_df = pd.DataFrame(results["results"])
                    results_df = results_df.merge(df[['docno', 'text']], on='docno', how='left')
                    results_df = results_df.rename(columns={
                        "docno": "Document ID",
                        "text": "Constitution Text",
                        "score": "Relevance Score"
                    })
                    # Truncate text for display
                    results_df["Constitution Text"] = results_df["Constitution Text"].apply(
                        lambda x: x[:200] + "..." if len(x) > 200 else x
                    )
                    st.dataframe(results_df[['Document ID', 'Constitution Text', 'Relevance Score']], use_container_width=True)
                else:
                    st.warning("No results found. Try a different query or ensure the text file is valid.")
        else:
            st.warning("Please enter a search query.")

    # Term Comparison Section
    st.header("Compare Constitution Terms")
    term1 = st.text_input("First Term (e.g., 'amendment')", key="term1")
    term2 = st.text_input("Second Term (e.g., 'article')", key="term2")
    col1, col2 = st.columns(2)
    with col1:
        model1 = st.radio("Select First Search Model", ["TF-IDF", "BM25"], key="compare_model1")
    with col2:
        model2 = st.radio("Select Second Search Model", ["TF-IDF", "BM25"], key="compare_model2")
    if st.button("Compare"):
        if term1 and term2:
            if df.empty or (model1 == "TF-IDF" and (tfidf_vectorizer is None or tfidf_matrix is None)) or (model1 == "BM25" and bm25 is None):
                st.warning("Error: No valid Constitution data or search models available.")
            else:
                with st.spinner("Comparing terms..."):
                    results = compare_terms(term1, term2, model1, model2, df, tfidf_vectorizer, tfidf_matrix, bm25, w2v_model)
                    st.subheader("Comparison Results")
                    
                    # Display Word2Vec similarity
                    st.markdown(f"**{results['Word2Vec']}**")
                    
                    # Display search results for each term and model
                    for term in [term1, term2]:
                        st.markdown(f"### Results for '{term}'")
                        for model in [model1, model2]:
                            st.markdown(f"#### {model}")
                            if results[term][model]["results"]:
                                results_df = pd.DataFrame(results[term][model]["results"])
                                results_df = results_df.merge(df[['docno', 'text']], on='docno', how='left')
                                results_df = results_df.rename(columns={
                                    "docno": "Document ID",
                                    "text": "Constitution Text",
                                    "score": "Relevance Score"
                                })
                                results_df["Constitution Text"] = results_df["Constitution Text"].apply(
                                    lambda x: x[:200] + "..." if len(x) > 200 else x
                                )
                                st.dataframe(results_df[['Document ID', "Constitution Text", "Relevance Score"]], use_container_width=True)
                            else:
                                st.write("No results found.")
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
