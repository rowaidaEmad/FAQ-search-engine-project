"""Reusable lexical, semantic, and hybrid search for Banking77."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import spmatrix
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from data_loader import load_corpus, preprocess_text


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TFIDF_WEIGHT = 0.4
SEMANTIC_WEIGHT = 0.6


@dataclass
class SearchResources:
    corpus: pd.DataFrame
    tfidf_vectorizer: TfidfVectorizer
    corpus_tfidf_matrix: spmatrix
    semantic_model: SentenceTransformer
    corpus_embeddings: np.ndarray


@st.cache_resource(show_spinner=False)
def get_search_resources() -> SearchResources:
    """Build each expensive search resource once per Streamlit cache lifecycle."""
    corpus = load_corpus()

    tfidf_vectorizer = TfidfVectorizer()
    corpus_tfidf_matrix = tfidf_vectorizer.fit_transform(corpus["clean_text"])

    semantic_model = SentenceTransformer(MODEL_NAME)
    corpus_embeddings = semantic_model.encode(
        corpus["text"].tolist(),
        batch_size=64,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    return SearchResources(
        corpus=corpus,
        tfidf_vectorizer=tfidf_vectorizer,
        corpus_tfidf_matrix=corpus_tfidf_matrix,
        semantic_model=semantic_model,
        corpus_embeddings=corpus_embeddings,
    )


def _validate_query(query: object) -> str:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("The query must be a non-empty string.")
    return query.strip()


def _validate_top_k(top_k: int, corpus_size: int) -> int:
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")
    return min(top_k, corpus_size)


def _format_results(
    resources: SearchResources,
    top_indices: np.ndarray,
    scores: np.ndarray,
    method: str,
) -> pd.DataFrame:
    results = resources.corpus.iloc[top_indices][["text", "category"]].copy()
    results.insert(0, "rank", range(1, len(results) + 1))
    results["score"] = scores[top_indices]
    results["method"] = method
    return results.reset_index(drop=True)


def tfidf_search(query: str, top_k: int = 5) -> pd.DataFrame:
    """Return the Top-N questions by TF-IDF cosine similarity."""
    query = _validate_query(query)
    clean_query = preprocess_text(query)
    if not clean_query:
        raise ValueError(
            "The query is empty after preprocessing. Please use more descriptive words."
        )

    resources = get_search_resources()
    top_k = _validate_top_k(top_k, len(resources.corpus))
    query_vector = resources.tfidf_vectorizer.transform([clean_query])
    scores = cosine_similarity(
        query_vector, resources.corpus_tfidf_matrix
    ).flatten()
    top_indices = np.argsort(scores)[::-1][:top_k]

    return _format_results(resources, top_indices, scores, "TF-IDF")


def semantic_search(query: str, top_k: int = 5) -> pd.DataFrame:
    """Return the Top-N questions by normalized MiniLM embedding similarity."""
    query = _validate_query(query)
    resources = get_search_resources()
    top_k = _validate_top_k(top_k, len(resources.corpus))

    query_embedding = resources.semantic_model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    scores = resources.corpus_embeddings @ query_embedding
    top_indices = np.argsort(scores)[::-1][:top_k]

    return _format_results(
        resources, top_indices, scores, "SentenceTransformer"
    )


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Min-max normalize scores exactly as in the notebook."""
    score_range = scores.max() - scores.min()
    if score_range == 0:
        return np.zeros_like(scores)
    return (scores - scores.min()) / score_range


def hybrid_search(
    query: str,
    top_k: int = 5,
    tfidf_weight: float = TFIDF_WEIGHT,
    semantic_weight: float = SEMANTIC_WEIGHT,
) -> pd.DataFrame:
    """Combine normalized TF-IDF and semantic scores with 0.4/0.6 weights."""
    query = _validate_query(query)
    if not np.isclose(tfidf_weight + semantic_weight, 1.0):
        raise ValueError("TF-IDF and semantic weights must add up to 1.")

    clean_query = preprocess_text(query)
    if not clean_query:
        raise ValueError(
            "The query is empty after preprocessing. Please use more descriptive words."
        )

    resources = get_search_resources()
    top_k = _validate_top_k(top_k, len(resources.corpus))

    query_tfidf_vector = resources.tfidf_vectorizer.transform([clean_query])
    tfidf_scores = cosine_similarity(
        query_tfidf_vector, resources.corpus_tfidf_matrix
    ).flatten()

    query_embedding = resources.semantic_model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    semantic_scores = resources.corpus_embeddings @ query_embedding

    hybrid_scores = (
        tfidf_weight * normalize_scores(tfidf_scores)
        + semantic_weight * normalize_scores(semantic_scores)
    )
    top_indices = np.argsort(hybrid_scores)[::-1][:top_k]

    results = _format_results(
        resources, top_indices, hybrid_scores, "Hybrid Weighted"
    )
    results["tfidf_score"] = tfidf_scores[top_indices]
    results["semantic_score"] = semantic_scores[top_indices]
    return results
