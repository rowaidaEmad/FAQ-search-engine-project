"""Load and prepare the Banking77 corpus used by the search engine."""

import re
from functools import lru_cache

import nltk
import pandas as pd
import streamlit as st
from datasets import load_dataset
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


DATASET_NAME = "gtfintechlab/banking77"


@lru_cache(maxsize=1)
def _get_preprocessing_tools() -> tuple[set[str], WordNetLemmatizer]:
    """Load the NLTK resources used by the original notebook."""
    try:
        stop_words = set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        stop_words = set(stopwords.words("english"))

    lemmatizer = WordNetLemmatizer()
    try:
        lemmatizer.lemmatize("questions")
    except LookupError:
        nltk.download("wordnet", quiet=True)

    return stop_words, lemmatizer


def preprocess_text(text: object) -> str:
    """Apply the notebook's lowercase, token, stop-word, and lemma steps."""
    if not isinstance(text, str):
        return ""

    stop_words, lemmatizer = _get_preprocessing_tools()
    tokens = re.findall(r"[a-z]+", text.lower())
    tokens = [
        word
        for word in tokens
        if len(word) > 1 and word not in stop_words
    ]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)


@st.cache_data(show_spinner=False)
def load_corpus() -> pd.DataFrame:
    """Load the Banking77 training split and prepare the searchable corpus."""
    train_dataset = load_dataset(DATASET_NAME, split="train")
    label_names = train_dataset.features["label"].names

    corpus_df = train_dataset.to_pandas().reset_index(drop=True)
    corpus_df["category"] = corpus_df["label"].map(
        lambda label_number: label_names[label_number]
    )
    corpus_df["clean_text"] = corpus_df["text"].apply(preprocess_text)

    return corpus_df[["text", "label", "category", "clean_text"]]
