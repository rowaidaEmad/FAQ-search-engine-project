# Similar Question Search

A small Streamlit application that retrieves similar questions from the
[Banking77](https://huggingface.co/datasets/gtfintechlab/banking77) training
split.

It preserves the three retrieval methods from the original notebook:

- **Lexical:** preprocesses text and ranks questions with TF-IDF cosine
  similarity.
- **Semantic:** uses normalized embeddings from
  `sentence-transformers/all-MiniLM-L6-v2` and ranks questions by dot-product
  similarity.
- **Hybrid:** min-max normalizes both score arrays, then combines them using
  the notebook's weights: 0.4 TF-IDF and 0.6 semantic.

## Setup

Python 3.10 or newer is recommended. From this folder, install the required
packages:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

The first search downloads Banking77, the NLTK stop-word and WordNet data, and
the SentenceTransformer model. Later searches reuse Streamlit's cached dataset,
TF-IDF index, model, and corpus embeddings.

## Folder structure

```text
question_search/
├── app.py             # Streamlit interface and search routing
├── data_loader.py     # Banking77 loading and corpus preprocessing
├── search_engine.py   # Lexical, semantic, and hybrid retrieval
├── requirements.txt   # Runtime dependencies
└── README.md          # Setup and project overview
```

The Banking77 training split is the searchable corpus, matching the notebook.
The test split was used only for notebook evaluation and is not needed by the
interactive application.
