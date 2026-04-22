# Restaurant Information Retrieval (Arizona)

A search engine for Arizona restaurants built using the Yelp Open Dataset. Supports three retrieval methods — BM25, Semantic Search, and a Hybrid approach — with a Streamlit UI for interactive querying.

---

## Project Overview

This project indexes ~974 open Arizona restaurants from Yelp, combining their name, categories, ratings, and up to 20 reviews into a single searchable document per restaurant. Users can query the system in natural language and receive ranked results.

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the Yelp Dataset

Download the Yelp Open Dataset from https://business.yelp.com/data/resources/open-dataset/ and extract the following files into the `data/` folder:

```
data/
├── yelp_academic_dataset_business.json
└── yelp_academic_dataset_review.json
```

> The `data/` folder is in `.gitignore` and is not tracked by git.

### 3. Run the app

```bash
streamlit run app.py
```

This opens the search UI in your browser. On first load it builds the index and loads the embedding model — this takes a moment but is cached for the rest of the session.

---

## Retrieval Methods

| Method | Description |
|---|---|
| **BM25** | Keyword-based ranking using term frequency and inverse document frequency |
| **Semantic** | Dense vector search using `all-MiniLM-L6-v2` with query expansion |
| **Hybrid** | Weighted combination of BM25 and Semantic scores (20% BM25 / 80% Semantic) |

Semantic search is set as the default and achieved a high NDCG@10 score in evaluation and yielded best results when checking invdividually.

---

## Python Module Breakdown (`py/`)

### `py/preprocessing.py`
Two text preprocessing functions used throughout the pipeline:
- `preprocess` — lowercase, tokenize, remove stopwords and punctuation
- `preprocessv2` — adds regex tokenization and lemmatization (used for queries and indexing)

### `py/query_expansion.py`
Expands a query to allow for more vocabulary words to be searched for
- `query_expand` — encodes the query and finds the 3 nearest vocabulary words by cosine similarity using the sentence transformer model
- `syn_expand` — WordNet synonym expansion (tested but not used in final pipeline)

### `py/search.py`
The three core retrieval functions:
- `bm25_search` — tokenizes the query and scores documents using BM25Okapi
- `semantic_search` — expands the query, encodes it, and ranks by cosine similarity against document embeddings
- `hybrid_search` — scores with both methods and combines them with a weighted sum (80/20)

### `py/evaluation.py`
Relevance evaluation using NDCG@10:
- `relevance` — maps ranked doc IDs to their human-labelled relevance scores
- `dcg_f` — computes Discounted Cumulative Gain
- `ndcg` — normalizes DCG against the ideal ranking
- `evaluation` — averages NDCG@10 across all queries for a retrieval method

### `py/data_loader.py`
Handles all data loading and index construction:
- `load_data` — loads the preprocessed corpus and business DataFrame from `.pkl` files
- `build_index` — tokenizes documents, builds the inverted index, IDF table, and BM25 object
- `load_model` — loads the `all-MiniLM-L6-v2` sentence transformer
- `build_doc_embeddings` — encodes all documents into dense vectors
- `build_vocab_embeddings` — encodes the vocabulary for use in query expansion

---

## Notebook (`notebooks/retrieval.ipynb`)

The notebook documents the full research progression:

| Section | Description |
|---|---|
| **0 - Imports** | All library imports |
| **1 - Data Extraction** | Loading the Yelp JSONs, filtering for open Arizona restaurants with 5+ reviews, concatenating up to 20 reviews per business, building the document corpus |
| **2 - Preprocessing** | Developing and comparing two preprocessing pipelines (basic tokenization vs. lemmatization with regex) |
| **3 - Indexing & Scoring** | Building the inverted index, TF-IDF scoring, and initial BM25 testing |
| **4 - Query Expansion** | Embedding-based vocabulary expansion using `SentenceTransformer` and WordNet synonym expansion (tested and compared) |
| **5 - Semantic Search** | Dense retrieval using document embeddings and cosine similarity |
| **6 - Hybrid Scoring** | Combining BM25 and semantic scores with a tunable alpha weight |
| **7 - Relevance Labelling** | Manually grading results (0–4) across 9 test queries for all three methods |
| **8 - Evaluation** | NDCG@10 results — Hybrid: 0.948, Semantic: 0.945, BM25: 0.888 |
| **9 - UI Testing** | Initial Streamlit prototype built and tested inside the notebook |

---

## Results

Evaluated on 9 natural language queries (food preferences, price constraints, dietary restrictions, occasion-based):

| Method | NDCG@10 |
|---|---|
| BM25 | 0.8876 |
| Semantic | 0.9454 |
| **Hybrid** | **0.9482** | 

-- though Semantic will be used as default as it seemed to outperform Hybrid in my opinion with less obvious errors

---

## Data

| File | Description |
|---|---|
| `data/arizona_restuarant_corpus.pkl` | Preprocessed document corpus (Series of strings) |
| `data/az_dict.pkl` | Full business DataFrame with all metadata |
| `data/qrels.json` | Human relevance labels for evaluation queries |
