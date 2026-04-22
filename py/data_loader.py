import math
from collections import defaultdict

import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from py.preprocessing import preprocessv2


def load_data(corpus_path='data/arizona_restuarant_corpus.pkl', docs_path='data/az_dict.pkl'):
    documents = pd.read_pickle(corpus_path)
    docs = pd.read_pickle(docs_path)
    return documents, docs


def build_index(documents):
    tokens = [preprocessv2(doc) for doc in documents]

    invert_index = defaultdict(dict)
    for doc_id, token_list in enumerate(tokens):
        for term in token_list:
            invert_index[term][doc_id] = invert_index[term].get(doc_id, 0) + 1

    N = len(tokens)
    idf = {term: math.log(N / len(postings)) for term, postings in invert_index.items()}

    bm25 = BM25Okapi(tokens)
    return tokens, invert_index, idf, bm25


def load_model(model_name='all-MiniLM-L6-v2'):
    return SentenceTransformer(model_name)


def build_doc_embeddings(model_s, documents, show_progress=True):
    return model_s.encode(list(documents), show_progress_bar=show_progress)


def build_vocab_embeddings(model_s, invert_index):
    vocab = list(invert_index.keys())
    embeddings = model_s.encode(vocab)
    return vocab, embeddings
