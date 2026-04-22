import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from py.preprocessing import preprocessv2
from py.query_expansion import query_expand


def bm25_search(query, bm25, k=20):
    tokens = preprocessv2(query)
    scores = bm25.get_scores(tokens)
    ranked = np.argsort(scores)[-k:][::-1]
    return list(set(ranked))


def semantic_search(query, model_s, doc_embeds, embeddings, vocab, k=20):
    q_tokens = preprocessv2(query)
    exp_tokens = query_expand(query, model_s, embeddings, vocab)
    extra_tokens = [x for x in exp_tokens if x not in q_tokens]
    exp_text = " ".join(q_tokens + extra_tokens)

    vector = model_s.encode(exp_text)
    similarity = cosine_similarity([vector], doc_embeds)[0]
    top_docs = np.argsort(similarity)[-k:][::-1]
    return list(set(top_docs))


def hybrid_search(query, bm25, model_s, doc_embeds, embeddings, vocab, alpha=0.2, k=20):
    q_tokens = preprocessv2(query)
    exp_tokens = query_expand(query, model_s, embeddings, vocab)
    extra_tokens = [x for x in exp_tokens if x not in q_tokens]
    exp_text = " ".join(q_tokens + extra_tokens)

    bm25_scores = (
        1.0 * np.array(bm25.get_scores(q_tokens)) +
        0.3 * np.array(bm25.get_scores(extra_tokens))
    )

    vector = model_s.encode(exp_text)
    semantic_scores = cosine_similarity([vector], doc_embeds)[0]

    final = alpha * bm25_scores + (1 - alpha) * semantic_scores
    top_docs = np.argsort(final)[-k:][::-1]
    return list(set(top_docs))
