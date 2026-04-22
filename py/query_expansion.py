import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import wordnet

from py.preprocessing import preprocessv2

#Embedding based Query Expansion - main
def query_expand(query, model_s, embeddings, vocab):
    q_tokens = preprocessv2(query)
    query_embedding = model_s.encode(query)
    prod = cosine_similarity([query_embedding], embeddings)[0]
    top = np.argsort(prod)[-3:][::-1]
    expansion = [vocab[i] for i in top if vocab[i] not in q_tokens]
    return q_tokens + expansion

#not used
def syn_expand(query):
    q_tokens = preprocessv2(query)
    expanded = set(q_tokens)
    for word in q_tokens:
        syns = wordnet.synsets(word)
        for syn in syns[:2]:
            for lemma in syn.lemmas():
                expanded.add(lemma.name().replace("_", " "))
    return list(expanded)
