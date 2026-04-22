import math


def relevance(ranked, qrels, k=10):
    return [qrels.get(doc, 0) for doc in ranked[:k]]


def dcg_f(rels, k=10):
    return sum(rels[i] / math.log2(i + 2) for i in range(min(k, len(rels))))


def ndcg(rels, k=10):
    dcg = dcg_f(rels, k)
    ideal = sorted(rels, reverse=True)
    idcg = dcg_f(ideal, k)
    return dcg / idcg if idcg > 0 else 0


def evaluation(results, qrels, k=10):
    scores = []
    for q in results:
        rels = relevance(results[q], qrels[q], k)
        score = ndcg(rels, k)
        scores.append(score)
    return sum(scores) / len(scores)
