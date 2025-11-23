# evaluation.py
import pandas as pd
import numpy as np

def precision_at_k(retrieved, relevant, k):
    retrieved_k = retrieved[:k]
    return sum(1 for d in retrieved_k if d in relevant) / k

def recall_at_k(retrieved, relevant, k):
    retrieved_k = set(retrieved[:k])
    if len(relevant) == 0:
        return 0
    return len(retrieved_k & relevant) / len(relevant)

def average_precision(retrieved, relevant):
    hits = 0
    ap_sum = 0

    for i, d in enumerate(retrieved, 1):
        if d in relevant:
            hits += 1
            ap_sum += hits / i

    return ap_sum / hits if hits > 0 else 0

def mean_average_precision(results, qrels_df):
    """
    results = { query_id : [doc_ids ranked...] }
    """
    qrels = (
        qrels_df[qrels_df["relevance"] > 0]
        .groupby("query_id")["doc_id"]
        .apply(set)
        .to_dict()
    )

    ap = []

    for qid, retrieved in results.items():
        relevant = qrels.get(qid, set())
        ap.append(average_precision(retrieved, relevant))

    return sum(ap)/len(ap)

def load_qrels(path):
    return pd.read_csv(path, sep="\t", header=None,
                       names=["query_id", "iter", "doc_id", "relevance"])
