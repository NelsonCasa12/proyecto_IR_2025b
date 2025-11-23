# retrieval.py
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# =============================
# TF-IDF
# =============================

def build_tfidf_index(docs, **tfidf_kwargs):
    vect = TfidfVectorizer(**tfidf_kwargs)
    tfidf_matrix = vect.fit_transform(docs)
    return vect, tfidf_matrix

def tfidf_search(query, vectorizer, tfidf_matrix, top_k=10):
    qv = vectorizer.transform([query])
    scores = cosine_similarity(qv, tfidf_matrix).flatten()
    idxs = np.argsort(scores)[::-1][:top_k]
    return idxs, scores[idxs]

# =============================
# BM25 (implementación pura)
# =============================

def build_bm25(docs):
    N = len(docs)
    doc_len = np.array([len(d) for d in docs], dtype=np.float64)
    avgdl = doc_len.mean()

    term_freqs = []
    df = {}

    for d in docs:
        freqs = {}
        for t in d:
            freqs[t] = freqs.get(t, 0) + 1
        term_freqs.append(freqs)

        for t in freqs:
            df[t] = df.get(t, 0) + 1
    
    return {
        "N": N,
        "avgdl": avgdl,
        "doc_len": doc_len,
        "term_freqs": term_freqs,
        "df": df
    }

def bm25_score(tokens, bm25_index, k1=1.5, b=0.75):
    N = bm25_index["N"]
    avgdl = bm25_index["avgdl"]
    doc_len = bm25_index["doc_len"]
    term_freqs = bm25_index["term_freqs"]
    df = bm25_index["df"]

    scores = np.zeros(N)

    for term in tokens:
        if term not in df:
            continue
        # IDF estándar BM25
        idf = np.log(1 + (N - df[term] + 0.5) / (df[term] + 0.5))

        for i in range(N):
            f = term_freqs[i].get(term, 0)
            denom = f + k1*(1 - b + b*(doc_len[i]/avgdl))
            if denom > 0:
                scores[i] += idf * (f*(k1+1) / denom)
    return scores

def bm25_search(tokens, bm25_index, top_k=10):
    scores = bm25_score(tokens, bm25_index)
    idxs = np.argsort(scores)[::-1][:top_k]
    return idxs, scores[idxs]

# =============================
# Jaccard
# =============================

def jaccard_search(query, docs, top_k=10):
    vect = CountVectorizer(binary=True)
    X = vect.fit_transform(docs)
    q_vec = vect.transform([query]).toarray()[0]
    X_arr = X.toarray()

    intersection = (X_arr & q_vec).sum(axis=1)
    union = (X_arr | q_vec).sum(axis=1)

    with np.errstate(divide='ignore'):
        scores = np.where(union == 0, 0, intersection/union)

    idxs = np.argsort(scores)[::-1][:top_k]
    return idxs, scores[idxs]
