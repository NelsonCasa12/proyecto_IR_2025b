from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Función para buscar documentos usando el algoritmo TF-IDF
def tfidf_search(query, docs):
    vect = TfidfVectorizer()
    tfidf_matrix = vect.fit_transform(docs)
    query_vec = vect.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    return np.argsort(scores)[::-1], scores

# Función para buscar documentos usando el algoritmo BM25
# k1 y b son parámetros de ajuste del algoritmo BM25
def bm25_search(query, docs, k1=1.5, b=0.75):
    from sklearn.feature_extraction.text import CountVectorizer
    import numpy as np

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(docs)
    terms = vectorizer.get_feature_names_out()

    query_terms = query.lower().split()
    doc_len = X.sum(axis=1)
    avg_len = doc_len.mean()
    scores = []

    # Itera a través de cada documento en el corpus
    for i in range(X.shape[0]):
        score = 0
        for term in query_terms:
            if term in terms:
                df = np.count_nonzero(X[:, vectorizer.vocabulary_[term]].toarray())
                idf = np.log((X.shape[0] - df + 0.5) / (df + 0.5) + 1)
                tf = X[i, vectorizer.vocabulary_[term]]
                denom = tf + k1 * (1 - b + b * doc_len[i, 0] / avg_len)
                score += idf * (tf * (k1 + 1)) / denom
        # Agrega la puntuación total del documento a la lista de puntuaciones
        scores.append(score)
    # Devuelve los índices de los documentos ordenados por puntuación en orden descendente
    # y las puntuaciones originales
    return np.argsort(scores)[::-1], scores

# Modelo Jaccard
def jaccard_search(query, docs):
    """
    Búsqueda utilizando Similitud de Jaccard con Vectores Binarios.
    J(A,B) = |Intersección| / |Unión|
    """
    from sklearn.feature_extraction.text import CountVectorizer
    import numpy as np
    
    # 1. Crear vectores BINARIOS (binary=True hace que sea 0 o 1, ignorando frecuencia)
    vectorizer = CountVectorizer(binary=True)
    
    # X es la matriz documento-término (binaria)
    X = vectorizer.fit_transform(docs)
    
    # q_vec es el vector de la consulta (binario)
    q_vec = vectorizer.transform([query])
    
    # 2. Calcular Intersección (A ∩ B)
    # Matemáticamente, es el producto punto de los vectores binarios
    intersection = X.dot(q_vec.T).toarray().flatten()
    
    # 3. Calcular Unión (A ∪ B)
    # Unión = (Suma bits Doc) + (Suma bits Query) - Intersección
    doc_sums = X.sum(axis=1).A1  # Suma de 1s en cada documento
    query_sum = q_vec.sum()      # Suma de 1s en la query
    
    union = doc_sums + query_sum - intersection
    
    # 4. Calcular Coeficiente Jaccard
    # Evitamos división por cero agregando una pequeña epsilon o manejando ceros
    with np.errstate(divide='ignore', invalid='ignore'):
        scores = intersection / union
        scores[union == 0] = 0  # Si la unión es 0, la similitud es 0
    
    # 5. Retornar índices ordenados por score descendente
    return np.argsort(scores)[::-1], scores