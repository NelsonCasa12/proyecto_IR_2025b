import pandas as pd

def evaluate(qrels_path, retrieved_ids, query_id=None, top_k=10, debug=False):
    qrels = pd.read_csv(qrels_path, sep='\t', header=0)
    qrels["relevance"] = pd.to_numeric(qrels["relevance"], errors="coerce")
    qrels = qrels.dropna(subset=["relevance"])
    qrels["relevance"] = qrels["relevance"].astype(int)
    if query_id is not None:
         relevant_docs = set(qrels[(qrels["query_id"] == query_id) & (qrels["relevance"] > 0)]["doc_id"])
    else:
        relevant_docs = set(qrels[qrels["relevance"] > 0]["doc_id"])


    # Imprime información de depuración si debug es True
    if debug:
        print("¡DEBUG ACTIVADO!")
        if query_id is not None:
             print(f"Relevantes para query_id {query_id}: {relevant_docs}")
        else:
             print(f"Todos los relevantes en qrels: {relevant_docs}")
        print(f"Top recuperados: {retrieved_ids[:top_k]}")

    # Si no hay documentos relevantes en las qrels, las métricas son 0
    if not relevant_docs:
        return 0.0, 0.0, 0.0

    # Obtiene los IDs de los documentos recuperados en el top K
    retrieved_at_k = retrieved_ids[:top_k]
    # Identifica cuáles de los documentos recuperados en el top K son realmente relevantes
    retrieved_relevant = [doc for doc in retrieved_at_k if doc in relevant_docs]

    # Imprime información de depuración sobre los documentos relevantes recuperados
    if debug:
        print(f"Coincidencias relevantes encontradas: {retrieved_relevant}")

    # Calcula la Precisión: proporción de documentos relevantes entre los recuperados en el top K
    precision = len(retrieved_relevant) / top_k
    # Calcula el Recall: proporción de documentos relevantes recuperados entre todos los relevantes
    recall = len(retrieved_relevant) / len(relevant_docs)

    # Calcula el Mean Average Precision (MAP)
    # Si no se recuperaron documentos relevantes, el MAP es 0
    if not retrieved_relevant:
        map_score = 0.0
    else:
        map_score = sum([(i + 1) / (rank + 1) for i, rank in enumerate(
            [i for i, doc in enumerate(retrieved_at_k) if doc in relevant_docs])]) / len(relevant_docs)
    return precision, recall, map_score