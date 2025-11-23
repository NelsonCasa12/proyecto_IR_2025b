from flask import Flask, request, render_template_string
from preprocessing import preprocess_query
from retrieval import tfidf_search, bm25_search
import pandas as pd
import importlib
import evaluation
importlib.reload(evaluation)

app = Flask(__name__)

# Carga del corpus de documentos preprocesados
df = pd.read_csv("data/corpus_arguana_preprocesado.csv")
df = df.dropna(subset=["Texto_preprocesado", "Texto_original"]).reset_index(drop=True)
docs = df["Texto_preprocesado"].tolist()
doc_ids = df["Doc_ID"].tolist()

# Carga de los Qrels
df_qrels = pd.read_csv("data/qrels.tsv", sep="\t")

# Carga y preprocesamiento de las consultas
query_texts = pd.read_csv("data/queries_preprocessed.tsv", sep="\t")
query_texts["text_proc"] = query_texts["text"].apply(preprocess_query)

HTML = """
<form method="post">
  <label>Consulta:</label><br>
  <input name="query" style="width: 300px;" value="{{ query }}"><br><br>
  <label>Método de búsqueda:</label><br>
  <select name="method">
    <option value="tfidf" {% if method == 'tfidf' %}selected{% endif %}>TF-IDF</option>
    <option value="bm25" {% if method == 'bm25' %}selected{% endif %}>BM25</option>
  </select><br><br>
  <input type="submit" value="Buscar">
</form>

{% if query %}
  <p><strong>Consulta ingresada (preprocesada):</strong> {{ query_pre }}</p>
{% endif %}

{% for i, score, doc in results %}
  <p><b>Doc ID: {{ i }} - Rank: {{ score }}</b></p>
  <p>{{ doc }}</p>
  <hr>
{% endfor %}
"""

@app.route("/", methods=["GET", "POST"])
def search():
    results = []
    query = ""
    method = "tfidf"
    query_id = ""
    query_pre = ""
    precision = recall = map_score = 0.0

    if request.method == "POST":
         # Se obtiene la consulta del formulario, método seleccionado y la consulta
        query = request.form["query"]
        method = request.form["method"]
        query_pre = preprocess_query(query)

        # Se intenta encontrar el ID de la consulta preprocesada
        matched = query_texts[query_texts["text_proc"].str.contains(query_pre, na=False)]
        if not matched.empty:
            query_id = matched.iloc[0]["query_id"]

         # Ejecuta la búsqueda según el método seleccionado
        if method == "bm25":
            ranked_indices, scores = bm25_search(query_pre, docs)
        else:
            ranked_indices, scores = tfidf_search(query_pre, docs)
        
        ranked_doc_ids = [doc_ids[i] for i in ranked_indices] # Se obtienen los IDs de documentos correspondientes a los índices rankeados

        # Evalua la búsqueda si hay consulta preprocesada
        if query_pre:
            try:
                print("Evaluate Iniciado")
                precision, recall, map_score = evaluation.evaluate(
                    "data/qrels.tsv", ranked_doc_ids, query_id=query_id, debug=True
                )
                print(f"[Evaluación] Query ID: {query_id}")
                print(f"Precisión: {precision:.3f} | Recall: {recall:.3f} | MAP: {map_score:.3f}")
            except Exception as e:
                print("ERROR en evaluate:", e)

    # Prepara los resultados para mostrar los top 10
        results = [(i, f"{scores[i]:.3f}", df['Texto_original'][i]) for i in ranked_indices[:10]]

    return render_template_string(
        HTML,
        results=results,
        query=query,
        query_pre=query_pre,
        method=method
    )

if __name__ == "__main__":
    app.run(debug=True)