from flask import Flask, request, render_template_string
import pandas as pd
import os

from preprocessing import preprocess_query
from retrieval import (
    build_tfidf_index, tfidf_search,
    build_bm25, bm25_search,
    jaccard_search
)

app = Flask(__name__)

DATA_PATH = "data/corpus_climate_fever_preprocesado.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"No existe {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

# identificar columna con texto preprocesado
text_col = None
for col in ["text_proc", "Texto_preprocesado", "preprocessed", "text"]:
    if col in df.columns:
        text_col = col
        break

if text_col is None:
    raise ValueError("No se encontró ninguna columna con texto preprocesado.")

# identificar columna original si existe
orig_text_col = None
for col in ["Texto_original", "text_original", "original"]:
    if col in df.columns:
        orig_text_col = col
        break

# columna Doc_ID
docid_col = None
for col in ["Doc_ID", "doc_id", "id"]:
    if col in df.columns:
        docid_col = col
        break

# lista de documentos preprocesados
docs = df[text_col].astype(str).tolist()

# construir TF-IDF y BM25
vectorizer, tfidf_matrix = build_tfidf_index(docs)
bm25_index = build_bm25([d.split() for d in docs])

# ---------------------------
# HTML con Bootstrap
# ---------------------------
HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Sistema de Recuperación de Información</title>
    <link rel="stylesheet"
          href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
    <style>
        body { background: #f5f7fa; }
        .container {
            max-width: 900px;
            margin-top: 40px;
            padding: 25px;
            background: white;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        }
        h2 { text-align: center; margin-bottom: 30px; }
        .result-box {
            background: #eef2f7;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 15px;
        }
        .query-info {
            background: #f0f0fd;
            padding: 10px;
            border-left: 5px solid #4c59cf;
            margin-bottom: 25px;
            border-radius: 4px;
        }
        .score {
            font-weight: bold;
            color: #364fc7;
        }
    </style>
</head>
<body>

<div class="container">
    <h2>🔎 Sistema de Recuperación de Información</h2>

    <form method="POST" class="row g-3 mb-4">
        <div class="col-12">
            <input class="form-control form-control-lg" name="q"
                   placeholder="Escribe tu consulta..." required>
        </div>

        <div class="col-6">
            <select class="form-select" name="method">
                <option value="tfidf">TF-IDF</option>
                <option value="bm25">BM25</option>
                <option value="jaccard">Jaccard</option>
            </select>
        </div>

        <div class="col-6">
            <button class="btn btn-primary w-100 btn-lg">Buscar</button>
        </div>
    </form>

    {% if q_original %}
    <div class="query-info">
        <p><b>Query original:</b> {{ q_original }}</p>
        <p><b>Query procesada:</b> <code>{{ q_procesada }}</code></p>
        <p><b>Método:</b> {{ metodo }}</p>
    </div>
    {% endif %}

    {% if results %}
        <h4>Resultados:</h4>
        <table class="table table-bordered">
            <thead class="table-secondary">
                <tr>
                    <th>Rank</th>
                    <th>Doc ID</th>
                    <th>Score</th>
                    <th>Texto</th>
                </tr>
            </thead>
            <tbody>
            {% for r in results %}
                <tr>
                    <td>{{ loop.index }}</td>
                    <td>{{ r.doc_id }}</td>
                    <td class="score">{{ r.score }}</td>
                    <td>
                        <b>Original:</b> {{ r.text_original }}<br>
                        <b>Preprocesado:</b> <i>{{ r.text_proc }}</i>
                    </td>
                </tr>
            {% endfor %}
            </tbody>
        </table>
    {% endif %}

</div>

</body>
</html>
"""


# ---------------------------
# ROUTE
# ---------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    q_original = None
    q_procesada = None
    metodo = None

    if request.method == "POST":
        q_original = request.form["q"]
        metodo = request.form["method"]

        q_procesada = preprocess_query(q_original)

        # ejecutar búsqueda
        if metodo == "tfidf":
            idxs, scores = tfidf_search(q_procesada, vectorizer, tfidf_matrix)
        elif metodo == "bm25":
            idxs, scores = bm25_search(q_procesada.split(), bm25_index)
        else:
            idxs, scores = jaccard_search(q_procesada, docs)

        # construir resultados enriquecidos
        for i, s in zip(idxs, scores):
            doc_original = df[orig_text_col].iloc[i] if orig_text_col else "No disponible"
            doc_id = df[docid_col].iloc[i] if docid_col else i

            results.append({
                "doc_id": doc_id,
                "score": round(float(s), 4),
                "text_original": str(doc_original),
                "text_proc": docs[i]
            })

    return render_template_string(
        HTML,
        results=results,
        q_original=q_original,
        q_procesada=q_procesada,
        metodo=metodo
    )


if __name__ == "__main__":
    app.run(debug=True)
