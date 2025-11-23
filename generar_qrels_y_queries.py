import ir_datasets
import pandas as pd
# Importamos la función para preprocesar las queries al final
from preprocessing import preprocess_queries_tsv 

# Cargar dataset
print("Cargando dataset beir/climate-fever para queries y qrels...")
dataset = ir_datasets.load("beir/climate-fever")

# Generar qrels.tsv
qrels = dataset.qrels_iter() # Itera sobre los juicios de relevancia
rows = []
for qrel in qrels:
    rows.append([qrel.query_id, 0, qrel.doc_id, qrel.relevance])

df_qrels = pd.DataFrame(rows, columns=["query_id", "iter", "doc_id", "relevance"])
df_qrels.to_csv("data/qrels.tsv", sep="\t", index=False)
print("Archivo qrels.tsv guardado en la carpeta 'data/'")

# Generar queries.tsv
queries = dataset.queries_iter()# Iterador sobre las consultas
query_rows = []
for query in queries:
    query_rows.append([query.query_id, query.text])

df_queries = pd.DataFrame(query_rows, columns=["query_id", "text"])
df_queries.to_csv("data/queries.tsv", sep="\t", index=False)
print("Archivo queries.tsv guardado en la carpeta 'data/'")

print("Generando queries_preprocessed.tsv (necesario para evaluación)...")
preprocess_queries_tsv("data/queries.tsv", "data/queries_preprocessed.tsv")