# Proyecto_RI — Sistema de Recuperación de Información
**Curso:** Recuperación de Información  
**Profesor:** Iván Carrera  
**Entrega:** 26 de noviembre de 2025

---

## Descripción
Este repositorio implementa un sistema de recuperación de información que indexa documentos en texto plano y permite consultas de texto libre usando:
- Modelo binario (Jaccard)
- Modelo vectorial (TF-IDF + similitud coseno)
- Modelo probabilístico BM25

Incluye además scripts para preprocesamiento, generación de qrels/queries (BEIR), evaluación (Precision, Recall, MAP) y una interfaz web simple para probar búsquedas.

---

## Estructura del repositorio
proyecto_rdi/
│
├── data/
│ ├── corpus_climate_fever_preprocesado.csv # Corpus preprocesado (si lo tienes)
│ ├── queries.tsv # Queries originales (generado por script)
│ ├── queries_preprocessed.tsv # Queries preprocesadas
│ └── qrels.tsv # Juicios de relevancia (query_id, doc_id, relevance)
│
├── preprocessing.py # Preprocesamiento de texto
├── retrieval.py # TF-IDF, BM25 y Jaccard
├── evaluation.py # Precision, Recall, MAP
├── generar_qrels_y_queries.py # Genera queries/qrels desde BEIR
├── extract_corpus.py # (si lo incluyes) extrae/guarda corpus
├── web_app.py # Interfaz web (Flask) mínima
├── run_evaluation.py # Script opcional para ejecutar evaluación
├── README.md
└── requirements.txt


---

## Requisitos
- Python 3.8+  
- RAM/espacio según tamaño del corpus (BEIR climate-fever es moderado)
- Internet para descargar `ir_datasets` y el modelo spaCy (solo la primera vez)

Dependencias principales:
- numpy, pandas, scikit-learn, nltk, spacy, ir-datasets, flask

Instalación rápida:
```bash
python -m venv venv
# Linux / macOS
source venv/bin/activate
# Windows (PowerShell)
venv\Scripts\Activate.ps1

pip install -r requirements.txt
python -m spacy download en_core_web_sm
