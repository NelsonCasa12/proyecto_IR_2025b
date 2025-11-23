# Proyecto_RI — Sistema de Recuperación de Información

## Resumen
Proyecto para el curso de Recuperación de Información. Implementa:
- Preprocesamiento (NLTK + spaCy)
- Índice invertido implícito a través de TF-IDF
- Recuperación con 3 modelos: Jaccard (binario), TF-IDF (coseno) y BM25
- Interfaz web mínima (Flask) para probar consultas
- Evaluación con Precision, Recall y MAP

---

## Requisitos
- Python 3.8+
- Recomendado entorno virtual (venv)

## Estructura del Proyecto
proyecto_rdi/
│
├── data/
│ ├── corpus_climate_fever_preprocesado.csv
│ ├── queries.tsv
│ ├── queries_preprocessed.tsv
│ └── qrels.tsv
│
├── src/
│ ├── preprocessing.py
│ ├── retrieval.py
│ ├── evaluation.py
│ ├── generar_qrels_y_queries.py
│ ├── extract_corpus.py
│ ├── run_evaluation.py
│ └── web_app.py
│
├── notebooks/
│ └── quick_tests.ipynb
│
├── README.md
├── requirements.txt
└── Proyecto_RI_fixed.zip (versión corregida que generé)

### Instalación rápida
```bash
python -m venv venv
# macOS / Linux
source venv/bin/activate
# Windows (PowerShell)
venv\Scripts\Activate.ps1

pip install -r requirements.txt
python -m spacy download en_core_web_sm
