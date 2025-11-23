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

## Estructura del Proyecto

```text
proyecto_rdi/
│
├── data/                               # Datos de entrada y corpus
│   ├── corpus_climate_fever_preprocesado.csv
│   ├── queries.tsv                     # Consultas originales
│   ├── queries_preprocessed.tsv        # Consultas procesadas
│   └── qrels.tsv                       # Juicios de relevancia
│
├── src/                                # Código fuente principal
│   ├── preprocessing.py                # Limpieza y tokenización
│   ├── retrieval.py                    # Modelos (Jaccard, TF-IDF, BM25)
│   ├── evaluation.py                   # Métricas (Precision, Recall, MAP)
│   ├── generar_qrels_y_queries.py      # Scripts auxiliares
│   ├── extract_corpus.py               # Extracción de corpus
│   ├── run_evaluation.py               # Script maestro de evaluación
│   └── web_app.py                      # Interfaz web (Flask)
│
├── notebooks/
│   └── quick_tests.ipynb
│
├── README.md
```

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
