import pandas as pd
import re
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Descargar recursos necesarios de NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Cargar modelo de spaCy y recursos de NLTK
nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Función general de preprocesamiento de texto
def preprocess(text):
    text = text.lower()                                # Convertir a minúsculas
    text = re.sub(r'[^a-z\s]', '', text)               # Eliminar caracteres no alfabéticos
    tokens = nltk.word_tokenize(text)                  # Tokenizar texto
    tokens = [stemmer.stem(t) for t in tokens if t not in stop_words]  # Stemming y eliminación de stopwords
    doc = nlp(' '.join(tokens))                        # Procesamiento con spaCy
    lemmatized = [token.lemma_ for token in doc]       # Lematización
    return ' '.join(lemmatized)                        # Devolver texto limpio

# Preprocesar un corpus en formato CSV que contenga una columna llamada "Texto"
def preprocess_corpus(csv_path):
    df = pd.read_csv(csv_path)                         # Leer archivo CSV
    df["Texto_preprocesado"] = df["Texto"].apply(preprocess)  # Aplicar preprocesamiento
    return df

# Preprocesar un archivo TSV con queries (consulta) y guardar el resultado
def preprocess_queries_tsv(tsv_path, output_path="queries_preprocessed.tsv"):
    df = pd.read_csv(tsv_path, sep="\t")               # Leer archivo TSV
    df["text_proc"] = df["text"].apply(preprocess)     # Preprocesar columna 'text'
    df.to_csv(output_path, sep="\t", index=False)      # Guardar nuevo archivo TSV
    print(f"Archivo guardado como {output_path}")
    return df

# Preprocesar una sola consulta (string) manualmente
def preprocess_query(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [stemmer.stem(t) for t in tokens if t not in stop_words]
    doc = nlp(' '.join(tokens))
    lemmatized = [token.lemma_ for token in doc]
    return ' '.join(lemmatized)