# preprocessing.py
import re
import pandas as pd
import nltk
import spacy
from nltk.corpus import stopwords

# Descargar recursos válidos
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Cargar spaCy
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise OSError(
        "El modelo SpaCy 'en_core_web_sm' no está instalado.\n"
        "Ejecute: python -m spacy download en_core_web_sm"
    )

stop_words = set(stopwords.words('english'))

def normalize_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess(text, remove_stopwords=True, do_lemmatize=True):
    text = normalize_text(text)
    tokens = nltk.word_tokenize(text)

    if remove_stopwords:
        tokens = [t for t in tokens if t not in stop_words and len(t) > 1]

    doc = nlp(" ".join(tokens))

    if do_lemmatize:
        lemmas = [token.lemma_.lower() for token in doc if token.lemma_ != '-PRON-']
        return " ".join(lemmas)
    else:
        return " ".join(tokens)

def preprocess_queries_tsv(tsv_path, output_path):
    df = pd.read_csv(tsv_path, sep="\t")
    
    # Detecta columna correcta
    if 'text' in df.columns:
        col = 'text'
    elif 'query' in df.columns:
        col = 'query'
    else:
        raise ValueError("El archivo debe tener columna 'text' o 'query'.")

    df["text_proc"] = df[col].astype(str).apply(preprocess)
    df.to_csv(output_path, sep="\t", index=False)
    print(f"Archivo guardado como {output_path}")
    return df

def preprocess_query(text):
    return preprocess(text)
