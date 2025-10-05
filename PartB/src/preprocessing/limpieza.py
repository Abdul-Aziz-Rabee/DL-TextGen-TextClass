# Modulo de Preprocesamiento para el corpus de texto

import pandas as pd
import re 
import os
import spacy #type: ignore
from nltk.stem import SnowballStemmer

# Carga spaCy solo una vez
try:
    nlp = spacy.load("es_core_news_sm")
except Exception:
    nlp = None
    print("spaCy 'es_core_news_sm' no está instalado.")

stemmer = SnowballStemmer("spanish")

def lematizar_o_stem(texto, metodo='lematizar'):
    """
    Aplica lematización (spaCy) o stemming (NLTK) al texto.
    metodo: "lematizar", "stem", o None
    """
    if metodo == "lematizar" and nlp is not None:
        doc = nlp(texto)
        return " ".join([token.lemma_ for token in doc])
    elif metodo == "stem":
        palabras = texto.split()
        return " ".join([stemmer.stem(p) for p in palabras])
    else:
        return texto

def reemplazar_mojibake_comun(texto):
    reemplazos = {
        'Ã²': 'ó',
        'Ã³': 'ó',
        'Ã¡': 'á',
        'Ã©': 'é',
        'Ã­': 'í',
        'Ãº': 'ú',
        'Ã±': 'ñ',
        'Ã': 'Á',
        'Ã‰': 'É',
        'Ã': 'Í',
        'Ã“': 'Ó',
        'Ãš': 'Ú',
        'Ã‘': 'Ñ',
        'maiorÃ': 'mayor',
        'Ãl': 'Él',
        'mayorÃ': 'mayoría',
        'Alimentosâ¦': 'Alimentos...',
        '»': '',
        '': '',
        'Â': ''
        # Agrega más según lo que observes
    }
    for malo, bueno in reemplazos.items():
        texto = texto.replace(malo, bueno)
    return texto

def fix_mojibake(texto):
    # Intenta decodificar con latin1->utf-8, si falla regresa el texto original
    try:
        return texto.encode('latin1').decode('utf-8')
    except:
        return texto

def limpiar_mojibake_ftfy(texto):
    try:
        import ftfy #type: ignore
        return ftfy.fix_text(texto)
    except ImportError:
        print("ftfy no está instalado. Regresando texto original.")
        return texto



def limpieza_basica(texto, metodo='ftfy'):
    # Reemplazar saltos de línea por espacios
    texto = texto.replace('\n', ' ')
    # Eliminar múltiples espacios
    texto = re.sub(r'\s+', ' ', texto)
    # Elimina patrones de reseñas truncadas tipo "...Más"
    texto = re.sub(r'\.{2,}\s*Más\b.*', '', texto, flags=re.IGNORECASE)
    # Limpiar mojibake usando el método seleccionado
    if metodo == 'ftfy':
        texto = limpiar_mojibake_ftfy(texto)
    else:
        texto = reemplazar_mojibake_comun(texto)
        texto = fix_mojibake(texto)
    return texto



def normalizar(texto):
    # Eliminar puntos, comas y signos de puntuación
    texto = re.sub(r'[^\w\s]', '', texto)
    # Eliminar digitos
    texto = re.sub(r'\d+', '', texto)
    # Convertir a minúsculas
    texto = texto.lower()
    return texto.strip()

def eliminar_stopwords(texto, stopwords_set):
    palabras = texto.split()
    return ' '.join([p for p in palabras if p not in stopwords_set])


def procesar_corpus(
    df,
    eliminar_stop=False,
    stopwords_set=None,
    normalizar_texto=False,
    lematizar_stem=False,
    metodo_lematizar='lematizar',
    custom_words=None
):
    df = df.copy()
    def procesar(texto):
        t = texto
        if normalizar_texto:
            t = normalizar(t)
        if eliminar_stop and stopwords_set is not None:
            t = eliminar_stopwords(t, stopwords_set)
        if custom_words is not None:
            t = ' '.join([w for w in t.split() if w.lower() not in custom_words])
        if lematizar_stem:
            t = lematizar_o_stem(t, metodo=metodo_lematizar)
        return t
    df['Review'] = df['Review'].astype(str).apply(procesar)
    return df


def leer_corpus(ruta, archivo, metodo='ftfy'):
    data_path = os.path.join(ruta, "data", "raw", archivo)
    corpus = pd.read_csv(data_path, encoding="utf-8")
    print('\nCorpus cargado. Realizando limpieza básica...')
    corpus['Review'] = corpus['Review'].astype(str).apply(lambda x: limpieza_basica(x, metodo=metodo))
    corpus['Polarity'] = corpus['Polarity'].astype(int)
    return corpus