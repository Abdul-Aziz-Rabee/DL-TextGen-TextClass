import re
import os
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import nltk
# nltk.download("stopwords") # Descargar stopwords si no están disponibles
from nltk.corpus import stopwords

def cargar_corpus(path="canciones_clean.txt"):
    with open(path, "r", encoding="utf-8") as f:
        contenido = f.read()

    bloques = re.split(r"<\|startsong\|>", contenido)
    canciones = []
    for bloque in bloques:
        if "<|endsong|>" not in bloque:
            continue
        letra = bloque.split("<|endsong|>")[0].strip()
        if letra:
            canciones.append(letra)
    return canciones

def explorar_corpus(canciones):
    n_canciones = len(canciones)
    longitudes = [len(c.split()) for c in canciones]
    promedio = sum(longitudes) / n_canciones
    max_len = max(longitudes)
    min_len = min(longitudes)

    print("📊 Estadísticas del corpus")
    print(f"Total de canciones: {n_canciones}")
    print(f"Promedio de palabras por canción: {promedio:.2f}")
    print(f"Máximo: {max_len} palabras")
    print(f"Mínimo: {min_len} palabras")

    # Obtener stopwords de NLTK
    nltk_stopwords = set(stopwords.words("spanish"))

    todas_palabras = []
    for c in canciones:
        todas_palabras.extend(re.findall(r"\b\w+\b", c.lower()))

    # Top palabras más comunes (incluyendo stopwords)
    top_incluyendo = Counter(todas_palabras).most_common(20)
    df_incluyendo = pd.DataFrame(top_incluyendo, columns=["Palabra", "Frecuencia"])
    print("\n🔝 Top 20 palabras más frecuentes (incluyendo stopwords):")
    print(df_incluyendo.to_string(index=False))

    # Top palabras más comunes (excluyendo stopwords)
    palabras_filtradas = [p for p in todas_palabras if p not in nltk_stopwords]
    top_excluyendo = Counter(palabras_filtradas).most_common(20)
    df_excluyendo = pd.DataFrame(top_excluyendo, columns=["Palabra", "Frecuencia"])
    print("\n🔝 Top 20 palabras más frecuentes (sin stopwords):")
    print(df_excluyendo.to_string(index=False))

    # Histograma de longitudes
    plt.hist(longitudes, bins=20, color="skyblue", edgecolor="black")
    plt.title("Distribución de longitud de canciones (en palabras)")
    plt.xlabel("Número de palabras")
    plt.ylabel("Frecuencia")
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vis_path = os.path.join(base_path, "results","figures" ,"hist_longitudes.png")
    plt.savefig(vis_path)
    plt.show()

if __name__ == "__main__":
    import nltk
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PATH = os.path.join(base_path, "data", "canciones_clean.txt")
    canciones = cargar_corpus(PATH)
    explorar_corpus(canciones)
    print("Estadísticas impresas y histograma guardado como 'hist_longitudes.png'.")


