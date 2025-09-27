import re
import os
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

def cargar_corpus(path="canciones_clean.txt"):
    with open(path, "r", encoding="utf-8") as f:
        contenido = f.read()

    # separar canciones por delimitadores
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
    # Estadísticas básicas
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

    # Top palabras más comunes (sin stopwords muy básicas)
    stopwords = {"de","la","que","el","y","a","en","un","una","los","las","se","del","al"}
    todas_palabras = []
    for c in canciones:
        todas_palabras.extend(re.findall(r"\b\w+\b", c.lower()))

    palabras_filtradas = [p for p in todas_palabras if p not in stopwords]
    top = Counter(palabras_filtradas).most_common(20)

    print("\n🔝 Top 20 palabras más frecuentes (sin stopwords):")
    for palabra, freq in top:
        print(f"{palabra}: {freq}")

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
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PATH = os.path.join(base_path, "data", "canciones_clean.txt")
    canciones = cargar_corpus(PATH)
    explorar_corpus(canciones)
    print("Estadísticas impresas y histograma guardado como 'hist_longitudes.png'.")


