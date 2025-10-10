#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
prepare_meia_transformer.py
Prepara el dataset MeIA_2025 para su uso con modelos Transformer (BETO, MarIA, etc.)
Compatible con Hugging Face Datasets y el pipeline de entrenamiento LoRA.
"""

import os
import pandas as pd
from datasets import Dataset
from src.preprocessing.limpieza import leer_corpus, procesar_corpus


def load_meia_for_transformer(
    csv_path: str,
    eliminar_stop: bool = False,
    normalizar_texto: bool = True,
    lematizar_stem: bool = False,
    metodo_lematizar: str = "lematizar",
    verbose: bool = True
) -> Dataset:
    """
    Carga, limpia y convierte el corpus MeIA_2025 en un Dataset de Hugging Face.
    Args:
        csv_path: ruta absoluta o relativa al archivo CSV del corpus.
        eliminar_stop: eliminar stopwords en español (por defecto False).
        normalizar_texto: aplicar normalización (minúsculas, sin signos, sin dígitos).
        lematizar_stem: aplicar lematización o stemming según método.
        metodo_lematizar: 'lematizar' (spaCy) o 'stem' (SnowballStemmer).
        verbose: mostrar progreso por consola.
    Returns:
        datasets.Dataset listo para tokenización con Hugging Face.
    """

    base_dir = os.path.dirname(csv_path)
    archivo = os.path.basename(csv_path)

    if verbose:
        print(f"📂 Cargando corpus desde: {csv_path}")

    # === 1. Carga con limpieza básica (ftfy + mojibake fix) ===
    df = leer_corpus(base_dir, archivo, metodo="ftfy")

    # === 2. Limpieza semántica y normalización ===
    '''
    df = procesar_corpus(
        df,
        eliminar_stop=eliminar_stop,
        normalizar_texto=normalizar_texto,
        lematizar_stem=lematizar_stem,
        metodo_lematizar=metodo_lematizar,
        custom_words={"hotel", "restaurante", "comida", "servicio", "playa", "lugar"}
    )
    '''

    # === 3. Formato final (texto + etiqueta) ===
    df = df[["Review", "Polarity"]].copy()
    df = df.rename(columns={"Review": "text", "Polarity": "label"})

    # Asegurar tipo entero 0–4 (originalmente 1–5)
    df["label"] = df["label"].astype(int) - 1

    if verbose:
        print("✅ Limpieza completada.")
        print(df.head(2))

    # === 4. Conversión a Dataset de Hugging Face ===
    hf_dataset = Dataset.from_pandas(df)
    if verbose:
        print(f"✅ Dataset Hugging Face creado: {len(hf_dataset)} muestras.")

    return hf_dataset


if __name__ == "__main__":
    # Ejemplo de uso rápido
    dataset = load_meia_for_transformer("data/raw/MeIA_2025_train.csv")
    print(dataset)
