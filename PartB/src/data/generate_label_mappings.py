#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
generate_label_mappings.py
Crea los mapeos de etiquetas originales (orden de clases)
a partir del dataset Rest-Mex_2025_train.csv, para replicar
el flujo de etiquetado del proyecto LLMs-sentiment-analysis-mx.
"""
import os
import pandas as pd
import json
from pathlib import Path

# === CONFIGURACIÓN ===
BASE_DIR = Path(__file__).resolve().parents[2]  # sube dos niveles (PartB/)


basedir = "C:\\Users\\uzgre\\Codes\\Python\\Projects\\LLMs-sentiment-analysis-mx\\data"
ruta = os.path.join(basedir, 'Rest-Mex_2025_train.csv')
DATA_PATH = Path(ruta)
#DATA_PATH = BASE_DIR / "data" / "raw" / "Rest-Mex_2025_train.csv"

OUT_DIR = BASE_DIR / "data" / "label_mappings"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"📂 Leyendo dataset base: {DATA_PATH}")

# === CARGA DEL DATASET ===
df = pd.read_csv(DATA_PATH)

# === POLARITY ===
# En el modelo original, la codificación era df['Polarity'] - 1,
# pero el orden de aparición fue el que dictó los índices internos.
polarity_labels = list(df["Polarity"].dropna().unique())
# Garantizar tipo float -> int
polarity_labels = [int(p) if not pd.isna(p) else None for p in polarity_labels]
print(f"🔢 Polarity classes ({len(polarity_labels)}): {polarity_labels}")

# === TYPE ===
type_labels, type_categories = pd.factorize(df["Type"])
type_labels_list = list(type_categories)
print(f"🏨 Type classes ({len(type_labels_list)}): {type_labels_list}")

# === TOWN ===
town_labels, town_categories = pd.factorize(df["Town"])
town_labels_list = list(town_categories)
print(f"🏙️ Town classes ({len(town_labels_list)}): {town_labels_list[:5]} ... ({len(town_labels_list)} total)")

# === GUARDAR EN JSON ===
mappings = {
    "polarity": polarity_labels,
    "type": type_labels_list,
    "town": town_labels_list
}

for name, values in mappings.items():
    out_file = OUT_DIR / f"{name}_labels_original.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(values, f, ensure_ascii=False, indent=2)
    print(f"✅ Guardado: {out_file}")

print("\n✨ Mapeos originales generados exitosamente.")
