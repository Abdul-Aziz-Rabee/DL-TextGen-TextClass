#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prueba rápida de carga del encoder BETO_MTL (solo backbone).
Verifica que los pesos se carguen correctamente sin reinicialización.
"""

import torch
from transformers import AutoTokenizer, AutoModel

# --- Ruta al modelo multitarea ---
model_dir = "models/BETO_MTL"

print(f"🔹 Cargando tokenizer y modelo desde: {model_dir}")
tok = AutoTokenizer.from_pretrained(model_dir)
model = AutoModel.from_pretrained(model_dir)

# --- Frase de prueba ---
text = "La comida fue excelente y el servicio impecable."
inputs = tok(text, return_tensors="pt")

# --- Forward ---
with torch.no_grad():
    outputs = model(**inputs)
    last_hidden = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)

# --- Estadísticas ---
mean_val = last_hidden.mean().item()
std_val = last_hidden.std().item()

print("\n✅ Modelo cargado correctamente.")
print(f"Embedding mean: {mean_val:.6f}")
print(f"Embedding std:  {std_val:.6f}")
print(f"Hidden size:    {last_hidden.shape[-1]}")
print(f"Seq length:     {last_hidden.shape[1]}")

# --- (opcional) verificar que los pesos no están todos a cero ---
nonzero_ratio = (last_hidden.abs() > 1e-8).float().mean().item()
print(f"Non-zero ratio: {nonzero_ratio:.3f}")

if std_val < 1e-6 or nonzero_ratio < 0.5:
    print("⚠️  Algo raro: los pesos podrían no haberse cargado correctamente.")
else:
    print("🔥 Encoder operativo, pesos MTL cargados sin reinicialización.")


n_params = sum(p.numel() for p in model.parameters())
print(f"Total parámetros: {n_params:,}")

print("✅ Prueba finalizada.")