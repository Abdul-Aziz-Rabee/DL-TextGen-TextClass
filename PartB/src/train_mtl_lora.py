#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_mtl_lora.py
Entrenamiento multitarea (Polarity, Type, Town) sobre el dataset MeIA 2025,
reutilizando el modelo BETO_MTL preentrenado e inyectando LoRA sólo al encoder.

Autor: Uziel Luján (Uzi)
Proyecto: DL-TextGen-TextClass — Parte B, Fase Transformers
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
print(f"🔹 GPUs visibles: {torch.cuda.device_count()}")
import numpy as np
from transformers import (
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from sklearn.metrics import f1_score
from datasets import Dataset
from data.prepare_meia_mtl import load_and_prepare_meia_for_mtl
from src.eval.eval_utils_mtl import compute_and_save_mtl_metrics
from src.train.train_mtl import MultiTaskModel, MultiTaskTrainer  # del proyecto original
from peft import get_peft_model

# ============================================================
# === CONFIGURACIÓN BÁSICA ==================================
# ============================================================

MODEL_DIR = "models/BETO_MTL"           # Modelo multitarea original
RUN_NAME = "BETO_MTL_LoRA_MeIA"         # Nombre del experimento actual
MAX_LEN = 256
EPOCHS = 3
BATCH_SIZE = 16
LR = 2e-5
USE_WANDB = False                       # Cambiar a True si deseas loggear

# ============================================================
# === 1. CARGA DE DATOS Y TOKENIZACIÓN =======================
# ============================================================

print("\n📥 Cargando dataset MeIA para multitarea...")
data = load_and_prepare_meia_for_mtl()
train_dataset, eval_dataset = data["train"], data["eval"]
label_mappings = data["label_mappings"]

print("🔤 Cargando tokenizador...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
    )

print("⚙️ Tokenizando datasets...")
train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# ============================================================
# === 2. CARGA DEL MODELO MULTITAREA =========================
# ============================================================

from transformers import AutoConfig
from safetensors.torch import load_file
import torch

print("\n🧠 Cargando modelo BETO_MTL preentrenado...")

# ------------------------------------------------------------
# Cargar configuración del modelo base BETO
# ------------------------------------------------------------
config = AutoConfig.from_pretrained(MODEL_DIR)

# Crear instancia explícita del modelo multitarea
# (evita que Transformers cargue erróneamente un BertModel plano)
model = MultiTaskModel(
    config=config,
    model_name=None,  # No cargar pesos aquí
    num_labels_polarity=len(label_mappings["polarity"]),
    num_labels_type=len(label_mappings["type"]),
    num_labels_town=len(label_mappings["town"]),
)

# ------------------------------------------------------------
# Cargar pesos .safetensors del modelo multitarea
# ------------------------------------------------------------

state_path = os.path.join(MODEL_DIR, "model.safetensors")
if not os.path.exists(state_path):
    raise FileNotFoundError(f"No se encontró el archivo de pesos en: {state_path}")

# Cargar pesos en formato seguro .safetensors
state_dict = load_file(state_path)

missing, unexpected = model.load_state_dict(state_dict, strict=False)

print(f"⚙️  Pesos cargados desde: {state_path}")
print(f"   🔹 Pesos faltantes: {len(missing)}")
print(f"   🔹 Pesos inesperados: {len(unexpected)}")

# Mostrar los primeros 5 pesos faltantes (si existen)
if len(missing) > 0:
    print(f"   ↳ Ejemplo de pesos faltantes: {missing[:5]}")
if len(unexpected) > 0:
    print(f"   ↳ Ejemplo de pesos inesperados: {unexpected[:5]}")

# Confirmar que es efectivamente el modelo multitarea
print(f"✅ Modelo cargado correctamente: {type(model)}")
print("Cabezas activas: Polarity, Type y Town")

# ------------------------------------------------------------
# Congelar pesos base (encoder) para aplicar LoRA después
# ------------------------------------------------------------
for name, param in model.named_parameters():
    param.requires_grad = False

# ============================================================
# === 3. INYECCIÓN DE LoRA (APLICADA AL MODELO COMPLETO) ====
# ============================================================
from peft import LoraConfig, get_peft_model

print("💡 Inyectando LoRA sobre todo el modelo multitarea (compatible con DDP)...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["query", "value"],  # puedes añadir "key" si quieres cubrir QKV
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_CLS",
)

# ⚠️ LoRA al modelo completo, NO al submódulo encoder.
model = get_peft_model(model, lora_config)

# (opcional) ver params entrenables
try:
    model.print_trainable_parameters()
except Exception:
    pass

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"✅ Parámetros entrenables: {trainable_params:,} / {total_params:,} "
      f"({100*trainable_params/total_params:.3f}%)")

# ============================================================
# === 4. MÉTRICAS DE EVALUACIÓN ==============================
# ============================================================

def compute_metrics_mtl(p):
    logits_polarity, logits_type, logits_town = p.predictions
    preds_polarity = np.argmax(logits_polarity, axis=1)
    preds_type = np.argmax(logits_type, axis=1)
    preds_town = np.argmax(logits_town, axis=1)

    labels_polarity, labels_type, labels_town = p.label_ids

    f1_polarity = f1_score(labels_polarity, preds_polarity, average="weighted")
    f1_type = f1_score(labels_type, preds_type, average="weighted")
    f1_town = f1_score(labels_town, preds_town, average="weighted")

    score = (2 * f1_polarity + 1 * f1_type + 3 * f1_town) / 6.0

    return {
        "Score": score,
        "f1_polarity": f1_polarity,
        "f1_type": f1_type,
        "f1_town": f1_town,
    }

# ============================================================
# === 5. CONFIGURACIÓN DE ENTRENAMIENTO ======================
# ============================================================

print("\n⚙️ Configurando entrenamiento...")
output_dir = os.path.join("models", RUN_NAME)
os.makedirs(output_dir, exist_ok=True)

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LR,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="Score",
    greater_is_better=True,
    save_total_limit=1,
    report_to="wandb" if USE_WANDB else "none",
    fp16=torch.cuda.is_available(),
    label_names=["polarity_label", "type_label", "town_label"],
    remove_unused_columns=False,
    ddp_find_unused_parameters=False,
)

trainer = MultiTaskTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics_mtl,
)

# ============================================================
# === 6. ENTRENAMIENTO =======================================
# ============================================================

print(f"\n🚀 Iniciando entrenamiento LoRA MTL: {RUN_NAME}")
trainer.train()
print("✅ Entrenamiento completado")

# ============================================================
# === 7. EVALUACIÓN FINAL Y GUARDADO =========================
# ============================================================

print("\n📊 Evaluando el mejor modelo...")
final_predictions = trainer.predict(eval_dataset)

compute_and_save_mtl_metrics(
    predictions=final_predictions.predictions,
    labels=final_predictions.label_ids,
    run_name=RUN_NAME,
    label_mappings=label_mappings,
    results_dir="results"
)

trainer.save_model(output_dir)
print(f"🧩 Modelo final guardado en {output_dir}")
print(f"\n🏁 Ejecución '{RUN_NAME}' finalizada correctamente.")
