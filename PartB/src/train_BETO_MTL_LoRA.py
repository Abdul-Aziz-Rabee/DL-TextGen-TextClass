#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fine-tuning ligero con LoRA sobre BETO-MTL (solo tarea Polarity).
Integrado con el módulo de limpieza y preparación de datos MeIA.
Listo para ejecutarse en cluster o local.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from transformers.utils import logging
logging.set_verbosity_info()
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

from peft import LoraConfig, get_peft_model
from sklearn.metrics import f1_score, accuracy_score

# 🔹 Importamos el módulo de preparación del dataset
from src.data.prepare_meia_transformer import load_meia_for_transformer


# ============================================================
# CONFIGURACIÓN PRINCIPAL
# ============================================================
BASE_MODEL_DIR = "models/BETO_MTL_encoder"  # ← encoder BETO-MTL exportado
DATA_PATH = "data/raw/MeIA_2025_train.csv"  # ← tu dataset real
OUTPUT_DIR = "models/beto_mtl_lora_polarity"
NUM_LABELS = 5                              # 5 niveles de polaridad (1–5 → 0–4)
MAX_LEN = 512
EPOCHS = 10
BATCH_SIZE = 32
LR = 1e-5
SEED = 42

torch.manual_seed(SEED)

# ============================================================
# FUNCIONES AUXILIARES
# ============================================================
def compute_metrics(pred):
    preds = pred.predictions.argmax(-1)
    labels = pred.label_ids
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1_weighted": f1}


# ============================================================
# CARGAR Y LIMPIAR DATASET (con flujo unificado)
# ============================================================
print(f"📂 Cargando y preparando dataset desde {DATA_PATH} ...")
dataset = load_meia_for_transformer(DATA_PATH)
dataset = dataset.train_test_split(test_size=0.2, seed=SEED)
train_ds, val_ds = dataset["train"], dataset["test"]

print(f"✅ Dataset cargado. Train: {len(train_ds)} | Val: {len(val_ds)}")

# ============================================================
# TOKENIZACIÓN
# ============================================================
print(f"🔹 Cargando tokenizer desde {BASE_MODEL_DIR}")
tok = AutoTokenizer.from_pretrained(BASE_MODEL_DIR)

def tokenize(batch):
    return tok(batch["text"], truncation=True, padding="max_length", max_length=MAX_LEN)

train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)

train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# ============================================================
# CARGAR MODELO BASE + INYECTAR LoRA
# ============================================================
print(f"⚙️ Cargando modelo base desde {BASE_MODEL_DIR}")
base_model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL_DIR,
    num_labels=NUM_LABELS
)

print("⚙️ Inyectando adaptadores LoRA...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_CLS",
)
model = get_peft_model(base_model, lora_config)

# Verificación rápida
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"✅ Modelo con LoRA listo. Entrenables: {trainable_params:,} ({100*trainable_params/total_params:.2f}% del total)")


# ============================================================
# CONFIGURAR TRAINER
# ============================================================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LR,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,             # ✅ requerido por EarlyStoppingCallback
    metric_for_best_model="f1_weighted",     # ✅ métrica de monitoreo
    greater_is_better=True,
    save_total_limit=1,
    fp16=torch.cuda.is_available(),
    logging_dir=os.path.join(OUTPUT_DIR, "logs"),
    logging_steps=50,
    report_to="none",
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
    tokenizer=tok,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

# ============================================================
# ENTRENAR
# ============================================================
print("\n🚀 Iniciando fine-tuning ligero con LoRA (polarity)...")
trainer.train()
print("✅ Entrenamiento finalizado.")

# ============================================================
# GUARDAR MODELO AJUSTADO
# ============================================================
trainer.save_model(OUTPUT_DIR)
print(f"\n✅ Modelo con LoRA ajustado guardado en: {OUTPUT_DIR}")
