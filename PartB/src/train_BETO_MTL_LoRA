#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fine-tuning ligero con LoRA sobre BETO-MTL (solo tarea Polarity).
Listo para ejecutarse en cluster o local.
"""

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model
from sklearn.metrics import f1_score, accuracy_score


# ============================================================
# CONFIGURACIÓN PRINCIPAL
# ============================================================
BASE_MODEL_DIR = "models/BETO_MTL"
DATA_PATH = "data/processed/classif/polarity_small.csv"   # <-- ajusta esta ruta
OUTPUT_DIR = "models/beto_mtl_lora_polarity"
NUM_LABELS = 3                                            # negativo / neutro / positivo
MAX_LEN = 256
EPOCHS = 5
BATCH_SIZE = 32
LR = 2e-5

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
# PREPARAR TOKENIZADOR Y DATOS
# ============================================================
print(f"🔹 Cargando tokenizer desde {BASE_MODEL_DIR}")
tok = AutoTokenizer.from_pretrained(BASE_MODEL_DIR)

# ⚠️ Sustituye esto por tu carga real (HuggingFace dataset o CSV)
print(f"📄 Cargando dataset desde {DATA_PATH}")
dataset = load_dataset("csv", data_files={"train": DATA_PATH, "validation": DATA_PATH})  # temporal

def tokenize(batch):
    return tok(batch["text"], truncation=True, padding="max_length", max_length=MAX_LEN)

dataset = dataset.map(tokenize, batched=True)
dataset = dataset.rename_column("label", "labels")
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

train_dataset = dataset["train"]
eval_dataset = dataset["validation"]

# ============================================================
# CARGAR MODELO BASE + INYECTAR LoRA
# ============================================================
print(f"⚙️ Cargando modelo base desde {BASE_MODEL_DIR}")
base_model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL_DIR, num_labels=NUM_LABELS)

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
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_weighted",
    greater_is_better=True,
    save_total_limit=1,
    fp16=torch.cuda.is_available(),
    logging_dir=os.path.join(OUTPUT_DIR, "logs"),
    logging_steps=50,
    report_to="none",  # cambia a "wandb" si quieres registrar
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
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
