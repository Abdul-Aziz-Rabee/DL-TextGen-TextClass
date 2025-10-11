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

BASE_MODEL_DIR = "models/BETO_local"    # Encoder base BETO (offline)
MODEL_DIR = "models/BETO_MTL_SO"           # Modelo multitarea original
RUN_NAME = "BETO_MTL_LoRA_MeIA"
MAX_LEN = 256
EPOCHS = 5
BATCH_SIZE = 16
LR = 2e-5
USE_WANDB = False

# ============================================================
# === 1. CARGA DE DATOS Y TOKENIZACIÓN =======================
# ============================================================

from accelerate import Accelerator
acc = Accelerator()

if acc.is_main_process:
    print("📥 Cargando dataset MeIA para multitarea...")
data = load_and_prepare_meia_for_mtl()
train_dataset, eval_dataset = data["train"], data["eval"]
label_mappings = data["label_mappings"]

# === Derivar número de clases correctamente con soporte a formatos nuevos/viejos ===
def _num_classes(mapping):
    # Formato nuevo: {"label2id": {...}, "id2label": {...}}
    if isinstance(mapping, dict) and "id2label" in mapping:
        id2label = mapping["id2label"]
        # claves pueden ser "0","1",... o ints
        return len(id2label)
    # Formato viejo: {0:"Hotel",1:"Restaurant",...} o lista ["Hotel",...]
    if isinstance(mapping, dict):
        return len(mapping)
    if isinstance(mapping, list):
        return len(mapping)
    raise ValueError(f"Formato de mapping no soportado: {type(mapping)}")

num_labels_polarity = _num_classes(label_mappings["polarity"])
num_labels_type     = _num_classes(label_mappings["type"])
num_labels_town     = _num_classes(label_mappings["town"])

print(f"🔢 Clases -> polarity:{num_labels_polarity} | type:{num_labels_type} | town:{num_labels_town}")

# ------------------------------------------------------------
# Tokenizador del modelo base BETO_local (offline)
# ------------------------------------------------------------
print(f"🔤 Cargando tokenizador desde {BASE_MODEL_DIR} ...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR)

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
# === 2. CARGA DEL MODELO MULTITAREA (USANDO BETO_LOCAL) =====
# ============================================================

from transformers import AutoConfig
from safetensors.torch import load_file
import torch

# --- Rutas de los modelos ---

print("\n🧠 Cargando modelo BETO_MTL preentrenado (usando BETO_local como base)...")

# ------------------------------------------------------------
# Cargar configuración multitarea
# ------------------------------------------------------------
config = AutoConfig.from_pretrained(MODEL_DIR)

# Crear instancia explícita del modelo multitarea
# Usa el encoder base BETO_local como punto de partida (idéntico al flujo original)
model = MultiTaskModel(
    config=config,
    model_name=BASE_MODEL_DIR,  # ⚙️ encoder base BETO
    num_labels_polarity=num_labels_polarity,
    num_labels_type=num_labels_type,
    num_labels_town=num_labels_town,
)

# ------------------------------------------------------------
# Cargar pesos multitarea (MTL) desde el archivo .safetensors
# ------------------------------------------------------------
state_path = os.path.join(MODEL_DIR, "model.safetensors")
if not os.path.exists(state_path):
    raise FileNotFoundError(f"No se encontró el archivo de pesos en: {state_path}")

state_dict = load_file(state_path)
missing, unexpected = model.load_state_dict(state_dict, strict=False)

print(f"⚙️  Pesos multitarea cargados desde: {state_path}")
print(f"   🔹 Pesos faltantes: {len(missing)}")
print(f"   🔹 Pesos inesperados: {len(unexpected)}")
if len(missing) > 0:
    print(f"   ↳ Ejemplo de pesos faltantes: {missing[:5]}")
if len(unexpected) > 0:
    print(f"   ↳ Ejemplo de pesos inesperados: {unexpected[:5]}")

# Confirmar tipo y estructura
print(f"✅ Modelo cargado correctamente: {type(model)}")
print("Cabezas activas: Polarity, Type y Town")

# ------------------------------------------------------------
# Congelar el encoder antes de aplicar LoRA
# ------------------------------------------------------------
for name, param in model.named_parameters():
    if name.startswith("transformer"):
        param.requires_grad = False

print("🔒 Encoder base congelado, listo para inyección LoRA.")


# ============================================================
# === 3. INYECCIÓN DE LoRA (APLICADA AL MODELO COMPLETO) ====
# ============================================================
from peft import LoraConfig, get_peft_model

print("💡 Inyectando LoRA SOLO en el encoder (transformer)...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["query", "value"],  # opcional añadir "key"
    lora_dropout=0.05,
    bias="none",
    task_type="FEATURE_EXTRACTION",     # <- neutro, no cambia la semántica
)

# ⚠️ APLICAR AL ENCODER, NO AL MODELO ENTERO
model.transformer = get_peft_model(model.transformer, lora_config)

# (opcional) ver params entrenables del encoder
try:
    model.transformer.print_trainable_parameters()
except Exception:
    pass

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"✅ Parámetros entrenables: {trainable_params:,} / {total_params:,} "
      f"({100*trainable_params/total_params:.3f}%)")

# ============================================================
# === VERIFICACIÓN DE ESTRUCTURA DEL MODELO + LORA ===========
# ============================================================

print("\n🔍 Verificando estructura del modelo tras inyección LoRA...\n")

# Tipo principal del modelo
print(f"🧠 Tipo de modelo principal: {type(model)}")

# Si el modelo está envuelto por PEFT (LoRA)
try:
    from peft import PeftModel
    if isinstance(model, PeftModel):
        print("✅ LoRA aplicada correctamente (modelo es instancia de PeftModel)")
        print(f"   ↳ Base model: {type(model.base_model)}")
    else:
        print("⚠️ El modelo no es una instancia de PeftModel (LoRA no aplicada o solo parcial)")
except Exception as e:
    print(f"❌ Error verificando PEFT: {e}")

# Buscar si las capas query/value del encoder tienen adaptadores LoRA
found_lora = []
for name, module in model.named_modules():
    if hasattr(module, "lora_A") or hasattr(module, "lora_B"):
        found_lora.append(name)

if found_lora:
    print(f"✅ Se detectaron {len(found_lora)} capas con adaptadores LoRA.")
    print("   Ejemplo de capas modificadas:")
    for n in found_lora[:5]:
        print(f"     - {n}")
else:
    print("⚠️ No se detectaron capas con adaptadores LoRA (inyección fallida o parcial)")

print("\n===========================================================\n")



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
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="Score",
    greater_is_better=True,
    save_total_limit=1,
    report_to="wandb" if USE_WANDB else "none",
    fp16=torch.cuda.is_available(),
    remove_unused_columns=False,
    label_names=["polarity_label", "type_label", "town_label"],
    ddp_find_unused_parameters=False,
)
print("🧪 Tipos antes de entrenar:")
print("   - type(model):", type(model))
print("   - type(model.transformer):", type(model.transformer))
print("   - hasattr(model, 'forward'):", hasattr(model, 'forward'))

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
