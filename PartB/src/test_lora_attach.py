"""
Prueba rápida para verificar que el encoder BETO_MTL
acepta correctamente adaptadores LoRA.
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import logging
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, PeftModel

# --- Ruta al modelo MTL ---
model_dir = "models/BETO_MTL"

print(f"🔹 Cargando backbone desde: {model_dir}")
tok = AutoTokenizer.from_pretrained(model_dir)
#base_model = AutoModel.from_pretrained(model_dir)
base_model = AutoModelForSequenceClassification.from_pretrained(model_dir)
# --- Configuración LoRA ---
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_CLS",  # tipo de tarea; basta con esto para inicializar
)

# --- Inyectar adaptadores LoRA ---
print("⚙️  Inyectando adaptadores LoRA...")
model_lora = get_peft_model(base_model, lora_config)

# --- Congelar base y verificar ---
total_params = sum(p.numel() for p in model_lora.parameters())
trainable_params = sum(p.numel() for p in model_lora.parameters() if p.requires_grad)
ratio = 100 * trainable_params / total_params

print(f"\n✅ Modelo con LoRA creado correctamente.")
print(f"Total parámetros: {total_params:,}")
print(f"Parámetros entrenables: {trainable_params:,}  ({ratio:.2f}% del total)")

# --- Probar un forward rápido ---
text = "Me encantó la película, la actuación fue increíble."
inputs = tok(text, return_tensors="pt")

with torch.no_grad():
    outputs = model_lora(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"]
    )
    logits = outputs.logits

print(f"\nLogits mean: {logits.mean().item():.6f}")
print(f"Logits std:  {logits.std().item():.6f}")

# --- Confirmar estructura ---
print("\nCapas LoRA detectadas:")
for name, module in model_lora.named_modules():
    if "lora" in name.lower():
        print("  •", name)

print("\n✅ Prueba finalizada. Modelo listo para fine-tuning ligero con LoRA.")
