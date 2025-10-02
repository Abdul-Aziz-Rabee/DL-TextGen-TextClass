import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

# Ruta base del proyecto
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 1. Parámetros principales
MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
MODEL_CACHE_PATH = os.path.join(base_path, 'models/models--meta-llama--Meta-Llama-3-8B')  # Ruta absoluta al modelo base
DATA_PATH = os.path.join(base_path, "data/train_lyrics.jsonl")
OUTPUT_DIR = os.path.join(base_path, "models/llama3_lora")
RUN_NAME = "llama3_lora_lyrics"

EPOCHS = 3
BATCH_SIZE = 2
LR = 2e-4
BLOCK_SIZE = 256

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. Cargar y procesar datos
dataset = load_dataset("json", data_files=DATA_PATH)["train"]
dataset = dataset.train_test_split(test_size=0.2, seed=42)
tokenizer = AutoTokenizer.from_pretrained(MODEL_CACHE_PATH, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_fn(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=BLOCK_SIZE,
    )

tokenized_datasets = dataset.map(tokenize_fn, batched=True)

# 3. Cargar modelo y preparar LoRA
model = AutoModelForCausalLM.from_pretrained(
    MODEL_CACHE_PATH,
    torch_dtype=torch.float16,
    device_map="cuda",
    load_in_4bit=True,
)

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 4. Argumentos de entrenamiento
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LR,
    logging_dir=f"{OUTPUT_DIR}/logs",
    evaluation_strategy="steps",
    eval_steps=50,
    save_steps=100,
    logging_steps=10,
    save_total_limit=2,
    fp16=True,
    report_to="none",
    run_name=RUN_NAME,
    dataloader_num_workers=4,
)

# 5. ¡Entrena!
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
)

trainer.train()
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Fine-tuning terminado. Modelo guardado en:", OUTPUT_DIR)
