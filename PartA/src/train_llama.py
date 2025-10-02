import os
import torch
import argparse
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='../models/Meta-Llama-3-8B/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920/')
parser.add_argument('--data_path', type=str, default='../data/train_lyrics.jsonl')
parser.add_argument('--run_name', type=str, default='llama3_ft_v1')
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--block_size', type=int, default=256)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--dropout', type=float, default=0.05)
parser.add_argument('--save_dir', type=str, default='../models/llama3_ft_v1')
parser.add_argument('--output_dir', type=str, default='../results/llama3_ft_v1')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# 1. Parámetros principales

# Esto siempre te da el path al directorio donde está tu script .py
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

# Así generas rutas absolutas y evitas confusiones
MODEL_CACHE_PATH = os.path.join(BASE_DIR, "models/Meta-Llama-3-8B/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920/")
DATA_PATH = os.path.join(BASE_DIR, "data/train_lyrics.jsonl")
OUTPUT_DIR = os.path.join(BASE_DIR, "results", args.run_name)

RUN_NAME = args.run_name
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
BLOCK_SIZE = args.block_size

# 2. Cargar y procesar datos
dataset = load_dataset("json", data_files=DATA_PATH)["train"]
dataset = dataset.train_test_split(test_size=0.2, seed=42)
tokenizer = AutoTokenizer.from_pretrained(MODEL_CACHE_PATH, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_fn(batch):
    outputs = tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=BLOCK_SIZE,
    )
    outputs["labels"] = outputs["input_ids"].copy()
    return outputs

tokenized_datasets = dataset.map(tokenize_fn, batched=True)

# 3. Cargar modelo y preparar LoRA
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_CACHE_PATH,
    dtype=torch.float16,
    device_map="auto",
    quantization_config=bnb_config,
)

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=args.dropout,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 4. Argumentos de entrenamiento
'''
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=2e-4,
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
'''
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    save_steps=100,
    save_total_limit=2,
    fp16=True,
    learning_rate=args.lr,
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

# 6. Guarda el modelo y el tokenizador
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

eval_result = trainer.evaluate(eval_dataset=tokenized_datasets["test"])
print("Validation loss:", eval_result["eval_loss"])
print("Perplexity:", np.exp(eval_result["eval_loss"]))

print("Fine-tuning terminado. Modelo guardado en:", OUTPUT_DIR)
