# src/train_mtl.py

from distutils.command import config
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import wandb
from datasets import Dataset, ClassLabel
import transformers
from transformers import (
    AutoTokenizer,
    AutoModel,
    TrainingArguments,
    Trainer,
    EvalPrediction,
    PreTrainedModel,
    AutoConfig
)
from sklearn.metrics import f1_score

# Import our custom modules
from data.prepare_meia_mtl import load_and_prepare_meia_for_mtl as load_and_prepare_dataset_for_mtl
from eval.eval_utils_mtl import compute_and_save_mtl_metrics

# --- Custom Model for Multi-Task Learning ---
class MultiTaskModel(PreTrainedModel):
    config_class = AutoConfig
    
    def __init__(self, config, model_name, num_labels_polarity, num_labels_type, num_labels_town):
        super().__init__(config)
        self.num_labels_polarity = num_labels_polarity
        self.num_labels_type = num_labels_type
        self.num_labels_town = num_labels_town

        if model_name is None:
            self.transformer = AutoModel.from_config(config)   # <- 100% offline
        else:
            self.transformer = AutoModel.from_pretrained(model_name, config=config)

        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.classifier_polarity = nn.Linear(config.hidden_size, self.num_labels_polarity)
        self.classifier_type = nn.Linear(config.hidden_size, self.num_labels_type)
        self.classifier_town = nn.Linear(config.hidden_size, self.num_labels_town)

def forward(
    self,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    polarity_label=None,
    type_label=None,
    town_label=None,
    return_dict=None,
    **kwargs,  # <- esto captura cualquier argumento extra (como 'labels')
):
    """
    Forward multitarea compatible con LoRA y Hugging Face Trainer.

    - Recibe los tres conjuntos de etiquetas (polarity, type, town)
    - Calcula las tres pérdidas y la loss combinada ponderada 2:1:3
    - Evita pasar argumentos no válidos ('labels', etc.) al encoder
    """

    # Asegurar que siempre regrese un diccionario
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # --- 🔹 Filtrar argumentos válidos para el encoder ---
    encoder_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "return_dict": return_dict,
    }

    # Solo mantener los que no sean None
    encoder_kwargs = {k: v for k, v in encoder_kwargs.items() if v is not None}

    # === Paso por el encoder ===
    # Evita pasar accidentalmente 'labels' al encoder (causa del error original)
    outputs = self.transformer(**encoder_kwargs)

    # === Representación de la secuencia ===
    # Hugging Face BERT devuelve (last_hidden_state, pooled_output, ...)
    if isinstance(outputs, dict):
        pooled_output = outputs.get("pooler_output", None)
        if pooled_output is None:
            # Algunos modelos (como RoBERTa) no usan pooler_output
            pooled_output = outputs["last_hidden_state"][:, 0]
    else:
        pooled_output = outputs[1]  # Compatibilidad con tuple outputs

    pooled_output = self.dropout(pooled_output)

    # === Cálculo de logits para cada tarea ===
    logits_polarity = self.classifier_polarity(pooled_output)
    logits_type = self.classifier_type(pooled_output)
    logits_town = self.classifier_town(pooled_output)

    # === Cálculo de pérdida multitarea ===
    loss = None
    if polarity_label is not None and type_label is not None and town_label is not None:
        loss_fct = torch.nn.CrossEntropyLoss()
        loss_polarity = loss_fct(
            logits_polarity.view(-1, self.num_labels_polarity),
            polarity_label.view(-1)
        )
        loss_type = loss_fct(
            logits_type.view(-1, self.num_labels_type),
            type_label.view(-1)
        )
        loss_town = loss_fct(
            logits_town.view(-1, self.num_labels_town),
            town_label.view(-1)
        )

        # Ponderación 2:1:3 (como en el MTL original)
        loss = (2 * loss_polarity + 1 * loss_type + 3 * loss_town) / 6.0

    # === Formato de salida compatible con Trainer ===
    if not return_dict:
        output = (logits_polarity, logits_type, logits_town)
        return (loss, output) if loss is not None else (None, output)

    return {
        "loss": loss,
        "logits": (logits_polarity, logits_type, logits_town),
    }


# --- Custom Trainer for Multi-Task Learning ---
# ¡NUEVO! Hacemos el Trainer explícito para que siempre funcione.
class MultiTaskTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss
        
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            outputs = model(**inputs)
            # Para la predicción, la pérdida será None, lo cual es correcto.
            loss = outputs.get("loss")
            # Los logits son la tupla que empaquetamos en el forward.
            logits = outputs.get("logits")
            # Las etiquetas se pasan por separado.
            labels = tuple(inputs.get(name) for name in self.label_names)
        return (loss, logits, labels)


# --- El resto del archivo no necesita cambios ---
def main(args):
    if args.use_wandb:
        wandb.init(project="LLMs-sentiment-analysis-mx", name=args.run_name, config=args)

    print("Loading and preparing data for MTL...")
    data = load_and_prepare_dataset_for_mtl()
    train_dataset = data['train']
    eval_dataset = data['eval']
    label_mappings = data['label_mappings']
    
    print(f"Loading tokenizer for model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=args.max_length)

    print(f"Tokenizing datasets with max_length: {args.max_length}...")
    train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    
    print("Casting label columns to correct data types...")
    for label_name, mapping in label_mappings.items():
        num_classes = len(mapping)
        class_label_feature = ClassLabel(num_classes=num_classes)
        train_dataset = train_dataset.cast_column(f"{label_name}_label", class_label_feature)
        eval_dataset = eval_dataset.cast_column(f"{label_name}_label", class_label_feature)
    
    print(f"Loading base model config: {args.model_name}")
    config = AutoConfig.from_pretrained(args.model_name)
    
    print("Instantiating Multi-Task Model...")
    model = MultiTaskModel(
        config=config,
        model_name=args.model_name,
        num_labels_polarity=len(label_mappings['polarity']),
        num_labels_type=len(label_mappings['type']),
        num_labels_town=len(label_mappings['town'])
    )

    def compute_metrics_mtl(p: EvalPrediction):
        # p.predictions ahora contendrá la tupla de logits que empaquetamos.
        logits_polarity, logits_type, logits_town = p.predictions
        preds_polarity = np.argmax(logits_polarity, axis=1)
        preds_type = np.argmax(logits_type, axis=1)
        preds_town = np.argmax(logits_town, axis=1)
        
        labels_polarity, labels_type, labels_town = p.label_ids

        f1_polarity = f1_score(labels_polarity, preds_polarity, average="weighted")
        f1_type = f1_score(labels_type, preds_type, average="weighted")
        f1_town = f1_score(labels_town, preds_town, average="weighted")
        
        final_score = (2 * f1_polarity + 1 * f1_type + 3 * f1_town) / 6.0

        return {
            "Score": final_score,
            "polarity_f1": f1_polarity,
            "type_f1": f1_type,
            "town_f1": f1_town,
        }

    training_args = TrainingArguments(
        output_dir=os.path.join("models", args.run_name),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="Score",
        greater_is_better=True,
        save_total_limit=1,
        report_to="wandb" if args.use_wandb else "none",
        fp16=torch.cuda.is_available(),
        label_names=["polarity_label", "type_label", "town_label"],
        # Aseguramos que el modelo siempre devuelva un diccionario para consistencia
        remove_unused_columns=False, 
    )

    trainer = MultiTaskTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_mtl,
    )

    print(f"\n--- Starting MTL training for run: {args.run_name} ---")
    trainer.train()
    print("--- Training finished ---")

    print("\nEvaluating the best MTL model on the evaluation set...")
    final_predictions = trainer.predict(eval_dataset)
    
    compute_and_save_mtl_metrics(
        predictions=final_predictions.predictions,
        labels=final_predictions.label_ids,
        run_name=args.run_name,
        label_mappings=label_mappings
    )

    output_dir = os.path.join("models", args.run_name)
    trainer.save_model(output_dir)
    print(f"Final MTL model saved to {output_dir}")
    
    print(f"\n✅ Run '{args.run_name}' completed. Best metrics: {final_predictions.metrics}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Multi-Task Transformer model.")
    
    parser.add_argument("--model_name", type=str, required=True, help="Path to the local base model or name from Hugging Face Hub.")
    parser.add_argument("--run_name", type=str, required=True, help="A unique name for this training run.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size PER DEVICE.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for the AdamW optimizer.")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum sequence length for tokenization.")
    parser.add_argument("--use_wandb", action="store_true", help="Set this flag to enable logging with Weights & Biases.")
    
    args = parser.parse_args()
    main(args)

