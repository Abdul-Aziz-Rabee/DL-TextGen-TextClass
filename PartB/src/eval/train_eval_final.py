#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from src.models.rnn_text import RNNClassifier
from src.models.cnn_text import CNNClassifier
from src.data.datasets import load_clean_corpus, build_vocab, TextClassificationDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento final y matriz de confusión (todas las arquitecturas)")
    parser.add_argument("--arch_type", type=str, default="lstm", choices=["rnn", "lstm", "gru", "cnn"])
    parser.add_argument("--level", type=str, default="word", choices=["word", "char"])
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--pool", type=str, default="max", choices=["max", "mean"])
    parser.add_argument("--emb_dropout", type=float, default=0.2)
    parser.add_argument("--rnn_dropout", type=float, default=0.2)
    parser.add_argument("--proj_dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max_len", type=int, default=300)
    parser.add_argument("--min_freq", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_class_weights", action="store_true")

    args = parser.parse_args()


# ==========================================
# CONFIGURACIÓN
# ==========================================
MODEL_TYPE = args.arch_type       # ← el mejor que detectaste
LEVEL = args.level               # "word" o "char"
EMBED_DIM = args.embed_dim
HIDDEN_SIZE = args.hidden_size
NUM_LAYERS = args.num_layers
BIDIRECTIONAL = args.bidirectional
POOL = args.pool
BATCH_SIZE = args.batch_size
LR = args.lr
EPOCHS = args.epochs
MAX_LEN = args.max_len
MIN_FREQ = args.min_freq
SEED = args.seed

torch.manual_seed(SEED)
np.random.seed(SEED)

# ==========================================
# 1. Cargar corpus completo y dividir test
# ==========================================
texts, labels = load_clean_corpus(base_dir="data/raw", archivo="MeIA_2025_train.csv", level=LEVEL)

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, stratify=labels, random_state=SEED
)

# ==========================================
# 2. Construir vocabulario y datasets
# ==========================================
vocab = build_vocab(X_train, min_freq=2)
num_classes = len(set(labels))

from src.data.datasets import TextClassificationDataset
train_ds = TextClassificationDataset(X_train, y_train, vocab, max_len=MAX_LEN)
test_ds = TextClassificationDataset(X_test, y_test, vocab, max_len=MAX_LEN)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# ==========================================
# 3. Modelo
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if MODEL_TYPE == "cnn":
    model = CNNClassifier(
        vocab_size=len(vocab),
        embed_dim=EMBED_DIM,
        num_classes=num_classes,
        num_filters=100,
        kernel_sizes=(3, 4, 5),
        emb_dropout=0.2,
        proj_dropout=0.3,
        pad_idx=0
    ).to(device)
else:
    model = RNNClassifier(
        vocab_size=len(vocab),
        embed_dim=EMBED_DIM,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=num_classes,
        rnn_type=MODEL_TYPE,
        bidirectional=BIDIRECTIONAL,
        pool=POOL,
        emb_dropout=0.2,
        rnn_dropout=0.2,
        proj_dropout=0.3,
        pad_idx=0
    ).to(device)

optimizer = optim.Adam(model.parameters(), lr=LR)
# Manejo de clases desbalanceadas
if args.use_class_weights:
    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
else:
    criterion = nn.CrossEntropyLoss()

# ==========================================
# 4. Entrenar
# ==========================================
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch}/{EPOCHS} - Loss: {running_loss/len(train_loader):.4f}")

# ==========================================
# 5. Evaluar y generar matriz
# ==========================================
model.eval()
all_preds, all_golds = [], []
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_golds.extend(y.numpy())

cm = confusion_matrix(all_golds, all_preds)

plt.figure(figsize=(7,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title(f"Matriz de confusión - {MODEL_TYPE.upper()} ({LEVEL}-level)")
plt.xlabel("Predicción")
plt.ylabel("Etiqueta real")
plt.tight_layout()

os.makedirs("results/reports", exist_ok=True)
plt.savefig(f"results/reports/confusion_{MODEL_TYPE}.png")
plt.close()

print("✅ Matriz de confusión guardada en results/reports/")
# ==========================================
# 6. Reporte de clasificación
report = classification_report(all_golds, all_preds, digits=3)
print("\n", report)

# Guardar el reporte en un archivo de texto
report_path = f"results/reports/classification_report_{MODEL_TYPE}.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report)
print(f"✅ Reporte de clasificación guardado en {report_path}")



