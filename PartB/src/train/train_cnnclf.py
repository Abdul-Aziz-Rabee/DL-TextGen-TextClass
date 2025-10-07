#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_cnnclf.py
Entrenamiento K-Fold para TextCNN (Kim, 2014).
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from src.models.cnn_text import CNNClassifier
from src.data.datasets import load_clean_corpus, create_kfold_loaders
import pandas as pd
import random


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, preds, golds = 0.0, [], []
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
        preds.extend(logits.argmax(1).detach().cpu().numpy())
        golds.extend(y.cpu().numpy())

    acc = accuracy_score(golds, preds)
    f1m = f1_score(golds, preds, average="macro")
    f1w = f1_score(golds, preds, average="weighted")
    return running_loss / len(dataloader.dataset), acc, f1m, f1w


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss, preds, golds = 0.0, [], []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            running_loss += loss.item() * x.size(0)
            preds.extend(logits.argmax(1).cpu().numpy())
            golds.extend(y.cpu().numpy())

    acc = accuracy_score(golds, preds)
    f1m = f1_score(golds, preds, average="macro")
    f1w = f1_score(golds, preds, average="weighted")
    return running_loss / len(dataloader.dataset), acc, f1m, f1w


def train_kfold(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🧠 Dispositivo usado: {device}\n")
    set_seed(args.seed)

    texts, labels = load_clean_corpus(
        base_dir=os.path.join("data", "raw"),
        archivo="MeIA_2025_train.csv",
        level=args.level
    )

    vocab, folds = create_kfold_loaders(
        texts, labels,
        k=args.kfolds,
        batch_size=args.batch_size,
        max_len=args.max_len,
        min_freq=args.min_freq
    )

    results = []

    for fold_idx, train_loader, val_loader in folds:
        print(f"\n===== Fold {fold_idx+1}/{args.kfolds} =====")

        model = CNNClassifier(
            vocab_size=len(vocab),
            embed_dim=args.embed_dim,
            num_filters=args.num_filters,
            kernel_sizes=args.kernel_sizes,
            num_classes=len(set(labels)),
            emb_dropout=args.emb_dropout,
            proj_dropout=args.proj_dropout,
            pad_idx=0
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss()

        best_val_f1 = -np.inf
        best_state = None

        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc, train_f1m, _ = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, val_f1m, val_f1w = evaluate(model, val_loader, criterion, device)

            print(f"[Epoch {epoch:02d}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.4f} | Val F1m: {val_f1m:.4f}")

            if val_f1m > best_val_f1:
                best_val_f1 = val_f1m
                best_state = model.state_dict().copy()

        model.load_state_dict(best_state)
        val_loss, val_acc, val_f1m, val_f1w = evaluate(model, val_loader, criterion, device)
        results.append({
            "fold": fold_idx + 1,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1_macro": val_f1m,
            "val_f1_weighted": val_f1w
        })

    df = pd.DataFrame(results)
    mean_vals = df.mean(numeric_only=True)
    std_vals = df.std(numeric_only=True)

    print("\n📊 Resultados finales (promedio ± desviación):")
    print(df)
    print(f"\nAccuracy: {mean_vals['val_acc']:.4f} ± {std_vals['val_acc']:.4f}")
    print(f"F1 Macro: {mean_vals['val_f1_macro']:.4f} ± {std_vals['val_f1_macro']:.4f}")
    print(f"F1 Weighted: {mean_vals['val_f1_weighted']:.4f} ± {std_vals['val_f1_weighted']:.4f}")

    os.makedirs("results/reports", exist_ok=True)
    out_path = f"results/reports/textcnn_kfold_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\n✅ Guardado en: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento K-Fold de TextCNN")
    parser.add_argument("--level", type=str, default="word", choices=["word", "char"])
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--num_filters", type=int, default=100)
    parser.add_argument("--kernel_sizes", nargs="+", type=int, default=[3,4,5])
    parser.add_argument("--emb_dropout", type=float, default=0.2)
    parser.add_argument("--proj_dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max_len", type=int, default=300)
    parser.add_argument("--min_freq", type=int, default=2)
    parser.add_argument("--kfolds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_kfold(args)
