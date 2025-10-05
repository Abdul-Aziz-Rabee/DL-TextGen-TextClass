#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
datasets.py
Manejo de corpus y splits K-Fold para la Parte B (clasificación de texto)
"""

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from collections import Counter
import numpy as np
from src.preprocessing.limpieza import leer_corpus, procesar_corpus
from nltk.tokenize import word_tokenize

# ==========================================================
# 1. Dataset base
# ==========================================================

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=256):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.texts[idx]
        ids = [self.vocab.get(tok, self.vocab["<unk>"]) for tok in tokens]
        if len(ids) < self.max_len:
            ids += [self.vocab["<pad>"]] * (self.max_len - len(ids))
        else:
            ids = ids[:self.max_len]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


# ==========================================================
# 2. Construcción de vocabulario
# ==========================================================

def build_vocab(tokenized_texts, min_freq=2):
    counter = Counter([tok for text in tokenized_texts for tok in text])
    vocab = {"<pad>": 0, "<unk>": 1}
    for tok, freq in counter.items():
        if freq >= min_freq:
            vocab[tok] = len(vocab)
    return vocab


# ==========================================================
# 3. Carga y preparación del corpus
# ==========================================================

def load_clean_corpus(base_dir=".", archivo="MeIA_2025_train.csv", level="word"):
    df = leer_corpus(base_dir, archivo, metodo="ftfy")
    df = procesar_corpus(
        df,
        eliminar_stop=False,
        normalizar_texto=True,
        lematizar_stem=False,
        custom_words={'hotel', 'restaurante', 'lugar', 'playa', 'comida', 'servicio'}
    )
    if level == "word":
        texts = [word_tokenize(t) for t in df["Review"].tolist()]
    elif level == "char":
        texts = [list(t) for t in df["Review"].tolist()]
    else:
        raise ValueError("Nivel de tokenización no soportado. Use 'word' o 'char'.")

    labels = df["Polarity"].astype(int).values
    return texts, labels


# ==========================================================
# 4. Generador de folds
# ==========================================================

def create_kfold_loaders(texts, labels, k=5, batch_size=64, max_len=256, min_freq=2):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    vocab = build_vocab(texts, min_freq=min_freq)

    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
        train_texts = [texts[i] for i in train_idx]
        val_texts   = [texts[i] for i in val_idx]
        train_labels = labels[train_idx]
        val_labels   = labels[val_idx]

        train_ds = TextClassificationDataset(train_texts, train_labels, vocab, max_len)
        val_ds   = TextClassificationDataset(val_texts, val_labels, vocab, max_len)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        folds.append((fold_idx, train_loader, val_loader))
    return vocab, folds
