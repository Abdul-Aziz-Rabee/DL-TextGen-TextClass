import argparse
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

# ====================
# Utils y Dataset
# ====================
from utils_textgen import (
    load_and_tokenize,
    build_vocab,
    TextDataset,
    sample_from_model,
    save_log
)

# ====================
# LightningModule
# ====================
import torch.nn as nn

class TextGenLightningModule(pl.LightningModule):
    def __init__(self, arch, vocab_size, embedding_dim, hidden_size, lr, level, idx2token, token2idx):
        super().__init__()
        self.save_hyperparameters()
        self.level = level
        self.idx2token = idx2token
        self.token2idx = token2idx
        self.vocab_size = vocab_size
        
        # Embedding solo si word-level
        if level == 'word':
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        else:
            self.embedding = None

        # Selección de arquitectura
        rnn_cls = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}[arch]
        self.rnn = rnn_cls(
            input_size=embedding_dim if level == 'word' else vocab_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, x, hidden=None):
        # x: (batch, seq_len)
        if self.level == 'word':
            x = self.embedding(x)  # (batch, seq_len, emb)
        else:
            x = torch.nn.functional.one_hot(x, num_classes=self.vocab_size).float()
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out)
        return out, hidden

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits, _ = self(x)
        loss = self.loss_fn(logits.view(-1, self.vocab_size), y.view(-1))
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits, _ = self(x)
        loss = self.loss_fn(logits.view(-1, self.vocab_size), y.view(-1))
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

# ====================
# Main: Argumentos y Entrenamiento
# ====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', choices=['rnn', 'lstm', 'gru'], required=True)
    parser.add_argument('--level', choices=['char', 'word'], required=True)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seq_len', type=int, default=50)
    parser.add_argument('--data_path', type=str, default='../data/canciones_clean.txt')
    parser.add_argument('--save_dir', type=str, default='../models/')
    parser.add_argument('--results_dir', type=str, default='../results/')
    parser.add_argument('--mode', choices=['train', 'generate'], default='train')
    parser.add_argument('--prompt', type=str, default=None)
    parser.add_argument('--length', type=int, default=100)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model_path', type=str, default=None)
    args = parser.parse_args()

    # Semilla reproducible
    pl.seed_everything(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    # Carga y tokenización de datos
    data, idx2token, token2idx = load_and_tokenize(
        args.data_path, level=args.level
    )
    vocab_size = len(idx2token)

    if args.mode == 'train':
        # Split train/val
        split_idx = int(0.9 * len(data))
        train_data = data[:split_idx]
        val_data = data[split_idx:]

        train_ds = TextDataset(train_data, args.seq_len, token2idx)
        val_ds = TextDataset(val_data, args.seq_len, token2idx)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

        # Modelo
        model = TextGenLightningModule(
            arch=args.arch,
            vocab_size=vocab_size,
            embedding_dim=args.embedding_dim,
            hidden_size=args.hidden_size,
            lr=args.lr,
            level=args.level,
            idx2token=idx2token,
            token2idx=token2idx
        )
        trainer = pl.Trainer(
            max_epochs=args.epochs,
            default_root_dir=args.save_dir,
            log_every_n_steps=10
        )
        trainer.fit(model, train_loader, val_loader)

        # Guardar modelo
        model_path = os.path.join(args.save_dir, f'{args.arch}_{args.level}_final.ckpt')
        trainer.save_checkpoint(model_path)
        print(f'[INFO] Modelo guardado en {model_path}')
    else:
        # Modo generación
        assert args.model_path is not None, "Debes especificar --model_path"
        model = TextGenLightningModule.load_from_checkpoint(
            args.model_path,
            arch=args.arch,
            vocab_size=vocab_size,
            embedding_dim=args.embedding_dim,
            hidden_size=args.hidden_size,
            lr=args.lr,
            level=args.level,
            idx2token=idx2token,
            token2idx=token2idx
        )
        model.eval()
        generated = sample_from_model(
            model, args.prompt, args.length, args.temperature, token2idx, idx2token, args.level
        )
        # Guardar resultado
        log_path = os.path.join(args.results_dir, f'sample_{args.arch}_{args.level}.txt')
        save_log(generated, log_path)
        print(f'[INFO] Letra generada guardada en {log_path}')

if __name__ == '__main__':
    main()
