import argparse
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils_textgen import (
    load_and_tokenize,
    TextDataset,
    sample_from_model,
    save_log
)
import torch.nn as nn

# ====================
# Modelo base: RNN, LSTM, GRU
# ====================
class TextGenModel(nn.Module):
    def __init__(self, arch, vocab_size, embedding_dim, hidden_size, level):
        super().__init__()
        self.level = level
        self.vocab_size = vocab_size

        # Embedding solo si word-level
        if level == 'word':
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            rnn_input_size = embedding_dim
        else:
            self.embedding = None
            rnn_input_size = vocab_size  # one-hot para char-level

        rnn_cls = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}[arch]
        self.rnn = rnn_cls(
            input_size=rnn_input_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        if self.level == 'word':
            x = self.embedding(x)
        else:
            # Convierte a one-hot
            x = torch.nn.functional.one_hot(x, num_classes=self.vocab_size).float()
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out)
        return out, hidden

# ====================
# Entrenamiento 
# ====================
def train(model, train_loader, val_loader, device, epochs, lr, vocab_size, save_dir, arch, level):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    train_losses, val_losses = [], []

    best_val_loss = float('inf')
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, f'trainlog_{arch}_{level}.csv')
    with open(log_file, 'w') as f:
        f.write('epoch,train_loss,val_loss\n')

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out, _ = model(x)
            loss = criterion(out.view(-1, vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validación
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out, _ = model(x)
                loss = criterion(out.view(-1, vocab_size), y.view(-1))
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"[Epoch {epoch:02d}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Guardar log
        with open(log_file, 'a') as f:
            f.write(f"{epoch},{avg_train_loss:.4f},{avg_val_loss:.4f}\n")

        # Guardar mejor modelo
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = os.path.join(save_dir, f"{arch}_{level}_best.pt")
            torch.save(model.state_dict(), model_path)
            print(f"    [INFO] Mejor modelo guardado en {model_path}")

    return train_losses, val_losses

# ====================
# Main
# ====================
def main():
    parser = argparse.ArgumentParser()
    # Parámetros esenciales (requeridos)
    parser.add_argument('--arch', choices=['rnn', 'lstm', 'gru'], required=True)
    parser.add_argument('--level', choices=['char', 'word'], required=True)
    parser.add_argument('--mode', choices=['train', 'generate'], required=True)
    parser.add_argument('--model_path', type=str, required='generate' in '--mode')
    parser.add_argument('--prompt', type=str, required='generate' in '--mode')

    # Parámetros opcionales (con valores predeterminados)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seq_len', type=int, default=50)
    parser.add_argument('--data_path', type=str, default='../data/canciones_clean.txt')
    parser.add_argument('--save_dir', type=str, default='../models/')
    parser.add_argument('--results_dir', type=str, default='../results/')
    parser.add_argument('--length', type=int, default=100)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Semilla reproducible
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Usando dispositivo: {device}")

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
        model = TextGenModel(
            arch=args.arch,
            vocab_size=vocab_size,
            embedding_dim=args.embedding_dim,
            hidden_size=args.hidden_size,
            level=args.level
        ).to(device)

        # Entrenamiento
        train(model, train_loader, val_loader, device, args.epochs, args.lr, vocab_size, args.save_dir, args.arch, args.level)

    else:
        # Modo generación
        model = TextGenModel(
            arch=args.arch,
            vocab_size=vocab_size,
            embedding_dim=args.embedding_dim,
            hidden_size=args.hidden_size,
            level=args.level
        ).to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
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
