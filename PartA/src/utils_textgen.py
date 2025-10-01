import os
import torch
from torch.utils.data import Dataset
import numpy as np
import random

def load_and_split_songs2(path, level='char', val_frac=0.2, seed=42):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Split por delimitador, pero NO elimines los delimitadores
    canciones = [c.strip() for c in text.split("<|startsong|>") if c.strip()]
    # NO elimines <|endsong|>
    # canciones = [c.replace("<|endsong|>", "") for c in canciones]  # <-- Elimina esta línea

    print(f"[INFO] Total canciones cargadas: {len(canciones)}")
    random.seed(seed)
    random.shuffle(canciones)

    n_total = len(canciones)
    n_val = int(val_frac * n_total)
    val_canciones = canciones[:n_val]
    train_canciones = canciones[n_val:]

    def tokenize_list(cancion_list, level):
        joined = "\n".join(cancion_list)
        joined = joined.replace("<|startsong|>", " <SS> ")
        joined = joined.replace("<|endsong|>", " <ES> ")
        if level == 'char':
            tokens = list(joined)
        elif level == 'word':
            # Reemplaza saltos de línea reales por un token especial antes de split
            joined = joined.replace('\n', ' <NL> ')
            tokens = joined.split()
        else:
            raise ValueError('Nivel no soportado')
        vocab = sorted(set(tokens))
        idx2token = {i: t for i, t in enumerate(vocab)}
        token2idx = {t: i for i, t in idx2token.items()}
        return tokens, idx2token, token2idx

    train_tokens, train_idx2token, train_token2idx = tokenize_list(train_canciones, level)
    val_tokens, _, _ = tokenize_list(val_canciones, level)

    return train_tokens, val_tokens, train_idx2token, train_token2idx


def load_and_split_songs(path, level='char', val_frac=0.2, seed=42):
    # 1. Leer todas las canciones como lista
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Split por delimitador
    # Quita cualquier espacio vacío por si acaso
    canciones = [c.strip() for c in text.split("<|startsong|>") if c.strip()]
    # Remueve posibles delimitadores de cierre si existen
    canciones = [c.replace("<|endsong|>", "") for c in canciones]
    print(f"[INFO] Total canciones cargadas: {len(canciones)}")
    # 2. Shuffle a nivel canción
    random.seed(seed)
    random.shuffle(canciones)

    # 3. Split en train/val (por canción)
    n_total = len(canciones)
    n_val = int(val_frac * n_total)
    val_canciones = canciones[:n_val]
    train_canciones = canciones[n_val:]

    # 4. Tokeniza cada subset como corpus grande
    def tokenize_list(cancion_list, level):
        joined = "\n".join(cancion_list)
        if level == 'char':
            tokens = list(joined)
        elif level == 'word':
            tokens = joined.replace('\n', ' \n ').split()
        else:
            raise ValueError('Nivel no soportado')
        vocab = sorted(set(tokens))
        idx2token = {i: t for i, t in enumerate(vocab)}
        token2idx = {t: i for i, t in idx2token.items()}
        return tokens, idx2token, token2idx

    train_tokens, train_idx2token, train_token2idx = tokenize_list(train_canciones, level)
    val_tokens, _, _ = tokenize_list(val_canciones, level)  # Usa mismo vocab que el train

    return train_tokens, val_tokens, train_idx2token, train_token2idx

# ==========================
# Tokenización y vocabulario
# ==========================
def load_and_tokenize(path, level='char'):
    """
    Carga un archivo de texto y tokeniza a nivel carácter o palabra.
    Devuelve lista de tokens, idx2token y token2idx.
    """
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    # Eliminar tokens especiales
    text = text.replace("<|endsong|>", "").replace("<|startsong|>", "")

    if level == 'char':
        tokens = list(text)
    elif level == 'word':
        tokens = text.replace('\n', ' \n ').split()
    else:
        raise ValueError('Nivel no soportado')
    vocab = sorted(set(tokens))
    idx2token = {i: t for i, t in enumerate(vocab)}
    token2idx = {t: i for i, t in idx2token.items()}
    return tokens, idx2token, token2idx

# ==========================
# Dataset para DataLoader
# ==========================
class TextDataset(Dataset):
    def __init__(self, tokens, seq_len, token2idx):
        self.seq_len = seq_len
        self.token2idx = token2idx
        # Si el token no está en el vocabulario, asigna 0 (o el idx de <unk> si lo tienes)
        self.data = [token2idx.get(t, 0) for t in tokens]
        self.length = len(self.data) - seq_len

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx+self.seq_len], dtype=torch.long)
        y = torch.tensor(self.data[idx+1:idx+self.seq_len+1], dtype=torch.long)
        return x, y


# ==========================
# Sampling/generación de texto
# ==========================
def sample_from_model(model, prompt, length, temperature, token2idx, idx2token, level):
    """
    Genera una secuencia de texto a partir de un prompt y un modelo entrenado.
    """
    device = next(model.parameters()).device
    model.eval()
    if not prompt or (level == 'word' and len(prompt.strip()) == 0):
        prompt = '\n' if level == 'char' else '<SS>'
    if level == 'char':
        tokens = list(prompt)
    else:
        tokens = prompt.split()
    if len(tokens) == 0:
        tokens = ['<SS>'] if level == 'word' else ['\n']
    generated = tokens.copy()
    input_seq = [token2idx.get(t, 0) for t in tokens]
    input_tensor = torch.tensor([input_seq], dtype=torch.long).to(device)

    hidden = None
    seq_len = model.seq_len if hasattr(model, 'seq_len') else 50  # fallback por si acaso
    with torch.no_grad():
        for _ in range(length):
            out, hidden = model(input_tensor, hidden)
            logits = out[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1).cpu().numpy().flatten()
            next_idx = np.random.choice(len(probs), p=probs)
            next_token = idx2token[next_idx]
            generated.append(next_token)
            # actualizar input_tensor (deslizar)
            input_seq = input_seq[1:] + [next_idx] if len(input_seq) >= seq_len else input_seq + [next_idx]
            input_tensor = torch.tensor([input_seq], dtype=torch.long).to(device)
    # Decodifica
    if level == 'char':
        return ''.join(generated)
    else:
        # Une palabras y reemplaza el token \n por salto de línea real
        text = ' '.join(generated)
        text = text.replace(' <NL> ', '\n').replace('<NL>', '\n')
        text = text.replace(' <SS> ', '\n').replace('<SS>', '\n')
        text = text.replace(' <ES> ', '\n').replace('<ES>', '\n')
        return text

# ==========================
# Guardar logs y muestras
# ==========================
def save_log(text, path):
    """
    Guarda texto o lista de líneas en un archivo.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        if isinstance(text, list):
            for line in text:
                f.write(str(line) + '\n')
        else:
            f.write(str(text))
