import os
import torch
from torch.utils.data import Dataset
import numpy as np

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
    """
    Dataset para modelado de lenguaje (input = seq, target = seq shift).
    """
    def __init__(self, tokens, seq_len, token2idx):
        self.seq_len = seq_len
        self.token2idx = token2idx
        self.data = [token2idx[t] for t in tokens]
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
    if prompt is None:
        prompt = '\n'
    if level == 'char':
        tokens = list(prompt)
    else:
        tokens = prompt.split()
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
        return ' '.join(generated)

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
