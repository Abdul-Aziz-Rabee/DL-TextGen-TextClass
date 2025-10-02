# batch_generator.py
import os
import json
from utils_textgen import sample_from_model
import torch
from train_textgen import TextGenModel

import random
import numpy as np
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)



# Definir las configuraciones de los modelos y prompts
models = [
    {'arch': 'gru', 'level': 'char', 'path': '../models/gru_char/gru_char_best.pt'},
    {'arch': 'gru', 'level': 'word', 'path': '../models/gru_word/gru_word_best.pt'},
    {'arch': 'lstm', 'level': 'char', 'path': '../models/lstm_char/lstm_char_best.pt'},
    {'arch': 'lstm', 'level': 'word', 'path': '../models/lstm_word/lstm_word_best.pt'},
    {'arch': 'rnn', 'level': 'char', 'path': '../models/rnn_char/rnn_char_best.pt'},
    {'arch': 'rnn', 'level': 'word', 'path': '../models/rnn_word/rnn_word_best.pt'},
]

prompts = [
    "",  # Prompt vacío
    "En la penumbra del día",  
    "Bailando bajo la lluvia",  
]
temperatures = [0.8, 1.0, 1.2]
length_char = 400
length_word = 80
results_dir = "../results/batch/"
os.makedirs(results_dir, exist_ok=True)

for m in models:
    # Carga modelo y checkpoint
    checkpoint = torch.load(m['path'], map_location='cuda')
    model = TextGenModel(
        arch=checkpoint['arch'],
        vocab_size=checkpoint['vocab_size'],
        embedding_dim=checkpoint['embedding_dim'],
        hidden_size=checkpoint['hidden_size'],
        level=checkpoint['level'],
        num_layers=checkpoint['num_layers'],
        dropout=checkpoint['dropout'],
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    idx2token = checkpoint['idx2token']
    token2idx = checkpoint['token2idx']
    
    curr_length = length_char if m['level'] == 'char' else length_word

    for pid, prompt in enumerate(prompts):
        for temp in temperatures:
            # Genera letra
            letra = sample_from_model(
                model, prompt, curr_length, temp, token2idx, idx2token, m['level']
            )
            # Define nombre de archivo
            fname = f"{m['arch']}_{m['level']}_prompt{pid}_temp{temp}.txt"
            fmeta = f"{m['arch']}_{m['level']}_prompt{pid}_temp{temp}.json"
            # Guarda letra
            with open(os.path.join(results_dir, fname), "w", encoding='utf-8') as f:
                f.write(letra)
            # Guarda metadatos
            meta = {
                "arch": m['arch'],
                "level": m['level'],
                "model_path": m['path'],
                "prompt_id": pid,
                "prompt": prompt,
                "temperature": temp,
                "length": curr_length,
            }
            with open(os.path.join(results_dir, fmeta), "w", encoding='utf-8') as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
            print(f"[INFO] Generado: {fname}")

print("Batch generation complete! 🚀")
