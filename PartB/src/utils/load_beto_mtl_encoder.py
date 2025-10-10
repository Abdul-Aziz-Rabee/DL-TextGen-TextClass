#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn
from transformers import AutoModel
from safetensors.torch import load_file


class MultiTaskModel(nn.Module):
    def __init__(self, model_name: str, hidden_size: int = 768,
                 num_labels_polarity: int = 5,
                 num_labels_type: int = 3,
                 num_labels_town: int = 40):
        super(MultiTaskModel, self).__init__()
        # Carga local sin internet
        self.transformer = AutoModel.from_pretrained(model_name, local_files_only=True)
        self.dropout = nn.Dropout(0.1)
        self.classifier_polarity = nn.Linear(hidden_size, num_labels_polarity)
        self.classifier_type = nn.Linear(hidden_size, num_labels_type)
        self.classifier_town = nn.Linear(hidden_size, num_labels_town)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        return {
            "polarity": self.classifier_polarity(pooled_output),
            "type": self.classifier_type(pooled_output),
            "town": self.classifier_town(pooled_output)
        }


if __name__ == "__main__":
    model_path = "models/BETO_MTL"
    output_dir = "models/BETO_MTL_encoder"

    print(f"🔹 Cargando arquitectura desde: {model_path}")
    model = MultiTaskModel(model_name=model_path)

    print(f"🔹 Cargando pesos (formato safetensors) desde: {model_path}/model.safetensors")
    state_dict = load_file(f"{model_path}/model.safetensors")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"✅ Pesos cargados (missing={len(missing)}, unexpected={len(unexpected)})")

    encoder = model.transformer
    encoder.save_pretrained(output_dir)
    print(f"✅ Encoder exportado a: {output_dir}")

