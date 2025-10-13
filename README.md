# 🧠 DL-TextGen-TextClass

Proyecto de Deep Learning aplicado a **generación y clasificación de texto en español**, desarrollado en el contexto de la **Tarea 2** del curso *Procesamiento de Texto e Imágenes con Deep Learning* (Maestría en Cómputo Estadístico — CIMAT).

---

## 📂 Estructura del proyecto

```bash
DL-TextGen-TextClass/
├── PartA/                         # Generación de letras de canciones
│   ├── data/                      # Canciones crudas, limpias y JSONL
│   ├── src/                       # Scripts fuente de preprocesamiento, entrenamiento y generación
│   ├── models/                    # Modelos entrenados (RNN/LSTM/GRU/LLaMA)
│   ├── results/                   # Letras generadas, métricas y figuras
│   ├── logs/                      # Logs de SLURM
│   └── README_A.md                # Instrucciones específicas Parte A
│   ├── run_textgen.sh             # Entrenamiento clásico (RNN/LSTM/GRU)
│   ├── run_train_LlaMA.sh         # Fine-tuning LLaMA 3 + LoRA
│   └── run_generate_Llama.sh      # Generación con modelo LLaMA 3
│
├── PartB/                         # Clasificación de reseñas turísticas
│   ├── data/                      # Dataset MeIA 2025 y mapeos de etiquetas
│   ├── src/                       # Scripts de preprocesamiento, entrenamiento y evaluación
│   ├── models/                    # Checkpoints entrenados (clásicos y BETO MTL + LoRA)
│   ├── results/                   # Métricas, figuras y reportes
│   ├── logs/                      # Logs de SLURM
│   └── README_B.md                # Instrucciones específicas Parte B
│   ├── run_kfolds.sh              # Entrenamiento clásico (5-Fold)
│   ├── run_eval_archs.sh          # Evaluación final clásica
│   └── run_train_mtl_lora.sh      # Fine-tuning multitarea BETO + LoRA
│
├── environment.yml                # Entorno reproducible (Conda)
└── README.md                      # Descripción general del proyecto
```


---

## Dependencias y entorno

Para crear el entorno reproducible, ejecuta:

```bash
conda env create -f environment.yml -n tarea2-nlp
conda activate tarea2-nlp
```

**Librerías principales:**

- PyTorch ≥ 2.2
- Transformers, Datasets, PEFT, BitsAndBytes
- Scikit-learn, Numpy, Pandas, Matplotlib
- FTFY, TQDM, NLTK

---

## Ejecución general

El proyecto puede ejecutarse tanto en local como en el clúster de CIMAT (Lab-SB).

### Parte A — Generación de texto

```bash
cd PartA
sbatch run_textgen.sh gru word 2 128 0.2 30 64 20 256 5e-5 data/canciones_clean.txt models/ results/
```

O bien, para el modelo Transformer:

```bash
sbatch run_train_LlaMA.sh llama3_v4 5 2 256 1e-4 0.2
```

### Parte B — Clasificación de texto

```bash
cd PartB
sbatch run_kfolds.sh         # Entrenamiento clásico (RNN, LSTM, GRU, CNN)
sbatch run_train_mtl_lora.sh # Fine-tuning multitarea BETO + LoRA
```

---

## 📊 Resultados esperados

| Parte | Modelos | Métricas principales | Mejores resultados |
|-------|---------|----------------------|--------------------|
| A — Generación de letras | RNN, LSTM, GRU, LLaMA 3 + LoRA | Perplejidad (PPL), coherencia cualitativa | PPL ≈ 8.75 (LLaMA 3 + LoRA) |
| B — Clasificación de reseñas | CNN, RNN, LSTM, GRU, BETO MTL + LoRA | Accuracy, F1 (Macro/Weighted) | Score oficial = 0.7677 (BETO-MTL + LoRA) |

---

## 🔁 Reproducibilidad

- Semilla global = 42 en todos los scripts
- Entrenamientos y evaluaciones totalmente parametrizables vía argparse
- Logs y métricas almacenados automáticamente en `logs/` y `results/`
- Compatibilidad garantizada con CPU y GPU (Titan RTX 24 GB probada)
- Scripts `.sh` listos para SLURM (uso de `sbatch`, `torchrun`, etc.)

---

## 📘 Documentación detallada

- `PartA/README_A.md`: pipeline completo de generación de texto
- `PartB/README_B.md`: pipeline completo de clasificación multitarea


## Autor
Uziel Isaí Lujan López — M.Sc. in Statistical Computing at CIMAT

'uziel.lujan@cimat.mx'

[LinkedIn](https://www.linkedin.com/in/uziel-lujan/) | [GitHub](https://github.com/UzielLujan)