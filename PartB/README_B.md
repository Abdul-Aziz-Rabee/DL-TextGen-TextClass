Estructura del Proyecto
=======================
La estructura del proyecto está organizada de la siguiente manera:

```bash
PartB/
│
├── data/
│   ├── raw/
│   │   └── MeIA_2025_train.csv
│   └── label_mappings/
│       └── polarity_labels_original.json
│       └── town_labels_original.json
│       └── type_labels_original.json
├── src/
│   ├── data/
│   │   └── datasets.py
│   │   └── generate_label_mappings.py    # Genera mapeos de etiquetas originales
│   │   └── prepare_meia_mtl.py      # Prepara datos para multitarea
│   ├── preprocessing/
│   │   └── limpieza.py       # Implementa funciones de preprocesamiento
│   ├── train/
│   │   ├── train_rnnclf.py
│   │   └── train_cnnclf.py
│   │   └── train_mtl.py
│   │   └── train_mtl_lora.py        # BETO MTL + LoRA
│   ├── models/
│   │   ├── rnn_text.py
│   │   └── cnn_text.py
│   └── eval/
│       └── eval_utils_mtl.py  # Funciones de evaluación para BETO MTL
│       └── train_eval_final.py   # Evaluación final de modelos clásicos
├── logs/           # Logs de ejecución SLURM
├── results/                           # Métricas, gráficas, reportes
├── run_kfolds.sh                      # Entrenamiento clásico (5-Fold)
├── run_eval_archs.sh                  # Evaluación final clásica
└── run_train_mtl_lora.sh              # Fine-tuning multitarea BETO + LoRA
```

---

## Dependencias y entorno

Usa el **mismo entorno Conda** que la Parte A:

```bash
conda activate tarea2-nlp
```

**Librerías clave:**

- torch, torchvision, transformers, datasets, peft
- scikit-learn, numpy, pandas, matplotlib
- ftfy, tqdm

---

## Flujo de trabajo

### 1️⃣ Preparación del dataset

Archivo original en:

```
data/raw/MeIA_2025_train.csv
```

Preprocesamiento:

```bash
python -m src.preprocessing.limpieza
python -m src.data.prepare_meia_mtl
python -m src.data.generate_label_mappings
```

Esto genera los archivos tokenizados y los `label_mappings` en `data/label_mappings/`.

---

### 2️⃣ Entrenamiento de modelos clásicos (CNN, RNN, LSTM, GRU)

**Ejecución local**

- Ejemplo — entrenar una LSTM a nivel palabra:

```bash
python -m src.train.train_rnnclf \
  --rnn_type lstm --level word --epochs 30 \
  --batch_size 64 --lr 1e-4 \
  --data_dir data/processed/classif/five \
  --out_dir models/lstm_word \
  --patience 5
```

**Ejecución en el clúster (Lab-SB / CIMAT):**

- Entrenamiento con validación cruzada 5-Fold:

```bash
sbatch run_kfolds.sh
```

- Evaluación final sobre 80/20:

```bash
sbatch run_eval_archs.sh
```

Los modelos y métricas se guardan en `results/reports/`.
Las figuras (matrices de confusión, curvas, etc.) se guardan en `results/figures/`.

---

### 3️⃣ Fine-tuning multitarea con BETO + LoRA (Transformers)

El modelo base BETO MTL_SO debe estar disponible en:

```
models/BETO_MTL_SO/
```

Ejecuta en el clúster con soporte DDP (2× GPUs):

```bash
sbatch run_train_mtl_lora.sh
```

Esto lanza el entrenamiento multitarea completo (`Polarity`, `Type`, `Town`) utilizando `torchrun --nproc_per_node=2` para entrenamiento distribuido.

---

### 4️⃣ Evaluación y análisis

Los resultados se almacenan automáticamente en:

- `results/BETO_MTL_LoRA_MeIA_*.csv`
- `results/reports/`

Cada salida incluye:

- Accuracy y F1 por tarea (Polarity, Type, Town)
- Matrices de confusión en `.png`.
- Logs detallados de SLURM y métricas en formato CSV

---

## Reproducibilidad

- Semilla global fijada a 42 en todos los scripts
- Directorios de salida creados automáticamente
- Logs de ejecución y métricas almacenados en `results/` y `logs/`
- Compatible con CPU o GPU (Titan RTX, 24 GB VRAM probado)
- Scripts configurados vía argparse y rutas relativas para su uso en local o clúster

---

## Resultados esperados

### Modelos clásicos (CNN, RNN, LSTM, GRU)

| Modelo | Accuracy | F1 Macro | F1 Weighted | Observaciones |
|:--|:--:|:--:|:--:|:--|
| RNN | 0.464 | 0.459 | 0.462 | Coherente pero tendencia a confundir clases intermedias (2–3). |
| LSTM | **0.481** | **0.478** | **0.481** | Mejor desempeño global; equilibrio entre precisión y recall. |
| GRU | 0.461 | 0.459 | 0.459 | Estabilidad en validación cruzada (mínima varianza). |
| CNN | 0.470 | 0.465 | 0.467 | Buen rendimiento general, pero menos robusta semánticamente. |

**Conclusión:**  
Las arquitecturas recurrentes capturan dependencias locales con desempeño cercano (≈0.47–0.48 F1), mostrando limitaciones en reseñas con matices neutros o ambiguos.

---

### Transformer multitarea (BETO MTL + LoRA)

| Variable | Accuracy | F1 Weighted |
|-----------|-----------|-------------|
| Polarity | 0.687 | 0.683 |
| Type | 0.975 | 0.975 |
| Town | 0.754 | 0.755 |
| **Score oficial (2–1–3)** | — | **0.768** |

**Análisis:**  
- **Polarity:** mejora de +0.20 en F1 frente al mejor modelo clásico (LSTM).  
- **Type:** clasificación casi perfecta entre categorías de tipo turístico.  
- **Town:** desempeño competitivo pese a las 40 clases geográficas.  
- **Tiempo de entrenamiento:** ~1 min 12 s en 2× TITAN RTX (vs. >90 min del fine-tuning completo).  

**Conclusión:**  
El modelo **BETO MTL SO + LoRA** logra **una mejora del 60 % en eficiencia de entrenamiento** y un incremento claro en desempeño global (Score ≈ 0.77).  
El enfoque multitarea conserva el conocimiento previo del dominio REST-MEX y se adapta eficazmente al nuevo corpus MeIA 2025.

---

**Resumen general:**
> El paso de modelos clásicos a Transformers multitarea produjo un salto sustancial en rendimiento y eficiencia, validando el enfoque de transferencia y adaptación ligera (LoRA) en entornos HPC.

---


# Autor

**Uziel Luján**

**Maestría en Cómputo Estadístico — CIMAT**