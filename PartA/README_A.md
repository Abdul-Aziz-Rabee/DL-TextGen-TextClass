# Parte A — Generación de letras de canciones

Esta parte del proyecto se centra en la **generación de texto** (letras de canciones) usando modelos RNN, LSTM, GRU y Transformers.



# Parte A — Generación de letras de canciones

Esta parte implementa **modelos de generación de texto** (letras de canciones) en español utilizando dos familias de arquitecturas:

1. **Modelos clásicos:** RNN, LSTM y GRU (a nivel carácter y palabra).  
2. **Modelos Transformer:** LLaMA 3 + LoRA (fine-tuning 4-bit).



## 📂 Estructura

```bash
PartA/
├── data/           # Canciones limpias y splits (train/valid)
├── src/            # Código fuente (preprocesamiento, entrenamiento, generación)
├── models/         # Modelos entrenados y checkpoints
├── results/        # Métricas y gráficas
├── notebooks/      # Notebooks exploratorios y de pruebas
├── logs/           # Logs de ejecución SLURM
├── run_textgen.sh          # Entrenamiento clásico (RNN/LSTM/GRU)
├── run_train_LlaMA.sh      # Fine-tuning LLaMA 3 + LoRA
└── run_generate_Llama.sh   # Generación con modelo LLaMA 3
```

## Dependencias y entorno

Todas las librerías necesarias están especificadas en el archivo environment.yml.
Para crear el entorno (en local o en el clúster):

```bash
conda env create -f environment.yml
conda activate tarea2-nlp
```

## Flujo de trabajo

1️⃣ Descarga de canciones

Usa la API de Genius para descargar ≥ 100 letras:

```bash
python src/extraer_canciones.py
```
2️⃣ Limpieza y preprocesamiento
Genera `data/canciones_clean.txt` libre de metadatos y ruido.

```bash
python src/limpiar_canciones.py
```
Opcionalmente, analiza estadisticas del corpus:

```bash
python src/song_statistics.py
```
Genera histogramas en `results/figures/hist_longitudes.png`.



3️⃣ **Conversión a JSONL (para Transformers)**

Convierte el corpus limpio a formato JSONL, necesario para entrenar modelos tipo Transformer:

```bash
python src/convert_to_train_lyrics.py
```

Esto crea `data/train_lyrics.jsonl`, con una canción por línea (clave `"text"`).

---

4️⃣ **Entrenamiento de modelos clásicos (RNN/LSTM/GRU)**

**Ejecución local** (ejemplo: GRU a nivel palabra):

```bash
python -u src/train_textgen.py \
  --arch gru --level word --epochs 30 \
  --hidden_size 128 --embedding_dim 256 \
  --lr 5e-5 --batch_size 64 --seq_len 20 \
  --data_path data/canciones_clean.txt \
  --save_dir models/ --results_dir results/ \
  --mode train
```

**Ejecución en clúster (Lab-SB / CIMAT):**

```bash
sbatch run_textgen.sh gru word 2 128 0.2 30 64 20 256 5e-5 data/canciones_clean.txt models/ results/
```

Los logs se guardan en `logs/textgen-<jobid>.log` y los modelos/métricas en `models/gru_word/`.

---

5️⃣ **Generación de letras (con modelos clásicos)**

Para generar múltiples combinaciones de prompt y temperatura:

```bash
python src/batch_generator.py
```

Esto genera archivos `.txt` y metadatos `.json` en `results/batch/`.

---

6️⃣ **Fine-tuning de LLaMA 3 + LoRA (Transformers)**

El modelo base debe estar cacheado en `models/Meta-Llama-3-8B/`.

**Ejecución local:**

```bash
python -u src/train_llama.py \
  --model_path models/Meta-Llama-3-8B/snapshots/.../ \
  --data_path data/train_lyrics.jsonl \
  --run_name llama3_v4 --epochs 5 \
  --batch_size 2 --block_size 256 \
  --lr 1e-4 --dropout 0.2
```

**Ejecución en clúster:**

```bash
sbatch run_train_LlaMA.sh llama3_v4 5 2 256 1e-4 0.2
```

Los logs se guardan en `logs/llama3-<jobid>.log` y el modelo fine-tuneado en `models/llama3_v4/`.

---

7️⃣ **Generación de letras con LLaMA 3 fine-tuneado**

**Ejecución local:**

```bash
python -u src/generate_llama.py \
  --model_path results/llama3_v4 \
  --prompt "En la penumbra del día" \
  --max_new_tokens 200 --temperature 1.0 --top_p 0.95 \
  --output_dir results/LLaMA
```

**Ejecución en clúster:**

```bash
sbatch run_generate_Llama.sh "Bailando bajo la lluvia" 200 1.0 0.95 llama3_out
```

Esto genera `.txt` y `.json` con los parámetros en `results/llama3_out/`.


# 🔁 Reproducibilidad

- Semilla global fijada a 42 en todos los scripts.
- Logs de entrenamiento y configuraciones guardados en .csv y .json.
- Todos los scripts aceptan parámetros por línea de comando (argparse).
- Directorios de salida automáticamente creados.
- Compatible con CPU y GPU (Titan RTX, 24 GB VRAM probado en CIMAT).


# Autor

**Uziel Luján**

**Maestría en Cómputo Estadístico — CIMAT**
