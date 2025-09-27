# Parte A — Generación de letras de canciones

Esta parte del proyecto se centra en la **generación de texto** (letras de canciones) usando modelos RNN, LSTM, GRU y Transformers.

## 📂 Estructura

```bash
PartA/
├── data/           # Canciones limpias y splits (train/valid)
├── src/            # Código fuente (preprocesamiento, entrenamiento, generación)
├── models/         # Modelos entrenados y checkpoints
├── results/        # Métricas y gráficas
├── logs/           # Logs de SLURM
└── run_scripts.sh  #  Scripts de lanzamiento para el clúster Lab-SB
```
# Flujo de trabajo

1. Preprocesamiento
Generar vocabularios y splits a nivel carácter y palabra:
```bash
python src/01_preprocesamiento.py
```


2. Entrenamiento
En el clúster Lab-SB (usando 2 GPUs):

- RNN a nivel carácter:
```bash
sbatch jobs/run_char.sh char_rnn_v1
```

- LSTM/GRU a nivel palabra:

```bash
sbatch jobs/run_word.sh word_lstm_v1
```

- Fine-tuning GPT-2 (previamente descargado con download_model_gen.py):

```bash
sbatch jobs/run_gpt2.sh models/gpt2_local gpt2_ft_v1
sbatch jobs/run_gpt2.sh models/gpt2_local gpt2_ft_v1
```

3. Generación de muestras
Una vez entrenado un modelo, se pueden generar letras:

```bash
python src/05_generate_samples.py \
    --model_dir models/gpt2_ft_v1 \
    --prompt "La vida es un sueño" \
    --max_length 200 \
    --temperature 0.8 \
    --top_k 50
```
# 📦 Dependencias

Se requieren las mismas dependencias descritas en el README global.
En caso de usar Conda:

```bash
conda env create -f jobs/environment.yml
```

# 📊 Resultados esperados

- Cuantitativos: Perplejidad (PPL) en validación.

- Cualitativos: ≥3 letras generadas por modelo, probando distintos parámetros:

  - Temperatura
  - Top-k / Top-p
  - Longitud objetivo
  - Prompt inicial

Los resultados deben analizarse en cuanto a:

- Coherencia temática
- Fluidez
- Repetición y clichés
- Variación léxica

# 📝 Notas

- Los modelos y tokenizadores deben estar descargados previamente en local y subidos a models/ ya que los nodos del clúster no tienen internet.

- Para reproducibilidad, se fijan semillas en los scripts (`utils.py`).

- Cada ejecución crea un subdirectorio dentro de `results`/ y `models`/ con el nombre del `run_name` indicado.