#!/bin/bash
#SBATCH --job-name=beto_mtl_lora      # Nombre del job
#SBATCH --partition=GPU                        # Cola de GPUs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0                              # Memoria (0 = toda la memoria del nodo)
#SBATCH --time=08:00:00                        # Ajusta según la duración esperada
#SBATCH --chdir=/home/est_posgrado_uziel.lujan/DL-TextGen-TextClass/PartB
#SBATCH --output=logs/%x-%j.log  # Archivo de salida

# === Inicialización del entorno ===
echo "========================================================"
echo " Job ID: $SLURM_JOB_ID"
echo " Job Name: $SLURM_JOB_NAME"
echo "========================================================"

mkdir -p logs

# === Parámetros ===
MODEL_DIR="models/BETO_MTL"
OUT_DIR="models/beto_mtl_lora_polarity"
DATA_PATH=${1:-"data/processed/classif/polarity_small.csv"}  # <-- Actualízalo cuando tengas el dataset
EPOCHS=${2:-5}
BATCH_SIZE=${3:-32}
LR=${4:-2e-5}
MAX_LEN=${5:-256}

# === Ejecución ===
echo "Iniciando fine-tuning LoRA sobre BETO-MTL..."
echo "Dataset: $DATA_PATH"
echo "Salida:  $OUT_DIR"

python -u src/train_BETO_MTL_LoRA.py \
    --base_model "$MODEL_DIR" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUT_DIR" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LR" \
    --max_length "$MAX_LEN"

echo "========================================================"
echo " Entrenamiento completado para BETO-MTL + LoRA"
echo "========================================================"
