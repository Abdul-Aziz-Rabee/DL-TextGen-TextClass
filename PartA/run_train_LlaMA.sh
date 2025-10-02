#!/bin/bash
#SBATCH --job-name=llama3
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --chdir=/home/est_posgrado_uziel.lujan/DL-TextGen-TextClass/PartA
#SBATCH --output=logs/%x-%j.log

set -e

MODEL_PATH="models/Meta-Llama-3-8B/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920/" # Ruta local al modelo cacheado
DATA_PATH="data/train_lyrics.jsonl"
RUN_NAME=${1:-"llama3_ft_v1"}
EPOCHS=${2:-3}
BATCH_SIZE=${3:-2}
BLOCK_SIZE=${4:-256}
LR=${5:-"1e-4"}
DROPOUT=${6:-"0.05"}

mkdir -p logs

echo "Iniciando fine-tuning LLaMA..."

python -u src/train_llama.py \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --run_name "$RUN_NAME" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --block_size "$BLOCK_SIZE" \
    --lr "$LR" \
    --dropout "$DROPOUT" \
    --save_dir "models/$RUN_NAME" \


echo "Fine-tuning completado."

