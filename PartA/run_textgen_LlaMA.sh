#!/bin/bash
#SBATCH --job-name=partA-gpt2
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --chdir=/home/est_posgrado_uziel.lujan/DL-TextGen-TextClass/PartA
#SBATCH --output=logs/%x-%j.log

set -e

MODEL_PATH=${1:-"models/gpt2_local"}
RUN_NAME=${2:-"gpt2_ft_v1"}
EPOCHS=${3:-5}
BATCH_SIZE=${4:-16}
BLOCK_SIZE=${5:-256}

mkdir -p logs results models

export PATH="/opt/anaconda_python311/bin:$PATH"

echo "Iniciando fine-tuning LLaMA con 2 GPUs..."
conda run -n tarea2-nlp torchrun --nproc_per_node=2 src/train_llama.py \
    --model_name "$MODEL_PATH" \
    --train_file data/train_word.txt \
    --valid_file data/valid_word.txt \
    --run_name "$RUN_NAME" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --block_size "$BLOCK_SIZE" \
    --save_dir models/"$RUN_NAME" \
    --results_dir results/"$RUN_NAME"

echo "Fine-tuning completado."
