#!/bin/bash
#SBATCH --job-name=partA-char
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --chdir=/home/est_posgrado_uziel.lujan/DL-TextGen-TextClass/PartA
#SBATCH --output=logs/%x-%j.log

set -e

RUN_NAME=${1:-"char_rnn_v1"}
EPOCHS=${2:-30}
BATCH_SIZE=${3:-64}
SEQ_LEN=${4:-256}

mkdir -p logs results models

export PATH="/opt/anaconda_python311/bin:$PATH"

echo "Iniciando entrenamiento carácter-level RNN con 2 GPUs..."
conda run -n llms-mx-env torchrun --nproc_per_node=2 src/02_train_rnn_char.py \
    --train_file data/train_char.txt \
    --valid_file data/valid_char.txt \
    --run_name "$RUN_NAME" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --seq_len "$SEQ_LEN" \
    --save_dir models/"$RUN_NAME" \
    --results_dir results/"$RUN_NAME"

echo "Entrenamiento completado."
