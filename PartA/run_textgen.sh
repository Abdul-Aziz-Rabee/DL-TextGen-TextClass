#!/bin/bash
#SBATCH --job-name=textgen
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --chdir=/home/est_posgrado_uziel.lujan/DL-TextGen-TextClass/PartA
#SBATCH --output=logs/%x-%j.log

set -e

# Parámetros clave como argumentos con valores por defecto
ARCH=${1:-gru}
LEVEL=${2:-char}
NUM_LAYERS=${3:-3}
HIDDEN_SIZE=${4:-128}
DROPOUT=${5:-0.3}
EPOCHS=${6:-50}
BATCH_SIZE=${7:-32}
SEQ_LEN=${8:-50}
EMBED_DIM=${9:-128} 
LR=${10:-0.001}

DATA_PATH=${11:-data/canciones_clean.txt}
SAVE_DIR=${12:-models/}
RESULTS_DIR=${13:-results/}

mkdir -p logs

export PATH="/opt/anaconda_python311/bin:$PATH"

echo "Iniciando entrenamiento para $ARCH-$LEVEL en el cluster..."
conda run -n tarea2-nlp python src/train_textgen.py \
    --arch $ARCH \
    --level $LEVEL \
    --num_layers $NUM_LAYERS \
    --hidden_size $HIDDEN_SIZE \
    --dropout $DROPOUT \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --seq_len $SEQ_LEN \
    --embedding_dim $EMBED_DIM \
    --lr $LR \
    --data_path $DATA_PATH \
    --save_dir $SAVE_DIR \
    --results_dir $RESULTS_DIR \
    --mode train

echo "Entrenamiento completado para $ARCH-$LEVEL"
