#!/bin/bash
#SBATCH --job-name=textclfkfolds
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --chdir=/home/est_posgrado_uziel.lujan/DL-TextGen-TextClass/PartB
#SBATCH --output=logs/%x-%j.log

set -e # Salir si hay error

# === CREAR CARPETA DE LOGS SI NO EXISTE ===
mkdir -p logs

# === FUNCIONA COMO WRAPPER PARA TODAS LAS ARQUITECTURAS ===

echo "=== Entrenando RNN ==="
python -m src.train.train_rnnclf --rnn_type rnn --level word --epochs 30

echo "=== Entrenando LSTM ==="
python -m src.train.train_rnnclf --rnn_type lstm --level word --epochs 30

echo "=== Entrenando GRU ==="
python -m src.train.train_rnnclf --rnn_type gru --level word --epochs 30

echo "=== Entrenando CNN ==="
python -m src.train.train_cnnclf --level word --epochs 30

echo "=== TODO FINALIZADO ==="
