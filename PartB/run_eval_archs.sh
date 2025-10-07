#!/bin/bash
#SBATCH --job-name=textclf
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
python -m src.eval.train_eval_final --arch_type rnn --level word --bidirectional --epochs 25 --pool max

echo "=== Entrenando LSTM ==="
python -m src.eval.train_eval_final --arch_type lstm --level word --bidirectional --epochs 25 --pool max

echo "=== Entrenando GRU ==="
python -m src.eval.train_eval_final --arch_type gru --level word --bidirectional --epochs 25 --pool max

echo "=== Entrenando CNN ==="
python -m src.eval.train_eval_final --arch_type cnn --level word --epochs 25

echo "=== TODO FINALIZADO ==="
