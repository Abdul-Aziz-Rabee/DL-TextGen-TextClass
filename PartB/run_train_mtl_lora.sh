#!/bin/bash
#SBATCH --job-name=beto_mtl_lora_ddp
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --chdir=/home/est_posgrado_uziel.lujan/DL-TextGen-TextClass/PartB
#SBATCH --output=logs/%x-%j.log

set -e
mkdir -p logs

echo "========================================================"
echo " Job ID: $SLURM_JOB_ID"
echo " Job Name: $SLURM_JOB_NAME"
echo "========================================================"


echo "🚀 Iniciando entrenamiento multitarea con LoRA (DDP) ..."

# === Ejecutar con torchrun para usar 2 GPUs ===
torchrun --nproc_per_node=2 src/train_mtl_lora.py

EXIT_CODE=$?
echo "========================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Entrenamiento completado exitosamente (BETO-MTL + LoRA + DDP)"
else
    echo "❌ Entrenamiento finalizado con errores (exit code: $EXIT_CODE)"
fi
echo "========================================================"

