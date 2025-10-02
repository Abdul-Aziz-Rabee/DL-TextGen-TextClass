#!/bin/bash
#SBATCH --job-name=llama3_generate
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --chdir=/home/est_posgrado_uziel.lujan/DL-TextGen-TextClass/PartA
#SBATCH --output=logs/%x-%j.log

set -e

MODEL_PATH="results/llama3_v3"
PROMPT="En la penumbra del día,\n"
MAX_NEW_TOKENS=200
TEMPERATURE=1.0
TOP_P=0.95
OUTPUT_DIR="results/LlaMA"

mkdir -p logs

echo "Iniciando generación de texto con LLaMA..."

python -u src/generate_llama.py \
    --model_path "$MODEL_PATH" \
    --prompt "$PROMPT" \
    --max_new_tokens $MAX_NEW_TOKENS \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --output_dir "$OUTPUT_DIR"

echo "Generación de texto completada."