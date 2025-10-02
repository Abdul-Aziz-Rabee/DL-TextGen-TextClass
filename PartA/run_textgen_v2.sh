#!/bin/bash
# ==============================================================================
# SBATCH -- Directivas para el gestor de trabajos SLURM
# ==============================================================================
# -- Nombre del trabajo (usando el primer argumento, default: 'exp')
#SBATCH --job-name=textgen_
# -- Partición a la que se envía el trabajo (probablemente una con GPUs)
#SBATCH --partition=GPU
# -- Nodos y Tareas: 1 nodo, 1 tarea por nodo
#SBATCH --nodes=1
#SBATCH --ntasks=1
# -- CPUs por tarea: pides 16, lo cual es bueno para el preprocesamiento de datos
#SBATCH --cpus-per-task=16
#SBATCH --mem=0  # -- Memoria: 0 significa usar toda la memoria del nodo
# -- Tiempo máximo de ejecución del trabajo
#SBATCH --time=08:00:00
# -- Directorio de trabajo: el script se ejecutará desde aquí
#SBATCH --chdir=/home/est_posgrado_uziel.lujan/DL-TextGen-TextClass/PartA
# -- Archivo de salida: %x se reemplaza por el nombre del job, %j por el Job ID
#SBATCH --output=logs/%x-%j.log

# ==============================================================================
# CONFIGURACIÓN DEL ENTORNO Y DIAGNÓSTICOS
# ==============================================================================
# -- Salir inmediatamente si un comando falla
set -e

echo "========================================================"
echo "          INICIANDO TRABAJO DE ENTRENAMIENTO"
echo "========================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Ejecutando en el nodo: $(hostname)"
echo "Directorio de trabajo: $(pwd)"
echo "--------------------------------------------------------"

# ==============================================================================
# PARÁMETROS Y EJECUCIÓN DEL SCRIPT
# ==============================================================================
echo "Configurando parámetros de entrenamiento..."
# -- Parámetros clave como argumentos con valores por defecto
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

# -- Se mantienen los paths por defecto, pero pueden ser sobrescritos
DATA_PATH=${11:-data/canciones_clean.txt}
SAVE_DIR=${12:-models/}
RESULTS_DIR=${13:-results/}

echo "Iniciando script de Python para el modelo $ARCH-$LEVEL..."
# -- 3. Ejecutar el script de Python
#    La clave aquí es el flag '-u' para la salida sin búfer (dinámica)
python -u src/train_textgen.py \
    --arch "$ARCH" \
    --level "$LEVEL" \
    --num_layers "$NUM_LAYERS" \
    --hidden_size "$HIDDEN_SIZE" \
    --dropout "$DROPOUT" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --seq_len "$SEQ_LEN" \
    --embedding_dim "$EMBED_DIM" \
    --lr "$LR" \
    --data_path "$DATA_PATH" \
    --save_dir "$SAVE_DIR" \
    --results_dir "$RESULTS_DIR" \
    --mode train

echo "========================================================"
echo "          TRABAJO DE ENTRENAMIENTO COMPLETADO"
echo "========================================================"