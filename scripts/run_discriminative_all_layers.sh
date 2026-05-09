#!/bin/bash
#SBATCH --job-name=mvprobe_all_layers
#SBATCH -o ./slurm/o/%x.o%j
#SBATCH -e ./slurm/e/%x.e%j
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Train MVProbe on all layers for a single model
# Usage: sbatch scripts/run_discriminative_all_layers.sh
# Modify MODEL, N_LAYERS, IS_RESNET below for each architecture

DATA_ROOT="/path/to/dataset"
OUTPUT="./outputs/all_layers"
SEED=1

MODEL="SupViT"
N_LAYERS=74
IS_RESNET="False"

# MODEL="ResNet"
# N_LAYERS=105
# IS_RESNET="True"

# MODEL="MAE"
# N_LAYERS=74
# IS_RESNET="False"

# MODEL="DINO"
# N_LAYERS=74
# IS_RESNET="False"

echo "=== ${MODEL} (All ${N_LAYERS} layers) ==="
for layer in $(seq 0 $((N_LAYERS - 1))); do
    echo ">>> ${MODEL} Layer $layer"
    python train_discriminative_probex.py \
        --input_path "${DATA_ROOT}/${MODEL}" \
        --output_path "${OUTPUT}/${MODEL}" \
        --is_resnet ${IS_RESNET} \
        --start_layer $layer --n_layers 1 \
        --seed ${SEED} \
        --cleanup_checkpoints True
done

echo "=== ${MODEL} Done: $(date) ==="
