#!/bin/bash
#SBATCH --job-name=mvprobe_disc
#SBATCH -o ./slurm/o/%x.o%j
#SBATCH -e ./slurm/e/%x.e%j
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Reproduce Table 1: Discriminative (Model Jungle) best-layer results
# Usage: sbatch scripts/run_discriminative_best_layer.sh

DATA_ROOT="/path/to/dataset"
OUTPUT="./outputs/discriminative"
SEED=1

for MODEL_INFO in "SupViT:59:False" "ResNet:67:True" "MAE:64:False" "DINO:47:False"; do
    IFS=':' read -r MODEL LAYER IS_RESNET <<< "$MODEL_INFO"
    echo ">>> ${MODEL} Layer ${LAYER} (seed=${SEED})"
    python train_discriminative_probex.py \
        --input_path "${DATA_ROOT}/${MODEL}" \
        --output_path "${OUTPUT}/${MODEL}_L${LAYER}" \
        --is_resnet ${IS_RESNET} \
        --start_layer ${LAYER} --n_layers 1 \
        --seed ${SEED}
done

echo "=== Done: $(date) ==="
