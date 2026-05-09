#!/bin/bash
#SBATCH --job-name=mvprobe_ablation
#SBATCH -o ./slurm/o/%x.o%j
#SBATCH -e ./slurm/e/%x.e%j
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Reproduce ablation study: branch subset analysis
# Usage: sbatch scripts/run_ablation.sh

DATA_ROOT="/path/to/dataset"
OUTPUT="./outputs/ablation"
SEED=1

MODEL="SupViT"
LAYER=59
IS_RESNET="False"

for ABL in xu xtu xxtu xtxu first_order second_order none; do
    echo ">>> ${MODEL} L${LAYER} ablation=${ABL}"
    python train_discriminative_probex.py \
        --input_path "${DATA_ROOT}/${MODEL}" \
        --output_path "${OUTPUT}/${MODEL}_L${LAYER}_abl_${ABL}" \
        --is_resnet ${IS_RESNET} \
        --start_layer ${LAYER} --n_layers 1 \
        --ablation ${ABL} \
        --seed ${SEED}
done

echo "=== Done: $(date) ==="
