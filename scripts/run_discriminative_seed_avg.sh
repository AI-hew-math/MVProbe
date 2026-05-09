#!/bin/bash
#SBATCH --job-name=mvprobe_disc_seeds
#SBATCH -o ./slurm/o/%x.o%j
#SBATCH -e ./slurm/e/%x.e%j
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Reproduce Table 1: Discriminative (Model Jungle) with seeds 1-5
# Usage: sbatch scripts/run_discriminative_seed_avg.sh

DATA_ROOT="/path/to/dataset"
OUTPUT="./outputs/discriminative_seed_avg"

for MODEL_INFO in "SupViT:59:False" "ResNet:67:True" "MAE:64:False" "DINO:47:False"; do
    IFS=':' read -r MODEL LAYER IS_RESNET <<< "$MODEL_INFO"
    for SEED in 1 2 3 4 5; do
        echo ">>> ${MODEL} Layer ${LAYER} seed=${SEED}"
        python train_discriminative_probex.py \
            --input_path "${DATA_ROOT}/${MODEL}" \
            --output_path "${OUTPUT}/${MODEL}_L${LAYER}_seed${SEED}" \
            --is_resnet ${IS_RESNET} \
            --start_layer ${LAYER} --n_layers 1 \
            --seed ${SEED}
    done
done

echo "=== Done: $(date) ==="
