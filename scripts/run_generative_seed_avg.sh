#!/bin/bash
#SBATCH --job-name=mvprobe_gen_seeds
#SBATCH -o ./slurm/o/%x.o%j
#SBATCH -e ./slurm/e/%x.e%j
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Reproduce Table 2: Generative (SD_200 + SD_1k) with seeds 1-5
# Usage: sbatch scripts/run_generative_seed_avg.sh

DATA_ROOT="/path/to/dataset"
OUTPUT="./outputs/generative_seed_avg"

for DATASET in SD_200 SD_1k; do
    for SEED in 1 2 3 4 5; do
        echo ">>> ${DATASET} Layer 46 seed=${SEED}"
        python train_generative_probex.py \
            --input_path "${DATA_ROOT}/${DATASET}" \
            --output_path "${OUTPUT}/${DATASET}_L46_seed${SEED}" \
            --subset ${DATASET} \
            --start_layer 46 --n_layers 1 \
            --seed ${SEED}
    done
done

echo "=== Done: $(date) ==="
