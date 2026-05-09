#!/bin/bash
#SBATCH --job-name=mvprobe_downstream
#SBATCH -o ./slurm/o/%x.o%j
#SBATCH -e ./slurm/e/%x.e%j
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Downstream evaluation: OCC, kNN classification, retrieval
# Usage: sbatch scripts/run_downstream.sh

DATA_ROOT="/path/to/dataset"
CHECKPOINT="./outputs/generative/SD_200_L46/ProbeX/abl-none/n_probes-128/proj_dim-128/rep_dim-512/checkpoints/best_val_holdout_layer-46__*.safetensors"
SUBSET="SD_200"

python downstream_generative.py \
    --task all \
    --input_path "${DATA_ROOT}/${SUBSET}" \
    --checkpoint_path ${CHECKPOINT} \
    --subset ${SUBSET} \
    --layer_idx 46

echo "=== Done: $(date) ==="
