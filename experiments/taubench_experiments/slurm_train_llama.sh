#!/bin/bash
#SBATCH --job-name="agentchord_ltrain"
#SBATCH --output="logs/%j.%N.agentchord_train.out"
#SBATCH --partition=gpuA40x4
#SBATCH --mem=208G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  # could be 1 for py-torch
#SBATCH --cpus-per-task=16   # spread out to use 1 core per numa, set to 64 if tasks is 1
#SBATCH --constraint="work"
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=closest   # select a cpu close to gpu on pci bus topology
#SBATCH --account=bghs-delta-gpu
#SBATCH --exclusive  # dedicated node for this job
#SBATCH --no-requeue
#SBATCH -t 48:00:00
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out

module reset # drop modules and explicitly load the ones needed
             # (good job metadata and reproducibility)
             # $WORK and $SCRATCH are now set
module list  # job documentation and metadata

# Load necessary modules (e.g., CUDA, cuDNN, etc.) if required by your training script
source /work/hdd/bghs/xyang7/miniconda3/etc/profile.d/conda.sh
conda activate


# Training script for experiments with different connection strategies

echo "Starting training for experiments..."

# Train with max_l1_norm connection strategy
echo "=========================================="
echo "Training: max_l1_norm"
echo "=========================================="
uv run python experiments/taubench_experiments/[Llama3.3-70B-Instruct]_[product_probs]_[max_l1_norm]/training/train.py


# # Train with max_product_input connection strategy
# echo "=========================================="
# echo "Training: max_product_input"
# echo "=========================================="
# uv run python experiments/taubench_experiments/[Llama3.3-70B-Instruct]_[product_probs]_[max_product_input]/training/train.py


# # Train with mean_l1_norm connection strategy
# echo "=========================================="
# echo "Training: mean_l1_norm"
# echo "=========================================="
# uv run python experiments/taubench_experiments/[Llama3.3-70B-Instruct]_[product_probs]_[mean_l1_norm]/training/train.py


# # Train with mean_product_input connection strategy
# echo "=========================================="
# echo "Training: mean_product_input"
# echo "=========================================="
# uv run python experiments/taubench_experiments/[Llama3.3-70B-Instruct]_[product_probs]_[mean_product_input]/training/train.py


echo "=========================================="
echo "All training completed!"
echo "=========================================="
