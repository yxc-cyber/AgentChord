srun --account=bghs-delta-gpu --partition=gpuA40x4-interactive \
  --nodes=1 --gpus-per-node=4 --tasks=1 \
  --tasks-per-node=16 --cpus-per-task=1 --mem=64g \
  --pty bash