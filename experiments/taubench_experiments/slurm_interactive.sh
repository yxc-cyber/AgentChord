srun --account=bghs-delta-gpu --partition=gpuA40x4-interactive \
  --nodes=1 --gpus-per-node=4 \
  --tasks-per-node=4 --cpus-per-task=4 --mem=64g \
  --time=01:00:00 --pty bash