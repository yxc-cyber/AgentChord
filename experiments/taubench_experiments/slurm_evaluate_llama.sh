#!/bin/bash
#SBATCH --job-name="agentchord_leval"
#SBATCH --output="logs/%j.%N.agentchord_eval.out"
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
source $SCRATCH/miniconda3/etc/profile.d/conda.sh
conda activate


# Run the vLLM host server in the background
vllm serve $SCRATCH/models/Llama-3.3-70B-Instruct \
        --served-model-name Llama3.3-70B-Instruct \
        --enable-auto-tool-choice \
        --tool-call-parser llama3_json \
        --chat-template src/agentchord/model/chat_templates/tool_chat_template_llama3.3_json.jinja \
        --tensor-parallel-size 4 \
        --quantization fp8 \
        --max-model-len 16384 \
        --max-num-seqs 1 \
        --gpu-memory-utilization 0.80 \
        --host 0.0.0.0 \
        --port 1911 \
        --api-key thisisakey \
        > vllm_server.log 2>&1 &
# vllm serve $SCRATCH/models/Qwen3-32B \
#         --served-model-name Qwen3-32B \
#         --enable-auto-tool-choice \
#         --tool-call-parser hermes \
#         --chat-template src/agentchord/model/chat_templates/tool_chat_template_qwen3_json_nothink.jinja \
#         --tensor-parallel-size 4 \
#         --quantization fp8 \
#         --max-model-len 16384 \
#         --max-num-seqs 1 \
#         --gpu-memory-utilization 0.80 \
#         --host 0.0.0.0 \
#         --port 1911 \
#         --api-key thisisakey \
#         > vllm_server.log 2>&1 &

VLLM_SERVER_PID=$!
echo "Started vLLM server with PID: $VLLM_SERVER_PID"
# Wait until the server is ready by checking /health endpoint
until [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:1911/health)" = "200" ]; do
    echo "Waiting for vLLM server to be ready..."
    sleep 5
done


# Evaluation script for experiments with different connection strategies

echo "Starting evaluation for experiments..."

# # Evaluate with baseline
# echo "=========================================="
# echo "Evaluation: baseline"
# echo "=========================================="
# uv run python experiments/taubench_experiments/[Qwen3-32B]_[baseline]_[Manager-Worker]/retail/evaluate.py

# # Evaluate with max_l1_norm connection strategy
# echo "=========================================="
# echo "Evaluation: max_l1_norm"
# echo "=========================================="
# uv run python experiments/taubench_experiments/[Llama3.3-70B-Instruct]_[product_probs]_[max_l1_norm]/evaluation_[Llama3.3-70B-Instruct]/evaluate.py


# # Evaluate with max_product_input connection strategy
# echo "=========================================="
# echo "Evaluation: max_product_input"
# echo "=========================================="
# uv run python experiments/taubench_experiments/[Llama3.3-70B-Instruct]_[product_probs]_[max_product_input]/evaluation_[Llama3.3-70B-Instruct]/evaluate.py


# Evaluate with mean_l1_norm connection strategy
echo "=========================================="
echo "Evaluation: mean_l1_norm"
echo "=========================================="
uv run python experiments/taubench_experiments/[Llama3.3-70B-Instruct]_[product_probs]_[mean_l1_norm]/evaluation_[Llama3.3-70B-Instruct]/evaluate.py


# Evaluate with mean_product_input connection strategy
echo "=========================================="
echo "Evaluation: mean_product_input"
echo "=========================================="
uv run python experiments/taubench_experiments/[Llama3.3-70B-Instruct]_[product_probs]_[mean_product_input]/evaluation_[Llama3.3-70B-Instruct]/evaluate.py


echo "=========================================="
echo "All evaluation completed!"
echo "=========================================="


# Kill the vLLM server after evaluation is done
echo "Killing vLLM server with PID: $VLLM_SERVER_PID"
kill $VLLM_SERVER_PID