# Evaluation script for experiments with different connection strategies

echo "Starting evaluation for experiments..."

# # Evaluate with baseline
# echo "=========================================="
# echo "Evaluation: baseline"
# echo "=========================================="
# uv run python experiments/taubench_experiments/[Qwen3-32B]_[baseline]_[Manager-Worker]/retail/evaluate.py

# Evaluate with max_l1_norm connection strategy
echo "=========================================="
echo "Evaluation: max_l1_norm"
echo "=========================================="
uv run python experiments/taubench_experiments/[Qwen3-32B]_[product_probs]_[max_l1_norm]/evaluation_[Qwen3-32B]/evaluate.py


# Evaluate with max_product_input connection strategy
echo "=========================================="
echo "Evaluation: max_product_input"
echo "=========================================="
uv run python experiments/taubench_experiments/[Qwen3-32B]_[product_probs]_[max_product_input]/evaluation_[Qwen3-32B]/evaluate.py


# # Evaluate with mean_l1_norm connection strategy
# echo "=========================================="
# echo "Evaluation: mean_l1_norm"
# echo "=========================================="
# uv run python experiments/taubench_experiments/[Qwen3-32B]_[product_probs]_[mean_l1_norm]/evaluation_[Qwen3-32B]/evaluate.py


# Evaluate with mean_product_input connection strategy
echo "=========================================="
echo "Evaluation: mean_product_input"
echo "=========================================="
uv run python experiments/taubench_experiments/[Qwen3-32B]_[product_probs]_[mean_product_input]/evaluation_[Qwen3-32B]/evaluate.py


echo "=========================================="
echo "All evaluation completed!"
echo "=========================================="