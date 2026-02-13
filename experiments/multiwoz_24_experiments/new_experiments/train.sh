#!/bin/bash

# Training script for Qwen3-32B experiments with different connection strategies

echo "Starting training for Qwen3-32B experiments..."

# Train with max_l1_norm connection strategy
echo "=========================================="
echo "Training: max_l1_norm"
echo "=========================================="
uv run python experiments/multiwoz_24_experiments/new_experiments/[Qwen3-32B]_[product_probs]_[max_l1_norm]/training/train.py


# Train with max_product_input connection strategy
echo "=========================================="
echo "Training: max_product_input"
echo "=========================================="
uv run python experiments/multiwoz_24_experiments/new_experiments/[Qwen3-32B]_[product_probs]_[max_product_input]/training/train.py


# Train with mean_l1_norm connection strategy
echo "=========================================="
echo "Training: mean_l1_norm"
echo "=========================================="
uv run python experiments/multiwoz_24_experiments/new_experiments/[Qwen3-32B]_[product_probs]_[mean_l1_norm]/training/train.py


# Train with mean_product_input connection strategy
echo "=========================================="
echo "Training: mean_product_input"
echo "=========================================="
uv run python experiments/multiwoz_24_experiments/new_experiments/[Qwen3-32B]_[product_probs]_[mean_product_input]/training/train.py


echo "=========================================="
echo "All training completed successfully!"
echo "=========================================="
