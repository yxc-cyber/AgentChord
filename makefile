train:
	@echo "Training..."
	@sbatch experiments/multiwoz_24_experiments/new_experiments/slurm_train.sh

evaluate:
	@echo "Evaluating..."
	@sbatch experiments/multiwoz_24_experiments/new_experiments/slurm_evaluate.sh

interact:
	@echo "Interacting..."
	@sbatch experiments/multiwoz_24_experiments/new_experiments/slurm_interact.sh