#!/bin/bash
#SBATCH --partition=gc3-h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --job-name=debug
#SBATCH --output=slurm-logs/jobs/output.%N.%j.log
#SBATCH --error=slurm-logs/jobs/error.%N.%j.log
#SBATCH --mem=64G
#SBATCH --account=project55
#SBATCH --time=0-00:2:00

# Unset SLURM_CPUS_PER_TASK to avoid conflict
unset SLURM_CPUS_PER_TASK

# Load necessary modules (if needed)
# module load python/3.10

# Activate your virtual environment (if needed)
# source /home/kinwai.cheuk/mambaforge/envs/mini_v2/bin/activate

source /home/aa409938/miniforge3/bin/activate
conda activate /home/aa409938/miniforge3/envs/debug

# Run the Python script
srun python train.py \
trainer.devices=2 \
trainer.max_epochs=200
