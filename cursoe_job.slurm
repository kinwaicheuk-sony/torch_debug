#!/bin/bash
#SBATCH --partition=sharedp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --job-name=debug
#SBATCH --output=slurm-logs/jobs/output.%N.%j.log
#SBATCH --error=slurm-logs/jobs/error.%N.%j.log
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --account=mt
#SBATCH --time=0-00:2:00
#SBATCH --signal=SIGUSR1@30
#SBATCH --requeue

# Load necessary modules (if needed)
# module load python/3.10

# Activate your virtual environment (if needed)
# source /home/kinwai.cheuk/mambaforge/envs/mini_v2/bin/activate

source /home/kinwai.cheuk/mambaforge/bin/activate
conda activate mini_1.9.0

# Run the Python script
srun python /mnt/beegfs/group/mfm/data/kinwai/torch_debug/train.py \
trainer.devices=2 \
trainer.max_epochs=20000