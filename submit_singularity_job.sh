#!/bin/bash
#SBATCH --partition=mt_l40s
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --job-name=SIG_job
#SBATCH --output=slurm-logs/sig/output.%N.%j.log
#SBATCH --error=slurm-logs/sig/error.%N.%j.log
#SBATCH --gres=gpu:2
#SBATCH --account=mt
#SBATCH --time=14-00:00:00
#SBATCH --nodelist=mfmcl3

# Create logs directory
mkdir -p logs

# Install Python and pip, then install requirements and run training
singularity exec --nv \
    --bind $PWD:/workspace \
    singularity/pytorch.sif \
    bash -c "
        cd /workspace && \
        python3 -m pip install --user -r /opt/app/requirements.txt && \
        python3 train.py
    "