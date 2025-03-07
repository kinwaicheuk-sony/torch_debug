#!/bin/bash

# Set desired number of tasks per node (which also sets GPUs per node)
NUM_TASKS_PER_NODE=2  # Change this value as needed

# Generate the job script dynamically
cat <<EOF > job.slurm
#!/bin/bash
#SBATCH --partition=gc3-h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=$NUM_TASKS_PER_NODE
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:$NUM_TASKS_PER_NODE
#SBATCH --job-name=debug
#SBATCH --output=slurm-logs/jobs/output.%N.%j.log
#SBATCH --error=slurm-logs/jobs/error.%N.%j.log
#SBATCH --mem=64G
#SBATCH --account=project55
#SBATCH --time=0-00:60:00

# Unset SLURM_CPUS_PER_TASK to avoid conflict
unset SLURM_CPUS_PER_TASK

# Load necessary modules (if needed)
# module load python/3.10

# Activate your virtual environment
source /home/aa409938/miniforge3/bin/activate
conda activate /home/aa409938/miniforge3/envs/debug

# Ensure SLURM_NTASKS is set
if [[ -z "\$SLURM_NTASKS" ]]; then
    echo "Error: SLURM_NTASKS is not set."
    exit 1
fi

# Run the Python script
time srun python train.py \
    trainer.devices="\$SLURM_NTASKS" \
    trainer.max_epochs=200
EOF

# Submit the generated job
sbatch job.slurm
