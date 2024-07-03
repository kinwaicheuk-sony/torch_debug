# Step 1: prepare job scripts
The error logs will be stored in the directory where you submit the job via `sbatch train_job.slurm`. It is required to have the folder `slurm-logs` generated before the logs could be generated.


## Using venv
If you have an `activate` file inside `bin`, you can do the following.

```bash
#!/bin/bash
#SBATCH --partition=mt,sharedp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --job-name=my_job
#SBATCH --output=slurm-logs/output.%N.%j.log
#SBATCH --error=slurm-logs/error.%N.%j.log
#SBATCH --mem=10G
#SBATCH --gres=gpu:1
#SBATCH --account=mt
#SBATCH --requeue

# Load necessary modules (if needed)
module load python/3.10

# Activate your virtual environment (if needed)
source /path/to/your/venv/bin/activate

# Run the Python script
python train.py
```

If you create the environment using `conda create -n xxx`, the `activate` file might be missing. To get the `activate` file from conda environment, `conda-pack` is required. After unpacking the env using `tar`
```bash
conda pack -n my_env 
mkdir -p my_env
tar -xzf my_env.tar.gz -C my_env
```

## Using conda

```bash
#!/bin/bash
#SBATCH --partition=mt,sharedp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --job-name=my_job
#SBATCH --output=slurm-logs/output.%N.%j.log
#SBATCH --error=slurm-logs/error.%N.%j.log
#SBATCH --mem=10G
#SBATCH --gres=gpu:1
#SBATCH --account=mt
#SBATCH --requeue

# Load necessary modules (if needed)
# module load python/3.10

# Activate your virtual environment (if needed)
source /home/xxx.xxx/mambaforge/bin/activate
conda activate mini

# Run the Python script
python /mnt/beegfs/group/mt/kinwai/torch_debug/train.py
```

# Step 2: Submit jobs
```bash
sbatch train_job.slurm
```

To cancel jobs
```bash
scancel job_id
```

# Step 3: monitor jobs
To view the jobs from a specific user
```bash
squeue -u username
```

It will show you
```
 JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
  1924        mt train_ai username PD       0:00      1 (QOSGrpGRES)
```

`ST`: job state. `PD` means pending.

You can check the detail of the job using
`scontrol show job 1924`.

---

To view the all jobs from all users
```bash
squeue -a
```
---
To format the output column width
```bash
squeue -u your_username -o "%.18i %.30j %.9u %.2t %.10M %.6D %R"
```
- %.18i: Job ID (i) with a width of 18 characters.
- %.30j: Job name (j) with a width of 30 characters.
- %.9u: User name (u) with a width of 9 characters.
- %.2t: Job state (t) with a width of 2 characters.
- %.10M: Time used by the job (M) with a width of 10 characters.
- %.6D: Number of nodes (D) with a width of 6 characters.
- %R: Reason or node list (R), width is not specified to let it take the remaining space.




# Useful commands

### Interactive mode

```bash
sbash --partition=sharedp --cpus-per-task=16 --gpus=4 --mem=16G mfmc13
```

### Check disk quota
```bash
show_quota
```

### Check cluster usage
```bash
sig
```

# TODO
When using this config, I cannot send the jobs
```bash
#SBATCH --partition=mt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=my_job
#SBATCH --output=slurm-logs/output.%N.%j.log
#SBATCH --error=slurm-logs/error.%N.%j.log
#SBATCH --account=mt
#SBATCH --requeue
#SBATCH --time=1-00:00:00
```