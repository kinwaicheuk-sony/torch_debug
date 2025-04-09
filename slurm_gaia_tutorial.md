# Useful commands
conda can be installed in the login node

### Check Node usage
```bash
sinfo
```

### Show resource allocated to a job
```
scontrol show job <job_id>
```

Example output
```
UserId=xxxx GroupId=xxxx MCS_label=N/A
Priority=1003333 Nice=0 Account=xxxx QOS=normal
JobState=RUNNING Reason=None Dependency=(null)
Requeue=0 Restarts=0 BatchFlag=1 Reboot=0 ExitCode=0:0
RunTime=00:28:08 TimeLimit=7-00:00:00 TimeMin=N/A
SubmitTime=2025-03-17T01:11:44 EligibleTime=2025-03-17T01:11:44
AccrueTime=2025-03-17T01:11:44
StartTime=2025-03-17T01:11:44 EndTime=2025-03-24T01:11:44 Deadline=N/A
SuspendTime=None SecsPreSuspend=0 LastSchedEval=2025-03-17T01:11:44 Scheduler=Main[6/259]
Partition=gc3-h100 AllocNode:Sid=0.0.0.0:1421319
ReqNodeList=(null) ExcNodeList=comp-h100-[9-51]
NodeList=comp-h100-1
BatchHost=comp-h100-1
NumNodes=1 NumCPUs=224 NumTasks=8 CPUs/Task=28 ReqB:S:C:T=0:0:*:*
ReqTRES=cpu=28,mem=245056M,node=1,billing=80,gres/gpu=8
AllocTRES=cpu=224,mem=1914.50G,node=1,billing=80,gres/gpu=8,gres/gpu:h100=8
Socks/Node=* NtasksPerN:B:S:C=8:0:*:1 CoreSpec=*
MinCPUsNode=224 MinMemoryCPU=8752M MinTmpDiskNode=0
Features=(null) DelayBoot=00:00:00
OverSubscribe=OK Contiguous=0 Licenses=(null) Network=(null)
```

### Interactive mode

```bash
salloc --partition=gc3-h100  --exclude=comp-h100-[9-51] --ntasks=8 --ntasks-per-node=8 --gpus-per-node=8 --cpus-per-task=32 --time=60
```

Attach to existing interactive seesion
```bash
srun --jobid=<JOBID> --pty bash
```

# Singularity 
https://github.com/bdusell/singularity-tutorial

In GAIA, singularity is not loaded by default. To load the modules:
```bash
source /etc/profile.d/modules.sh
module load singularity/3.11.5
module load openmpi/3.1.6
```

### building 
```bash
singularity build --fakeroot my_python_env.sif singularity/container.def
```

### using

Interactive mode
```bash
singularity shell --fakeroot --writable my_python_env.sif 
```

If you don't want the home dir

```bash
singularity shell --fakeroot --no-home --writable my_python_env.sif 
```

```bash
singularity exec my_python_env.sif python train.py
```
```bash
singularity exec --bind /home/kinwai/miniforge3/envs/debug:/home/kinwai/miniforge3/envs/debug my_python_env.sif python train.py
```


# Example files
slurm job with adjustable arguments

The following example can be submitted using
`sbatch --export=ALL,SILENCE_RATIO=0.25 silence_check.slurm`

The `#!/bin/bash` at the very beginning is very important, without this `conda activate` might not work.

```
#!/bin/bash
#SBATCH --partition=gc3-h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:8
#SBATCH --job-name=check0.25
#SBATCH --output=slurm-logs/jobs2/output.%x.%j.log
#SBATCH --error=slurm-logs/jobs2/error.%x.%j.log
#SBATCH --mem=128G
#SBATCH --account=project55
#SBATCH --time=1-00:00:00


# Unset SLURM_CPUS_PER_TASK to avoid conflict
unset SLURM_CPUS_PER_TASK

# Load necessary modules (if needed)
# module load python/3.10

# Activate your virtual environment
source /home/aa409938/miniforge3/bin/activate
conda activate /home/aa409938/miniforge3/envs/fsa2

# Ensure SLURM_NTASKS is set
if [[ -z "\$SLURM_NTASKS" ]]; then
    echo "Error: SLURM_NTASKS is not set."
    exit 1
fi

# Default value if not passed in externally
SILENCE_RATIO=${SILENCE_RATIO:-0.75}

# Run the Python script
srun python silence_check.py --silence-ratio $SILENCE_RATIO
```