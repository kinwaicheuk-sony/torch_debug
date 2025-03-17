# Useful commands
conda can be installed in the login node

### Check Node usage
```bash
sinfo
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
