# Useful commands
### Check Node usage
```bash
sinfo
```
### Interactive mode

```bash
salloc --partition=gc3-h100  --ntasks=1 --ntasks-per-node=1 --gpus-per-node=1 --cpus-per-task=32 --time=60
```

### Singularity 
```bash
source /etc/profile.d/modules.sh
module load singularity/3.11.5
module load openmpi/3.1.6
```

### building 
```bash
singularity build --fakeroot my_python_env.sif singularity/container.def
```
