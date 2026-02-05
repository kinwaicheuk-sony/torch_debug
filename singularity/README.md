# building Singularity Images on HPC
<details><summary>--fakeroot is not working</summary>
`--fakeroot` is needed, since we don't have root access on HPC.
Due to the permission problem on this particular HPC system (Crusoe), we need to build Singularity images in our home directory.

Move the `.def` and the `requirements.txt` file to home directory and run:

As an example, the content of `pytorch.def` could be:

```
Bootstrap: docker
From: nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

%files
    requirements.txt /opt/app/requirements.txt

%runscript
    exec python "$@"
```

```bash
singularity build --fakeroot pytorch.sif pytorch.def
```

This `.def` creates a Singularity image `.sif` with CUDA, but without installing the Python packages specified in `requirements.txt`. As `--fakeroot` causes problems with installing packages using `pip` inside the Singularity image, we will install the packages at runtime.
</details>

Use `singularity build --remote` instead.

To set up singularity remote, go to https://cloud.sylabs.io, then copy the access token.

On your machine, type `singularity remote login` and then paste the access token.

Then run:

```bash
singularity build --remote pytorch.sif pytorch.def
```


<details><summary>The example singularity config is as follows:</summary>

```bash
Bootstrap: docker
From: nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

%files
    requirements.txt /opt/app/requirements.txt
    
%post
    # Install wget
    apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*
    
    # Install Miniforge
    wget https://github.com/conda-forge/miniforge/releases/download/25.11.0-1/Miniforge3-25.11.0-1-Linux-x86_64.sh -O /tmp/miniforge3.sh
    bash /tmp/miniforge3.sh -b -p /opt/miniforge3
    rm /tmp/miniforge3.sh
    
    # Create environment with specific Python version
    /opt/miniforge3/bin/mamba create -n myenv python=3.10.2 -y
    
    # Install your requirements
    /opt/miniforge3/envs/myenv/bin/pip install -r /opt/app/requirements.txt

%environment
    export PATH=/opt/mambaforge/envs/myenv/bin:$PATH

%runscript
    exec python "$@"
```
</details>