# Method 1: Local server

Step 1: Environment Setup
```
pip install -r requirements.txt
```

Step 2: Training

```
python train.py trainer.devices=1
```

Change the value of `trainer.devices` for the number of GPUs to use.

Step 3: Result
The trained model will be saved in the `outputs` directory.

# Method 2: Docker
First building the Docker image following the instructions in [docker/how_to_build_docker_images.md](docker/how_to_build_docker_images.md).

Then running the container and mounting the current directory to `/workspace` in the container.

```bash
docker run --rm --gpus all -v $(pwd):/workspace debug_code:v1 python train.py
```



# Multi-GPU experiments

## 1. Logging
When using `wandb`, be careful that only rank 0 has access to `wandb_logger.experiment.id`.
Other ranks will return an object like this `<bound method _DummyExperiment.nop of <lightning_fabric.loggers.logger._DummyExperiment object at 0x7f952633aa90>>`.

In the case we use `wandb_logger.experiment.id` as the output path, there will be a folder with this strange name.

## 2. Gradient accumulating (Fisher)
