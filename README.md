# Local server

Step 1: Environment Setup
```
pip install -r requirements.txt
```

Step 2: Training

```
python train.py
```

Step 3: Result
The trained model will be saved in the `outputs` directory.



# Multi-GPU experiments

## 1. Logging
When using `wandb`, be careful that only rank 0 has access to `wandb_logger.experiment.id`.
Other ranks will return an object like this `<bound method _DummyExperiment.nop of <lightning_fabric.loggers.logger._DummyExperiment object at 0x7f952633aa90>>`.

In the case we use `wandb_logger.experiment.id` as the output path, there will be a folder with this strange name.

## 2. Gradient accumulating (Fisher)
