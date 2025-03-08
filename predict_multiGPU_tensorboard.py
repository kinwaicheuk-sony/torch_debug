import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import pytorch_lightning as pl
from pytorch_lightning.plugins.environments import SLURMEnvironment
from pytorch_lightning.callbacks import BasePredictionWriter
from torch.utils.data import DataLoader, TensorDataset
from hydra import main
from omegaconf import DictConfig
import signal
import os

# set fixed seed for reproducibility
torch.manual_seed(0)

class LossInfo:
    def __init__(self, path, chunk_id, loss):
        self.path = path
        self.chunk_id = chunk_id
        self.loss = loss

    def get_csv_row(self, delimiter='|'):
        return delimiter.join([self.path, self.chunk_id])

    def to_dict(self):
        return {
            "path": self.path,
            "chunk_id": self.chunk_id,
            "loss": self.loss}
        
    def __repr__(self):
        return f"DataType(path={self.path}, chunk_id={self.chunk_id}, loss={self.loss})"

class LossWriter(BasePredictionWriter):

    def __init__(self, output_dir, write_interval='batch_and_epoch', write_every_n_steps=10):
        super().__init__(write_interval)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.write_every_n_steps = write_every_n_steps
        self.call_cnt = 0
        self.buffer: List[LossInfo] = []
        
    def write_on_batch_end(
            self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx
        ):
        self.call_cnt += 1
        self.buffer.extend(prediction)
        if self.call_cnt % self.write_every_n_steps == 0:
            self.flush(trainer.global_rank, self.call_cnt)
    
    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        if len(self.buffer) > 0:
            self.flush(trainer.global_rank, self.call_cnt)
        
    def flush(self, rank, call_cnt):
        
        with open(os.path.join(self.output_dir, f"losses_rank_{rank}_call_cnt_{call_cnt}.pt"), "wb") as f:
            torch.save(self.buffer, f)
        self.buffer.clear()

# Define a simple neural network using PyTorch Lightning
class SimpleNN(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, output_dim, lr):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.criterion = nn.MSELoss()
        self.lr = lr

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = self.criterion(outputs, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def predict_step(self, batch, batch_idx):
        x, y = batch

        # take mean on all dimensions
        outputs = self(x)
        loss = F.mse_loss(outputs, y, reduction='none').flatten()
        
        log_list =[]
        for i, loss in enumerate(loss):
            log_list.append(
                [
                LossInfo(
                    {batch_idx},
                    i,
                    loss.item()
                    )
                ]
            )

        return log_list        

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

# Generate artificial data
torch.manual_seed(0)
X = torch.randn(1000, 10)
y = torch.randn(1000, 1)

# Create DataLoader
dataset = TensorDataset(X, y)
train_loader = DataLoader(dataset, batch_size=10, shuffle=False)

# Hydra configuration
@main(config_path="config", config_name="config")
def train(cfg: DictConfig):
    # Create a directory for the experiment
    experiment_dir = os.path.join('runs', f'simple_nn_experiment')
    os.makedirs(experiment_dir, exist_ok=True)

    # Initialize the model
    model = SimpleNN(cfg.model.input_dim, cfg.model.hidden_dim, cfg.model.output_dim, cfg.model.lr)

    # Initialize TensorBoard logger and loss writer
    tb_logger = pl.loggers.TensorBoardLogger(save_dir='runs', name=f'simple_nn_experiment')
    loss_writer = LossWriter(f'./', write_every_n_steps=10)

    # Initialize the trainer
    if 'SLURM_JOB_ID' in os.environ:
        cfg.trainer.enable_progress_bar = False
        trainer = pl.Trainer(
            **cfg.trainer,
            logger=tb_logger,
            callbacks=[loss_writer],
            plugins=[SLURMEnvironment(requeue_signal=signal.SIGUSR1)]
        )
    else:
        cfg.trainer.enable_progress_bar = True
        trainer = pl.Trainer(
            **cfg.trainer,    
            logger=tb_logger,
            callbacks=[loss_writer]
        )

    # Train the model
    trainer.predict(model, train_loader)

if __name__ == "__main__":
    train()
