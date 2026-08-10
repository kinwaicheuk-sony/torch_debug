import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.plugins.environments import ClusterEnvironment
from pytorch_lightning.plugins.environments import SLURMEnvironment
from torch.utils.data import DataLoader, TensorDataset
from hydra import main
from omegaconf import DictConfig
import signal
import os

class StandardClusterEnvironment(ClusterEnvironment):
    @property
    def creates_processes_externally(self) -> bool:
        """Return True if the cluster is managed (you don't launch processes yourself)"""
        return True

    def world_size(self) -> int:
        return int(os.environ["WORLD_SIZE"])

    def global_rank(self) -> int:
        return int(os.environ["RANK"])

    def local_rank(self) -> int:
        return int(os.environ["LOCAL_RANK"])

    def node_rank(self) -> int:
        return 0  # int(os.environ["NODE_RANK"])

    def main_address(self) -> str:
        return os.environ["MASTER_ADDR"]

    def main_port(self) -> int:
        return int(os.environ["MASTER_PORT"])

    def detect() -> bool:
        """Detects the environment settings corresponding to this cluster and returns ``True`` if they match."""
        return True

    def set_world_size(self, size: int) -> None:
        pass

    def set_global_rank(self, rank: int) -> None:
        pass

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

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

# Generate artificial data
torch.manual_seed(0)
X = torch.randn(10000, 10)
y = torch.randn(10000, 1)

# Create DataLoader
dataset = TensorDataset(X, y)
train_loader = DataLoader(dataset, batch_size=128, shuffle=True)

# Hydra configuration
@main(config_path="config", config_name="config")
def train(cfg: DictConfig):
    torch.distributed.init_process_group(init_method="env://")
    
    # Create a directory for the experiment
    experiment_dir = os.path.join('runs', f'simple_nn_experiment')
    os.makedirs(experiment_dir, exist_ok=True)

    # Initialize the model
    model = SimpleNN(cfg.model.input_dim, cfg.model.hidden_dim, cfg.model.output_dim, cfg.model.lr)

    # Initialize TensorBoard logger
    tb_logger = pl.loggers.TensorBoardLogger(save_dir='runs', name=f'simple_nn_experiment')

    # Initialize the ModelCheckpoint callback
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=experiment_dir,
        filename='checkpoint-{epoch:02d}-{train_loss:.2f}',
        save_top_k=1,
        verbose=False,
        monitor='train_loss',
        mode='min',
        save_last=True
    )

    from pytorch_lightning.strategies import DDPStrategy
    cluster_env = StandardClusterEnvironment()
    strategy = DDPStrategy(find_unused_parameters=True, cluster_environment=cluster_env, process_group_backend="nccl")

    # Initialize the trainer
    if 'SLURM_JOB_ID' in os.environ:
        cfg.trainer.enable_progress_bar = False
        
        trainer = pl.Trainer(
            **cfg.trainer,
            logger=tb_logger,
            callbacks=[checkpoint_callback],
            strategy=strategy,
            # plugins=[SLURMEnvironment(requeue_signal=signal.SIGUSR1)]
        )
    else:
        cfg.trainer.enable_progress_bar = True
        trainer = pl.Trainer(
            **cfg.trainer,
            strategy=strategy,
            logger=tb_logger,
            callbacks=[checkpoint_callback]
        )

    # Train the model
    trainer.fit(model, train_loader)

if __name__ == "__main__":
    train()
