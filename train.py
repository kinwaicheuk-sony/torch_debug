import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from hydra import main, initialize, compose
from omegaconf import DictConfig
import datetime
import os

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

    # Initialize the trainer
    trainer = pl.Trainer(
        **cfg.trainer,
        logger=tb_logger,
        callbacks=[checkpoint_callback],
    )

    # Train the model
    trainer.fit(model, train_loader)

if __name__ == "__main__":
    train()
