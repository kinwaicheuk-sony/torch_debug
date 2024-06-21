import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import datetime
import os

# Define a simple neural network using PyTorch Lightning
class SimpleNN(pl.LightningModule):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 1)
        self.criterion = nn.MSELoss()

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
        return torch.optim.Adam(self.parameters(), lr=0.001)

# Generate artificial data
torch.manual_seed(0)
X = torch.randn(10000, 10)
y = torch.randn(10000, 1)

# Create DataLoader
dataset = TensorDataset(X, y)
train_loader = DataLoader(dataset, batch_size=128, shuffle=True)

# Create a timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Create a directory for the experiment
experiment_dir = os.path.join('runs', f'simple_nn_experiment_{timestamp}')
os.makedirs(experiment_dir, exist_ok=True)

# Initialize TensorBoard logger
tb_logger = pl.loggers.TensorBoardLogger(save_dir='runs', name=f'simple_nn_experiment_{timestamp}')

# Initialize the model
model = SimpleNN()

# Initialize the trainer
trainer = pl.Trainer(max_epochs=200, logger=tb_logger, devices=1)

# Train the model
trainer.fit(model, train_loader)

# Save the model weights
torch.save(model.state_dict(), os.path.join(experiment_dir, 'checkpoint.pt'))
