import torch
from torch import nn, optim
import torchmetrics as tm
import lightning as pl
from torchmetrics.regression import R2Score, MeanAbsoluteError


class MSE4DLoss(nn.Module):
	def __init__(self):
		super(MSE4DLoss, self).__init__()
	
	def forward(self, y_pred, y_true):
		# Calculate MSE for each dimension
		mse = ((y_pred - y_true) ** 2).mean(dim=1).sum()
		return mse


class FlightLSTM(pl.LightningModule):
	def __init__(self, input_size=42, hidden_size=300, num_layers=10, output_size=4, learning_rate=1e-4, dropout=0.1):
		super().__init__()
		self.save_hyperparameters()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.output_size = output_size
		self.learning_rate = learning_rate
		self.dropout = dropout
		
		# Define LSTM layers
		self.lstm = nn.LSTM(
			batch_first=True,
			input_size=self.input_size,
			hidden_size=self.hidden_size,
			num_layers=self.num_layers,
			dropout=self.dropout,
		)
		
		# Define a fully connected layer for each time step
		self.fc = nn.Linear(self.hidden_size, self.output_size)
		
		# Define ReLU activation function
		self.relu = nn.ReLU()
		
		# Define the custom 4D MSE loss function
		self.loss = MSE4DLoss()
		
		# Define the metrics
		self.mse = tm.MeanSquaredError()
		self.mae = MeanAbsoluteError()
		self.r2 = R2Score(num_outputs=self.output_size)
		
		# Set example input array for TensorBoard graph logging
		self.example_input_array = torch.zeros((1, 6616, self.input_size))
	
	def forward(self, x):
		# Initialize hidden state with zeros
		h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
		c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
		
		# Forward propagate LSTM
		out, _ = self.lstm(x, (h0, c0))
		
		# Apply ReLU activation function
		out = self.relu(out)
		
		# Apply fully connected layer to each time step
		out = self.fc(out)
		return out
	
	def _flatten(self, y_hat, y):
		# Flatten the predictions and targets to 2D tensors
		y_hat_flat = y_hat.view(-1, self.output_size)
		y_flat = y.view(-1, self.output_size)
		return y_hat_flat, y_flat
	
	def training_step(self, batch, batch_idx):
		x, y = batch
		y_hat = self(x)
		y_hat_flat, y_flat = self._flatten(y_hat, y)
		loss = self.loss(y_hat_flat, y_flat)
		mse = self.mse(y_hat_flat, y_flat)
		mae = self.mae(y_hat_flat, y_flat)
		r2 = self.r2(y_hat_flat, y_flat)
		self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
		self.log('train_mse', mse, on_step=True, on_epoch=True, prog_bar=True, logger=True)
		self.log('train_mae', mae, on_step=True, on_epoch=True, prog_bar=True, logger=True)
		self.log('train_r2', r2, on_step=True, on_epoch=True, prog_bar=True, logger=True)
		return loss
	
	def validation_step(self, batch, batch_idx):
		x, y = batch
		y_hat = self(x)
		y_hat_flat, y_flat = self._flatten(y_hat, y)
		loss = self.loss(y_hat_flat, y_flat)
		mse = self.mse(y_hat_flat, y_flat)
		mae = self.mae(y_hat_flat, y_flat)
		r2 = self.r2(y_hat_flat, y_flat)
		self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
		self.log('val_mse', mse, on_step=True, on_epoch=True, prog_bar=True, logger=True)
		self.log('val_mae', mae, on_step=True, on_epoch=True, prog_bar=True, logger=True)
		self.log('val_r2', r2, on_step=True, on_epoch=True, prog_bar=True, logger=True)
		return loss
	
	def test_step(self, batch, batch_idx):
		x, y = batch
		y_hat = self(x)
		y_hat_flat, y_flat = self._flatten(y_hat, y)
		loss = self.loss(y_hat_flat, y_flat)
		mse = self.mse(y_hat_flat, y_flat)
		mae = self.mae(y_hat_flat, y_flat)
		r2 = self.r2(y_hat_flat, y_flat)
		self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
		self.log('test_mse', mse, on_step=True, on_epoch=True, prog_bar=True, logger=True)
		self.log('test_mae', mae, on_step=True, on_epoch=True, prog_bar=True, logger=True)
		self.log('test_r2', r2, on_step=True, on_epoch=True, prog_bar=True, logger=True)
		return loss
	
	def configure_optimizers(self):
		optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
		scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
		return [optimizer], [scheduler]
