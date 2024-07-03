import torch
from torch import nn, optim
import torchmetrics as tm
import lightning as pl


class FlightLSTM(pl.LightningModule):
	def __init__(self, input_size=36, hidden_size=300, num_layers=5, output_size=4, learning_rate=1e-4):
		super().__init__()
		
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.output_size = output_size
		self.learning_rate = learning_rate
		
		# Define LSTM layers
		self.lstm = nn.LSTM(
			input_size=self.input_size,
			hidden_size=self.hidden_size,
			num_layers=self.num_layers,
			batch_first=True
		)
		
		# Define a fully connected layer for each time step
		self.fc = nn.Linear(self.hidden_size, self.output_size)
		
		# Define the loss function
		self.loss = nn.MSELoss()  # Use MSELoss for regression
		
		# Define the metric for regression
		self.mse = tm.MeanSquaredError()
		
		# Set example input array for TensorBoard graph logging
		self.example_input_array = torch.zeros((1, 6616, self.input_size))
	
	def forward(self, x):
		# Initialize hidden state with zeros
		h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
		c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
		
		# Forward propagate LSTM
		out, _ = self.lstm(x, (h0, c0))
		
		# Apply fully connected layer to each time step
		out = self.fc(out)
		return out
	
	def training_step(self, batch, batch_idx):
		x, y = batch
		y_hat = self(x)
		loss = self.loss(y_hat, y)
		mse = self.mse(y_hat, y)
		self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
		self.log('train_mse', mse, on_step=True, on_epoch=True, prog_bar=True, logger=True)
		return loss
	
	def validation_step(self, batch, batch_idx):
		x, y = batch
		y_hat = self(x)
		loss = self.loss(y_hat, y)
		mse = self.mse(y_hat, y)
		self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
		self.log('val_mse', mse, on_step=True, on_epoch=True, prog_bar=True, logger=True)
		return loss
	
	def test_step(self, batch, batch_idx):
		x, y = batch
		y_hat = self(x)
		loss = self.loss(y_hat, y)
		mse = self.mse(y_hat, y)
		self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
		self.log('test_mse', mse, on_step=True, on_epoch=True, prog_bar=True, logger=True)
		return loss
	
	def configure_optimizers(self):
		optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
		scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
		return [optimizer], [scheduler]
