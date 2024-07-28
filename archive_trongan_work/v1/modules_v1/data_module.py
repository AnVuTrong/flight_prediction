import os
import torch
import lightning as pl


class FlightDataModule(pl.LightningDataModule):
	def __init__(self, train_dataset, val_dataset, test_dataset, batch_size: int = 32):
		super().__init__()
		self.train_dataset = train_dataset
		self.val_dataset = val_dataset
		self.test_dataset = test_dataset
		self.batch_size = batch_size
		self.num_workers = os.cpu_count()
	
	def train_dataloader(self):
		return torch.utils.data.DataLoader(
			self.train_dataset,
			batch_size=self.batch_size,
			shuffle=True,
			num_workers=self.num_workers,
		)
	
	def val_dataloader(self):
		return torch.utils.data.DataLoader(
			self.val_dataset,
			batch_size=self.batch_size,
			shuffle=False,
			num_workers=self.num_workers,
		)
	
	def test_dataloader(self):
		return torch.utils.data.DataLoader(
			self.test_dataset,
			batch_size=self.batch_size,
			shuffle=False,
			num_workers=self.num_workers,
		)
