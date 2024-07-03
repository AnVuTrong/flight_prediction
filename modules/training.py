import os

import lightning as pl
from modules.models import FlightLSTM
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme


class TrainingFlightModel:
	def __init__(
			self,
			early_stopping_lstm,
			model_checkpoint_lstm,
			logger_lstm,
			data_module,
			max_epoch,
	):
		progress_bar = RichProgressBar()
		self.trainer_lstm = pl.Trainer(
			max_epochs=max_epoch,
			accelerator='gpu',
			devices=1,
			callbacks=[early_stopping_lstm, model_checkpoint_lstm],
			log_every_n_steps=1,
			logger=logger_lstm,
			enable_progress_bar=True,
			gradient_clip_val=1.0,
		)
		self.data_module = data_module
		self.best_model_path_lstm = None
		self.loaded_model_lstm = None
	
	def training(self, input_size=36, hidden_size=300, num_layers=5, output_size=4, learning_rate=1e-4):
		# Create and train the LSTM model
		model_lstm = FlightLSTM(
			input_size=input_size,
			hidden_size=hidden_size,
			num_layers=num_layers,
			output_size=output_size,
			learning_rate=learning_rate
		)
		self.trainer_lstm.fit(model_lstm, self.data_module)
	
	def validate_and_testing_lstm(self, data_module):
		# Validate the LSTM model with the best checkpoint
		best_model_path_lstm = self.trainer_lstm.checkpoint_callback.best_model_path
		if not os.path.isfile(best_model_path_lstm):
			raise FileNotFoundError(f"Checkpoint path '{best_model_path_lstm}' is not a valid file.")
		loaded_model_lstm = FlightLSTM.load_from_checkpoint(best_model_path_lstm)
		
		self.trainer_lstm.validate(model=loaded_model_lstm, dataloaders=data_module.val_dataloader())
		self.trainer_lstm.test(model=loaded_model_lstm, dataloaders=data_module.test_dataloader())
