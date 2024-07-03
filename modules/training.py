import lightning as pl
from modules.models import FlightLSTM


class TrainingFlightModel:
	def __init__(
			self,
			early_stopping_lstm,
			model_checkpoint_lstm,
			logger_lstm,
			data_module,
	):
		self.trainer_lstm = pl.Trainer(
			max_epochs=100,
			accelerator='gpu',
			devices=1,
			callbacks=[early_stopping_lstm, model_checkpoint_lstm],
			log_every_n_steps=1,
			logger=logger_lstm,
			enable_progress_bar=True,
			gradient_clip_val=1.0,
		)
		self.data_module = data_module
	
	def training(self, input_size=36, hidden_size=300, num_layers=5, output_size=4, learning_rate=1e-5):
		# Create and train the LSTM model
		model_lstm = FlightLSTM(
			input_size=input_size,
			hidden_size=hidden_size,
			num_layers=num_layers,
			output_size=output_size,
			learning_rate=learning_rate
		)
		self.trainer_lstm.fit(model_lstm, self.data_module)
