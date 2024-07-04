import os
import random
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from modules.models import FlightLSTM
from modules.feature_engineering import FeatureEngineeringV1


class FlightVisualizer:
	def __init__(self, model_checkpoint_path, data_path):
		self.model_checkpoint_path = model_checkpoint_path
		self.data_path = data_path
		self.feature_engineering = FeatureEngineeringV1()
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.model = self.load_model()
	
	def load_model(self):
		if not os.path.isfile(self.model_checkpoint_path):
			raise FileNotFoundError(f"Checkpoint path '{self.model_checkpoint_path}' is not a valid file.")
		model = FlightLSTM.load_from_checkpoint(self.model_checkpoint_path)
		model.to(self.device)
		model.eval()
		return model
	
	def preprocess_data(self):
		df = pd.read_csv(self.data_path)
		df = self.feature_engineering.common_preprocessing(df)
		df = self.feature_engineering.remove_unreasonable_time(df)
		df_normalized = self.feature_engineering.normalize_features(df)
		return df, df_normalized
	
	def get_random_flight(self, df, df_normalized):
		flight_ids = df['Ac_id'].unique()
		random_flight_id = random.choice(flight_ids)
		flight_data = df[df['Ac_id'] == random_flight_id]
		flight_data_normalized = df_normalized[df_normalized['Ac_id'] == random_flight_id]
		return flight_data, flight_data_normalized
	
	def predict_flight_path(self, flight_data_normalized):
		wind_conditions = flight_data_normalized[
			[col for col in flight_data_normalized.columns if 'wind_speed' in col or 'wind_dir' in col]].values
		wind_conditions = torch.FloatTensor(wind_conditions).unsqueeze(0).to(
			self.device)  # Add batch dimension and move to device
		with torch.no_grad():
			# Remove batch dimension and move to CPU
			predicted_path = self.model(wind_conditions).squeeze(0).cpu().numpy()
		return predicted_path
	
	def visualize_flight_path(self, actual_flight_data, predicted_path):
		fig = plt.figure(figsize=(10,20))
		ax = fig.add_subplot(111, projection='3d')
		
		# Actual path
		ax.plot(actual_flight_data['Ac_Lon'], actual_flight_data['Ac_Lat'], actual_flight_data['Ac_feet'],
		        label='Actual Path')
		
		# Predicted path
		predicted_path_decoded = self.feature_engineering.decode_features(
			pd.DataFrame(predicted_path, columns=['Ac_kts', 'Ac_Lat', 'Ac_Lon', 'Ac_feet']))
		ax.plot(predicted_path_decoded['Ac_Lon'], predicted_path_decoded['Ac_Lat'], predicted_path_decoded['Ac_feet'],
		        label='Predicted Path', linestyle='--')
		
		ax.set_xlabel('Longitude')
		ax.set_ylabel('Latitude')
		ax.set_zlabel('Altitude (feet)')
		ax.legend()
		plt.show()
	
	def visualize_flight_path_plotly(self, actual_flight_data, predicted_path):
		# Predicted path
		predicted_path_df = pd.DataFrame(predicted_path, columns=['Ac_kts', 'Ac_Lat', 'Ac_Lon', 'Ac_feet'])
		predicted_path_decoded = self.feature_engineering.decode_features(predicted_path_df)
		
		# Create 3D plot
		fig = make_subplots(
			rows=1, cols=1,
			specs=[[{'type': 'scatter3d'}]]
		)
		
		# Actual path
		actual_trace = go.Scatter3d(
			x=actual_flight_data['Ac_Lon'],
			y=actual_flight_data['Ac_Lat'],
			z=actual_flight_data['Ac_feet'],
			mode='lines',
			name='Actual Path'
		)
		
		# Predicted path
		predicted_trace = go.Scatter3d(
			x=predicted_path_decoded['Ac_Lon'],
			y=predicted_path_decoded['Ac_Lat'],
			z=predicted_path_decoded['Ac_feet'],
			mode='lines',
			name='Predicted Path',
			line=dict(dash='dash')
		)
		
		fig.add_trace(actual_trace, row=1, col=1)
		fig.add_trace(predicted_trace, row=1, col=1)
		
		fig.update_layout(
			scene=dict(
				xaxis_title='Longitude',
				yaxis_title='Latitude',
				zaxis_title='Altitude (feet)'
			),
			title='Actual vs Predicted Flight Path'
		)
		
		fig.show()
	
	def run_visualization(self, use_plotly=False):
		df, df_normalized = self.preprocess_data()
		actual_flight_data, flight_data_normalized = self.get_random_flight(df, df_normalized)
		predicted_path = self.predict_flight_path(flight_data_normalized)
		if use_plotly:
			self.visualize_flight_path_plotly(actual_flight_data, predicted_path)
		else:
			self.visualize_flight_path(actual_flight_data, predicted_path)


if __name__ == "__main__":
	checkpoint_path = '../log/epoch=55-val_loss=0.00.ckpt'
	data_path = '../data/csv/processed.csv'
	
	visualizer = FlightVisualizer(checkpoint_path, data_path)
	visualizer.run_visualization(use_plotly=True)
