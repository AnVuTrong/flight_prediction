import os
import dotenv
import random
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import folium
from folium import plugins
from modules.models import FlightLSTM
from modules.feature_engineering import FeatureEngineeringV1

dotenv.load_dotenv()

class FlightVisualizer:
	def __init__(self, model_checkpoint_path, data_path):
		self.map_box_api_key = os.getenv("MAPBOX_FLIGHT_PATH_RNN_API_KEY")
		self.model_checkpoint_path = model_checkpoint_path
		self.data_path = data_path
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.model = self.load_model()
		self.fe = FeatureEngineeringV1()
	
	def load_model(self):
		model = FlightLSTM.load_from_checkpoint(self.model_checkpoint_path)
		model.to(self.device)
		model.eval()
		return model
	
	def get_random_flight(self, df):
		flight_ids = df['Ac_id'].unique()
		random_flight_id = random.choice(flight_ids)
		flight_data = df[df['Ac_id'] == random_flight_id]
		return random_flight_id, flight_data
	
	def predict_flight_path(self, flight_data_normalized):
		wind_conditions = flight_data_normalized[
			[col for col in flight_data_normalized.columns if 'wind_speed' in col or 'wind_dir' in col]
		].values
		
		ac_type_columns = [col for col in flight_data_normalized.columns if col.startswith('Ac_type_')]
		phase_columns = [col for col in flight_data_normalized.columns if col.startswith('Phase_')]
		other_features = flight_data_normalized[ac_type_columns + phase_columns + ['Time_step']].values
		
		# Combine all features into one tensor
		all_features = np.concatenate((wind_conditions, other_features), axis=1)
		all_features = torch.FloatTensor(all_features).unsqueeze(0).to(self.device)
		
		with torch.no_grad():
			predicted_path = self.model(all_features).squeeze(0).cpu().numpy()
		return predicted_path
	
	def visualize_flight_path(self, actual_flight_data, predicted_path):
		fig = plt.figure(figsize=(10, 20))
		ax = fig.add_subplot(111, projection='3d')
		
		# Actual path
		ax.plot(actual_flight_data['Ac_Lon'], actual_flight_data['Ac_Lat'], actual_flight_data['Ac_feet'],
		        label='Actual Path')
		
		# Predicted path
		predicted_path_decoded = self.fe.decode_features(
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
		predicted_path_decoded = self.fe.decode_features(predicted_path_df)
		
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
	
	def visualize_flight_path_folium(self, actual_flight_data, predicted_path):
		# Predicted path
		predicted_path_df = pd.DataFrame(predicted_path, columns=['Ac_kts', 'Ac_Lat', 'Ac_Lon', 'Ac_feet'])
		predicted_path_decoded = self.fe.decode_features(predicted_path_df)
		
		# Create map
		m = folium.Map(location=[actual_flight_data['Ac_Lat'].mean(), actual_flight_data['Ac_Lon'].mean()],
		               zoom_start=6)
		
		# Actual path
		actual_coords = actual_flight_data[['Ac_Lat', 'Ac_Lon']].values
		folium.PolyLine(actual_coords, color='blue', weight=2.5, opacity=1).add_to(m)
		
		# Predicted path
		predicted_coords = predicted_path_decoded[['Ac_Lat', 'Ac_Lon']].values
		folium.PolyLine(predicted_coords, color='red', weight=2.5, opacity=1, dash_array='5').add_to(m)
		
		# Add markers for start and end points
		folium.Marker(actual_coords[0], tooltip='Take off', icon=folium.Icon(color='green')).add_to(m)
		folium.Marker(actual_coords[-1], tooltip='Landing', icon=folium.Icon(color='red')).add_to(m)
		
		return m
	
	def visualize_flight_path_mapbox(self, actual_flight_data, predicted_path):
		# Predicted path
		predicted_path_df = pd.DataFrame(predicted_path, columns=['Ac_kts', 'Ac_Lat', 'Ac_Lon', 'Ac_feet'])
		predicted_path_decoded = self.fe.decode_features(predicted_path_df)
		
		# Create 3D plot with Mapbox
		fig = go.Figure()
		
		# Actual path
		fig.add_trace(go.Scattermapbox(
			lon=actual_flight_data['Ac_Lon'],
			lat=actual_flight_data['Ac_Lat'],
			mode='markers+lines',
			marker=dict(size=4),
			name='Actual Path'
		))
		
		# Predicted path
		fig.add_trace(go.Scattermapbox(
			lon=predicted_path_decoded['Ac_Lon'],
			lat=predicted_path_decoded['Ac_Lat'],
			mode='markers+lines',
			marker=dict(size=4),
			name='Predicted Path',
			line=dict(width=2)
		))
		
		fig.update_layout(
			mapbox=dict(
				accesstoken=self.map_box_api_key,
				style='mapbox://styles/mapbox/streets-v11',
				zoom=4,
				center=dict(lat=actual_flight_data['Ac_Lat'].mean(), lon=actual_flight_data['Ac_Lon'].mean())
			),
			title='Actual vs Predicted Flight Path with Speed and Time',
			height=800,
		)
		
		fig.show()
	
	def print_actual_path(self, actual_flight_data):
		actual_path_df = actual_flight_data[['Ac_kts', 'Ac_Lat', 'Ac_Lon', 'Ac_feet']]
		print("Actual Path (Target Variables):")
		print(actual_path_df)
	
	def print_predicted_path(self, predicted_path):
		predicted_path_df = pd.DataFrame(predicted_path, columns=['Ac_kts', 'Ac_Lat', 'Ac_Lon', 'Ac_feet'])
		predicted_path_decoded = self.fe.decode_features(predicted_path_df)
		print("Predicted Path (Target Variables):")
		print(predicted_path_decoded)

	def run_visualization(self, name='flight_path_map'):
		df = pd.read_csv(self.data_path)
		df_normalized = self.fe.process_data(self.data_path)
		random_flight_id, actual_flight_data = self.get_random_flight(df_normalized)
		self.print_actual_path(df[df['Ac_id'] == random_flight_id])
		predicted_path = self.predict_flight_path(actual_flight_data)
		self.print_predicted_path(predicted_path)
		
		m = self.visualize_flight_path_folium(
			df[df['Ac_id'] == random_flight_id],
			predicted_path
		)
		m.save(f'../output/{name}.html')
		print(f"Map saved as {name}.html")
		self.visualize_flight_path_plotly(actual_flight_data, predicted_path)
		self.visualize_flight_path_mapbox(actual_flight_data, predicted_path)
		self.visualize_flight_path(actual_flight_data, predicted_path)


if __name__ == "__main__":
	checkpoint_path = '../log/LSTM-epoch=33-val_loss=0.09058.ckpt'
	data_path = '../data/csv/raw.csv'
	
	visualizer = FlightVisualizer(checkpoint_path, data_path)
	visualizer.run_visualization()
