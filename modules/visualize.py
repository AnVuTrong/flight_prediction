import os
import dotenv
import random
import torch
import folium
import pandas as pd
import numpy as np
import pydeck as pdk
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from modules.models import FlightLSTM
from modules.feature_engineering import FeatureEngineeringV1

dotenv.load_dotenv()


class FlightVisualizer:
	def __init__(self, model_checkpoint_path, csv_path):
		self.map_box_api_key = os.getenv("MAPBOX_FLIGHT_PATH_RNN_API_KEY")
		self.model_checkpoint_path = model_checkpoint_path
		self.data_path = csv_path
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.model = self.load_model()
		self.fe = FeatureEngineeringV1()
		self.plotly_template = 'plotly'
	
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
	
	def visualize_flight_path(self, df, random_flight_id, predicted_path):
		actual_flight_data = df[df['Ac_id'] == random_flight_id]
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
		
		# Add points for the takeoff and landing
		ax.scatter(actual_flight_data['Ac_Lon'].iloc[0], actual_flight_data['Ac_Lat'].iloc[0],
		           actual_flight_data['Ac_feet'].iloc[0], color='green', label='Takeoff')
		ax.scatter(actual_flight_data['Ac_Lon'].iloc[-1], actual_flight_data['Ac_Lat'].iloc[-1],
		           actual_flight_data['Ac_feet'].iloc[-1], color='red', label='Landing')
		
		ax.set_xlabel('Longitude')
		ax.set_ylabel('Latitude')
		ax.set_zlabel('Altitude (feet)')
		ax.legend()
		plt.title('Actual vs Predicted Flight Path img')
		plt.show()
	
	def visualize_flight_path_cartopy(self, df, random_flight_id, predicted_path):
		actual_flight_data = df[df['Ac_id'] == random_flight_id]
		
		# Predicted path
		predicted_path_df = pd.DataFrame(predicted_path, columns=['Ac_kts', 'Ac_Lat', 'Ac_Lon', 'Ac_feet'])
		predicted_path_decoded = self.fe.decode_features(predicted_path_df)
		
		fig = plt.figure(figsize=(10, 15))
		ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
		
		# Plot the actual flight path
		ax.plot(
			actual_flight_data['Ac_Lon'],
			actual_flight_data['Ac_Lat'], 'b-',
			label='Actual Path',
			transform=ccrs.Geodetic(),
		)
		
		# Plot the predicted flight path
		ax.plot(
			predicted_path_decoded['Ac_Lon'],
			predicted_path_decoded['Ac_Lat'], 'r--',
			label='Predicted Path',
			transform=ccrs.Geodetic()
		)
		
		# Plot takeoff and landing points
		ax.plot(
			actual_flight_data['Ac_Lon'].iloc[0],
			actual_flight_data['Ac_Lat'].iloc[0], 'go',
			label='Takeoff',
			transform=ccrs.Geodetic()
		)
		ax.plot(
			actual_flight_data['Ac_Lon'].iloc[-1],
			actual_flight_data['Ac_Lat'].iloc[-1], 'ro',
			label='Landing', transform=ccrs.Geodetic()
		)
		
		# Add coastlines and borders
		ax.coastlines()
		ax.add_feature(cfeature.BORDERS)
		
		# Set the limits for the longitude to widen the span
		ax.set_xlim(100, 112)  # Adjust these values as needed
		
		ax.gridlines(draw_labels=True)
		ax.set_title('Flight Path Visualization using Cartopy')
		plt.legend()
		plt.show()
	
	def visualize_flight_path_plotly(self, df, random_flight_id, predicted_path):
		actual_flight_data = df[df['Ac_id'] == random_flight_id]
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
		
		# Add points for takeoff and landing
		takeoff_trace = go.Scatter3d(
			x=[actual_flight_data['Ac_Lon'].iloc[0]],
			y=[actual_flight_data['Ac_Lat'].iloc[0]],
			z=[actual_flight_data['Ac_feet'].iloc[0]],
			mode='markers',
			marker=dict(size=10, color='green'),
			name='Takeoff'
		)
		landing_trace = go.Scatter3d(
			x=[actual_flight_data['Ac_Lon'].iloc[-1]],
			y=[actual_flight_data['Ac_Lat'].iloc[-1]],
			z=[actual_flight_data['Ac_feet'].iloc[-1]],
			mode='markers',
			marker=dict(size=10, color='red'),
			name='Landing'
		)
		
		fig.add_trace(actual_trace, row=1, col=1)
		fig.add_trace(predicted_trace, row=1, col=1)
		fig.add_trace(takeoff_trace, row=1, col=1)
		fig.add_trace(landing_trace, row=1, col=1)

		fig.update_layout(
			scene=dict(
				xaxis_title='Longitude',
				yaxis_title='Latitude',
				zaxis_title='Altitude (feet)'
			),
			title='Actual vs Predicted Flight Path with Plotly',
			template=self.plotly_template,
			
		)
		fig.show()
	
	def visualize_flight_path_scattergeo(self, df, random_flight_id, predicted_path):
		actual_flight_data = df[df['Ac_id'] == random_flight_id]
		# Predicted path
		predicted_path_df = pd.DataFrame(predicted_path, columns=['Ac_kts', 'Ac_Lat', 'Ac_Lon', 'Ac_feet'])
		predicted_path_decoded = self.fe.decode_features(predicted_path_df)
		
		# Create the plot
		fig = go.Figure()
		
		# Actual path
		actual_trace = go.Scattergeo(
			lon=actual_flight_data['Ac_Lon'],
			lat=actual_flight_data['Ac_Lat'],
			mode='lines',
			name='Actual Path'
		)
		
		# Predicted path
		predicted_trace = go.Scattergeo(
			lon=predicted_path_decoded['Ac_Lon'],
			lat=predicted_path_decoded['Ac_Lat'],
			mode='lines',
			name='Predicted Path',
			line=dict(dash='dash')
		)
		
		# Add points for takeoff and landing
		takeoff_trace = go.Scattergeo(
			lon=[actual_flight_data['Ac_Lon'].iloc[0]],
			lat=[actual_flight_data['Ac_Lat'].iloc[0]],
			mode='markers',
			marker=dict(size=10, color='green'),
			name='Takeoff'
		)
		landing_trace = go.Scattergeo(
			lon=[actual_flight_data['Ac_Lon'].iloc[-1]],
			lat=[actual_flight_data['Ac_Lat'].iloc[-1]],
			mode='markers',
			marker=dict(size=10, color='red'),
			name='Landing'
		)
		
		fig.add_trace(actual_trace)
		fig.add_trace(predicted_trace)
		fig.add_trace(takeoff_trace)
		fig.add_trace(landing_trace)
		
		fig.update_layout(
			geo=dict(
				projection_type='natural earth',
				showland=True,
				landcolor='rgb(217, 217, 217)',
				subunitwidth=1,
				countrywidth=1,
				subunitcolor="rgb(255, 255, 255)",
				countrycolor="rgb(255, 255, 255)"
			),
			title='Actual vs Predicted Flight Path with Plotly',
			template=self.plotly_template,
		)
		
		fig.show()
	
	def visualize_flight_path_pydeck(self, df, random_flight_id, predicted_path):
		actual_flight_data = df[df['Ac_id'] == random_flight_id]
		predicted_path_df = pd.DataFrame(predicted_path, columns=['Ac_kts', 'Ac_Lat', 'Ac_Lon', 'Ac_feet'])
		predicted_path_decoded = self.fe.decode_features(predicted_path_df)
		
		# Actual path layer
		actual_path_layer = pdk.Layer(
			'PathLayer',
			actual_flight_data,
			get_path='[["Ac_Lon", "Ac_Lat", "Ac_feet"]]',
			get_color=[0, 0, 255],
			width_scale=20,
			width_min_pixels=2,
			get_width=5,
			pickable=True
		)
		
		# Predicted path layer
		predicted_path_layer = pdk.Layer(
			'PathLayer',
			predicted_path_decoded,
			get_path='[["Ac_Lon", "Ac_Lat", "Ac_feet"]]',
			get_color=[255, 0, 0],
			width_scale=20,
			width_min_pixels=2,
			get_width=5,
			pickable=True,
			dash_size=20,
			dash_gap_size=10
		)
		
		# Start and end points
		start_point = pdk.Layer(
			'ScatterplotLayer',
			data=[{'Lon': actual_flight_data['Ac_Lon'].iloc[0], 'Lat': actual_flight_data['Ac_Lat'].iloc[0],
			       'Alt': actual_flight_data['Ac_feet'].iloc[0]}],
			get_position='[Lon, Lat, Alt]',
			get_color=[0, 255, 0],
			get_radius=100,
		)
		
		end_point = pdk.Layer(
			'ScatterplotLayer',
			data=[{'Lon': actual_flight_data['Ac_Lon'].iloc[-1], 'Lat': actual_flight_data['Ac_Lat'].iloc[-1],
			       'Alt': actual_flight_data['Ac_feet'].iloc[-1]}],
			get_position='[Lon, Lat, Alt]',
			get_color=[255, 0, 0],
			get_radius=100,
		)
		
		# View
		view_state = pdk.ViewState(
			latitude=actual_flight_data['Ac_Lat'].mean(),
			longitude=actual_flight_data['Ac_Lon'].mean(),
			zoom=5,
			pitch=45,
			bearing=0
		)
		
		# Deck
		r = pdk.Deck(
			layers=[actual_path_layer, predicted_path_layer, start_point, end_point],
			initial_view_state=view_state,
			tooltip={"text": "{Ac_id}"}
		)
		
		r.to_html("../output/flight_path_pydeck.html")
		print("3D Flight path visualization saved as flight_path_pydeck.html")
	
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
	
	def save_folium_file(self, df, flight_id, predicted_path, name):
		m = self.visualize_flight_path_folium(
			df[df['Ac_id'] == flight_id],
			predicted_path
		)
		m.save(f'../output/{name}.html')
		print(f"Map saved as {name}.html")
	
	def print_actual_path(self, actual_flight_data):
		actual_path_df = actual_flight_data[['Ac_kts', 'Ac_Lat', 'Ac_Lon', 'Ac_feet']]
		print(f"Actual Path of flight{actual_flight_data['Ac_id'].iloc[0]}:")
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
		predicted_path = self.predict_flight_path(actual_flight_data)
		
		self.print_actual_path(df[df['Ac_id'] == random_flight_id])
		self.print_predicted_path(predicted_path)
		
		self.visualize_flight_path(
			df, random_flight_id, predicted_path
		)
		self.visualize_flight_path_cartopy(
			df, random_flight_id, predicted_path
		)
		self.visualize_flight_path_plotly(
			df, random_flight_id, predicted_path
		)
		self.visualize_flight_path_scattergeo(
			df, random_flight_id, predicted_path
		)
		self.visualize_flight_path_pydeck(
			df, random_flight_id, predicted_path
		)
		self.save_folium_file(
			df, random_flight_id, predicted_path, name
		)


if __name__ == "__main__":
	checkpoint_path = '../log/LSTM-epoch=33-val_loss=0.09058.ckpt'
	data_path = '../data/csv/raw.csv'
	
	visualizer = FlightVisualizer(checkpoint_path, data_path)
	visualizer.run_visualization()
