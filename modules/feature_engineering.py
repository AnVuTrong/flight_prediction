import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader


class FeatureEngineeringV1:
	def __init__(self):
		self.scalers = {
			'wind_speed': MinMaxScaler(),
			'wind_dir'  : MinMaxScaler(),
			'lat_lon'   : MinMaxScaler(feature_range=(-1, 1)),
			'altitude'  : MinMaxScaler()
		}
		self.fitted = False
	
	def normalize_features(self, df):
		wind_speed_columns = df.columns[df.columns.str.startswith('wind_speed')]
		wind_dir_columns = df.columns[df.columns.str.startswith('wind_dir')]
		lat_lon_columns = ['Ac_Lat', 'Ac_Lon']
		altitude_columns = ['Ac_feet']
		
		df[wind_speed_columns] = self.scalers['wind_speed'].fit_transform(df[wind_speed_columns])
		df[wind_dir_columns] = self.scalers['wind_dir'].fit_transform(df[wind_dir_columns])
		df[lat_lon_columns] = self.scalers['lat_lon'].fit_transform(df[lat_lon_columns])
		df[altitude_columns] = self.scalers['altitude'].fit_transform(df[altitude_columns])
		
		self.fitted = True
		return df
	
	def remove_unreasonable_time(self, df, max_threshold=10000, min_flight_time=4000):
		# Remove rows with Time_step greater than max_threshold
		cleaned_df = df[df['Time_step'] < max_threshold]
		
		# Identify flights with maximum Time_step less than min_flight_time
		flights_to_remove = cleaned_df.groupby('Ac_id')['Time_step'].max() < min_flight_time
		id_to_remove = flights_to_remove[flights_to_remove].index
		
		# Remove these flights from the cleaned dataframe
		cleaned_df = cleaned_df[~cleaned_df['Ac_id'].isin(id_to_remove)]
		
		return cleaned_df
	
	def process_data(self, csv_path):
		df = pd.read_csv(csv_path)
		df = self.normalize_features(df)
		df = self.remove_unreasonable_time(df)
		return df
	
	def decode_features(self, df):
		if not self.fitted:
			raise Exception("Scalers not fitted. Call process_data first.")
		
		wind_speed_columns = df.columns[df.columns.str.startswith('wind_speed')]
		wind_dir_columns = df.columns[df.columns.str.startswith('wind_dir')]
		lat_lon_columns = ['Ac_Lat', 'Ac_Lon']
		altitude_columns = ['Ac_feet']
		
		df[wind_speed_columns] = self.scalers['wind_speed'].inverse_transform(df[wind_speed_columns])
		df[wind_dir_columns] = self.scalers['wind_dir'].inverse_transform(df[wind_dir_columns])
		df[lat_lon_columns] = self.scalers['lat_lon'].inverse_transform(df[lat_lon_columns])
		df[altitude_columns] = self.scalers['altitude'].inverse_transform(df[altitude_columns])
		
		return df
	
	def padding_features(self, df):
		total_flights = df['Ac_id'].nunique()
		max_time_step = df['Time_step'].max()
		num_wind_conditions = len([col for col in df.columns if 'wind_speed' in col or 'wind_dir' in col])
		X = np.zeros((total_flights, max_time_step, num_wind_conditions))
		y = np.zeros((total_flights, max_time_step, 4))  # For Ac_kts, Ac_Lat, Ac_Lon, Ac_feet
		
		ids = df['Ac_id'].unique()
		for i, id in enumerate(ids):
			id_data = df[df['Ac_id'] == id]
			wind_conditions = id_data[[col for col in df.columns if 'wind_speed' in col or 'wind_dir' in col]].values
			flight_info = id_data[['Ac_kts', 'Ac_Lat', 'Ac_Lon', 'Ac_feet']].values
			
			X[i, :len(wind_conditions), :] = wind_conditions
			y[i, :len(flight_info), :] = flight_info
		
		return X, y
	
	def split_train_test(self, X, y, test_size=0.2, val_size=0.25):
		X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=42)
		X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size, random_state=42)
		return X_train, X_val, X_test, y_train, y_val, y_test


# class FeatureEngineeringV0:
# 	def __init__(self, sequence_length=10):
# 		self.sequence_length = sequence_length
# 		self.scaler = MinMaxScaler()
#
# 	def normalize_features(self, df):
# 		# Select columns to normalize
# 		columns_to_normalize = df.columns[
# 			df.columns.str.startswith(('wind_speed', 'wind_dir', 'Ac_Lat', 'Ac_Lon', 'Ac_feet'))]
# 		df[columns_to_normalize] = self.scaler.fit_transform(df[columns_to_normalize])
# 		return df
#
# 	def create_sequences(self, df):
# 		sequences = []
# 		targets = []
# 		# Group by 'Ac_id' to handle each flight
# 		grouped = df.groupby('Ac_id')
# 		for ac_id, group in tqdm(grouped):
# 			for i in range(len(group) - self.sequence_length):
# 				sequence = group.iloc[i:i + self.sequence_length].drop(columns=['Route', 'Ac_id', 'Ac_code', 'Ac_type'])
# 				target = group.iloc[i + self.sequence_length][['Ac_Lat', 'Ac_Lon', 'Ac_feet']]
# 				sequences.append(sequence.values)
# 				targets.append(target.values)
# 		return np.array(sequences), np.array(targets)
#
# 	def process_data(self, csv_path):
# 		df = pd.read_csv(csv_path)
# 		df = self.normalize_features(df)
# 		sequences, targets = self.create_sequences(df)
# 		return sequences, targets
