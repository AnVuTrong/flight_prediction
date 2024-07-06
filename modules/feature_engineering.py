import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset


class FeatureEngineeringV1:
	def __init__(self):
		self.scalers = {
			'wind_speed': StandardScaler(),
			'wind_dir'  : StandardScaler(),
			'lat_lon'   : StandardScaler(),
			'altitude'  : StandardScaler(),
			'speed'     : StandardScaler(),
			'time_step' : StandardScaler()
		}
		self.encoders = {
			'ac_type': OneHotEncoder(sparse_output=False),
			'phase'  : OneHotEncoder(sparse_output=False),
		}
		self.fitted = False
	
	def common_preprocessing(self, df):
		df = df.dropna()
		df = df.drop_duplicates().reset_index(drop=True)
		return df
	
	def normalize_features(self, df):
		wind_speed_columns = df.columns[df.columns.str.startswith('wind_speed')]
		wind_dir_columns = df.columns[df.columns.str.startswith('wind_dir')]
		lat_lon_columns = ['Ac_Lat', 'Ac_Lon']
		altitude_columns = ['Ac_feet']
		speed_columns = ['Ac_kts']
		time_step_column = ['Time_step']
		
		df[wind_speed_columns] = self.scalers['wind_speed'].fit_transform(df[wind_speed_columns])
		df[wind_dir_columns] = self.scalers['wind_dir'].fit_transform(df[wind_dir_columns])
		df[lat_lon_columns] = self.scalers['lat_lon'].fit_transform(df[lat_lon_columns])
		df[altitude_columns] = self.scalers['altitude'].fit_transform(df[altitude_columns])
		df[speed_columns] = self.scalers['speed'].fit_transform(df[speed_columns])
		df[time_step_column] = self.scalers['time_step'].fit_transform(df[time_step_column])
		
		ac_type_encoded = self.encoders['ac_type'].fit_transform(df[['Ac_type']])
		phase_encoded = self.encoders['phase'].fit_transform(df[['Phase']])
		
		ac_type_df = pd.DataFrame(
			ac_type_encoded,
			columns=self.encoders['ac_type'].get_feature_names_out(['Ac_type'])
		).reset_index(drop=True)
		
		phase_df = pd.DataFrame(
			phase_encoded,
			columns=self.encoders['phase'].get_feature_names_out(['Phase'])
		).reset_index(drop=True)
		
		df = pd.concat(
			[
				df.reset_index(drop=True),
				ac_type_df,
				phase_df,
			], axis=1
		)
		df = df.drop(['Ac_type', 'Phase'], axis=1)
		
		self.fitted = True
		return df
	
	def remove_unreasonable_time(self, df, max_threshold=10000, min_flight_time=4000):
		cleaned_df = df[df['Time_step'] < max_threshold]
		flights_to_remove = cleaned_df.groupby('Ac_id')['Time_step'].max() < min_flight_time
		id_to_remove = flights_to_remove[flights_to_remove].index
		cleaned_df = cleaned_df[~cleaned_df['Ac_id'].isin(id_to_remove)]
		cleaned_df.reset_index(drop=True)
		return cleaned_df
	
	def process_data(self, csv_path):
		df = pd.read_csv(csv_path)
		df = self.common_preprocessing(df)
		df = self.remove_unreasonable_time(df)
		df = self.normalize_features(df)
		
		return df
	
	def decode_features(self, df):
		if not self.fitted:
			raise Exception("Scalers not fitted. Call process_data first.")
		
		lat_lon_columns = ['Ac_Lat', 'Ac_Lon']
		altitude_columns = ['Ac_feet']
		speed_columns = ['Ac_kts']
		time_step_column = ['Time_step']
		
		
		df[lat_lon_columns] = self.scalers['lat_lon'].inverse_transform(df[lat_lon_columns])
		df[altitude_columns] = self.scalers['altitude'].inverse_transform(df[altitude_columns])
		df[speed_columns] = self.scalers['speed'].inverse_transform(df[speed_columns])
		df[time_step_column] = self.scalers['time_step'].inverse_transform(df[time_step_column])
		
		return df
	
	def padding_features(self, df):
		total_flights = df['Ac_id'].nunique()
		
		wind_condition_columns = [col for col in df.columns if 'wind_speed' in col or 'wind_dir' in col]
		ac_type_columns = [col for col in df.columns if col.startswith('Ac_type_')]
		phase_columns = [col for col in df.columns if col.startswith('Phase_')]
		other_features_columns = ac_type_columns + phase_columns + ['Time_step']
		
		num_features = len(wind_condition_columns) + len(other_features_columns)
		
		X_list = []
		y_list = []
		
		ids = df['Ac_id'].unique()
		for id in ids:
			id_data = df[df['Ac_id'] == id]
			wind_conditions = id_data[wind_condition_columns].values
			other_features = id_data[other_features_columns].values
			flight_info = id_data[['Ac_kts', 'Ac_Lat', 'Ac_Lon', 'Ac_feet']].values
			
			features = np.concatenate((wind_conditions, other_features), axis=1)
			
			X_list.append(features)
			y_list.append(flight_info)
		
		return X_list, y_list
	
	def split_train_test(self, X, y, test_size=0.2, val_size=0.25):
		X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=42)
		X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size, random_state=42)
		return X_train, X_val, X_test, y_train, y_val, y_test
	
	def make_tensor_dataset(self, X_train, X_val, X_test, y_train, y_val, y_test):
		X_train = torch.FloatTensor(X_train)
		X_val = torch.FloatTensor(X_val)
		X_test = torch.FloatFloatTensor(X_test)
		y_train = torch.FloatFloatTensor(y_train)
		y_val = torch.FloatFloatTensor(y_val)
		y_test = torch.FloatFloatTensor(y_test)
		
		train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
		val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
		test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
		return train_dataset, val_dataset, test_dataset
