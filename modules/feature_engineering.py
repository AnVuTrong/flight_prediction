import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


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
		# Normalizing wind speed and direction
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
	
	def process_data(self, csv_path):
		df = pd.read_csv(csv_path)
		df = self.normalize_features(df)
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


# Example usage
if __name__ == "__main__":
	fe = FeatureEngineeringV1()
	processed_df = fe.process_data('data/csv/raw.csv')
	print(processed_df.head())
	
	decoded_df = fe.decode_features(processed_df.copy())
	print(decoded_df.head())
