import pandas as pd


class FileHandler:
	def __init__(self):
		pass
	
	def convert_excel_to_csv(self, excel_path, csv_path):
		"""
		Converts an Excel file to a CSV file.

		:param excel_path: str, the path to the Excel file.
		:param csv_path: str, the path to save the converted CSV file.
		"""
		try:
			excel_data = pd.read_excel(excel_path)
			excel_data.to_csv(csv_path, index=False)
			print(f"File converted successfully and saved to {csv_path}")
		except Exception as e:
			print(f"An error occurred: {e}")
