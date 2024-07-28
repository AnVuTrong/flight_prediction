import os
import pandas as pd
from tqdm import tqdm
from modules.file_handler import FileHandler

class DataPreprocessing:
    def __init__(self):
        self.file_handler = FileHandler()
        tqdm.pandas(ncols=30)

    def handle_init_files(self, list_path_of_files, combined_csv_path):
        try:
            if os.path.exists(combined_csv_path):
                print(f"The file '{combined_csv_path}' already exists. Skipping file creation.")
                return

            dataframes = []
            for file_path in tqdm(list_path_of_files):
                df = pd.read_excel(file_path)
                dataframes.append(df)
            combined_df = pd.concat(dataframes, ignore_index=True)
            combined_df.to_csv(combined_csv_path, index=False)
            print(f"All files combined successfully and saved to {combined_csv_path}")
        except Exception as e:
            print(f"An error occurred: {e}")
            