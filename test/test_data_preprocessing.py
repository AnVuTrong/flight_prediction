import unittest
import os
import pandas as pd
from modules.data_preprocessing import DataPreprocessing

class TestDataPreprocessing(unittest.TestCase):
    def setUp(self):
        self.files = ["test_file1.xlsx", "test_file2.xlsx"]
        for file in self.files:
            df = pd.DataFrame({
                "A": [1, 2, 3],
                "B": [4, 5, 6]
            })
            df.to_excel(file, index=False)
        self.combined_csv_path = "combined_test_output.csv"

    def tearDown(self):
        for file in self.files:
            if os.path.exists(file):
                os.remove(file)
        if os.path.exists(self.combined_csv_path):
            os.remove(self.combined_csv_path)

    def test_handle_init_files(self):
        data_preprocessing = DataPreprocessing()
        data_preprocessing.handle_init_files(self.files, self.combined_csv_path)
        self.assertTrue(os.path.exists(self.combined_csv_path))

        df_combined = pd.read_csv(self.combined_csv_path)
        self.assertEqual(df_combined.shape, (6, 2))
        self.assertEqual(list(df_combined.columns), ["A", "B"])

if __name__ == '__main__':
    unittest.main()
