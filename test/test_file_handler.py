import unittest
import os
import pandas as pd
from modules.file_handler import FileHandler

class TestFileHandler(unittest.TestCase):
    def setUp(self):
        self.excel_path = "test_example.xlsx"
        self.csv_path = "test_example.csv"
        df = pd.DataFrame({
            "A": [1, 2, 3],
            "B": [4, 5, 6]
        })
        df.to_excel(self.excel_path, index=False)

    def tearDown(self):
        if os.path.exists(self.excel_path):
            os.remove(self.excel_path)
        if os.path.exists(self.csv_path):
            os.remove(self.csv_path)

    def test_convert_excel_to_csv(self):
        file_handler = FileHandler()
        file_handler.convert_excel_to_csv(self.excel_path, self.csv_path)
        self.assertTrue(os.path.exists(self.csv_path))

        df_csv = pd.read_csv(self.csv_path)
        self.assertEqual(df_csv.shape, (3, 2))
        self.assertEqual(list(df_csv.columns), ["A", "B"])

if __name__ == '__main__':
    unittest.main()
