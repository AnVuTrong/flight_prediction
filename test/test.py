import unittest
from tqdm import tqdm
import os
import sys

# Add the parent directory to the Python path to recognize the modules package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Discover and load all test modules
test_loader = unittest.TestLoader()
test_dir = os.path.dirname(os.path.abspath(__file__))
suite = test_loader.discover(start_dir=test_dir, pattern='test_*.py')

# Create a custom runner to include a progress bar
class TQDMTestRunner(unittest.TextTestRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self, test):
        test_result = self._makeResult()
        tqdm_bar = tqdm(total=test.countTestCases(), ncols=70)

        def tqdm_progress(test):
            tqdm_bar.update()

        test_result.startTest = tqdm_progress
        test_result.startTestRun = lambda: tqdm_bar
        test_result.stopTestRun = tqdm_bar.close
        test_result.stopTest = lambda x: None

        return super().run(test)

if __name__ == '__main__':
    runner = TQDMTestRunner(verbosity=2)
    runner.run(suite)
