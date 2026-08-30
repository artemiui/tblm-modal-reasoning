import sys
import unittest
from pathlib import Path

# Add modal-logic-mi to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(Path(__file__).parent), pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        sys.exit(1)
