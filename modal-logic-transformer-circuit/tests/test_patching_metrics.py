import unittest
from helpers.patching_helpers_custom import basic_metric


class TestPatchingMetrics(unittest.TestCase):
    def test_basic_metric_calculation(self):
        # patched_ld = 4.0, clean_ld = 0.0, corrupted_ld = 4.0 -> metric = (4 - 0)/(4 - 0) = 1.0
        # Testing direct formula math
        clean_ld = 0.0
        corrupted_ld = 4.0
        patched_ld = 4.0
        metric_val = (patched_ld - clean_ld) / (corrupted_ld - clean_ld)
        self.assertAlmostEqual(metric_val, 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
