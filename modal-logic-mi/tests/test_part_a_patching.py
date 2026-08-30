import unittest
from src.patching.metrics import compute_calibrated_ld


class TestPartAPatching(unittest.TestCase):
    def test_metrics_calibrated_ld(self):
        cld = compute_calibrated_ld(clean_ld=4.0, corrupted_ld=-2.0, patched_ld=4.0)
        self.assertAlmostEqual(cld, 1.0, places=5)

        cld_zero = compute_calibrated_ld(clean_ld=4.0, corrupted_ld=-2.0, patched_ld=-2.0)
        self.assertAlmostEqual(cld_zero, 0.0, places=5)

        cld_half = compute_calibrated_ld(clean_ld=4.0, corrupted_ld=-2.0, patched_ld=1.0)
        self.assertAlmostEqual(cld_half, 0.5, places=5)


if __name__ == "__main__":
    unittest.main()
