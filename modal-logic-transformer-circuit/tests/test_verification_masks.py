import unittest
from helpers.verification import circuit_specification


class TestVerification(unittest.TestCase):
    def test_circuit_specification_keys(self):
        circuit, pos = circuit_specification("full")
        self.assertIn("MOH", circuit)
        self.assertIn("WAH", circuit)
        self.assertIn("CRH", circuit)
        self.assertIn("QRLH", circuit)
        self.assertIn("QRMH", circuit)
        self.assertIn("FPH", circuit)
        self.assertIn("DH", circuit)

        c_no_moh, _ = circuit_specification("no_moh")
        self.assertEqual(len(c_no_moh["MOH"]), 0)

        c_no_crh, _ = circuit_specification("no_crh")
        self.assertEqual(len(c_no_crh["CRH"]), 0)


if __name__ == "__main__":
    unittest.main()
