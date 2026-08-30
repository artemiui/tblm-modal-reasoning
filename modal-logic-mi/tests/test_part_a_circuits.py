import unittest
from src.circuits.head_classify import classify_heads
from src.circuits.sufficiency_table import verify_sufficiency


class DummyCfg:
    n_layers = 12
    n_heads = 8


class DummyModel:
    cfg = DummyCfg()


class TestCircuits(unittest.TestCase):
    def test_family_classification_keys(self):
        model = DummyModel()
        candidate_heads = [
            {"layer": 1, "head": 0},
            {"layer": 4, "head": 1},
            {"layer": 8, "head": 2},
        ]
        pairs_by_type = {}
        families = classify_heads(model, candidate_heads, pairs_by_type)
        self.assertIn("MOH", families)
        self.assertIn("WAH", families)
        self.assertIn("CRH", families)
        self.assertIn("GMH", families)
        self.assertIn("QRLH", families)
        self.assertIn("QRMH", families)
        self.assertIn("FPH", families)
        self.assertIn("DH", families)


if __name__ == "__main__":
    unittest.main()
