import unittest
from src.data_gen.mi_pairs import generate_modal_mi_dataset


class TestCalibration(unittest.TestCase):
    def test_dataset_generation_properties(self):
        samples = generate_modal_mi_dataset(n_samples=20, seed=42, prompt_order="facts_first")
        self.assertEqual(len(samples), 20)
        for s in samples:
            self.assertIn("clean_prompt_symbolic", s)
            self.assertIn("corrupted_prompt_symbolic", s)
            self.assertIn("label", s)
            self.assertIn("label_corrupted", s)
            self.assertNotEqual(s["label"], s["label_corrupted"])
            self.assertIn("clean_valuation", s)
            self.assertIn("clean_accessibility", s)


if __name__ == "__main__":
    unittest.main()
