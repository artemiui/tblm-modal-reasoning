import random
import unittest
from src.data_gen.circuit_pairs import (
    query_flip_pairs, modal_operator_flip_pairs, accessibility_flip_pairs,
    fact_flip_pairs, rule_location_swap_pairs
)


class TestCircuitPairs(unittest.TestCase):
    def test_query_flip_pairs(self):
        rng = random.Random(42)
        p = query_flip_pairs(rng)
        self.assertEqual(p.pair_type, "query_flip")
        self.assertNotEqual(p.clean_label, p.counterfactual_label)
        self.assertIn("Query:", p.clean_prompt)

    def test_modal_operator_flip_pairs(self):
        rng = random.Random(42)
        p = modal_operator_flip_pairs(rng)
        self.assertEqual(p.pair_type, "modal_operator_flip")
        self.assertIn("necessarily", p.clean_prompt)
        self.assertIn("possibly", p.counterfactual_prompt)

    def test_accessibility_flip_pairs(self):
        rng = random.Random(42)
        p = accessibility_flip_pairs(rng)
        self.assertEqual(p.pair_type, "accessibility_flip")
        self.assertNotEqual(p.clean_label, p.counterfactual_label)

    def test_fact_flip_pairs(self):
        rng = random.Random(42)
        p = fact_flip_pairs(rng)
        self.assertEqual(p.pair_type, "fact_flip")
        self.assertNotEqual(p.clean_label, p.counterfactual_label)

    def test_rule_location_swap_pairs(self):
        rng = random.Random(42)
        p = rule_location_swap_pairs(rng)
        self.assertEqual(p.pair_type, "rule_location_swap")
        self.assertEqual(p.clean_label, p.counterfactual_label)


if __name__ == "__main__":
    unittest.main()
