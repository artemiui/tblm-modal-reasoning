import random
import unittest
from src.data_gen.circuit_pairs import (
    query_flip_pairs, modal_operator_flip_pairs, modal_proposition_flip_pairs,
    accessibility_flip_pairs, fact_flip_pairs, rule_location_swap_pairs,
    connective_flip_pairs, generate_all_circuit_pairs
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

    def test_modal_proposition_flip_pairs(self):
        rng = random.Random(42)
        p = modal_proposition_flip_pairs(rng)
        self.assertEqual(p.pair_type, "modal_proposition_flip")
        self.assertNotEqual(p.clean_label, p.counterfactual_label)
        self.assertIn("necessarily P implies possibly P", p.clean_prompt)

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

    def test_connective_flip_pairs(self):
        rng = random.Random(42)
        p = connective_flip_pairs(rng)
        self.assertEqual(p.pair_type, "connective_flip")
        self.assertTrue(p.clean_label)
        self.assertFalse(p.counterfactual_label)
        self.assertIn("or", p.clean_prompt)
        self.assertIn("and", p.counterfactual_prompt)

    def test_generate_all_circuit_pairs_count(self):
        pairs = generate_all_circuit_pairs(n_per_type=5, seed=42)
        # 6 pair types * 5 = 30
        self.assertEqual(len(pairs), 30)


if __name__ == "__main__":
    unittest.main()
