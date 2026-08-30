import random
import unittest
from src.data_gen.mi_pairs import MODAL_RULE_CATEGORIES
from src.data_gen.modal_grammar import eval_modal_expr


class TestPartBGrammar(unittest.TestCase):
    def test_all_10_modal_rule_categories(self):
        rng = random.Random(42)
        self.assertEqual(len(MODAL_RULE_CATEGORIES), 10)
        for cat in MODAL_RULE_CATEGORIES:
            expr, model, base_world = cat.build_fn(rng)
            label = eval_modal_expr(expr, model, base_world)
            self.assertIsInstance(label, bool)


if __name__ == "__main__":
    unittest.main()
