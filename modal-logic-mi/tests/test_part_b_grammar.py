import random
import unittest
from src.data_gen.mi_pairs import MODAL_RULE_CATEGORIES
from src.data_gen.modal_grammar import eval_modal_expr


class TestPartBGrammar(unittest.TestCase):
    def test_all_modal_rule_categories(self):
        rng = random.Random(42)
        self.assertEqual(len(MODAL_RULE_CATEGORIES), 11)
        cat_names = [c.name for c in MODAL_RULE_CATEGORIES]
        self.assertNotIn("cross_world_composition", cat_names)
        self.assertIn("b_axiom", cat_names)
        self.assertIn("d_axiom", cat_names)
        self.assertIn("four_axiom", cat_names)
        self.assertIn("five_axiom", cat_names)

        for cat in MODAL_RULE_CATEGORIES:
            expr, model, base_world = cat.build_fn(rng)
            label = eval_modal_expr(expr, model, base_world)
            self.assertIsInstance(label, bool)


if __name__ == "__main__":
    unittest.main()
