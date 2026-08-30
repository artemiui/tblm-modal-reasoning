import random
import unittest
from src.data_gen.modal_grammar import Box, Diamond, Implies, KripkeFrame, KripkeModel, Var, eval_modal_expr
from src.data_gen.corruptions import (
    find_label_flipping_fact_corruption,
    find_label_flipping_accessibility_corruption
)


class TestPartBCorruptions(unittest.TestCase):
    def test_fact_corruption(self):
        rng = random.Random(42)
        frame = KripkeFrame(worlds=["w0", "w1"], accessibility={"w0": ["w0", "w1"]})
        val = {"w0": {"P": True, "Q": True}, "w1": {"P": True, "Q": True}}
        model = KripkeModel(frame, val)
        expr = Box(Var("P"))

        corrupted_model, meta = find_label_flipping_fact_corruption(model, expr, "w0", rng)
        clean_val = eval_modal_expr(expr, model, "w0")
        corrupt_val = eval_modal_expr(expr, corrupted_model, "w0")
        self.assertNotEqual(clean_val, corrupt_val)
        self.assertEqual(meta["corruption_type"], "fact_corruption")

    def test_accessibility_corruption(self):
        rng = random.Random(42)
        frame = KripkeFrame(worlds=["w0", "w1"], accessibility={"w0": ["w0", "w1"]})
        val = {"w0": {"P": False}, "w1": {"P": True}}
        model = KripkeModel(frame, val)
        expr = Diamond(Var("P"))

        corrupted_model, meta = find_label_flipping_accessibility_corruption(model, expr, "w0", rng)
        clean_val = eval_modal_expr(expr, model, "w0")
        corrupt_val = eval_modal_expr(expr, corrupted_model, "w0")
        self.assertNotEqual(clean_val, corrupt_val)
        self.assertEqual(meta["corruption_type"], "accessibility_corruption")


if __name__ == "__main__":
    unittest.main()
