import unittest
from src.data_gen.modal_grammar import (
    Box, Const, Diamond, Expr, Iff, Implies, KripkeFrame, KripkeModel,
    Not, Or, And, Var, eval_modal_expr, to_symbolic
)


class TestModalGrammar(unittest.TestCase):
    def test_modal_evaluator_box(self):
        frame = KripkeFrame(worlds=["w0", "w1"], accessibility={"w0": ["w0", "w1"]})
        val1 = {"w0": {"P": True}, "w1": {"P": True}}
        m1 = KripkeModel(frame, val1)
        self.assertTrue(eval_modal_expr(Box(Var("P")), m1, "w0"))

        val2 = {"w0": {"P": True}, "w1": {"P": False}}
        m2 = KripkeModel(frame, val2)
        self.assertFalse(eval_modal_expr(Box(Var("P")), m2, "w0"))

    def test_modal_evaluator_diamond(self):
        frame = KripkeFrame(worlds=["w0", "w1"], accessibility={"w0": ["w0", "w1"]})
        val1 = {"w0": {"P": False}, "w1": {"P": True}}
        m1 = KripkeModel(frame, val1)
        self.assertTrue(eval_modal_expr(Diamond(Var("P")), m1, "w0"))

        val2 = {"w0": {"P": False}, "w1": {"P": False}}
        m2 = KripkeModel(frame, val2)
        self.assertFalse(eval_modal_expr(Diamond(Var("P")), m2, "w0"))

    def test_modal_evaluator_inaccessible_world(self):
        frame = KripkeFrame(worlds=["w0", "w1"], accessibility={"w0": ["w0"]})
        val = {"w0": {"P": True}, "w1": {"P": False}}
        m = KripkeModel(frame, val)
        self.assertTrue(eval_modal_expr(Box(Var("P")), m, "w0"))


if __name__ == "__main__":
    unittest.main()
