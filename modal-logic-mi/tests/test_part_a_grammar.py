import unittest
from src.data_gen.modal_grammar import (
    Box, Certainly, Const, Diamond, Expr, Iff, Implies, KripkeFrame, KripkeModel,
    Not, Or, Probably, Unlikely, And, Var, Xor, eval_modal_expr, to_symbolic
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

    def test_modal_evaluator_probably_and_certainly(self):
        frame = KripkeFrame(worlds=["w0", "w1", "w2"], accessibility={"w0": ["w0", "w1", "w2"]})
        # 2 of 3 worlds True (66.7% > 50%) -> probably is True, certainly is False
        val1 = {"w0": {"P": True}, "w1": {"P": True}, "w2": {"P": False}}
        m1 = KripkeModel(frame, val1)
        self.assertTrue(eval_modal_expr(Probably(Var("P")), m1, "w0"))
        self.assertFalse(eval_modal_expr(Certainly(Var("P")), m1, "w0"))

        # 3 of 3 worlds True (100%) -> both probably and certainly are True
        val2 = {"w0": {"P": True}, "w1": {"P": True}, "w2": {"P": True}}
        m2 = KripkeModel(frame, val2)
        self.assertTrue(eval_modal_expr(Probably(Var("P")), m2, "w0"))
        self.assertTrue(eval_modal_expr(Certainly(Var("P")), m2, "w0"))

        # 1 of 3 worlds True (33.3% < 50%) -> probably is False, unlikely is True
        val3 = {"w0": {"P": True}, "w1": {"P": False}, "w2": {"P": False}}
        m3 = KripkeModel(frame, val3)
        self.assertFalse(eval_modal_expr(Probably(Var("P")), m3, "w0"))
        self.assertTrue(eval_modal_expr(Unlikely(Var("P")), m3, "w0"))

    def test_boolean_connective_xor(self):
        frame = KripkeFrame(worlds=["w0"], accessibility={"w0": ["w0"]})
        m_tf = KripkeModel(frame, {"w0": {"P": True, "Q": False}})
        m_tt = KripkeModel(frame, {"w0": {"P": True, "Q": True}})
        self.assertTrue(eval_modal_expr(Xor(Var("P"), Var("Q")), m_tf, "w0"))
        self.assertFalse(eval_modal_expr(Xor(Var("P"), Var("Q")), m_tt, "w0"))


if __name__ == "__main__":
    unittest.main()
