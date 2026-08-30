import unittest
from helpers.modal_problem_generation import (
    sample_modal_chain,
    generate_modal_sample_tokens,
    convert_to_english,
    generate_cot_question_query_based,
    generate_cot_question_operator_based,
    generate_cot_question_accessibility_based,
    generate_cot_question_graded_operator_based,
    generate_cot_question_connective_based,
)


class TestModalProblemGeneration(unittest.TestCase):
    def test_sample_modal_chain(self):
        s_m, s_l = sample_modal_chain(depth=2, operator="BOX", accessible_worlds=["w0", "w1"], truths=["TRUE", "TRUE", "TRUE"])
        self.assertEqual(s_m["query_type"], "modal")
        self.assertEqual(s_l["query_type"], "linear")
        self.assertEqual(s_m["answer"], "TRUE")

    def test_sample_modal_chain_graded(self):
        # 1 True out of 2 worlds -> Probably is False (50% not >50%), Certainly is False
        s_p, _ = sample_modal_chain(depth=2, operator="PROBABLY", accessible_worlds=["w0", "w1"], truths=["TRUE", "FALSE", "FALSE"])
        self.assertEqual(s_p["answer"], "UNDETERMINED")

        # 2 True out of 2 worlds -> Probably is True, Certainly is True
        s_c, _ = sample_modal_chain(depth=2, operator="CERTAINLY", accessible_worlds=["w0", "w1"], truths=["TRUE", "TRUE", "FALSE"])
        self.assertEqual(s_c["answer"], "TRUE")

    def test_token_and_english_conversion(self):
        s_m, _ = sample_modal_chain(depth=2, operator="BOX")
        ctx, ans, seq_len = generate_modal_sample_tokens(s_m)
        self.assertGreater(seq_len, 0)
        _, ctx_eng = convert_to_english(ctx)
        self.assertIn("ACCESS_START", ctx_eng)
        self.assertIn("Rules:", ctx_eng)
        self.assertIn("Facts:", ctx_eng)

    def test_query_flip(self):
        p_m, g_m, p_l, g_l, i_m, i_l = generate_cot_question_query_based()
        self.assertIn("necessarily", p_m)
        self.assertNotEqual(p_m, p_l)

    def test_operator_flip(self):
        p_b, g_b, p_d, g_d, _, _ = generate_cot_question_operator_based()
        self.assertIn("necessarily", p_b)
        self.assertIn("possibly", p_d)

    def test_accessibility_flip(self):
        p_2, g_2, p_1, g_1, _, _ = generate_cot_question_accessibility_based()
        self.assertIn("w0 w1", p_2)
        self.assertIn("w0", p_1)

    def test_graded_operator_flip(self):
        p_p, g_p, p_c, g_c, _, _ = generate_cot_question_graded_operator_based()
        self.assertIn("probably", p_p)
        self.assertIn("certainly", p_c)

    def test_connective_flip(self):
        p_o, g_o, p_a, g_a, _, _ = generate_cot_question_connective_based()
        self.assertNotEqual(p_o, p_a)


if __name__ == "__main__":
    unittest.main()
