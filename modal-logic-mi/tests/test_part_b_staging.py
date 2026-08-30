import unittest
from src.data_gen.formatters import build_4region_char_spans, build_modal_mi_prompt
from src.data_gen.modal_grammar import KripkeFrame


class TestPartBStaging(unittest.TestCase):
    def test_4region_char_spans(self):
        frame = KripkeFrame(worlds=["w0", "w1"], accessibility={"w0": ["w0", "w1"]})
        valuation = {"w0": {"P": True}, "w1": {"P": False}}
        expr_text = "box(P)"

        prompt = build_modal_mi_prompt(
            valuation=valuation,
            frame=frame,
            expr_text=expr_text,
            base_world="w0",
            prompt_order="facts_first",
        )

        spans = build_4region_char_spans(
            prompt=prompt,
            valuation=valuation,
            frame=frame,
            expr_text=expr_text,
            base_world="w0",
            prompt_order="facts_first",
        )

        self.assertTrue("facts_region" in spans and len(spans["facts_region"]) > 0)
        self.assertTrue("accessibility_region" in spans and len(spans["accessibility_region"]) > 0)
        self.assertTrue("expression_region" in spans and len(spans["expression_region"]) > 0)
        self.assertTrue("query_region" in spans and len(spans["query_region"]) > 0)


if __name__ == "__main__":
    unittest.main()
