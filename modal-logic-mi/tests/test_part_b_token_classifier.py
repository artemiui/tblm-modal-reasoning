import unittest
from src.staged.info_transmission import classify_modal_tokens


class TestPartBTokenClassifier(unittest.TestCase):
    def test_modal_token_classifier(self):
        str_tokens = ["In", "w0", ":", "P", "is", "True", ".", "Accessibility", ":", "box", "(", "P", ")", "is", "True", "or", "False"]
        cats = classify_modal_tokens(str_tokens)
        self.assertEqual(len(cats), len(str_tokens))
        self.assertIn("accessibility_boundary", cats)
        self.assertIn("facts_value", cats)
        self.assertIn("operator", cats)
        self.assertIn("query_token", cats)


if __name__ == "__main__":
    unittest.main()
