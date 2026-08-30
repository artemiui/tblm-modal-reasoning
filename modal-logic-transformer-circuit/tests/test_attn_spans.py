import unittest
from helpers.attn_analysis_helpers import _find_all_subseq


class TestAttnSpans(unittest.TestCase):
    def test_find_all_subseq(self):
        sntc = [1, 2, 3, 4, 5, 2, 3, 6]
        subseq = [2, 3]
        matches = _find_all_subseq(sntc, subseq)
        self.assertEqual(matches, [1, 5])


if __name__ == "__main__":
    unittest.main()
