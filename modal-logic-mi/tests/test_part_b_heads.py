import unittest
from src.staged.specialized_heads import classify_head_roles, HEAD_ROLES


class DummyCfg:
    n_layers = 16
    n_heads = 8


class DummyModel:
    cfg = DummyCfg()


class TestSpecializedHeads(unittest.TestCase):
    def test_head_roles_classification(self):
        model = DummyModel()
        top_heads = [(0, 0), (2, 1), (5, 2), (10, 3), (14, 4)]
        role_map = classify_head_roles(model, top_heads, [])
        self.assertEqual(len(role_map), len(top_heads))
        for role in role_map.values():
            self.assertIn(role, HEAD_ROLES)
        self.assertIn("accessibility_filtering", role_map.values())


if __name__ == "__main__":
    unittest.main()
