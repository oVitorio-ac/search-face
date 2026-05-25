import unittest

import numpy as np

from app.domain.matcher import FaceMatcher
from app.domain.models import KnownPerson


class FaceMatcherTests(unittest.TestCase):
    def test_returns_all_matches_sorted_by_distance(self):
        matcher = FaceMatcher()
        query = np.array([0.0, 0.0])
        known = [
            KnownPerson(name="bob", encoding=np.array([0.2, 0.2])),
            KnownPerson(name="alice", encoding=np.array([0.1, 0.1])),
            KnownPerson(name="far", encoding=np.array([2.0, 2.0])),
        ]

        matches = matcher.all_matches(query, known, threshold=0.6)

        self.assertEqual(len(matches), 2)
        self.assertEqual(matches[0].name, "alice")
        self.assertEqual(matches[1].name, "bob")

    def test_returns_known_name_below_threshold(self):
        matcher = FaceMatcher()
        query = np.array([0.0, 0.0])
        known = [KnownPerson(name="alice", encoding=np.array([0.1, 0.1]))]

        decision = matcher.best_match(query, known, threshold=0.6)

        self.assertEqual(decision.name, "alice")
        self.assertLess(decision.distance, 0.6)

    def test_returns_unknown_above_threshold(self):
        matcher = FaceMatcher()
        query = np.array([0.0, 0.0])
        known = [KnownPerson(name="alice", encoding=np.array([2.0, 2.0]))]

        decision = matcher.best_match(query, known, threshold=0.6)

        self.assertEqual(decision.name, "unknown")
        self.assertGreater(decision.distance, 0.6)


if __name__ == "__main__":
    unittest.main()
