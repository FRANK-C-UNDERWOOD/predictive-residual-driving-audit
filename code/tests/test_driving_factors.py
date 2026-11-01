"""Protect subgroup semantics: annotation onset, clipping and gap-aware motion."""
import json, unittest
from pathlib import Path
import numpy as np
from riskaware.driving_factors import annotation_factors, group_identities

CFG = json.loads((Path(__file__).resolve().parents[1] / 'configs/driving_factors_v01.json').read_text())
DT = [('t', 'i8'), ('x', 'f4'), ('y', 'f4'), ('w', 'f4'), ('h', 'f4'), ('class_id', 'u1'), ('track_id', 'i8')]

class FactorSemantics(unittest.TestCase):
    def rows(self, values):
        return np.array(values, dtype=DT)
    def test_initial_tracks_are_censored_and_appearance_uses_full_grid(self):
        grid = [0, 50000, 100000, 150000, 200000]
        rows = self.rows([(t, 0, 0, 10, 10, 2, ident) for ident, ts in [(1, grid), (2, grid[1:])] for t in ts])
        factors, _ = annotation_factors(grid, rows, CFG)
        self.assertEqual(factors[200000, 1]['stage'], 'left_censored')
        self.assertEqual(factors[150000, 2]['stage'], 'early')
        self.assertEqual(factors[200000, 2]['stage'], 'established')
        self.assertEqual(factors[200000, 2]['appearance_number'], 4)
    def test_motion_uses_same_identity_and_adjacent_native_query(self):
        grid = [0, 50000, 100000, 150000]
        rows = self.rows([(0, 0, 0, 10, 10, 2, 1), (50000, 4, 0, 10, 10, 2, 1),
                          (50000, 100, 0, 10, 10, 2, 2), (150000, 8, 0, 10, 10, 2, 1)])
        factors, _ = annotation_factors(grid, rows, CFG)
        self.assertAlmostEqual(factors[50000, 1]['motion_diagonal_per_second'], .1)
        self.assertEqual(factors[50000, 1]['motion'], 'motion_high')
        self.assertEqual(factors[50000, 2]['motion'], 'motion_unknown')
        self.assertEqual(factors[150000, 1]['motion'], 'motion_unknown')
    def test_native_time_gap_is_not_bridged(self):
        grid = [0, 50000, 100000, 500000]
        rows = self.rows([(100000, 0, 0, 10, 10, 2, 1), (500000, 20, 0, 10, 10, 2, 1)])
        factors, _ = annotation_factors(grid, rows, CFG)
        self.assertEqual(factors[500000, 1]['motion'], 'motion_unknown')
    def test_clip_area_and_class_exclusion(self):
        rows = self.rows([(0, -10, 0, 20, 20, 2, 1), (0, 0, 0, 20, 20, 1, 2)])
        factors, _ = annotation_factors([0, 50000], rows, CFG)
        self.assertEqual(set(factors), {(0, 1)})
        self.assertEqual(factors[0, 1]['box'], [0., 0., 10., 20.])
        groups = group_identities(list(factors.values()), CFG['groups'])
        self.assertEqual(groups['small'], {1})
        self.assertEqual(groups['early'], set())
    def test_duplicate_identity_is_rejected(self):
        rows = self.rows([(0, 0, 0, 10, 10, 2, 1)] * 2)
        with self.assertRaises(ValueError):
            annotation_factors([0, 50000], rows, CFG)

if __name__ == '__main__':
    unittest.main()
