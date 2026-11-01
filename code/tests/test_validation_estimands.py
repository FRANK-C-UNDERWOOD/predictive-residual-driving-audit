import unittest
from riskaware.validation_estimands import motion_interaction, weighted_cell_interaction, recording_summary

class ValidationEstimands(unittest.TestCase):
    def test_recall_interaction_uses_gt_not_dense_denominator(self):
        self.assertEqual(motion_interaction(10, 20, 100, 40, 40, 200), -10.)
    def test_missing_motion_population_is_undefined(self):
        self.assertIsNone(motion_interaction(0, 0, 0, 1, 2, 10))
    def test_missing_cell_is_not_dropped_and_weights_not_renormalized(self):
        cells = {'a': dict(low_a=1, low_b=2, low_gt=10, high_a=5, high_b=5, high_gt=10)}
        self.assertIsNone(weighted_cell_interaction(cells, {'a': .5, 'b': .5}))
        self.assertEqual(weighted_cell_interaction(cells, {'a': 1., 'b': 0.}), -10.)
    def test_recordings_are_equal_units_and_missing_not_zero(self):
        result = recording_summary([-10., 2., None])
        self.assertEqual(result['mean'], -4.)
        self.assertEqual(result['defined'], 2)
        self.assertFalse(result['all_planned_defined'])

if __name__ == '__main__':
    unittest.main()
