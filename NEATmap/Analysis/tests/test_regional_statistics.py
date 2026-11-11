import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from Analysis import Statistics as stats


class RegionalStatisticsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = self.directory.name

    def sample(self, name, filename, values):
        folder = os.path.join(self.root, name, 'whole_brain_cell_counts')
        os.makedirs(folder, exist_ok=True)
        pd.DataFrame(values).to_csv(os.path.join(folder, filename), index=False)

    def test_bh_bonferroni_empty_and_missing_values(self):
        np.testing.assert_allclose(stats.correct_p_values([0.01, 0.04, 0.03]), [0.03, 0.04, 0.04])
        np.testing.assert_allclose(stats.correct_p_values([0.01, 0.04, 0.03], 'B'), [0.03, 0.12, 0.09])
        adjusted = stats.correct_p_values([0.01, np.nan, 0.03])
        np.testing.assert_allclose(adjusted[[0, 2]], [0.02, 0.03])
        self.assertTrue(np.isnan(adjusted[1]))
        self.assertEqual(stats.correct_p_values([]).size, 0)
        with self.assertRaises(ValueError):
            stats.correct_p_values([0.2], 'unknown')

    def test_fractional_unequal_groups_and_region_alignment_match_scipy(self):
        filename = 'cell-counts-level6-sum.csv'
        experiment = np.array([[0.1, 0.3, 0.5], [1.2, 1.7, 1.9]])
        control = np.array([[0.7, 0.9], [1.1, 1.4]])
        for index in range(3):
            self.sample('e{}'.format(index), filename,
                        {'Region': ['A', 'B'], 'counts': experiment[:, index]})
        for index in range(2):
            self.sample('c{}'.format(index), filename,
                        {'Region': ['B', 'A'], 'counts': control[::-1, index]})
        stats.ttest_level6(self.root, ['e0', 'e1', 'e2'], ['c0', 'c1'], region_num=2)
        actual = pd.read_csv(os.path.join(self.root, 'Hypothesis_test_level6.csv'))
        expected = scipy_stats.ttest_ind(experiment, control, axis=1, equal_var=True).pvalue
        np.testing.assert_allclose(actual['p_val'], stats.correct_p_values(expected))
        self.assertEqual(list(actual['Region']), ['A', 'B'])
        self.assertEqual(list(actual.columns), ['Region', 'p_val', 'Confidence level'])

    def test_mismatched_and_duplicate_regions_fail_before_comparing(self):
        filename = 'cell-counts-level6-sum.csv'
        self.sample('e', filename, {'Region': ['A', 'B'], 'counts': [1, 2]})
        self.sample('c', filename, {'Region': ['A', 'C'], 'counts': [1, 2]})
        with self.assertRaisesRegex(ValueError, 'Region labels'):
            stats.ttest_level6(self.root, ['e'], ['c'], region_num=2)
        self.sample('c', filename, {'Region': ['A', 'A'], 'counts': [1, 2]})
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            stats.ttest_level6(self.root, ['e'], ['c'], region_num=2)

    def test_boundary_pvalues_and_nan_keep_all_rows(self):
        source = os.path.join(self.root, 'Hypothesis_test_level6_left_right.csv')
        pd.DataFrame({'Region': ['A', 'B', 'C', 'D'],
                      'left_Confidence level': [0.001, 0.01, 0.05, np.nan]}).to_csv(source, index=False)
        stats.symbol_left_right_ci(self.root, self.root)
        result = pd.read_csv(os.path.join(self.root, 'Symbol_level6_left.csv'), keep_default_na=False)
        self.assertEqual(list(result['Left Confidence level']), ['**', '*', 'ns', 'NA'])

    def test_hemisphere_aggregation_does_not_require_adjacent_rows(self):
        source = os.path.join(self.root, 'counts.csv')
        pd.DataFrame({'acronym': ['A', 'B', 'A', 'B'], 'count': [1.25, 3, 2.75, 5]}).to_csv(source, index=False)
        stats.level6_sum(source, self.root)
        stats.level6_left_right_brain(source, self.root)
        result = pd.read_csv(os.path.join(self.root, 'cell-counts-level6-sum.csv'))
        np.testing.assert_allclose(result['counts'], [4, 8])
        sides = pd.read_csv(os.path.join(self.root, 'cell-counts-level6-left_right.csv'))
        np.testing.assert_allclose(sides['right_counts'], [2.75, 5])

    def test_layer_statistics_remove_nan_false_is_defined(self):
        for name, value in [('e0', 0.2), ('e1', 0.4), ('c0', 0.6), ('c1', 0.9)]:
            self.sample(name, 'layer_density.csv', {'Region_name': ['A'], 'layer 1': [value]})
        stats.ttest_multi_layer(self.root, ['e0', 'e1'], ['c0', 'c1'],
                               'layer 1', 'L1_', 'layer_density', region_num=1, remove_nan=False)
        result = pd.read_csv(os.path.join(self.root, 'stats_layer', 'L1_layer_density_p_values.csv'))
        expected = scipy_stats.ttest_ind([0.2, 0.4], [0.6, 0.9], equal_var=True).pvalue
        np.testing.assert_allclose(result['P value'], [expected])

    def test_level2_supports_arbitrary_sample_count(self):
        frame = pd.DataFrame({'Region': np.arange(205), 'ctrl': np.ones(205), 'exp': np.full(205, 2.5)})
        frame.to_csv(os.path.join(self.root, 'level6_total_cell_count.csv'), index=False)
        configuration = {'con_data_name': ['ctrl'], 'es_data_name': ['exp'], 'sds_data_name': []}
        with patch.object(stats, 'Stats', configuration):
            stats.level2_count(self.root, self.root)
        result = pd.read_csv(os.path.join(self.root, 'level2_total_count.csv'))
        self.assertEqual(result.shape, (12, 3))
        self.assertEqual(result.loc[0, 'exp'], 42.5)

    def test_constant_normalization_is_finite(self):
        np.testing.assert_array_equal(stats.norm_data([3, 3]), [0, 0])
        np.testing.assert_allclose(stats.norm_data([2, 4, 6]), [0, 0.5, 1])

    def test_missing_counts_are_not_silently_summed_as_zero(self):
        source = os.path.join(self.root, 'counts.csv')
        pd.DataFrame({'acronym': ['A', 'A'], 'count': [np.nan, 5]}).to_csv(source, index=False)
        with self.assertRaisesRegex(ValueError, 'finite'):
            stats.level6_sum(source, self.root)

    def test_model_comparison_keeps_existing_marker_thresholds(self):
        for name in ['A', 'B']:
            pd.DataFrame({'Dice': [0.6, 0.8]}).to_csv(os.path.join(self.root, name + '_patch_results.csv'), index=False)
        with patch.object(stats.ss, 'ranksums', side_effect=[(0, 0.02), (0, 0.5)]):
            stats.DNN_result_ranksum(self.root, ['A', 'B'])
        with open(os.path.join(self.root, 'Dice_ranksums.txt')) as output:
            self.assertEqual(output.read().splitlines(), ['** | A,B', '* | B,A'])


if __name__ == '__main__':
    unittest.main()
