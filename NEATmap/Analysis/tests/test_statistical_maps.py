"""Regression cases for regional statistics and density-map boundary rules."""

import os
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from scipy import stats
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Analysis import cvandzscore as cv
from Analysis import log2Change as lc
from Analysis import p_values as pv


class StatisticalMapsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = self.temp.name

    def counts(self, path, values, regions=('A', 'B', 'C')):
        parent = os.path.dirname(path)
        os.makedirs(parent, exist_ok=True)
        pd.DataFrame({'Region': regions, 'counts': values}).to_csv(path, index=False)

    def test_dynamic_cv_and_zscore_align_reordered_regions(self):
        for treatment in ('es', 'sds'):
            self.counts(os.path.join(self.root, 'cell-counts-level6-sum_{}_a.csv'.format(treatment)), [3, 8, 0])
            self.counts(os.path.join(self.root, 'cell-counts-level6-sum_{}_b.csv'.format(treatment)), [0, 12, 5], ('C', 'B', 'A'))
        self.counts(os.path.join(self.root, 'cell-counts-level6-sum_con_a.csv'), [1, 4, 0])
        self.counts(os.path.join(self.root, 'cell-counts-level6-sum_con_b.csv'), [0, 8, 3], ('C', 'B', 'A'))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            cv.coefficient_of_variation(self.root, self.root, ['a', 'b'])
            cv.z_score(self.root, self.root, ['a', 'b'])
        result = pd.read_csv(os.path.join(self.root, 'O_C_level6_CV.csv'))
        self.assertEqual(result['Region'].tolist(), ['A', 'B', 'C'])
        np.testing.assert_allclose(result['SDS_CV'], [0.25, 0.2, np.nan], equal_nan=True)
        scores = pd.read_csv(os.path.join(self.root, 'I_C_level6_z_score.csv'))
        self.assertEqual(list(scores.columns), ['Region', 'CONa', 'CONb', 'SDSa', 'SDSb'])
        np.testing.assert_allclose(scores.iloc[:, 1:].values,
                                   [[-1, 1, 1, 3], [-1, 1, 1, 3], [np.nan] * 4], equal_nan=True)

    def test_duplicate_regions_fail_explicitly(self):
        path = os.path.join(self.root, 'duplicate.csv')
        self.counts(path, [1, 2, 3], ('A', 'A', 'C'))
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            cv._read_counts(path)

    def test_fold_change_keeps_floats_and_unequal_group_sizes(self):
        treatment = np.array([[1.5, 2.5, 3.5], [8.0, 10.0, 12.0], [0.0, 0.0, 0.0]])
        control = np.array([[0.5, 1.5], [4.0, 8.0], [0.0, 0.0]])
        for i, values in enumerate(treatment.T):
            self.counts(os.path.join(self.root, 't{}'.format(i), 'whole_brain_cell_counts', 'cell-counts-level6-sum.csv'), values)
        for i, values in enumerate(control.T):
            self.counts(os.path.join(self.root, 'c{}'.format(i), 'whole_brain_cell_counts', 'cell-counts-level6-sum.csv'), values[::-1], ('C', 'B', 'A'))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            lc.log2change(self.root, ['t0', 't1', 't2'], ['c0', 'c1'], region_num=3)
            expected_p = stats.ttest_ind(treatment, control, axis=1, equal_var=True)[1]
        result = pd.read_csv(os.path.join(self.root, 'O_I_difference_level6.csv'))
        np.testing.assert_allclose(result['log2FoldChange'], [np.log2(2.5), np.log2(10.0 / 6.0), np.nan], equal_nan=True)
        np.testing.assert_allclose(result['pValue'], expected_p, equal_nan=True)
        np.testing.assert_allclose(result['qValue'], lc.correct_p_values(expected_p), equal_nan=True)

    def test_multiple_test_correction_does_not_spread_nan(self):
        pvalues = np.array([0.01, 0.04, np.nan, 0.03])
        np.testing.assert_allclose(lc.correct_p_values(pvalues), [0.03, 0.04, np.nan, 0.04], equal_nan=True)
        np.testing.assert_allclose(lc.correct_p_values(pvalues, 'B'), [0.03, 0.12, np.nan, 0.09], equal_nan=True)
        self.assertEqual(lc.correct_p_values([]).size, 0)
        with self.assertRaises(ValueError):
            lc.correct_p_values([0.1], 'unknown')
        with self.assertRaises(ValueError):
            lc.correct_p_values([1.1])

    def test_density_matches_reference_with_duplicates_and_outside_points(self):
        coordinates = np.array([[0, 0], [0, 0], [-1, 0], [4, 4], [5, 4], [2, 2]])
        offsets = pv.search_indices_sphere((2, 2))
        expected = np.zeros((5, 5), dtype=np.uint32)
        for point in coordinates:
            for offset in offsets:
                target = point + offset
                if np.all((target >= 0) & (target < expected.shape)):
                    expected[tuple(target)] += 1
        actual = np.zeros_like(expected)
        pv._accumulate_density(actual, coordinates, offsets, chunk_size=2)
        np.testing.assert_array_equal(actual, expected)

    def test_high_density_is_not_wrapped_to_uint16(self):
        sink = np.array([[65536, 3], [2, 0]], dtype=np.uint32)
        path = os.path.join(self.root, 'density.tif')
        with self.assertWarns(RuntimeWarning):
            pv._save_density(path, sink, (1, 0))
        result = tifffile.imread(path)
        self.assertEqual(result.dtype, np.dtype('uint32'))
        np.testing.assert_array_equal(result, sink.T)

    def test_density_internal_overflow_fails_instead_of_wrapping(self):
        sink = np.full((1, 1), np.iinfo(np.uint32).max, dtype=np.uint32)
        with self.assertRaises(OverflowError):
            pv._accumulate_density(sink, np.array([[0, 0]]), np.array([[0, 0]]))

    def test_no_pvalue_cutoff_returns_an_independent_copy(self):
        values = np.array([0.01, 0.7])
        actual = pv.cutoff_p_values(values, None)
        actual[0] = 0.9
        self.assertEqual(values[0], 0.01)
        np.testing.assert_allclose(pv.cutoff_p_values(values), [0.01, 0.05])

    def test_no_cutoff_allows_rgb_without_unused_trend_vectors(self):
        result = pv.color_p_values(np.array([0.01, 0.05]), np.array([1, -1]),
                                    positive=[1, 0, 0], negative=[0, 1, 0])
        np.testing.assert_allclose(result, [[0.04, 0, 0], [0, 0, 0]])

    def test_default_ttest_without_cutoff_matches_scipy(self):
        treatment = np.array([[2, 0], [4, 0], [5, 0]], dtype=float)
        control = np.array([[1, 0], [2, 0]], dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            result, signs = pv.t_test_voxelization(treatment, control, signed=True)
            expected = stats.ttest_ind(treatment, control, axis=0, equal_var=True)
        np.testing.assert_allclose(result, [expected[1][0], 1.0])
        np.testing.assert_array_equal(signs, [np.sign(expected[0][0]), 0])

    def test_volume_color_channels_are_the_last_axis(self):
        values = np.linspace(0.0, 0.05, 24).reshape(2, 3, 4)
        captured = {}
        def save(path, result):
            captured[os.path.basename(path)] = result.copy()
        with mock.patch.object(pv, 't_test_voxelization', return_value=(values, np.ones_like(values))):
            with mock.patch.object(pv.tifffile, 'imwrite', side_effect=save):
                pv.generate_p_mapping([], [], self.root, 'example')
        np.testing.assert_allclose(captured['example_pos.tif'], (0.05 - values).astype('float32'))
        np.testing.assert_array_equal(captured['example_neg.tif'], np.zeros((2, 3, 4)))

    def test_thumbnail_listing_accepts_nonconsecutive_ids(self):
        for name in ('0002_25.0.tif.csv', '0009_25.0.tif.csv'):
            with open(os.path.join(self.root, name), 'w') as handle:
                handle.write('metadata\nmetadata\nPosition X,Position Y,Position Z\n0,0,0\n')
        with open(os.path.join(self.root, 'unrelated.txt'), 'w') as handle:
            handle.write('not a CSV')
        with mock.patch.object(pv, '_save_density') as save:
            pv.whole_brain_cell_heatmap(self.root, self.root, 2, 'whole')
        sink = save.call_args[0][1]
        self.assertEqual(sink[0, 0, 0], 2)
        self.assertEqual(save.call_args[0][2], (2, 1, 0))

    def test_invalid_volume_shapes_and_empty_groups_fail_explicitly(self):
        with self.assertRaises(ValueError):
            pv.read_group([])
        with self.assertRaises(ValueError):
            pv.read_group([np.zeros((2, 3)), np.zeros((3, 3))])
        with self.assertRaises(ValueError):
            pv._density_shape((5, 2.5))
        self.assertEqual(pv.read_group([], combine=False), [])
        volumes = pv.read_group([np.zeros((2, 3)), np.zeros((3, 3))], combine=False)
        self.assertEqual([volume.shape for volume in volumes], [(1, 2, 3), (1, 3, 3)])


if __name__ == '__main__':
    unittest.main()
