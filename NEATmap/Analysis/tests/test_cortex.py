"""Regression checks for the existing cortex row mapping and rounding rules."""

import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from Analysis.cortex_cell_counts import cortex_stats


class CortexCountsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / 'whole_brain_cell_counts'
        self.source.mkdir()

    def write_counts(self, values):
        pd.DataFrame({'count': values}).to_csv(self.source / 'cell-counting.csv', index=False)

    def test_atlas_order_four_layer_and_five_layer_rounding_are_unchanged(self):
        self.write_counts(np.arange(600))
        cortex_stats(self.source)
        with (self.root / 'cortex_cells_counts.csv').open(newline='', encoding='utf-8') as stream:
            rows = list(csv.reader(stream))
        self.assertEqual(rows[0], ['Region', 'layer 1', 'layer 2/3', 'layer 4',
                                  'layer 5', 'layer 6', 'SUM'])
        self.assertEqual([row[0] for row in rows[1:]], [
            'FRP', 'MOp', 'MOs', 'SSp-n', 'SSp-bfd', 'SSp-ll', 'SSp-m', 'SSp-ul',
            'SSp-tr', 'SSp-un', 'SSs', 'AUDd', 'AUDp', 'AUDpo', 'AUDv', 'VISal',
            'VISam', 'VISl', 'VISp', 'VISpl', 'VISpm', 'VISli', 'VISpor', 'RSPagl',
            'RSPd', 'RSPv', 'VISa', 'VISrl', 'ORBl', 'ORBm', 'ORBvl', 'ACAd',
            'ACAv', 'PL', 'ILA', 'VISC', 'GU', 'AId', 'AIp', 'AIv', 'PERl', 'TEa', 'ECT',
        ])
        result = {row[0]: [float(value) for value in row[1:]] for row in rows[1:]}
        self.assertEqual(result['FRP'], [12, 14, 0, 16, 19, 61])
        self.assertEqual(result['VISp'], [296, 298, 300, 302, 305, 1501])
        # Keep the historical SSp-bfd layer-6 slice [6:] until atlas review.
        self.assertEqual(result['SSp-bfd'], [68, 70, 72, 74, 76, 360])
        self.assertEqual(result['ECT'], [586, 588, 0, 590, 593, 2357])

    def test_fractional_counts_keep_floor_pairs_and_truncated_layer_six_means(self):
        self.write_counts(np.arange(600) + 0.75)
        cortex_stats(self.source)
        result = pd.read_csv(self.root / 'cortex_cells_counts.csv').set_index('Region')
        self.assertEqual(result.loc['FRP'].tolist(), [13, 15, 0, 17, 20, 65])

    def test_invalid_schema_short_atlas_and_invalid_counts_fail_before_output(self):
        pd.DataFrame({'wrong': np.arange(600)}).to_csv(self.source / 'cell-counting.csv', index=False)
        with self.assertRaisesRegex(ValueError, 'count.*column'):
            cortex_stats(self.source)
        self.write_counts(np.arange(595))
        with self.assertRaisesRegex(ValueError, '596'):
            cortex_stats(self.source)
        for invalid in (np.nan, np.inf, -1, 'not-a-count'):
            with self.subTest(invalid=invalid):
                values = np.arange(600).astype(object)
                values[20] = invalid
                self.write_counts(values)
                with self.assertRaises(ValueError):
                    cortex_stats(self.source)
        self.assertFalse((self.root / 'cortex_cells_counts.csv').exists())

    def test_malformed_rows_are_not_skipped_and_cannot_shift_atlas_indices(self):
        source = self.source / 'cell-counting.csv'
        source.write_text('Region,count\n' + 'region,1\n' * 596 + 'region,1,extra\n',
                          encoding='utf-8')
        with self.assertRaises(pd.errors.ParserError):
            cortex_stats(self.source)


if __name__ == '__main__':
    unittest.main()
