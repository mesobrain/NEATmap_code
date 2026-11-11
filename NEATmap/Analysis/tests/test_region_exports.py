"""Check export order and historical layer label grouping without image I/O."""

from collections import OrderedDict
import unittest
from unittest.mock import call, patch

from Analysis import get_region_layer as exports


class RegionExportTests(unittest.TestCase):
    def test_region_mapping_order_is_preserved(self):
        regions = OrderedDict((('VISp', [1, 2]), ('FRP', [3, 4])))
        with patch.object(exports, 'genSingleRegion') as export, patch('builtins.print'):
            exports.get_multi_region(regions, 'images', 'atlas', 'regions', 'points')
        self.assertEqual(export.call_args_list, [
            call('atlas', 'regions', 'VISp', [1, 2]),
            call('atlas', 'regions', 'FRP', [3, 4]),
        ])

    def test_four_and_five_layer_label_groups_keep_remaining_layer_six_ids(self):
        regions = OrderedDict((('FRP', list(range(10))), ('VISp', list(range(20, 32)))))
        with patch.object(exports, 'genSingleRegion') as export, patch('builtins.print'):
            exports.get_multi_layer(regions, 'images', 'atlas', 'regions', 'points')
        self.assertEqual(export.call_args_list, [
            call('atlas', 'regions', 'FRP_layer_1', [0, 1]),
            call('atlas', 'regions', 'FRP_layer_2_3', [2, 3]),
            call('atlas', 'regions', 'FRP_layer_5', [4, 5]),
            call('atlas', 'regions', 'FRP_layer_6', [6, 7, 8, 9]),
            call('atlas', 'regions', 'VISp_layer_1', [20, 21]),
            call('atlas', 'regions', 'VISp_layer_2_3', [22, 23]),
            call('atlas', 'regions', 'VISp_layer_4', [24, 25]),
            call('atlas', 'regions', 'VISp_layer_5', [26, 27]),
            call('atlas', 'regions', 'VISp_layer_6', [28, 29, 30, 31]),
        ])

    def test_point_export_keeps_historical_names_instead_of_guessing_anatomy(self):
        regions = OrderedDict((('ORBI', []), ('ORBl', []), ('TEa', [])))
        with patch.object(exports, 'genPointsSingleRegion') as export, \
                patch.object(exports, 'genSingleRegion') as mask_export, patch('builtins.print'):
            exports.get_multi_layer(regions, 'images', 'atlas', 'regions', 'points', True)
        self.assertEqual([item[0][-1] for item in export.call_args_list], [
            'ORBI_layer_1', 'ORBI_layer_2_3', 'ORBI_layer_5', 'ORBI_layer_6',
            'ORBl_layer_1', 'ORBl_layer_2_3', 'ORBl_layer_4', 'ORBl_layer_5', 'ORBl_layer_6',
            'TEa_layer_1', 'TEa_layer_2_3', 'TEa_layer_5', 'TEa_layer_6',
        ])
        mask_export.assert_not_called()


if __name__ == '__main__':
    unittest.main()
