"""Synthetic-volume regression checks; no experiment data is required."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Analysis import Density, Intensity, Position_stats, Volume_ratio


def write_tiff(path, array):
    tifffile.imwrite(str(path), array, photometric='minisblack')


def legacy_measure(image, mask, minimum, maximum):
    """Original component/weighted-center calculation without deprecated np.int."""
    labels, _ = ndi.label(mask, ndi.generate_binary_structure(image.ndim, 1))
    ids, counts = np.unique(labels, return_counts=True)
    selected = [(i, count) for i, count in zip(ids, counts)
                if i and minimum < count <= maximum]
    if not selected:
        return []
    centers = np.rint(ndi.center_of_mass(image, labels, [item[0] for item in selected])).astype(np.intp)
    rows = []
    for (object_id, count), center in zip(selected, centers):
        def neighborhood(radius):
            slices = tuple(slice(max(c - radius, 0), min(c + radius + 1, size))
                           for c, size in zip(center, image.shape))
            return image[slices]
        background = np.percentile(neighborhood(4), 5)
        signal = neighborhood(1).max()
        rows.append(list(center[::-1]) + [background, max(signal, background), count])
    return rows


class IntensityTests(unittest.TestCase):
    def test_measurement_matches_original_in_two_and_three_dimensions(self):
        for shape in [(8, 9), (5, 8, 9)]:
            image = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape) + 1
            mask = np.zeros(shape, dtype=np.uint8)
            if len(shape) == 2:
                mask[0, 0:3] = 255
                mask[3, 1:3] = 255  # Exactly minimum: excluded.
                mask[6, 2:6] = 255
            else:
                mask[0, 0, 0:3] = 255
                mask[2, 3, 1:3] = 255
                mask[4, 6, 2:6] = 255
            expected = legacy_measure(image, mask, 2, 4)
            actual = Intensity._measure_intensity(image, mask, 2, 4)
            np.testing.assert_allclose(actual, expected)

    def test_zero_signal_object_uses_finite_geometric_center(self):
        image = np.zeros((3, 4, 5), dtype=np.uint16)
        mask = np.zeros_like(image, dtype=np.uint8)
        mask[1, 2, 1:4] = 255
        rows = Intensity._measure_intensity(image, mask, 2, 64)
        np.testing.assert_array_equal(rows, [[2, 2, 1, 0, 0, 3]])

    def test_intensity_modes_match_signal_definition_without_mutating_input(self):
        for shape in [(5, 6), (4, 5, 6)]:
            image = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
            before = image.copy()
            image.setflags(write=False)
            labels = np.zeros(shape, dtype=np.uint8)
            labels.flat[0:3] = 1
            center = (0,) * len(shape)
            local = image[(slice(0, 2),) * len(shape)]
            background = np.percentile(image, 5)  # Radius 9 covers this fixture.
            signals = {'max': local.max(), 'local_mean': local.mean(),
                       'obj_mean': image[labels == 1].mean()}
            measure = Intensity.get_intensity if len(shape) == 3 else Intensity.get_2d_intensity
            for mode, signal in signals.items():
                actual = measure(image, labels, 1, center, mode=mode, min_rad=9)
                np.testing.assert_allclose(actual, (max(signal - background, 0), background))
            np.testing.assert_array_equal(image, before)
            with self.assertRaisesRegex(ValueError, 'Unknown intensity mode'):
                measure(image, labels, 1, center, mode='unsupported')

    def test_local_neighborhood_is_clipped_view(self):
        image = np.zeros((4, 5, 6))
        neighborhood = Intensity.get_local_voxels(image, 0, 0, 0, 1)
        self.assertEqual(neighborhood.shape, (2, 2, 2))
        self.assertTrue(np.shares_memory(image, neighborhood))

    def test_batch_reads_the_brain_image_once_and_keeps_csv_schema(self):
        image = np.arange(125, dtype=np.uint16).reshape(5, 5, 5) + 1
        mask = np.zeros_like(image, dtype=np.uint8)
        mask[2, 2, 1:4] = 255
        reads = []
        def read(path):
            reads.append(os.path.basename(path))
            return image if str(path).endswith('_whole_brain_25mic.tif') else mask
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.object(Intensity.tifffile, 'imread', side_effect=read):
                Intensity.get_intensity_csv({'A': [], 'B': []}, 'images', 'labels', temporary, id='sample')
            self.assertEqual(reads.count('sample_whole_brain_25mic.tif'), 1)
            result = pd.read_csv(os.path.join(temporary, 'A_intensity.csv'))
            self.assertEqual(list(result.columns), ['X', 'Y', 'Z', 'bg_percentile', 'intensity', 'volume'])
            self.assertEqual(len(result), 1)

    def test_patch_writes_empty_csv_for_no_objects(self):
        image = np.zeros((5, 6), dtype=np.uint16)
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.object(Intensity.tifffile, 'imread', return_value=image):
                Intensity.patch_intensity('', '', 'image.tif', 'mask.tif', temporary)
            frame = pd.read_csv(os.path.join(temporary, 'patch_intensity.csv'))
            self.assertTrue(frame.empty)
            self.assertEqual(list(frame.columns), ['X', 'Y', 'bg_percentile', 'intensity', 'volume'])

    def test_histograms_are_associated_by_region_name(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'histograms'
            path.mkdir()
            pd.DataFrame({'normalized_count': [8, 9]}).to_csv(str(path / 'B.csv'), index=False)
            pd.DataFrame({'normalized_count': [1, 2]}).to_csv(str(path / 'A.csv'), index=False)
            (path / 'unrelated.txt').write_text('not a histogram')
            with mock.patch.object(Intensity.os, 'listdir', return_value=['B.csv', 'unrelated.txt', 'A.csv']):
                Intensity.get_whole_norm_counts(str(path), {'A': [], 'B': []}, 'combined.csv')
            frame = pd.read_csv(str(Path(temporary) / 'combined.csv'))
            self.assertEqual(frame['A'].tolist(), [1, 2])
            self.assertEqual(frame['B'].tolist(), [8, 9])


class RegionQuantificationTests(unittest.TestCase):
    def test_density_aligns_named_counts_and_preserves_original_scale(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            region = np.zeros((2, 2, 2), dtype=np.uint8)
            region.flat[:2] = 255
            write_tiff(path / 'Region_A.tif', region)
            write_tiff(path / 'Region_B.tif', np.full((2, 2, 2), 255, dtype=np.uint8))
            pd.DataFrame({'Region': ['B', 'A'], 'SUM': [8, 6]}).to_csv(str(path / 'cortex_cells_counts.csv'), index=False)
            Density.get_density_csv(str(path), str(path), {'A': [], 'B': []}, str(path / 'output'))
            result = pd.read_csv(str(path / 'output' / 'cortex_density.csv'))
            np.testing.assert_allclose(result['Density'], [3 * 15626, 15626])
            self.assertEqual(result['Region'].tolist(), ['A', 'B'])

    def test_density_missing_region_is_reported(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'cortex_cells_counts.csv'
            pd.DataFrame({'Region': ['B'], 'SUM': [8]}).to_csv(str(path), index=False)
            with self.assertRaisesRegex(ValueError, 'Missing regions'):
                Density.get_density_csv(temporary, temporary, {'A': []}, temporary)

    def test_volume_filter_keeps_strict_minimum_and_inclusive_maximum(self):
        mask = np.zeros((4, 5, 7), dtype=np.uint8)
        mask[0, 0, :3] = 255
        mask[3, 0, :2] = 255
        mask[3, 4, :4] = 255
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            write_tiff(path / 'Region_A.tif', np.full(mask.shape, 255, dtype=np.uint8))
            write_tiff(path / 'Points_Region_A.tif', mask)
            ratio = Volume_ratio.volume_ratio(temporary, temporary, 'A', min_volume=2, max_volume=3)
            self.assertEqual(ratio, np.round(3 / mask.size * 100, 2))
            write_tiff(path / 'Region_A.tif', np.zeros_like(mask))
            self.assertEqual(Volume_ratio.volume_ratio(temporary, temporary, 'A'), 0)

    def test_group_means_align_regions_across_sample_files(self):
        columns = ['layer 1', 'layer 2_3', 'layer 4', 'layer 5', 'layer 6']
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, sample in enumerate(['sample1', 'sample2']):
                path = root / sample / 'whole_brain_cell_counts'
                path.mkdir(parents=True)
                frame = pd.DataFrame({column: [2 + index * 2, 10 + index * 2] for column in columns})
                frame.insert(0, 'Region_name', ['A', 'B'])
                if index:
                    frame = frame.iloc[::-1]
                for filename in ['layer_density.csv', 'layer_volume.csv']:
                    frame.to_csv(str(path / filename), index=False)
            for module, kind in [(Density, 'density'), (Volume_ratio, 'volume')]:
                measure = getattr(module, 'get_whole_layer_mean_' + kind)
                measure(temporary, ['sample1', 'sample2'], {'A': [], 'B': []}, temporary, 'test')
                frame = pd.read_csv(str(root / ('test_mean_' + kind + '.csv')))
                np.testing.assert_allclose(frame[['layer1', 'layer2_3', 'layer4', 'layer5', 'layer6']],
                                           [[3] * 5, [11] * 5])
                with self.assertRaisesRegex(ValueError, 'At least one sample'):
                    measure(temporary, [], {'A': [], 'B': []}, temporary, 'empty')

    def test_position_counts_cover_all_slices_and_preserve_split(self):
        image = np.zeros((23, 3, 2), dtype=np.uint8)
        image[[0, 9, 10, 22], 0, 0] = 255
        image[2, 1, 1] = 128
        with tempfile.TemporaryDirectory() as temporary:
            write_tiff(Path(temporary) / 'position.tif', image)
            with mock.patch.dict(Position_stats.Stats, {'position_group': 2}):
                self.assertEqual(Position_stats.position_stats(temporary, 'position.tif'), (2, 2))


if __name__ == '__main__':
    unittest.main()
