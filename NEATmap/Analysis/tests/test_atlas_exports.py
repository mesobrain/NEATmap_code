import os
import tempfile
import unittest

import numpy as np
import tifffile

from Analysis import freesia_export_to_BrainRegion as atlas


def reference_edges(image):
    result = image.copy()
    offsets = [(0, -1), (0, 1), (1, 0), (1, 1), (1, -1), (-1, 0), (-1, -1), (-1, 1)]
    for row in range(1, image.shape[0] - 1):
        for column in range(1, image.shape[1] - 1):
            values = [image[row + dr, column + dc] for dr, dc in offsets]
            if len(np.unique(values)) == 1:
                result[row, column] = 0
    return result


class AtlasExportTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = self.directory.name

    def folder(self, name):
        path = os.path.join(self.root, name)
        os.makedirs(path, exist_ok=True)
        return path

    def test_vectorized_edges_match_original_rule_and_borders(self):
        random = np.random.RandomState(2026)
        for shape in [(1, 4), (3, 3), (17, 23)]:
            image = random.randint(0, 4, size=shape).astype(np.uint16)
            np.testing.assert_array_equal(atlas._atlas_edges(image), reference_edges(image))
        image = np.full((5, 5), 11, dtype=np.uint16)
        image[2, 2] = 99
        np.testing.assert_array_equal(atlas._atlas_edges(image), reference_edges(image))

    def test_corner_markers_use_valid_neighbors_and_keep_input_unchanged(self):
        image = np.array([[1000, 7], [7, 1000]], dtype=np.uint16)
        original = image.copy()
        np.testing.assert_array_equal(atlas.getMaskSinge(image), np.full((2, 2), 7))
        np.testing.assert_array_equal(image, original)

    def test_concat_infers_shape_accepts_integral_float_and_does_not_fill_source(self):
        source, output = self.folder('source'), self.folder('output')
        plane = np.arange(12, dtype=np.uint16).reshape(3, 4)
        tifffile.imwrite(os.path.join(source, '0000_25.tif'), plane)
        tifffile.imwrite(os.path.join(source, '0002_25.tif'), plane + 10)
        atlas.concat_image(source, output, 3.0, 'volume')
        volume = tifffile.imread(os.path.join(output, 'volume.tif'))
        self.assertEqual(volume.shape, (3, 3, 4))
        np.testing.assert_array_equal(volume[1], np.zeros_like(plane))
        self.assertFalse(os.path.exists(os.path.join(source, '0001_25.tif')))

    def test_region_mask_and_points_match_direct_boolean_reference(self):
        source, masks, regions, output = [self.folder(name) for name in ['images', 'atlas', 'regions', 'points']]
        labels = np.array([[[0, 1, 2], [2, 3, 0]], [[2, 1, 0], [0, 2, 3]]], dtype=np.uint16)
        points = np.array([[[1000, 1000, 0], [1000, 0, 0]], [[1000, 1000, 0], [1000, 1000, 0]]], dtype=np.uint16)
        for index in range(2):
            filename = '{:04d}_25.tif'.format(index)
            tifffile.imwrite(os.path.join(masks, filename), labels[index])
            tifffile.imwrite(os.path.join(source, filename), points[index])
        with open(os.path.join(masks, 'README.txt'), 'w') as stream:
            stream.write('Not an image')
        atlas.genSingleRegion(masks, regions, 'A', [1, 2, 2])
        expected_mask = np.isin(labels, [1, 2]).astype(np.uint16) * 255
        np.testing.assert_array_equal(tifffile.imread(os.path.join(regions, 'Region_A.tif')), expected_mask)
        atlas.genPointsSingleRegion(source, regions, output, 'A')
        expected_points = ((points == 1000) & (expected_mask > 0)).astype(np.uint16) * 255
        np.testing.assert_array_equal(tifffile.imread(os.path.join(output, 'Points_Region_A.tif')), expected_points)

    def test_invalid_stack_depth_is_reported(self):
        source, regions, output = [self.folder(name) for name in ['images', 'regions', 'points']]
        tifffile.imwrite(os.path.join(source, '0000.tif'), np.zeros((2, 3), dtype=np.uint16))
        tifffile.imwrite(os.path.join(regions, 'Region_A.tif'), np.zeros((2, 2, 3), dtype=np.uint16), photometric='minisblack')
        with self.assertRaisesRegex(ValueError, 'depth'):
            atlas.genPointsSingleRegion(source, regions, output, 'A')


if __name__ == '__main__':
    unittest.main()
