"""Atlas and point exports using slice-wise operations and bounded memory."""
import os
import shutil

import numpy as np
import tifffile


def _tiff_files(root):
    files = sorted(name for name in os.listdir(root)
                   if name.lower().endswith(('.tif', '.tiff'))
                   and os.path.isfile(os.path.join(root, name)))
    if not files:
        raise ValueError('No TIFF images found in {}'.format(root))
    return files


def _read_plane(path, shape=None):
    plane = tifffile.imread(path)
    if plane.ndim != 2 or (shape is not None and plane.shape != shape):
        raise ValueError('Expected a 2-D image of shape {}, found {} in {}'.format(shape, plane.shape, path))
    return plane


def concat_image(path, save_path, z_num, save_name):
    if int(z_num) != z_num or z_num <= 0:
        raise ValueError('z_num must be a positive integer.')
    z_num = int(z_num)
    paths = [os.path.join(path, '{:04d}_25.tif'.format(index)) for index in range(z_num)]
    available = next((filename for filename in paths if os.path.isfile(filename)), None)
    shape = _read_plane(available).shape if available else (400, 560)
    volume = np.zeros((z_num,) + shape, dtype=np.uint16)
    for index, filename in enumerate(paths):
        if os.path.isfile(filename):
            volume[index] = _read_plane(filename, shape)
        print('finished {} image'.format(index))
    os.makedirs(save_path, exist_ok=True)
    # Missing slices remain zero in the output; the source directory is not modified.
    tifffile.imwrite(os.path.join(save_path, save_name + '.tif'), volume,
                     photometric='minisblack')


def getMaskSinge(image):
    """Replace point markers with the modal valid 8-neighbour atlas label.

    Interior tie-breaking and sequential replacement match the original code.
    Border neighbours are clipped to the image instead of wrapping or raising.
    """
    atlas = np.array(image, copy=True)
    if atlas.ndim != 2:
        raise ValueError('Atlas images must be two-dimensional.')
    height, width = atlas.shape
    neighbors = [(0, -1), (0, 1), (1, 0), (1, 1), (1, -1),
                 (-1, 0), (-1, -1), (-1, 1)]
    for row, column in zip(*np.where(atlas == 1000)):
        values = [atlas[row + dr, column + dc] for dr, dc in neighbors
                  if 0 <= row + dr < height and 0 <= column + dc < width]
        if values:
            labels, counts = np.unique(values, return_counts=True)
            atlas[row, column] = labels[np.argmax(counts)]
        else:
            atlas[row, column] = 0
    return atlas


def regenAtlas(image_root):
    atlas_root = os.path.join(image_root, '..', 'atlas')
    os.makedirs(atlas_root, exist_ok=True)
    for filename in _tiff_files(image_root):
        atlas = getMaskSinge(_read_plane(os.path.join(image_root, filename)))
        tifffile.imwrite(os.path.join(atlas_root, filename), atlas)


def _atlas_edges(atlas):
    """Preserve the legacy eight-neighbour rule, including the unchanged border."""
    output = atlas.copy()
    height, width = atlas.shape
    if height < 3 or width < 3:
        return output
    reference = atlas[1:-1, :-2]
    uniform = np.ones((height - 2, width - 2), dtype=bool)
    for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1),
                   (-1, 0), (-1, -1), (-1, 1)]:
        uniform &= atlas[1 + dr:height - 1 + dr, 1 + dc:width - 1 + dc] == reference
    output[1:-1, 1:-1][uniform] = 0
    return output


def genEdgefromAtlas(atlas_root, Edge_root):
    os.makedirs(Edge_root, exist_ok=True)
    for filename in _tiff_files(atlas_root):
        atlas = _read_plane(os.path.join(atlas_root, filename))
        tifffile.imwrite(os.path.join(Edge_root, filename), _atlas_edges(atlas))


def genPoints(image_root, Points_root):
    os.makedirs(Points_root, exist_ok=True)
    for filename in _tiff_files(image_root):
        image = _read_plane(os.path.join(image_root, filename))
        points = (image == 1000).astype(np.uint16) * 255
        tifffile.imwrite(os.path.join(Points_root, filename), points)


def genSingleRegion(atlas_root, Region_root, Region_name, Mask_id):
    files = _tiff_files(atlas_root)
    shape = _read_plane(os.path.join(atlas_root, files[0])).shape
    region = np.empty((len(files),) + shape, dtype=np.uint16)
    labels = np.asarray(list(Mask_id))
    # Keep only one atlas slice in memory; do not allocate an atlas volume or
    # another full-size temporary volume for every label.
    for index, filename in enumerate(files):
        plane = _read_plane(os.path.join(atlas_root, filename), shape)
        region[index] = np.isin(plane, labels).astype(np.uint16) * 255
    os.makedirs(Region_root, exist_ok=True)
    tifffile.imwrite(os.path.join(Region_root, 'Region_' + Region_name + '.tif'),
                     region, photometric='minisblack')
    print('finished')


def genPointsSingleRegion(image_root, Region_root, Points_SingleRegion_root, Region_name):
    files = _tiff_files(image_root)
    region = tifffile.imread(os.path.join(Region_root, 'Region_' + Region_name + '.tif'))
    if region.ndim == 2 and len(files) == 1:
        region = region[np.newaxis, :, :]
    if region.ndim != 3 or region.shape[0] != len(files):
        raise ValueError('Region mask depth must match the number of input slices.')
    points = np.empty(region.shape, dtype=np.uint16)
    for index, filename in enumerate(files):
        plane = _read_plane(os.path.join(image_root, filename), region.shape[1:])
        points[index] = ((plane == 1000) & (region[index] > 0)).astype(np.uint16) * 255
    os.makedirs(Points_SingleRegion_root, exist_ok=True)
    tifffile.imwrite(os.path.join(Points_SingleRegion_root, 'Points_Region_' + Region_name + '.tif'),
                     points, photometric='minisblack')
    print('finished')


def merge(root):
    files = _tiff_files(root)
    shape = _read_plane(os.path.join(root, files[0])).shape
    total = np.zeros(shape, dtype=float)
    for filename in files:
        total += _read_plane(os.path.join(root, filename), shape)
    total[total > 1] = 255
    tifffile.imwrite(os.path.join(root, '..', 'merge_points.tif'), total.astype(np.uint8))



def label_parse(lable_root):
    """
    parse the WHS rat atlas label.
    :param lable_root:
    :return:
    """
    with open(lable_root, 'r') as f:
        lines = f.readlines()
    name_index = []
    for i in range(15, len(lines)):
        line = lines[i]
        id  = line.strip().split(' ')[0]
        index = line.find('\"')
        # print(id, line[index+1:-1-1])
        name_index.append((id, line[index+1:-1-1]))
    print(name_index)
    return name_index


def genMergeRaw(Edge_root, Points_root, Save_root, Snapshot_id):
    if not os.path.exists(Save_root):
        os.mkdir(Save_root)
    for i in range(len(Snapshot_id)):
        dir = os.path.join(Save_root,str(i).zfill(2))
        if not os.path.exists(dir):
            os.mkdir(dir)
        point_save_dir = os.path.join(dir, "Points")
        if not os.path.exists(point_save_dir):
            os.mkdir(point_save_dir)
        Edge_name = str(Snapshot_id[i]).zfill(4)+'_25.tif'
        print(Edge_name)
        shutil.copyfile(os.path.join(Edge_root,Edge_name),os.path.join(dir,Edge_name))
        for offset in range(0,8):
            if i >= 5:
                offset = -offset
            Point_index = Snapshot_id[i] - offset
            Point_name = str(Point_index).zfill(4) +'_25.tif'
            shutil.copyfile(os.path.join(Points_root, Point_name), os.path.join(point_save_dir, Point_name))
        merge(point_save_dir)
        raw = tifffile.imread(os.path.join(Edge_root,Edge_name))
        tifffile.imwrite(os.path.join(dir,Edge_name),raw.astype('uint8'))


if __name__ == "__main__":
    from Parameter import *
    # freesia export cell counts
    image_root = os.path.join(freesia_export_path, 'images')
    regenAtlas(image_root)

    atlas_root = os.path.join(image_root, '..', 'atlas')

    Edge_root = os.path.join(image_root, '..', 'Edge')
    genEdgefromAtlas(atlas_root, Edge_root)

    Points_root = os.path.join(image_root, '..', 'Points')
    genPoints(image_root,Points_root)

    Region_name = 'CBX'
    Mask_id = Whole_brain_region_level2[Region_name]#CBX
    
    Region_root = os.path.join(image_root, '..', 'Region')
    Points_SingleRegion_root = os.path.join(image_root, '..', 'Points_Single_Region')
    genSingleRegion(atlas_root, Region_root, Region_name, Mask_id)
    genPointsSingleRegion(image_root, Region_root, Points_SingleRegion_root, Region_name)
    save_path = os.path.join(Points_root, '..', '3d_result')
    save_name = 'whole_brain_point'
    altas_num = Preprocessing['brain_total_num'] / Spot_map['group_num']
    concat_image(Points_root, save_path, z_num=altas_num, save_name=save_name)

    Save_root = Atlas_edge_pionts_save_path
    genMergeRaw(Edge_root, Points_root, Save_root, Atlas_snapshot_id)
             