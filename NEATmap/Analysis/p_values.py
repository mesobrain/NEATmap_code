"""Cell-density maps and equal-variance voxel-wise t-tests."""

import os
import re
import warnings

import numpy as np
import pandas as pd
import tifffile
from scipy import stats


def search_indices_sphere(radius):
    """Return relative integer offsets in an ellipsoid (a scalar means 3-D)."""
    if np.isscalar(radius):
        radius = (radius,) * 3
    radii = np.asarray(radius, dtype=float)
    if (radii.ndim != 1 or not len(radii) or not np.isfinite(radii).all()
            or (radii < 0).any() or not np.equal(radii, np.floor(radii)).all()):
        raise ValueError('radius must contain non-negative integers')
    radii = radii.astype(np.int64)
    grid = [np.arange(-value, value + 1, dtype=float) / max(1, value) for value in radii]
    normalized = np.array(np.meshgrid(*grid, indexing='ij'))
    distances = np.sum(normalized * normalized, axis=0)
    order = np.argsort(distances.reshape(-1))
    order = order[distances.reshape(-1)[order] <= 1]
    return np.array(np.unravel_index(order, distances.shape)).T - radii


def _density_shape(shape):
    values = np.asarray(shape, dtype=float)
    if (values.ndim != 1 or not np.isfinite(values).all() or (values <= 0).any()
            or not np.equal(values, np.floor(values)).all()):
        raise ValueError('Density-map dimensions must be positive integers')
    return tuple(int(value) for value in values)


def _read_coordinates(path, columns, skiprows=None):
    frame = pd.read_csv(path, skiprows=skiprows)
    if not set(columns).issubset(frame.columns):
        raise ValueError('{} must contain {}'.format(path, ', '.join(columns)))
    values = frame.loc[:, list(columns)].apply(pd.to_numeric, errors='raise').values.astype(float)
    if not np.isfinite(values).all():
        raise ValueError('{} contains non-finite coordinates'.format(path))
    return np.floor(values / 25).astype(np.int64)


def _accumulate_density(sink, coordinates, offsets, chunk_size=65536):
    """Accumulate exact counts with bounded coordinate temporaries.

    Fixed-offset targets are unique after duplicate points are combined, so
    indexed addition is safe. Points outside the image are retained when their
    neighbourhood overlaps the image, matching the original boundary rule.
    """
    coordinates = np.asarray(coordinates)
    offsets = np.asarray(offsets)
    if coordinates.ndim != 2 or coordinates.shape[1] != sink.ndim:
        raise ValueError('Coordinate dimensions do not match the density map')
    if offsets.ndim != 2 or offsets.shape[1] != sink.ndim:
        raise ValueError('Neighbourhood dimensions do not match the density map')
    if chunk_size <= 0:
        raise ValueError('chunk_size must be positive')
    shape = np.asarray(sink.shape)
    maximum = np.iinfo(sink.dtype).max
    for start in range(0, len(coordinates), chunk_size):
        points, counts = np.unique(coordinates[start:start + chunk_size], axis=0, return_counts=True)
        for offset in offsets:
            positions = points + offset
            valid = np.all((positions >= 0) & (positions < shape), axis=1)
            if not valid.any():
                continue
            target = tuple(positions[valid].T)
            increment = counts[valid].astype(sink.dtype)
            current = sink[target]
            if np.any(current > maximum - increment):
                raise OverflowError('Density counts exceed the accumulation dtype')
            sink[target] = current + increment
    return sink


def _save_density(path, sink, axes):
    # Preserve the existing 16-bit output when lossless; never wrap high counts.
    dtype = np.uint16
    if sink.size and sink.max() > np.iinfo(np.uint16).max:
        dtype = np.uint32
        warnings.warn('Density exceeds uint16; writing uint32 TIFF to preserve counts', RuntimeWarning)
    tifffile.imwrite(path, np.transpose(sink, axes).astype(dtype))


def _single_density(path, save_path, save_name, columns, shape, skiprows=None):
    shape = _density_shape(shape)
    coordinates = _read_coordinates(path, columns, skiprows=skiprows)
    sink = np.zeros(shape, dtype=np.uint32)
    offsets = search_indices_sphere((6,) * len(shape))
    _accumulate_density(sink, coordinates, offsets)
    os.makedirs(save_path, exist_ok=True)
    _save_density(os.path.join(save_path, save_name + '_cell_density.tif'), sink, (1, 0))


def cell_heatmap(data_path, save_path, fname):
    """Write a coronal density map at 25 microns per voxel."""
    _single_density(os.path.join(data_path, fname + '.csv'), save_path, fname,
                    ('Position X', 'Position Y'), (560, 400))


def group_cell_heatmap(data_path, save_path, id):
    path = os.path.join(data_path, 'whole_brain_cell_counts', 'Thumbnail_CSV',
                        '{:04d}_25.0.tif.csv'.format(id))
    _single_density(path, save_path, str(id), ('Position X', 'Position Y'),
                    (560, 400), skiprows=[0, 1])


def sagittal_cell_heatmap(data_path, save_path, fname, z_length):
    _single_density(os.path.join(data_path, fname + '.csv'), save_path, fname,
                    ('Position Y', 'Position Z'), (400, z_length))


def horizontal_cell_heatmap(data_path, save_path, fname, z_length):
    _single_density(os.path.join(data_path, fname + '.csv'), save_path, fname,
                    ('Position X', 'Position Z'), (560, z_length))


def whole_brain_cell_heatmap(csv_path, save_path, z_length, save_name):
    """Accumulate the actual thumbnail CSVs, allowing non-consecutive slice IDs."""
    names = [name for name in os.listdir(csv_path)
             if re.match(r'^\d+_25\.0\.tif\.csv$', name)]
    names.sort(key=lambda name: int(name.split('_', 1)[0]))
    if not names:
        raise ValueError('No thumbnail coordinate CSVs found in {}'.format(csv_path))
    sink = np.zeros(_density_shape((560, 400, z_length)), dtype=np.uint32)
    offsets = search_indices_sphere((6, 6, 6))
    for name in names:
        coordinates = _read_coordinates(os.path.join(csv_path, name),
                                        ('Position X', 'Position Y', 'Position Z'),
                                        skiprows=[0, 1])
        _accumulate_density(sink, coordinates, offsets)
        print('finished {} csv'.format(name))
    os.makedirs(save_path, exist_ok=True)
    _save_density(os.path.join(save_path, save_name + '_cell_density.tif'), sink, (2, 1, 0))


def read_group(sources, combine=True):
    """Read matching TIFFs; sample arrays retain a leading sample axis."""
    if isinstance(sources, np.ndarray):
        return sources
    group = []
    shape = None
    for source in sources:
        data = source if isinstance(source, np.ndarray) else tifffile.imread(source)
        data = np.asarray(data)
        if shape is None:
            shape = data.shape
        elif combine and data.shape != shape:
            raise ValueError('All volumes in a group must have identical shapes')
        group.append(np.expand_dims(data, axis=0))
    if not group:
        if combine:
            raise ValueError('A group must contain at least one volume')
        return []
    return np.concatenate(group, axis=0) if combine else group


def cutoff_p_values(pvals, p_cutoff=0.05):
    """Return a copy, optionally truncating p-values above the cutoff."""
    values = np.array(pvals, dtype=float, copy=True)
    if p_cutoff is None:
        return values
    if not np.isfinite(p_cutoff) or not 0 <= p_cutoff <= 1:
        raise ValueError('p_cutoff must be between 0 and 1, or None')
    return np.minimum(values, p_cutoff)


def color_p_values(pvals, psign, positive=[1, 0], negative=[0, 1], p_cutoff=None,
                   positive_trend=[0, 1], negative_trend=[1, 0], pmax=None):
    values = np.asarray(pvals, dtype=float)
    signs = np.asarray(psign)
    if values.shape != signs.shape:
        raise ValueError('pvals and psign must have the same shape')
    colors = [np.asarray(color, dtype=float) for color in
              (positive, negative, positive_trend, negative_trend)]
    active_colors = colors[:2] if p_cutoff is None else colors
    if any(color.ndim != 1 or len(color) != len(colors[0]) for color in active_colors):
        raise ValueError('All color vectors must be one-dimensional and equally sized')
    if pmax is None:
        pmax = values.max() if values.size else 0.0
    inverse = pmax - values
    result = np.zeros(values.shape + (len(colors[0]),), dtype=float)
    if p_cutoff is None:
        masks = [(signs > 0, colors[0]), (signs < 0, colors[1])]
    else:
        if not np.isfinite(p_cutoff) or not 0 <= p_cutoff <= 1:
            raise ValueError('p_cutoff must be between 0 and 1, or None')
        significant = values < p_cutoff
        positive_sign = signs > 0
        masks = [(positive_sign & significant, colors[0]),
                 (~positive_sign & significant, colors[1]),
                 (positive_sign & ~significant, colors[2]),
                 (~positive_sign & ~significant, colors[3])]
    for mask, color in masks:
        result[mask] = inverse[mask][:, None] * color
    return result


def t_test_voxelization(group1, group2, signed=False, remove_nan=True, p_cutoff=None):
    """Compare sample stacks with the existing equal-variance, two-sided t-test."""
    group1, group2 = np.asarray(read_group(group1)), np.asarray(read_group(group2))
    if (group1.ndim < 1 or group2.ndim < 1 or not group1.shape[0]
            or not group2.shape[0] or group1.shape[1:] != group2.shape[1:]):
        raise ValueError('Groups need samples with matching spatial shapes')
    tvals, pvals = stats.ttest_ind(group1, group2, axis=0, equal_var=True)
    if remove_nan:
        undefined = np.isnan(pvals)
        pvals = np.where(undefined, 1.0, pvals)
        tvals = np.where(undefined, 0.0, tvals)
    pvals = cutoff_p_values(pvals, p_cutoff=p_cutoff)
    return (pvals, np.sign(tvals)) if signed else pvals


def generate_p_mapping(group1, group2, save_path, save_name):
    """Write positive and negative maps for both 2-D and 3-D sample volumes."""
    pvals, psign = t_test_voxelization(group1, group2, signed=True,
                                      remove_nan=True, p_cutoff=0.05)
    colors = color_p_values(pvals, psign, positive=[0, 1], negative=[1, 0], p_cutoff=0.05)
    os.makedirs(save_path, exist_ok=True)
    for channel, suffix in ((1, '_pos.tif'), (0, '_neg.tif')):
        result = np.squeeze(colors[..., channel]).astype('float32')
        tifffile.imwrite(os.path.join(save_path, save_name + suffix), result)

if __name__ == "__main__":
    from Parameter import Data_root, Preprocessing, Spot_map, Stats

    root = Data_root
    fname = {'FS8':'0490_25', 'Ctrl8':'0490_25'}
    z_length = Preprocessing['brain_total_num'] / Spot_map['group_num']

    data_name = [ '', '', ...]
    group_name = list(fname.keys())
    for i in range(len(data_name)):
      data_path = os.path.join(root, data_name[i], 'coordinate')
      save_path = os.path.join(root, data_name[i], 'cell_density')
      cell_heatmap(data_path, save_path, fname=fname[group_name[i]])
      print('finished {}'.format(group_name[i]))
    sagittal_cell_heatmap(data_path, save_path, fname, z_length)
    horizontal_cell_heatmap(data_path, save_path, fname, z_length)

    for i in range(len(data_name)):
      for j in range(35):
        index = j*16
        group_cell_heatmap(data_path=os.path.join(root, data_name[i]), save_path=os.path.join(root, data_name[i], 'group_cell_denisty'), id=index)
        if index >= z_length:
          break
        print('finish {}'.format(index))
      print('finished' + data_name[i])
    group1 = ['', '', ...]
    group2 = ['', '', ...]
    fs_name = Stats['Exp_data_list']
    ctrl_name = Stats['Ctrl_data_list']

    for i in range(35):
      index = i*16
      group_fs = [os.path.join(root, fs_name[j], 'group_cell_denisty', '{}_cell_density.tif'.format(index)) for j in range(len(fs_name))]
      # group1.append(group_fs)
      group_ctrl = [os.path.join(root, ctrl_name[k], 'group_cell_denisty', '{}_cell_density.tif'.format(index)) for k in range(len(ctrl_name))]
      # group2.append(group_ctrl)
      save_p_map_path = os.path.join(Data_root, '', 'group_p_map')
      save_name = str(index) + '_p_map' # Select according to the 300th image of FS6 in 25 microns. For example, the 25-micron FS6 has a total of 648 images. 300/648 ~= 0.463.
                                  #Select other groups at this rate
      generate_p_mapping(group_fs, group_ctrl, save_p_map_path, save_name)
      print('finished {}'.format(index))

    csv_path = os.path.join(Data_root, '', r'whole_brain_cell_counts\Thumbnail_CSV')
    save_path = os.path.join(Data_root, '', 'cell_density')
    save_name = ''
    whole_brain_cell_heatmap(csv_path, save_path, z_length=len(os.listdir(csv_path)), save_name=save_name)
