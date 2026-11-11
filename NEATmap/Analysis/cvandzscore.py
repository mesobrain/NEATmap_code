"""Regional coefficients of variation and control-referenced z-scores."""

import csv
import os

import numpy as np
import pandas as pd
from scipy.stats import variation


def _read_counts(path, regions=None):
    frame = pd.read_csv(path)
    if not {'Region', 'counts'}.issubset(frame.columns):
        raise ValueError('{} must contain Region and counts columns'.format(path))
    if frame['Region'].isnull().any() or frame['Region'].duplicated().any():
        raise ValueError('{} contains missing or duplicate regions'.format(path))
    if regions is None:
        regions = frame['Region'].values.copy()
    elif len(frame) != len(regions) or set(frame['Region']) != set(regions):
        raise ValueError('{} does not contain the same regions as the first sample'.format(path))
    counts = pd.to_numeric(frame.set_index('Region').reindex(regions)['counts'], errors='raise').values.astype(float)
    if not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError('{} contains non-finite or negative counts'.format(path))
    return regions, counts


def _read_groups(csv_path, index, treatment):
    index = [str(value) for value in index]
    if not index:
        raise ValueError('index must contain at least one sample')
    regions = None
    treatment_counts, control_counts = [], []
    for sample in index:
        path = os.path.join(csv_path, 'cell-counts-level6-sum_{}_{}.csv'.format(treatment, sample))
        regions, counts = _read_counts(path, regions)
        treatment_counts.append(counts)
        path = os.path.join(csv_path, 'cell-counts-level6-sum_con_{}.csv'.format(sample))
        regions, counts = _read_counts(path, regions)
        control_counts.append(counts)
    return regions, np.column_stack(treatment_counts), np.column_stack(control_counts), index


def coefficient_of_variation(csv_path, save_path, index):
    """Write population CVs, preserving the historical ES/control filenames."""
    regions, treatment, control, _ = _read_groups(csv_path, index, 'es')
    os.makedirs(save_path, exist_ok=True)
    values = pd.DataFrame({'Region': regions,
                           'SDS_CV': variation(treatment, axis=1),
                           'CON_CV': variation(control, axis=1)},
                          columns=['Region', 'SDS_CV', 'CON_CV'])
    values.to_csv(os.path.join(save_path, 'O_C_level6_CV.csv'), index=False)
    print('Finished')


def _zscore_headers(index):
    # Configuration is optional for library users and is not imported eagerly.
    try:
        from Parameter import Stats
    except ImportError:
        Stats = {}
    configured_index = [str(value) for value in Stats.get('group_index', [])]
    names = list(Stats.get('group_data_name', []))
    if (len(names) == 2 * len(configured_index) and configured_index
            and all(value in configured_index for value in index)):
        positions = [configured_index.index(value) for value in index]
        return ([names[position] for position in positions]
                + [names[len(configured_index) + position] for position in positions])
    if len(names) == 2 * len(index):
        return names
    return ['CON' + value for value in index] + ['SDS' + value for value in index]


def z_score(csv_path, save_path, index):
    """Write scores using the control mean and population standard deviation.

    Scores for zero control variance remain NaN/inf, rather than being
    silently replaced by a numerically meaningful score.
    """
    regions, treatment, control, index = _read_groups(csv_path, index, 'sds')
    mean = np.mean(control, axis=1, keepdims=True)
    std = np.std(control, axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        treatment_scores = (treatment - mean) / std
        control_scores = (control - mean) / std
    scores = np.concatenate((control_scores, treatment_scores), axis=1)
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, 'I_C_level6_z_score.csv'), 'w', newline='') as handle:
        writer = csv.writer(handle, dialect='excel')
        writer.writerow(['Region'] + _zscore_headers(index))
        for region, row in zip(regions, scores):
            writer.writerow([region] + row.tolist())
    print('Finished')


if __name__ == '__main__':
    from Parameter import Stats

    csv_path = Stats['level6_cell_counts_path']
    index = Stats['group_index']
    coefficient_of_variation(csv_path=csv_path, save_path=csv_path, index=index)
    z_score(csv_path=csv_path, save_path=csv_path, index=index)
