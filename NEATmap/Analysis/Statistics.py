"""CSV-based regional statistics, independent of the neural-network runtime."""
import csv
import os

import numpy as np
import pandas as pd
import scipy.stats as ss

try:
    from ..Parameter import Data_root, Stats
except ImportError:
    from Parameter import Data_root, Stats


def _save_frame(frame, path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    frame.to_csv(path, index=False)


def _confidence(p, symbols=None):
    symbols = symbols or ['***', '**', '*', 'ns']
    if not np.isfinite(p):
        return 'NA'
    for threshold, symbol in zip([0.001, 0.01, 0.05], symbols):
        if p < threshold:
            return symbol
    return symbols[3]


def _read_group(root, names, filename, region_column, columns,
                region_num, regions=None):
    if not names:
        raise ValueError('Each comparison group must contain at least one sample.')
    arrays = []
    for name in names:
        path = os.path.join(root, name, 'whole_brain_cell_counts', filename)
        frame = pd.read_csv(path)
        needed = [region_column] + list(columns)
        missing = [column for column in needed if column not in frame]
        if missing:
            raise ValueError('{} is missing columns: {}'.format(path, missing))
        labels = frame[region_column]
        if labels.isnull().any() or labels.duplicated().any():
            raise ValueError('{} contains missing or duplicate region labels.'.format(path))
        if regions is None:
            regions = list(labels)
            if region_num is not None and len(regions) != region_num:
                raise ValueError('Expected {} regions, found {} in {}.'.format(region_num, len(regions), path))
        if set(labels) != set(regions):
            raise ValueError('Region labels do not match the reference sample: {}'.format(path))
        values = frame.set_index(region_column).loc[regions, columns].values.astype(float)
        arrays.append(values)
    return regions, np.stack(arrays, axis=1)


def _comparison(root, fs_name, ctrl_name, filename, region_column,
                columns, region_num):
    regions, experiment = _read_group(root, fs_name, filename, region_column,
                                      columns, region_num)
    _, control = _read_group(root, ctrl_name, filename, region_column,
                            columns, region_num, regions)
    return regions, experiment, control


def DNN_result_ranksum(data_path, model_name):
    # Read each model only once, while retaining the original ordered comparisons.
    scores = {name: pd.read_csv(os.path.join(data_path, str(name) + '_patch_results.csv'))['Dice'].values
              for name in model_name}
    with open(os.path.join(data_path, 'Dice_ranksums.txt'), 'w') as output:
        for first in model_name:
            for second in model_name:
                if first != second:
                    _, p = ss.ranksums(scores[first], scores[second])
                    # Retain the historical model-comparison marker convention.
                    if not np.isfinite(p):
                        marker = 'NA'
                    elif p <= 0.0001:
                        marker = '***'
                    elif p < 0.05:
                        marker = '**'
                    else:
                        marker = '*'
                    output.write('{} | {},{}\n'.format(marker, first, second))


def select_level(csv_path, save_path, level=6):
    frame = pd.read_csv(csv_path)
    columns = ['id', 'acronym', 'count', 'density', 'volume', 'path', 'level', 'name']
    selected = frame.loc[frame['level'] == level, columns]
    _save_frame(selected, os.path.join(save_path, 'cell-counts-level{}.csv'.format(level)))


def norm_data(data):
    values = np.asarray(data, dtype=float)
    if values.size == 0:
        return values.copy()
    minimum, maximum = np.min(values), np.max(values)
    if maximum == minimum:
        return np.zeros_like(values)
    return (values - minimum) / (maximum - minimum)


def _read_level6_counts(csv_path):
    frame = pd.read_csv(csv_path)
    if frame['acronym'].isnull().any():
        raise ValueError('Region acronyms must not be missing.')
    frame['count'] = pd.to_numeric(frame['count'], errors='raise')
    values = frame['count'].values
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError('Cell counts must be finite and nonnegative.')
    return frame


def level6_sum(csv_path, save_path):
    frame = _read_level6_counts(csv_path)
    counts = frame.groupby('acronym', sort=False)['count'].sum()
    result = pd.DataFrame({'Region': counts.index, 'counts': counts.values}, columns=['Region', 'counts'])
    _save_frame(result, os.path.join(save_path, 'cell-counts-level6-sum.csv'))


def level6_left_right_brain(csv_path, save_path):
    frame = _read_level6_counts(csv_path)
    rows = []
    for region, group in frame.groupby('acronym', sort=False):
        if len(group) != 2:
            raise ValueError('Expected two hemisphere rows for region {}.'.format(region))
        values = group['count'].values
        rows.append([region, values[0], values[1]])
    _save_frame(pd.DataFrame(rows, columns=['Region', 'left_counts', 'right_counts']),
                os.path.join(save_path, 'cell-counts-level6-left_right.csv'))


def correct_p_values(pvalues, method='BH'):
    """BH/FDR or Bonferroni adjustment; missing tests remain NaN.

    Finite tests form the correction family. Callers that treat missing tests
    conservatively can replace them with 1 before calling this function.
    """
    values = np.asarray(pvalues, dtype=float)
    if values.ndim != 1:
        raise ValueError('pvalues must be one-dimensional.')
    key = method.lower()
    if key not in ('bh', 'fdr', 'b', 'fwer'):
        raise ValueError('Unknown multiple-testing method: {}'.format(method))
    finite = np.isfinite(values)
    if np.any((values[finite] < 0) | (values[finite] > 1)):
        raise ValueError('Finite p-values must lie between 0 and 1.')
    result = np.full(values.shape, np.nan)
    valid = values[finite]
    n = valid.size
    if not n:
        return result
    if key in ('b', 'fwer'):
        result[finite] = np.minimum(valid * n, 1.0)
    else:
        order = np.argsort(valid)
        adjusted = valid[order] * n / np.arange(1, n + 1, dtype=float)
        adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
        corrected = np.empty(n, dtype=float)
        corrected[order] = np.minimum(adjusted, 1.0)
        result[finite] = corrected
    return result


def ttest_level6(root, fs_name, ctrl_name, region_num=205):
    regions, experiment, control = _comparison(root, fs_name, ctrl_name,
        'cell-counts-level6-sum.csv', 'Region', ['counts'], region_num)
    pvalues = ss.ttest_ind(experiment[:, :, 0], control[:, :, 0], axis=1, equal_var=True)[1]
    adjusted = correct_p_values(pvalues, method='FDR')
    symbols = Stats.get('symbol', ['***', '**', '*', 'ns'])
    result = pd.DataFrame({'Region': regions, 'p_val': adjusted,
        'Confidence level': [_confidence(p, symbols) for p in adjusted]},
        columns=['Region', 'p_val', 'Confidence level'])
    _save_frame(result, os.path.join(root, 'Hypothesis_test_level6.csv'))
    print('{} brain regions showed significant differences'.format(np.count_nonzero(adjusted < 0.05)))


def ttest_level6_left_right(root, fs_name, ctrl_name, region_num=205):
    regions, experiment, control = _comparison(root, fs_name, ctrl_name,
        'cell-counts-level6-left_right.csv', 'Region', ['left_counts', 'right_counts'], region_num)
    pvalues = ss.ttest_ind(experiment, control, axis=1, equal_var=True)[1]
    adjusted = np.column_stack([correct_p_values(pvalues[:, i], 'FDR') for i in range(2)])
    result = pd.DataFrame({'Region': regions,
        'fs_right_mean': experiment[:, :, 1].mean(axis=1),
        'fs_left_mean': experiment[:, :, 0].mean(axis=1),
        'ctrl_right_mean': control[:, :, 1].mean(axis=1),
        'ctrl_left_mean': control[:, :, 0].mean(axis=1),
        'right_Confidence level': adjusted[:, 1], 'left_Confidence level': adjusted[:, 0]},
        columns=['Region', 'fs_right_mean', 'fs_left_mean', 'ctrl_right_mean',
                 'ctrl_left_mean', 'right_Confidence level', 'left_Confidence level'])
    _save_frame(result, os.path.join(root, 'Hypothesis_test_level6_left_right.csv'))


def symbol_left_right_ci(csv_path, save_csv, select_left=True):
    frame = pd.read_csv(os.path.join(csv_path, 'Hypothesis_test_level6_left_right.csv'))
    side = 'left' if select_left else 'right'
    result = pd.DataFrame({'Region': frame['Region'],
        side.title() + ' Confidence level': [_confidence(p) for p in frame[side + '_Confidence level']]},
        columns=['Region', side.title() + ' Confidence level'])
    _save_frame(result, os.path.join(save_csv, 'Symbol_level6_' + side + '.csv'))


def ttest_multi_region_vol(root, fs_name, ctrl_name, region_num=43):
    regions, experiment, control = _comparison(root, fs_name, ctrl_name,
        'cortex_volume.csv', 'Region_name', ['Volume ratio %'], region_num)
    pvalues = ss.ttest_ind(experiment[:, :, 0], control[:, :, 0], axis=1, equal_var=True)[1]
    adjusted = correct_p_values(pvalues, 'FDR')
    result = pd.DataFrame({'Region': regions, 'Confidence level': [_confidence(p) for p in adjusted]},
                          columns=['Region', 'Confidence level'])
    _save_frame(result, os.path.join(root, 'Hypothesis_test_cortex_volume.csv'))
    print('{} brain regions showed significant differences'.format(np.count_nonzero(adjusted < 0.05)))


def get_multi_cl_counts(root, data_name, save_path, symbol, pvals_cutoff):
    if len(symbol) != len(pvals_cutoff):
        raise ValueError('Each significance symbol must have a matching cutoff.')
    frame = pd.read_csv(os.path.join(root, data_name, 'whole_brain_cell_counts', 'cell-counts-level6-sum.csv'))
    confidence = pd.read_csv(os.path.join(root, 'Hypothesis_test_level6.csv'))
    for label, cutoff in zip(symbol, pvals_cutoff):
        regions = confidence.loc[confidence['Confidence level'] == label, 'Region']
        result = frame.loc[frame['Region'].isin(regions), ['Region', 'counts']]
        _save_frame(result, os.path.join(save_path, 'Region_{}.csv'.format(cutoff)))


def get_plot_region_counts(root, data_name, plot_region_name, save_path, pvals_cutoff):
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, 'plot_region.csv'), 'w', newline='') as output:
        writer = csv.writer(output)
        for cutoff in pvals_cutoff:
            frame = pd.read_csv(os.path.join(root, data_name, 'whole_brain_cell_counts', 'csv', 'Region_{}.csv'.format(cutoff)))
            writer.writerow([cutoff])
            writer.writerow(['Region', 'counts'])
            selected = frame.loc[frame['Region'].isin(plot_region_name), ['Region', 'counts']]
            writer.writerows(selected.values)


def ttest_multi_layer(root, fs_name, ctrl_name, layer_name, save_layer_name,
                     stats_dtype, region_num=43, remove_nan=True):
    regions, experiment, control = _comparison(root, fs_name, ctrl_name,
        stats_dtype + '.csv', 'Region_name', [layer_name], region_num)
    pvalues = ss.ttest_ind(experiment[:, :, 0], control[:, :, 0], axis=1, equal_var=True)[1]
    if remove_nan:
        pvalues = np.where(np.isfinite(pvalues), pvalues, 1.0)
    adjusted = correct_p_values(pvalues, 'FDR')
    _save_frame(pd.DataFrame({'Region': regions, 'P value': adjusted}, columns=['Region', 'P value']),
        os.path.join(root, 'stats_layer', save_layer_name + stats_dtype + '_p_values.csv'))


def get_layer_ci(root, layer_name, save_layer_name, save_csv):
    frame = pd.read_csv(os.path.join(root, 'layer_p_value.csv'))
    classes = {'***': 0.001, '**': 0.01, '*': 0.05, 'ns': 0.1, 'NA': np.nan}
    result = pd.DataFrame({'Region': frame['Region'],
        'class': [classes[_confidence(p)] for p in frame[layer_name]]}, columns=['Region', 'class'])
    _save_frame(result, os.path.join(save_csv, save_layer_name + 'class.csv'))


def level2_count(csv_path, save_csv):
    frame = pd.read_csv(os.path.join(csv_path, 'level6_total_cell_count.csv'))
    samples = Stats['con_data_name'] + Stats['es_data_name'] + Stats['sds_data_name']
    if len(frame) < 200:
        raise ValueError('The configured level-2 atlas mapping requires at least 200 rows.')
    values = frame.loc[:, samples].values.astype(float)
    # Preserve the existing atlas row mapping; sample count is no longer fixed at 18.
    boundaries = [0, 17, 27, 29, 36, 48, 56, 69, 108, 134, 160, 196, 200]
    regions = ['Isocortex', 'OLF', 'HPF', 'CTXsp', 'STR', 'PAL', 'TH', 'HY', 'MB', 'P', 'MY', 'CBX']
    totals = np.vstack([values[start:end].sum(axis=0) for start, end in zip(boundaries[:-1], boundaries[1:])])
    result = pd.DataFrame(totals, columns=samples)
    result.insert(0, 'Region', regions)
    _save_frame(result, os.path.join(save_csv, 'level2_total_count.csv'))


if __name__ == "__main__":

    DNN_result_ranksum(data_path=Stats['DNN_results'], model_name=Stats['Model_name'])
    root = Data_root
    data_name = Stats['Pred_data_list']
    for i in range(0, len(data_name)):
        csv_path = os.path.join(root, data_name[i], 'whole_brain_cell_counts/cell-counting.csv')
        save_path = os.path.join(root, data_name[i], 'whole_brain_cell_counts')
        select_level(csv_path, save_path)
        print('finished {}'.format(data_name[i]))

    for i in range(len(data_name)):
        csv_path = os.path.join(root, data_name[i], 'whole_brain_cell_counts/cell-counts-level6.csv')
        save_path = os.path.join(root, data_name[i], 'whole_brain_cell_counts')
        level6_sum(csv_path, save_path)
        level6_left_right_brain(csv_path, save_path)
        print('finished {}'.format(data_name[i]))

    fs_name = Stats['FST_data_list']
    ctrl_name = Stats['Ctrl_data_list']
    ttest_level6(root, fs_name, ctrl_name)
    ttest_level6_left_right(root, fs_name, ctrl_name)
    symbol_left_right_ci(csv_path=root, save_csv=root, select_left=False)
    ttest_multi_region_vol(root, fs_name, ctrl_name)
    symbol = Stats['symbol']
    pvals_cutoff = Stats['pvals_cutoff']
    for i in range(len(data_name)):
        save_path = os.path.join(root, data_name[i], 'whole_brain_cell_counts/csv')
        os.makedirs(save_path,exist_ok=True) 
        get_multi_cl_counts(root, data_name[i], save_path, symbol, pvals_cutoff)
        print('finished {}'.format(data_name[i]))
    layer_name = Stats['layer_name']
    save_layer_name = Stats['save_layer_name']
    stats_dtype = Stats['stats_dtype']
    for ln in range(len(layer_name)):

        ttest_multi_layer(root, fs_name, ctrl_name, layer_name[ln], save_layer_name[ln], stats_dtype[2])
        get_layer_ci(root, layer_name[ln], save_layer_name[ln], save_csv=root)
        print(layer_name[ln])

    plot_region_name = Stats['display_region_name']
    for i in range(len(data_name)):
        save_path = os.path.join(root, data_name[i], 'whole_brain_cell_counts/csv')
        os.makedirs(save_path,exist_ok=True) 
        get_plot_region_counts(root, data_name[i], plot_region_name, save_path, pvals_cutoff)
        print('finished {}'.format(data_name[i])) 
    level2_count(csv_path=Data_root, save_csv=Data_root)
