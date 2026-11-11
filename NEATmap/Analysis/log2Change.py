"""Two-group regional fold changes with raw and FDR-adjusted p-values."""

import csv
import os

import numpy as np
import pandas as pd
import scipy.stats as ss


def correct_p_values(pvalues, method='BH'):
    """Apply BH/FDR or Bonferroni correction to finite tests.

    Missing (NaN) tests remain NaN and are excluded from the tested family.
    Input and output retain the same one-dimensional order.
    """
    values = np.asarray(pvalues, dtype=float)
    if values.ndim != 1:
        raise ValueError('pvalues must be one-dimensional')
    finite = np.isfinite(values)
    if np.isinf(values).any() or ((values[finite] < 0) | (values[finite] > 1)).any():
        raise ValueError('pvalues must be between 0 and 1, or NaN')
    if not isinstance(method, str):
        raise ValueError('method must be BH/FDR or Bonferroni')
    method = method.lower()
    if method not in ('bh', 'fdr', 'benjamini-hochberg', 'b', 'fwer', 'bonferroni', 'bonferoni'):
        raise ValueError('Unknown p-value correction method: {}'.format(method))
    result = np.full(values.shape, np.nan, dtype=float)
    valid = values[finite]
    n_tests = len(valid)
    if not n_tests:
        return result
    if method in ('bh', 'fdr', 'benjamini-hochberg'):
        order = np.argsort(valid)
        adjusted = valid[order] * n_tests / np.arange(1, n_tests + 1, dtype=float)
        adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
        restored = np.empty_like(adjusted)
        restored[order] = np.minimum(adjusted, 1.0)
        result[finite] = restored
    else:
        result[finite] = np.minimum(valid * n_tests, 1.0)
    return result


def _read_group(root, sample_names, regions=None):
    sample_names = list(sample_names)
    if not sample_names:
        raise ValueError('Each group must contain at least one sample')
    columns = []
    for name in sample_names:
        path = os.path.join(root, name, 'whole_brain_cell_counts', 'cell-counts-level6-sum.csv')
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
        columns.append(counts)
    return regions, np.column_stack(columns)


def log2change(root, fs_name, ctrl_name, region_num=205):
    """Compare two independent groups using the existing equal-variance t-test.

    ``region_num=None`` infers the region count. Zero means retain the usual
    extended log-ratio values (NaN or +/-inf), without adding a pseudocount.
    ``pValue`` and the legacy label threshold retain their raw-p semantics;
    ``qValue`` additionally exposes the FDR correction.
    """
    regions, treatment = _read_group(root, fs_name)
    _, control = _read_group(root, ctrl_name, regions)
    if region_num is not None:
        if (not isinstance(region_num, (int, np.integer)) or region_num <= 0
                or region_num != len(regions)):
            raise ValueError('region_num must match the number of CSV regions, or be None')
    _, pvalues = ss.ttest_ind(treatment, control, axis=1, equal_var=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        foldchange = np.log2(np.mean(treatment, axis=1) / np.mean(control, axis=1))
    qvalues = correct_p_values(pvalues, method='FDR')
    labels = [region if np.isfinite(pvalue) and pvalue <= 0.01 else 'NA'
              for region, pvalue in zip(regions, pvalues)]
    csv_name = os.path.join(root, 'O_I_difference_level6.csv')
    with open(csv_name, 'w', newline='') as handle:
        writer = csv.writer(handle, dialect='excel')
        writer.writerow(['Region', 'log2FoldChange', 'pValue', 'label', 'qValue'])
        writer.writerows(zip(regions, foldchange, pvalues, labels, qvalues))
    print('Finished')

if __name__ == "__main__":
    root = ''
    exp_name = ['Model_SLJ_MHW_I3_8_1_seg_pre', 'Model_SLJ_WH_I4_11_1_seg_pre', 'Model_SLJ_WH_I5_14_1_seg_pre',
                            'Model_GLX_MHW_I6_17_1_seg_pre', 'Model_GLX_MHW_I7_20_1_seg_pre', 'Model_GLX_MHW_I8_23_1_seg_pre']
    obser_name = ['Model_GLX_MHW_O3_9_1_seg_pre', 'Model_GLX_MHW_O4_12_1_seg_pre', 'Model_GLX_MHW_O5_15_1_seg_pre',
                                'Model_GLX_MHW_O6_18_1_seg_pre', 'Model_SLJ_MHW_O7_21_1_seg_pre', 'Model_GLX_MHW_O8_24_1_seg_pre']
    con_name = ['Model_GLX_MHW_C3_7_1_seg_pre', 'Model_SLJ_WH_C4_10_1_seg_pre', 'Model_GLX_MHW_C5_13_1_seg_pre',
                            'Model_GLX_MHW_C6_16_1_seg_pre', 'Model_GLX_MHW_C7_19_1_seg_pre', 'Model_SLJ_MHW_C8_22_1_seg_pre']

    log2change(root, obser_name, exp_name)
