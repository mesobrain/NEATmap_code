import os, csv, math
import scipy.stats as ss
import pandas as pd
import numpy as np

def correct_p_values(pvalues, method = 'BH'):
  """Corrects p-values for multiple testing using various methods. 
  
  Arguments
  ---------
  pvalues : array
    List of p values to be corrected.
  method : str 
    Optional method to use: 'BH' = 'FDR' = 'Benjamini-Hochberg', 
    'B' = 'FWER' = 'Bonferoni'.
  
  Returns
  -------
  qvalues : array
    Corrected p values.
  
  References
  ----------
  - `Benjamini Hochberg, 1995 <http://www.jstor.org/stable/2346101?seq=1#page_scan_tab_contents>`_
  - `Bonferoni correction <http://www.tandfonline.com/doi/abs/10.1080/01621459.1961.10482090#.VmHWUHbH6KE>`_
  - `R statistics package <https://www.r-project.org/>`_
  
  Notes
  -----
  Modified from http://statsmodels.sourceforge.net/ipdirective/generated/scikits.statsmodels.sandbox.stats.multicomp.multipletests.html.
  """
  
  pvals = np.asarray(pvalues);

  if method.lower() in ['bh', 'fdr']:
    pvals_sorted_ids = np.argsort(pvals);
    pvals_sorted = pvals[pvals_sorted_ids]
    sorted_ids_inv = pvals_sorted_ids.argsort()

    n = len(pvals);
    bhfactor = np.arange(1,n+1)/float(n);

    pvals_corrected_raw = pvals_sorted / bhfactor;
    pvals_corrected = np.minimum.accumulate(pvals_corrected_raw[::-1])[::-1]
    pvals_corrected[pvals_corrected>1] = 1;

    return pvals_corrected[sorted_ids_inv];
  
  elif method.lower() in ['b', 'fwer']:
    n = len(pvals);        
    
    pvals_corrected = n * pvals;
    pvals_corrected[pvals_corrected>1] = 1;
    
    return pvals_corrected;

def log2change(root, fs_name, ctrl_name, region_num=205):
    fs_array = np.zeros((region_num, len(fs_name))).astype('int')
    ctrl_array = np.zeros((region_num, len(ctrl_name))).astype('int')
    for i in range(len(fs_name)):
        fs_csv_path = os.path.join(root, fs_name[i], 'whole_brain_cell_counts/cell-counts-level6-sum.csv')
        ctrl_csv_path = os.path.join(root, ctrl_name[i], 'whole_brain_cell_counts/cell-counts-level6-sum.csv')
        fs_file, ctrl_file = open(fs_csv_path), open(ctrl_csv_path)
        fs_df, ctrl_df = pd.read_csv(fs_file), pd.read_csv(ctrl_file)
        fs_count, ctrl_count = fs_df['counts'].values, ctrl_df['counts'].values
        region_name = fs_df['Region'].values
        fs_array[:, i] = fs_count
        ctrl_array[:, i] = ctrl_count
    pval_list = []
    foldchange = []
    for rn in range(region_num):
        tvals, pvals = ss.ttest_ind(fs_array[rn, :], ctrl_array[rn, :], axis=0, equal_var=True)
        log2foldchange = math.log2(np.mean(fs_array[rn, :]) / np.mean(ctrl_array[rn, :]))
        pval_list.append(pvals)
        foldchange.append(log2foldchange)
    pvals_corrected = correct_p_values(pval_list, method='FDR')
    label = [0 for c in range(len(pvals_corrected))]
    for p in range(len(pvals_corrected)):
        if pval_list[p] > 0.01:
            label[p] = 'NA'
        else:
            label[p] = region_name[p]
    csv_name = os.path.join(root, 'O_I_difference_level6.csv')
    first_line = ['Region', 'log2FoldChange', 'pValue', 'label']
    with open(csv_name, 'w+', newline='') as f:
        csv_write = csv.writer(f, dialect='excel')
        csv_write.writerow(first_line)
        for pc in range(len(pvals_corrected)):
            write_lines = [region_name[pc], foldchange[pc], pval_list[pc], label[pc]]
            csv_write.writerow(write_lines)

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