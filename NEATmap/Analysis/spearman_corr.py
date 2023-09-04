import os, csv
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

def spearman(data, ctrl, region_number):
    sds_counts = data[region_number].values
    ctrl_counts = ctrl[region_number].values
    sds_rank = pd.Series(sds_counts).rank()
    ctrl_rank = pd.Series(ctrl_counts).rank()
    stat, _ = ss.spearmanr(sds_rank, ctrl_rank)
    _, pvals = ss.ttest_ind(sds_counts, ctrl_counts, equal_var=True)
    return stat, pvals

def spearman_corr(data_root, save_root):
    data = pd.read_csv(os.path.join(data_root, 'SDS_pearson_data.csv'))
    ctrl = pd.read_csv(os.path.join(data_root, 'ES_pearson_data.csv'))
    column_names = list(data.columns)
    corr = []
    group = []
    pvalue = []
    pos_neg = []
    for i in range(len(column_names)):
        stat, pval = spearman(data,ctrl, column_names[i])
        corr.append(stat)
        pvalue.append(pval)
    pvalue = correct_p_values(pvalue, method='FDR')
    csv_name = os.path.join(save_root, 'sd_es_corr.csv')
    first_line = ['Region', 'corr', 'group', 'pvalue', 'neg_pos']
    for co in range(len(corr)):
        if corr[co] > 0:
            pos_neg.append('Pos')
        elif corr[co] < 0:
            pos_neg.append('Neg')

    with open(csv_name, 'w+', newline='') as f:
        csv_write = csv.writer(f, dialect='excel')
        csv_write.writerow(first_line)
        for c in range(len(corr)):
            if pvalue[c] < 0.001:
                write_line = [column_names[c], corr[c], 'SDS vs ES', '<0.001', pos_neg[c]]
            elif 0.001 < pvalue[c] < 0.01:
                write_line = [column_names[c], corr[c], 'SDS vs ES', '<0.01', pos_neg[c]]
            elif 0.01 < pvalue[c] < 0.05:
                write_line = [column_names[c], corr[c], 'SDS vs ES', '<0.05', pos_neg[c]]
            elif pvalue[c] >= 0.05:
                write_line = [column_names[c], corr[c], 'SDS vs ES', '>0.05', pos_neg[c]]
            csv_write.writerow(write_line)

    print('finished')

if __name__ == "__main__":
    data_root = ''
    spearman_corr(data_root, data_root)