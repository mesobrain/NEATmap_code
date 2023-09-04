from Environment import *
from Parameter import *

def DNN_result_ranksum(data_path, model_name):

    file = open(os.path.join(data_path, 'Dice_ranksums.txt'), 'w+')
    symbol = ['***', '**', '*']
    for i in model_name:
        for j in model_name:
            if i == j:
                pass
            else:
                f1 = open(os.path.join(data_path, str(i) + '_patch_results.csv'))
                f2 = open(os.path.join(data_path, str(j) + '_patch_results.csv'))
                df1 = pd.read_csv(f1)
                df2 = pd.read_csv(f2)
                stat, p = ss.ranksums(df1['Dice'], df2['Dice'])
                if p <=0.0001:
                    file.writelines(str(symbol[0]) + ' | ')
                    file.writelines(str(i) + ',' + str(j) + '\n')
                if 0.0001 < p < 0.05:
                    file.writelines(str(symbol[1]) + ' | ')
                    file.writelines(str(i) + ',' + str(j) + '\n')
                if p >= 0.05:
                    file.writelines(str(symbol[2]) + ' | ')
                    file.writelines(str(i) + ',' + str(j) + '\n')
    file.close()

def select_level(csv_path, save_path, level=6):

    file = open(csv_path)
    df = pd.read_csv(file, error_bad_lines=False)
    select_df = df[df['level'] == level]
    save_name = 'cell-counts-level6.csv'
    savepath = os.path.join(save_path, save_name)
    savecsv = pd.DataFrame(columns=['id', 'acronym', 'count', 'density', 'volume', 'path', 'level', 'name'])
    savecsv['id'] = select_df['id']
    savecsv['acronym'] = select_df['acronym']
    savecsv['count'] = select_df['count']
    savecsv['density'] = select_df['density']
    savecsv['volume'] = select_df['volume']
    savecsv['path'] = select_df['path']
    savecsv['level'] = select_df['level']
    savecsv['name'] = select_df['name']
    savecsv.to_csv(savepath, index=False)

def norm_data(data):
    data1 = data - np.min(data) 
    data = data1 * 1.0 / (np.max(data) - np.min(data) )

    return data
def level6_sum(csv_path, save_path):

    file = open(csv_path)
    df = pd.read_csv(file, error_bad_lines=False)
    region_name = df['acronym'].values
    region_name = list(region_name)
    region_name = sorted(set(region_name), key=region_name.index)
    counts = df['count'].values
    sum_counts = []
    index = 0
    for i in range(len(counts)):
        values = counts[index : index + 2]
        index = 2*(i+1)
        if i < len(counts) // 2:
            sum_counts.append(sum(values))
        else:
            break
    
    # sum_counts = norm_data(sum_counts)
    # sum_counts = [k / 1078 for k in sum_counts]
    savename = 'cell-counts-level6-sum.csv'
    savepath = os.path.join(save_path, savename)
    savecsv = pd.DataFrame(columns=['Region', 'counts'])
    savecsv['Region'] = region_name
    savecsv['counts'] = sum_counts
    savecsv.to_csv(savepath, index=False)

def level6_left_right_brain(csv_path, save_path):

    file = open(csv_path)
    df = pd.read_csv(file, error_bad_lines=False)
    region_name = df['acronym'].values
    region_name = list(region_name)
    region_name = sorted(set(region_name), key=region_name.index)
    counts = df['count'].values
    left_counts = []
    right_counts = []
    index = 0
    for i in range(len(counts)):
        values = counts[index : index + 2]
        index = 2*(i+1)
        if i < len(counts) // 2:
            left_count = values[0]
            right_count = values[1]
            left_counts.append(left_count)
            right_counts.append(right_count)
        else:
            break
    savename = 'cell-counts-level6-left_right.csv'
    savepath = os.path.join(save_path, savename)
    savecsv = pd.DataFrame(columns=['Region', 'left_counts', 'right_counts'])
    savecsv['Region'] = region_name
    savecsv['left_counts'] = left_counts
    savecsv['right_counts'] = right_counts
    savecsv.to_csv(savepath, index=False)

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

def ttest_level6(root, fs_name, ctrl_name, region_num=205):
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
    for rn in range(region_num):
        tvals, pvals = ss.ttest_ind(fs_array[rn, :], ctrl_array[rn, :], axis=0, equal_var=True)
        pval_list.append(pvals)
    pvals_corrected = correct_p_values(pval_list, method='FDR')
    csv_name = os.path.join(root, 'Hypothesis_test_level6.csv')
    first_line = ['Region', 'p_val', 'Confidence level']
    significant_num = 0
    symbol = Stats['symbol']
    with open(csv_name, 'w+', newline='') as f:
        csv_write = csv.writer(f, dialect='excel')
        csv_write.writerow(first_line)
        for pc in range(len(pvals_corrected)):
            if pvals_corrected[pc] < 0.001:
                write_lines = [region_name[pc], pvals_corrected[pc], str(symbol[0])]
                csv_write.writerow(write_lines)
                significant_num += 1
            if 0.001 < pvals_corrected[pc] < 0.01:
                write_lines = [region_name[pc], pvals_corrected[pc], str(symbol[1])]
                csv_write.writerow(write_lines)
                significant_num += 1
            if 0.01 < pvals_corrected[pc] < 0.05:
                write_lines = [region_name[pc], pvals_corrected[pc], str(symbol[2])]
                csv_write.writerow(write_lines)
                significant_num += 1
            if pvals_corrected[pc] >= 0.05:
                write_lines = [region_name[pc], pvals_corrected[pc], str(symbol[3])]
                csv_write.writerow(write_lines)

    print('{} brain regions showed significant differences'.format(significant_num))

def ttest_level6_left_right(root, fs_name, ctrl_name, region_num=205):
    fs_left_array = np.zeros((region_num, len(fs_name))).astype('int')
    ctrl_left_array = np.zeros((region_num, len(ctrl_name))).astype('int')
    fs_right_array = np.zeros((region_num, len(fs_name))).astype('int')
    ctrl_right_array = np.zeros((region_num, len(ctrl_name))).astype('int')
    for i in range(len(fs_name)):
        fs_csv_path = os.path.join(root, fs_name[i], 'whole_brain_cell_counts/cell-counts-level6-left_right.csv')
        ctrl_csv_path = os.path.join(root, ctrl_name[i], 'whole_brain_cell_counts/cell-counts-level6-left_right.csv')
        fs_file, ctrl_file = open(fs_csv_path), open(ctrl_csv_path)
        fs_df, ctrl_df = pd.read_csv(fs_file), pd.read_csv(ctrl_file)
        fs_left_count, ctrl_left_count = fs_df['left_counts'].values, ctrl_df['left_counts'].values
        fs_right_count, ctrl_right_count = fs_df['right_counts'].values, ctrl_df['right_counts'].values
        region_name = fs_df['Region'].values
        fs_left_array[:, i] = fs_left_count
        ctrl_left_array[:, i] = ctrl_left_count
        fs_right_array[:, i] = fs_right_count
        ctrl_right_array[:, i] = ctrl_right_count
    
    fs_left_mean = np.mean(fs_left_array, 1)
    fs_right_mean = np.mean(fs_right_array, 1)
    ctrl_left_mean = np.mean(ctrl_left_array, 1)
    ctrl_right_mean = np.mean(ctrl_right_array, 1)
    left_pval_list = []
    right_pval_list = []
    for rn in range(region_num):
        left_tvals, left_pvals = ss.ttest_ind(fs_left_array[rn, :], ctrl_left_array[rn, :], axis=0, equal_var=True)
        left_pval_list.append(left_pvals)
        right_tvals, right_pvals = ss.ttest_ind(fs_right_array[rn, :], ctrl_right_array[rn, :], axis=0, equal_var=True)
        right_pval_list.append(right_pvals)
    left_pvals_corrected = correct_p_values(left_pval_list, method='FDR')
    right_pvals_corrected = correct_p_values(right_pval_list, method='FDR')
    csv_name = os.path.join(root, 'Hypothesis_test_level6_left_right.csv')

    first_line = ['Region', 'fs_right_mean', 'fs_left_mean', 'ctrl_right_mean', 'ctrl_left_mean', 'right_Confidence level',
                 'left_Confidence level']

    with open(csv_name, 'w+', newline='') as f:
        csv_write = csv.writer(f, dialect='excel')
        csv_write.writerow(first_line)
        for pc in range(len(left_pvals_corrected)):
            
            write_lines = [region_name[pc], fs_right_mean[pc], fs_left_mean[pc], ctrl_right_mean[pc], ctrl_left_mean[pc],
                            right_pvals_corrected[pc], left_pvals_corrected[pc]]
            csv_write.writerow(write_lines)

    print('Finished')

def symbol_left_right_ci(csv_path, save_csv, select_left=True):
    file = open(os.path.join(csv_path, 'Hypothesis_test_level6_left_right.csv'))
    df = pd.read_csv(file)
    region_name = df['Region'].values
    if select_left:
        left_p_correct = df['left_Confidence level'].values
        pvals_corrected = list(left_p_correct)
        first_line = ['Region', 'Left Confidence level']
        csv_name = os.path.join(save_csv, 'Symbol_level6_left.csv')
    else:
        right_p_correct = df['right_Confidence level'].values
        pvals_corrected = list(right_p_correct)
        first_line = ['Region', 'Right Confidence level']
        csv_name = os.path.join(save_csv, 'Symbol_level6_right.csv')
    symbol = ['***', '**', '*', 'ns'] # ***:p<0.001, **:p<0.01, *:p<0.05, ns(no significant): p>=0.05
    significant_num = 0
    with open(csv_name, 'w+', newline='') as f:
        csv_write = csv.writer(f, dialect='excel')
        csv_write.writerow(first_line)
        for pc in range(len(pvals_corrected)):
            if pvals_corrected[pc] < 0.001:
                write_lines = [region_name[pc], str(symbol[0])]
                csv_write.writerow(write_lines)
                significant_num += 1
            if 0.001 < pvals_corrected[pc] < 0.01:
                write_lines = [region_name[pc], str(symbol[1])]
                csv_write.writerow(write_lines)
                significant_num += 1
            if 0.01 < pvals_corrected[pc] < 0.05:
                write_lines = [region_name[pc], str(symbol[2])]
                csv_write.writerow(write_lines)
                significant_num += 1
            if pvals_corrected[pc] >= 0.05:
                write_lines = [region_name[pc], str(symbol[3])]
                csv_write.writerow(write_lines)

    print('{} brain regions showed significant differences'.format(significant_num))

def ttest_multi_region_vol(root, fs_name, ctrl_name, region_num=43):

    fs_array = np.zeros((region_num, len(fs_name))).astype('int')
    ctrl_array = np.zeros((region_num, len(ctrl_name))).astype('int')
    for i in range(len(fs_name)):
        fs_csv_path = os.path.join(root, fs_name[i], 'whole_brain_cell_counts/cortex_volume.csv')
        ctrl_csv_path = os.path.join(root, ctrl_name[i], 'whole_brain_cell_counts/cortex_volume.csv')
        fs_file, ctrl_file = open(fs_csv_path), open(ctrl_csv_path)
        fs_df, ctrl_df = pd.read_csv(fs_file), pd.read_csv(ctrl_file)
        fs_count, ctrl_count = fs_df['Volume ratio %'].values, ctrl_df['Volume ratio %'].values
        region_name = fs_df['Region_name'].values
        fs_array[:, i] = fs_count
        ctrl_array[:, i] = ctrl_count
    pval_list = []
    for rn in range(region_num):
        tvals, pvals = ss.ttest_ind(fs_array[rn, :], ctrl_array[rn, :], axis=0, equal_var=True)
        pval_list.append(pvals)
    pvals_corrected = correct_p_values(pval_list, method='FDR')
    csv_name = os.path.join(root, 'Hypothesis_test_cortex_volume.csv')
    symbol = ['***', '**', '*', 'ns'] # ***:p<0.001, **:p<0.01, *:p<0.05, ns(no significant): p>=0.05
    first_line = ['Region', 'Confidence level']
    significant_num = 0
    with open(csv_name, 'w+', newline='') as f:
        csv_write = csv.writer(f, dialect='excel')
        csv_write.writerow(first_line)
        for pc in range(len(pvals_corrected)):
            if pvals_corrected[pc] < 0.001:
                write_lines = [region_name[pc], str(symbol[0])]
                csv_write.writerow(write_lines)
                significant_num += 1
            if 0.001 < pvals_corrected[pc] < 0.01:
                write_lines = [region_name[pc], str(symbol[1])]
                csv_write.writerow(write_lines)
                significant_num += 1
            if 0.01 < pvals_corrected[pc] < 0.05:
                write_lines = [region_name[pc], str(symbol[2])]
                csv_write.writerow(write_lines)
                significant_num += 1
            if pvals_corrected[pc] >= 0.05:
                write_lines = [region_name[pc], str(symbol[3])]
                csv_write.writerow(write_lines)

    print('{} brain regions showed significant differences'.format(significant_num))

def get_multi_cl_counts(root, data_name, save_path, symbol, pvals_cutoff):

    file = open(os.path.join(root, data_name, 'whole_brain_cell_counts/cell-counts-level6-sum.csv'))
    cl_file = open(os.path.join(root, 'Hypothesis_test_level6.csv'))
    df = pd.read_csv(file)
    cl_df = pd.read_csv(cl_file)
    for s in range(len(symbol)):
        select_cl_df = cl_df[cl_df['Confidence level'] == symbol[s]]
        select_region = select_cl_df['Region'].values
        Region = df['Region'].values
        counts = df['counts'].values
        select_region_list = []
        select_counts_list = []

        for i in range(len(Region)):

            if Region[i] in select_region:
                select_region_list.append(Region[i])
                select_counts_list.append(counts[i])
            else:
                pass
        csv_name = os.path.join(save_path, 'Region_' + str(pvals_cutoff[s]) + '.csv')
        savecsv = pd.DataFrame(columns=['Region', 'counts'])
        savecsv['Region'] = select_region_list
        savecsv['counts'] = select_counts_list
        savecsv.to_csv(csv_name, index=False)

def get_plot_region_counts(root, data_name, plot_region_name, save_path, pvals_cutoff):

    csv_name = os.path.join(save_path, 'plot_region.csv')
    lines = ['Region', 'counts']
    with open(csv_name, 'w+', newline='') as f:
        csv_writer = csv.writer(f, dialect='excel')
        for num in range(len(pvals_cutoff)):
            file = open(os.path.join(root, data_name, 'whole_brain_cell_counts/csv', 'Region_' + pvals_cutoff[num] + '.csv'))
            df = pd.read_csv(file)
            region_name = df['Region'].values
            counts = df['counts'].values
            csv_writer.writerow(pvals_cutoff[num])
            csv_writer.writerow(lines)
            for rn in range(len(region_name)):
                if region_name[rn] in plot_region_name:              
                    write_lines = [region_name[rn], counts[rn]]
                    csv_writer.writerow(write_lines)


def ttest_multi_layer(root, fs_name, ctrl_name, layer_name, save_layer_name, stats_dtype, region_num=43, remove_nan=True):

    fs_array = np.zeros((region_num, len(fs_name))).astype('int')
    ctrl_array = np.zeros((region_num, len(ctrl_name))).astype('int')
    for i in range(len(fs_name)):
        fs_csv_path = os.path.join(root, fs_name[i], 'whole_brain_cell_counts', stats_dtype + '.csv')
        ctrl_csv_path = os.path.join(root, ctrl_name[i], 'whole_brain_cell_counts', stats_dtype + '.csv')
        fs_file, ctrl_file = open(fs_csv_path), open(ctrl_csv_path)
        fs_df, ctrl_df = pd.read_csv(fs_file), pd.read_csv(ctrl_file)
        fs_count, ctrl_count = fs_df[layer_name].values, ctrl_df[layer_name].values
        region_name = fs_df['Region_name'].values
        fs_array[:, i] = fs_count
        ctrl_array[:, i] = ctrl_count
    pval_list = []
    for rn in range(region_num):
        tvals, pvals = ss.ttest_ind(fs_array[rn, :], ctrl_array[rn, :], axis=0, equal_var=True)
        pval_list.append(pvals)

    if remove_nan: 
        pval_list_rn = [1.0 if math.isnan(p) else p for p in pval_list]

    pvals_corrected = correct_p_values(pval_list_rn, method='FDR')
    csv_name = os.path.join(root, 'stats_layer', save_layer_name + stats_dtype + '_p_values.csv')
    first_line = ['Region', 'P value']
    with open(csv_name, 'w+', newline='') as f:
        csv_write = csv.writer(f, dialect='excel')
        csv_write.writerow(first_line)
        for pc in range(len(pvals_corrected)):
            write_lines = [region_name[pc], pvals_corrected[pc]]
            csv_write.writerow(write_lines)

def get_layer_ci(root, layer_name, save_layer_name, save_csv):
    class_index = [0.001, 0.01, 0.05, 0.1]
    symbol_list = []
    file = open(os.path.join(root, 'layer_p_value.csv'))
    df = pd.read_csv(file)
    p_val = df[layer_name].values
    region_name = df['Region'].values
    p_val_list = list(p_val)
    for p in p_val_list:
        if p < 0.001:
            symbol_list.append(class_index[0])
        if 0.001 < p < 0.01:
            symbol_list.append(class_index[1])
        if 0.01 < p < 0.05:
            symbol_list.append(class_index[2])
        if p >= 0.05:
            symbol_list.append(class_index[3])
    csv_name = os.path.join(save_csv, save_layer_name + 'class' + '.csv')
    savecsv = pd.DataFrame(columns=['Region', 'class'])
    savecsv['Region'] = region_name
    savecsv['class'] = symbol_list
    savecsv.to_csv(csv_name, index=False)

def level2_count(csv_path, save_csv):
    file = open(os.path.join(csv_path, 'level6_total_cell_count.csv'))
    df = pd.read_csv(file)
    Ctrl = {}
    Fst = {}
    Sds = {}
    for name in Stats['con_data_name']:
        Ctrl[name] = df[name].values
    for name in Stats['es_data_name']:
        Fst[name] = df[name].values
    for name in Stats['sds_data_name']:
        Sds[name] = df[name].values
    num = len(Stats['con_data_name'] + Stats['es_data_name'] + Stats['sds_data_name'])
    total = np.zeros((205, num)).astype(int)
    for i in range(len(Stats['con_data_name'])):
        total[:, i] = Ctrl[Stats['con_data_name'][i]]
    for j in range(len(Stats['es_data_name'])):
        total[:, j + len(Stats['con_data_name'])] = Fst[Stats['es_data_name'][j]]
    for k in range(len(Stats['sds_data_name'])):
        total[:, k + len(Stats['con_data_name']) + len(Stats['es_data_name'])] = Sds[Stats['sds_data_name'][k]]

    Isocortex = np.sum(total[0:17, :], axis=0)
    Olf = np.sum(total[17:27, :], axis=0)
    Hpf = np.sum(total[27:29, :], axis=0)
    Ctexsp = np.sum(total[29:36, :], axis=0)
    Str = np.sum(total[36:48, :], axis=0)
    Pal = np.sum(total[48:56, :], axis=0)
    Th = np.sum(total[56:69, :], axis=0)
    Hy = np.sum(total[69:108, :], axis=0)
    Mb = np.sum(total[108:134, :], axis=0)
    p = np.sum(total[134:160, :], axis=0)
    My = np.sum(total[160:196, :], axis=0)
    Cbx = np.sum(total[196:200, :], axis=0)
    total_level = np.zeros((12, num))
    total_level[0,:] = Isocortex
    total_level[1,:] = Olf
    total_level[2,:] = Hpf
    total_level[3,:] = Ctexsp
    total_level[4,:] = Str
    total_level[5,:] = Pal
    total_level[6,:] = Th
    total_level[7,:] = Hy
    total_level[8,:] = Mb
    total_level[9,:] = p
    total_level[10,:] = My
    total_level[11,:] = Cbx
    Region = ['Isocortex', 'OLF', 'HPF', 'CTXsp', 'STR', 'PAL', 'TH', 'HY', 'MB', 'P', 'MY', 'CBX']
    csv_name = os.path.join(save_csv, 'level2_total_count.csv')
    first_line = ['Region'] + Stats['con_data_name'] + Stats['es_data_name'] + Stats['sds_data_name']
    with open(csv_name, 'w+', newline='') as f:
        csv_write = csv.writer(f, dialect='excel')
        csv_write.writerow(first_line)
        for r in range(len(Region)):
            write_lines = [Region[r], total_level[r,0], total_level[r,1], total_level[r,2], total_level[r,3], total_level[r,4], total_level[r,5], total_level[r,6],
                            total_level[r,7], total_level[r,8], total_level[r,9], total_level[r,10], total_level[r,11], total_level[r,12], total_level[r,13],
                            total_level[r,14], total_level[r,15], total_level[r,16], total_level[r,17]]
            csv_write.writerow(write_lines)

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