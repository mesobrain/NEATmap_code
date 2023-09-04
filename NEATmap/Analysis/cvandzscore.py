from Environment import *
from Parameter import *
from scipy.stats import variation
def coefficient_of_variation(csv_path, save_path, index):

    fs_array = np.zeros((205, len(Stats['fst_data_name'])))
    ctrl_array = np.zeros_like(fs_array)
    for i in range(len(index)):
        fs_file = open(os.path.join(csv_path, 'cell-counts-level6-sum_es_' + index[i] + '.csv'))
        ctrl_file = open(os.path.join(csv_path, 'cell-counts-level6-sum_con_' + index[i] + '.csv'))
        fs = pd.read_csv(fs_file)
        ctrl = pd.read_csv(ctrl_file)
        fs_array[:, i] = fs['counts'].values
        ctrl_array[:, i] = ctrl['counts'].values
        region_name = fs['Region'].values
    
    # fs_mean = np.mean(fs_array, 1)
    # ctrl_mean = np.mean(ctrl_array, 1)
    # fs_std = np.std(fs_array, 1)
    # ctrl_std = np.std(ctrl_array, 1)
    # fs_cv = fs_std / fs_mean
    # ctrl_cv = ctrl_std / ctrl_mean
    fs_cv = np.zeros((fs_array.shape[0], 1))
    ctrl_cv = np.zeros((fs_array.shape[0], 1))
    for r in range(fs_array.shape[0]):
        fs_cv[r, :] = variation(fs_array[r, :])
        ctrl_cv[r, :] = variation(ctrl_array[r, :])

    savename = os.path.join(save_path, 'O_C_level6_CV.csv')
    values = pd.DataFrame(columns=['Region', 'SDS_CV', 'CON_CV'])
    values['Region'] = region_name
    values['SDS_CV'] = fs_cv
    values['CON_CV'] = ctrl_cv
    values.to_csv(savename, index=False)
    print('Finished')

def z_score(csv_path, save_path, index):

    fs_array = np.zeros((205, len(Stats['fst_data_name'])))
    ctrl_array = np.zeros_like(fs_array)
    fs_z_score = np.zeros_like(fs_array)
    ctrl_z_score = np.zeros_like(fs_array)

    for i in range(len(index)):
        fs_file = open(os.path.join(csv_path, 'cell-counts-level6-sum_sds_' + index[i] + '.csv'))
        ctrl_file = open(os.path.join(csv_path, 'cell-counts-level6-sum_con_' + index[i] + '.csv'))
        fs = pd.read_csv(fs_file)
        ctrl = pd.read_csv(ctrl_file)
        fs_array[:, i] = fs['counts'].values
        ctrl_array[:, i] = ctrl['counts'].values
        region_name = fs['Region'].values
    
    ctrl_mean = np.mean(ctrl_array, 1)
    ctrl_std = np.std(ctrl_array, 1)

    for j in range(len(index)):
        fs_z_score[:, j] = (fs_array[:, j] - ctrl_mean) / ctrl_std
        ctrl_z_score[:, j] = (ctrl_array[:, j] - ctrl_mean) / ctrl_std
    save_csv = os.path.join(save_path, 'I_C_level6_z_score.csv')
    list_name = ['Region'] + Stats['group_data_name']
    with open(save_csv, 'w+', newline='') as f:
        csv_writer = csv.writer(f, dialect='excel')
        csv_writer.writerow(list_name)
        for k in range(fs_array.shape[0]):
            write_lines = [region_name[k], ctrl_z_score[k, 0], ctrl_z_score[k, 1], ctrl_z_score[k, 2], ctrl_z_score[k, 3], 
                            ctrl_z_score[k, 4], ctrl_z_score[k, 5],
                            fs_z_score[k, 0], fs_z_score[k, 1], fs_z_score[k, 2], fs_z_score[k, 3],
                            fs_z_score[k, 4], fs_z_score[k, 5]]
            csv_writer.writerow(write_lines)
    print('Finished')

if __name__ == '__main__':

    csv_path = Stats['level6_cell_counts_path']
    index = Stats['group_index']
    coefficient_of_variation(csv_path=csv_path, save_path=csv_path, index=index)
    z_score(csv_path=csv_path, save_path=csv_path, index=index)