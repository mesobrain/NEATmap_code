import scipy.ndimage.measurements
from Environment import *
from scipy import stats
from Parameter import *
def position_stats(data_path, data_name):
    
    image = tifffile.imread(os.path.join(data_path, data_name))
    # z, _, _ = image.shape
    group_num = Stats['position_group']
    group1 = image[0 : group_num, :, :]
    group2 = image[group_num : 2*group_num, :, :]
    group3 = image[2*group_num : 3*group_num, :, :]
    group4 = image[3*group_num : 4*group_num, :, :]
    group5 = image[4*group_num : 5*group_num, :, :]
    group6 = image[5*group_num : 6*group_num, :, :]
    group7 = image[6*group_num : 7*group_num, :, :]
    group8 = image[7*group_num : 8*group_num, :, :]
    group9 = image[8*group_num : 9*group_num, :, :]
    group10 = image[9*group_num:, :, :]
    count1 = scipy.ndimage.measurements.sum(np.ones(group1.shape, dtype=bool), labels=group1, index=255);
    count2 = scipy.ndimage.measurements.sum(np.ones(group2.shape, dtype=bool), labels=group2, index=255);
    count3 = scipy.ndimage.measurements.sum(np.ones(group3.shape, dtype=bool), labels=group3, index=255);
    count4 = scipy.ndimage.measurements.sum(np.ones(group4.shape, dtype=bool), labels=group4, index=255);
    count5 = scipy.ndimage.measurements.sum(np.ones(group5.shape, dtype=bool), labels=group5, index=255);
    count6 = scipy.ndimage.measurements.sum(np.ones(group6.shape, dtype=bool), labels=group6, index=255);
    count7 = scipy.ndimage.measurements.sum(np.ones(group7.shape, dtype=bool), labels=group7, index=255);
    count8 = scipy.ndimage.measurements.sum(np.ones(group8.shape, dtype=bool), labels=group8, index=255);
    count9 = scipy.ndimage.measurements.sum(np.ones(group9.shape, dtype=bool), labels=group9, index=255);
    count10 = scipy.ndimage.measurements.sum(np.ones(group10.shape, dtype=bool), labels=group10, index=255);
    total = [count1, count2, count3, count4, count5, count6, count7, count8, count9, count10]
    print(total)
    num = len(total) / 2
    anterior = sum(total[0:num])
    posterior = sum(total[num:])
    return anterior, posterior

def t_test(fst, ctrl):
    t, p = stats.ttest_ind(fst, ctrl, axis=0, equal_var=True)
    print(p)

if __name__ == "__main__":
    fst_anterior = []
    fst_posterior = []
    ctrl_anterior = []
    ctrl_posterior = []
    for data_path in Stats['FST_data_list']:
        anterior, posterior = position_stats(data_path)
        fst_anterior.append(anterior)
        fst_posterior.append(posterior)
    for data_path in Stats['Ctrl_data_list']:
        anterior, posterior = position_stats(data_path)
        ctrl_anterior.append(anterior)
        ctrl_posterior.append(posterior)   

    t_test(fst_anterior, ctrl_anterior)
    t_test(fst_posterior, ctrl_posterior) 