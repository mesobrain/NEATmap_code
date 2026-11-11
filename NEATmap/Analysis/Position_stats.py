import os

import numpy as np
import tifffile
from scipy import stats
from Parameter import Stats
def position_stats(data_path, data_name):
    
    image = tifffile.imread(os.path.join(data_path, data_name))
    if image.ndim != 3:
        raise ValueError('Position statistics require a three-dimensional image')
    group_num = Stats['position_group']
    if not isinstance(group_num, (int, np.integer)) or group_num <= 0:
        raise ValueError("Stats['position_group'] must be a positive integer")
    # Keep the original ten slabs, with all remaining slices in the last slab.
    total = [np.count_nonzero(image[i * group_num:(i + 1) * group_num] == 255)
             for i in range(9)]
    total.append(np.count_nonzero(image[9 * group_num:] == 255))
    print(total)
    num = len(total) // 2
    anterior = sum(total[0:num])
    posterior = sum(total[num:])
    return anterior, posterior

def t_test(fst, ctrl):
    t, p = stats.ttest_ind(fst, ctrl, axis=0, equal_var=True)
    print(p)

if __name__ == "__main__":
    from Parameter import Data_root

    position_name = Stats.get('position_name')
    if not position_name:
        raise ValueError("Set Stats['position_name'] to the input TIFF filename")
    fst_anterior = []
    fst_posterior = []
    ctrl_anterior = []
    ctrl_posterior = []
    for data_path in Stats['FST_data_list']:
        anterior, posterior = position_stats(os.path.join(Data_root, data_path), position_name)
        fst_anterior.append(anterior)
        fst_posterior.append(posterior)
    for data_path in Stats['Ctrl_data_list']:
        anterior, posterior = position_stats(os.path.join(Data_root, data_path), position_name)
        ctrl_anterior.append(anterior)
        ctrl_posterior.append(posterior)   

    t_test(fst_anterior, ctrl_anterior)
    t_test(fst_posterior, ctrl_posterior)
