import csv
import os

import numpy as np
import pandas as pd
import tifffile


def _read_region_rows(csv_path, region_names, name_column):
    """Align named CSV rows; retain positional support for legacy CSV files."""
    frame = pd.read_csv(csv_path)
    if name_column in frame.columns:
        if frame[name_column].duplicated().any():
            raise ValueError('Duplicate region names in {}'.format(csv_path))
        frame = frame.set_index(name_column)
        missing = [name for name in region_names if name not in frame.index]
        if missing:
            raise ValueError('Missing regions in {}: {}'.format(csv_path, missing))
        return frame.loc[region_names]
    if len(frame) != len(region_names):
        raise ValueError('Region count does not match {}'.format(csv_path))
    return frame


def get_density_csv(region_path, counts_csv, region_color, save_csv):
    '''cell counts/region pixel nums'''
    region_name = list(region_color.keys())
    df = _read_region_rows(os.path.join(counts_csv, 'cortex_cells_counts.csv'),
                           region_name, 'Region')
    counts = df['SUM'].values
    os.makedirs(save_csv or os.curdir, exist_ok=True)
    csv_save = os.path.join(save_csv, 'cortex_density.csv')
    first_lines = ['Region', 'Density']
    with open(csv_save, 'w+', newline='') as f:
        csv_write = csv.writer(f, dialect='excel')
        csv_write.writerow(first_lines)
        for rn in range(len(region_name)):
            region = tifffile.imread(os.path.join(region_path, 'Region_' + region_name[rn] + '.tif'))
            region_counts = np.count_nonzero(region == 255)
            if region_counts != 0:
                # Retain the original scaling factor for existing result compatibility.
                cell_density = counts[rn] / region_counts * 15626
                cell_density = np.round(cell_density, 4)
            else:
                cell_density = 0
            write_lines = [region_name[rn], cell_density]
            csv_write.writerow(write_lines)
            print('finished {}'.format(region_name[rn]))

def get_whole_layer_density(region_color, root, layer_path, data_name, savecsv):
    region_name = list(region_color.keys())
    df = _read_region_rows(os.path.join(root, data_name, 'cortex_cells_counts.csv'),
                           region_name, 'Region')
    os.makedirs(savecsv or os.curdir, exist_ok=True)
    csv_name = 'layer_density.csv'
    csv_root = os.path.join(savecsv, csv_name)
    first_line = ['Region_name', 'layer 1', 'layer 2_3', 'layer 4', 'layer 5', 'layer 6']
    no_layer4_region = ['FRP', 'MOp', 'MOs', 'RSPagl', 'RSPd', 'RSPv', 'ORBI', 'ORBm', 'ORBvl', 'ACAd', 'ACAv', 'PL', 'ILA', 'Ald', 'Alp', 'Alv', 
                    'PERI', 'TEa', 'ECT']
    layers = [('1', 'layer 1'), ('2_3', 'layer 2/3'), ('4', 'layer 4'),
              ('5', 'layer 5'), ('6', 'layer 6')]
    with open(csv_root, 'w+', newline='') as f:
        csv_write = csv.writer(f, dialect='excel')
        csv_write.writerow(first_line)
        for i in range(len(region_name)):
            densities = []
            for layer, column in layers:
                if layer == '4' and region_name[i] in no_layer4_region:
                    densities.append(0)
                    continue
                layer_file = 'Region_{}_layer_{}.tif'.format(region_name[i], layer)
                region = tifffile.imread(os.path.join(layer_path, layer_file))
                region_counts = np.count_nonzero(region == 255)
                count = df[column].iloc[i]
                densities.append(np.round(count / region_counts, 4)
                                 if region_counts else 0)
            write_lines = [region_name[i]] + densities
            csv_write.writerow(write_lines)
            print('finished {}'.format(region_name[i]))

def get_whole_layer_mean_density(root, data_name, region_color, save_csv, group_name):
    if not len(data_name):
        raise ValueError('At least one sample is required to calculate mean density')
    region_names = list(region_color.keys())
    columns = ['layer 1', 'layer 2_3', 'layer 4', 'layer 5', 'layer 6']
    totals = np.zeros((len(region_names), len(columns)), dtype=float)
    for sample in data_name:
        csv_path = os.path.join(root, sample, 'whole_brain_cell_counts', 'layer_density.csv')
        frame = _read_region_rows(csv_path, region_names, 'Region_name')
        totals += frame[columns].values
    means = np.round(totals / len(data_name), 4)
    os.makedirs(save_csv or os.curdir, exist_ok=True)
    csv_name = group_name + '_mean_density.csv'
    csv_root = os.path.join(save_csv, csv_name)
    values = pd.DataFrame(means, columns=['layer1', 'layer2_3', 'layer4', 'layer5', 'layer6'])
    values.insert(0, 'Region_name', region_names)
    values.to_csv(csv_root, index=False)
    print('finishe {} mean density'.format(group_name))

if __name__ == "__main__":
    from Parameter import Data_root, Isocortex, Stats

    root = Data_root
    fs_name = Stats['FST_data_list']
    ctrl_name = Stats['Ctrl_data_list']  
    data_name = Stats['Pred_data_list']

    for ind in range(len(data_name)):
        region_path = os.path.join(root, data_name[ind], 'whole_brain_cell_counts', 'cortex_region')
        csv_path = os.path.join(root, data_name[ind])
        save_density = os.path.join(region_path, '..')
        get_density_csv(region_path, csv_path, Isocortex, save_density)
        print('finished {}'.format(data_name[ind]))

    for ind in range(len(data_name)):
        path = os.path.join(root, data_name[ind], 'whole_brain_cell_counts')
        layer_path = os.path.join(path, 'cortex_layer')
        layer_points_path = os.path.join(path, 'points_cortex_layer')
        get_whole_layer_density(Isocortex, root, layer_path, data_name[ind], path)
        print('finished {}'.format(data_name[ind]))

    get_whole_layer_mean_density(root, fs_name, Isocortex, save_csv=root, group_name='FS') # group_name = 'FS' or 'Ctrl'
