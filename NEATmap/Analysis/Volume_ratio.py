import csv
import os

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi

from Parameter import Stats

def volume_ratio(region_path, label_path, region_name, min_volume=Stats['min_volume'], max_volume=Stats['max_volume']):

    '''volume ratio (computed as (total plaque volume in the region)/(region volume))'''
    region = tifffile.imread(os.path.join(region_path, 'Region_' + region_name + '.tif'))
    label = tifffile.imread(os.path.join(label_path, 'Points_Region_' + region_name + '.tif'))
    if region.ndim != 3 or label.shape != region.shape:
        raise ValueError('Region and label must be matching three-dimensional images')
    if min_volume < 0 or max_volume < min_volume:
        raise ValueError('Expected 0 <= min_volume <= max_volume')
    struct = ndi.generate_binary_structure(3,1)
    label, _ = ndi.label(label, struct)
    counts = np.bincount(label.ravel())[1:]  # Label 0 is background.
    volumes = counts[(counts > min_volume) & (counts <= max_volume)]
    region_counts = np.count_nonzero(region == 255)
    if region_counts != 0:
        ratios = (volumes.sum() / region_counts) * 100
        ratios = np.round(ratios, 2)
    else:
        ratios = 0 

    return ratios

def get_volume_ratio_csv(region_color, region_path, label_path, save_csv_root):
    region_name = list(region_color.keys())
    os.makedirs(save_csv_root, exist_ok=True)
    csv_name = 'cortex_volume.csv'
    csv_root = os.path.join(save_csv_root, csv_name)
    first_line = ['Region_name', 'Volume ratio %']
    with open(csv_root, 'w+', newline='') as f:
        csv_write = csv.writer(f, dialect='excel')
        csv_write.writerow(first_line)
        for i in range(len(region_name)):
            ratio = volume_ratio(region_path, label_path, region_name[i])
            write_lines = [region_name[i], ratio]
            csv_write.writerow(write_lines)
            print('finished {}'.format(region_name[i]))

def get_whole_layer_volume(region_color, layer_path, layer_points_path, savecsv):
    region_name = list(region_color.keys())
    os.makedirs(savecsv or os.curdir, exist_ok=True)
    csv_name = 'layer_volume.csv'
    csv_root = os.path.join(savecsv, csv_name)
    first_line = ['Region_name', 'layer 1', 'layer 2_3', 'layer 4', 'layer 5', 'layer 6']
    no_layer4_region = ['FRP', 'MOp', 'MOs', 'RSPagl', 'RSPd', 'RSPv', 'ORBI', 'ORBm', 'ORBvl', 'ACAd', 'ACAv', 'PL', 'ILA', 'Ald', 'Alp', 'Alv', 
                        'PERI', 'TEa', 'ECT']
    with open(csv_root, 'w+', newline='') as f:
        csv_write = csv.writer(f, dialect='excel')
        csv_write.writerow(first_line)
        for i in range(len(region_name)):
            layer_1_ratio = volume_ratio(layer_path, layer_points_path, region_name[i] + '_layer_1')
            layer_2_3_ratio = volume_ratio(layer_path, layer_points_path, region_name[i] + '_layer_2_3')
            if region_name[i] not in no_layer4_region:
                layer_4_ratio = volume_ratio(layer_path, layer_points_path, region_name[i] + '_layer_4')
            else:
                layer_4_ratio = 0 
            layer_5_ratio = volume_ratio(layer_path, layer_points_path, region_name[i] + '_layer_5')
            layer_6_ratio = volume_ratio(layer_path, layer_points_path, region_name[i] + '_layer_6')
            write_lines = [region_name[i], layer_1_ratio, layer_2_3_ratio, layer_4_ratio, layer_5_ratio, layer_6_ratio]
            csv_write.writerow(write_lines)
            print('finished {}'.format(region_name[i]))

def get_whole_layer_mean_volume(root, data_name, region_color, save_csv, group_name):
    if not len(data_name):
        raise ValueError('At least one sample is required to calculate mean volume')
    region_names = list(region_color.keys())
    columns = ['layer 1', 'layer 2_3', 'layer 4', 'layer 5', 'layer 6']
    totals = np.zeros((len(region_names), len(columns)), dtype=float)
    for sample in data_name:
        csv_path = os.path.join(root, sample, 'whole_brain_cell_counts', 'layer_volume.csv')
        frame = pd.read_csv(csv_path)
        if 'Region_name' in frame.columns:
            if frame['Region_name'].duplicated().any():
                raise ValueError('Duplicate region names in {}'.format(csv_path))
            frame = frame.set_index('Region_name')
            missing = [name for name in region_names if name not in frame.index]
            if missing:
                raise ValueError('Missing regions in {}: {}'.format(csv_path, missing))
            frame = frame.loc[region_names]
        elif len(frame) != len(region_names):
            raise ValueError('Region count does not match {}'.format(csv_path))
        totals += frame[columns].values
    means = np.round(totals / len(data_name), 2)
    os.makedirs(save_csv or os.curdir, exist_ok=True)
    csv_name = group_name + '_mean_volume.csv'
    csv_root = os.path.join(save_csv, csv_name)
    values = pd.DataFrame(means, columns=['layer1', 'layer2_3', 'layer4', 'layer5', 'layer6'])
    values.insert(0, 'Region_name', region_names)
    values.to_csv(csv_root, index=False)
    print('finishe {} mean volume'.format(group_name))

if __name__ == "__main__":
    from Parameter import Data_root, Isocortex

    root = Data_root
    data_name = Stats['group_data_name']

    for ind in range(len(data_name)):
        region_path = os.path.join(root, data_name[ind], 'whole_brain_cell_counts', 'cortex_region')
        points_label_path = os.path.join(root, data_name[ind], 'whole_brain_cell_counts', 'points_cortex_region')
        save_volume_ratio = os.path.join(region_path, '..')
        get_volume_ratio_csv(Isocortex, region_path, points_label_path, save_volume_ratio)
        print('finished {}'.format(data_name[ind]))

    for ind in range(0, len(data_name)):
        path = os.path.join(root, data_name[ind], 'whole_brain_cell_counts')
        layer_path = os.path.join(path, 'cortex_layer')
        layer_points_path = os.path.join(path, 'points_cortex_layer')
        get_whole_layer_volume(Isocortex, layer_path, layer_points_path, savecsv=path)
        print('finished {}'.format(data_name[ind]))

    ctrl_name = Stats['Ctrl_data_list']
    fs_name = Stats['FST_data_list']
    get_whole_layer_mean_volume(root, ctrl_name, Isocortex, save_csv=root, group_name='Ctrl')
