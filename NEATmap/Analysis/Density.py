from Environment import *
from Parameter import *

def get_density_csv(region_path, counts_csv, region_color, save_csv):
    '''cell counts/region pixel nums'''
    region_name = list(region_color.keys())
    file = open(os.path.join(counts_csv, 'cortex_cells_counts.csv'))
    df = pd.read_csv(file)
    counts = df['SUM'].values
    csv_save = os.path.join(save_csv, 'cortex_density.csv')
    first_lines = ['Region', 'Density']
    with open(csv_save, 'w+', newline='') as f:
        csv_write = csv.writer(f, dialect='excel')
        csv_write.writerow(first_lines)
        for rn in range(len(region_name)):
            region = tifffile.imread(os.path.join(region_path, 'Region_' + region_name[rn] + '.tif'))
            x, _, _ = np.where(region==255)
            region_counts = len(x)
            if region_counts != 0:
                cell_density = counts[rn] / (region_counts) * 15626 # 15625 = 25*25*25
                cell_density = np.round(cell_density, 4)
            else:
                cell_density = 0
            write_lines = [region_name[rn], cell_density]
            csv_write.writerow(write_lines)
            print('finished {}'.format(region_name[rn]))

def get_whole_layer_density(region_color, root, layer_path, data_name, savecsv):
    file = open(os.path.join(root, data_name, 'cortex_cells_counts.csv'))
    df = pd.read_csv(file)
    region_name = list(region_color.keys())
    csv_name = 'layer_density.csv'
    csv_root = os.path.join(savecsv, csv_name)
    first_line = ['Region_name', 'layer 1', 'layer 2_3', 'layer 4', 'layer 5', 'layer 6']
    no_layer4_region = ['FRP', 'MOp', 'MOs', 'RSPagl', 'RSPd', 'RSPv', 'ORBI', 'ORBm', 'ORBvl', 'ACAd', 'ACAv', 'PL', 'ILA', 'Ald', 'Alp', 'Alv', 
                    'PERI', 'TEa', 'ECT']
    with open(csv_root, 'w+', newline='') as f:
        csv_write = csv.writer(f, dialect='excel')
        csv_write.writerow(first_line)
        for i in range(len(region_name)):
            region_layer1 = tifffile.imread(os.path.join(layer_path,  'Region_' + region_name[i] + '_layer_1.tif'))
            layer1_counts = df['layer 1']
            x1, _, _ = np.where(region_layer1==255)
            region_counts_1 = len(x1)
            if region_counts_1 != 0:
                cell_density = layer1_counts[i] / (region_counts_1)
                cell_density1 = np.round(cell_density, 4)
            else:
                cell_density1 = 0
            
            region_layer2_3 = tifffile.imread(os.path.join(layer_path,  'Region_' + region_name[i] + '_layer_2_3.tif'))
            layer2_3_counts = df['layer 2/3']
            x23, _, _ = np.where(region_layer2_3==255)
            region_counts_2_3 = len(x23)
            if region_counts_2_3 != 0:
                cell_density = layer2_3_counts[i] / (region_counts_2_3)
                cell_density2_3 = np.round(cell_density, 4)
            else:
                cell_density2_3 = 0

            if region_name[i] not in no_layer4_region:
                region_layer4 = tifffile.imread(os.path.join(layer_path,  'Region_' + region_name[i] + '_layer_4.tif'))
                layer4_counts = df['layer 4']
                x4, _, _ = np.where(region_layer4==255)
                region_counts_4 = len(x4)
                if region_counts_4 != 0:
                    cell_density = layer4_counts[i] / (region_counts_4)
                    cell_density4 = np.round(cell_density, 4)
                else:
                    cell_density4 = 0
            else:
                cell_density4 = 0
            
            region_layer5 = tifffile.imread(os.path.join(layer_path,  'Region_' + region_name[i] + '_layer_5.tif'))
            layer5_counts = df['layer 5']
            x5, _, _ = np.where(region_layer5==255)
            region_counts_5 = len(x5)
            if region_counts_5 != 0:
                cell_density = layer5_counts[i] / (region_counts_5)
                cell_density5 = np.round(cell_density, 4)
            else:
                cell_density5 = 0

            region_layer6 = tifffile.imread(os.path.join(layer_path,  'Region_' + region_name[i] + '_layer_6.tif'))
            layer6_counts = df['layer 6']
            x6, _, _ = np.where(region_layer6==255)
            region_counts_6 = len(x6)
            if region_counts_6 != 0:
                cell_density = layer6_counts[i] / (region_counts_6)
                cell_density6 = np.round(cell_density, 4)
            else:
                cell_density6 = 0
            write_lines = [region_name[i], cell_density1, cell_density2_3, cell_density4, cell_density5, cell_density6]
            csv_write.writerow(write_lines)
            print('finished {}'.format(region_name[i]))

def get_whole_layer_mean_density(root, data_name, region_color, save_csv, group_name):

    region_num = len(list(region_color.keys()))
    layer1_array = np.zeros((region_num, len(data_name)))
    layer2_3_array = np.zeros((region_num, len(data_name)))
    layer4_array = np.zeros((region_num, len(data_name)))
    layer5_array = np.zeros((region_num, len(data_name)))
    layer6_array = np.zeros((region_num, len(data_name)))

    for i in range(len(data_name)):
        file = open(os.path.join(root, data_name[i], 'whole_brain_cell_counts/layer_density.csv'))
        df = pd.read_csv(file)
        layer1 = df['layer 1']
        layer2_3 = df['layer 2_3']
        layer4 = df['layer 4']
        layer5 = df['layer 5']
        layer6 = df['layer 6']
        layer1_array[:, i] = layer1
        layer2_3_array[:, i] = layer2_3
        layer4_array[:, i] = layer4
        layer5_array[:, i] = layer5
        layer6_array[:, i] = layer6

    mean_layer1 = np.mean(layer1_array, axis=1)
    mean_layer2_3 = np.mean(layer2_3_array, axis=1)
    mean_layer4 = np.mean(layer4_array, axis=1)
    mean_layer5 = np.mean(layer5_array, axis=1)
    mean_layer6 = np.mean(layer6_array, axis=1)

    csv_name = group_name + '_mean_density.csv'
    csv_root = os.path.join(save_csv, csv_name)
    values = pd.DataFrame(columns=['Region_name', 'layer1', 'layer2_3', 'layer4', 'layer5', 'layer6'])
    values['Region_name'] = list(region_color.keys())
    values['layer1'] = np.round(mean_layer1, 4)
    values['layer2_3'] = np.round(mean_layer2_3, 4)
    values['layer4'] = np.round(mean_layer4, 4)
    values['layer5'] = np.round(mean_layer5, 4)
    values['layer6'] = np.round(mean_layer6, 4)
    values.to_csv(csv_root, index=False)
    print('finishe {} mean density'.format(group_name))

if __name__ == "__main__":
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