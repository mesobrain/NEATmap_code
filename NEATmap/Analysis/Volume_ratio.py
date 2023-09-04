from Environment import *
from Parameter import *

def volume_ratio(region_path, label_path, region_name, min_volume=Stats['min_volume'], max_volume=Stats['max_volume']):

    '''volume ratio (computed as (total plaque volume in the region)/(region volume))'''
    region = tifffile.imread(os.path.join(region_path, 'Region_' + region_name + '.tif'))
    label = tifffile.imread(os.path.join(label_path, 'Points_Region_' + region_name + '.tif'))
    struct = ndi.generate_binary_structure(3,1)
    label, _ = ndi.label(label, struct)
    unique, counts = np.unique(label, return_counts=True)
    # _, region_counts = np.unique(region_label, return_counts=True)
    _, _, x = np.where(region==255)
    region_counts = len(x)
    small, medium, large = [], [], []
    for uq, ct in zip(unique, counts):
        if uq == 0:
            continue # skip zero!
        if ct <= min_volume:
            small.append([uq, ct]) # if object is smaller than mimimum size, it gets excluded
        elif min_volume < ct <= max_volume:
            medium.append([uq, ct] )
        else:
            large.append([uq, ct])
    volumes = [e[1] for e in medium]
    if region_counts != 0:
        ratios = (sum(volumes) / region_counts) * 100
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

    region_num = len(list(region_color.keys()))
    layer1_array = np.zeros((region_num, len(data_name)))
    layer2_3_array = np.zeros((region_num, len(data_name)))
    layer4_array = np.zeros((region_num, len(data_name)))
    layer5_array = np.zeros((region_num, len(data_name)))
    layer6_array = np.zeros((region_num, len(data_name)))

    for i in range(len(data_name)):
        file = open(os.path.join(root, data_name[i], 'whole_brain_cell_counts/layer_volume.csv'))
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

    csv_name = group_name + '_mean_volume.csv'
    csv_root = os.path.join(save_csv, csv_name)
    values = pd.DataFrame(columns=['Region_name', 'layer1', 'layer2_3', 'layer4', 'layer5', 'layer6'])
    values['Region_name'] = list(region_color.keys())
    values['layer1'] = np.round(mean_layer1, 2)
    values['layer2_3'] = np.round(mean_layer2_3, 2)
    values['layer4'] = np.round(mean_layer4, 2)
    values['layer5'] = np.round(mean_layer5, 2)
    values['layer6'] = np.round(mean_layer6, 2)
    values.to_csv(csv_root, index=False)
    print('finishe {} mean volume'.format(group_name))

if __name__ == "__main__":
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