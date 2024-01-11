from Environment import *
from Parameter import *
from Analysis.freesia_export_to_BrainRegion import genSingleRegion, genPointsSingleRegion

def get_multi_region(region_color, image_root, atlas_root, Region_root, Points_SingleRegion_root, get_point=False):
    region_name = region_color.keys()
    color = region_color.values()
    for i in range(len(list(region_name))):
        if get_point:
            genPointsSingleRegion(image_root, Region_root, Points_SingleRegion_root, list(region_name)[i])
            print('{} region'.format(list(region_name)[i]))
        else:
            genSingleRegion(atlas_root ,Region_root, list(region_name)[i], list(color)[i])
            print('{} region'.format(list(region_name)[i]))

def get_multi_layer(region_color, image_root, atlas_root, Region_root, Points_SingleRegion_root, get_point=False):

    region_name = list(region_color.keys())
    color = list(region_color.values())
    layer_name_5 = ['_layer_1', '_layer_2_3', '_layer_4', '_layer_5', '_layer_6']
    layer_name_4 = ['_layer_1', '_layer_2_3', '_layer_5', '_layer_6']
    no_layer4_region = ['FRP', 'MOp', 'MOs', 'RSPagl', 'RSPd', 'RSPv', 'ORBI', 'ORBm', 'ORBvl', 'ACAd', 'ACAv', 'PL', 'ILA', 'Ald', 'Alp', 'Alv', 
                        'PERI', 'TEa', 'ECT']
    for i in range(len(region_name)):
        if get_point:
            if region_name[i] in no_layer4_region:
                for num in range(0, 4):
                    genPointsSingleRegion(image_root, Region_root, Points_SingleRegion_root, region_name[i] + layer_name_4[num])
                    print('{} region'.format(region_name[i] + layer_name_4[num]))
            else:
                for num in range(0, 5):
                    genPointsSingleRegion(image_root, Region_root, Points_SingleRegion_root, region_name[i] + layer_name_5[num])
                    print('{} region'.format(region_name[i] + layer_name_5[num]))
        else:
            if region_name[i] in no_layer4_region:
                index = 0
                for num in range(0, 4):
                    if index == 6:
                        genSingleRegion(atlas_root ,Region_root, region_name[i] + layer_name_4[num], color[i][index:])
                        print('{} region'.format(region_name[i] + layer_name_4[num]))
                    else:
                        genSingleRegion(atlas_root ,Region_root, region_name[i] + layer_name_4[num], color[i][index : index + 2])
                        print('{} region'.format(region_name[i] + layer_name_4[num]))
                    index += 2
            else:
                index = 0
                for num in range(0, 5):
                    if index == 8:
                        genSingleRegion(atlas_root ,Region_root, region_name[i] + layer_name_5[num], color[i][index: ])
                        print('{} region'.format(region_name[i] + layer_name_5[num]))
                    else:
                        genSingleRegion(atlas_root ,Region_root, region_name[i] + layer_name_5[num], color[i][index : index + 2])
                        print('{} region'.format(region_name[i] + layer_name_5[num]))
                    index += 2

if __name__ == "__main__":
    root = Data_root
    data_name = Stats['Pred_data_list']
    
    for i in range(len(data_name)):
        image_root = os.path.join(root, data_name[i], 'whole_brain_cell_counts/images')
        altas_root =  os.path.join(image_root, '..', 'atlas')
        save_region_root = os.path.join(image_root, '..', 'cortex_region')
        save_points_region_root = os.path.join(image_root, '..', 'points_cortex_region')
        get_multi_region(Isocortex, image_root, altas_root, save_region_root, save_points_region_root, get_point=True)

        save_layer_root = os.path.join(image_root, '..', 'cortex_layer')
        save_points_layer_root = os.path.join(image_root, '..', 'points_cortex_layer')
        get_multi_layer(Isocortex, image_root, altas_root, save_layer_root, save_points_layer_root, get_point=True)
        print('finished {}'.format(data_name[i])) 