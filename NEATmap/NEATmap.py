# -*- coding: utf-8 -*-
"""
NEATmap
=======
"""

__author__    = 'Weijie Zheng'
__license__   = 'GPLv3 - GNU General Pulic License v3'
__copyright__ = 'Copyright © 2023 by Weijie Zheng'
__webpage__   = 'https://github.com/mesobrain/NEATMap'

from Environment import *
from Parameter import *

if __name__ == "__main__":
    
    #%%############################################################################
    ### Data preprocessing
    ###############################################################################
    """
        VISoR reconstruction of the generated BrainImage path with 561 channels and 488 channels to 
        synthesize 3D images with (z, x, y) size (64,3500,2500)
    """
    ## 561nm
    start = time.time()
    root = Data_root
    json_path = os.path.join(Raw_data_root, '..', 'freesia_4.0_'+ Channels['488nm_index'] + '_488nm_10X.json')
    with open(json_path) as f:
        brain = json.load(f)
        images = brain['images']
        total_num = len(images)
    save_brain3d_561nm_path = os.path.join(Data_root, 'brain_image_64_' + Channels['staining'])
    b2t3.brain2dto3d(Raw_data_root, save_brain3d_561nm_path, total_num, z_num=Preprocessing['z_num'], channel_index=Channels['561nm_index'])
    # ## 488nm
    save_brain3d_path = os.path.join(Data_root, 'brain_image_64_' + Channels['autofluo'])
    b2t3.brain2dto3d(Raw_data_root, save_brain3d_path, total_num, z_num=Preprocessing['z_num'], channel_index=Channels['488nm_index'])

    """
        Synthetic 3d brain slices were cut into 70 patches. If you get the training data, you need to input the parameters train_path and label_path.
    """
    # Cut 561nm channel
    cut.cut(root=Data_root, cut_size=Preprocessing['cut_size'], channel=Channels['staining'],
    cut_index_x=Preprocessing['cut_index_x'], cut_index_y=Preprocessing['cut_index_y'], 
    patch_weight_num=Preprocessing['patch_weight_num'], patch_hegiht_num=Preprocessing['patch_height_num'])

    # Cut 488nm channel
    cut.cut(root=Data_root, cut_size=Preprocessing['cut_size'], channel=Channels['autofluo'],
    cut_index_x=Preprocessing['cut_index_x'], cut_index_y=Preprocessing['cut_index_y'], 
    patch_weight_num=Preprocessing['patch_weight_num'], patch_hegiht_num=Preprocessing['patch_height_num'])

    """
        Get the pixel-level labels of the training data.
    """
    image_path = Train_config['train_image_path']
    label_path = Train_config['train_label_path']
    sps.spot_seg(train_image_path=image_path, save_seg_path=label_path, channel=Channels['staining'], 
            scaling_param=Spot_seg_config['scaling_param'], wrapper_param=Spot_seg_config['wrapper_param'])

    """
        Images and labels make up the training data.
    """
    save_path = os.path.join(Train_config['train_data_path'], 'train_data')
    tn.get_train_data(image_path, label_path, save_path)

    """
        Write the training data name.
    """
    wt.data_list(train_data=Network['train_data'], ratio=Network['train_valid_ratio'], test_id=Test_config['test_id'],
            save_list=Network['save_data_list'], test_singel=Network['test_slice'], test_whole_brain=Network['test_whole_brain'])
    """
        Training neural networks
    """
    from Network.model.swin_transform import swin_tiny_patch4_window8_256
    # from Network.model.FCN_resnet3d import fcn_resnet34
    # from Network.model.unet3d import UNet3D
    # from Network.model.ASPP_unet3d import UNet3D
    # from Network.model.FCN import FCN
    from Network.utils import weights_init
    ## Whether to perform transfer learning or not, the default parameter is False.
    pre_train = Network['pre_train']
    checkpoint_path = os.path.join(pre_train, 'epoch_9.pth')
    ##
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    net = swin_tiny_patch4_window8_256(in_channels=Network['swin_in_channels'], num_classes=Network['num_classes']).to(device)
    # net = FCN(in_channels=1, num_classes=Network['num_classes']).to(device)
    # net = fcn_resnet34(num_classes=Network['num_classes']).to(device)
    # net = UNet3D(in_channel=1, n_classes=Network['num_classes']).to(device)
    # net = HRNet(num_classes=Network['num_classes']).to(device)
    net = net.apply(weights_init)
    print('net initialization succeeds !')
    train.train(net, checkpoint_path, pre_train, device)

    """
        Preparing test data
    """
    patch_num = Preprocessing['patch_weight_num'] * Preprocessing['patch_height_num']
    ths.slice_test_data(image_patch_path=Test_config['test_image_path'], label_patch_path=Test_config['test_label_path'], 
                    channel=Channels['staining'], save_test_path=Test_config['test_save_path'], patch_num=patch_num, 
                    test_id=Test_config['test_id'])
    """
        Testing the performance of the network.
    """
    from Network.dataset_VISoR import VISoR_dataset
    parser = argparse.ArgumentParser()
    parser.add_argument('--volume_path', type=str,
                        default='', help='root dir for validation volume data') 
    parser.add_argument('--dataset', type=str,
                        default='VISoR', help='experiment_name')
    parser.add_argument('--max_epochs', type=int,
                        default=Network['max_epochs'], help='maximum epoch number to train')
    parser.add_argument('--num_classes', type=int,
                        default=Network['num_classes'], help='output channel of network')
    parser.add_argument('--list_dir', type=str,
                        default='./lists', help='list dir')
    parser.add_argument('--model_name', type=str,
                        default='swin', help='model_name')

    parser.add_argument('--batch_size', type=int, default=Network['test_batch_size'], help='batch_size per gpu')
    parser.add_argument('--img_size', type=int, default=Preprocessing['cut_size'], help='input patch size of network input')
    parser.add_argument('--is_savetif', action="store_true", default=True, help='whether to save results during inference')
    parser.add_argument('--test_id', type=int, default=30, help='test the 30th brain slice')
    parser.add_argument('--test_save_dir', type=str, default='../predictions', help='saving prediction as tif!')
    parser.add_argument('--base_lr', type=float,  default=Network['lr'], help='segmentation network learning rate')
    parser.add_argument('--patch_csv_root', type=str,  default=Network['save_patch_metrics'], help='The path to save the patch evaluation result csv')
    parser.add_argument('--slice_csv_root', type=str,  default=Network['save_metrics'], help='The path to save the slice evaluation result csv')
    args = parser.parse_args()

    dataset_config = {
        'VISoR': {
            'Dataset': VISoR_dataset,
            'volume_path': Test_config['test_save_path'],
            'list_dir': '/home/weijie/3dHRNet/lists',
            'num_classes': Network['num_classes'],
            'z_spacing': Network['z_spacing'],
        },
    }
    dataset_name = args.dataset
    args.exp = dataset_name + str(args.img_size)
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.test_id = Test_config['test_id']
    args.is_savetif = True

    # name the same snapshot defined in train script!
    snapshot_path = Network['snapshot_path']

    # net = HRNet(num_classes=args.num_classes, width=Model['hrnet_width']).cuda()
    # net = UNet3D(in_channel=Network['in_channels'], n_classes=args.num_classes).cuda()
    # net = FCN(in_channels=Network['in_channels'], num_classes=args.num_classes).cuda()
    # net = fcn_resnet34(num_classes=args.num_classes).cuda()
    net = swin_tiny_patch4_window8_256(in_channels=Network['swin_in_channels'], num_classes=args.num_classes).cuda()

    snapshot = os.path.join(snapshot_path, 'best_model.pth')
    if not os.path.exists(snapshot): snapshot = snapshot.replace('best_model', 'epoch_'+str(args.max_epochs-1))
    net.load_state_dict(torch.load(snapshot))
    snapshot_name = snapshot_path.split('/')[-1]

    log_folder = 'test_log/test_log_' + args.exp
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    if args.is_savenii:
        args.test_save_dir = os.path.join(Network['save_prediction_path'], 'brain_predications_' + args.test_id + '_' + args.model_name + '_epoch' + str(args.max_epochs))
        test_save_path = os.path.join(args.test_save_dir, args.exp, snapshot_name)
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    test.test(args, net, test_save_path)

    """
        Automated whole brain c-Fos expression for cellular segmentation.
    """
    from Network.dataset_whole_brain import Whole_brain_dataset
    data_path = Network['whole_brain_path']
    test_path = os.path.join(data_path, 'PatchImage_' + args.channel)
    for ind in range(1, len(os.listdir(test_path)) + 1):
        dataset_config = {
            'VISoR': {
                'Dataset': Whole_brain_dataset,
                'volume_path': test_path + '/patchimage{}'.format(ind),
                'list_dir': Network['whole_brain_list'],
                'name_list': 'Z{:05d}_test'.format(ind),
                'z_spacing': Network['z_spacing'],
            },
        }
        dataset_name = args.dataset
        args.exp = dataset_name + str(args.img_size)
        args.volume_path = dataset_config[dataset_name]['volume_path']
        args.Dataset = dataset_config[dataset_name]['Dataset']
        args.list_dir = dataset_config[dataset_name]['list_dir']
        args.name_list = dataset_config[dataset_name]['name_list']
        args.z_spacing = dataset_config[dataset_name]['z_spacing']
        args.is_pretrain = True

        # name the same snapshot defined in train script!
        snapshot_path = Network['snapshot_path']

        net = swin_tiny_patch4_window8_256(in_channels=Network['swin_in_channels'], num_classes=args.num_classes).cuda()

        snapshot = os.path.join(snapshot_path, 'best_model.pth')
        if not os.path.exists(snapshot): snapshot = snapshot.replace('best_model', 'epoch_'+str(args.max_epochs-1))
        net.load_state_dict(torch.load(snapshot))
        snapshot_name = snapshot_path.split('/')[-1]

        log_folder = 'test_log/test_log_' + args.exp
        os.makedirs(log_folder, exist_ok=True)
        logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        logging.info(str(args))
        logging.info(snapshot_name)

        if args.is_savenii:
            args.test_save_dir = data_path + '/whole_predications_' + args.channel + 'brain_predications_{}_swin_epoch10'.format(ind)
            test_save_path = os.path.join(args.test_save_dir, args.exp, snapshot_name)
            os.makedirs(test_save_path, exist_ok=True)
        else:
            test_save_path = None
        infer_whole.infer_whole_brain(args, net, ind, test_save_path)

    """
        Automatic segmentation of volume image decomposition into 2-dimensional images.
        The segmentation result (z,x,y) is stitched into a volume image of size (64,3500,2500).
    """

    channel = Channels['staining']
    pred_root = os.path.join(Data_root, 'whole_predications_' + channel)
    path = os.path.join(Data_root, 'brain_image_64_' + channel)
    save_path_3d = os.path.join(Splicing['save_splicing_path'], 'whole_brain_pred_' + channel)
    os.makedirs(save_path_3d, exist_ok=True)
    infer_total_num = len(os.listdir(path))
    resuidual_z = total_num - (Preprocessing['z_num'] * infer_total_num)
    patch_total_num = Preprocessing['patch_weight_num'] * Preprocessing['patch_height_num']
    for num in range(1, len(os.listdir(path)) + 1):
        list = rt.load(path, num, total_patch_num=patch_total_num)
        rt.concat(list, save_path_3d, num)
        print('finished {} image '.format(num))
    rt.create_residual_image(save_path_3d, infer_total_num + 1, resuidual_z)


    """
        Post-processing of the predicted segmentation results.
    """
    image_path = os.path.join(Data_root, 'brain_image_64_' + Channels['staining'])
    path = Splicing['save_splicing_path']
    pred_path = os.path.join(path, 'whole_brain_pred_' + Channels['staining'])
    path_488nm = os.path.join(path, 'whole_brain_pred_' + Channels['autofluo'])
    save_path = os.path.join(path, 'whole_brain_pred_post_filter')
    post.spot_filter(image_path, pred_path, path_488nm, save_path, lower_limit=Post['intensity_lower_differ'], 
                        upper_limit=Post['intensity_upper_differ'], min_size=Post['point_min_size'], max_intensity=Post['big_object_size'])

    """
        Register to Allen Brain Atlas CCFv3.
    """
    reconstruction_root = os.path.join(Registration['Raw_brain_path'], 'Reconstruction')
    image_list_file = os.path.join(reconstruction_root, 'BrainImage', 'freesia_4.0_C1_488nm_10X.json')
    output_path = Registration['output_path'] 
    os.makedirs(output_path, exist_ok=True)
    template_file = 'Registration/data/ccf_v3_template.json'
    output_name = Registration['output_name']
    transform_path = os.path.join(reconstruction_root, 'BrainTransform', 'visor_brain.txt')
    br.register_brain(image_list_file=image_list_file, output_path=output_path, template_file=template_file, output_name=output_name, 
                    brain_transform_path=transform_path)

    """
        The results of the segmentation were composed into brain slices.
    """
    with tempfile.TemporaryDirectory() as TEMP_PATH:
        splice_path = Splicing['save_splicing_path']
        data_path = os.path.join(splice_path, 'whole_brain_pred_post_filter')
        save_path = os.path.join(TEMP_PATH, 'whole_brain_pred_2d')
        s3t2.seg3d_to_2d(data_path, save_path)
        save_root = os.path.join(splice_path, 'whole_brain_pred_3d')
        st.BrainImage_cat(save_path, save_root, flip=False)

    """
        The coordinates of the center of mass of each cell were counted.
    """
    BrainImage_root = os.path.join(Save_root, 'whole_brain_pred_3d')
    csv_root = os.path.join(Save_root, 'whole_brain_cell_counts')
    st.BrainImage2Spot(BrainImage_root, csv_root)

    """
        Segmentation of cells mapped to the Allen Brain Atlas. (Coronal plane)
    """
    save_csv_root = os.path.join(csv_root, 'Thumbnail_CSV')
    total_path = os.path.join(csv_root, 'total.txt')
    st.Spot_csv(total_path, save_csv_root, brainimage2d_num=total_num, group_num=Spot_map['group_num'])

    """
        Segmentation of cells mapped to the Allen Brain Atlas. (Sagittal plane)
    """
    st.Sagittal_spot_csv(total_path, csv_root, brainimage_x=Brain['weight'], group_num=Spot_map['group_num'])
    
    """
        Segmentation of cells mapped to the Allen Brain Atlas. (Horizontal plane)
    """
    st.Horizontal_spot_csv(total_path, csv_root, brainimage_y=Brain['height'], group_num=Spot_map['group_num'])

    """
        Remove the cell center-of-mass coordinate points from the freesia-derived image to obtain atlas.
    """
    ##freesia export cell counts
    image_root = os.path.join(freesia_export_path, 'images')
    fb.regenAtlas(image_root)
    """
        Obtain the boundaries of the brain atlas.
    """
    atlas_root = os.path.join(image_root, '..', 'atlas')
    Edge_root = os.path.join(image_root, '..', 'Edge')
    fb.genEdgefromAtlas(atlas_root, Edge_root)
    """
        Obtain a brain atlas of the segmented to cells.
    """
    Points_root = os.path.join(image_root, '..', 'Points')
    fb.genPoints(image_root,Points_root)  
    """
        Obtain segmentation masks and cellular distribution of brain regions of interest.
    """
    Region_name = 'SFO'
    Mask_id = Region_mask[Region_name]#CBX
    Region_root = os.path.join(image_root, '..', 'Region')
    Points_SingleRegion_root = os.path.join(image_root, '..', 'Points_Single_Region')
    fb.genSingleRegion(atlas_root, Region_root, Region_name, Mask_id)
    fb.genPointsSingleRegion(image_root, Region_root, Points_SingleRegion_root, Region_name)
    """
        Acquisition of a whole brain cell activity map.
    """
    save_path = os.path.join(atlas_root, '..', '3d_result')
    save_name = 'altas'
    save_name = 'whole_brain_points'
    altas_num = int(total_num / Spot_map['group_num'])
    fb.concat_image(atlas_root, save_path, z_num=altas_num, save_name=save_name)
    fb.concat_image(Points_root, save_path, z_num=altas_num, save_name=save_name)
    """
        Display the whole brain cell activity map.
    """
    Save_root = Atlas_edge_pionts_save_path
    fb.genMergeRaw(Edge_root, Points_root, Save_root, Atlas_snapshot_id)

    """
        The number of cells in each layer in different regions of the cerebral cortex was obtained.
    """ 
    root = os.path.join(Data_root, '..')
    data_name = Stats['Pred_data_list']
    for k in range(len(data_name)):
        csv_path = os.path.join(root, data_name[k], 'whole_brain_cell_counts')
        ct.cortex_stats(csv_path)
        print('finished {}'.format(data_name[k]))

        
    """
        Access to different areas of the cerebral cortex.
    """
    root = os.path.join(Data_root, '..')
    data_name = Stats['Pred_data_list']
    
    for i in range(len(data_name)):
        image_root = os.path.join(root, data_name[i], 'whole_brain_cell_counts/images')
        altas_root =  os.path.join(image_root, '..', 'atlas')
        save_region_root = os.path.join(image_root, '..', 'cortex_region')
        save_points_region_root = os.path.join(image_root, '..', 'points_cortex_region')
        rl.get_multi_region(Isocortex, image_root, altas_root, save_region_root, save_points_region_root, get_point=False)

        save_layer_root = os.path.join(image_root, '..', 'cortex_layer')
        save_points_layer_root = os.path.join(image_root, '..', 'points_cortex_layer')
        rl.get_multi_layer(Isocortex, image_root, altas_root, save_layer_root, save_points_layer_root, get_point=True)
        print('finished {}'.format(data_name[i]))   

    """
        Calculated region and cell volume ratio in each layer
    """
    root = Data_root
    data_name = Stats['group_data_name']

    for ind in range(len(data_name)):
        region_path = os.path.join(root, data_name[ind], 'whole_brain_cell_counts', 'cortex_region')
        points_label_path = os.path.join(root, data_name[ind], 'whole_brain_cell_counts', 'points_cortex_region')
        save_volume_ratio = os.path.join(region_path, '..')
        vr.get_volume_ratio_csv(Isocortex, region_path, points_label_path, save_volume_ratio)
        print('finished {}'.format(data_name[ind]))

    for ind in range(0, len(data_name)):
        path = os.path.join(root, data_name[ind], 'whole_brain_cell_counts')
        layer_path = os.path.join(path, 'cortex_layer')
        layer_points_path = os.path.join(path, 'points_cortex_layer')
        vr.get_whole_layer_volume(Isocortex, layer_path, layer_points_path, savecsv=path)
        print('finished {}'.format(data_name[ind]))
    """
        Calculate the mean value of the cell volume ratio in the area and in each layer of the experimental or control group
    """
    ctrl_name = Stats['Ctrl_data_list']
    fs_name = Stats['FST_data_list']
    vr.get_whole_layer_mean_volume(root, ctrl_name, Isocortex, save_csv=root, group_name='Ctrl')

    """
        Calculated region and cell density in each layer
    """
    root = Data_root
    fs_name = Stats['FST_data_list']
    ctrl_name = Stats['Ctrl_data_list']  
    data_name = Stats['Pred_data_list']

    for ind in range(len(data_name)):
        region_path = os.path.join(root, data_name[ind], 'whole_brain_cell_counts', 'cortex_region')
        csv_path = os.path.join(root, data_name[ind])
        save_density = os.path.join(region_path, '..')
        ds.get_density_csv(region_path, csv_path, Isocortex, save_density)
        print('finished {}'.format(data_name[ind]))

    for ind in range(len(data_name)):
        path = os.path.join(root, data_name[ind], 'whole_brain_cell_counts')
        layer_path = os.path.join(path, 'cortex_layer')
        layer_points_path = os.path.join(path, 'points_cortex_layer')
        ds.get_whole_layer_density(Isocortex, root, layer_path, data_name[ind], path)
        print('finished {}'.format(data_name[ind]))
    """
        Calculate the mean value of the cell density in the area and in each layer of the experimental or control group
    """
    ds.get_whole_layer_mean_density(root, fs_name, Isocortex, save_csv=root, group_name='FS') # group_name = 'FS' or 'Ctrl'

    """
        Calculate the intensity values of cell signals in different regions.
    """
    root = Data_root
    data_path = os.path.join(root, 'Brainimage_25mic')
    label_path = os.path.join(root, '', r'whole_brain_cell_counts\points_cortex_region')
    save_csv_root = os.path.join(root, '', r'whole_brain_cell_counts\cortex_intensity')
    it.get_intensity_csv(Isocortex, data_path, label_path, save_csv_root, id='Ctrl10')
    """
        The percentage of intensity values of cell signals in different regions was calculated.
    """
    intensity_root = os.path.join(Data_root, '', r'whole_brain_cell_counts\cortex_intensity')
    savepath = os.path.join(intensity_root, '..', 'intensity_normalized_counts')
    it.normalized_counts(Isocortex, intensity_root, savepath)
    it.get_whole_norm_counts(path=savepath, region_color=Isocortex, csv_name='Norm_counts.csv')
    """
        The intensity values of the cell signals in each subvolume were calculated.
    """
    patch_path = ''
    it.patch_intensity(data_path=patch_path, label_path=patch_path, save_csv_root=patch_path, min_volume=Stats['min_volume'], max_volume=Stats['max_volume'])
    """
        Statistical hypothesis testing
    """
    ## Statistical hypothesis testing of metrics for evaluating DNN segmentation results
    stat.DNN_result_ranksum(data_path=Stats['DNN_results'], model_name=Stats['Model_name'])

    ## The cell count results derived from freesia are selected for a particular level of brain area.
    root = os.path.join(Data_root, '..')
    data_name = Stats['Pred_data_list']
    for i in range(0, len(data_name)):
        csv_path = os.path.join(root, data_name[i], 'whole_brain_cell_counts/cell-counting.csv')
        save_path = os.path.join(root, data_name[i], 'whole_brain_cell_counts')
        stat.select_level(csv_path, save_path, level=6)
        print('finished {}'.format(data_name[i]))

    ## The total number of cells in different brain regions or the total number of cells in the left and right parts of different regions were exported.
    for i in range(len(data_name)):
        csv_path = os.path.join(root, data_name[i], 'whole_brain_cell_counts/cell-counts-level6.csv')
        save_path = os.path.join(root, data_name[i], 'whole_brain_cell_counts')
        stat.level6_sum(csv_path, save_path)
        # stat.level6_left_right_brain(csv_path, save_path)
        print('finished {}'.format(data_name[i]))
    ## Statistical hypotheses were tested for the number of cells in different brain regions or the number of cells in the left and right parts of different regions.
    exp_name = Stats['Observe_data_list']
    ctrl_name = Stats['Exp_data_list']
    stat.ttest_level6(root, exp_name, ctrl_name)
    stat.ttest_level6_left_right(root, fs_name, ctrl_name)
    ## Exported statistical hypothesis test signs for the number of cells in the left and right brain halves.
    stat.symbol_left_right_ci(csv_path=root, save_csv=root, select_left=False)
    ## Statistical hypothesis tests were performed on the volume ratio of left and right brain cell halves.
    stat.ttest_multi_region_vol(root, fs_name, ctrl_name)
    ## Results were derived for the number of cells in brain regions at different significant levels.
    symbol = Stats['symbol']
    pvals_cutoff = Stats['pvals_cutoff']
    for i in range(len(data_name)):
        save_path = os.path.join(root, data_name[i], 'whole_brain_cell_counts/csv')
        os.makedirs(save_path,exist_ok=True) 
        stat.get_multi_cl_counts(root, data_name[i], save_path, symbol, pvals_cutoff)
        print('finished {}'.format(data_name[i]))
    ## Statistical hypothesis testing was performed on the results for different regions of the cerebral cortex and the number of cells each layer.
    ## Results were derived for the number of cells in each layer at different significant levels.
    layer_name = Stats['layer_name']
    save_layer_name = Stats['save_layer_name']
    stats_dtype = Stats['stats_dtype']
    for ln in range(len(layer_name)):
        stat.ttest_multi_layer(root, fs_name, ctrl_name, layer_name[ln], save_layer_name[ln], stats_dtype[2])
        stat.get_layer_ci(root, layer_name[ln], save_layer_name[ln], save_csv=root)
        print(layer_name[ln])
    ## Demonstrate the number of cells in the brain region of interest.
    plot_region_name = Stats['display_region_name']
    for i in range(len(data_name)):
        save_path = os.path.join(root, data_name[i], 'whole_brain_cell_counts/csv')
        os.makedirs(save_path,exist_ok=True) 
        stat.get_plot_region_counts(root, data_name[i], plot_region_name, save_path, pvals_cutoff)
        print('finished {}'.format(data_name[i]))
    ## The number of cells in statistical level 2 brain regions. 
    stat.level2_count(csv_path=root, save_csv=root)
    """
        P value map
    """
    fname = {'FS8':'0490_25', 'Ctrl8':'0490_25'}
    z_length = 648
    ## Heat map of cellular activity in the coronal, horizontal and sagittal planes.
    data_name = Stats['Pred_data_list']
    group_name = list(fname.keys())
    for i in range(len(data_name)):
      data_path = os.path.join(root, data_name[i], 'coordinate')
      save_path = os.path.join(root, data_name[i], 'cell_density')
      pv.cell_heatmap(data_path, save_path, fname=fname[group_name[i]])
      pv.sagittal_cell_heatmap(data_path, save_path, fname[group_name[i]], z_length)
      pv.horizontal_cell_heatmap(data_path, save_path, fname[group_name[i]], z_length)
      print('finished {}'.format(group_name[i]))
    ## Calculate the cell density mapped at 25 microns
    for i in range(len(data_name)):
      for j in range(35):
        index = j*16
        pv.group_cell_heatmap(data_path=os.path.join(root, data_name[i]), save_path=os.path.join(root, data_name[i], 'group_cell_denisty'), id=index)
        if index >= z_length:
          break
        print('finish {}'.format(index))
      print('finished' + data_name[i])
    ## Generate a P-value map.
    group1 = ['', '', ...]
    group2 = ['', '', ...]
    fs_name = Stats['Exp_data_list']
    ctrl_name = Stats['Observe_data_list']
    for i in range(35):
      index = i*16
      group_fs = [os.path.join(root, fs_name[j], 'group_cell_denisty\{}_cell_density.tif'.format(index)) for j in range(len(fs_name))]
      # group1.append(group_fs)
      group_ctrl = [os.path.join(root, ctrl_name[k], 'group_cell_denisty\{}_cell_density.tif'.format(index)) for k in range(len(ctrl_name))]
      # group2.append(group_ctrl)
      save_p_map_path = os.path.join(Data_root, '..', 'I_O_group_p_map')
      save_name = str(index) + '_p_map' 
      pv.generate_p_mapping(group_fs, group_ctrl, save_p_map_path, save_name)
      print('finished {}'.format(index))
    ## Whole brain cell heat map
    csv_path = os.path.join(Save_root, r'whole_brain_cell_counts\Thumbnail_CSV')
    save_path = os.path.join(Save_root, 'cell_density')
    save_name = 'O6'
    pv.whole_brain_cell_heatmap(csv_path, save_path, z_length=len(os.listdir(csv_path)), save_name=save_name)

    """
        Calculate the coefficient of variation (CV) and z-score.
    """
    csv_path = Stats['level6_cell_counts_path']
    index = Stats['group_index']
    cz.coefficient_of_variation(csv_path=csv_path, save_path=csv_path, index=index)
    cz.z_score(csv_path=csv_path, save_path=csv_path, index=index)
    """
        Calculate the log2(fold change).
    """
    root = Data_root
    exp_name = Stats['Exp_data_list'] 
    obser_name = Stats['Observe_data_list'] 
    con_name = Stats['Ctrl_data_list']
    lc.log2change(root, exp_name, obser_name, con_name) 

    """
        Calculate the spearman's correlation.
    """
    data_root = Data_root
    save_root = Data_root
    sc.spearman_corr(data_root, save_root)