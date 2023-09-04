# 3D-HSFormer parameter
import os

data_name = '150819_0_8X-cfos_16-55-11'
name = data_name.split('_')
if len(name) == 6:
    name_seg = name[1] + str('_') + name[2] + str('_') + name[3] + str('_') + name[4] + str('_') + name[5]
else:
    print('Recheck the name of the data')
Raw_data_root = r'G:\cyx/' + data_name + '/Reconstruction/BrainImage/4.0'
# Data_root = r'R:\WeijieZheng\Data_proc_' + data_name
Data_root = r'R:\WeijieZheng\Haloperidol\haloperidol\1269' + data_name
Save_root = r'R:\WeijieZheng\Model_'+ name_seg +'_seg_pre'

Channels = {}
Channels['autofluo'] = '488nm'
Channels['staining'] = '561nm'
Channels['561nm_index'] = 'C2'
Channels['488nm_index'] = 'C1'

Brain = {}
Brain['z_num'] = 75
Brain['height'] = 2500
Brain['weight'] = 3500
Brain['voxel_size'] = 4

Isocortex = {'FRP' : [14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
                'MOp' : [38, 39, 40, 41, 42, 43, 44, 45, 46, 47],
                'Mos' : [50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
                'SSp-n' : [90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101],
                'SSp-bfd' : [104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115],
                'SSp-ll' : [132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143],
                'SSp-m' : [146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157],
                'SSp-ul' : [160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171],
                'SSp-tr' : [174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185],
                'SSp-un' : [188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199],
                'SSs' : [202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213],
                'AUDd' : [246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257],
                'AUDp' : [274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285],
                'AUDpo' : [288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299],
                'AUDv' : [302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313],
                'VISal' : [330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341],
                'VISam': [344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355],
                'VISI' : [358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369],
                'VISp' : [372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383],
                'VISpl' : [386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397],
                'VISpm' : [400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411],
                'VISli' : [414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425],
                'VISpor' : [428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439],
                'RSPagl' : [598, 599, 600, 601, 602, 603, 604, 605, 606, 607],
                'RSPd' : [652, 653, 654, 657, 658, 659, 660, 661, 662, 663],
                'RSPv' : [666, 667, 670, 671, 672, 673, 674, 675, 676, 677],
                'VISa' : [694, 695, 696, 697, 698, 699, 700, 701, 702, 703, 704, 705],
                'VISrl' : [708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719],
                'ORBI' : [518, 519, 520, 521, 522, 523, 524, 525, 526, 527],
                'ORBm' : [530, 531, 534, 535, 536, 537, 538, 539, 540, 541],
                'ORBvl' : [546, 547, 548, 549, 550, 551, 552, 553, 554, 555],
                'ACAd' : [454, 455, 456, 457, 458, 459, 460, 461, 462, 463], 
                'ACAv' : [466, 467, 468, 469, 470, 471, 472, 473, 474, 475],
                'PL' : [482, 483, 484, 485, 486, 487, 488, 489, 490, 491],
                'ILA' : [492, 493, 496, 497, 498, 499, 500, 501, 502, 503],
                'VISC' : [230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241],
                'GU' : [216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227],
                'Ald' : [560, 561, 562, 563, 564, 565, 566, 567, 568, 569],
                'Alp' : [572, 573, 574, 575, 576, 577, 578, 579, 580, 581],
                'Alv' : [584, 585, 596, 587, 588, 589, 590, 591, 592, 593],
                'PERI' : [736, 737, 738, 739, 740, 741, 742, 743, 744, 755],
                'TEa' : [722, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732, 733],
                'ECT' : [748, 749, 750, 751, 752, 753, 754, 755, 756, 757]}

#The index corresponding to each brain region in the count column of the cell-count.csv exported by freesia.
# FRP: 12-21
# MOp: 26-35
# MOs: 38-47
# SSp-n: 54-65
# SSp-bfd: 68-79
# SSp-ll: 82-93
# SSp-m: 96-107
# SSp-ul: 110-121
# SSp-tr: 124-135
# SSp-un: 138-149
# SSs: 152-163
# AUDd: 196-207
# AUDp: 210-221
# AUDpo: 224-235
# AUDv: 238-249
# VIsal: 254- 265
# VISam: 268-279
# VISI: 282-293
# VISp: 296-307
# VISpl: 310-321
# VISpm: 324-335
# VISli: 338-349
# VISpor: 352-363
# RSPagl: 494-503
# RSPd: 506-515
# RSPv: 518-527
# VISa: 532-543
# VISrl: 546-557
# ORBI: 418-427
# ORBm: 430-439
# ORBvl: 442-451
# ACAd: 368-377
# ACAv: 380-389
# PL: 392-401
# ILA: 404-413
# VISC: 180-191
# GU: 166-177
# Ald: 456-465
# Alp: 468-477
# Alv: 480-489
# PERI: 574-583
# TEa: 560-571
# ECT: 586-595

Preprocessing = {}
Preprocessing['z_num'] = 64
Preprocessing['cut_size'] = 256
Preprocessing['cut_index_x'] = 470
Preprocessing['cut_index_y'] = 245
Preprocessing['patch_weight_num'] = 10
Preprocessing['patch_height_num'] = 7

Spot_seg_config = {}
Spot_seg_config['scaling_param'] = [1, 40]
Spot_seg_config['gaussian_smoothing_sigma'] = 1
Spot_seg_config['wrapper_param'] = [[1, 0.01], [1, 0.006]] # [[mu1, sigma1], [mu2, sigma2], ...]
Spot_seg_config['save_seg_path'] = ''

Train_config = {}
Train_config['train_total_num'] = 40
Train_config['train_image_path'] = None
Train_config['train_label_path'] = None
Train_config['train_data_path'] = ''

Test_config = {}
Test_config['test_id'] = 30
Test_config['test_path'] = os.path.join('', 'brain_label_64')
Test_config['test_image_path'] = ''
Test_config['test_label_path'] = ''
Test_config['test_save_path'] = ''

Network = {}
Network['train_data'] = ''
Network['test_slice'] = ''
Network['train_valid_ratio'] = 0.9
Network['test_whole_brain'] = ''
Network['save_data_list'] = ''
Network['lr'] = 0.01
Network['max_epochs'] = 10
Network['num_classes'] = 2
Network['save_checkpoint'] = ''
Network['train_batch_size'] = 2
Network['valid_batch_size'] = 1
Network['test_batch_size'] = 1
Network['train_shuffle'] = True
Network['valid_shuffle'] = False
Network['train_num_workers'] = 4
Network['valid_num_workers'] = 1
Network['pin_memory'] = True
Network['loss_ratio'] = 0.5
Network['pre_train'] = False
Network['Pre_train_path'] = ''
Network['save_patch_metrics'] = ''
Network['save_metrics'] = ''
Network['z_spacing'] = 1
Infer = {}
Infer['snapshot_path'] = '/home/weijie/VISoRMap/Network/checkpoint/swin_T_checkpoint_22_01_09'
Infer['whole_brain_path'] = '/data/weijie/Test_VISoRMap/save'
Infer['whole_brain_list'] = '/home/weijie/VISoRMap/Network/whole_brain_lists'
Infer['swin_in_channels'] = 16

Optimizer = {}
Optimizer['momentum'] = 0.9
Optimizer['SGD_weight_decay'] = 0.0001
Optimizer['Adam_weight_decay'] = 0.005
Optimizer['AdamW_weight_decay'] = 5E-2

Splicing = {}
Splicing['whole_predications_path'] = r'R:\WeijieZheng\Data_prco_20210824_GLX_MHW_C5_13_1'
Splicing['checkpoint_name'] = 'swin_T_checkpoint_22_01_09'
Splicing['save_root'] = Save_root
Splicing['save_splicing_path'] = Save_root

Post = {}
Post['intensity_lower_differ'] = -95
Post['intensity_upper_differ'] = -47
Post['point_min_size'] = 1
Post['big_object_size'] = 3000

Registration = {}
Registration['parameters_root'] = 'R:/WeijieZheng/VISoRMap_Code/VISoRMap/Registration/parameters'
Registration['Raw_brain_path'] = r'U:\VISoRData\MHW\MHW-SD-B2\MHW-SD-B2-part2\20210830_GLX_MHW_C9_25_1'
Registration['output_path'] = r'R:\WeijieZheng\Model_GLX_MHW_C9_25_1_seg_pre'
Registration['output_name'] = 'C9'
Registration['registration_param'] = ['tp_brain_registration_rigid.txt','tp_brain_registration_bspline.txt']
Registration['inverse_param'] = 'tp_inverse_bspline.txt'

Spot_map = {}
Spot_map['group_num'] = 6.25
Spot_map['filter_area_lower'] = 1
Spot_map['filter_area_upper'] = 27

freesia_export_path = r'R:\WeijieZheng\Model_'+ name_seg +'_seg_pre\whole_brain_cell_counts'
Region_mask = {
    'Hippo': [914, 915, 924, 925, 934, 935, 948, 949, 950, 951, 952, 953],
    'Central_amy_nuc': [846, 847, 848, 849, 851, 854, 855, 864, 865, 876, 877],
    'nuc_accumbens': [1150, 1151],
    'infralimbic_area': [490, 491, 492, 493, 496, 497, 498, 499, 500, 501],
    'BST': [1242, 1243], 
    'MPN': [1552, 1553],
    'VTA': [1646, 1647],
    'ACB': [1150, 1151],
    'SFO': [1506, 1507],
    'FRP': [i + 1 for i in range(12, 22)],
    'Mop': [i + 1 for i in range(37, 47)]
}
Whole_brain_region_level2 = {
    'Isocortex': [i + 1 for i in range(12, 757)],
    'OLF': [i + 1 for i in range(759, 899)],
    'HPF': [i + 1 for i in range(913, 1109)],
    'CTXsp': [i + 1 for i in range(1113, 1139)],
    'STR': [i + 1 for i in range(1145, 1201)],
    'PAL': [i + 1 for i in range(1219, 1277)],
    'TH': [i + 1 for i in range(1287, 1331)],
    'HY': [i + 1 for i in range(1433, 1611)],
    'MB': [i + 1 for i in range(1617, 1763)],
    'P': [i + 1 for i in range(1769, 1869)],
    'MY': [i + 1 for i in range(1873, 2027)],
    'CBX': [i + 1 for i in range(2049, 2153)]
}

Atlas_snapshot_id = [109,156,179,245,339,394,414,435,459,493,488]
Atlas_edge_pionts_save_path = ''

Stats = {}
Stats['Pred_data_list'] = ['Model_SLJ_MHW_I3_8_1_seg_pre', 'Model_GLX_MHW_O3_9_1_seg_pre', 'Model_GLX_MHW_C3_7_1_seg_pre',
                            'Model_SLJ_WH_I4_11_1_seg_pre', 'Model_GLX_MHW_O4_12_1_seg_pre', 'Model_SLJ_WH_C4_10_1_seg_pre',
                            'Model_SLJ_WH_I5_14_1_seg_pre', 'Model_GLX_MHW_O5_15_1_seg_pre', 'Model_GLX_MHW_C5_13_1_seg_pre',
                            'Model_GLX_MHW_I6_17_1_seg_pre', 'Model_GLX_MHW_O6_18_1_seg_pre', 'Model_GLX_MHW_C6_16_1_seg_pre',
                            'Model_GLX_MHW_I7_20_1_seg_pre', 'Model_SLJ_MHW_O7_21_1_seg_pre', 'Model_GLX_MHW_C7_19_1_seg_pre',
                            'Model_GLX_MHW_I8_23_1_seg_pre', 'Model_GLX_MHW_O8_24_1_seg_pre', 'Model_SLJ_MHW_C8_22_1_seg_pre']
Stats['Exp_data_list'] = ['Model_SLJ_MHW_I3_8_1_seg_pre', 'Model_SLJ_WH_I4_11_1_seg_pre', 'Model_SLJ_WH_I5_14_1_seg_pre',
                            'Model_GLX_MHW_I6_17_1_seg_pre', 'Model_GLX_MHW_I7_20_1_seg_pre', 'Model_GLX_MHW_I8_23_1_seg_pre']
Stats['Observe_data_list'] = ['Model_GLX_MHW_O3_9_1_seg_pre', 'Model_GLX_MHW_O4_12_1_seg_pre', 'Model_GLX_MHW_O5_15_1_seg_pre',
                                'Model_GLX_MHW_O6_18_1_seg_pre', 'Model_SLJ_MHW_O7_21_1_seg_pre', 'Model_GLX_MHW_O8_24_1_seg_pre']
Stats['Ctrl_data_list'] = ['Model_GLX_MHW_C3_7_1_seg_pre', 'Model_SLJ_WH_C4_10_1_seg_pre', 'Model_GLX_MHW_C5_13_1_seg_pre',
                            'Model_GLX_MHW_C6_16_1_seg_pre', 'Model_GLX_MHW_C7_19_1_seg_pre', 'Model_SLJ_MHW_C8_22_1_seg_pre']
Stats['position_group'] = 13
Stats['position_name'] = ''
Stats['DNN_results'] = ''
Stats['Model_name'] = ['swin_T', 'fcn', 'hrnet_fuse', 'hrnet_keep', 'hrnet_multi', 'resnet34', 'unet', 'aspp_unet']
Stats['symbol'] = ['***', '**', '*', 'ns']
Stats['pvals_cutoff'] = ['0.001', '0.01', '0.05', 'ns']
Stats['layer_name'] = ['layer 1', 'layer 2_3', 'layer 4', 'layer 5', 'layer 6']
Stats['save_layer_name'] = ['layer_1', 'layer_2_3', 'layer_4', 'layer_5', 'layer_6']
Stats['stats_dtype'] = ['cortex_cells_counts', 'layer_volume', 'layer_density']
Stats['display_region_name'] = ['MBO', 'PG', 'TT', 'PVHd', 'VLPO', 'SOC', 'DR', 'HIP', 'FRP', 'PMv', 'IPN', 'IO','SUT', 'PAA', 
                        'ACB', 'SI', 'NTB', 'PD', 'CP', 'BMAa', 'AAA', 'PCG', 'PVp', 'LRN', 'RPA', 'SCH', 'ILM', 'MDRN', 
                        'NLL', 'AOB', 'SCig', 'RSP', 'PAS', 'DTN', 'BMAp', 'PeF', 'ISN', 'RHP',	'OT', 'SPVC','GPi', 'PSV',	
                        'ND',  'MPN', 'GENd','PDTg', 'SG', 'FS', 'TU', 'AUD',	'PVa',	'NPC',	'PC5']

Stats['ctrl_data_name'] = ['ctrl6', 'ctrl7', 'ctrl8', 'ctrl9', 'ctrl10', 'ctrl11']
Stats['fst_data_name'] = ['fst6', 'fst7', 'fst8', 'fst9', 'fst10', 'fst11']

Stats['con_data_name'] = ['CON3', 'CON4', 'CON5', 'CON6', 'CON7', 'CON8']
Stats['sds_data_name'] = ['SDS3', 'SDS4', 'SDS5', 'SDS6', 'SDS7', 'SDS8']
Stats['es_data_name'] = ['ES3', 'ES4', 'ES5', 'ES6', 'ES7', 'ES8']

Stats['min_volume'] = 2
Stats['max_volume'] = 64

Stats['level6_cell_counts_path'] = r'R:\WeijieZheng\Cell_counts_level6'
Stats['group_data_name'] = ['CON3', 'CON4', 'CON5', 'CON6', 'CON7', 'CON8',
                            'SDS3', 'SDS4', 'SDS5', 'SDS6', 'SDS7', 'SDS8']
# Stats['group_data_name'] = ['CON3', 'CON4', 'CON5', 'CON6', 'CON7', 'CON8',
#                             'ES3', 'ES4', 'ES5', 'ES6', 'ES7', 'ES8']
Stats['group_index'] = ['3', '4', '5', '6', '7', '8']
