# from Environment import *
# from Parameter import *
import numpy as np
import pandas as pd
import tifffile, os, csv
from scipy import ndimage as ndi
Stats = {}
Stats['min_volume'] = 2
Stats['max_volume'] = 64

def get_local_voxels(block, zc, yc, xc, rad ):
    """
    """
    xst = xc-rad if xc-rad>=0 else 0
    yst = yc-rad if yc-rad>=0 else 0
    zst = zc-rad if zc-rad>=0 else 0
    xen = xc+rad+1 if xc+rad+1<=block.shape[2] else block.shape[2]
    yen = yc+rad+1 if yc+rad+1<=block.shape[1] else block.shape[1]
    zen = zc+rad+1 if zc+rad+1<=block.shape[0] else block.shape[0]

    return block[ zst:zen, yst:yen, xst:xen ]

def get_intensity(blk, label, obj_id, arr_idx,
                    mode='max', max_rad=1, min_rad=4, min_percentile=5):
    """
    """
    block = blk.copy()
    c0, c1, c2 = arr_idx

    # compute bagkground level
    local = get_local_voxels( block, c0, c1, c2, min_rad )
    bg = np.percentile(local, min_percentile )

    if mode=='max':
        local = get_local_voxels( block, c0, c1, c2, max_rad )
        sig = local.max()
    elif mode=='local_mean':
        local = get_local_voxels( block, c0, c1, c2, max_rad )
        sig = local.mean()
    elif mode=='obj_mean':
        sig = ndi.mean( block, label, obj_id )

    #compute delta
    delta = sig - bg
    if delta < 0:
        delta = 0

    return delta, bg

def intensity(data_path, label_path, region_name, save_csv_root, id, min_volume=Stats['min_volume'], max_volume=Stats['max_volume']):

    image = tifffile.imread(os.path.join(data_path, id + '_whole_brain_25mic.tif'))
    label = tifffile.imread(os.path.join(label_path, 'Points_Region_' + region_name + '.tif'))
    raw = image.copy()
    struct = ndi.generate_binary_structure(3,1)
    label, _ = ndi.label(label, struct)
    unique, counts = np.unique(label, return_counts=True)
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
    detected_object = []
    object_ids = [e[0] for e in medium]
    volumes = [e[1] for e in medium]
    if object_ids: # skip if empty
        center_mass = ndi.center_of_mass(raw, label, object_ids )
        coms = np.array(center_mass).round().astype(np.int)
        for i, com in enumerate(coms):
            this_idx = object_ids[i]
            deltaI, bg = get_intensity(raw, label, this_idx, com, mode='max')
            vol = volumes[i]
            obj = [com[2], com[1], com[0], bg, bg + deltaI, vol] # X, Y, Z, intensity, volume
            detected_object.append(obj)
    csv_name = region_name + '_intensity.csv'
    csv_path = os.path.join(save_csv_root, csv_name)
    list_name = ['X', 'Y', 'Z', 'bg_percentile', 'intensity', 'volume']
    with open(csv_path,'w+',newline='')as f:
        csv_write = csv.writer(f, dialect='excel')
        csv_write.writerow(list_name)
        for k in range(len(detected_object)):
            csv_write.writerow(detected_object[k])
    return print('Finished ' + region_name)

def get_2d_local_voxels(block, yc, xc, rad ):
    """
    """
    xst = xc-rad if xc-rad>=0 else 0
    yst = yc-rad if yc-rad>=0 else 0
    xen = xc+rad+1 if xc+rad+1<=block.shape[1] else block.shape[1]
    yen = yc+rad+1 if yc+rad+1<=block.shape[0] else block.shape[0]

    return block[ yst:yen, xst:xen ]

def get_2d_intensity(blk, label, obj_id, arr_idx,
                    mode='max', max_rad=1, min_rad=4, min_percentile=5):
    """
    """
    block = blk.copy()
    c0, c1 = arr_idx

    # compute bagkground level
    local = get_2d_local_voxels( block, c0, c1, min_rad )
    bg = np.percentile(local, min_percentile )

    if mode=='max':
        local = get_2d_local_voxels( block, c0, c1, max_rad )
        sig = local.max()
    elif mode=='local_mean':
        local = get_2d_local_voxels( block, c0, c1, max_rad )
        sig = local.mean()
    elif mode=='obj_mean':
        sig = ndi.mean( block, label, obj_id )

    #compute delta
    delta = sig - bg
    if delta < 0:
        delta = 0

    return delta, bg

def patch_intensity(data_path, label_path, image_name, label_name, save_csv_root, min_volume=2, max_volume=64):
    image = tifffile.imread(os.path.join(data_path, image_name))
    label = tifffile.imread(os.path.join(label_path, label_name))
    raw = image.copy()
    struct = ndi.generate_binary_structure(2, 1)
    label, _ = ndi.label(label, struct)
    unique, counts = np.unique(label, return_counts=True)
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
    detected_object = []
    object_ids = [e[0] for e in medium]
    volumes = [e[1] for e in medium]
    if object_ids: # skip if empty
        center_mass = ndi.center_of_mass(raw, label, object_ids )
        coms = np.array(center_mass).round().astype(np.int)
        for i, com in enumerate(coms):
            this_idx = object_ids[i]
            deltaI, bg = get_2d_intensity(raw, label, this_idx, com, mode='max')
            vol = volumes[i]
            obj = [com[1], com[0], bg, bg + deltaI, vol] # X, Y, Z, intensity, volume
            detected_object.append(obj)
    csv_name = 'patch_intensity.csv'
    csv_path = os.path.join(save_csv_root, csv_name)
    list_name = ['X', 'Y', 'bg_percentile', 'intensity', 'volume']
    with open(csv_path,'w+',newline='')as f:
        csv_write = csv.writer(f, dialect='excel')
        csv_write.writerow(list_name)
        for k in range(len(detected_object)):
            csv_write.writerow(detected_object[k])
    return print('Finished ')

def get_intensity_csv(region_color, data_path, label_path, save_csv_root, id='FS10'):

    os.makedirs(save_csv_root, exist_ok=True)
    region_name = list(region_color.keys())
    for i in range(len(region_name)):
        intensity(data_path, label_path, region_name=region_name[i], save_csv_root=save_csv_root, id=id)

def normalized_counts(region_color, intensity_root, savepath):
    region_name = list(region_color.keys())
    os.makedirs(savepath, exist_ok=True)
    for i in range(len(region_name)):
        csv_path = os.path.join(intensity_root, region_name[i] + '_intensity.csv')
        file = open(csv_path)
        df = pd.read_csv(file)
        intensity = df['intensity'].values
        # intensity_range = (intensity.min(), intensity.max())
        intensity_range = (180, 780)
        hist, bin_edges = np.histogram(intensity, bins=20,
                                        range=intensity_range,
                                        density=False)
        bins = ( bin_edges[:-1] + bin_edges[1:] ) / 2
        savename = os.path.join(savepath, region_name[i] + '.csv')
        values = pd.DataFrame(columns=['intensity', 'normalized_count'])
        values['intensity'] = bins
        values['normalized_count'] = hist
        values.to_csv(savename, index=False)
        print('finished {}'.format(region_name[i]))

def get_whole_norm_counts(path, region_color, csv_name):
    ''' Obtained Normalized counts of partial cortical regions '''
    region_name = os.listdir(path)
    csv_path = os.path.join(path, '..', csv_name )
    first_line = list(region_color.keys())
    values = pd.DataFrame(columns=first_line)
    for i in range(len(first_line)):
        file = open(os.path.join(path, region_name[i]))
        df = pd.read_csv(file)

        norm_count = df['normalized_count'].values
        values[first_line[i]] = norm_count

    values.to_csv(csv_path, index=False)


if __name__ == "__main__":
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
    root = Data_root= r'R:\WeijieZheng\Forced_swimming\Model_GLX_MHW_FS10_20_1_seg_pred'
    data_path = os.path.join(root, 'Brainimage_25mic')
    label_path = os.path.join(root, '', r'whole_brain_cell_counts\points_cortex_region')
    save_csv_root = os.path.join(root, '', r'whole_brain_cell_counts\cortex_intensity')
    # get_intensity_csv(Isocortex, data_path, label_path, save_csv_root, id='Ctrl10')
    intensity_root = os.path.join(Data_root, '', r'whole_brain_cell_counts\cortex_intensity')
    savepath = os.path.join(intensity_root, '..', 'intensity_normalized_counts')
    normalized_counts(Isocortex, intensity_root, savepath)
    # get_whole_norm_counts(path=savepath, region_color=Isocortex, csv_name='Norm_counts.csv')

    # desktop = ''
    # patch_intensity(data_path=desktop, label_path=desktop, save_csv_root=desktop, min_volume=Stats['min_volume'], max_volume=Stats['max_volume'])