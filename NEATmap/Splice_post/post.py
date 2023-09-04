from Environment import *
from scipy import ndimage as ndi
from skimage import measure
from Splice_post.utils import segmentation_quick_view
from Parameter import *

def post_488nm(pred_result, seg_488nm):

    seg_488nm[seg_488nm == 255] = 300
    result = pred_result + seg_488nm
    result[result == 555] = 0
    result[result == 300] = 0

    return result

def remove_piont(image, min_size, connectivity=1, in_place=False):
    if in_place:
        out = image
    else:
        out = image.copy()

    if min_size == 0:  
        return out

    if out.dtype == bool:
        selem = ndi.generate_binary_structure(image.ndim, connectivity)
        ccs = np.zeros_like(image, dtype=np.int32)
        ndi.label(image, selem, output=ccs)
    else:
        ccs = out
    component_sizes = np.bincount(ccs.ravel())

    too_small = component_sizes <= min_size
    too_small_mask = too_small[ccs]
    out[too_small_mask] = 0

    return out

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

def big_object_filter(filter_matrix, seg, limit):

    filter_big_seg = seg
    bool_matrix = filter_matrix >= limit
    index_matrix = np.where(bool_matrix == True)
    for ind in range(len(index_matrix[0])):
        filter_big_seg[index_matrix[0][ind], index_matrix[1][ind], index_matrix[2][ind]] = 0

    return filter_big_seg

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

def intensity_filter(image, label,  min_volume=Stats['min_volume'], max_volume=Stats['max_volume']):

    raw = image.copy()
    mask = label.copy
    struct = ndi.generate_binary_structure(3,1)
    mask, _ = ndi.label(mask, struct)
    unique, counts = np.unique(mask, return_counts=True)
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
    object_ids = [e[0] for e in medium]
    if object_ids: # skip if empty
        # center_mass = ndi.center_of_mass(raw, label, object_ids)
        labeled_img = measure.label(mask, connectivity=1)
        properties = measure.regionprops(labeled_img)
        # coms = np.array(center_mass).round().astype(np.int)
        for i, pro in enumerate(properties):
            this_idx = object_ids[i]
            centroid = pro.centroid
            coord = pro.coords
            deltaI, bg = get_intensity(raw, mask, this_idx, centroid, mode='obj_mean')
            intensiy = deltaI + bg
            if intensiy <= 195:
                mask[[cor[0] for cor in coord], [cor[1] for cor in coord], [cor[2] for cor in coord]] == 0

    return mask


def spot_filter(image_path, pred_path, path_488nm, save_path, lower_limit, upper_limit, min_size, max_intensity):

    os.makedirs(save_path, exist_ok=True)
    for i in range(1, len(os.listdir(pred_path)) + 1):
        if i < len(os.listdir(pred_path)):
            img = tifffile.imread(os.path.join(image_path, 'Z{:05d}.tif'.format(i)))
            seg = tifffile.imread(os.path.join(pred_path, 'Z{:05d}_seg.tif'.format(i)))
            seg_488nm = tifffile.imread(os.path.join(path_488nm, 'Z{:05d}_seg.tif'.format(i)))
            post_seg = post_488nm(seg, seg_488nm)
            resMatrix = remove_piont(post_seg>0, min_size)
            resMatrix = segmentation_quick_view(resMatrix)
            new_seg = intensity_filter(img, resMatrix)
            # filter_matrix = img.astype(np.float32) - resMatrix.astype(np.float32)
            # bool_matrix = np.logical_and(filter_matrix >= lower_limit, filter_matrix <= upper_limit)
            # index_matrix = np.where(bool_matrix == True)

            # for j in range(len(index_matrix[0])):
            #     resMatrix[index_matrix[0][j], index_matrix[1][j], index_matrix[2][j]] = 0
            
            # new_seg = big_object_filter(filter_matrix, resMatrix, limit=max_intensity)
        else:
            new_seg = tifffile.imread(os.path.join(pred_path, 'Z{:05d}_seg.tif'.format(i)))
        tifffile.imwrite(os.path.join(save_path, 'Z{:05d}_filter.tif'.format(i)), new_seg.astype('uint16'))
        print('finished {} segmentation filter'.format(i))

if __name__ == "__main__":
    start = time.time()
    image_path = Save_root
    path = Splicing['save_splicing_path']
    pred_path = os.path.join(path, 'whole_brain_pred_' + Channels['staining'])
    path_488nm = os.path.join(path, 'whole_brain_pred_' + Channels['autofl'])
    save_path = os.path.join(path, 'whole_brain_pred_post_filter')
    spot_filter(image_path, pred_path, path_488nm, save_path, lower_limit=Post['intensity_lower_differ'], upper_limit=Post['intensity_upper_differ'], min_size=Post['point_min_size'],
                max_intensity=Post['big_object_size'])
    end = time.time()
    print('Running time: %s Seconds'%(end-start))