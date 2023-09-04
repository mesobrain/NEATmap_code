from Environment import *
from Splice_post.utils import segmentation_quick_view
from Parameter import *

def concat_image(path, save_path, z_num, save_name):

    os.makedirs(save_path, exist_ok=True)
    concat_image = []
    for i in range(z_num):
        if not os.path.exists(os.path.join(path, '{:04d}_25.tif'.format(i))):
            array = np.zeros((400, 560)).astype('uint16')
            tifffile.imwrite(os.path.join(path, '{:04d}_25.tif'.format(i)), array)
        else:
            image = sitk.ReadImage(os.path.join(path, '{:04d}_25.tif'.format(i)))
            array = sitk.GetArrayFromImage(image)
        concat_image.append(array)
        print('finished {} image'.format(i))
    concat_image = np.array(concat_image)
    save_image = sitk.GetImageFromArray(concat_image)
    sitk.WriteImage(save_image, os.path.join(save_path, save_name + '.tif'))

def getMaskSinge(image):
    """
    Process single image, Get atlas from freessia export, remove points.
    :param image:
    :return:
    """
    atlas_mask = image
    x,y = np.where(atlas_mask==1000)
    # print(x,y)
    neighbor_8 = [(0,-1),(0,1), (1,0),(1,1),(1,-1), (-1,0),(-1,-1),(-1,1)]
    for i in range(len(x)):
        xx = x[i]
        yy = y[i]
        bg = []
        for neighbor in neighbor_8:
            # print(neighbor[0],neighbor[1])
            bgs = atlas_mask[xx+neighbor[0]][yy+neighbor[1]]
            # print(bgs)
            bg.append(bgs)
        counts = np.bincount(bg)
        a = np.argmax(counts)
        atlas_mask[xx][yy] = a
        # print(a)
    return atlas_mask

def regenAtlas(image_root):
    """
        Get atlas from freessia export, remove points.
        :param image:
        :return:
        """
    atlas_root = os.path.join(image_root,'..','atlas')
    image_list = os.listdir(image_root)
    if not os.path.exists(atlas_root):
        os.mkdir(atlas_root)
    for i in range(0, len(image_list)):
        print('img '+image_list[i])
        image = tifffile.imread(os.path.join(image_root, image_list[i]))
        atlas_mask = getMaskSinge(image)

        # print(atlas_mask)
        tifffile.imwrite(os.path.join(atlas_root, image_list[i]),atlas_mask)

def label_parse(lable_root):
    """
    parse the WHS rat atlas label.
    :param lable_root:
    :return:
    """
    with open(lable_root, 'r') as f:
        lines = f.readlines()
    name_index = []
    for i in range(15, len(lines)):
        line = lines[i]
        id  = line.strip().split(' ')[0]
        index = line.find('\"')
        # print(id, line[index+1:-1-1])
        name_index.append((id, line[index+1:-1-1]))
    print(name_index)
    return name_index

def genEdgefromAtlas(atlas_root,Edge_root):
    """
    Generate brain atlas' edges from atlas greyscale image.
    :param atlas_root:
    :param Edge_root:
    :return:
    """
    if not os.path.exists(Edge_root):
        os.mkdir(Edge_root)
    atlas_list = os.listdir(atlas_root)
    for k in range(len(atlas_list)):
        print(os.path.join(atlas_list[k]))
        atlas = tifffile.imread(os.path.join(atlas_root,atlas_list[k]))
        atlas_new = atlas.copy() #copy
        neighbor_8 = [(0, -1), (0, 1), (1, 0), (1, 1), (1, -1), (-1, 0), (-1, -1), (-1, 1)]
        # print(atlas.shape[0]-1, atlas.shape[1]-1)
        for i in range(1, atlas.shape[0]-1):
            for j in range(1, atlas.shape[1]-1):
                bg = []
                for neighbor in neighbor_8:
                    bgs = atlas[i+neighbor[0]][j+neighbor[1]]
                    # print(bgs,i+neighbor[0],j+neighbor[1])
                    bg.append(bgs)
                # print(i,j,bg,len(np.unique(bg)))
                if len(np.unique(bg))==1:
                    atlas_new[i][j] = 0
        tifffile.imwrite(os.path.join(Edge_root,atlas_list[k]),atlas_new)

def genPoints(image_root,Points_root):
    """
    Generate points from freessia export.
    :param image_root:
    :param Points_root:
    :return:
    """
    if not os.path.exists(Points_root):
        os.mkdir(Points_root)
    image_list = os.listdir(image_root)
    for i in range(len(image_list)):
        print(os.path.join(image_root,image_list[i]))
        image = tifffile.imread(os.path.join(image_root,image_list[i]))
        image_new = np.where(image==1000, 255, 0)
        tifffile.imwrite(os.path.join(Points_root,image_list[i]),image_new.astype('uint16'))

def genSingleRegion(atlas_root, Region_root, Region_name, Mask_id):
    """
    Using label index, generate single brain region, like hippocampus, cortex, etc.
    :param atlas_root:
    :param Region_root:
    :param Region_name:
    :param Mask_id:
    :return:
    """
    if not os.path.exists(Region_root):
        os.mkdir(Region_root)
    atlas_list = os.listdir(atlas_root)
    atlas_list.sort()
    shape = tifffile.imread(os.path.join(atlas_root, atlas_list[0])).shape
    atlas_3d = np.zeros((len(atlas_list), shape[0], shape[1])).astype('uint16')
    for i in range(len(atlas_list)):
        atlas_3d[i,:,:] = tifffile.imread(os.path.join(atlas_root, atlas_list[i])).astype('uint16')

    region_3d = np.zeros((len(atlas_list), shape[0], shape[1])).astype('uint16')
    for i in range(len(Mask_id)):
        tmp = np.where(atlas_3d==Mask_id[i],255,0).astype('uint16')
        region_3d+=tmp
    tifffile.imwrite(os.path.join(Region_root,'Region_' + Region_name + '.tif'), region_3d.astype('uint16'))
    print('finished')

def genPointsSingleRegion(image_root, Region_root, Points_SingleRegion_root, Region_name):
    """
    Generate points inside the region mask.
    :param image_root:
    :param Region_root:
    :param Points_SingleRegion_root:
    :param Region_name:
    :return:
    """
    if not os.path.exists(Points_SingleRegion_root):
        os.mkdir(Points_SingleRegion_root)
    img_list = os.listdir(image_root)
    img_list.sort()
    shape = tifffile.imread(os.path.join(image_root, img_list[0])).shape
    image_3d = np.zeros((len(img_list),shape[0], shape[1])).astype('uint16')
    for i in range(len(img_list)):
        image_3d[i,:,:] = tifffile.imread(os.path.join(image_root, img_list[i])).astype('uint16')

    Region_3d = tifffile.imread(os.path.join(Region_root,'Region_' + Region_name + '.tif'))
    Points_3d = np.where(image_3d == 1000,1,0)&Region_3d
    Points_3d = segmentation_quick_view(Points_3d)

    tifffile.imwrite(os.path.join(Points_SingleRegion_root,'Points_Region_' + Region_name + '.tif'),Points_3d.astype('uint16'))

    print('finished')

def merge(root):
    im_lst = os.listdir(root)
    shape = tifffile.imread(os.path.join(root,im_lst[0])).shape
    image = np.zeros((shape[0],shape[1]))
    for i in range(len(im_lst)):
        tmp = tifffile.imread(os.path.join(root,im_lst[i]))
        image += tmp
    image[image>1] = 255
    tifffile.imwrite(os.path.join(root, '..','merge_points.tif'),image.astype('uint8'))

def genMergeRaw(Edge_root, Points_root, Save_root, Snapshot_id):
    if not os.path.exists(Save_root):
        os.mkdir(Save_root)
    for i in range(len(Snapshot_id)):
        dir = os.path.join(Save_root,str(i).zfill(2))
        if not os.path.exists(dir):
            os.mkdir(dir)
        point_save_dir = os.path.join(dir, "Points")
        if not os.path.exists(point_save_dir):
            os.mkdir(point_save_dir)
        Edge_name = str(Snapshot_id[i]).zfill(4)+'_25.tif'
        print(Edge_name)
        shutil.copyfile(os.path.join(Edge_root,Edge_name),os.path.join(dir,Edge_name))
        for offset in range(0,8):
            if i >= 5:
                offset = -offset
            Point_index = Snapshot_id[i] - offset
            Point_name = str(Point_index).zfill(4) +'_25.tif'
            shutil.copyfile(os.path.join(Points_root, Point_name), os.path.join(point_save_dir, Point_name))
        merge(point_save_dir)
        raw = tifffile.imread(os.path.join(Edge_root,Edge_name))
        tifffile.imwrite(os.path.join(dir,Edge_name),raw.astype('uint8'))

if __name__ == "__main__":
    # freesia export cell counts
    image_root = os.path.join(freesia_export_path, 'images')
    regenAtlas(image_root)

    atlas_root = os.path.join(image_root, '..', 'atlas')

    Edge_root = os.path.join(image_root, '..', 'Edge')
    genEdgefromAtlas(atlas_root, Edge_root)

    Points_root = os.path.join(image_root, '..', 'Points')
    genPoints(image_root,Points_root)

    Region_name = 'CBX'
    Mask_id = Whole_brain_region_level2[Region_name]#CBX
    
    Region_root = os.path.join(image_root, '..', 'Region')
    Points_SingleRegion_root = os.path.join(image_root, '..', 'Points_Single_Region')
    genSingleRegion(atlas_root, Region_root, Region_name, Mask_id)
    genPointsSingleRegion(image_root, Region_root, Points_SingleRegion_root, Region_name)
    save_path = os.path.join(Points_root, '..', '3d_result')
    save_name = 'whole_brain_point'
    altas_num = Preprocessing['brain_total_num'] / Spot_map['group_num']
    concat_image(Points_root, save_path, z_num=altas_num, save_name=save_name)

    Save_root = Atlas_edge_pionts_save_path
    genMergeRaw(Edge_root, Points_root, Save_root, Atlas_snapshot_id)
             