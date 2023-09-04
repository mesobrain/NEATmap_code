from PIL import Image
from Environment import *
from Parameter import *
from Splice_post.utils import segmentation_quick_view

def concat(image_list, save_path, index):
    start_x, start_y= 0, 0
    restored_array = np.zeros((int(Preprocessing['z_num']), int(Preprocessing['patch_weight_num'])*int(Preprocessing['cut_size'] ), 
                                int(Preprocessing['patch_height_num'])*int(Preprocessing['cut_size'] )), dtype=np.uint16)
    for i in range(len(image_list)):
        crop_shape = image_list[i][1].shape
        restored_array[:, start_x:start_x+crop_shape[1], start_y:start_y+crop_shape[2]] = image_list[i][1].transpose([0, 2, 1])

        start_x += crop_shape[1]
        if start_x >= int(Preprocessing['patch_weight_num'])*int(Preprocessing['patch_height_num']):
            start_x = 0
            start_y += crop_shape[1]

    depth, row, col = restored_array.shape
    x_zeros_up = np.zeros((depth, row, int(Preprocessing['cut_index_y'])))
    x_zeros_down = np.zeros((depth, row, int(Brain['height']) - (restored_array.shape[2] + int(Preprocessing['cut_index_y']))))
    y_zeros = np.zeros((depth, (int(Brain['weight']) - row) // 2, int(Brain['height'])))
    img_resize = np.concatenate((restored_array, x_zeros_down), axis = 2)
    img_resize = np.concatenate((x_zeros_up, img_resize), axis = 2)
    img_resize = np.concatenate((img_resize, y_zeros), axis = 1)
    img_resize = np.concatenate((y_zeros, img_resize), axis = 1)
    img_resize = img_resize.transpose([0, 2, 1])
    restored_image = sitk.GetImageFromArray(img_resize)
    restored_image = sitk.Cast(restored_image, sitk.sitkUInt16)

    sitk.WriteImage(restored_image, os.path.join(save_path, 'Z{:05d}_seg.tif'.format(index)))


def load(path, index, total_patch_num):
    images = []
    for i in range(1, total_patch_num + 1):
        image_path = os.path.join(path, 'Z{:05d}_patch_{}_pred.tif'.format(index, i))   
        image = sitk.ReadImage(image_path)
        array = sitk.GetArrayFromImage(image)
        images.append((image_path, array)) 
    sorted_images = sorted(images, key=lambda x: int(re.search(r'\d+', x[0]).group()))
    return sorted_images
    
def load_image(file):
    image = sitk.ReadImage(file)
    array = sitk.GetArrayFromImage(image)
    image_PIL = Image.fromarray(array)

    return image_PIL

def make_up_3d(restore_save_path, save_path_3d, num, seg=False):

    j = 1
    new = []
    for j in range(1, Preprocessing['z_num'] + 1):
        patch_data_path = restore_save_path
        file = os.path.join(patch_data_path, 'image_{}.tif'.format(j))
        image = sitk.ReadImage(file)
        array = sitk.GetArrayFromImage(image)
        new.append(array)

    result = np.array(new)
    if seg:
        result = segmentation_quick_view(result)
    result = sitk.GetImageFromArray(result)
    sitk.WriteImage(result, save_path_3d + '/Z{:05d}_seg.tif'.format(num))

def create_residual_image(save_path, total_num, resuidual_z):

    resuidual_array = np.zeros((resuidual_z, Brain['height'], Brain['weight']))
    resuidual_image = sitk.GetImageFromArray(resuidual_array)
    resuidual_image = sitk.Cast(resuidual_image, sitk.sitkUInt16)
    sitk.WriteImage(resuidual_image, save_path + '/Z{:05d}_seg.tif'.format(total_num))
   
if __name__ == '__main__':
    raw_path = Raw_data_root
    path = os.path.join(Save_root, 'brain_image_64_' + Channels['staining'])
    save_path_3d = os.path.join(Splicing['save_splicing_path'], 'whole_brain_pred_' + Channels['staining'])
    os.makedirs(save_path_3d, exist_ok=True)
    with open(raw_path) as f:
        brain = json.load(f)
        file = brain['images']
        raw_total_num = len(file)
    total_num = len(os.listdir(path))
    total_num = len(os.listdir(path))
    resuidual_z = raw_total_num - (Preprocessing['z_num'] * total_num)
    patch_total_num = Preprocessing['patch_weight_num'] * Preprocessing['patch_height_num']
    strat_index = Preprocessing['z_num'] * total_num + Brain['z_num']
    with tempfile.TemporaryDirectory() as TEMP_PATH:
        for num in range(1, len(os.listdir(path)) + 1):
            for ind in range(1, Preprocessing['z_num'] + 1):
                path = 'D:/UserData/weijie/whole_brain_split' + '/split_image{}'.format(num) + '/{}'.format(ind)
                list = load(path, num=patch_total_num)
                save_path = concat(list, ind, num, cut_num_x=Preprocessing['patch_weight_num'], 
                                    cut_num_y=Preprocessing['patch_height_num'], temp_path=TEMP_PATH)
                print('finished {} frame '.format(ind))
            make_up_3d(save_path, save_path_3d, num, seg=True)
            print('finished {} image'.format(num))
        create_residual_image(save_path_3d, total_num + 1, resuidual_z, strat_index, raw_total_num)