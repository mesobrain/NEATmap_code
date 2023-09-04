# Synthetic 3d brain slices were cut into 70 patches.
from Environment import *
from Parameter import *

def single_cutting(files, index, save_path, cut_size, cut_index_x, cut_index_y, patch_weight_num, patch_hegiht_num):

    image = sitk.ReadImage(files)
    image_ROI = image[cut_index_x : cut_size*patch_weight_num + cut_index_x, cut_index_y : cut_size*patch_hegiht_num + cut_index_y]
    image_array = sitk.GetArrayFromImage(image_ROI)
    
    num = 1
    start_x, start_y= 0, 0
    crop_width = cut_size
    crop_height = cut_size
    while start_x < image_array.shape[1]:
        end_x = min(start_x + crop_width, image_array.shape[1])
        end_y = start_y + crop_height
        cropped_image = image_array[:, start_x:end_x, start_y:end_y]
        cropped_sitk_image = sitk.GetImageFromArray(cropped_image)
        os.makedirs(os.path.join(save_path, 'patchimage{}'.format(index)), exist_ok=True)
        sitk.WriteImage(cropped_sitk_image, os.path.join(save_path, 'patchimage{}'.format(index), 'Z{:05d}_patch_{}.tif'.format(index, num)))
        num += 1
        start_y += crop_height
        if start_y >= image_array.shape[2]:
            start_y = 0
            start_x += crop_width

def make_up_3d(total_num, name, ind, save_root, temp_path, z_num, cut_label):
    j = 1
    new = []
    for i in range(1, total_num + 1):
        for j in range(1, z_num + 1):
            patch_data_path = os.path.join(temp_path, 'single_cutting', '{}'.format(j))
            file = os.path.join(patch_data_path, 'image_{}.tif'.format(i))
            image = sitk.ReadImage(file)
            array = sitk.GetArrayFromImage(image)
            new.append(array)

        tif = np.array(new)
        tif = sitk.GetImageFromArray(tif)
        if cut_label:
            save_path = save_root + '/patchseg{}/'.format(ind)
            os.makedirs(save_path, exist_ok=True)
            sitk.WriteImage(tif, save_path + name + '_patch_seg_{}.tif'.format(i))
        else:
            save_path = save_root + '/patchimage{}/'.format(ind)
            os.makedirs(save_path, exist_ok=True)
            sitk.WriteImage(tif, save_path + name + '_patch_{}.tif'.format(i))
        new = []
        print('finished {} patch 3dimage'.format(i))

def cut(root, cut_size, channel, cut_index_x, cut_index_y, patch_weight_num, 
            patch_hegiht_num, train_path=None, label_path=None, cut_label=False):
    if cut_label:
        data_path = os.path.join(root, 'brain_label_64_' + channel)
        save_path = os.path.join(root, 'PatchSeg_' + channel)
    else:
        data_path = os.path.join(root, 'brain_image_64_' + channel)
        save_path = os.path.join(root, 'PatchImage_' + channel)
    if train_path is not None:
        save_path = os.path.join(root, 'train_image')
    if label_path is not None:
        save_path = os.path.join(root, 'train_label')
    ind = 1

    for i in range(1, len(os.listdir(data_path)) + 1):
        if cut_label:
            name = 'Z{:05d}_seg'.format(i)
        else:
            name = 'Z{:05d}'.format(i)

        image = os.path.join(data_path, name + '.tif')
        single_cutting(image, save_path, cut_size, cut_index_x, cut_index_y, patch_weight_num, patch_hegiht_num)
        ind += 1

if __name__=="__main__":
    cut(root=Data_root, cut_size=Preprocessing['cut_size'],  channel=Channels['staining'],
        cut_index_x=Preprocessing['cut_index_x'], cut_index_y=Preprocessing['cut_index_y'], 
        patch_weight_num=Preprocessing['patch_weight_num'], patch_hegiht_num=Preprocessing['patch_height_num'],
        train_path=Train_config['train_image_path'], label_path=Train_config['train_label_path'])