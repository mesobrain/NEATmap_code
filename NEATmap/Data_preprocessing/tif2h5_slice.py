# Test the preparation of a brain slice data.
from Environment import *
from Parameter import *

def slice_test_data(image_patch_path, label_patch_path, channel, save_test_path, patch_num, test_id):
    for i in range(1, patch_num + 1):
        file_path = os.path.join(image_patch_path, 'PatchImage_' + channel, 'patchimage' + test_id + '/Z{:05d}'.format(test_id) + '_patch_{}.tif'.format(i))
        label_file = os.path.join(label_patch_path, 'PatchSeg_' + channel, 'patchseg' + test_id + '/Z{:05d}'.format(test_id) + '_patch_seg_{}.tif'.format(i))
        img, label = sitk.ReadImage(file_path), sitk.ReadImage(label_file)
        img, label = sitk.Cast(img, sitk.sitkFloat32), sitk.Cast(label, sitk.sitkFloat32)
        img_array, label_array = sitk.GetArrayFromImage(img), sitk.GetArrayFromImage(label)
        img_array, label_array = img_array, label_array
        label_array[label_array == 255] = 1
        img_array, label_array = img_array.transpose(0, 2, 1), label_array.transpose(0, 2, 1)

        save_path = os.path.join(save_test_path, 'brain_test_' + test_id) 
        os.makedirs(save_path, exist_ok=True)
        with h5py.File(save_path + '/Z{:05d}'.format(test_id) + '_patch_{}.h5'.format(i), 'w') as h5file:
            h5file.create_dataset('image', data=img_array, dtype='float32')
            h5file.create_dataset('label', data=label_array, dtype='float32')
            h5file.create_dataset('case_name', data='Z{:05d}'.format(test_id) + '_patch_{}'.format(i))
            h5file.close()
        print('finished {} h5'.format(i))

if __name__ == "__main__":
    patch_num = Preprocessing['patch_weight_num'] * Preprocessing['patch_height_num']
    slice_test_data(image_patch_path=Test_config['test_image_path'], label_patch_path=Test_config['test_label_path'], 
                    channel=Channels['staining'], save_test_path=Test_config['test_save_path'], patch_num=patch_num, 
                    test_id=Test_config['test_id'])