from Environment import *
from Data_preprocessing.utils import intensity_normalization, segmentation_quick_view, dot_2d_slice_by_slice_wrapper
from Data_preprocessing.edge_dection import remove_edge
from Parameter import *
def spot_seg(train_image_path, save_seg_path, channel, scaling_param, wrapper_param):

    data_path = os.path.join(train_image_path, 'brain_image_64_' + channel)
    for i in range(1, len(os.listdir(data_path)) + 1):
        raw_data_path = os.path.join(data_path, 'brain_image_64/Z{:05d}.tif'.format(i))
        image_561nm = sitk.ReadImage(raw_data_path)
        raw_data = sitk.ReadImage(raw_data_path)
        array_561nm = sitk.GetArrayFromImage(image_561nm)
        raw_array = sitk.GetArrayFromImage(raw_data)

        struct_img = intensity_normalization(array_561nm, scaling_param=scaling_param)

        # gaussian_smoothing_sigma = Spot_seg_config['gaussian_smoothing_sigma']
        # structure_img_smooth = image_smoothing_gaussian_slice_by_slice(struct_img, sigma=gaussian_smoothing_sigma)

        bw_561nm = dot_2d_slice_by_slice_wrapper(struct_img, wrapper_param)
        result_561nm = segmentation_quick_view(bw_561nm)
        
        result = remove_edge(raw_array, result_561nm)

        seg = sitk.GetImageFromArray(result)
        os.makedirs(save_seg_path, exist_ok=True)
        sitk.WriteImage(seg, save_seg_path + '/Z{:05d}_seg.tif'.format(i))
        print('Finished {} seg'.format(i))

if __name__ == "__main__":
    start = time.time()
    spot_seg(train_image_path=Train_config['train_image_path'], save_seg_path=Train_config['train_label_path'], channel=Channels['staining'], 
            scaling_param=Spot_seg_config['scaling_param'], wrapper_param=Spot_seg_config['wrapper_param'])
    end = time.time()
    print('Running time: %s Seconds'%(end-start))