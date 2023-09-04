from Environment import *
from Parameter import *
# VISoR reconstruction of the generated BrainImage path with 561 channels and 488 channels to 
# synthesize 3D images with (z, x, y) size (64,3500,2500)
def brain2dto3d(data_path, save_path, total_num, z_num, channel_index):
    
    temp = []
    j = 0
    name_index = 1
    os.makedirs(save_path, exist_ok=True)
    for i in range(0, total_num):
        image = sitk.ReadImage(os.path.join(data_path, 'Z{:05d}_'.format(i) + channel_index +'.tif'))
        array = sitk.GetArrayFromImage(image)
        temp.append(array)
        if i == z_num - 1 + j:
            tif = np.array(temp)
            tif = sitk.GetImageFromArray(tif)
            sitk.WriteImage(tif, save_path + '/Z{:05d}.tif'.format(name_index))
            j += z_num
            print('finished {} image'.format(name_index))
            name_index += 1
            temp = []

if __name__ == "__main__":
    path = 'G:\Analysis\20210409_CYX_CFOS_NON-SOCIAL_2_1'
    save_path = os.path.join(Save_root, 'brain_image_64_' + Channels['staining'])#561nm
    start = time.time()
    brain2dto3d(data_path=Data_root, save_path=save_path, z_num=Preprocessing['z_num'])
    end = time.time()
    print('Running time: %s Seconds'%(end-start))