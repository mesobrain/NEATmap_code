# Production of training data
from Environment import *
from Parameter import *

def load_tif2array(file):
    
    image = sitk.ReadImage(file)
    image = sitk.Cast(image, sitk.sitkFloat32)
    image_array = sitk.GetArrayFromImage(image)
    image_array = np.transpose(image_array, [0, 2, 1])
    return image_array

def get_train_data(image_path, label_path, save_path):
    for i in range(1, len(os.listdir(image_path)) + 1):
        image_file = os.path.join(image_path, 'image_patch_{}.tif').format(i)
        label_file = os.path.join(label_path, 'label_patch_{}.tif').format(i)
        if os.path.isfile(image_file) and os.path.isfile(label_file):
            image_array = load_tif2array(image_file)
            label_array = load_tif2array(label_file)
            label_array[label_array == 255] = 1
            np.savez(os.path.join(save_path, 'data_patch_{}.npz').format(i), image_q=image_array, image_k=label_array)
            print('finished {}'.format(i))
        else:
            continue

if __name__=="__main__":
    image_path = Train_config['train_image_path']
    label_path = Train_config['train_label_path']
    save_path = os.path.join(Train_config['train_data_path'], 'train_data')
    get_train_data(image_path, label_path, save_path)
