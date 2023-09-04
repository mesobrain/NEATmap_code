from Environment import *
from scipy import ndimage
from torch.utils.data import Dataset
from scipy import ndimage as ndi
from scipy import stats

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    for zz in range(image.shape[0]):
        image[zz,:,:] = np.rot90(image[zz,:,:], k)
        label[zz,:,:] = np.rot90(label[zz,:,:], k)
        axis = np.random.randint(0, 2)
        image[zz,:,:] = np.flip(image[zz,:,:], axis=axis).copy()
        label[zz,:,:] = np.flip(label[zz,:,:], axis=axis).copy()
    return image, label

def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    for zz in range(image.shape[0]):
        image[zz,:,:] = ndimage.rotate(image[zz,:,:], angle, order=0, reshape=False)
        label[zz,:,:] = ndimage.rotate(label[zz,:,:], angle, order=0, reshape=False)
    return image, label

def norm_data(data):
    data1 = data - np.min(data) 
    data = data1 * 1.0 / (np.max(data) - np.min(data) )
    return data

def simple_norm(img, a, b, m_high=-1, m_low=-1):
    idx = np.ones(img.shape, dtype=bool)
    if m_high>0:
        idx = np.logical_and(idx, img<m_high)
    if m_low>0:
        idx = np.logical_and(idx, img>m_low)
    img_valid = img[idx]
    m,s = stats.norm.fit(img_valid.flat)
    strech_min = max(m - a*s, img.min())
    strech_max = min(m + b*s, img.max())
    img[img>strech_max]=strech_max
    img[img<strech_min]=strech_min
    img = (img- strech_min)/(strech_max - strech_min)
    return img

def background_sub(img, r):
    struct_img_smooth = ndi.gaussian_filter(img, sigma=r, mode='nearest', truncate=3.0)
    struct_img_smooth_sub = img - struct_img_smooth
    struct_img = (struct_img_smooth_sub - struct_img_smooth_sub.min())/(struct_img_smooth_sub.max()-struct_img_smooth_sub.min())
    return struct_img

def gamma(image, c, v):
    new_image = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if 300 <= image[i,j] <= 680:
                value = image[i,j] / 255
                new_image[i,j] = c * np.power(value, v) * 255
            else:
                new_image[i,j] = c * image[i,j]
    output_image = np.uint16(new_image + 0.5)

    return output_image

class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # image = background_sub(image, 2)
        # image = norm_data(image) 
        # contrast = random.random()
        # new_image = np.zeros_like(image)
        # if contrast < 0.5:
        #     for zz in range(image.shape[0]):
        #         new_image[zz,:,:] = gamma(image[zz,:,:], 1, 2) 
        image = np.expand_dims(image, axis=0)
        label = np.expand_dims(label, axis=0)
        image = torch.from_numpy(image.astype(np.float32))
        label = torch.from_numpy(label.astype(np.float32)).squeeze(0)
        sample = {'image': image, 'label': label.long()}

        return sample

class VISoR_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train" or self.split == "valid":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
            
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
