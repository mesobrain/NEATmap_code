from skimage import measure
from math import ceil,floor
from Parameter import *
from Environment import *

def BrainImage_cat(BrainImage_root,Save_root,flip=True, slices=75):
    x = 2500
    y = 3500
    if not os.path.exists(Save_root):
        os.mkdir(Save_root)
    brainimage_list =  os.listdir(BrainImage_root)
    brainimage_list.sort()
    for i in range(floor(len(brainimage_list)/slices)):
        slice_list = brainimage_list[slices*i:slices*(i+1)]
        slice_image = np.zeros((slices,x,y)).astype('uint16')
        for j in range(len(slice_list)):
            print(j, os.path.join(BrainImage_root,slice_list[j]))
            im = tifffile.imread(os.path.join(BrainImage_root,slice_list[j]))
            if flip:
                im = np.fliplr(im)
            slice_image[j,: ,:] = im
        Save_path = os.path.join(Save_root,str(i).zfill(5)+'.tif')
        tifffile.imwrite(Save_path,slice_image, compress=2)

def BrainImage2Spot(BrainImage_root,csv_root1):
    if not os.path.exists(csv_root1):
        os.mkdir(csv_root1)
    brainimage_list = os.listdir(BrainImage_root)
    save_file = open(os.path.join(csv_root1,'total.txt'),'w+')
    save_file.writelines('Id X Y Z Area\n')
    for i in range(0,len(brainimage_list)):
        brainimage_path = os.path.join(BrainImage_root,brainimage_list[i])
        print(brainimage_path)
        print('image reading...')
        binary_image = tifffile.imread(brainimage_path)
        labeled_img = measure.label(binary_image, connectivity=1)
        properties = measure.regionprops(labeled_img)
        centroid_list = []
        area_list = []
        print('cell counting...')
        for pro in properties:
            centroid = pro.centroid
            centroid_list.append(centroid)
            area = pro.area
            area_list.append(area)
        centroid_list.sort()
        for j in range(len(centroid_list)):
            z = ceil(centroid_list[j][0])
            y = ceil(centroid_list[j][1])
            x = ceil(centroid_list[j][2])
            area = area_list[j]
            if area == Spot_map['filter_area_lower'] or area >= Spot_map['filter_area_upper']:
                pass
            else:
                z_index = z + i*Brain['z_num'] 
                print(x, y, z_index, area, '---', j)
                content = str(j) + ' ' + str(x) + ' '+ str(y) +' ' + str(z_index) +' ' + str(area) + '\n'
                save_file.writelines(content)
    save_file.close()

# group_num is the number of brainimage2d_num divided by the number of thumbnails (thumbnails are obtained after alignment)
def Spot_csv(total_path, csv_root, brainimage2d_num, group_num):
    if not os.path.exists(csv_root):
        os.mkdir(csv_root)
    f = open(total_path,'r')
    spots = []
    for spot in f.readlines()[1:]:
        a = spot.strip().split(' ')
        a = np.array(a).astype(dtype=int).tolist()
        spots.append(a)
    # print(spots)

    # group with z = spots[3] , for every 6.25 slices
    for i in range(floor(brainimage2d_num/group_num)):
        print('----Thumbnail'+str(i))
        csv_name = str(i).zfill(4)+'_25.0.tif.csv'
        csv_path = os.path.join(csv_root,csv_name)
        count = 0
        list_name = ['Position X','Position Y','Position Z', 'Unit', 'Category', 'Collection','Time','ID']
        with open(csv_path,'w+',newline='')as f:
            csv_write = csv.writer(f, dialect='excel')
            csv_write.writerow(['25.0'])
            csv_write.writerow(['=========='])
            csv_write.writerow(list_name)
            for j in range(len(spots)):
                if i * group_num<=spots[j][3]<=i * group_num + (group_num-1):
                    print(spots[j])
                    x ,y ,z = str(spots[j][1]*Brain['voxel_size']), str(spots[j][2]*Brain['voxel_size']), str(spots[j][3]*Brain['voxel_size'])
                    writeline = [x,y,z,'um','Spot','Position','1',str(count)]
                    count += 1
                    csv_write.writerow(writeline)
                    print(count)

def Sagittal_spot_csv(total_path, csv_root, brainimage_x, group_num):
    if not os.path.exists(csv_root):
        os.mkdir(csv_root)
    f = open(total_path,'r')
    spots = []
    for spot in f.readlines()[1:]:
        a = spot.strip().split(' ')
        a = np.array(a).astype(dtype=int).tolist()
        spots.append(a)
    # print(spots)

    for i in range(floor(brainimage_x/group_num)):
        print('----Thumbnail'+str(i))
        csv_name = str(i).zfill(4)+'_25.csv'
        csv_path = os.path.join(csv_root,csv_name)
        count = 0
        list_name = ['Position X','Position Y','Position Z', 'Unit', 'Category', 'Collection','Time','ID']
        with open(csv_path,'w+',newline='')as f:
            csv_write = csv.writer(f, dialect='excel')
            csv_write.writerow(list_name)
            for j in range(len(spots)):
                if i * group_num<=spots[j][1]<=i * group_num + (group_num-1):
                    print(spots[j])
                    x ,y ,z = str(spots[j][1]*Brain['voxel_size']), str(spots[j][2]*Brain['voxel_size']), str(spots[j][3]*Brain['voxel_size'])
                    writeline = [x,y,z,'um','Spot','Position','1',str(count)]
                    count += 1
                    csv_write.writerow(writeline)
                    print(count)

def Horizontal_spot_csv(total_path, csv_root, brainimage_y, group_num):
    if not os.path.exists(csv_root):
        os.mkdir(csv_root)
    f = open(total_path,'r')
    spots = []
    for spot in f.readlines()[1:]:
        a = spot.strip().split(' ')
        a = np.array(a).astype(dtype=int).tolist()
        spots.append(a)
    # print(spots)

    for i in range(floor(brainimage_y/group_num)):
        print('----Thumbnail'+str(i))
        csv_name = str(i).zfill(4)+'_25.csv'
        csv_path = os.path.join(csv_root,csv_name)
        count = 0
        list_name = ['Position X','Position Y','Position Z', 'Unit', 'Category', 'Collection','Time','ID']
        with open(csv_path,'w+',newline='')as f:
            csv_write = csv.writer(f, dialect='excel')
            csv_write.writerow(list_name)
            for j in range(len(spots)):
                if i * group_num<=spots[j][2]<=i * group_num + (group_num-1):
                    print(spots[j])
                    x ,y ,z = str(spots[j][1]*Brain['voxel_size']), str(spots[j][2]*Brain['voxel_size']), str(spots[j][3]*Brain['voxel_size'])
                    writeline = [x,y,z,'um','Spot','Position','1',str(count)]
                    count += 1
                    csv_write.writerow(writeline)
                    print(count)


if __name__ == '__main__':
    path = Splicing['save_splicing_path']
    BrainImage_root = os.path.join(path, 'whole_brain_pred_2d')
    Save_root = os.path.join(path, 'whole_brain_pred_3d')
    BrainImage_cat(BrainImage_root,Save_root, flip=False)
    BrainImage_root = os.path.join(path, 'whole_brain_pred_3d')
    csv_root1 = os.path.join(path, 'whole_brain_cell_counts')
    BrainImage2Spot(BrainImage_root,csv_root1)
    csv_root = os.path.join(csv_root1, 'Thumbnail_CSV')
    total_path = os.path.join(csv_root1, 'total.txt')
    Spot_csv(total_path, csv_root, brainimage2d_num=Preprocessing['brain_total_num'], group_num=Spot_map['group_num'])
    Sagittal_spot_csv(total_path, csv_root, brainimage_x=Brain['weight'], group_num=Spot_map['group_num'])
    Horizontal_spot_csv(total_path, csv_root, brainimage_y=Brain['height'], group_num=Spot_map['group_num'])
    
