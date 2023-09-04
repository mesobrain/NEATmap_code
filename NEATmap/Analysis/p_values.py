from Environment import *
from scipy import stats
from Parameter import *

### Calculate cell heatmap

def search_indices_sphere(radius):
  """Creates all relative indices within a sphere of specified radius.
  
  Arguments
  ---------
  radius : tuple or int
    Radius of the sphere of the search index list.
  
  Returns
  -------
  indices : array
     Array of ints of relative indices for the search area voxels.
  """
  #create coordiante grid          
  grid = [np.arange(-r,r+1, dtype=float)/np.maximum(1,r) for r in radius];                    
  grid = np.array(np.meshgrid(*grid, indexing = 'ij'));
  
  #sort indices by radius  
  dist = np.sum(grid*grid, axis = 0);
  dist_shape = dist.shape;
  dist = dist.reshape(-1);            
  dist_index = np.argsort(dist);
  dist_sorted = dist[dist_index];
  keep = dist_sorted <= 1;
  dist_index = dist_index[keep];
  
  # convert to relative coordinates
  indices = np.array(np.unravel_index(dist_index, dist_shape)).T;
  indices -= radius;                    
  
  return indices

def cell_heatmap(data_path, save_path, fname):
    """ Calculate the cell density mapped at 25 microns """
    os.makedirs(save_path, exist_ok=True)
    f = open(os.path.join(data_path, fname + '.csv'))
    df = pd.read_csv(f)
    x = df['Position X'] // 25
    y = df['Position Y'] // 25
    source = [x, y]
    coordinates = np.array(source).T
    strides = [1, 1]
    n_points = coordinates.shape[0]
    radius = (6, 6)
    indices = search_indices_sphere(radius)
    n_indices = indices.shape[0]
    n_strides = len(strides)
    shape = [560, 400]
    sink = np.zeros((560, 400))

    for n in range(n_points):
        for i in range(n_indices):
            j = 0
            v = 1
            for d in range(n_strides):
                k = coordinates[n, d] + indices[i, d]
                if not (0 <= k and k < shape[d]):
                    v = 0
                    break
                else:
                    j = j + k * strides[d]
                if d == 0:
                    index_x = j
                    j = 0
                if d == 1:
                    index_y = j
                    j = 0
            if v == 1:
                sink[(index_x, index_y)] += 1
    tif = np.transpose(sink, [1, 0])
    tifffile.imwrite(os.path.join(save_path, fname + '_cell_density.tif'), tif.astype('uint16'))

def group_cell_heatmap(data_path, save_path, id):
    """ Calculate the cell density mapped at 25 microns """
    os.makedirs(save_path, exist_ok=True)
    f = open(os.path.join(data_path, 'whole_brain_cell_counts\Thumbnail_CSV\{:04d}_25.0.tif.csv'.format(id)))
    df = pd.read_csv(f, skiprows=lambda x : x in [0, 1])
    x = df['Position X'] // 25
    y = df['Position Y'] // 25
    source = [x, y]
    coordinates = np.array(source).T
    strides = [1, 1]
    n_points = coordinates.shape[0]
    radius = (6, 6)
    indices = search_indices_sphere(radius)
    n_indices = indices.shape[0]
    n_strides = len(strides)
    shape = [560, 400]
    sink = np.zeros((560, 400))

    for n in range(n_points):
        for i in range(n_indices):
            j = 0
            v = 1
            for d in range(n_strides):
                k = coordinates[n, d] + indices[i, d]
                if not (0 <= k and k < shape[d]):
                    v = 0
                    break
                else:
                    j = j + k * strides[d]
                if d == 0:
                    index_x = j
                    j = 0
                if d == 1:
                    index_y = j
                    j = 0
            if v == 1:
                sink[(index_x, index_y)] += 1
    tif = np.transpose(sink, [1, 0])
    tifffile.imwrite(os.path.join(save_path, str(id) + '_cell_density.tif'), tif.astype('uint16'))

def sagittal_cell_heatmap(data_path, save_path, fname, z_length):
    """ Calculate the sagittal cell density mapped at 25 microns """
    os.makedirs(save_path, exist_ok=True)
    f = open(os.path.join(data_path, fname + '.csv'))
    df = pd.read_csv(f)
    y = df['Position Y'] // 25
    z = df['Position Z'] // 25
    source = [y.values, z.values]
    coordinates = np.array(source).T
    strides = [1, 1]
    n_points = coordinates.shape[0]
    radius = (6, 6)
    indices = search_indices_sphere(radius)
    n_indices = indices.shape[0]
    n_strides = len(strides)
    shape = [400, z_length]
    sink = np.zeros((400, z_length))

    for n in range(n_points):
        for i in range(n_indices):
            j = 0
            v = 1
            for d in range(n_strides):
                k = coordinates[n, d] + indices[i, d]
                if not (0 <= k and k < shape[d]):
                    v = 0
                    break
                else:
                    j = j + k * strides[d]
                if d == 0:
                    index_y = j
                    j = 0
                if d == 1:
                    index_z = j
                    j = 0
            if v == 1:
                sink[(index_y, index_z)] += 1
    tif = np.transpose(sink, [1, 0])
    tifffile.imwrite(os.path.join(save_path, fname + '_cell_density.tif'), tif.astype('uint16'))

def horizontal_cell_heatmap(data_path, save_path, fname, z_length):
    """ Calculate the horizontal cell density mapped at 25 microns """
    os.makedirs(save_path, exist_ok=True)
    f = open(os.path.join(data_path, fname + '.csv'))
    df = pd.read_csv(f)
    x = df['Position X'] // 25
    z = df['Position Z'] // 25
    source = [x.values, z.values]
    coordinates = np.array(source).T
    strides = [1, 1]
    n_points = coordinates.shape[0]
    radius = (6, 6)
    indices = search_indices_sphere(radius)
    n_indices = indices.shape[0]
    n_strides = len(strides)
    shape = [560, z_length]
    sink = np.zeros((560, z_length))

    for n in range(n_points):
        for i in range(n_indices):
            j = 0
            v = 1
            for d in range(n_strides):
                k = coordinates[n, d] + indices[i, d]
                if not (0 <= k and k < shape[d]):
                    v = 0
                    break
                else:
                    j = j + k * strides[d]
                if d == 0:
                    index_x = j
                    j = 0
                if d == 1:
                    index_z = j
                    j = 0
            if v == 1:
                sink[(index_x, index_z)] += 1
    tif = np.transpose(sink, [1, 0])
    tifffile.imwrite(os.path.join(save_path, fname + '_cell_density.tif'), tif.astype('uint16'))

def whole_brain_cell_heatmap(csv_path, save_path, z_length, save_name):

  os.makedirs(save_path, exist_ok=True)
  strides = [1, 1, 1]
  radius = (6, 6, 6)
  indices = search_indices_sphere(radius)
  n_indices = indices.shape[0]
  n_strides = len(strides)
  shape = [560, 400, z_length]
  sink = np.zeros((560, 400, z_length))
  for id in range(len(os.listdir(csv_path))):
    file = open(os.path.join(csv_path, '{:04d}_25.0.tif.csv'.format(id)))
    df = pd.read_csv(file, error_bad_lines=False, skiprows=lambda x: x in [0, 1])
    x = df['Position X'] // 25
    y = df['Position Y'] // 25
    z = df['Position Z'] // 25
    source = [x.values, y.values, z.values]
    coordinates = np.array(source).T
    n_points = coordinates.shape[0]
    for n in range(n_points):
        for i in range(n_indices):
            j = 0
            v = 1
            for d in range(n_strides):
                k = coordinates[n, d] + indices[i, d]
                if not (0 <= k and k < shape[d]):
                    v = 0
                    break
                else:
                    j = j + k * strides[d]
                if d == 0:
                    index_x = j
                    j = 0
                if d == 1:
                    index_y = j
                    j = 0
                if d == 2:
                    index_z = j
                    j = 0
            if v == 1:
                sink[(index_x, index_y, index_z)] += 1
    print('finished {} csv'.format(id))
  tif = np.transpose(sink, [2, 1, 0])
  tifffile.imwrite(os.path.join(save_path, save_name + '_cell_density.tif'), tif.astype('uint16'))

### Calculate the P value

def read_group(sources, combine = True):
  """Turn a list of sources for data into a numpy stack.
  
  Arguments
  ---------
  sources : list of str or sources
     The sources to combine.
  combine : bool
    If true combine the sources to ndarray, oterhwise return a list.
  
  Returns
  -------
  group : array or list
    The gorup data.
  """
  
  #check if stack already:
  if isinstance(sources, np.ndarray):
    return sources;
  
  #read the individual files
  group = [];
  for f in sources:
    data = tifffile.imread(f)
    data = np.reshape(data, (1,) + data.shape);
    group.append(data);
  
  if combine:
    return np.vstack(group);
  else:
    return group;
        
def cutoff_p_values(pvals, p_cutoff = 0.05):
  """cutt of p-values above a threshold.
  
  Arguments
  ---------
  p_valiues : array
    The p values to truncate.
  p_cutoff : float or None
    The p-value cutoff. If None, do not cut off.

  Returns
  -------
  p_values : array
    Cut off p-values.
  """
  pvals2 = pvals.copy();
  pvals2[pvals2 > p_cutoff]  = p_cutoff;
  return pvals2;
    

def color_p_values(pvals, psign, positive = [1,0], negative = [0,1], p_cutoff = None, positive_trend = [0,1], negative_trend = [1,0], pmax = None):
    
    pvalsinv = pvals.copy();
    if pmax is None:
        pmax = pvals.max();    
    pvalsinv = pmax - pvalsinv;    
    
    if p_cutoff is None:  # color given p values
        
        d = len(positive);
        ds = pvals.shape + (d,);
        pvc = np.zeros(ds);
    
        #color
        ids = psign > 0;
        pvalsi = pvalsinv[ids];
        for i in range(d):
            pvc[ids, i] = pvalsi * positive[i];
    
        ids = psign < 0;
        pvalsi = pvalsinv[ids];
        for i in range(d):
            pvc[ids, i] = pvalsi * negative[i];
        
        return pvc;
        
    else:  # split pvalues according to cutoff
    
        d = len(positive_trend);
        
        if d != len(positive) or  d != len(negative) or  d != len(negative_trend) :
            raise RuntimeError('colorPValues: postive, negative, postivetrend and negativetrend option must be equal length!');
        
        ds = pvals.shape + (d,);
        pvc = np.zeros(ds);
    
        idc = pvals < p_cutoff;
        ids = psign > 0;

        ##color 
        # significant postive
        ii = np.logical_and(ids, idc);
        pvalsi = pvalsinv[ii];
        w = positive;
        for i in range(d):
            pvc[ii, i] = pvalsi * w[i];
    
        #non significant postive
        ii = np.logical_and(ids, ~idc);
        pvalsi = pvalsinv[ii];
        w = positive_trend;
        for i in range(d):
            pvc[ii, i] = pvalsi * w[i];
            
         # significant negative
        ii = np.logical_and(~ids, idc);
        pvalsi = pvalsinv[ii];
        w = negative;
        for i in range(d):
            pvc[ii, i] = pvalsi * w[i];
    
        #non significant postive
        ii = np.logical_and(~ids, ~idc)
        pvalsi = pvalsinv[ii];
        w = negative_trend;
        for i in range(d):
            pvc[ii, i] = pvalsi * w[i];
        
        return pvc;

def t_test_voxelization(group1, group2, signed = False, remove_nan = True, p_cutoff = None):
  """t-Test on differences between the individual voxels in group1 and group2
  
  Arguments
  ---------
  group1, group2 : array of arrays
    The group of voxelizations to compare.
  signed : bool
    If True, return also the direction of the changes as +1 or -1.
  remove_nan : bool
    Remove Nan values from the data.
  p_cutoff : None or float
    Optional cutoff for the p-values.
  
  Returns
  -------
  p_values : array
    The p values for the group wise comparison.
  """
  group1 = read_group(group1);  
  group2 = read_group(group2);  
  
  tvals, pvals = stats.ttest_ind(group1, group2, axis=0, equal_var=True);
  
  #remove nans
  if remove_nan: 
    pi = np.isnan(pvals);
    pvals[pi] = 1.0;
    tvals[pi] = 0;

  pvals = cutoff_p_values(pvals, p_cutoff=p_cutoff);

  #return
  if signed:
      return pvals, np.sign(tvals);
  else:
      return pvals;

def generate_p_mapping(group1, group2, save_path, save_name):
    os.makedirs(save_path, exist_ok=True)
    """ Generate a P-value map """
    pvals, psign = t_test_voxelization(group1, group2, signed = True, remove_nan=True, p_cutoff=0.05)
    pvalscol = color_p_values(pvals, psign, positive = [0, 1], negative = [1, 0], p_cutoff=0.05)
    # if len(pvalscol.shape) == 3:
    #   pvalscol = np.transpose(pvalscol, [2, 0, 1])
    # elif len(pvalscol.shape) == 4:
    pvalscol_neg = pvalscol[:, :, 0]
    pvalscol_pos = pvalscol[:, :, 1]
    pvalscol_pos = np.squeeze(pvalscol_pos)
    pvalscol_neg = np.squeeze(pvalscol_neg)
    tifffile.imwrite(os.path.join(save_path, save_name + '_pos.tif'), pvalscol_pos.astype('float32')) 
    tifffile.imwrite(os.path.join(save_path, save_name + '_neg.tif'), pvalscol_neg.astype('float32')) 

if __name__ == "__main__":
    root = Data_root
    fname = {'FS8':'0490_25', 'Ctrl8':'0490_25'}
    z_length = Preprocessing['brain_total_num'] / Spot_map['group_num']

    data_name = [ '', '', ...]
    group_name = list(fname.keys())
    for i in range(len(data_name)):
      data_path = os.path.join(root, data_name[i], 'coordinate')
      save_path = os.path.join(root, data_name[i], 'cell_density')
      cell_heatmap(data_path, save_path, fname=fname[group_name[i]])
      print('finished {}'.format(group_name[i]))
    sagittal_cell_heatmap(data_path, save_path, fname, z_length)
    horizontal_cell_heatmap(data_path, save_path, fname, z_length)

    for i in range(len(data_name)):
      for j in range(35):
        index = j*16
        group_cell_heatmap(data_path=os.path.join(root, data_name[i]), save_path=os.path.join(root, data_name[i], 'group_cell_denisty'), id=index)
        if index >= z_length:
          break
        print('finish {}'.format(index))
      print('finished' + data_name[i])
    group1 = ['', '', ...]
    group2 = ['', '', ...]
    fs_name = Stats['FST_data_list']
    ctrl_name = Stats['Ctrl_data_list']

    for i in range(35):
      index = i*16
      group_fs = [os.path.join(root, fs_name[j], 'group_cell_denisty\{}_cell_density.tif'.format(index)) for j in range(len(fs_name))]
      # group1.append(group_fs)
      group_ctrl = [os.path.join(root, ctrl_name[k], 'group_cell_denisty\{}_cell_density.tif'.format(index)) for k in range(len(ctrl_name))]
      # group2.append(group_ctrl)
      save_p_map_path = os.path.join(Data_root, '', 'group_p_map')
      save_name = str(index) + '_p_map' # Select according to the 300th image of FS6 in 25 microns. For example, the 25-micron FS6 has a total of 648 images. 300/648 ~= 0.463.
                                  #Select other groups at this rate
      generate_p_mapping(group_fs, group_ctrl, save_p_map_path, save_name)
      print('finished {}'.format(index))

    csv_path = os.path.join(Data_root, '', r'whole_brain_cell_counts\Thumbnail_CSV')
    save_path = os.path.join(Data_root, '', 'cell_density')
    save_name = ''
    whole_brain_cell_heatmap(csv_path, save_path, z_length=len(os.listdir(csv_path)), save_name=save_name)