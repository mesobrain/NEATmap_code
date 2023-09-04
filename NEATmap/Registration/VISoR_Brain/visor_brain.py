from Environment import *
from Registration.VISoR_Brain.visor_sample import VISoRSample

VERSION = '0.9.6'
class VISoRBrain:
    def __init__(self, path=None):
        self.slices = {}
        self._transforms = {}
        self.transform_source = {}
        self.atlas_transform = None
        self.slice_spheres = {}
        self.sphere_map = None
        self.sphere = [[0, 0, 0], [0, 0, 0]]
        self.version = VERSION

        if path is not None:
            self.load(path)

    def save(self, file_path):
        _info = {}
        _info['sphere'] = self.sphere
        _info['slices_spheres'] = self.slice_spheres
        _info['transforms'] = {}
        _info['slices'] = {}
        _info['version'] = VERSION
        path = os.path.dirname(file_path)
        if not os.path.exists(path):
            os.mkdir(path)

        if self.sphere_map is not None:
            _info['sphere_map'] = os.path.join(path, 'sphere_map.mha')
            sphere_map = sitk.GetImageFromArray(self.sphere_map)
            sitk.WriteImage(sphere_map, _info['sphere_map'])
        else:
            _info['sphere_map'] = None

        t_path = os.path.join(path, 'slices')
        if not os.path.exists(t_path):
            os.mkdir(t_path)
        for k, f in self.slices.items():
            p = os.path.join(t_path, str(k) + '.txt')
            f.save(p)
            _info['slices'][k] = (os.path.relpath(p, path))
        t_path = os.path.join(path, 'transforms')
        if not os.path.exists(t_path):
            os.mkdir(t_path)
        for k, f in self._transforms.items():
            p = os.path.join(t_path, '{0}.mha').format(k)
            if f is not None:
                sitk.WriteImage(sitk.Cast(f.GetDisplacementField(), sitk.sitkVectorFloat32), p)
            _info['transforms'][k] = (os.path.relpath(p, path))
        if self.atlas_transform is not None:
            sitk.WriteTransform(self.atlas_transform, os.path.join(path, 'atlas_transform.txt'))
            _info['atlas_transform'] = 'atlas_transform.txt'

        info = yaml.dump(_info)
        info_file = open(os.path.join(path, 'visor_brain.txt'), 'w')
        info_file.write(info)
        info_file.close()

    def load(self, file_path):
        path = os.path.dirname(file_path)
        with open(file_path, 'r') as info_file:
            _info = yaml.load(info_file)
        self.sphere = _info['sphere']
        self.slice_spheres = _info['slices_spheres']
        self._transforms = {}
        if 'version' in _info:
            self.version = _info['version']
        else:
            self.version = '0.2.0'
        for k, f in _info['slices'].items():
            sl = VISoRSample()
            sl.load(os.path.join(path, f))
            self.slices[k] = sl
        for k, f in _info['transforms'].items():
            self.transform_source[k] = os.path.join(path, f)
            self._transforms[k] = None
        if _info['sphere_map'] is not None:
            self.sphere_map = sitk.ReadImage(_info['sphere_map'])
            self.sphere_map = sitk.GetArrayFromImage(self.sphere_map)
        try:
            if _info['atlas_transform'] is not None:
                self.atlas_transform = sitk.ReadTransform(os.path.join(path, _info['atlas_transform']))
        except KeyError:
            pass