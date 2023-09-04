from Environment import *
from contextlib import contextmanager
from distutils.version import LooseVersion

VERSION = '0.9.6'
class VISoRSample:
    def __init__(self, filename=None):
        self.transforms = {}
        self.inverse_transforms = {}
        self.column_spheres = {}
        self.sphere = [[0, 0, 0], [0, 0, 0]]
        self.column_images = {}
        self.image = None
        self.column_source = None
        self.image_source = None
        self.device_file = None
        self.image_origin = [0, 0, 0]
        self.image_spacing = [1, 1, 1]
        self.version = VERSION

        # Private attributes for optimization
        self._column_spheres = None

        if filename is not None:
            self.load(filename)

    @contextmanager
    def load(self, filename):
        if filename.split('.')[-1] == 'tar':
            with tempfile.TemporaryDirectory() as tmpdir:
                self.version = '0.1'
                tar = tarfile.open(filename, 'r')
                tar.extractall(tmpdir)
                info_file = open(os.path.join(tmpdir, 'info.txt'), 'r')
                _info = yaml.load(info_file)
                info_file.close()
                self.sphere = _info['sphere']
                self.column_spheres = _info['stack_spheres']
                self._update_spheres()
                self.column_source = _info['stack_source']
                if _info['image_source'] is not None:
                    self.image_source = os.path.join(os.path.dirname(filename), _info['image_source'])
                    self.image_origin = _info['image_origin']
                    self.image_spacing = _info['image_spacing']
                self.device_file = _info['device_file']
                for i in range(len(_info['transforms'])):
                    tf_path = os.path.join(tmpdir, _info['transforms'][i])
                    tf = sitk.ReadTransform(tf_path)
                    self.transforms[i] = tf
                self.calculate_transforms()
                return
        with open(filename, 'r') as info_file:
            _info = json.load(info_file)
            self.sphere = _info['sphere']
            if 'version' in _info:
                self.version = _info['version']
            else:
                self.version = '0.2.0'
            self.column_spheres = {int(i): v for i, v in _info['column_spheres'].items()}
            self._update_spheres()
            self.column_source = _info['stack_source']
            if _info['image_source'] is not None:
                self.image_source = os.path.join(os.path.dirname(filename), _info['image_source'])
                self.image_origin = _info['image_origin']
                self.image_spacing = _info['image_spacing']
            if LooseVersion(self.version) < LooseVersion('0.4.0'):
                self.device_file = _info['device_file']
            for i in range(len(_info['transforms'])):
                tf = sitk.AffineTransform(3)
                tf.SetParameters(_info['transforms'][str(i)])
                self.transforms[i] = tf
            self.calculate_transforms()

    @contextmanager
    def save(self, filename):
        _info = {'sphere': self.sphere,
                 'transforms': {},
                 'image_source': None,
                 'stack_source': self.column_source}

        if self.image_source is not None:
            _info['image_source'] = os.path.relpath(self.image_source, os.path.dirname(filename))
            _info['image_origin'] = self.image_origin
            _info['image_spacing'] = self.image_spacing

        _info['version'] = VERSION
        # old way
        if filename.split('.')[-1] == 'tar':
            _info['stack_spheres'] = self.column_spheres
            with tempfile.TemporaryDirectory() as tmpdir:
                t_path = os.path.join(tmpdir, 'transforms')
                os.mkdir(t_path)
                for i in self.transforms.keys():
                    tf_path = os.path.join(t_path, str(i) + '.txt')
                    sitk.WriteTransform(self.transforms[i], tf_path)
                    _info['transforms'][i] = os.path.relpath(tf_path, tmpdir)

                info = yaml.dump(_info)
                info_file = open(os.path.join(tmpdir, 'info.txt'), 'w')
                info_file.write(info)
                info_file.close()
                tar = tarfile.open(filename, 'w')
                for root,dir_,files in os.walk(tmpdir):
                    for file in files:
                        fullpath = os.path.join(root, file)
                        tar.add(fullpath, os.path.relpath(fullpath, tmpdir))
                tar.close()
            return
        # new way
        _info['column_spheres'] = self.column_spheres
        for i in self.transforms:
            _info['transforms'][i] = self.transforms[i].GetParameters()
        with open(filename, 'w') as f:
            json.dump(_info, f, indent=2)
            
    def calculate_transforms(self):
        self.inverse_transforms = {i: self.transforms[i].GetInverse() for i in self.transforms}        
        
    def _update_spheres(self):
        self._column_spheres = [self.column_spheres[i][j][k]
                                for i in range(len(self.column_spheres))
                                for j in range(2)
                                for k in range(3)]