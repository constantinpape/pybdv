import os
import unittest
from abc import ABC
from shutil import rmtree

import numpy as np
from pybdv import make_bdv

try:
    from elf.io import open_file
    WITH_ELF = True
except ImportError:
    import h5py
    open_file = h5py.File
    WITH_ELF = False


class MetadataTestMixin(ABC):
    tmp_folder = './tmp'
    xml_path = './tmp/test.xml'
    shape = (64,) * 3
    chunks = (32,) * 3

    resolution1 = [4.4, 3.8, 4.]
    resolution2 = [1., 3.1415926, 42.]

    def setUp(self):
        os.makedirs(self.tmp_folder, exist_ok=True)
        data = np.random.rand(*self.shape)
        make_bdv(data, self.out_path, resolution=self.resolution1, chunks=self.chunks)
        make_bdv(data, self.out_path, resolution=self.resolution2, chunks=self.chunks,
                 setup_id=1)

    def tearDown(self):
        try:
            rmtree(self.tmp_folder)
        except OSError:
            pass

    def test_get_bdv_format(self):
        from pybdv.metadata import get_bdv_format
        bdv_format = get_bdv_format(self.xml_path)
        self.assertEqual(bdv_format, self.bdv_format)

    def test_get_resolution(self):
        from pybdv.metadata import get_resolution

        resolution = get_resolution(self.xml_path, 0)
        self.assertEqual(resolution, self.resolution1)

        resolution = get_resolution(self.xml_path, 1)
        self.assertEqual(resolution, self.resolution2)

    def test_get_data_path(self):
        from pybdv.metadata import get_data_path
        path = get_data_path(self.xml_path)
        exp_path = os.path.split(self.out_path)[1]
        self.assertEqual(path, exp_path)
        abs_path = get_data_path(self.xml_path, return_absolute_path=True)
        self.assertEqual(abs_path, os.path.abspath(self.out_path))


class TestMetadataH5(MetadataTestMixin, unittest.TestCase):
    in_path = './tmp/in.h5'
    out_path = './tmp/test.h5'
    bdv_format = 'bdv.hdf5'


@unittest.skipUnless(WITH_ELF, "Need elf for n5 support")
class TestMetadataN5(MetadataTestMixin, unittest.TestCase):
    in_path = './tmp/in.n5'
    out_path = './tmp/test.n5'
    bdv_format = 'bdv.n5'


if __name__ == '__main__':
    unittest.main()
