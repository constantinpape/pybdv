import os
import unittest
import numpy as np
from shutil import rmtree
from pybdv import make_bdv
try:
    import z5py
except ImportError:
    z5py = None


class TestMetadata(unittest.TestCase):
    path_h5 = 'tmp_vol1.h5'
    path_h5_xml = 'tmp_vol1.xml'
    path_n5 = 'tmp_vol2.n5'
    path_n5_xml = 'tmp_vol2.xml'
    shape = (128,) * 3
    resolution = [4.4, 3.8, 4.]
    scale_factors = [[2, 2, 2]] * 3

    def tearDown(self):
        if os.path.exists(self.path_h5):
            os.remove(self.path_h5)
            os.remove(self.path_h5_xml)
        if os.path.exists(self.path_n5):
            rmtree(self.path_n5)
            os.remove(self.path_n5_xml)

    def _make_h5(self):
        data = np.random.rand(*self.shape)
        make_bdv(data, self.path_h5,
                 resolution=self.resolution)

    def _make_n5(self):
        data = np.random.rand(*self.shape)
        make_bdv(data, self.path_n5,
                 resolution=self.resolution,
                 chunks=(64,) * 3)

    def test_get_bdv_format(self):
        from pybdv.metadata import get_bdv_format
        self._make_h5()
        bdv_format = get_bdv_format(self.path_h5_xml)
        self.assertEqual(bdv_format, 'bdv.hdf5')
        if z5py is not None:
            self._make_n5()
            bdv_format = get_bdv_format(self.path_n5_xml)
            self.assertEqual(bdv_format, 'bdv.n5')

    def test_get_resolution(self):
        from pybdv.metadata import get_resolution
        self._make_h5()
        resolution = get_resolution(self.path_h5_xml, 0)
        self.assertEqual(resolution, self.resolution)
        if z5py is not None:
            self._make_n5()
            resolution = get_resolution(self.path_n5_xml, 0)
            self.assertEqual(resolution, self.resolution)

    def test_get_data_path(self):
        from pybdv.metadata import get_data_path
        self._make_h5()
        path = get_data_path(self.path_h5_xml)
        self.assertEqual(path, self.path_h5)
        abs_path = get_data_path(self.path_h5_xml, return_absolute_path=True)
        self.assertEqual(abs_path, os.path.abspath(self.path_h5))

        if z5py is not None:
            self._make_n5()
            path = get_data_path(self.path_n5_xml)
            self.assertEqual(path, self.path_n5)
            abs_path = get_data_path(self.path_n5_xml, return_absolute_path=True)
            self.assertEqual(abs_path, os.path.abspath(self.path_n5))


if __name__ == '__main__':
    unittest.main()
