import os
import unittest
import numpy as np
from shutil import rmtree
from pybdv import make_bdv
try:
    import z5py
except ImportError:
    z5py = None


class TestUtil(unittest.TestCase):
    path_h5 = 'tmp_vol1.h5'
    path_h5_xml = 'tmp_vol1.xml'
    path_n5 = 'tmp_vol2.n5'
    path_n5_xml = 'tmp_vol2.xml'
    shape = (128,) * 3
    scale_factors = [[2, 2, 2]] * 3
    abs_scale_factors = [[1., 1., 1.],
                         [2., 2., 2.],
                         [4., 4., 4.],
                         [8., 8., 8.],]

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
                 downscale_factors=self.scale_factors)

    def _make_n5(self):
        data = np.random.rand(*self.shape)
        make_bdv(data, self.path_n5,
                 downscale_factors=self.scale_factors,
                 chunks=(64,) * 3)

    def test_get_number_of_scales(self):
        from pybdv.util import get_number_of_scales
        self._make_h5()
        n_scales = get_number_of_scales(self.path_h5, 0, 0)
        self.assertEqual(n_scales, 4)
        if z5py is not None:
            self._make_n5()
            n_scales = get_number_of_scales(self.path_n5, 0, 0)
            self.assertEqual(n_scales, 4)

    def test_get_scale_factors(self):
        from pybdv.util import get_scale_factors
        self._make_h5()
        scale_factors = get_scale_factors(self.path_h5, 0)
        self.assertEqual(scale_factors, self.abs_scale_factors)
        if z5py is not None:
            self._make_n5()
            scale_factors = get_scale_factors(self.path_n5, 0)
            self.assertEqual(scale_factors, self.abs_scale_factors)


if __name__ == '__main__':
    unittest.main()
