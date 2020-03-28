import os
import unittest
from abc import ABC
from shutil import rmtree

import numpy as np
from pybdv import make_bdv
from pybdv.util import HAVE_ELF


class UtilTestMixin(ABC):
    tmp_folder = './tmp'
    xml_path = './tmp/test.xml'
    shape = (128,) * 3
    scale_factors = [[2, 2, 2]] * 3
    abs_scale_factors = [[1., 1., 1.],
                         [2., 2., 2.],
                         [4., 4., 4.],
                         [8., 8., 8.]]

    def tearDown(self):
        try:
            rmtree(self.tmp_folder)
        except OSError:
            pass

    def setUp(self):
        os.makedirs(self.tmp_folder, exist_ok=True)
        data = np.random.rand(*self.shape)
        make_bdv(data, self.out_path, downscale_factors=self.scale_factors)

    def test_get_number_of_scales(self):
        from pybdv.util import get_number_of_scales
        n_scales = get_number_of_scales(self.out_path, 0, 0)
        self.assertEqual(n_scales, 4)

    def test_get_scale_factors(self):
        from pybdv.util import get_scale_factors
        scale_factors = get_scale_factors(self.out_path, 0)
        self.assertEqual(scale_factors, self.abs_scale_factors)

    def test_abs_to_rel_sf(self):
        from pybdv.util import absolute_to_relative_scale_factors

        abs_sf = [[1, 1, 1], [2, 2, 2], [4, 4, 4], [8, 8, 8]]
        rel_sf = absolute_to_relative_scale_factors(abs_sf)
        exp_rel_sf = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
        self.assertEqual(rel_sf, exp_rel_sf)

        abs_sf = [[1, 1, 1], [1, 2, 2], [1, 4, 4], [2, 8, 8]]
        rel_sf = absolute_to_relative_scale_factors(abs_sf)
        exp_rel_sf = [[1, 1, 1], [1, 2, 2], [1, 2, 2], [2, 2, 2]]
        self.assertEqual(rel_sf, exp_rel_sf)

        abs_sf = [[1, 1, 1], [3, 3, 3], [6, 6, 6], [24, 24, 24]]
        rel_sf = absolute_to_relative_scale_factors(abs_sf)
        exp_rel_sf = [[1, 1, 1], [3, 3, 3], [2, 2, 2], [4, 4, 4]]
        self.assertEqual(rel_sf, exp_rel_sf)

    def test_rel_to_abs_sf(self):
        from pybdv.util import relative_to_absolute_scale_factors

        rel_sf = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
        abs_sf = relative_to_absolute_scale_factors(rel_sf)
        exp_abs_sf = [[1, 1, 1], [2, 2, 2], [4, 4, 4], [8, 8, 8]]
        self.assertEqual(abs_sf, exp_abs_sf)

        rel_sf = [[1, 1, 1], [1, 2, 2], [1, 2, 2], [2, 2, 2]]
        abs_sf = relative_to_absolute_scale_factors(rel_sf)
        exp_abs_sf = [[1, 1, 1], [1, 2, 2], [1, 4, 4], [2, 8, 8]]
        self.assertEqual(abs_sf, exp_abs_sf)

        rel_sf = [[1, 1, 1], [3, 3, 3], [2, 2, 2], [4, 4, 4]]
        abs_sf = relative_to_absolute_scale_factors(rel_sf)
        exp_abs_sf = [[1, 1, 1], [3, 3, 3], [6, 6, 6], [24, 24, 24]]
        self.assertEqual(abs_sf, exp_abs_sf)


class TestUtilH5(UtilTestMixin, unittest.TestCase):
    out_path = './tmp/test.h5'


@unittest.skipUnless(HAVE_ELF, "Need elf for n5 support")
class TestConvertToBdvN5(UtilTestMixin, unittest.TestCase):
    out_path = './tmp/test.n5'


if __name__ == '__main__':
    unittest.main()
