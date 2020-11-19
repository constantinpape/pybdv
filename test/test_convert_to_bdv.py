import os
import unittest
from abc import ABC
from subprocess import run
from shutil import rmtree

import numpy as np
from pybdv.util import get_key, open_file

try:
    import z5py
except ImportError:
    z5py = None


class ConvertToBdvTestMixin(ABC):
    tmp_folder = './tmp'
    xml_path = './tmp/test.xml'

    def _make_input(self, name, data, chunks):
        with open_file(self.in_path, 'a') as f:
            f.create_dataset(name, data=data, chunks=chunks)

    def setUp(self):
        os.makedirs(self.tmp_folder, exist_ok=True)
        shape = (64,) * 3
        self._make_input('data', np.random.rand(*shape), (32,) * 3)

    def tearDown(self):
        try:
            rmtree(self.tmp_folder)
        except OSError:
            pass

    def check_result(self):
        with open_file(self.in_path, 'r') as f:
            exp = f['data'][:]

        key = get_key(self.is_h5, timepoint=0, setup_id=0, scale=0)
        with open_file(self.out_path, 'r') as f:
            self.assertTrue(key in f)
            res = f[key][:]

        self.assertEqual(res.shape, exp.shape)
        self.assertTrue(np.allclose(res, exp))

    def test_simple(self):
        from pybdv import convert_to_bdv
        convert_to_bdv(self.in_path, 'data', self.out_path)
        self.check_result()

    def test_cli_simple(self):
        run(['convert_to_bdv', self.in_path, 'data', self.out_path])
        self.check_result()

    def _test_overwrite(self, mode):
        from pybdv import convert_to_bdv
        from pybdv.util import get_scale_factors, absolute_to_relative_scale_factors
        from pybdv.metadata import get_attributes, get_affine

        def _check(name, exp_sf, exp_attrs, exp_affine):
            with open_file(self.in_path, 'r') as f:
                exp_data = f[name][:]

            key = get_key(self.is_h5, timepoint=0, setup_id=0, scale=0)
            with open_file(self.out_path, 'r') as f:
                data = f[key][:]
            self.assertTrue(np.allclose(data, exp_data))

            sf = get_scale_factors(self.out_path, setup_id=0)
            sf = absolute_to_relative_scale_factors(sf)
            self.assertEqual(sf, [[1, 1, 1]] + exp_sf)

            attrs = get_attributes(self.xml_path, setup_id=0)
            self.assertEqual(attrs, exp_attrs)

            affine = get_affine(self.xml_path, setup_id=0, timepoint=0)['affine0']
            self.assertTrue(np.allclose(np.array(affine), np.array(exp_affine), atol=1e-4))

        sf1 = [[2, 2, 2]]
        attrs1 = {'channel': {'id': 1}, 'angle': {'id': 2}}
        affine1 = np.random.rand(12).tolist()

        self._make_input('data2', np.random.rand(72, 72, 72), (32, 32, 32))
        sf2 = [[1, 2, 2], [2, 2, 2]]
        attrs2 = {'channel': {'id': 3}, 'angle': {'id': 6}}
        affine2 = np.random.rand(12).tolist()

        convert_to_bdv(self.in_path, 'data', self.out_path, setup_id=0, timepoint=0,
                       downscale_factors=sf1, attributes=attrs1, affine=affine1)
        _check('data', sf1, attrs1, affine1)

        convert_to_bdv(self.in_path, 'data2', self.out_path, setup_id=0, timepoint=0,
                       downscale_factors=sf2, attributes=attrs2, affine=affine2,
                       overwrite=mode)

        if mode == 'skip':
            _check('data', sf1, attrs1, affine1)
        elif mode == 'all':
            _check('data2', sf2, attrs2, affine2)
        elif mode == 'data':
            _check('data2', sf2, attrs1, affine1)
        elif mode == 'metadata':
            _check('data', sf1, attrs2, affine2)
        else:
            raise ValueError("Invalid over-write mode")

    def test_overwrite_skip(self):
        self._test_overwrite('skip')

    def test_overwrite_all(self):
        self._test_overwrite('all')

    def test_overwrite_metadata(self):
        self._test_overwrite('metadata')

    def test_overwrite_data(self):
        self._test_overwrite('data')


class TestConvertToBdvH5(ConvertToBdvTestMixin, unittest.TestCase):
    in_path = './tmp/in.h5'
    out_path = './tmp/test.h5'
    is_h5 = True


@unittest.skipUnless(z5py is not None, "Need z5py for n5 support")
class TestConvertToBdvN5(ConvertToBdvTestMixin, unittest.TestCase):
    in_path = './tmp/in.n5'
    out_path = './tmp/test.n5'
    is_h5 = False


if __name__ == '__main__':
    unittest.main()
