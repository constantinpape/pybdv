import os
import unittest
from shutil import rmtree

import numpy as np
import h5py


# TODO expand tests
class TestMakeBdv(unittest.TestCase):
    def setUp(self):
        os.makedirs('./tmp', exist_ok=True)

    def tearDown(self):
        try:
            rmtree('./tmp')
        except OSError:
            pass

    def test_simple(self):
        from pybdv import make_bdv
        shape = (100, 100, 100)
        data = np.random.rand(*shape).astype('float32')

        out_path = './tmp/test.h5'
        make_bdv(data, out_path)

        key = 't00000/s00/0/cells'
        self.assertTrue(os.path.exists(out_path))
        with h5py.File(out_path, 'r') as f:
            self.assertTrue(key in f)
            ds = f[key]
            self.assertEqual(ds.shape, shape)
            out_data = ds[:]
        self.assertTrue(np.allclose(data, out_data))

    # TODO test views with different registrations
    def test_multi_setup(self):
        from pybdv import make_bdv
        shape = (128, 128, 128)
        out_path = './tmp/test.h5'

        n_views = 2
        out_path = './tmp/test.h5'

        data_dict = {}

        for vid in range(n_views):
            data = np.random.rand(*shape).astype('float32')
            make_bdv(data, out_path, setup_id=vid)
            data_dict[vid] = data

        with h5py.File(out_path, 'r') as f:
            for vid in range(n_views):
                expected_key = 't00000/s%02i/0/cells' % vid
                self.assertTrue(expected_key in f)

                exp_data = data_dict[vid]
                data = f[expected_key][:]
                self.assertTrue(np.allclose(data, exp_data))

        # TODO check the xml metadata

    def _test_ds(self, mode):
        from pybdv import make_bdv
        shape = (256, 256, 256)
        data = np.random.rand(*shape).astype('float32')

        out_path = './tmp/test.h5'
        n_scales = 4
        downscale_factors = n_scales * [[2, 2, 2]]
        make_bdv(data, out_path, downscale_factors,
                 downscale_mode=mode)

        exp_shape = shape
        self.assertTrue(os.path.exists(out_path))
        with h5py.File(out_path, 'r') as f:
            for scale in range(n_scales):
                key = 't00000/s00/%i/cells' % scale
                self.assertTrue(key in f)
                ds = f[key]
                self.assertEqual(ds.shape, exp_shape)
                exp_shape = tuple(sh // sf for sh, sf in zip(exp_shape, downscale_factors[scale]))

    def test_ds_nearest(self):
        self._test_ds('nearest')

    def test_ds_mean(self):
        self._test_ds('mean')


if __name__ == '__main__':
    unittest.main()
