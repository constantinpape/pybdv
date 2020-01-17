import os
import unittest
from shutil import rmtree

import numpy as np
import h5py

try:
    from elf.io import open_file
except ImportError:
    open_file = None


class TestMakeBdv(unittest.TestCase):
    def setUp(self):
        os.makedirs('./tmp', exist_ok=True)

    def tearDown(self):
        try:
            rmtree('./tmp')
        except OSError:
            pass

    def _test_simple(self, shape):
        from pybdv import make_bdv
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

    def test_simple_3d(self):
        shape = (100, 100, 100)
        self._test_simple(shape)

    # 2d is not supported yet
    @unittest.skip
    def test_simple_2d(self):
        shape = (256, 256)
        self._test_simple(shape)

    # TODO test views with different registrations
    def test_multi_setup(self):
        from pybdv import make_bdv
        shape = (64,) * 3
        out_path = './tmp/test.h5'

        n_views = 2
        out_path = './tmp/test.h5'

        data_dict = {}

        for vid in range(n_views):
            data = np.random.rand(*shape).astype('float32')
            make_bdv(data, out_path, setup_id=vid)
            data_dict[vid] = data

        # check implicit setup id
        data = np.random.rand(*shape).astype('float32')
        make_bdv(data, out_path)
        data_dict[n_views] = data

        with h5py.File(out_path, 'r') as f:
            for vid in range(n_views + 1):
                expected_key = 't00000/s%02i/0/cells' % vid
                self.assertTrue(expected_key in f)

                exp_data = data_dict[vid]
                data = f[expected_key][:]
                self.assertTrue(np.allclose(data, exp_data))

        # TODO check the xml metadata

    def _test_ds(self, shape, mode):
        from pybdv import make_bdv
        data = np.random.rand(*shape).astype('float32')

        out_path = './tmp/test.h5'
        n_scales = 4
        ndim = len(shape)
        downscale_factors = n_scales * [[2] * ndim]
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
                exp_shape = tuple(sh // sf
                                  for sh, sf in zip(exp_shape, downscale_factors[scale]))

    def test_ds_nearest_3d(self):
        shape = (256,) * 3
        self._test_ds(shape, 'nearest')

    def test_ds_mean_3d(self):
        shape = (256,) * 3
        self._test_ds(shape, 'mean')

    def test_dtype(self):
        from pybdv import make_bdv
        shape = (128,) * 3

        val = np.iinfo('uint16').max + 1
        data = np.full(shape, val, dtype='uint32')

        out_path = './tmp/test.h5'
        make_bdv(data, out_path, convert_dtype=False)
        with h5py.File(out_path, 'r') as f:
            d = f['t00000/s00/0/cells'][:]
        self.assertTrue(np.array_equal(d, data))

        with self.assertRaises(RuntimeError):
            make_bdv(d, './tmp.test2.h5', convert_dtype=True)

    def test_custom_chunks(self):
        from pybdv import make_bdv
        shape = (128,) * 3
        chunks = (64,) * 3

        out_path = './tmp/test.h5'
        data = np.random.rand(*shape)
        make_bdv(data, out_path, chunks=chunks)

        with h5py.File(out_path, 'r') as f:
            d = f['t00000/s00/0/cells'][:]
        self.assertTrue(np.allclose(d, data))

    @unittest.skipIf(open_file is None, "Need elf for n5 support")
    def test_n5(self):
        from pybdv import make_bdv
        shape = (128,) * 3
        chunks = (64,) * 3

        out_path = './tmp/test.n5'
        data = np.random.rand(*shape)
        make_bdv(data, out_path, chunks=chunks)

        with open_file(out_path, 'r') as f:
            d = f['setup0/timepoint0/s0'][:]
        self.assertTrue(np.allclose(d, data))

    @unittest.skipIf(open_file is None, "Need elf for n5 support")
    def test_multi_threaded(self):
        from pybdv import make_bdv
        from pybdv.util import get_key
        shape = (128,) * 3
        chunks = (64,) * 3

        out_paths = ['./tmp/test.h5', './tmp/test.n5']
        data = np.random.rand(*shape)
        scale_factors = 2 * [[2, 2, 2]]
        for out_path, is_h5 in zip(out_paths, (True, False)):
            make_bdv(data, out_path, chunks=chunks,
                     n_threads=4, downscale_factors=scale_factors)
            key = get_key(is_h5, time_point=0, setup_id=0, scale=0)
            with open_file(out_path, 'r') as f:
                d = f[key][:]
            self.assertTrue(np.allclose(d, data))

    # 2d is not supported yet
    @unittest.skip
    def test_ds_nearest_2d(self):
        shape = (512,) * 2
        self._test_ds(shape, 'nearest')

    # 2d is not supported yet
    @unittest.skip
    def test_ds_mean_2d(self):
        shape = (512,) * 2
        self._test_ds(shape, 'mean')


if __name__ == '__main__':
    unittest.main()
