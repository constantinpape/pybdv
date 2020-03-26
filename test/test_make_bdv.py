import os
import unittest
from abc import ABC
from shutil import rmtree

import numpy as np
from pybdv.util import get_key

try:
    from elf.io import open_file
    WITH_ELF = True
except ImportError:
    import h5py
    open_file = h5py.File
    WITH_ELF = False


class MakeBdvTestMixin(ABC):
    tmp_folder = './tmp'
    xml_path = './tmp/test.xml'

    def setUp(self):
        os.makedirs(self.tmp_folder, exist_ok=True)

    def tearDown(self):
        try:
            rmtree(self.tmp_folder)
        except OSError:
            pass

    def _test_simple(self, shape, affine=None):
        from pybdv import make_bdv
        data = np.random.rand(*shape).astype('float32')

        make_bdv(data, self.out_path, affine=affine)

        key = get_key(self.is_h5, timepoint=0, setup_id=0, scale=0)
        self.assertTrue(os.path.exists(self.out_path))
        with open_file(self.out_path, 'r') as f:
            self.assertTrue(key in f)
            ds = f[key]
            self.assertEqual(ds.shape, shape)
            out_data = ds[:]
        self.assertTrue(np.allclose(data, out_data))

    def test_simple(self):
        shape = (100, 100, 100)
        self._test_simple(shape)

    def test_affine(self):
        from pybdv.metadata import get_affine
        shape = (100, 100, 100)
        affine = np.random.rand(12).tolist()
        affine = [round(aff, 4) for aff in affine]
        self._test_simple(shape, affine)
        affine_out = get_affine(self.xml_path, 0)['affine0']
        self.assertEqual(affine, affine_out)

    def test_multi_setup(self):
        from pybdv import make_bdv
        from pybdv.metadata import get_affine
        shape = (64,) * 3
        n_views = 2

        data_dict = {}
        affine_dict = {}

        for vid in range(n_views):
            data = np.random.rand(*shape).astype('float32')
            affine = {'trafo1': [round(aff, 4) for aff in np.random.rand(12)],
                      'trafo2': [round(aff, 4) for aff in np.random.rand(12)]}
            make_bdv(data, self.out_path, setup_id=vid, affine=affine)
            data_dict[vid] = data
            affine_dict[vid] = affine

        # check implicit setup id
        data = np.random.rand(*shape).astype('float32')
        make_bdv(data, self.out_path)
        data_dict[n_views] = data

        with open_file(self.out_path, 'r') as f:
            for vid in range(n_views + 1):
                expected_key = get_key(self.is_h5, timepoint=0, setup_id=vid, scale=0)
                self.assertTrue(expected_key in f)

                exp_data = data_dict[vid]
                data = f[expected_key][:]
                self.assertTrue(np.allclose(data, exp_data))

        # check affine trafos (only for explicit setup-ids)
        for vid in range(n_views):
            affine = affine_dict[vid]
            affine_out = get_affine(self.xml_path, vid)
            self.assertEqual(affine, affine_out)

    def test_multi_timepoint(self):
        from pybdv import make_bdv
        from pybdv.metadata import get_time_range

        n_timepoints = 4
        shape = (64,) * 3

        tp_data = []
        tp_setups = []
        for tp in range(n_timepoints):
            data = np.random.rand(*shape)
            setup_id = np.random.randint(0, 10)
            make_bdv(data, self.out_path, setup_id=setup_id, timepoint=tp)
            tp_data.append(data)
            tp_setups.append(setup_id)

        tstart, tstop = get_time_range(self.xml_path)
        self.assertEqual(tstart, 0)
        self.assertEqual(tstop, n_timepoints - 1)

        with open_file(self.out_path, 'r') as f:
            for tp in range(n_timepoints):
                setup_id = tp_setups[tp]
                tp_key = get_key(self.is_h5, timepoint=tp, setup_id=setup_id, scale=0)
                data = f[tp_key][:]
                data_exp = tp_data[tp]
                self.assertTrue(np.allclose(data, data_exp))

    def _test_ds(self, shape, mode):
        from pybdv import make_bdv
        data = np.random.rand(*shape).astype('float32')

        n_scales = 4
        ndim = len(shape)
        downscale_factors = n_scales * [[2] * ndim]
        make_bdv(data, self.out_path, downscale_factors,
                 downscale_mode=mode)

        exp_shape = shape
        self.assertTrue(os.path.exists(self.out_path))
        with open_file(self.out_path, 'r') as f:
            for scale in range(n_scales):
                key = get_key(self.is_h5, timepoint=0, setup_id=0, scale=scale)
                self.assertTrue(key in f)
                ds = f[key]
                self.assertEqual(ds.shape, exp_shape)
                exp_shape = tuple(sh // sf
                                  for sh, sf in zip(exp_shape, downscale_factors[scale]))

    def test_ds_nearest(self):
        shape = (256,) * 3
        self._test_ds(shape, 'nearest')

    def test_ds_mean(self):
        shape = (256,) * 3
        self._test_ds(shape, 'mean')

    def test_dtype(self):
        if not self.is_h5:
            return

        from pybdv import make_bdv
        shape = (128,) * 3

        val = np.iinfo('uint16').max + 1
        data = np.full(shape, val, dtype='uint32')

        make_bdv(data, self.out_path, convert_dtype=False)
        with open_file(self.out_path, 'r') as f:
            key = get_key(self.is_h5, timepoint=0, setup_id=0, scale=0)
            d = f[key][:]
        self.assertTrue(np.array_equal(d, data))

        with self.assertRaises(RuntimeError):
            make_bdv(d, './tmp.test2.h5', convert_dtype=True)

    def test_custom_chunks(self):
        from pybdv import make_bdv
        shape = (128,) * 3
        chunks = (64, 42, 59)

        data = np.random.rand(*shape)
        make_bdv(data, self.out_path, chunks=chunks)

        key = get_key(self.is_h5, timepoint=0, setup_id=0, scale=0)
        with open_file(self.out_path, 'r') as f:
            ds = f[key]
            chunks_out = ds.chunks
            d = ds[:]
            self.assertEqual(chunks, chunks_out)
        self.assertTrue(np.allclose(d, data))

    def test_multi_threaded(self):
        from pybdv import make_bdv
        shape = (128,) * 3
        chunks = (64,) * 3

        data = np.random.rand(*shape)
        scale_factors = 2 * [[2, 2, 2]]

        make_bdv(data, self.out_path, chunks=chunks,
                 n_threads=4, downscale_factors=scale_factors)
        key = get_key(self.is_h5, timepoint=0, setup_id=0, scale=0)
        with open_file(self.out_path, 'r') as f:
            d = f[key][:]
        self.assertTrue(np.allclose(d, data))


class TestMakeBdvH5(MakeBdvTestMixin, unittest.TestCase):
    out_path = './tmp/test.h5'
    is_h5 = True


@unittest.skipUnless(WITH_ELF, "Need elf for n5 support")
class TestMakeBdvN5(MakeBdvTestMixin, unittest.TestCase):
    out_path = './tmp/test.n5'
    is_h5 = False


if __name__ == '__main__':
    unittest.main()
