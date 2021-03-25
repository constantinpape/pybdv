import os
import unittest
from abc import ABC
from shutil import rmtree

import numpy as np
from pybdv.util import get_key, open_file, n5_file


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

        for vid in range(n_views + 1):
            expected_key = get_key(self.is_h5, timepoint=0, setup_id=vid, scale=0)
            with open_file(self.out_path, 'r') as f:
                self.assertTrue(expected_key in f)
                data = f[expected_key][:]
            exp_data = data_dict[vid]
            self.assertTrue(np.allclose(data, exp_data))

        # check affine trafos (only for explicit setup-ids)
        for vid in range(n_views):
            affine = affine_dict[vid]
            affine_out = get_affine(self.xml_path, vid)
            self.assertEqual(affine, affine_out)

    def test_multi_timepoint(self):
        from pybdv import make_bdv
        from pybdv.metadata import get_time_range

        n_timepoints = 6
        shape = (64,) * 3

        tp_data = []
        tp_setups = []
        for tp in range(n_timepoints):
            data = np.random.rand(*shape)
            # make sure that we at least have 2 setup ids that agree
            setup_id = np.random.randint(0, 20) if tp > 1 else 0
            make_bdv(data, self.out_path, setup_id=setup_id, timepoint=tp)
            tp_data.append(data)
            tp_setups.append(setup_id)

        tstart, tstop = get_time_range(self.xml_path)
        self.assertEqual(tstart, 0)
        self.assertEqual(tstop, n_timepoints - 1)

        for tp in range(n_timepoints):
            setup_id = tp_setups[tp]
            tp_key = get_key(self.is_h5, timepoint=tp, setup_id=setup_id, scale=0)
            with open_file(self.out_path, 'r') as f:
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

    def test_custom_attributes(self):
        from pybdv import make_bdv
        from pybdv.metadata import get_attributes
        shape = (64,) * 3

        data = np.random.rand(*shape)

        chan_name = 'DAPI'
        tile_name = 'some-tile'
        angle_name = 'some-angle'
        # write setup 0
        make_bdv(data, self.out_path, setup_id=0,
                 attributes={'channel': {'id': None, 'name': chan_name},
                             'tile': {'id': 2, 'name': tile_name},
                             'angle': {'id': 0, 'name': angle_name}})
        attrs_out = get_attributes(self.xml_path, 0)
        attrs_exp = {'channel': {'id': 0, 'name': chan_name},
                     'tile': {'id': 2, 'name': tile_name},
                     'angle': {'id': 0, 'name': angle_name}}
        self.assertEqual(attrs_out, attrs_exp)

        # write setup 1
        make_bdv(data, self.out_path, setup_id=None,
                 attributes={'channel': {'id': None},
                             'tile': {'id': 2},
                             'angle': {'id': 0}})
        attrs_out = get_attributes(self.xml_path, 1)
        attrs_exp = {'channel': {'id': 1},
                     'tile': {'id': 2, 'name': tile_name},
                     'angle': {'id': 0, 'name': angle_name}}
        self.assertEqual(attrs_out, attrs_exp)

        # write to setup 0 again with different timepoint
        make_bdv(data, self.out_path, setup_id=0, timepoint=1,
                 attributes={'channel': {'id': None},
                             'tile': {'id': 2},
                             'angle': {'id': 0}})
        attrs_out = get_attributes(self.xml_path, 0)
        attrs_exp = {'channel': {'id': 0, 'name': chan_name},
                     'tile': {'id': 2, 'name': tile_name},
                     'angle': {'id': 0, 'name': angle_name}}
        self.assertEqual(attrs_out, attrs_exp)

        # write next setup id without specifying all attribute names
        # -> should fail
        with self.assertRaises(ValueError):
            make_bdv(data, self.out_path, setup_id=None,
                     attributes={'channel': {'id': 5}, 'tile': {'id': 2}})

        # write next setup id with a new attribute name
        # -> should fail
        with self.assertRaises(ValueError):
            make_bdv(data, self.out_path, setup_id=None,
                     attributes={'channel': {'id': 5}, 'settings': {'id': 2}})

        # write exisiting setup id with  different attribute setup
        # -> should fail
        with self.assertRaises(ValueError):
            make_bdv(data, self.out_path, setup_id=0, timepoint=2,
                     attributes={'channel': {'id': 5}, 'tile': {'id': 2}, 'angle': {'id': 0}})

    def _test_overwrite(self, mode):
        from pybdv import make_bdv
        from pybdv.util import get_scale_factors, absolute_to_relative_scale_factors
        from pybdv.metadata import get_attributes, get_affine

        def _check(exp_data, exp_sf, exp_attrs, exp_affine):
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

        shape1 = (64,) * 3
        data1 = np.random.rand(*shape1)
        sf1 = [[2, 2, 2]]
        attrs1 = {'channel': {'id': 1}, 'angle': {'id': 2}}
        affine1 = np.random.rand(12).tolist()

        shape2 = (72,) * 3
        data2 = np.random.rand(*shape2)
        sf2 = [[1, 2, 2], [2, 2, 2]] if mode != 'metadata' else sf1
        attrs2 = {'channel': {'id': 3}, 'angle': {'id': 6}}
        affine2 = np.random.rand(12).tolist()

        make_bdv(data1, self.out_path, setup_id=0, timepoint=0,
                 downscale_factors=sf1, attributes=attrs1,
                 affine=affine1)
        _check(data1, sf1, attrs1, affine1)

        make_bdv(data2, self.out_path, setup_id=0, timepoint=0,
                 downscale_factors=sf2, attributes=attrs2, affine=affine2,
                 overwrite=mode)

        if mode == 'skip':
            _check(data1, sf1, attrs1, affine1)
        elif mode == 'all':
            _check(data2, sf2, attrs2, affine2)
        elif mode == 'data':
            _check(data2, sf2, attrs1, affine1)
        elif mode == 'metadata':
            _check(data1, sf1, attrs2, affine2)
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


class TestMakeBdvH5(MakeBdvTestMixin, unittest.TestCase):
    out_path = './tmp/test.h5'
    is_h5 = True


@unittest.skipIf(n5_file is None, "Need zarr or z5py for n5 support")
class TestMakeBdvN5(MakeBdvTestMixin, unittest.TestCase):
    out_path = './tmp/test.n5'
    is_h5 = False


if __name__ == '__main__':
    unittest.main()
