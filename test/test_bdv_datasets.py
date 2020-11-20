import os
import unittest
from abc import ABC
from shutil import rmtree

import numpy as np
from pybdv import make_bdv, BdvDataset
from pybdv.util import open_file, get_key, HDF5_EXTENSIONS

try:
    import z5py
except ImportError:
    z5py = None

DOWNSCALE_MODE = 'mean'


class BdvDatasetTestMixin(ABC):
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
        # data = np.random.rand(*self.shape)
        data = np.zeros(self.shape)
        make_bdv(data, self.out_path, downscale_factors=self.scale_factors, downscale_mode=DOWNSCALE_MODE)

    def check_data(self, pos, shp):

        is_h5 = os.path.splitext(self.out_path)[1] in HDF5_EXTENSIONS

        # Make the reference data from scratch
        ref_data = np.zeros(self.shape)
        pos = np.array(pos)
        offset = np.array((0,) * 3)
        offset[pos < 0] = pos[pos < 0]
        pos[pos < 0] = 0
        ref_data[
            pos[0]: pos[0] + shp[0] + offset[0],
            pos[1]: pos[1] + shp[1] + offset[1],
            pos[2]: pos[2] + shp[2] + offset[2]
        ] = 1
        if is_h5:
            ref_out_path = os.path.join(self.tmp_folder, 'ref_tmp_data.h5')
        else:
            ref_out_path = os.path.join(self.tmp_folder, 'ref_tmp_data.n5')
        make_bdv(ref_data, ref_out_path, downscale_factors=self.scale_factors, downscale_mode=DOWNSCALE_MODE)

        for idx, scale in enumerate(self.abs_scale_factors):
            with open_file(self.out_path, mode='r') as f:
                data = f[get_key(is_h5, timepoint=0, setup_id=0, scale=idx)][:]
            with open_file(ref_out_path, mode='r') as f:
                ref_data = f[get_key(is_h5, timepoint=0, setup_id=0, scale=idx)][:]

            # Now check for the difference
            self.assertEqual(np.abs(data - ref_data).max(), 0)

    def test_write_data(self):

        pos = (32,) * 3
        shp = (64,) * 3
        print(f'pos = {pos}; shp = {shp}')

        vol = np.ones(shp)

        ds = BdvDataset(self.out_path, timepoint=0, setup_id=0, downscale_mode=DOWNSCALE_MODE)
        ds[
            pos[0]: pos[0] + shp[0],
            pos[1]: pos[1] + shp[1],
            pos[2]: pos[2] + shp[2]
        ] = vol

        self.check_data(pos, shp)

    def test_write_out_of_bounds_pos(self):

        pos = (96,) * 3
        shp = (64,) * 3
        print(f'pos = {pos}; shp = {shp}')

        vol = np.ones(shp)

        ds = BdvDataset(self.out_path, timepoint=0, setup_id=0, downscale_mode=DOWNSCALE_MODE)
        ds[
            pos[0]: pos[0] + shp[0],
            pos[1]: pos[1] + shp[1],
            pos[2]: pos[2] + shp[2]
        ] = vol

        self.check_data(pos, shp)

    def test_write_out_of_bounds_neg(self):

        pos = (-32,) * 3
        shp = (64,) * 3
        print(f'pos = {pos}; shp = {shp}')

        vol = np.ones(shp)

        ds = BdvDataset(self.out_path, timepoint=0, setup_id=0, downscale_mode=DOWNSCALE_MODE)
        ds[
            pos[0]: pos[0] + shp[0],
            pos[1]: pos[1] + shp[1],
            pos[2]: pos[2] + shp[2]
        ] = vol

        self.check_data(pos, shp)

    def test_write_data_subpx(self):

        pos = (35,) * 3
        shp = (66,) * 3
        print(f'pos = {pos}; shp = {shp}')

        vol = np.ones(shp)

        ds = BdvDataset(self.out_path, timepoint=0, setup_id=0, downscale_mode=DOWNSCALE_MODE, verbose=True)
        ds[
            pos[0]: pos[0] + shp[0],
            pos[1]: pos[1] + shp[1],
            pos[2]: pos[2] + shp[2]
        ] = vol

        self.check_data(pos, shp)

    def test_write_out_of_bounds_pos_subpx(self):

        pos = (99,) * 3
        shp = (66,) * 3
        print(f'pos = {pos}; shp = {shp}')

        vol = np.ones(shp)

        ds = BdvDataset(self.out_path, timepoint=0, setup_id=0, downscale_mode=DOWNSCALE_MODE)
        ds[
            pos[0]: pos[0] + shp[0],
            pos[1]: pos[1] + shp[1],
            pos[2]: pos[2] + shp[2]
        ] = vol

        self.check_data(pos, shp)

    def test_write_out_of_bounds_neg_subpx(self):

        pos = (-35,) * 3
        shp = (66,) * 3
        print(f'pos = {pos}; shp = {shp}')

        vol = np.ones(shp)

        ds = BdvDataset(self.out_path, timepoint=0, setup_id=0, downscale_mode=DOWNSCALE_MODE)
        ds[
            pos[0]: pos[0] + shp[0],
            pos[1]: pos[1] + shp[1],
            pos[2]: pos[2] + shp[2]
        ] = vol

        self.check_data(pos, shp)


class TestBdvDatasetH5(BdvDatasetTestMixin, unittest.TestCase):
    out_path = './tmp/test.h5'


@unittest.skipUnless(z5py is not None, "Need z5py for n5 support")
class TestBdvDatasetN5(BdvDatasetTestMixin, unittest.TestCase):
    out_path = './tmp/test.n5'


if __name__ == '__main__':
    unittest.main()
