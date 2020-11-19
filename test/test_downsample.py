import os
import unittest

import h5py
import numpy as np

from pybdv.downsample import get_downsampler, sample_shape


class TestDownsample(unittest.TestCase):
    test_path = 'tmp.h5'

    def tearDown(self):
        if os.path.exists(self.test_path):
            os.remove(self.test_path)

    def test_downsample(self):
        from pybdv.downsample import downsample

        shape = (256, 256, 256)
        block_shape = (32, 32, 32)
        scale_factor = [2, 2, 2]
        input_vol = np.random.rand(*shape)

        in_key = 'data_in'

        with h5py.File(self.test_path, 'a') as f:
            f.create_dataset(in_key, data=input_vol, chunks=block_shape)

        modes = ['nearest', 'mean', 'max', 'min', 'interpolate']
        for mode in modes:

            out_key = f'ds_{mode}'
            downsample(self.test_path, in_key, out_key,
                       scale_factor, mode, n_threads=4)
            with h5py.File(self.test_path, 'r') as f:
                vol = f[out_key][:]

            downsampler = get_downsampler(mode)
            exp_vol = downsampler(input_vol, scale_factor,
                                  sample_shape(input_vol.shape, scale_factor))
            # interpolate is more tricky ...
            if mode == 'interpolate':
                self.assertEqual(exp_vol.shape, vol.shape)
            else:
                self.assertTrue(np.allclose(vol, exp_vol))

    def test_downsample_in_memory(self):
        from pybdv.downsample import downsample_in_memory

        shape = (256, 256, 256)
        block_shape = (32, 32, 32)
        input_vol = np.random.rand(*shape)

        scale_factors = 4 * [[2, 2, 2]]

        modes = ['nearest', 'mean', 'max', 'min', 'interpolate']
        for mode in modes:
            exp_vol = input_vol.copy()
            ds_vols = downsample_in_memory(input_vol,
                                           scale_factors,
                                           mode,
                                           block_shape,
                                           n_threads=4)
            downsampler = get_downsampler(mode)
            for factor, vol in zip(scale_factors, ds_vols):
                exp_vol = downsampler(exp_vol, factor,
                                      sample_shape(exp_vol.shape, factor))
                # interpolate is more tricky ...
                if mode == 'interpolate':
                    self.assertEqual(exp_vol.shape, vol.shape)
                else:
                    self.assertTrue(np.allclose(vol, exp_vol))


if __name__ == '__main__':
    unittest.main()
