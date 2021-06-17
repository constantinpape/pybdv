import os
import unittest

import h5py
import numpy as np

from pybdv.downsample import get_downsampler, sample_shape

DOWNSCALING_MODES = ['nearest', 'mean', 'max', 'min', 'interpolate']


class TestDownsample(unittest.TestCase):
    test_path = 'tmp.h5'

    def tearDown(self):
        if os.path.exists(self.test_path):
            os.remove(self.test_path)

    def check_block_artifacts(self, vol, block_shape):
        first_block_border = np.s_[
            block_shape[0] - 1:block_shape[0] + 1,
            0:block_shape[1],
            0:block_shape[2]
        ]
        first_border = vol[first_block_border]
        for slice_ in first_border:
            self.assertFalse(np.allclose(slice_, 0))

    def _test_downsample(self, shape, block_shape, scale_factor,
                         modes=DOWNSCALING_MODES):
        from pybdv.downsample import downsample

        input_vol = np.random.rand(*shape)

        in_key = 'data_in'
        with h5py.File(self.test_path, 'a') as f:
            if in_key in f:
                del f[in_key]
            f.create_dataset(in_key, data=input_vol, chunks=block_shape)

        for mode in modes:

            out_key = f'ds_{mode}'
            downsample(self.test_path, in_key, out_key,
                       scale_factor, mode, n_threads=1)
            with h5py.File(self.test_path, 'r') as f:
                vol = f[out_key][:]

            downsampler = get_downsampler(mode)
            exp_vol = downsampler(input_vol, scale_factor,
                                  sample_shape(input_vol.shape, scale_factor))
            # interpolate is more tricky ...
            if mode == 'interpolate':
                self.assertEqual(exp_vol.shape, vol.shape)
                self.check_block_artifacts(vol, block_shape)
            else:
                self.assertTrue(np.allclose(vol, exp_vol))

    def test_downsample(self):
        # test with regular shape and chunks for all modes
        self._test_downsample(shape=(256, 256, 256),
                              block_shape=(32, 32, 32),
                              scale_factor=(2, 2, 2))

    # test for multiple shape / block shape combinations
    def test_downsample_shapes(self):
        shapes = [
            (128, 128, 144),
            (123, 71, 97)
        ]
        block_shapes = [
            (64, 64, 64),
            (33, 7, 93)
        ]
        for shape in shapes:
            for block_shape in block_shapes:
                self._test_downsample(shape=shape,
                                      block_shape=block_shape,
                                      scale_factor=(2, 2, 2),
                                      modes=['nearest'])
                os.remove(self.test_path)

    def test_downsample_in_memory(self):
        from pybdv.downsample import downsample_in_memory

        shape = (256, 256, 256)
        block_shape = (32, 32, 32)
        input_vol = np.random.rand(*shape)

        scale_factors = 4 * [[2, 2, 2]]

        for mode in DOWNSCALING_MODES:
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

    # def _test_block_artifacts(self, mode):
    #     from pybdv.downsample import downsample
    #     shape = (64, 64)
    #     scale_factor = 2
    #     input_vol = np.random.rand(*shape)
    #     out = downsample(input_vol)

    # # see https://github.com/constantinpape/pybdv/issues/38
    # def test_block_artifacts(self):
    #     for mode in DOWNSCALING_MODES:
    #         self._test_block_artifacts(mode)


if __name__ == '__main__':
    unittest.main()
