import os
import unittest
from subprocess import call
from shutil import rmtree

import numpy as np
import h5py


class TestConvertToBdv(unittest.TestCase):
    in_path = './tmp/in.h5'
    out_path = './tmp/out.h5'

    def setUp(self):
        os.makedirs('./tmp', exist_ok=True)
        with h5py.File(self.in_path) as f:
            f.create_dataset('data', data=np.random.rand(64, 64, 64), chunks=(32, 32, 32))

    def tearDown(self):
        try:
            rmtree('./tmp')
        except OSError:
            pass

    def check_result(self):
        with h5py.File(self.in_path, 'r') as f:
            exp = f['data'][:]

        with h5py.File(self.out_path, 'r') as f:
            self.assertTrue('t00000/s00/0/cells' in f)
            res = f['t00000/s00/0/cells'][:]

        self.assertEqual(res.shape, exp.shape)
        self.assertTrue(np.allclose(res, exp))

    def test_simple(self):
        from pybdv import convert_to_bdv
        convert_to_bdv(self.in_path, 'data', self.out_path)
        self.check_result()

    def test_cli(self):
        call(['convert_to_bdv', self.in_path, 'data', self.out_path])
        self.check_result()


if __name__ == '__main__':
    unittest.main()
