import os
import unittest
from abc import ABC
from subprocess import call
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


class ConvertToBdvTestMixin(ABC):
    tmp_folder = './tmp'

    def setUp(self):
        os.makedirs(self.tmp_folder, exist_ok=True)
        with open_file(self.in_path, 'a') as f:
            f.create_dataset('data', data=np.random.rand(64, 64, 64), chunks=(32, 32, 32))

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

    def test_cli(self):
        call(['convert_to_bdv', self.in_path, 'data', self.out_path])
        self.check_result()


class TestConvertToBdvH5(ConvertToBdvTestMixin, unittest.TestCase):
    in_path = './tmp/in.h5'
    out_path = './tmp/test.h5'
    is_h5 = True


@unittest.skipUnless(WITH_ELF, "Need elf for n5 support")
class TestConvertToBdvN5(ConvertToBdvTestMixin, unittest.TestCase):
    in_path = './tmp/in.n5'
    out_path = './tmp/test.n5'
    is_h5 = False


if __name__ == '__main__':
    unittest.main()
