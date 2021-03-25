import os
import unittest
from abc import ABC
from shutil import rmtree

import numpy as np
from pybdv import make_bdv
from pybdv.util import n5_file


class MetadataTestMixin(ABC):
    tmp_folder = './tmp'
    xml_path = './tmp/test.xml'
    xml_path2 = './tmp/test2.xml'

    shape = (64,) * 3
    chunks = (32,) * 3

    resolution1 = [4.4, 3.8, 4.]
    resolution2 = [1., 3.1415926, 42.]

    def setUp(self):
        os.makedirs(self.tmp_folder, exist_ok=True)
        data = np.random.rand(*self.shape)
        make_bdv(data, self.out_path, resolution=self.resolution1, chunks=self.chunks)
        make_bdv(data, self.out_path, resolution=self.resolution2, chunks=self.chunks,
                 setup_id=1)

    def tearDown(self):
        try:
            rmtree(self.tmp_folder)
        except OSError:
            pass

    def test_get_bdv_format(self):
        from pybdv.metadata import get_bdv_format
        bdv_format = get_bdv_format(self.xml_path)
        self.assertEqual(bdv_format, self.bdv_format)

    def test_get_resolution(self):
        from pybdv.metadata import get_resolution

        resolution = get_resolution(self.xml_path, 0)
        self.assertEqual(resolution, self.resolution1)

        resolution = get_resolution(self.xml_path, 1)
        self.assertEqual(resolution, self.resolution2)

    def test_get_data_path(self):
        from pybdv.metadata import get_data_path
        path = get_data_path(self.xml_path)
        exp_path = os.path.split(self.out_path)[1]
        self.assertEqual(path, exp_path)
        abs_path = get_data_path(self.xml_path, return_absolute_path=True)
        self.assertEqual(abs_path, os.path.abspath(self.out_path))

    def test_validate_attributes(self):
        from pybdv.metadata import validate_attributes

        attrs1 = {'channel': {'id': 0, 'name': 'foo'}}
        attrs1_ = validate_attributes(self.xml_path, attrs1, 0, True)
        self.assertEqual(attrs1, attrs1_)

        attrs2 = {'channel': {'id': None, 'name': 'foo'}}
        attrs2_exp = {'channel': {'id': 0, 'name': 'foo'}}
        attrs2_ = validate_attributes(self.xml_path, attrs2, 0, True)
        self.assertEqual(attrs2_exp, attrs2_)

        attrs3 = {'channel': {'name': 'bar'}}
        with self.assertRaises(ValueError):
            validate_attributes(self.xml_path, attrs3, 1, True)

        attrs4 = {'displaysettings': {'id': 0, 'name': 'baz', 'min': 0, 'max': 1, 'isset': True,
                                      'color': [255, 255, 255, 255]}}
        make_bdv(np.random.rand(*self.shape), self.out_path2, resolution=self.resolution1, chunks=self.chunks,
                 attributes=attrs4)
        attrs4 = {'displaysettings': {'id': 0, 'name': 'baz', 'min': 0, 'max': 1, 'isset': True,
                                      'color': [255, 255, 255, 255]}}
        attrs4_ = validate_attributes(self.xml_path2, attrs4, 1, True)
        self.assertEqual(attrs4, attrs4_)

        attrs5 = {'displaysettings': {'id': 1, 'name': 'biz', 'min': 0, 'max': 1}}
        with self.assertRaises(ValueError):
            validate_attributes(self.xml_path2, attrs5, 1, True)

    def test_get_name(self):
        from pybdv.metadata import get_name
        name = get_name(self.xml_path, setup_id=0)
        self.assertEqual(name, "Setup0")

    def test_write_name(self):
        from pybdv.metadata import get_name, write_name
        name = "MyName"
        write_name(self.xml_path, setup_id=0, name=name)
        got = get_name(self.xml_path, setup_id=0)
        self.assertEqual(name, got)


class TestMetadataH5(MetadataTestMixin, unittest.TestCase):
    out_path = './tmp/test.h5'
    out_path2 = './tmp/test2.h5'
    bdv_format = 'bdv.hdf5'


@unittest.skipIf(n5_file is None, "Need zarr or z5py for n5 support")
class TestMetadataN5(MetadataTestMixin, unittest.TestCase):
    out_path = './tmp/test.n5'
    out_path2 = './tmp/test2.n5'
    bdv_format = 'bdv.n5'


if __name__ == '__main__':
    unittest.main()
