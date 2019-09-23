import os
import sys
import numpy as np
import h5py
from tqdm import tqdm

try:
    import z5py
except ImportError:
    z5py = None

from .util import blocking
from .metadata import write_h5_metadata, write_xml_metadata
from .downsample import downsample

Z5_EXTENSIONS = ['.n5', '.zr', '.zarr']
HDF5_EXTENSIONS = ['.h5', '.hdf', '.hdf5']
XML_EXTENSIONS = ['.xml']


# TODO use elf.io.open_file instead
def file_reader(input_path, mode='r'):
    ext = os.path.splitext(input_path)[1]
    if ext.lower() in HDF5_EXTENSIONS:
        return h5py.File(input_path, mode)
    elif ext.lower() in Z5_EXTENSIONS:
        if z5py is None:
            raise RuntimeError("z5py was not found")
        return z5py.File(input_path, mode)
    else:
        raise RuntimeError("Unknown file extensions %s" % ext)


def handle_setup_id(setup_id, h5_path):
    if os.path.exists(h5_path):
        with h5py.File(h5_path, 'r') as f:
            setup_ids = list(f['t00000'].keys())
            setup_ids = [int(sid[1:]) for sid in setup_ids]
    else:
        setup_ids = [-1]
    if setup_id is None:
        setup_id = max(setup_ids) + 1
    else:
        if setup_id in setup_ids:
            overwrite = input("Setup-id %i is alread present. Do you want to over-write it? y / [n]:")
            if overwrite != 'y':
                sys.exit(0)
    assert setup_id < 100, "Only up to 100 set-ups are supported"
    return setup_id


def copy_dataset(input_path, input_key, output_path, output_key,
                 chunks=(64, 64, 64), dtype=None):

    with file_reader(input_path, 'r') as f_in,\
            h5py.File(output_path) as f_out:

        ds_in = f_in[input_key]
        shape = ds_in.shape
        out_dtype = ds_in.dtype if dtype is None else dtype
        ds_out = f_out.create_dataset(output_key, shape=shape, chunks=chunks,
                                      compression='gzip', dtype=out_dtype)

        def copy_chunk(bb):
            data = ds_in[bb]
            # skip empty chunks
            if data.sum() == 0:
                return
            ds_out[bb] = data

        print("Copy initial dataset from: %s:%s to %s:%s" % (input_path, input_key,
                                                             output_path, output_key))
        for bb in tqdm(blocking(shape, chunks)):
            copy_chunk(bb)


def normalize_output_path(output_path):
    # construct hdf5 output path and xml output path from output path
    base_path, ext = os.path.splitext(output_path)
    if ext == '':
        h5_path = output_path + '.h5'
        xml_path = output_path + '.xml'
    elif ext.lower() in HDF5_EXTENSIONS:
        h5_path = output_path
        xml_path = base_path + '.xml'
    elif ext.lower() in XML_EXTENSIONS:
        h5_path = base_path + '.h5'
        xml_path = output_path
    else:
        raise ValueError("File extension %s not supported" % ext)
    return h5_path, xml_path


def make_scales(h5_path, downscale_factors, downscale_mode, ndim, setup_id):
    assert downscale_mode in ('nearest', 'mean', 'max', 'min', 'interpolate')
    assert all(isinstance(factor, (int, tuple, list)) for factor in downscale_factors)
    assert all(len(factor) == 3 for factor in downscale_factors
               if isinstance(factor, (tuple, list)))
    # normalize all factors to be tuple or list
    factors = [ndim*[factor] if isinstance(factor, int) else factor
               for factor in downscale_factors]

    # run single downsampling stages
    for scale, factor in enumerate(factors):
        in_key = 't00000/s%02i/%i/cells' % (setup_id, scale)
        out_key = 't00000/s%02i/%i/cells' % (setup_id, scale + 1)
        print("Downsample scale %i / %i" % (scale + 1, len(factors)))
        downsample(h5_path, in_key, out_key, factor, downscale_mode)

    # add first level to factors
    factors = [[1, 1, 1]] + factors
    return factors


# TODO expose 'offsets' parameter
# TODO support multiple time-points
# TODO replace assertions with more meaningfull errors
def convert_to_bdv(input_path, input_key, output_path,
                   downscale_factors=None, downscale_mode='nearest',
                   resolution=[1., 1., 1.], unit='pixel', dtype=None,
                   setup_id=None, setup_name=None):
    """ Convert hdf5 volume to BigDatViewer format.

    Optionally downscale the input volume and write it
    to BigDataViewer scale pyramid.

    Args:
        input_path (str): path to hdf5 input volume
        input_key (str): path in hdf5 input file
        output_path (str): output path to bdv file
        downscale_factors (tuple or list): tuple or list of downscaling
            factors. Can be anisotropic. No downscaling by default.
        downscale_mode (str): mode used for downscaling.
            Can be 'mean', 'max', 'min', 'nearest' or 'interpolate' (default:'nerarest').
        resolution(list or tuple): resolution of the data
        unit (str): unit of measurement
        dtype (str or np.dtype): dtype of output dataset.
            By default, the dtype of the input data is used.
        setup_id (int): id of this view set-up. By default, the next free id is chosen (default: None).
        setup_name (str): name of this view set-up (default: None)
    """
    # validate input data arguments
    assert os.path.exists(input_path), input_path
    with file_reader(input_path, 'r') as f:
        assert input_key in f, "%s not in %s" % (input_key, input_path)
        shape = f[input_key].shape
        ndim = len(shape)
    assert ndim == 3, "Only support 3d"
    assert len(resolution) == ndim

    h5_path, xml_path = normalize_output_path(output_path)
    setup_id = handle_setup_id(setup_id, h5_path)

    # copy the initial dataset
    base_key = 't00000/s%02i/0/cells' % setup_id
    copy_dataset(input_path, input_key,
                 h5_path, base_key, dtype=dtype)

    # downsample if needed
    if downscale_factors is None:
        # set single level downscale factor
        factors = [[1, 1, 1]]
    else:
        factors = make_scales(h5_path, downscale_factors, downscale_mode, ndim, setup_id)

    # write bdv metadata
    write_h5_metadata(h5_path, factors, setup_id)
    write_xml_metadata(xml_path, h5_path, unit, resolution,
                       setup_id=setup_id,
                       setup_name=setup_name)


def make_bdv(data, output_path,
             downscale_factors=None, downscale_mode='nearest',
             resolution=[1., 1., 1.], unit='pixel',
             setup_id=None, setup_name=None):
    """ Write data to BigDatViewer format.

    Optionally downscale the input data to BigDataViewer scale pyramid.

    Args:
        data (np.ndarray): input data
        output_path (str): output path to bdv file
        downscale_factors (tuple or list): tuple or list of downscaling
            factors. Can be anisotropic. No downscaling by default.
        downscale_mode (str): mode used for downscaling.
            Can be 'mean', 'max', 'min', 'nearest' or 'interpolate' (default:'nerarest').
        resolution(list or tuple): resolution of the data
        unit (str): unit of measurement
        setup_id (int): id of this view set-up. By default, the next free id is chosen (default: None).
        setup_name (str): name of this view set-up (default: None)
    """
    # validate input data arguments
    assert isinstance(data, np.ndarray), "Input needs to be numpy array"
    ndim = data.ndim
    assert ndim == 3, "Only support 3d"
    assert len(resolution) == ndim

    h5_path, xml_path = normalize_output_path(output_path)
    setup_id = handle_setup_id(setup_id, h5_path)

    # write initial dataset
    base_key = 't00000/s%02i/0/cells' % setup_id
    with h5py.File(h5_path) as f:
        f.create_dataset(base_key, data=data, compression='gzip')

    # downsample if needed
    if downscale_factors is None:
        # set single level downscale factor
        factors = [[1, 1, 1]]
    else:
        factors = make_scales(h5_path, downscale_factors, downscale_mode, ndim, setup_id)

    # write bdv metadata
    write_h5_metadata(h5_path, factors, setup_id)
    write_xml_metadata(xml_path, h5_path,
                       unit, resolution,
                       setup_id=setup_id,
                       setup_name=setup_name)
