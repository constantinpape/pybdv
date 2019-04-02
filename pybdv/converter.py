import os
import numpy as np
import h5py

from .util import blocking
from .metadata import write_h5_metadata, write_xml_metadata
from .downsample import downsample

HDF5_EXTENSIONS = ['.h5', '.hdf', '.hdf5']
XML_EXTENSIONS = ['.xml']


def copy_dataset(input_path, input_key, output_path, output_key,
                 chunks=(64, 64, 64)):

    with h5py.File(input_path, 'r') as f_in,\
            h5py.File(output_path) as f_out:

        ds_in = f_in[input_key]
        shape = ds_in.shape
        ds_out = f_out.create_dataset(output_key, shape=shape, chunks=chunks,
                                      compression='gzip', dtype=ds_in.dtype)

        def copy_chunk(bb):
            data = ds_in[bb]
            ds_out[bb] = data

        for bb in blocking(shape, chunks):
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


def make_scales(h5_path, downscale_factors, downscale_mode, ndim):
    assert downscale_mode in ('nearest', 'interpolate')
    assert all(isinstance(factor, (int, tuple, list)) for factor in downscale_factors)
    assert all(len(factor) == 3 for factor in downscale_factors
               if isinstance(factor, (tuple, list)))
    # normalize all factors to be tuple or list
    factors = [ndim*[factor] if isinstance(factor, int) else factor
               for factor in downscale_factors]

    # run single downsampling stages
    for scale, factor in enumerate(factors):
        in_key = 't00000/s00/%i/cells' % scale
        out_key = 't00000/s00/%i/cells' % (scale + 1,)
        downsample(h5_path, in_key, out_key, factor, downscale_mode)

    # add first level to factors
    factors = [[1, 1, 1]] + factors
    return factors


# TODO expose 'offsets' parameter
# TODO support multiple time points and set-ups
# TODO replace assertions with more meaningfull errors
def convert_to_bdv(input_path, input_key, output_path,
                   downscale_factors=None, downscale_mode='nearest',
                   resolution=[1., 1., 1.], unit='pixel'):
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
            Either 'nearest' or 'interpolate'.
        resolution(list or tuple): resolution of the data
        unit (str): unit of measurement
    """
    # validate input data arguments
    assert os.path.exists(input_path), input_path
    with h5py.File(input_path, 'r') as f:
        assert input_key in f, "%s not in %s" % (input_key, input_path)
        shape = f[input_key].shape
        ndim = len(shape)
    # TODO support arbitrary dimensions
    assert ndim == 3, "Only support 3d"
    assert len(resolution) == ndim

    h5_path, xml_path = normalize_output_path(output_path)

    # copy the initial dataset
    base_key = 't00000/s00/0/cells'
    copy_dataset(input_path, input_key,
                 h5_path, base_key)

    # downsample if needed
    if downscale_factors is None:
        # set single level downscale factor
        factors = [[1, 1, 1]]
    else:
        factors = make_scales(h5_path, downscale_factors, downscale_mode, ndim)

    # write bdv metadata
    write_h5_metadata(h5_path, factors)
    write_xml_metadata(xml_path, h5_path, unit, resolution)


def make_bdv(data, output_path,
             downscale_factors=None, downscale_mode='nearest',
             resolution=[1., 1., 1.], unit='pixel'):
    """ Write data to BigDatViewer format.

    Optionally downscale the input data to BigDataViewer scale pyramid.

    Args:
        data (np.ndarray): input data
        output_path (str): output path to bdv file
        downscale_factors (tuple or list): tuple or list of downscaling
            factors. Can be anisotropic. No downscaling by default.
        downscale_mode (str): mode used for downscaling.
            Either 'nearest' or 'interpolate'.
        resolution(list or tuple): resolution of the data
        unit (str): unit of measurement
    """
    # validate input data arguments
    assert isinstance(data, np.ndarray), "Input needs to be numpy array"
    ndim = data.ndim
    # TODO support arbitrary dimensions
    assert ndim == 3, "Only support 3d"
    assert len(resolution) == ndim

    h5_path, xml_path = normalize_output_path(output_path)

    # write initial dataset
    base_key = 't00000/s00/0/cells'
    with h5py.File(h5_path) as f:
        f.create_dataset(base_key, data=data, compression='gzip')

    # downsample if needed
    if downscale_factors is None:
        # set single level downscale factor
        factors = [[1, 1, 1]]
    else:
        factors = make_scales(h5_path, downscale_factors, downscale_mode, ndim)

    # write bdv metadata
    write_h5_metadata(h5_path, factors)
    write_xml_metadata(xml_path, h5_path, unit, resolution)
