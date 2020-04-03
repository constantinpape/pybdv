import os
import numpy as np
from concurrent import futures
from tqdm import tqdm

from .util import (blocking, get_nblocks, open_file, get_key,
                   HAVE_ELF, HDF5_EXTENSIONS, N5_EXTENSIONS, XML_EXTENSIONS)
from .metadata import (validate_affine, validate_attributes,
                       write_h5_metadata, write_xml_metadata, write_n5_metadata)
from .downsample import downsample
from .dtypes import convert_to_bdv_dtype, get_new_dtype


def handle_setup_id(setup_id, data_path, is_h5, timepoint, overwrite):

    # get the existing setup ids
    if os.path.exists(data_path):
        with open_file(data_path, 'r') as f:
            if is_h5:
                tp_key = 't%05i' % timepoint
                if tp_key in f:
                    g = f[tp_key]
                    setup_ids = list(g.keys())
                    setup_ids = [int(sid[1:]) for sid in setup_ids]
                else:
                    setup_ids = []
            else:
                setup_ids = [key for key in f.keys()
                             if (key.startswith('setup') and
                                 'timepoint%i' % timepoint in f[key])]
                setup_ids = [int(sid[5:]) for sid in setup_ids]
    else:
        setup_ids = [-1]

    # we have three modes:
    # 0: normal mode
    # 1: the setup id is present already and we do not over-write -> skip
    # 2: the setup id is present already and we do over-write -> over-write
    mode = 0

    # setup_id is None
    # -> select the next available setup id
    if setup_id is None:
        setup_id = max(setup_ids) + 1

    # setup id is given, it exists already and overwrite is True
    # -> return this set-up id and overwrite it
    elif setup_id in setup_ids and overwrite:
        print("Setup id", setup_id, "is present already and will be over-written")
        mode = 2

    elif setup_id in setup_ids and not overwrite:
        print("Setup id", setup_id, "is present already and will be skipped")
        mode = 1

    if setup_id >= 100:
        raise ValueError("Only up to 100 set-ups are supported")
    return setup_id, mode


def copy_dataset(input_path, input_key, output_path, output_key, is_h5,
                 convert_dtype=False, chunks=None, n_threads=1, overwrite=False):

    with open_file(input_path, 'r') as f_in, open_file(output_path, 'a') as f_out:

        ds_in = f_in[input_key]
        shape = ds_in.shape

        # validate chunks
        if chunks is None:
            chunks_ = True if is_h5 else None
        else:
            chunks_ = tuple(min(ch, sh) for sh, ch in zip(shape, chunks))

        if convert_dtype:
            out_dtype = get_new_dtype(ds_in.dtype)
        else:
            out_dtype = ds_in.dtype

        if overwrite:
            del f_out[output_key]

        # create the output dataset and get the effective chunks
        ds_out = f_out.create_dataset(output_key, shape=shape, chunks=chunks_,
                                      compression='gzip', dtype=out_dtype)
        ds_chunks = ds_out.chunks

        def copy_chunk(bb):
            data = ds_in[bb]
            # skip empty chunks
            if data.sum() == 0:
                return
            if convert_dtype:
                data = convert_to_bdv_dtype(data)
            ds_out[bb] = data

        n_blocks = get_nblocks(shape, ds_chunks)
        print("Copy initial dataset from: %s:%s to %s:%s" % (input_path, input_key,
                                                             output_path, output_key))

        if n_threads > 1:
            with futures.ThreadPoolExecutor(n_threads) as tp:
                list(tqdm(tp.map(copy_chunk, blocking(shape, ds_chunks)), total=n_blocks))
        else:
            for bb in tqdm(blocking(shape, ds_chunks), total=n_blocks):
                copy_chunk(bb)


def normalize_output_path(output_path):
    # construct hdf5 output path and xml output path from output path
    base_path, ext = os.path.splitext(output_path)
    is_h5 = True
    if ext == '':
        data_path = output_path + '.h5'
        xml_path = output_path + '.xml'
    elif ext.lower() in HDF5_EXTENSIONS:
        data_path = output_path
        xml_path = base_path + '.xml'
    elif ext.lower() in XML_EXTENSIONS:
        data_path = base_path + '.h5'
        xml_path = output_path
    elif ext.lower() in N5_EXTENSIONS:
        if not HAVE_ELF:
            raise ValueError("Can only write n5 with elf.")
        data_path = output_path
        xml_path = base_path + '.xml'
        is_h5 = False
    else:
        raise ValueError("File extension %s not supported" % ext)
    return data_path, xml_path, is_h5


def make_scales(data_path, downscale_factors, downscale_mode,
                ndim, setup_id, is_h5,
                chunks=None, n_threads=1, timepoint=0, overwrite=False):
    ds_modes = ('nearest', 'mean', 'max', 'min', 'interpolate')
    if downscale_mode not in ds_modes:
        raise ValueError("Invalid downscale mode %s, choose one of %s" % downscale_mode, str(ds_modes))
    if not all(isinstance(factor, (int, tuple, list)) for factor in downscale_factors):
        raise ValueError("Invalid downscale factor")
    if not all(len(factor) == 3 for factor in downscale_factors
               if isinstance(factor, (tuple, list))):
        raise ValueError("Invalid downscale factor")
    # normalize all factors to be tuple or list
    factors = [ndim*[factor] if isinstance(factor, int) else factor
               for factor in downscale_factors]

    # run single downsampling stages
    for scale, factor in enumerate(factors):
        in_key = get_key(is_h5, timepoint=timepoint, setup_id=setup_id, scale=scale)
        out_key = get_key(is_h5, timepoint=timepoint, setup_id=setup_id, scale=scale + 1)
        print("Downsample scale %i / %i" % (scale + 1, len(factors)))
        downsample(data_path, in_key, out_key, factor, downscale_mode, n_threads,
                   overwrite=overwrite)

    # add first level to factors
    factors = [[1, 1, 1]] + factors
    return factors


def convert_to_bdv(input_path, input_key, output_path,
                   downscale_factors=None, downscale_mode='nearest',
                   resolution=[1., 1., 1.], unit='pixel',
                   setup_id=None, timepoint=0,
                   setup_name=None, affine=None, attributes={'channel': {'id': None}},
                   overwrite=False, convert_dtype=None, chunks=None, n_threads=1):
    """ Convert hdf5 volume to BigDatViewer format.

    Optionally downscale the input volume and write it
    to BigDataViewer scale pyramid.

    Arguments:
        input_path (str): path to hdf5 input volume
        input_key (str): path in hdf5 input file
        output_path (str): output path to bdv file
        downscale_factors (tuple or list): factors tused to create multi-scale pyramid.
            The factors need to be specified per dimension and are interpreted relative to the previous factor.
            If no argument is passed, pybdv does not create a multi-scale pyramid. (default: None)
        downscale_mode (str): mode used for downscaling.
            Can be 'mean', 'max', 'min', 'nearest' or 'interpolate' (default:'nerarest').
        resolution(list or tuple): resolution of the data
        unit (str): unit of measurement
        setup_id (int): id of this view set-up. By default, the next free id is chosen (default: None).
        timepoint (int): time point id to write (default: 0)
        setup_name (str): name of this view set-up (default: None)
        affine (list[float] or dict[str, list[float]]): affine view transformation(s) for this setup.
            Can either be a list for a single transformation or a dictionary for multiple transformations.
            Each transformation needs to be given in the bdv convention, i.e. using XYZ axis convention
            unlike the other parameters of pybdv, that expect ZYX axis convention. (default: None)
        attributes (dict[str, dict]): attributes associated with the view setups. Expects a dictionary
            which maps the attribute anmes to their settings (also dict).
            The setting dictionaries must contain the entry id is None.
            If this entry's value is None, it will be set to the current highest id + 1.
            (default: {'channel': {'id': None}})
        overwrite (bool): whether to over-write or skip existing setups with existing setup id and time-point.
            (default: False)
        convert_dtype (bool): convert the datatype to value range that is compatible with BigDataViewer.
            This will map unsigned types to signed and fail if the value range is too large. (default: None)
        chunks (tuple): chunks for the output dataset.
            By default the h5py auto chunks are used (default: None)
        n_threads (int): number of chunks used for copying and downscaling (default: 1)
    """
    # validate input data arguments
    if not os.path.exists(input_path):
        raise ValueError("Input file %s does not exist" % input_path)
    with open_file(input_path, 'r') as f:
        if input_key not in f:
            raise ValueError("%s not in %s" % (input_key, input_path))
        shape = f[input_key].shape
        ndim = len(shape)

    if ndim != 3 or len(resolution) != ndim:
        raise ValueError("Invalid input dimensionality")
    if affine is not None:
        validate_affine(affine)

    data_path, xml_path, is_h5 = normalize_output_path(output_path)
    setup_id, mode = handle_setup_id(setup_id, data_path, is_h5, timepoint, overwrite)
    # mode 1 -> skip
    if mode == 1:
        return
    # mode 2 -> over-write existing setup_id / timepoint
    overwrite_ = mode == 2

    # validate the attributes
    attributes_ = validate_attributes(xml_path, attributes, setup_id,
                                      overwrite=overwrite_)

    # we need to convert the dtype only for the hdf5 based storage
    if convert_dtype is None:
        convert_dtype = is_h5

    # copy the initial dataset
    base_key = get_key(is_h5, timepoint=timepoint, setup_id=setup_id, scale=0)
    copy_dataset(input_path, input_key,
                 data_path, base_key, is_h5, convert_dtype=convert_dtype,
                 chunks=chunks, n_threads=n_threads, overwrite=overwrite_)

    # downsample if needed
    if downscale_factors is None:
        # set single level downscale factor
        factors = [[1, 1, 1]]
    else:
        factors = make_scales(data_path, downscale_factors, downscale_mode,
                              ndim, setup_id, is_h5,
                              n_threads=n_threads, chunks=chunks, timepoint=timepoint,
                              overwrite=overwrite_)

    # write the format specific metadata in the output container
    if is_h5:
        write_h5_metadata(data_path, factors, setup_id, timepoint,
                          overwrite=overwrite_)
    else:
        write_n5_metadata(data_path, factors, resolution, setup_id, timepoint,
                          overwrite=overwrite_)

    # write bdv xml metadata
    write_xml_metadata(xml_path, data_path, unit,
                       resolution, is_h5,
                       setup_id=setup_id,
                       timepoint=timepoint,
                       setup_name=setup_name,
                       affine=affine,
                       attributes=attributes_,
                       overwrite=overwrite_)


def make_bdv(data, output_path,
             downscale_factors=None, downscale_mode='nearest',
             resolution=[1., 1., 1.], unit='pixel',
             setup_id=None, timepoint=0, setup_name=None,
             affine=None, attributes={'channel': {'id': None}},
             overwrite=False, convert_dtype=None, chunks=None, n_threads=1):
    """ Write data in BigDatViewer file format for one view setup and timepoint.

    Optionally downscale the input data to multi-scale image pyramid.

    Arguments:
        data (np.ndarray): input data
        output_path (str): output path to bdv file
        downscale_factors (tuple or list): factors tused to create multi-scale pyramid.
            The factors need to be specified per dimension and are interpreted relative to the previous factor.
            If no argument is passed, pybdv does not create a multi-scale pyramid. (default: None)
        downscale_mode (str): mode used for downscaling.
            Can be 'mean', 'max', 'min', 'nearest' or 'interpolate' (default:'nerarest').
        resolution(list or tuple): resolution of the data
        unit (str): unit of measurement
        setup_id (int): id of this view set-up. By default, the next free id is chosen (default: None).
        timepoint (int): time point id to write (default: 0)
        setup_name (str): name of this view set-up (default: None)
        affine (list[float] or dict[str, list[float]]): affine view transformation(s) for this setup.
            Can either be a list for a single transformation or a dictionary for multiple transformations.
            Each transformation needs to be given in the bdv convention, i.e. using XYZ axis convention
            unlike the other parameters of pybdv, that expect ZYX axis convention. (default: None)
        attributes (dict[str, dict]): attributes associated with the view setups. Expects a dictionary
            which maps the attribute anmes to their settings (also dict).
            The setting dictionaries must contain the entry id is None.
            If this entry's value is None, it will be set to the current highest id + 1.
            (default: {'channel': {'id': None}})
        overwrite (bool): whether to over-write or skip existing setups with existing setup id and time-point.
            (default: False)
        convert_dtype (bool): convert the datatype to value range that is compatible with BigDataViewer.
            This will map unsigned types to signed and fail if the value range is too large. (default: None)
        chunks (tuple): chunks for the output dataset.
            By default the h5py auto chunks are used (default: None)
        n_threads (int): number of chunks used for writing and downscaling (default: 1)
    """
    # validate input arguments
    if not isinstance(data, np.ndarray):
        raise ValueError("Input needs to be numpy array, got %s" % type(data))
    ndim = data.ndim
    if ndim != 3 or len(resolution) != ndim:
        raise ValueError("Invalid input dimensionality")
    if affine is not None:
        validate_affine(affine)

    data_path, xml_path, is_h5 = normalize_output_path(output_path)
    setup_id, mode = handle_setup_id(setup_id, data_path, is_h5, timepoint, overwrite)
    # mode 1 -> skip
    if mode == 1:
        return
    # mode 2 -> over-write existing setup_id / timepoint
    overwrite_ = mode == 2

    # validate the attributes
    attributes_ = validate_attributes(xml_path, attributes, setup_id, overwrite_)

    # we need to convert the dtype only for the hdf5 based storage
    if convert_dtype is None:
        convert_dtype = is_h5

    if convert_dtype:
        data = convert_to_bdv_dtype(data)

    # set proper chunks
    if chunks is None:
        chunks_ = True if is_h5 else None
    else:
        chunks_ = tuple(min(ch, sh) for sh, ch in zip(data.shape, chunks))

    # write initial dataset
    base_key = get_key(is_h5, timepoint=timepoint, setup_id=setup_id, scale=0)
    with open_file(data_path, 'a') as f:

        # need to remove the previous data-set if we over-write
        if overwrite_:
            del f[base_key]

        ds = f.create_dataset(base_key, shape=data.shape, compression='gzip',
                              chunks=chunks_, dtype=data.dtype)

        # if we have z5py, this will trigger multi-threaded write (otherwise no effect)
        ds.n_threads = n_threads
        ds[:] = data

    # downsample if needed
    if downscale_factors is None:
        # set single level downscale factor
        factors = [[1, 1, 1]]
    else:
        factors = make_scales(data_path, downscale_factors, downscale_mode,
                              ndim, setup_id, is_h5,
                              n_threads=n_threads, chunks=chunks, timepoint=timepoint,
                              overwrite=overwrite_)

    # write the format specific metadata in the output container
    if is_h5:
        write_h5_metadata(data_path, factors, setup_id, timepoint,
                          overwrite=overwrite_)
    else:
        write_n5_metadata(data_path, factors, resolution, setup_id, timepoint,
                          overwrite=overwrite_)

    # write bdv xml metadata
    write_xml_metadata(xml_path, data_path, unit,
                       resolution, is_h5,
                       setup_id=setup_id,
                       timepoint=timepoint,
                       setup_name=setup_name,
                       affine=affine,
                       attributes=attributes_,
                       overwrite=overwrite_)
