import os
from itertools import product
import numpy as np

try:
    from elf.io import open_file
    HAVE_ELF = True
except ImportError:
    import h5py
    HAVE_ELF = False

    # only supprt h5py if we don't have elf
    def open_file(input_path, mode='a'):
        return h5py.File(input_path, mode)


HDF5_EXTENSIONS = ['.h5', '.hdf', '.hdf5']
XML_EXTENSIONS = ['.xml']
N5_EXTENSIONS = ['.n5']


def get_key(is_h5, timepoint=None, setup_id=None, scale=None):
    sequence = []
    if is_h5:
        if timepoint is not None:
            sequence.append('t%05i' % timepoint)
        if setup_id is not None:
            sequence.append('s%02i' % setup_id)
        if scale is not None:
            sequence.append('%i/cells' % scale)
    else:
        if setup_id is not None:
            sequence.append('setup%i' % setup_id)
        if timepoint is not None:
            sequence.append('timepoint%i' % timepoint)
        if scale is not None:
            sequence.append('s%i' % scale)
    return '/'.join(sequence)


def blocking(shape, block_shape):
    """ Generator for nd blocking.

    Args:
        shape (tuple): nd shape
        block_shape (tuple): nd block shape
    """
    if len(shape) != len(block_shape):
        raise ValueError("Invalid number of dimensions.")

    # compute the ranges for the full shape
    ranges = [range(sha // bsha if sha % bsha == 0 else sha // bsha + 1)
              for sha, bsha in zip(shape, block_shape)]
    min_coords = [0] * len(shape)
    max_coords = shape

    start_points = product(*ranges)
    for start_point in start_points:
        positions = [sp * bshape for sp, bshape in zip(start_point, block_shape)]
        yield tuple(slice(max(pos, minc), min(pos + bsha, maxc))
                    for pos, bsha, minc, maxc in zip(positions, block_shape,
                                                     min_coords, max_coords))


def get_nblocks(shape, block_shape):
    n_blocks = [sh // bs + int((sh % bs) != 0)
                for sh, bs in zip(shape, block_shape)]
    return np.prod(n_blocks)


def grow_bounding_box(bb, halo, shape):
    if not (len(bb) == len(halo) == len(shape)):
        raise ValueError("Invalid number of dimensions.")
    bb_grown = tuple(slice(max(b.start - ha, 0), min(b.stop + ha, sh))
                     for b, ha, sh in zip(bb, halo, shape))
    bb_local = tuple(slice(b.start - bg.start, b.stop - bg.start)
                     for bg, b in zip(bb_grown, bb))
    return bb_grown, bb_local


def get_number_of_scales(path, timepoint, setup_id):
    ext = os.path.splitext(path)[1]
    is_h5 = ext in HDF5_EXTENSIONS
    key = get_key(is_h5, timepoint, setup_id)
    with open_file(path, 'r') as f:
        n_scales = len(f[key])
    return n_scales


def get_scale_factors(path, setup_id):
    ext = os.path.splitext(path)[1]
    is_h5 = ext in HDF5_EXTENSIONS

    with open_file(path, 'r') as f:
        if is_h5:
            key = 's%02i/resolutions' % setup_id
            ds = f[key]
            scale_factors = ds[:].tolist()
        else:
            key = get_key(is_h5, timepoint=None, setup_id=setup_id)
            scale_factors = f[key].attrs['downsamplingFactors']

    # need to switch from XYZ to ZYX
    scale_factors = [sc[::-1] for sc in scale_factors]
    return scale_factors
