import os
from itertools import product

# check if we have elf and use it's open_file implementation
try:
    from elf.io import open_file
except ImportError:
    open_file = None


HDF5_EXTENSIONS = ['.h5', '.hdf', '.hdf5']
XML_EXTENSIONS = ['.xml']
N5_EXTENSIONS = ['.n5']


# if we don't have elf, define a simplified open_file function
if open_file is None:
    import h5py

    try:
        import z5py
    except ImportError:
        z5py = None

    def open_file(path, mode='r'):
        ext = os.path.splitext(path)[1].lower()
        if ext in HDF5_EXTENSIONS:
            return h5py.File(path, mode=mode)
        elif ext in N5_EXTENSIONS:
            if z5py is None:
                raise ValueError("Need z5py to open n5 files")
            return z5py.File(path, mode=mode)
        else:
            raise ValueError(f"Invalid extension: {ext}")


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

    Argumentss:
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


def absolute_to_relative_scale_factors(scale_factors):
    """ Convert absolute to relative scale factors.

    Arguments:
        scale_factors (list[list[int]]): absolute scale factors
    """
    rel_scale_factors = [scale_factors[0]]
    for scale_factor, prev_scale_factor in zip(scale_factors[1:], scale_factors[:-1]):
        rel_scale_factor = [sf // prev_sf for sf, prev_sf in zip(scale_factor,
                                                                 prev_scale_factor)]
        rel_scale_factors.append(rel_scale_factor)

    return rel_scale_factors


def relative_to_absolute_scale_factors(scale_factors):
    """ Convert relative to absolure scale factors.

    Arguments:
        scale_factors (list[int]): relative scale factors
    """
    abs_scale_factors = [scale_factors[0]]
    for scale_factor in scale_factors[1:]:
        abs_scale_factor = [abs_sf * sf for abs_sf, sf in zip(abs_scale_factors[-1],
                                                              scale_factor)]
        abs_scale_factors.append(abs_scale_factor)

    return abs_scale_factors


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
