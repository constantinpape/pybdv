import h5py
import numpy as np

from .util import blocking


def downsample(path, in_key, out_key, factor, mode):
    """ Downsample input hdf5 volume
    """
