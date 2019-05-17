import h5py
import numpy as np

from .util import blocking


def ds_nearest():
    pass


def ds_mean():
    pass


def downsample(path, in_key, out_key, factor, mode):
    """ Downsample input hdf5 volume
    """

    # TODO set halo for mean interpolation ?
    if mode == 'nearest':
        downsample_function = ds_nearest
    elif mode == 'mean':
        downsample_function = ds_mean
    else:
        raise ValueError("Downsampling mode %s is not supported" % mode)

    with h5py.File(path) as f:
        ds_in = f[in_key]
        shape = ds_in.shape
        chunks = ds_in.chunks

        sampled_shape = tuple(sh // scale_factor for scale_factor in factor)
        chunks = tuple(min(sh, ch) for sh, ch in zip(sampled_shape, ds_in.chunks))

        ds_out = f.create_dataset(out_key, shape=sampled_shape, chunks=chunks,
                                  compression='gzip', dtype=ds_in.dtype)

        def sample_chunk(bb):
            bb_up = tuple(slice(b.start * scale_factor, b.stop * scale_factor)
                          for b, scale_factor in zip(bb, factor))
            inp = ds_in[bb_up]

            # don't sample empty blocks
            if inp.sum() == 0:
                return

            out_shape = tuple(b.stop - b.start for b in bb)
            outp = downsample_function(inp, scale_factor, out_shape)
            ds_out[bb] = outp

        # TODO use tdqm to measure progress
        for bb in blocking(sampled_shape, chunks):
            sample_chunk(bb)
