from functools import partial
from concurrent import futures
import numpy as np
from skimage.transform import resize
from skimage.measure import block_reduce
from tqdm import tqdm

from .util import blocking, grow_bounding_box, get_nblocks, open_file


def ds_interpolate(data, scale_factor, out_shape, order):
    dtype = data.dtype
    out = resize(data, out_shape, order=order, mode='constant',
                 anti_aliasing=order > 0, preserve_range=True)
    return out.astype(dtype)


def ds_block_reduce(data, scale_factor, out_shape, function):
    out = block_reduce(data, tuple(scale_factor), function)
    # crop if necessary
    if out.shape != out_shape:
        bb = tuple(slice(0, osh) for osh in out_shape)
        out = out[bb]
    return out


def sample_shape(shape, factor, add_incomplete_blocks=False):
    if add_incomplete_blocks:
        sampled = tuple(sh // scale_factor + int((sh % scale_factor) != 0)
                        for sh, scale_factor in zip(shape, factor))
    else:
        sampled = tuple(sh // scale_factor for sh, scale_factor in zip(shape, factor))
    sampled = tuple(max(1, sh) for sh in sampled)
    return sampled


def downsample(path, in_key, out_key, factor, mode, n_threads=1, overwrite=False):
    """ Downsample input hdf5 volume
    """

    if mode == 'nearest':
        downsample_function = partial(ds_interpolate, order=0)
    elif mode == 'mean':
        downsample_function = partial(ds_block_reduce, function=np.mean)
    elif mode == 'max':
        downsample_function = partial(ds_block_reduce, function=np.max)
    elif mode == 'min':
        downsample_function = partial(ds_block_reduce, function=np.min)
    elif mode == 'interpolate':
        downsample_function = partial(ds_interpolate, order=3)
    else:
        raise ValueError("Downsampling mode %s is not supported" % mode)
    halo = factor

    with open_file(path, 'a') as f:
        ds_in = f[in_key]
        shape = ds_in.shape
        chunks = ds_in.chunks

        sampled_shape = sample_shape(shape, factor)
        chunks = tuple(min(sh, ch) for sh, ch in zip(sampled_shape, ds_in.chunks))

        if overwrite and out_key in f:
            del f[out_key]

        ds_out = f.create_dataset(out_key, shape=sampled_shape, chunks=chunks,
                                  compression='gzip', dtype=ds_in.dtype)

        def sample_chunk(bb):

            # grow the bounding box if we have a halo
            if halo is not None:
                bb_grown, bb_local = grow_bounding_box(bb, halo, shape)
            else:
                bb_grown = bb
                bb_local = np.s_[:]

            bb_up = tuple(slice(b.start * scale_factor, b.stop * scale_factor)
                          for b, scale_factor in zip(bb_grown, factor))
            inp = ds_in[bb_up]

            # don't sample empty blocks
            if inp.sum() == 0:
                return

            out_shape = tuple(b.stop - b.start for b in bb_grown)
            outp = downsample_function(inp, factor, out_shape)
            ds_out[bb] = outp[bb_local]

        n_blocks = get_nblocks(sampled_shape, chunks)
        if n_threads > 1:
            with futures.ThreadPoolExecutor(n_threads) as tp:
                list(tqdm(tp.map(sample_chunk, blocking(shape, chunks)), total=n_blocks))
        else:
            for bb in tqdm(blocking(sampled_shape, chunks), total=n_blocks):
                sample_chunk(bb)
