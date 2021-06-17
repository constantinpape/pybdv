from functools import partial
from concurrent import futures
import numpy as np
from skimage.transform import resize
from skimage.measure import block_reduce
from tqdm import tqdm
from warnings import warn

from .util import blocking, grow_bounding_box, open_file


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


def get_downsampler(mode):
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
        warn("Downscaling with mode 'interpolate' may lead to different results depending on the chunk size")
    else:
        raise ValueError("Downsampling mode %s is not supported" % mode)
    return downsample_function


def _get_halo(halo, factor, ndim, mode):

    # case 1: halo is zero -> we don't use any halo (downstream function expects None for this case)
    if halo == 0:
        return None
    # case 2: halo is None and the mode is interpolate -> set a halo
    elif halo is None and mode == "interpolate":
        halo = 2 * factor if isinstance(factor, int) else [2 * fac for fac in factor]
    # case 3: halo is None and interpolation mode is not interpolsate
    # -> don't need a halo
    elif halo is None:
        return None

    # make halo nd if it's just a number
    if isinstance(halo, int):
        halo = [halo] * ndim
    assert isinstance(halo, (list, tuple))
    return halo


def downsample_in_memory(input_volume,
                         downscale_factors,
                         downscale_mode,
                         block_shape,
                         n_threads,
                         halo=0):
    downscaled_volumes = []
    downsample_function = get_downsampler(downscale_mode)

    def sample_chunk(bb, in_vol, out_vol, scale_factor, halo):

        bb_grown, bb_local = grow_bounding_box(bb, halo, shape)
        bb_up = tuple(slice(b.start * scale_factor, b.stop * scale_factor)
                      for b, scale_factor in zip(bb_grown, factor))
        inp = in_vol[bb_up]

        # don't sample empty blocks
        if inp.sum() == 0:
            return

        out_shape = tuple(b.stop - b.start for b in bb_grown)
        outp = downsample_function(inp, factor, out_shape)
        out_vol[bb] = outp[bb_local]

    in_vol = input_volume
    for factor in downscale_factors:
        shape = in_vol.shape

        halo = _get_halo(halo, factor, in_vol.ndim, downscale_mode)

        ds_shape = sample_shape(shape, factor)
        ds_vol = np.zeros(ds_shape, dtype=input_volume.dtype)

        sampler = partial(sample_chunk,
                          in_vol=in_vol,
                          out_vol=ds_vol,
                          scale_factor=factor,
                          halo=halo)

        with futures.ThreadPoolExecutor(n_threads) as tp:
            list(tp.map(sampler, blocking(ds_shape, block_shape)))

        downscaled_volumes.append(ds_vol)
        in_vol = ds_vol

    return downscaled_volumes


def downsample(path, in_key, out_key, factor, mode, n_threads=1, overwrite=False, halo=None):
    """ Downsample input volume.
    """

    downsample_function = get_downsampler(mode)

    with open_file(path, 'a') as f:
        ds_in = f[in_key]
        shape = ds_in.shape
        chunks = ds_in.chunks

        halo = _get_halo(halo, factor, ds_in.ndim, mode)

        sampled_shape = sample_shape(shape, factor)
        chunks = tuple(min(sh, ch) for sh, ch in zip(sampled_shape, ds_in.chunks))

        if overwrite and out_key in f:
            del f[out_key]

        ds_out = f.require_dataset(out_key, shape=sampled_shape, chunks=chunks,
                                   compression='gzip', dtype=ds_in.dtype)

        def sample_chunk(bb):

            # grow the bounding box if we have a halo
            bb_grown, bb_local = grow_bounding_box(bb, halo, shape)

            bb_up = tuple(slice(b.start * scale_factor, b.stop * scale_factor)
                          for b, scale_factor in zip(bb_grown, factor))
            inp = ds_in[bb_up]

            # don't sample empty blocks
            if inp.sum() == 0:
                return

            out_shape = tuple(b.stop - b.start for b in bb_grown)
            outp = downsample_function(inp, factor, out_shape)
            ds_out[bb] = outp[bb_local]

        blocks = list(blocking(sampled_shape, chunks))
        if n_threads > 1:
            with futures.ThreadPoolExecutor(n_threads) as tp:
                list(tqdm(tp.map(sample_chunk, blocks), total=len(blocks)))
        else:
            for bb in tqdm(blocks, total=len(blocks)):
                sample_chunk(bb)
