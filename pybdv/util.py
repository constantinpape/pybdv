from itertools import product
import numpy as np


def blocking(shape, block_shape):
    """ Generator for nd blocking.

    Args:
        shape (tuple): nd shape
        block_shape (tuple): nd block shape
    """
    assert len(shape) == len(block_shape), "Invalid number of dimensions."

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
    assert len(bb) == len(halo) == len(shape)
    bb_grown = tuple(slice(max(b.start - ha, 0), min(b.stop + ha, sh))
                     for b, ha, sh in zip(bb, halo, shape))
    bb_local = tuple(slice(b.start - bg.start, b.stop - bg.start)
                     for bg, b in zip(bb_grown, bb))
    return bb_grown, bb_local
