import numpy as np
import pybdv
from pybdv.bdv_datasets import BdvDataset


def on_the_fly_2d():
    x = np.random.rand(64, 128, 128)

    scale_factors = [
        (1, 2, 2), (1, 2, 2)
    ]

    path = "./data.n5"
    pybdv.initialize_bdv(path, shape=x.shape, dtype=x.dtype,
                         downscale_factors=scale_factors, chunks=(1, 64, 64))
    ds = BdvDataset(path, setup_id=0, timepoint=0)

    for z in range(x.shape[0]):
        # TODO support better slicing
        ds[z:z+1, 0:128, 0:128] = x[z:z+1, 0:128, 0:128]


# TODO
def on_the_fly_3d():
    pass


if __name__ == '__main__':
    on_the_fly_2d()
