import numpy as np
import pybdv
from pybdv.bdv_datasets import BdvDataset


def on_the_fly_2d():
    """Minimal example for writing to a single timepoint and setup slice by slice.
    """
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


def on_the_fly_3d():
    """Writing sub-regions to multiple timepoints and setups.
    """
    shape = (64, 64, 64)

    scale_factors = [(2, 2, 2), (2, 2, 2)]

    n_setups = 2
    n_timepoints = 2

    path = "./data.n5"
    # we use a nested dict to store the BdvDatasets for the individual
    # setup/timepoint configurations
    datasets = {setup_id: {} for setup_id in range(n_setups)}

    for setup_id in range(n_setups):
        for tp in range(n_timepoints):
            pybdv.initialize_bdv(path, shape=shape, dtype="float32",
                                 setup_id=setup_id, timepoint=tp,
                                 downscale_factors=scale_factors, chunks=(32, 32, 32))
            datasets[setup_id][tp] = BdvDataset(path, setup_id=setup_id, timepoint=tp)

    # write sub-region to setup 0, timepoint 0
    datasets[0][0][12:20, 32:64, 3:10] = np.random.rand(8, 32, 7)

    # write sub-region to setup 1, timepoint 0
    datasets[1][0][17:33, 0:32, 5:17] = np.random.rand(16, 32, 12)

    # write sub-region to setup 1, timepoint 1
    datasets[1][1][15:45, 32:48, 11:19] = np.random.rand(30, 16, 8)


if __name__ == '__main__':
    # on_the_fly_2d()
    on_the_fly_3d()
