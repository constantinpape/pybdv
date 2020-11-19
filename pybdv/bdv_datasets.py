
import os
import numpy as np
from .util import open_file, get_scale_factors, get_key, HDF5_EXTENSIONS
from .downsample import downsample_in_memory


def _check_for_out_of_bounds(position, volume, full_shape, verbose=False):

    position = np.array(position)
    full_shape = np.array(full_shape)

    vol_shape = np.array(volume.shape)
    if position.min() < 0 or (position + vol_shape - full_shape).max() > 0:

        print(f'position = {position}')

        too_large = (position + vol_shape - full_shape) > 0
        source_ends = vol_shape
        source_ends[too_large] = (full_shape - position)[too_large]

        source_starts = np.zeros((3,), dtype=int)
        source_starts[position < 0] = -position[position < 0]
        position[position < 0] = 0

        if verbose:
            print(f'source_starts = {source_starts}')
            print(f'source_ends = {source_ends}')
            print(f'position = {position}')

        volume = volume[
            source_starts[0]: source_ends[0],
            source_starts[1]: source_ends[1],
            source_starts[2]: source_ends[2]
        ]

    return position, volume


def _check_shape_and_position_scaling(max_scale, position, volume,
                                      data_path, key,
                                      verbose=False):
    vol_shape = np.array(volume.shape)
    if ((position / max_scale) - (position / max_scale).astype(int)).max()\
            or ((vol_shape / max_scale) - (vol_shape / max_scale).astype(int)).max():

        # They don't scale properly:
        # So, we have to read a volume from the target data (largest scale) that does and covers the area of where the
        # volume belongs, which we here call target_vol with respective properties target_pos and target_shape

        if verbose:
            print('----------------------')
            print(f'max_scale = {max_scale}')
            print(f'position = {position}')
            print(f'vol_shape = {vol_shape}')
            print('----------------------')
        target_pos = max_scale * (position / max_scale).astype(int)
        target_shape = max_scale * np.ceil((position + vol_shape) / max_scale).astype(int) - target_pos
        if verbose:
            print(f'target_pos = {target_pos}')
            print(f'target_shape = {target_shape}')
        with open_file(data_path, mode='r') as f:
            target_vol = f[key][
                target_pos[0]: target_pos[0] + target_shape[0],
                target_pos[1]: target_pos[1] + target_shape[1],
                target_pos[2]: target_pos[2] + target_shape[2]
            ]
        if verbose:
            print(f'target_vol.shape = {target_vol.shape}')

        # Now we have to put the volume to this target_vol at the proper position
        in_target_pos = position - target_pos
        target_vol[
            in_target_pos[0]: in_target_pos[0] + volume.shape[0],
            in_target_pos[1]: in_target_pos[1] + volume.shape[1],
            in_target_pos[2]: in_target_pos[2] + volume.shape[2]
        ] = volume

    else:

        # Everything scales nicely, so we just have to take care that the proper variables exist
        target_vol = volume
        target_pos = position
        target_shape = vol_shape

    return target_vol, target_pos, target_shape


def _scale_and_add_to_dataset(
        data_path, setup_id, timepoint,
        target_pos, target_vol, target_shape,
        scales, downscale_mode, n_threads):

    is_h5 = os.path.splitext(data_path)[1] in HDF5_EXTENSIONS

    scales = np.array(scales).astype(int)
    scale_factors = scales[1: 2].tolist()

    for scale in scales[2:]:
        scale_factors.append((scale / np.product(scale_factors, axis=0)).astype(int).tolist())

    # Scale the data
    downscaled_vols = [target_vol]
    downscaled_vols.extend(
        downsample_in_memory(
            target_vol,
            downscale_factors=scale_factors,
            downscale_mode=downscale_mode,
            block_shape=(64, 64, 64),
            n_threads=n_threads
        )
    )

    # Now, we just need to put it to the proper positions in the bdv file
    for scale_id, scale in enumerate(scales):

        # Position in the current scale
        pos_in_scale = (target_pos / scale).astype(int)
        shp_in_scale = (target_shape / scale).astype(int)

        scaled_vol = downscaled_vols[scale_id]

        with open_file(data_path, mode='a') as f:
            key = get_key(is_h5, timepoint=timepoint, setup_id=setup_id, scale=scale_id)
            f[key][
                pos_in_scale[0]: pos_in_scale[0] + shp_in_scale[0],
                pos_in_scale[1]: pos_in_scale[1] + shp_in_scale[1],
                pos_in_scale[2]: pos_in_scale[2] + shp_in_scale[2]
            ] = scaled_vol


class BdvDataset:
    """
    The basic BDV dataset to which volumes can be written using numpy nomenclature.

    The data is included into each of the down-sampling layers accordingly
    The full resolution area is padded, if necessary, to avoid sub-pixel locations in the down-sampling layers
    """

    def __init__(self, path, timepoint, setup_id, downscale_mode='mean', n_threads=1, verbose=False):

        self._path = path
        self._timepoint = timepoint
        self._setup_id = setup_id
        self._downscale_mode = downscale_mode
        self._n_threads = n_threads
        self._verbose = verbose

        # Check if it is h5 or n5
        self._is_h5 = os.path.splitext(path)[1] in HDF5_EXTENSIONS

        # Get the scales
        self._scales = np.array(get_scale_factors(self._path, self._setup_id)).astype(int)

        # Determine full dataset shape
        with open_file(self._path, mode='r') as f:
            self._key = get_key(self._is_h5, timepoint=timepoint, setup_id=setup_id, scale=0)
            self._full_shape = f[self._key].shape

    def _add_to_volume(self, position, volume):

        if self._verbose:
            print(f'scales = {self._scales}')
            print(f'full_shape = {self._full_shape}')

        # Check for out of bounds (and fix it if not)
        position, volume = _check_for_out_of_bounds(position, volume, self._full_shape, verbose=self._verbose)

        # Check if volume and position properly scale to the final scale level (and fix it if not)
        max_scale = self._scales[-1]
        target_vol, target_pos, target_shape = _check_shape_and_position_scaling(
            max_scale, position, volume,
            self._path, self._key,
            verbose=self._verbose)

        # Scale volume and write to target dataset
        _scale_and_add_to_dataset(self._path, self._setup_id, self._timepoint,
                                  target_pos, target_vol, target_shape,
                                  self._scales, self._downscale_mode,
                                  self._n_threads)

    def __setitem__(self, key, value):

        # We are assuming the index to be relative to scale 0 (full resolution)

        position = [k.start for k in key]
        shp = [k.stop - k.start for k in key]
        assert list(value.shape) == shp, f'Shape of array = {value.shape} does not match target shape = {shp}'

        self._add_to_volume(position, value)


# TODO Implement this one that includes stitching operations
class BdvDatasetWithStitching(BdvDataset):

    def __init__(self, path, timepoint, setup_id, downscale_mode='interpolate', n_threads=1, halo=None, verbose=False):

        self._halo = halo

        super().__init__(path, timepoint, setup_id, downscale_mode=downscale_mode, n_threads=n_threads, verbose=verbose)

    def set_halo(self, halo):
        """
        Adjust the halo any time you want
        """
        self._halo = halo

    def __setitem__(self, key, value):

        # TODO Do the stitching and stuff here

        # Now call the super with the properly stitched volume
        super().__setitem__(key, value)
