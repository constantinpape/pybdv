import xml.etree.ElementTree as ET
import h5py


def convert_to_bdv(input_path, input_key, output_path,
                   downscale_factors=None, downscaling_mode='nearest',
                   resolution=[1., 1., 1.], unit='pixel'):
    """ Convert hdf5 volume to BigDatViewer format.

    Optionally downscale the input volume and write it
    to BigDataViewer scale pyramid.

    Args:
        input_path (str): path to hdf5 input volume
        input_key (str): path in hdf5 input file
        output_path (str): output path to bdv file
        downscale_factors (tuple or list): tuple or list of downscaling
            factors. Can be anisotropic. No downscaling by default.
        downscaling_mode (str): mode used for downscaling.
            Either 'nearest' or 'interpolate'.
        resolution(list or tuple): resolution of the data
        unit (str): unit of measurement
    """
    # TODO validate all arguments
    assert os.path.exists(input_path), input_path
