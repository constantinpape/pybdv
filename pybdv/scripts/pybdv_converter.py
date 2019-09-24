#! /usr/bin/python

import argparse
import json
from pybdv import convert_to_bdv


def main():
    parser = argparse.ArgumentParser(description='Convert hdf5 dataset to bigdataviewer format.')
    parser.add_argument('input_path', type=str,
                        help='path to the hdf5 input file')
    parser.add_argument('input_key', type=str,
                        help='path in file to input dataset')
    parser.add_argument('output_path', type=str,
                        help='path to the output file')
    parser.add_argument('--downscale_factors', type=str, default=None,
                        help='factors used for creating downscaled image pyramid, needs to be encoded as json list')
    parser.add_argument('--downscale_mode', type=str, default='nerarest',
                        help='mode used for downscaling, can be nearest, mean, max, min or interpolate')
    parser.add_argument('--resolution', type=float, nargs=3, default=[1., 1., 1.],
                        help='resolution of the data')
    parser.add_argument('--unit', type=str, default='pixel',
                        help='unit of measurement')
    parser.add_argument('--setup_id', type=int, default=None,
                        help='id of the setup to write')
    parser.add_argument('--setup_name', type=str, default=None,
                        help='name of the setup to write')

    args = parser.parse_args()

    # decode downscale factors
    if args.downscale_factors is None:
        downscale_factors = None
    else:
        try:
            downscale_factors = json.loads(args.downscale_factors)
        except ValueError as e:
            raise ValueError("Decoding downscale_factors as json failed with %s" % str(e))

    convert_to_bdv(args.input_path, args.input_key, args.output_path,
                   downscale_factors, args.downscale_mode,
                   resolution=args.resolution, unit=args.unit,
                   setup_id=args.setup_id, setup_name=args.setup_name)


if __name__ == '__main__':
    main()
