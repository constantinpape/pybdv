#! /usr/bin/python

import argparse
import json
from pybdv import convert_to_bdv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str,
                        help='path to the hdf5 input file')
    parser.add_argument('input_key', type=str,
                        help='path in file to input dataset')
    parser.add_argument('output_path', type=str,
                        help='path to the output file')
    parser.add_argument('--downscale_factors', type=str, default=None,
                        help='factors used for creating downscaled image pyramid')
    parser.add_argument('--downscale_mode', type=str, default='nerarest',
                        help='mode used for downscaling (can be nearest or mean)')
    parser.add_argument('--resolution', type=float, nargs=3, default=[1., 1., 1.],
                        help='resolution of the data')
    parser.add_argument('--unit', type=str, default='pixel',
                        help='unit of measurement')

    args = parser.parse_args()

    # decode downscale factors
    if args.downscale_factors is None:
        downscale_factors = None
    else:
        try:
            downscale_factors = json.loads(args.downscale_factors)
        except ValueError as e:
            raise ValueError("Decoding downscale_factors as json failed with %s" % str(e))

    convert_to_bdv(args.input_path, args.input_key,
                   args.output_path, downscale_factors,
                   args.downscale_mode, args.resolution,
                   args.unit)


if __name__ == '__main__':
    main()
