#! /usr/bin/python

import argparse
import json
from pybdv import convert_to_bdv


def main():
    parser = argparse.ArgumentParser(description='Convert dataset from container (hdf5 or n5) to bigdataviewer format.')

    parser.add_argument('input_path', type=str,
                        help='path to the hdf5 input file')
    parser.add_argument('input_key', type=str,
                        help='path in file to input dataset')
    parser.add_argument('output_path', type=str,
                        help='path to the output file')

    parser.add_argument('--downscale_factors', type=str, default=None,
                        help='factors used for creating downscaled image pyramid, expects json encoded list')
    parser.add_argument('--downscale_mode', type=str, default='nerarest',
                        help='mode used for downscaling, can be nearest, mean, max, min or interpolate')
    parser.add_argument('--resolution', type=float, nargs=3, default=[1., 1., 1.],
                        help='resolution of the data')
    parser.add_argument('--unit', type=str, default='pixel',
                        help='unit of measurement')

    parser.add_argument('--setup_id', type=int, default=None,
                        help='id of the setup to write')
    parser.add_argument('--timepoint', type=int, default=0,
                        help='timepoint id to write')
    parser.add_argument('--setup_name', type=str, default=None,
                        help='name of the setup to write')

    parser.add_argument('--affine', type=str, default=None,
                        help='affine transformation(s) used for view registration, expects json encoded list or dict')
    default_attributes = {'channel': None}
    default_attributes = json.dumps(default_attributes)
    parser.add_argument('--attributes', type=str, default=default_attributes,
                        help='attributes for this view, expects json encoded dict')

    parser.add_argument('--overwrite', type=int, default=0,
                        help='whether to overwrite existing views')
    parser.add_argument('--chunks', type=str, default=None,
                        help='chunk settings, expects json encoded list')
    parser.add_argument('--n_threads', type=int, default=1,
                        help='number of threads used for copying and downsampling')

    args = parser.parse_args()

    # decode downscale factors
    if args.downscale_factors is None:
        downscale_factors = None
    else:
        try:
            downscale_factors = json.loads(args.downscale_factors)
        except ValueError as e:
            raise ValueError("Decoding downscale_factors as json failed with %s" % str(e))

    # decode affine
    if args.affine is None:
        affine = None
    else:
        try:
            affine = json.loads(args.affine)
        except ValueError as e:
            raise ValueError("Decoding affine as json failed with %s" % str(e))

    # decode attributes
    if args.attributes is None:
        attributes = None
    else:
        try:
            attributes = json.loads(args.attributes)
        except ValueError as e:
            raise ValueError("Decoding attributes as json failed with %s" % str(e))

    # decode chunks
    if args.chunks is None:
        chunks = None
    else:
        try:
            chunks = json.loads(args.chunks)
        except ValueError as e:
            raise ValueError("Decoding chunks as json failed with %s" % str(e))

    convert_to_bdv(args.input_path, args.input_key, args.output_path,
                   downscale_factors, args.downscale_mode,
                   resolution=args.resolution, unit=args.unit,
                   setup_id=args.setup_id, timepoint=args.timepoint, setup_name=args.setup_name,
                   affine=affine, attributes=attributes, overwrite=int(args.overwrite),
                   chunks=chunks, n_threads=args.n_threads)


if __name__ == '__main__':
    main()
