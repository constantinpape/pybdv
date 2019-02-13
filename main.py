import argparse
import json
from pybdv import convert_to_bdv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str)
    parser.add_argument('input_key', type=str)
    parser.add_argument('output_path', type=str)
    parser.add_argument('--downscale_factors', type=str, default=None)
    parser.add_argument('--downscale_mode', type=str, default='nerarest')
    parser.add_argument('--resolution', type=float, nargs=3, default=[1., 1., 1.])
    parser.add_argument('--unit', type=str, default='pixel')

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
