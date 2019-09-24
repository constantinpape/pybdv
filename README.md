# PyBdv

Python tools for [BigDataViewer](https://imagej.net/BigDataViewer).


## Installation

You can install the package from source
```
python setup.py install
```
or via conda:
```
conda install -c conda-forge -c cpape pybdv
```


## Usage

### Python

Write out numpy array `volume` to bdv format:
```python
from pybdv import make_bdv

out_path = '/path/to/out'

# the scale factors determine the levels of the multi-scale pyramid
# that will be created by pybdv.
# the downscaling factors are interpreted relative to the previous factor
# (rather than absolute) and the zeroth scale level (corresponding to [1, 1, 1])
# is implicit, i.e. DON'T specify it
scale_factors = [[2, 2, 2], [2, 2, 2], [4, 4, 4]]

# the downscale mode determines the method for downscaling:
# - interpolate: cubic interpolation
# - max: downscale by maximum
# - mean: downscale by averaging
# - min: downscale by minimum
# - nearest: nearest neighbor downscaling
mode = 'mean'

# specify a resolution of 0.5 micron per pixel (for zeroth scale level)
make_bdv(volume, out_path,
         downscale_factors=scale_factors, downscale_mode=mode,
         resolution=[0.5, 0.5, 0.5], unit='micrometer')
```

Convert hdf5 dataset to bdv format:
```python
from pybdv import convert_to_bdv

in_path = '/path/to/in.h5'
in_key = 'data'
out_path = '/path/to/out'

# keyword arguments are same as for 'make_bdv'
convert_to_bdv(in_path, in_key, out_path,
               resolution=[0.5, 0.5, 0.5], unit='micrometer')
```

### Command line

You can also call `convert_to_bdv` via the command line:
```bash
convert_to_bdv /path/to/in.h5 data /path/to/out --downscale_factors "[[2, 2, 2], [2, 2, 2], [4, 4, 4]]" --downscale_mode nearest --resolution 0.5 0.5 0.5 --unit micrometer
```

The downscale factors need to be encoded as json list.
