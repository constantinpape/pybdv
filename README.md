[![Build Status](https://github.com/constantinpape/pybdv/workflows/build/badge.svg)](https://github.com/constantinpape/pybdv/actions)
[![Conda Forge](https://img.shields.io/conda/vn/conda-forge/pybdv.svg)](https://anaconda.org/conda-forge/pybdv)

# pyBDV

Python tools for [BigDataViewer](https://imagej.net/BigDataViewer).


## Installation

You can install the package from source
```
python setup.py install
```
or via conda:
```
conda install -c conda-forge pybdv
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


### Conversion to n5-bdv format

Bigdatviewer core also supports an [n5 based data format](https://github.com/bigdataviewer/bigdataviewer-core/blob/master/BDV%20N5%20format.md). The data can be converted to this format by passing a path with n5 ending as output path: `/path/to/out.n5`. In order to support this, you need to install [z5py](https://github.com/constantinpape/z5).


### Advanced IO options

If [elf](https://github.com/constantinpape/elf) is available, additional file input formats are supported.
For example, it is possible to convert inputs from tif slices

```python
import os
import imageio
import numpy as np
from pybdv import convert_to_bdv


input_path = './slices'
os.makedirs(input_path, exist_ok=True)
n_slices = 25
shape = (256, 256)

for slice_id in range(n_slices):
    imageio.imsave('./slices/im%03i.tif', np.random.randint(0, 255, size=shape, dtype='uint8'))

input_key = '*.tif'
output_path = 'from_slices.h5'
convert_to_bdv(input_path, input_key, output_path)
```

or tif stacks:

```python
import imageio
import numpy as np
from pybdv import convert_to_bdv


input_path = './stack.tif'
shape = (25, 256, 256)

imageio.volsave(input_path, np.random.randint(0, 255, size=shape, dtype='uint8'))

input_key = ''
output_path = 'from_stack.h5'
convert_to_bdv(input_path, input_key, output_path)
```
