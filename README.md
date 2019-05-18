# PyBdv

Python tools for [BigDataViewer](https://imagej.net/BigDataViewer).


## Installation

You can install the package via python:
```
python setup.py install
```


## Usage

If you install the package, you can call the application `pybdv_converter.py` from
the command line to convert an input hdf5 file to the bigdataviewer format.
You can also use the library from python. It exposes two functions:
- `convert_to_bdv`, which makes a bdv file from an existing hd5 dataset
- `make_bdv`, which makes a bdv file from a numpy array
