from distutils.core import setup
from pybdv import __version__


setup(name='pybdv',
      version=__version__,
      description='python tools for BigDataViewer',
      author='Constantin Pape',
      packages=['pybdv'],
      scripts=['pybdv_converter.py'],
      install_requires=['numpy', 'h5py', 'scikit-image', 'tqdm'],
      include_package_dat=True)
