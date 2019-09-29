import runpy
import itertools
from setuptools import setup, find_packages

__version__ = runpy.run_path('pybdv/__version__.py')['__version__']

requires = [
    "numpy",
    "h5py",
    "scikit-image",
    "tqdm"
]

extras = {"elf": "elf"}
extras["all"] = list(itertools.chain.from_iterable(extras.values()))

setup(
    name='pybdv',
    packages=find_packages(exclude=['test']),
    version=__version__,
    description='Python tools for BigDataViewer',
    author='Constantin Pape',
    install_requires=requires,
    extras_require=extras,
    url='https://github.com/constantinpape/pybdv',
    license='MIT',
    entry_points={
        "console_scripts": ["convert_to_bdv = pybdv.scripts.pybdv_converter:main"]
    },
)
