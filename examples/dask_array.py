import dask.array as da
from pybdv.converter import make_bdv_from_dask_array

dask_array = da.random.randint(0,1000,(256,256,256))
downscale_factors = [[2,2,2], [2,2,2]]
make_bdv_from_dask_array(data=dask_array,
                         output_path='path/to/file.n5',
                         downscale_factors=downscale_factors,
                         downscale_mode='mean',
                         resolution=[1., 1., 1.], 
                         unit='pixel',
                         setup_id=None, 
                         timepoint=0,
                         setup_name=None,
                         affine=None, 
                         attributes={'channel': {'id': None}},
                         overwrite='skip', 
                         chunks=(64,64,64), # for s0 scale
                         downsample_chunks=[(32,32,32), (16,16,16)], # for s1, s2 scales
                         )