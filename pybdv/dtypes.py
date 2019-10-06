import numpy as np

# Mapping of integer dtypes
DTYPE_MAPPING = {np.dtype('int8'): np.dtype('int8'),
                 np.dtype('int16'): np.dtype('int16'),
                 np.dtype('int32'): np.dtype('int16'),
                 np.dtype('int64'): np.dtype('int16'),
                 np.dtype('uint8'): np.dtype('uint8'),
                 np.dtype('uint16'): np.dtype('int16'),
                 np.dtype('uint32'): np.dtype('int16'),
                 np.dtype('uint64'): np.dtype('int16')}


def get_new_dtype(dtype):
    return DTYPE_MAPPING.get(np.dtype(dtype), np.dtype(dtype))


def map_value_range(data, new_dtype):
    if data.dtype == new_dtype:
        return data

    # we need to make sure the value range is ok
    # we don't allow for negative numbers
    valid_min = 0

    if new_dtype == np.dtype('int8'):
        valid_max = np.iinfo('uint8').max
    else:
        valid_max = np.iinfo('uint16').max

    dmin, dmax = data.min(), data.max()
    if dmin < valid_min or dmax > valid_max:
        raise RuntimeError("Cannot convert value range %i:%i to bdv data-type" % (dmin, dmax))

    # TODO need to change that numpy and bdv dtype mapping actually agrees
    # we just map the numpy dtypes
    return data.astype(new_dtype)


def convert_to_bdv_dtype(data):
    """ Map the data to datatype and value range consistent with bdv datatype encoding.
    """
    dtype = data.dtype
    if np.issubdtype(dtype, np.integer):
        new_dtype = DTYPE_MAPPING[dtype]
        data = map_value_range(data, new_dtype)
    return data
