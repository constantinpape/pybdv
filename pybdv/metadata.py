import os
import numpy as np
import h5py
import xml.etree.ElementTree as ET


# pretty print xml, from:
# http://effbot.org/zone/element-lib.htm#prettyprint
def indent_xml(elem, level=0):
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent_xml(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


# TODO support multiple setups and timepoints
def write_xml_metadata(xml_path, h5_path, unit, resolution,
                       offsets=(0., 0., 0.)):
    """ Write bigdataviewer xml.

    Based on https://github.com/tlambert03/imarispy/blob/master/imarispy/bdv.py.
    """
    # channels and  time points to 1, but should support more channels
    nt, nc = 1, 1
    key = 't00000/s00/0/cells'
    with h5py.File(h5_path, 'r') as f:
        shape = f[key].shape
        dtype = f[key].dtype
    nz, ny, nx = tuple(shape)

    # format for tischis bdv extension
    bdv_dtype = 'bdv.hdf5.ulong' if np.dtype(dtype) == np.dtype('uint64') else 'bdv.hdf5'

    # write top-level data
    root = ET.Element('SpimData')
    root.set('version', '0.2')
    bp = ET.SubElement(root, 'BasePath')
    bp.set('type', 'relative')
    bp.text = '.'

    # read metadata from dict
    dz, dy, dx = resolution
    oz, oy, ox = offsets

    seqdesc = ET.SubElement(root, 'SequenceDescription')
    imgload = ET.SubElement(seqdesc, 'ImageLoader')
    imgload.set('format', bdv_dtype)
    el = ET.SubElement(imgload, 'hdf5')
    el.set('type', 'relative')
    el.text = os.path.basename(h5_path)
    viewsets = ET.SubElement(seqdesc, 'ViewSetups')
    attrs = ET.SubElement(viewsets, 'Attributes')
    attrs.set('name', 'channel')
    for c in range(nc):
        vs = ET.SubElement(viewsets, 'ViewSetup')
        ET.SubElement(vs, 'id').text = str(c)
        ET.SubElement(vs, 'name').text = 'channel {}'.format(c + 1)
        ET.SubElement(vs, 'size').text = '{} {} {}'.format(nx, ny, nz)
        vox = ET.SubElement(vs, 'voxelSize')
        ET.SubElement(vox, 'unit').text = unit
        ET.SubElement(vox, 'size').text = '{} {} {}'.format(dx, dy, dz)
        a = ET.SubElement(vs, 'attributes')
        ET.SubElement(a, 'channel').text = str(c + 1)
        chan = ET.SubElement(attrs, 'Channel')
        ET.SubElement(chan, 'id').text = str(c + 1)
        ET.SubElement(chan, 'name').text = str(c + 1)
    tpoints = ET.SubElement(seqdesc, 'Timepoints')
    tpoints.set('type', 'range')
    ET.SubElement(tpoints, 'first').text = str(0)
    ET.SubElement(tpoints, 'last').text = str(nt - 1)

    vregs = ET.SubElement(root, 'ViewRegistrations')
    for t in range(nt):
        for c in range(nc):
            vreg = ET.SubElement(vregs, 'ViewRegistration')
            vreg.set('timepoint', str(t))
            vreg.set('setup', str(c))
            vt = ET.SubElement(vreg, 'ViewTransform')
            vt.set('type', 'affine')
            ET.SubElement(vt, 'affine').text = '{} 0.0 0.0 {} 0.0 {} 0.0 {} 0.0 0.0 {} {}'.format(dx, ox, dy, oy, dz, oz)
    indent_xml(root)
    tree = ET.ElementTree(root)
    tree.write(xml_path)


# TODO support multiple setups and timepoints
def write_h5_metadata(path, scale_factors):
    effective_scale = [1, 1, 1]

    # scale factors and chunks
    scales = []
    chunks = []

    # iterate over the scales
    for scale, scale_factor in enumerate(scale_factors):
        # compute the effective scale at this level
        if isinstance(scale_factor, int):
            effective_scale = [eff * scale_factor for eff in effective_scale]
        else:
            effective_scale = [eff * sf for sf, eff in zip(scale_factor, effective_scale)]

        # get the chunk size for this level
        out_key = 't00000/s00/%i/cells' % scale
        with h5py.File(path, 'r') as f:
            # for some reason I don't understand we do not need to invert here
            chunk = f[out_key].chunks[::-1]

        scales.append(effective_scale[::-1])
        chunks.append(chunk)

    scales = np.array(scales).astype('float32')
    chunks = np.array(chunks).astype('int')
    with h5py.File(path) as f:
        f.create_dataset('s00/resolutions', data=scales)
        f.create_dataset('s00/subdivisions', data=chunks)
