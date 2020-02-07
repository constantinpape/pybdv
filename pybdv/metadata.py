import os
import numpy as np
import xml.etree.ElementTree as ET
from .util import open_file, get_key


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


#
# functions to write the metadata
#


# TODO support multiple timepoints and different types of views
# (right now, we only support multiple channels)
def write_xml_metadata(xml_path, h5_path, unit, resolution, is_h5,
                       offsets=(0., 0., 0.), setup_id=0,
                       setup_name=None):
    """ Write bigdataviewer xml.

    Based on https://github.com/tlambert03/imarispy/blob/master/imarispy/bdv.py.
    """
    # number of timepoints hard-coded to 1
    nt = 1
    setup_name = 'Setup%i' % setup_id if setup_name is None else setup_name
    key = get_key(is_h5, time_point=0, setup_id=setup_id, scale=0)
    with open_file(h5_path, 'r') as f:
        shape = f[key].shape
    nz, ny, nx = tuple(shape)

    format_type = 'hdf5' if is_h5 else 'n5'

    # check if we have an xml already
    if os.path.exists(xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # load the sequence description
        seqdesc = root.find('SequenceDescription')
        # TODO validate image loader

        # load the view descriptions
        viewsets = seqdesc.find('ViewSetups')

        # NOTE we only suport channel for now
        # load the attributes setup
        attrsets = viewsets.find('Attributes')

        # load the registration decriptions
        vregs = root.find('ViewRegistrations')
    else:
        # write top-level data
        root = ET.Element('SpimData')
        root.set('version', '0.2')
        bp = ET.SubElement(root, 'BasePath')
        bp.set('type', 'relative')
        bp.text = '.'

        # make the sequence description element
        seqdesc = ET.SubElement(root, 'SequenceDescription')
        # make the image loader
        imgload = ET.SubElement(seqdesc, 'ImageLoader')
        bdv_dtype = 'bdv.%s' % format_type
        imgload.set('format', bdv_dtype)
        el = ET.SubElement(imgload, format_type)
        el.set('type', 'relative')
        el.text = os.path.basename(h5_path)

        # make the view descriptions
        viewsets = ET.SubElement(seqdesc, 'ViewSetups')
        attrsets = ET.SubElement(viewsets, 'Attributes')
        attrsets.set('name', 'channel')

        # make the registration decriptions
        vregs = ET.SubElement(root, 'ViewRegistrations')

        # timepoint description
        tpoints = ET.SubElement(seqdesc, 'Timepoints')
        tpoints.set('type', 'range')
        ET.SubElement(tpoints, 'first').text = str(0)
        ET.SubElement(tpoints, 'last').text = str(nt - 1)

    # parse the resolution and offsets
    dz, dy, dx = resolution
    oz, oy, ox = offsets

    # setup for this view
    vs = ET.SubElement(viewsets, 'ViewSetup')
    # id, name and size
    ET.SubElement(vs, 'id').text = str(setup_id)
    ET.SubElement(vs, 'name').text = setup_name
    ET.SubElement(vs, 'size').text = '{} {} {}'.format(nx, ny, nz)
    vox = ET.SubElement(vs, 'voxelSize')
    ET.SubElement(vox, 'unit').text = unit
    ET.SubElement(vox, 'size').text = '{} {} {}'.format(dx, dy, dz)
    # attributes for this view setup.
    attrs = ET.SubElement(vs, 'attributes')
    ET.SubElement(attrs, 'channel').text = str(setup_id)

    # add channel attribute
    chan = ET.SubElement(attrsets, 'Channel')
    ET.SubElement(chan, 'id').text = str(setup_id)
    ET.SubElement(chan, 'name').text = str(setup_id)

    # TODO support different registrations here
    for t in range(nt):
        vreg = ET.SubElement(vregs, 'ViewRegistration')
        vreg.set('timepoint', str(t))
        vreg.set('setup', str(setup_id))
        vt = ET.SubElement(vreg, 'ViewTransform')
        vt.set('type', 'affine')
        ET.SubElement(vt, 'affine').text = '{} 0.0 0.0 {} 0.0 {} 0.0 {} 0.0 0.0 {} {}'.format(dx, ox,
                                                                                              dy, oy,
                                                                                              dz, oz)
    indent_xml(root)
    tree = ET.ElementTree(root)
    tree.write(xml_path)


# TODO support multiple timepoints
def write_h5_metadata(path, scale_factors, setup_id=0):
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
        out_key = get_key(True, time_point=0, setup_id=setup_id, scale=scale)
        with open_file(path, 'r') as f:
            # for some reason I don't understand we do not need to invert here
            chunk = f[out_key].chunks[::-1]

        scales.append(effective_scale[::-1])
        chunks.append(chunk)

    scales = np.array(scales).astype('float32')
    chunks = np.array(chunks).astype('int')
    with open_file(path, 'a') as f:
        f.create_dataset('s%02i/resolutions' % setup_id, data=scales)
        f.create_dataset('s%02i/subdivisions' % setup_id, data=chunks)


# n5 metadata format is specified here:
# https://github.com/bigdataviewer/bigdataviewer-core/blob/master/BDV%20N5%20format.md
def write_n5_metadata(path, scale_factors, resolution, setup_id=0):
    # build the effective scale factors
    effective_scales = [scale_factors[0]]
    for factor in scale_factors[1:]:
        effective_scales.append([eff * fac
                                 for eff, fac in zip(effective_scales[-1], factor[::-1])])

    with open_file(path) as f:
        key = get_key(False, time_point=0, setup_id=setup_id, scale=0)
        dtype = str(f[key].dtype)

        root_key = get_key(False, setup_id=setup_id)
        root = f[root_key]
        root.attrs['downsamplingFactors'] = effective_scales
        root.attrs['dataType'] = dtype

        group_key = get_key(False, time_point=0, setup_id=setup_id)
        g = f[group_key]
        g.attrs['multiScale'] = True
        g.attrs['resolution'] = resolution[::-1]

        effective_scale = [1, 1, 1]
        for scale_id, factor in enumerate(effective_scales):
            ds = g['s%i' % scale_id]
            effective_scale = [eff * sf for eff, sf in zip(effective_scale, factor)]
            ds.attrs['downsamplingFactors'] = factor


#
# helper functions to read attributes from the xml metadata
#


def get_bdv_format(xml_path):
    root = ET.parse(xml_path).getroot()
    seqdesc = root.find('SequenceDescription')
    imgload = seqdesc.find('ImageLoader')
    return imgload.attrib['format']


def get_resolution(xml_path, setup_id):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    seqdesc = root.find('SequenceDescription')
    viewsets = seqdesc.find('ViewSetups')
    vsetups = viewsets.findall('ViewSetup')
    for vs in vsetups:
        if vs.find('id').text == str(setup_id):
            vox = vs.find('voxelSize')
            resolution = vox.find('size').text
            return [float(res) for res in resolution.split()][::-1]
    raise RuntimeError("Could not find setup %s" % str(setup_id))


def get_data_path(xml_path, return_absolute_path=False):
    et = ET.parse(xml_path).getroot()
    et = et.find('SequenceDescription')
    et = et.find('ImageLoader')
    node = et.find('hdf5')
    if node is None:
        node = et.find('n5')
    if node is None:
        raise RuntimeError("Could not find valid data path in xml.")
    path = node.text
    # this assumes relative path in xml
    if return_absolute_path:
        path = os.path.join(os.path.split(xml_path)[0], path)
        path = os.path.abspath(os.path.relpath(path))
    return path
