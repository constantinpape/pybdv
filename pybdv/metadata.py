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


# TODO types of views (right now, we only support multiple channels)
def write_xml_metadata(xml_path, data_path, unit, resolution, is_h5,
                       setup_id=0, timepoint=0,
                       setup_name=None, affine=None):
    """ Write bigdataviewer xml.

    Based on https://github.com/tlambert03/imarispy/blob/master/imarispy/bdv.py.
    Arguments:
        xml_path (str): path to xml meta data
        data_path (str): path to the data (in h5 or n5 format)
        unit (str): physical unit of the data
        resolution (str): resolution / voxel size of the data at the original scale
        is_h5 (bool): is the data in h5 or n5 format
        setup_id (int): id of the set-up (default: None)
        timepoint (int): id of the time-point (default: None)
        setup_name (str): name of this set-up (default: None)
        affine (list[int] or dict[list[int]]): affine transformations for the view set-up (default: None)
    """
    # number of timepoints hard-coded to 1
    setup_name = 'Setup%i' % setup_id if setup_name is None else setup_name
    key = get_key(is_h5, timepoint=timepoint, setup_id=setup_id, scale=0)
    with open_file(data_path, 'r') as f:
        shape = f[key].shape
    nz, ny, nx = tuple(shape)

    format_type = 'hdf5' if is_h5 else 'n5'

    # check if we have xml with metadata already
    # -> yes we do
    if os.path.exists(xml_path):
        # parse the metadata from xml
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # load the sequence description
        seqdesc = root.find('SequenceDescription')

        # load the view descriptions
        viewsets = seqdesc.find('ViewSetups')

        # NOTE we only suport channel for now
        # load the attributes setup
        attrsets = viewsets.find('Attributes')

        # load the registration decriptions
        vregs = root.find('ViewRegistrations')

        # update the timepoint descriptions
        tpoints = seqdesc.find('Timepoints')
        first = tpoints.find('first')
        first.text = str(min(int(first.text), timepoint))
        last = tpoints.find('last')
        last.text = str(max(int(last.text), timepoint))

    # -> no we don't have an xml
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
        el.text = os.path.basename(data_path)

        # make the view descriptions
        viewsets = ET.SubElement(seqdesc, 'ViewSetups')
        attrsets = ET.SubElement(viewsets, 'Attributes')
        attrsets.set('name', 'channel')

        # make the registration decriptions
        vregs = ET.SubElement(root, 'ViewRegistrations')

        # timepoint description
        tpoints = ET.SubElement(seqdesc, 'Timepoints')
        tpoints.set('type', 'range')
        ET.SubElement(tpoints, 'first').text = str(timepoint)
        ET.SubElement(tpoints, 'last').text = str(timepoint)

    # parse the resolution and offsets
    dz, dy, dx = resolution

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

    if affine is None:
        _write_default_affine(vregs, setup_id, timepoint, dx, dy, dz)
    else:
        _write_affine(vregs, setup_id, timepoint, affine)
    indent_xml(root)
    tree = ET.ElementTree(root)
    tree.write(xml_path)


def write_h5_metadata(path, scale_factors, setup_id=0, timepoint=0):
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
        out_key = get_key(True, timepoint=timepoint, setup_id=setup_id, scale=scale)
        with open_file(path, 'r') as f:
            # for some reason I don't understand we do not need to invert here
            chunk = f[out_key].chunks[::-1]

        scales.append(effective_scale[::-1])
        chunks.append(chunk)

    scales = np.array(scales).astype('float32')
    chunks = np.array(chunks).astype('int')
    with open_file(path, 'a') as f:
        # write the resolution metadata for this set-up,
        # or if if we have this set-up already make sure
        # that the metadata is consistent
        key_res = 's%02i/resolutions' % setup_id
        if key_res in f:
            scales_expected = f[key_res][:]
            if not np.array_equal(scales_expected, scales):
                raise RuntimeError("Metadata for setup %i already exists and is inconsistent" % setup_id)
        else:
            f.create_dataset(key_res, data=scales)

        # write the chunk metadata for this set-up,
        # or if if we have this set-up already make sure
        # that the metadata is consistent
        key_chunks = 's%02i/subdivisions' % setup_id
        if key_chunks in f:
            chunks_expected = f[key_chunks][:]
            if not np.array_equal(chunks_expected, chunks):
                raise RuntimeError("Metadata for setup %i already exists and is inconsistent" % setup_id)
        else:
            f.create_dataset(key_chunks, data=chunks)


# n5 metadata format is specified here:
# https://github.com/bigdataviewer/bigdataviewer-core/blob/master/BDV%20N5%20format.md
def write_n5_metadata(path, scale_factors, resolution, setup_id=0, timepoint=0):
    # build the effective scale factors
    effective_scales = [scale_factors[0]]
    for factor in scale_factors[1:]:
        effective_scales.append([eff * fac
                                 for eff, fac in zip(effective_scales[-1], factor[::-1])])

    with open_file(path) as f:
        key = get_key(False, timepoint=timepoint, setup_id=setup_id, scale=0)
        dtype = str(f[key].dtype)

        root_key = get_key(False, setup_id=setup_id)
        root = f[root_key]
        attrs = root.attrs

        # write setup metadata / check for consistency if it already exists
        if 'downsamplingFactors' in attrs and attrs['downsamplingFactors'] != effective_scales:
            raise RuntimeError("Metadata for setup %i already exists and is inconsistent" % setup_id)
        else:
            root.attrs['downsamplingFactors'] = effective_scales

        if 'dataType' in attrs and attrs['dataType'] != dtype:
            raise RuntimeError("Metadata for setup %i already exists and is inconsistent" % setup_id)
        else:
            root.attrs['dataType'] = dtype

        group_key = get_key(False, timepoint=timepoint, setup_id=setup_id)
        g = f[group_key]
        g.attrs['multiScale'] = True
        g.attrs['resolution'] = resolution[::-1]

        effective_scale = [1, 1, 1]
        for scale_id, factor in enumerate(effective_scales):
            ds = g['s%i' % scale_id]
            effective_scale = [eff * sf for eff, sf in zip(effective_scale, factor)]
            ds.attrs['downsamplingFactors'] = factor


#
# helper functions to support affine transformations
#

def validate_affine(affine):

    def _check_affine(trafo):
        if len(trafo) != 12:
            raise ValueError("Invalid length of affine transformation, expect 12, got %i" % len(trafo))
        all_floats = all(isinstance(aff, float) for aff in trafo)
        if not all_floats:
            raise ValueError("Invalid datatype in affine transformation, expect list of floats")

    if isinstance(affine, list):
        _check_affine(affine)
    elif isinstance(affine, dict):
        for aff in affine.values():
            _check_affine(aff)
    else:
        raise ValueError("Invalid type for affine transformatin, expect list or dict, got %s" % type(affine))


def _write_default_affine(vregs, setup_id, timepoint, dx, dy, dz):
    vreg = ET.SubElement(vregs, 'ViewRegistration')
    vreg.set('timepoint', str(timepoint))
    vreg.set('setup', str(setup_id))
    vt = ET.SubElement(vreg, 'ViewTransform')
    vt.set('type', 'affine')
    ox, oy, oz = 0., 0., 0.
    ET.SubElement(vt, 'affine').text = '{} 0.0 0.0 {} 0.0 {} 0.0 {} 0.0 0.0 {} {}'.format(dx, ox,
                                                                                          dy, oy,
                                                                                          dz, oz)


def _write_affine(vregs, setup_id, timepoint, affine):
    vreg = ET.SubElement(vregs, 'ViewRegistration')
    vreg.set('timepoint', str(timepoint))
    vreg.set('setup', str(setup_id))
    if isinstance(affine, list):
        vt = ET.SubElement(vreg, 'ViewTransform')
        vt.set('type', 'affine')
        ET.SubElement(vt, 'affine').text = ' '.join([str(aff) for aff in affine])
    else:
        for name, affs in affine.items():
            vt = ET.SubElement(vreg, 'ViewTransform')
            vt.set('type', 'affine')
            ET.SubElement(vt, 'affine').text = ' '.join([str(aff) for aff in affs])
            ET.SubElement(vt, 'Name').text = name


def get_affine(xml_path, setup_id, timepoint=0):
    """ Get affine transformation for given setup id from xml.

    Arguments:
        xml_path (str): path to the xml file with the metadata
        setup_id (int): setup id for which the affine trafo(s) should be loaded
        timepoint (int): time point for which to load the affine (default: 0)
    Returns:
        dict: mapping name of transformation to its parameters
            If transformation does not have a name, will be called 'affine%i',
            where i is counting the number of transformations.
    """
    root = ET.parse(xml_path).getroot()
    vregs = root.find('ViewRegistrations')

    for vreg in vregs.findall('ViewRegistration'):
        setup = int(vreg.attrib['setup'])
        tp = int(vreg.attrib['timepoint'])
        if (setup != setup_id) or (timepoint != tp):
            continue

        ii = 0
        affine = {}
        for vt in vreg.findall('ViewTransform'):
            name = vt.find('Name')
            if name is None:
                name = 'affine%i' % ii
            else:
                name = name.text
            trafo = vt.find('affine').text
            trafo = [float(aff) for aff in trafo.split()]
            affine[name] = trafo
            ii += 1
        return affine

    raise RuntimeError("Could not find setup %i and timepoint %i" % (setup_id, timepoint))


#
# helper functions to read attributes from the xml metadata
#

def get_time_range(xml_path):
    """ Get the first and last timepoint present.

    Arguments:
        xml_path (str): path to the xml file with the metadata
    """
    root = ET.parse(xml_path).getroot()
    seqdesc = root.find('SequenceDescription')
    tpoints = seqdesc.find('Timepoints')
    first = int(tpoints.find('first').text)
    last = int(tpoints.find('last').text)
    return first, last


def get_bdv_format(xml_path):
    """ Get bigdataviewer data fromat.

    Arguments:
        xml_path (str): path to the xml file with the metadata
    """
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
    raise RuntimeError("Could not find setup %i" % setup_id)


def get_data_path(xml_path, return_absolute_path=False):
    """ Get path to the data.

    Arguments:
        xml_path (str): path to the xml file with the metadata
        return_absolute_path (bool): return the absolute path (default: False)
    """
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
