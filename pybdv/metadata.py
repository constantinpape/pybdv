import os
import numpy as np
import xml.etree.ElementTree as ET
from numbers import Number
from .util import open_file, get_key

MANDATORY_DISPLAY_SETTINGS = {'min', 'max', 'isset', 'color'}


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

def _require_view_setup(viewsets, setup_id, setup_name,
                        resolution, shape, attributes, unit,
                        overwrite, overwrite_data, enforce_consistency):

    # parse resolution and shape
    dz, dy, dx = resolution
    nz, ny, nx = tuple(shape)

    def _overwrite_setup(vs):
        # id, name and size
        vs.find('id').text = str(setup_id)
        vs.find('name').text = setup_name
        vs.find('size').text = '{} {} {}'.format(nx, ny, nz)

        # voxel size and unit of measurement
        vox = vs.find('voxelSize')
        vox.find('unit').text = unit
        vox.find('size').text = '{} {} {}'.format(dx, dy, dz)

        # attributes for this view setup
        attrs = vs.find('attributes')
        for att_name, att_values in attributes.items():
            attrs.find(att_name).text = str(att_values['id'])

    def _check_setup(vs):
        # check the name and size
        if vs.find('name').text != setup_name:
            raise ValueError("Incompatible setup name")
        shape_exp = vs.find('size').text.split()
        shape_exp = tuple(int(shp) for shp in shape_exp)
        if shape_exp != (nx, ny, nz):
            if overwrite_data:
                vs.find('size').text = '{} {} {}'.format(nx, ny, nz)
            else:
                raise ValueError(f"Incompatible dataset size: {shape_exp}, {(nx, ny, nz)}")

        # check the voxel size
        vox = vs.find('voxelSize')
        if vox.find('unit').text != unit:
            raise ValueError("Incompatible unit of measurement")
        res_exp = vox.find('size').text.split()
        res_exp = tuple(float(res) for res in res_exp)
        if res_exp != (dx, dy, dz):
            raise ValueError("Incompatible voxel size")

        # check the view attributes (only check for ids!)
        view_attrs = read_view_attributes(vs.find('attributes'))
        check_attrs = {k: v['id'] for k, v in attributes.items()}
        if view_attrs != check_attrs and enforce_consistency:
            raise ValueError("Incompatible view attributes")

    # check if we have the setup for this view already
    setups = viewsets.findall('ViewSetup')
    for vs in setups:
        # check if this set-up exists already
        if int(vs.find('id').text) == setup_id:

            # yes it exists and we are in over-write mode
            # -> over-write it
            if overwrite:
                _overwrite_setup(vs)

            # yes it exsits and we are not in over-write mode
            # -> check for consistency
            else:
                _check_setup(vs)

            return

    # we do not have this setup, so we write the setup configuration
    vs = ET.SubElement(viewsets, 'ViewSetup')

    # id, name and size
    ET.SubElement(vs, 'id').text = str(setup_id)
    ET.SubElement(vs, 'name').text = setup_name
    ET.SubElement(vs, 'size').text = '{} {} {}'.format(nx, ny, nz)

    # voxel size and unit of measurement
    vox = ET.SubElement(vs, 'voxelSize')
    ET.SubElement(vox, 'unit').text = unit
    ET.SubElement(vox, 'size').text = '{} {} {}'.format(dx, dy, dz)

    # attributes for this view setup
    attrs = ET.SubElement(vs, 'attributes')
    for att_name, att_values in attributes.items():
        ET.SubElement(attrs, att_name).text = str(att_values['id'])


def _initialize_attributes(viewsets, attributes):
    for att_name, att_values in attributes.items():
        attrsets = ET.SubElement(viewsets, 'Attributes')
        attrsets.attrib['name'] = att_name
        xml_name = att_name.capitalize()
        attr_setup = ET.SubElement(attrsets, xml_name)
        for name, val in att_values.items():
            ET.SubElement(attr_setup, name).text = " ".join(map(str, val)) if isinstance(val, list) else str(val)


def _update_attributes(viewsets, attributes, overwrite):
    attrsets = viewsets.findall('Attributes')
    for attrset in attrsets:
        this_name = attrset.attrib['name']

        # this attribute name should be present, otherwise 'validate_attributes' would
        # have thrown an error; so it's ok to just use assert here, because if this
        # throws it is a logic error, not a user error
        assert this_name in attributes, this_name
        this_values = attributes[this_name]
        assert 'id' in this_values
        this_id = this_values['id']

        xml_name = this_name.capitalize()
        attr_setups = attrset.findall(xml_name)
        attr_ids = [int(att_set.find('id').text) for att_set in attr_setups]

        # do we have this attribute id already?
        has_id = this_id in attr_ids

        # yes, we have it already and we are in over-write mode
        # -> over-write all the settings for this attribute
        if has_id and overwrite:
            # find this attribute setup, over-write existing keys
            # and add new keys
            for attr_setup in attr_setups:
                if int(attr_setup.find('id').text) != this_id:
                    continue

                for name, val in this_values.items():
                    elem = attr_setup.find(name)
                    if elem is None:
                        elem = ET.SubElement(attr_setup, name)
                    elem.text = str(val)

        # yes, we have it already, but we are not in over-write mode
        # -> do nothing
        elif has_id and not overwrite:
            continue

        # no, we don't have it
        # -> write the new attribute setup
        else:
            attr_setup = ET.SubElement(attrset, xml_name)
            for name, val in this_values.items():
                ET.SubElement(attr_setup, name).text = " ".join(map(str, val)) if isinstance(val, list) else str(val)


def write_xml_metadata(xml_path, data_path, unit, resolution, is_h5,
                       setup_id, timepoint, setup_name, affine, attributes,
                       overwrite, overwrite_data, enforce_consistency):
    """ Write bigdataviewer xml.

    Based on https://github.com/tlambert03/imarispy/blob/master/imarispy/bdv.py.
    Arguments:
        xml_path (str): path to xml meta data
        data_path (str): path to the data (in h5 or n5 format)
        unit (str): physical unit of the data
        resolution (str): resolution / voxel size of the data at the original scale
        is_h5 (bool): is the data in h5 or n5 format
        setup_id (int): id of the set-up
        timepoint (int): id of the time-point
        setup_name (str): name of this set-up
        affine (list[int] or dict[list[int]]): affine transformations for the view set-up
        attributes (dict[str, int]): view setup attributes
        overwrite (bool): whether to over-write existing setup id / timepoint
        overwrite_data (bool): whether to over-write purely data-related attributes
        enforce_consistency (bool): whether we enforce consistency of the setup attributes
    """
    # number of timepoints hard-coded to 1
    setup_name = 'Setup%i' % setup_id if setup_name is None else setup_name
    key = get_key(is_h5, timepoint=timepoint, setup_id=setup_id, scale=0)
    with open_file(data_path, 'r') as f:
        shape = f[key].shape

    format_type = 'hdf5' if is_h5 else 'n5'

    # check if we have xml with metadata already
    # -> yes we do
    if os.path.exists(xml_path):
        # parse the metadata from xml
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # load the sequence description
        seqdesc = root.find('SequenceDescription')

        # load the view descriptions and update the attributes
        viewsets = seqdesc.find('ViewSetups')
        _update_attributes(viewsets, attributes, overwrite)

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
        _initialize_attributes(viewsets, attributes)

        # make the registration decriptions
        vregs = ET.SubElement(root, 'ViewRegistrations')

        # timepoint description
        tpoints = ET.SubElement(seqdesc, 'Timepoints')
        tpoints.set('type', 'range')
        ET.SubElement(tpoints, 'first').text = str(timepoint)
        ET.SubElement(tpoints, 'last').text = str(timepoint)

    # require this view setup
    _require_view_setup(viewsets, setup_id, setup_name,
                        resolution, shape, attributes, unit,
                        overwrite, overwrite_data, enforce_consistency)

    # write the affine transformation(s) for this view registration
    _write_transformation(vregs, setup_id, timepoint, affine, resolution, overwrite)

    # write the xml
    indent_xml(root)
    tree = ET.ElementTree(root)
    tree.write(xml_path)


def write_h5_metadata(path, scale_factors, setup_id=0, timepoint=0, overwrite=False):
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
            if out_key not in f:
                continue
            # for some reason I don't understand we do not need to invert here
            chunk = f[out_key].chunks[::-1]

        scales.append(effective_scale[::-1])
        chunks.append(chunk)

    scales = np.array(scales).astype('float32')
    chunks = np.array(chunks).astype('int')
    with open_file(path, 'a') as f:

        # write the resolution metadata for this set-up,
        # or if if we have this set-up already make sure
        # that the metadata is consistent (unless in over-write mode)
        def _write_mdata(key, mdata):
            if key in f and not overwrite:
                return
            elif key in f and overwrite:
                del f[key]
                f.create_dataset(key, data=mdata)
            else:
                f.create_dataset(key, data=mdata)

        key_res = 's%02i/resolutions' % setup_id
        _write_mdata(key_res, scales)

        key_chunks = 's%02i/subdivisions' % setup_id
        _write_mdata(key_chunks, chunks)


# n5 metadata format is specified here:
# https://github.com/bigdataviewer/bigdataviewer-core/blob/master/BDV%20N5%20format.md
def write_n5_metadata(path, scale_factors, resolution, setup_id=0, timepoint=0, overwrite=False):
    # build the effective scale factors
    effective_scales = [scale_factors[0]]
    for factor in scale_factors[1:]:
        effective_scales.append([eff * fac
                                 for eff, fac in zip(effective_scales[-1], factor[::-1])])

    with open_file(path, 'a') as f:
        key = get_key(False, timepoint=timepoint, setup_id=setup_id, scale=0)
        dtype = str(f[key].dtype)

        root_key = get_key(False, setup_id=setup_id)
        root = f[root_key]
        attrs = root.attrs

        # write setup metadata / check for consistency if it already exists
        if 'downsamplingFactors' in attrs and not overwrite:
            return
        root.attrs['downsamplingFactors'] = effective_scales

        if 'dataType' in attrs and not overwrite:
            return
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
# helper functions to support attributes
#


def _validate_attribute_id(this_attributes, this_id, xml_ids, enforce_consistency, name):
    """ Validate attribute id.
    """

    # the given id is None and we don't have setup attributes
    # -> increase current max id for the attribute by 1
    if this_id is None and this_attributes is None:
        this_id = max(xml_ids) + 1

    # the given id is None and we do have setup attributes
    # set id to the id present in the setup
    elif this_id is None and this_attributes is not None:
        this_id = this_attributes[name]

    # the given id is not None and we do have setup attributes
    # -> check that the ids match (unless we are in over-write mode)
    elif this_id is not None and this_attributes is not None:
        if (this_id != this_attributes[name]) and enforce_consistency:
            raise ValueError("Expect id %i for attribute %s, got %i" % (this_attributes[name],
                                                                        name,
                                                                        this_id))
    return this_id


def _validate_attribute_dict(this_attributes, this_attribute_values,
                             xml_ids, enforce_consistency, name):
    """ Validate attribute values, which are a dict.
    For now, we only validate and update the id given in the dict and do
    not check for consistency of other values (but we check that these are not nested types).
    """
    try:
        this_id = this_attribute_values['id']
    except KeyError:
        raise ValueError("Attribute values muset to contain entry with key 'id'")

    # validate the id
    this_id = _validate_attribute_id(this_attributes, this_id, xml_ids, enforce_consistency, name)

    # extra checks for display settings, which needs some additional keys
    if name == 'displaysettings':
        value_names = set(this_attribute_values.keys())
        if len(MANDATORY_DISPLAY_SETTINGS - value_names) != 0:
            raise ValueError("Not all mandatory display settings were passed")

    # make output values and check that all other attribute values are of simple type
    values_out = {'id': this_id}
    for k, v in this_attribute_values.items():
        if k == 'id':
            continue
        # check simple type
        if not (isinstance(v, (str, list, Number))):
            raise ValueError("Attribute values must be list, string, bool or number, got %s" % type(v))
        if isinstance(v, list) and not all(isinstance(vv, (str, Number)) for vv in v):
            raise ValueError("List attributes can only contain simple types")
        values_out[k] = v

    return values_out


def _validate_existing_attributes(setups, setup_id, attributes, enforce_consistency):
    """ Validate the attributes with metadata already present, increase the 'id' if it's None.
    """
    # check if we have this view already, if we do load it's
    # attribute mapping
    this_attributes = None
    viewsets = setups.findall('ViewSetup')
    for viewset in viewsets:
        if int(viewset.find('id').text) == setup_id:
            this_attributes = viewset.find('attributes')
            this_attributes = read_view_attributes(this_attributes)
            break

    # get all the attribute setups
    attrs_xml = setups.findall('Attributes')
    all_names_xml = set()

    # iterate over the attributes and make sure that all attribute names exist
    # and check the attribute ids
    attrs_out = {}
    for attribute in attrs_xml:
        name = attribute.attrib['name']
        if name not in attributes:
            raise ValueError("Expected attributes to contain %s" % name)
        all_names_xml.update({name})

        xml_ids = [int(child.find('id').text) for child in attribute]
        this_attribute_values = attributes[name]

        if not isinstance(this_attribute_values, dict):
            raise ValueError("Expected or dict, got %s" % type(this_attribute_values))

        this_out = _validate_attribute_dict(this_attributes, this_attribute_values,
                                            xml_ids, enforce_consistency, name)
        attrs_out[name] = this_out

    # check that we don't have excess names in the input attributes
    this_names = set(attributes.keys())
    if len(this_names - all_names_xml) > 0:
        raise ValueError("Attributes contains unexpected names")

    return attrs_out


def _validate_new_attributes(attributes):

    def _validate_new(name, value):
        if not isinstance(value, dict):
            raise ValueError("Expected dict, got %s" % type(value))

        if 'id' not in value:
            raise ValueError("Attribute values must to contain entry 'id'")

        # extra checks for display settings, which needs some additional keys
        if name == 'displaysettings':
            value_names = set(value.keys())
            if len(MANDATORY_DISPLAY_SETTINGS - value_names) != 0:
                raise ValueError("Not all mandatory display settings were passed")

        new_value = {}
        for k, v in value.items():
            v = 0 if (k == 'id' and v is None) else v
            if isinstance(v, list) and not all(isinstance(vv, (str, Number)) for vv in v):
                raise ValueError("List attributes can only contain simple types")
            elif not isinstance(v, (str, Number, list)):
                raise ValueError("Attribute values must be list, string, bool or number, got %s" % type(v))
            new_value[k] = v
        return new_value

    attrs_out = {k: _validate_new(k, v) for k, v in attributes.items()}
    return attrs_out


def validate_attributes(xml_path, attributes, setup_id, enforce_consistency):
    if os.path.exists(xml_path):
        setups = ET.parse(xml_path).getroot().find('SequenceDescription').find('ViewSetups')
        attrs_out = _validate_existing_attributes(setups, setup_id, attributes, enforce_consistency)
    else:
        attrs_out = _validate_new_attributes(attributes)
    return attrs_out


def read_view_attributes(view_attrs):
    return {att.tag: int(att.text) for att in view_attrs}


def read_attributes(attributes):

    # cast from str to corresponding type
    def _cast(val):
        try:
            val = int(val)
            return val
        except ValueError:
            pass
        try:
            val = float(val)
            return val
        except ValueError:
            pass
        return val

    return {att.tag: _cast(att.text) for att in attributes}


def get_attributes(xml_path, setup_id):
    """ Read attributes for a given setup id

    Arguments:
        xml_path (str): path to the xml file with the metadata
        setup_id (int): setup id for which to read the attributes
    """
    root = ET.parse(xml_path).getroot()
    setups = root.find('SequenceDescription').find('ViewSetups')

    viewsets = setups.findall('ViewSetup')
    attr_ids = None
    for viewset in viewsets:
        if int(viewset.find('id').text) == setup_id:
            attr_ids = read_view_attributes(viewset.find('attributes'))

    if attr_ids is None:
        raise ValueError("Could not find setup %i" % setup_id)

    attributes = {}
    attribute_settings = setups.findall('Attributes')
    for attribute_setups in attribute_settings:
        name = attribute_setups.attrib['name']
        assert name in attr_ids, name
        this_id = attr_ids[name]

        this_attrs = None
        for attribute_setup in attribute_setups:
            attrs = read_attributes(attribute_setup)
            if attrs['id'] == this_id:
                this_attrs = attrs
                break

        assert this_attrs is not None

        attributes[name] = this_attrs

    return attributes


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
        raise ValueError("Invalid type for affine transformation, expect list or dict, got %s" % type(affine))


def _write_transformation(vregs, setup_id, timepoint, affine, resolution, overwrite):

    def write_trafo(vreg):
        if isinstance(affine, dict):
            for name, affs in affine.items():
                vt = ET.SubElement(vreg, 'ViewTransform')
                vt.set('type', 'affine')
                ET.SubElement(vt, 'affine').text = ' '.join(map(str, affs))
                # NOTE for CP: truncating the number of digits here is not a good idea
                # ET.SubElement(vt, 'affine').text = ' '.join(['{:.4f}'.format(aff) for aff in affs])
                ET.SubElement(vt, 'name').text = name
        else:
            if affine is None:
                dz, dy, dx = resolution
                ox, oy, oz = 0., 0., 0.
                trafo = '{} 0.0 0.0 {} 0.0 {} 0.0 {} 0.0 0.0 {} {}'.format(dx, ox,
                                                                           dy, oy,
                                                                           dz, oz)
            else:
                # NOTE for CP: truncating the number of digits here is not a good idea
                # trafo = ' '.join(['{:.4f}'.format(aff) for aff in affine])
                trafo = ' '.join(map(str, affine))
            vt = ET.SubElement(vreg, 'ViewTransform')
            vt.set('type', 'affine')
            ET.SubElement(vt, 'affine').text = trafo

    # check if we have the affine for this setup-id and timepoint already
    vreg = None
    for vreg_candidate in vregs.findall('ViewRegistration'):
        setup = int(vreg_candidate.attrib['setup'])
        tp = int(vreg_candidate.attrib['timepoint'])
        if (setup == setup_id) and (timepoint == tp):
            vreg = vreg_candidate
            break

    # if we don't have it yet, create the view registration
    if vreg is None:
        vreg = ET.SubElement(vregs, 'ViewRegistration')
    # if we have the view registration and over-write, clear the current trafo
    elif vreg is not None and overwrite:
        vreg.clear()
    # otherwise, the trafo exists already and we don't over-write -> do nothing
    else:
        return

    vreg.set('timepoint', str(timepoint))
    vreg.set('setup', str(setup_id))
    write_trafo(vreg)


def write_affine(xml_path, setup_id, affine, overwrite, timepoint=0):
    """ Write affine transformation for given setup id from xml.

    Arguments:
        xml_path (str): path to the xml file with the bdv metadata
        setup_id (int): setup id for which the affine trafo(s) should be loaded
        affine (list[int] or dict[list[int]]): affine transformation(s)
        overwrite (bool): whether to over-write existing transformations or add the new one
        timepoint (int): time point for which to load the affine (default: 0)
    """

    # validate the input transformation and write it to the metadta
    validate_affine(affine)
    root = ET.parse(xml_path).getroot()
    vregs = root.find('ViewRegistrations')
    _write_transformation(vregs, setup_id, timepoint,
                          affine=affine,
                          resolution=None,
                          overwrite=overwrite)

    # write the xml
    indent_xml(root)
    tree = ET.ElementTree(root)
    tree.write(xml_path)


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
            name = vt.find('name')
            if name is None:
                name = 'affine%i' % ii
            else:
                name = name.text
            trafo = vt.find('affine').text
            trafo = [float(aff) for aff in trafo.split()]
            affine[name] = trafo
            ii += 1
        return affine

    raise ValueError("Could not find setup %i and timepoint %i" % (setup_id, timepoint))


#
# helper functions to read additional data from the xml metadata
#


def get_setup_ids(xml_path):
    """ Get all available setup ids.

    Arguments:
        xml_path (str): path to the xml file with the metadata
    """
    ids = []
    root = ET.parse(xml_path).getroot()
    viewsets = root.find('SequenceDescription').find('ViewSetups')
    vsetups = viewsets.findall('ViewSetup')
    for vs in vsetups:
        ids.append(int(vs.find('id').text))
    return ids


def get_timeponts(xml_path, setup_id):
    """ Get timepoints for a given setup id.

    Arguments:
        xml_path (str): path to the xml file with the metadata
    """
    timepoints = []
    root = ET.parse(xml_path).getroot()
    viewregs = root.find('ViewRegistrations').findall('ViewRegistration')
    for reg in viewregs:
        this_id = int(reg.attrib['setup'])
        this_tp = int(reg.attrib['timepoint'])
        if this_id == setup_id:
            timepoints.append(this_tp)
    timepoints.sort()
    return timepoints


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
    raise ValueError("Could not find setup %i" % setup_id)


def get_size(xml_path, setup_id):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    seqdesc = root.find('SequenceDescription')
    viewsets = seqdesc.find('ViewSetups')
    vsetups = viewsets.findall('ViewSetup')
    for vs in vsetups:
        if vs.find('id').text == str(setup_id):
            size = vs.find('size').text
            return tuple(int(siz) for siz in size.split())[::-1]
    raise ValueError("Could not find setup %i" % setup_id)


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
        raise ValueError("Could not find valid data path in xml.")
    path = node.text
    # this assumes relative path in xml
    if return_absolute_path:
        path = os.path.join(os.path.split(xml_path)[0], path)
        path = os.path.abspath(os.path.relpath(path))
    return path


#
# helper functions to write additional xml metadata
#


def write_size_and_resolution(xml_path, setup_id, size, resolution):
    """ Write size and resolution data.
    """

    if size is not None and len(size) != 3:
        raise ValueError(f"Expected size of length 3 instead of {len(size)}")
    if resolution is not None and len(resolution) != 3:
        raise ValueError(f"Expected resolution of length 3 instead of {len(resolution)}")

    tree = ET.parse(xml_path)
    root = tree.getroot()
    seqdesc = root.find('SequenceDescription')
    viewsets = seqdesc.find('ViewSetups')
    vsetups = viewsets.findall('ViewSetup')

    found_setup = False
    for vs in vsetups:
        if vs.find('id').text == str(setup_id):
            size_elem = vs.find('size')
            if size is not None:
                size_elem.text = ' '.join(map(str, size[::-1]))

            res_elem = vs.find('voxelSize').find('size')
            if resolution is not None:
                res_elem.text = ' '.join(map(str, resolution[::-1]))

            found_setup = True

    if found_setup:
        # write the xml
        indent_xml(root)
        tree = ET.ElementTree(root)
        tree.write(xml_path)
    else:
        raise ValueError("Could not find setup %i" % setup_id)
