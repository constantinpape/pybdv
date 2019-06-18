import xml.etree.ElementTree as ET
import numpy as np
from .metadata import indent_xml


#
# switch between transformation vector and affine transformation matrix
# Affine Matrix:
# [[a00 a01 a02 a03]
#  [a10 a11 a12 a13]
#  [a20 a21 a22 a23]
#  [  0   0   0   1]]
# Transformation Vector (as stored by bdv)
# [a00 a01 a02 a03 a10 a11 a12 a13 a20 a21 a22 a23]
#


def transformation_to_matrix(transformation):
    """ From bdv transformation vector to affine matrix.
    """
    matrix = np.zeros((4, 4))

    matrix[0, :] = transformation[0:4]
    matrix[1, :] = transformation[4:8]
    matrix[2, :] = transformation[8:12]

    matrix[3, 3] = 1
    return matrix


def matrix_to_transformation(matrix):
    """ From affine matrix to bdv transformation vector.
    """
    expected_last_row = np.array([0, 0, 0, 1])
    assert matrix.shape == (4, 4)
    assert (matrix[3, :] == expected_last_row).all(),\
        "%s, %s" % (str(matrix[3, :]), str(expected_last_row))

    transformation = np.zeros(12)
    transformation[0:4] = matrix[0, :]
    transformation[4:8] = matrix[1, :]
    transformation[8:12] = matrix[2, :]

    return transformation


#
# read and write transformations from / to xml
# TODO support multiple set-ups / time-points
#


def read_resolution_and_transformation(xml):
    """ Read resolution and transformation vector from bdv xml.
    """
    tree = ET.ElementTree(file=xml)

    for elem in tree.iter():
        if elem.tag == 'voxelSize':
            for child in elem:
                if child.tag == 'size':
                    resolution = child.text
    resolution = [float(re) for re in resolution.split()]

    for elem in tree.iter():
        if elem.tag == 'ViewTransform':
            for child in elem:
                transformation = child.text
    transformation = transformation.split()
    transformation = [float(trafo) for trafo in transformation]

    return resolution, transformation


def write_resolution_and_transformation(xml_in, xml_out,
                                        resolution, transformation):
    """ Write resolution and transformation vector to bdv xml.

    Needs an input bdv xml and will copy everything except resolution
    and transformation to the output xml.
    """
    assert len(resolution) == 3
    assert len(transformation) == 12

    tree = ET.ElementTree(file=xml_in)

    res_string = ' '.join([str(res) for res in resolution])
    for elem in tree.iter():
        if elem.tag == 'voxelSize':
            for child in elem:
                if child.tag == 'size':
                    child.text = res_string

    trafo_string = ' '.join([str(trafo) for trafo in transformation])
    for elem in tree.iter():
        if elem.tag == 'ViewTransform':
            for child in elem:
                child.text = trafo_string

    root = tree.getroot()
    indent_xml(root)
    tree = ET.ElementTree(root)
    tree.write(xml_out)


def write_resolution_and_matrix(xml_in, xml_out, resolution, matrix):
    """ Write resolution and affine matrix describing transformation to bdv xml.

    Needs an input bdv xml and will copy everything except resolution
    and transformation to the output xml.
    """
    transformation = matrix_to_transformation(matrix)
    write_resolution_and_transformation(xml_in, xml_out, resolution, transformation)


def read_resolution_and_matrix(xml):
    """ Read resolution and affine matrix describing transfomration from bdv xml.
    """
    resolution, transformation = read_resolution_and_transformation(xml)
    matrix = transformation_to_matrix(transformation)
    return resolution, matrix

# operations for affine transformation matrix
# for details, see
# https://math.stackexchange.com/questions/237369/given-this-transformation-matrix-how-do-i-decompose-it-into-translation-rotati
# TODO It should not be necessary to do the decomposition; should be enought to just build the new affine matrix and apply
# before changing anything now, we need tests!


def get_translation_from_matrix(matrix):
    """ Extract translation matrix from affine matrix
    """
    t = np.zeros((4, 4))
    for i in range(3):
        t[i, 3] = matrix[i, 3]
        t[i, i] = 1
    t[3, 3] = 1
    return t


def get_scaling_from_matrix(matrix):
    """ Extract scaling matrix from affine matrix
    """
    s = np.zeros((4, 4))
    for i in range(3):
        s[i, i] = np.linalg.norm(matrix[:, i])
    s[3, 3] = 1
    return s


def get_rotation_and_shear_from_matrix(matrix):
    """ Extract rotation / shear matrix from affine matrix
    """
    r = np.zeros((4, 4))

    # need scaling to compute rotation/shear
    s = get_scaling_from_matrix(matrix)
    s = np.array([s[0, 0], s[1, 1], s[2, 2]])

    for i in range(3):
        r[i, :3] = matrix[i, :3] / s

    r[3, 3] = 1
    return r


def decompose_matrix(matrix):
    """ Decompose matrix into translation, scaling and rotation/shear.
    """
    t = get_translation_from_matrix(matrix)
    s = get_scaling_from_matrix(matrix)
    r = get_rotation_and_shear_from_matrix(matrix)
    return t, s, r


def scale_matrix(matrix, scale_factor):
    """ Scale affine transformation described by the matrix
    by a scale_factor.
    """
    if isinstance(scale_factor, int):
        scale_factor = 3*[scale_factor]
    assert len(scale_factor) == 3, "%i" % len(scale_factor)

    # decompose transformations
    t, s, r = decompose_matrix(matrix)

    # apply the scale factor to the scale_matrix
    s[0, 0] *= scale_factor[0]
    s[1, 1] *= scale_factor[1]
    s[2, 2] *= scale_factor[2]

    # compute new transformation
    matrix = np.matmul(np.matmul(t, s), r)
    return matrix


def translate_matrix(matrix, translation):
    """ Add translation to affine transformation described by the matrix.
    """
    if isinstance(translation, int):
        translation = 3*[translation]
    assert len(translation) == 3, "%i" % len(translation)

    # decompose transformations
    t, s, r = decompose_matrix(matrix)

    # apply the scale factor to the scale_matrix
    t[0, -1] += translation[0]
    t[1, -1] += translation[1]
    t[2, -1] += translation[2]

    # compute new transformation
    matrix = np.matmul(np.matmul(t, s), r)
    return matrix


# see https://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions
def build_rotation_matrix(phi, theta, psi):
    """ Rotation matrix from euler angles in degree.
    """
    phi, theta, psi = np.deg2rad([phi, theta, psi])
    r = np.zeros((4, 4))

    cos, sin = np.cos, np.sin

    r[0, 0] = cos(theta) * cos(psi)
    r[0, 1] = (- cos(phi) * sin(psi) + sin(phi) * sin(theta) * cos(psi))
    r[0, 2] = sin(phi) * sin(psi) + cos(phi) * sin(theta) * cos(psi)

    r[1, 0] = cos(theta) * sin(psi)
    r[1, 1] = cos(phi) * cos(psi) + sin(phi) * sin(theta) * sin(psi)
    r[1, 2] = (- sin(phi) * cos(psi) + cos(phi) * sin(theta) * sin(psi))

    r[2, 0] = - sin(theta)
    r[2, 1] = sin(phi) * cos(theta)
    r[2, 2] = cos(phi) * cos(theta)

    r[3, 3] = 1
    return r


def rotate_matrix(matrix, phi, theta, psi):
    """ Rotate affine transformation described by the matrix.
        The rotation is described by the three euler angles
        phi, theta, psi (in degree).

        This is untested!
    """
    # decompose transformations
    t, s, r = decompose_matrix(matrix)

    # build the rotation matrix
    rot = build_rotation_matrix(phi, theta, psi)

    # TODO is this the correct order of applying the trafos?
    r = np.matmul(rot, r)

    # compute new transformation
    matrix = np.matmul(np.matmul(t, s), r)
    return matrix
