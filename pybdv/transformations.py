import xml.etree.ElementTree as ET
import numpy as np


# TODO support multiple set-ups / time-points
def load_resolution_and_transformation(xml):
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
    transformation = [float(trafo) for trafo in transformation]

    return resolution, transformation


# TODO support multiple set-ups / time-points
def write_resolution_and_transformation(xml, resolution, transformation):
    assert len(resolution) == 3
    assert len(transformation) == 12

    tree = ET.ElementTree(file=xml)

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


def write_resolution_and_matrix(xml, resolution, matrix):
    transformation = matrix_to_transformation(matrix)
    write_resolution_and_transformation(xml, resolution, transformation)


# TODO figure out row-major etc. here
def transformation_to_matrix(transformation):
    matrix = np.zeros((4, 4))

    matrix[0, 0] = transformation[0]
    matrix[0, 1] = transformation[1]
    matrix[0, 2] = transformation[2]
    matrix[0, 3] = transformation[3]

    matrix[1, 0] = transformation[4]
    matrix[1, 1] = transformation[5]
    matrix[1, 2] = transformation[6]
    matrix[1, 3] = transformation[7]

    matrix[2, 0] = transformation[8]
    matrix[2, 1] = transformation[9]
    matrix[2, 2] = transformation[10]
    matrix[2, 3] = transformation[11]

    matrix[3, 3] = 1
    return matrix


def matrix_to_transformation(matrix):
    expected_col = np.array([0, 0, 0, 1])
    assert matrix.shape == (4, 4)
    assert (matrix[3, :] == expected_col).all()
    assert (matrix[:, 3] == expected_col).all()

    transformation = np.zeros(12)
    transformation[0:4] = matrix[0, :]
    transformation[4:8] = matrix[1, :]
    transformation[8:12] = matrix[2, :]

    return transformation


def load_resolution_and_matrix(xml):
    resolution, transformation = load_resolution_and_transformation(xml)
    matrix = transformation_to_matrix(transformation)
    return resolution, matrix


# see this for affine matrix operations:
# https://math.stackexchange.com/questions/237369/given-this-transformation-matrix-how-do-i-decompose-it-into-translation-rotati
def get_translation_from_matrix(matrix):
    t = np.zeros((4, 4))
    # TODO [0, i] or [i, 0] ???
    for i in range(3):
        t[i, 0] = matrix[i, 0]
    return t


def get_scaling_from_matrix(matrix):
    s = np.zeros((4, 4))
    for i in range(3):
        s[i, i] = np.linalg.norm(matrix[:, i])
    return s


def get_rotation_and_shear_from_matrix(matrix):
    r = np.zeros((4, 4))

    # need scaling to compute rotation/shear
    s = get_scaling_from_matrix(matrix)
    sx, sy, sz = s[0, 0], s[1, 1], s[2, 2]

    r[0, 0] = matrix[0, 0] / sx
    r[0, 1] = matrix[0, 1] / sy
    r[0, 2] = matrix[0, 2] / sz

    r[1, 0] = matrix[1, 0] / sx
    r[1, 1] = matrix[1, 1] / sy
    r[1, 2] = matrix[1, 2] / sz

    r[2, 0] = matrix[2, 0] / sx
    r[2, 1] = matrix[2, 1] / sy
    r[2, 2] = matrix[2, 2] / sz

    r[3, 3] = 1
    return r


def decompose_matrix(matrix):
    t = get_translation_from_matrix(matrix)
    s = get_scaling_from_matrix(matrix)
    r = get_rotation_and_shear_from_matrix(matrix)
    return t, s, r


def scale_matrix(matrix, scale_factor):
    if isinstance(scale_factor, int):
        scale_factor = 3*[scale_factor]
    assert len(scale_factor) == 3, "%i" % len(scale_factor)

    t, s, r = decompose_matrix(matrix)
    # apply the scale factor to the scale_matrix
    s[0, 0] *= scale_factor[0]
    s[1, 1] *= scale_factor[1]
    s[2, 2] *= scale_factor[2]

    # TODO is this the correct order ?
    matrix = np.matmul(np.matmul(t, s), r)

    return matrix
