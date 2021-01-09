import numpy as np
from numpy.testing import assert_array_equal

import fields

def test_vector_field_shape():
    vectors = fields.vector_field(np.zeros((10, 20)))

    assert vectors.shape == (10, 20, 2)


def test_downsize_field():
    field = np.array([[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]])
    expected = np.array([[1, 2], [3, 4]])

    assert_array_equal(fields.downsize_field(field, (2, 2)), expected)
