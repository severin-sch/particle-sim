import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from pytest import mark

import fields
import simulation


@mark.parametrize(
    "position, velocity, expected_position, expected_velocity",
    [
        ([0, 0], [0, 0], [1, 0], [1, 0]),
        ([0, 0], [0, 1], [1, 1], [1, 1]),
        ([1, 0], [0, 0], [1, 1], [0, 1]),
        ([0.8, 0.8], [0, 0], [1.8, 0.8], [1, 0]),
    ],
)
def test_next_position_velocity(
    position, velocity, expected_position, expected_velocity
):
    forces = np.array([[[1, 0], [0, 1]], [[0, 0], [0, 0]]])
    next_position, next_velocity = simulation.next_position_velocity(
        forces, 1, np.array(position), np.array(velocity)
    )

    assert_array_equal(next_position, expected_position)
    assert_array_equal(next_velocity, expected_velocity)


def test_simulate():
    """Smoke test to ensure it doesn't error out."""
    field = fields.scalar_field()
    vectors = fields.vector_field(field)

    simulation.simulate(vectors, 1, (12, 12))


@mark.parametrize(
    "position, shape, new_shape, expected",
    [
        # ([0, 0], [1, 1], [1, 1], [[1]]),
        ([0.5, 0.5], [2, 2], [2, 2], [[0.25, 0.25], [0.25, 0.25]]),
        ([0.5, 0], [2, 2], [2, 2], [[0.5, 0.5], [0, 0]]),
        ([0.8, 0.6], [2, 2], [2, 2], [[0.2 * 0.4, 0.8 * 0.4], [0.2 * 0.6, 0.8 * 0.6]]),
        ([1, 1], [3, 3], [2, 2], [[0.25, 0.25], [0.25, 0.25]]),
        ([1, 2], [4, 4], [3, 3], [[0, 0, 0], [2 / 9, 4 / 9, 0], [1 / 9, 2 / 9, 0]]),
    ],
)
def test_four_warm_position(position, shape, new_shape, expected):
    assert_array_almost_equal(
        simulation.four_warm_position(shape, new_shape, position), expected
    )
