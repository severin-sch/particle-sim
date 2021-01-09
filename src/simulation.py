"""Simulation of a particle in a force field"""
import numpy as np


def next_position_velocity(force_field, mass, position, velocity):
    r"""Calculate the position and velocity in the next time step.

    The x and y indices of the force vectors in the fields are assumed to apply
    in all positions :math:`(x', y') \in [x, x+1) \times [y, y+1)`.

    :param force_field: A force field as 3d, matrix-indexed array of shape (y, x, 2).
    :param mass: Mass of the particle.
    :param position: Current position of the particle, as an array of cartesian
                     coordinates ``(x, y)``.
    :param velocity: Current velocity of the particle in the given field, as an
                     array like ``position``.
    :returns: The next position and velocity of the particle, as tuple of
              cartesian coordinate arrays.
    """
    x, y = position
    acceleration = force_field[int(y), int(x)] / mass

    next_vel = velocity + acceleration
    next_pos = position + next_vel

    return next_pos, next_vel


def four_warm_position(prev_shape, new_shape, position):
    """Returns downsized "four-warm" encoded position of a particle

    :param prev_shape: A tuple ``(y, x)``, the coordinate system the position is
                       given in
    :param new_shape: A tuple ``(y, x)``, the shape of the field to four-warm
                      encode the position in
    :param position: A tuple ``(x, y)``, the original position
    :return: A 2d matrix-indexed array with shape ``new_shape``, where the
             position is encoded into the four nearest indices.
    """
    prev_y, prev_x = prev_shape
    new_y, new_x = new_shape

    factorx = (prev_x - 1) / (new_x - 1)
    factory = (prev_y - 1) / (new_y - 1)
    x = position[0] / factorx
    y = position[1] / factory

    int_x, dec_x = int(x), x % 1
    int_y, dec_y = int(y), y % 1

    pos_field = np.zeros(new_shape)
    pos_field[int_y, int_x] = (1 - dec_y) * (1 - dec_x)
    if int_y + 1 < new_shape[0]:
        pos_field[int_y + 1, int_x] = dec_y * (1 - dec_x)
    if int_x + 1 < new_shape[1]:
        pos_field[int_y, int_x + 1] = (1 - dec_y) * dec_x
    if int_y + 1 < new_shape[0] and int_x + 1 < new_shape[0]:
        pos_field[int_y + 1, int_x + 1] = dec_y * dec_x

    return pos_field


def simulate(force_field, mass, initial_position, steps=100, initial_velocity=(0, 0)):
    """Simulate a particle in a mass field

    :param force_field: A force field as 3d matrix-indexed array of shape (y, x, 2).
    :param mass: Mass of the particle to simulate.
    :param initial_position: Initial position of the particle to simulate, as
                             cartesian coordinates
    :param steps: Number of time steps to simulate over, including the initial step.
    :param initial_velocity: The initial velocity of the particle.
    :return: An array of shape ``(steps, 2)``, containing the cartesian
             coordinates of the particle for each time step.
    """
    positions = np.zeros((steps, 2))
    positions[0] = initial_position

    velocity = np.asarray(initial_velocity)
    for i in range(1, steps):
        position, velocity = next_position_velocity(
            force_field, mass, positions[i - 1], velocity
        )
        positions[i] = position

    return positions
