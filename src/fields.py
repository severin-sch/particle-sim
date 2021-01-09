import numpy as np
import noise


def trap(field, flatness=25, safety=1.1):
    """Convert the field to a trap

    :param field: Field to turn into a trap. Ideally with values from -1 to 1.
                  Matrix-indexed
    :param flatness: Controls how "wide" the edges of the bowl are
    :param safety: Height of the edges
    :return: The field turned into a trap
    """
    x, y = np.meshgrid(np.linspace(-flatness, flatness, field.shape[0]),
                       np.linspace(-flatness, flatness, field.shape[1]))
    trap = np.max([np.exp(np.abs(x) - flatness),
                   np.exp(np.abs(y) - flatness)], axis=0)
    return (1-trap) * field + safety * trap


def scalar_field(size=1000, rng=None):
    """Produces a square scalar field

    :param size: Dimensions of field
    :param rng: Random number generator for the field
    :returns: A scalar field of noise, as a 2d array
    """
    if rng is None:
        rng = np.random.default_rng(seed=0)

    scale = size
    octaves = 4
    persistence = 0.5
    lacunarity = 2
    offset_x, offset_y = (rng.random(2) - 0.5)*2e5

    field = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            field[i][j] = noise.pnoise2(
                offset_x + i / scale,
                offset_y + j / scale,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
            )
    field /= np.max(np.abs(field))
    return field


def vector_field(field):
    """Calculates gradient of scalar field

    The field is assumed to be matrix-indexed, while the gradient vectors will
    be cartesian.

    :param field: Scalar field, as 2d matrix-indexed array
    :returns: Gradient vectors as matrix-indexed array with shape ``(y, x, 2)``.
              Lowest dimension is cartesian coordinate tuples.
    """
    # Swapped order because field is cartesian indexed
    dys, dxs = np.gradient(field)

    # Because the gradient points in the direction of greatest ascent, we need
    # to go the other way
    # Moving the axes takes us from shape (2, x, y) to (x, y, 2)
    vectors = np.moveaxis([-dxs, -dys], 0, -1)
    return vectors


def downsize_field(field, shape):
    """Downsizes a field to the desired shape

    :param field: A scalar force field
    :param shape: (y,x) tuple of desired shape,
                  must be multiples of previous shape
    :returns: Reshaped scalar field
    """
    return field.reshape(
        (shape[0], field.shape[0] // shape[0], shape[1], field.shape[1] // shape[1])
    ).mean((1, 3))
