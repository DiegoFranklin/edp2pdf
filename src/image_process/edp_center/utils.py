import numpy as np
from typing import Tuple

def get_centered_crop_img(image, center):
    # crops the image to biggest area possible so its get centered on center
    c_x, c_y = map(round, center[::-1])

    x_lenth, y_lenth = image.shape

    radius = min(c_x, c_y, abs(x_lenth-c_x), abs(y_lenth-c_y))

    return image[c_x-radius:c_x+radius, c_y-radius:c_y+radius] 


def bilinear_interpolation(point: Tuple[float, float], 
                           neighbors: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]],
                           values: Tuple[float, float, float, float]) -> float:
    """
    Performs bilinear interpolation for a given point based on the values of its four neighbors.

    Args:
        point: A tuple (x, y) representing the coordinates where we want to interpolate.
        neighbors: A tuple containing the four corner integer coordinates:
                   ((x0, y0), (x0, y1), (x1, y0), (x1, y1))
        values: A tuple containing the values at the four corners:
                (value_00, value_01, value_10, value_11)

    Returns:
        The interpolated value at the specified point.
    """
    x, y = point 
    (x0, y0), (x0, y1), (x1, y0), (x1, y1) = neighbors
    value_00, value_01, value_10, value_11 = values

    Q11 = value_00 * (x1 - x) * (y1 - y)
    Q12 = value_01 * (x1 - x) * (y - y0)
    Q21 = value_10 * (x - x0) * (y1 - y)
    Q22 = value_11 * (x - x0) * (y - y0)

    # Bilinear interpolation
    interpolated_value = (Q11 + Q12 + Q21 + Q22) / ((x1 - x0) * (y1 - y0))

    return interpolated_value


def get_integer_neighbors(point: Tuple[float, float]) -> Tuple[Tuple[int, int], ...]:
    i_floor, j_floor = int(np.floor(point[0])), int(np.floor(point[1]))
    i_ceil, j_ceil = i_floor + 1, j_floor + 1

    neighbors = [
        (i_floor, j_floor),
        (i_floor, j_ceil),
        (i_ceil, j_floor),
        (i_ceil, j_ceil)
    ]

    return neighbors