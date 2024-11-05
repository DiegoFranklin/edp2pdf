from typing import Tuple
import numpy as np

def get_centered_crop_img(image, center):
    # crops the image to biggest area possible so its get centered on center
    c_x = round(center[1])
    c_y = round(center[0])

    x_lenth = image.shape[0]
    y_lenth = image.shape[1]

    radius = min(c_x, c_y, abs(x_lenth-c_x), abs(y_lenth-c_y))

    return image[c_x-radius:c_x+radius, c_y-radius:c_y+radius]

def distance(a, b):
    """
    Calculates the Euclidean distance between two points.

    Args:
        a: A numpy array or tuple representing the first point.
        b: A numpy array or tuple representing the second point.

    Returns:
        The Euclidean distance between point a and point b.
    """
    c = np.subtract(a, b)

    return np.linalg.norm(c)