import numpy as np

def get_centroid(image):

    if len(image.shape) != 2:
        raise ValueError("Input must be a 2D grayscale image.")

    height, width = image.shape

    y_coords, x_coords = np.indices((height, width))

    M = np.sum(image)

    if M == 0:
        return None

    x_com = np.sum(x_coords * image) / M
    y_com = np.sum(y_coords * image) / M

    return (x_com, y_com)
