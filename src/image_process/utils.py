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

def distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    # Convert tuples to numpy arrays
    p1_np = np.array(p1)
    p2_np = np.array(p2)
    
    # Calculate the Euclidean distance
    return np.linalg.norm(p1_np - p2_np)