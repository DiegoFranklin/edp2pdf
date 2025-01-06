import numpy as np

def get_centroid(image):
    """
    Calculates the centroid of a grayscale image using pixel intensity as mass.
    
    Args:
        image (numpy.ndarray): 2D array representing the grayscale image.
        
    Returns:
        tuple: The (x, y) coordinates of the centroid.
    """
    # Ensure the input is a 2D array
    if len(image.shape) != 2:
        raise ValueError("Input must be a 2D array representing a grayscale image.")
    
    # Get the dimensions of the image
    height, width = image.shape
    
    # Create coordinate grids
    x_coords = np.arange(width)
    y_coords = np.arange(height)
    
    # Compute the total mass (sum of pixel intensities)
    total_mass = np.sum(image)
    
    if total_mass == 0:
        raise ValueError("Total mass (sum of pixel intensities) is zero, cannot compute centroid.")
    
    # Calculate the weighted coordinates
    x_centroid = np.sum(image * x_coords[None, :]) / total_mass
    y_centroid = np.sum(image * y_coords[:, None]) / total_mass
    
    return x_centroid, y_centroid
