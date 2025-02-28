import pytest
import numpy as np
from typing import Tuple

from edp2pdf.image_process.diffraction_pattern import eDiffractionPattern

# Helper function to create a valid diffraction pattern
def create_valid_diffraction_pattern():
    data = np.random.rand(10, 10)
    center = (5, 5)
    mask = np.zeros((10, 10), dtype=bool)
    return data, center, mask

# Test valid initialization
def test_valid_initialization():
    data, center, mask = create_valid_diffraction_pattern()
    pattern = eDiffractionPattern(data, center, mask)
    
    assert np.array_equal(pattern.data, data)
    assert pattern.center == center
    assert np.array_equal(pattern.mask, mask)

# Test invalid data type for data
def test_invalid_data_type():
    data = [[1, 2], [3, 4]]  # Not a numpy array
    center = (1, 1)
    mask = np.zeros((2, 2), dtype=bool)
    
    with pytest.raises(TypeError):
        eDiffractionPattern(data, center, mask)

# Test invalid data type for mask
def test_invalid_mask_type():
    data = np.random.rand(2, 2)
    center = (1, 1)
    mask = [[True, False], [False, True]]  # Not a numpy array
    
    with pytest.raises(TypeError):
        eDiffractionPattern(data, center, mask)

# Test invalid data shape
def test_invalid_data_shape():
    data = np.random.rand(2, 2, 2)  # 3D array
    center = (1, 1)
    mask = np.zeros((2, 2), dtype=bool)
    
    with pytest.raises(ValueError):
        eDiffractionPattern(data, center, mask)

# Test invalid mask shape
def test_invalid_mask_shape():
    data = np.random.rand(2, 2)
    center = (1, 1)
    mask = np.zeros((3, 3), dtype=bool)  # Shape mismatch
    
    with pytest.raises(ValueError):
        eDiffractionPattern(data, center, mask)

# Test out-of-bounds center coordinates
def test_out_of_bounds_center():
    data = np.random.rand(2, 2)
    center = (3, 3)  # Out of bounds
    mask = np.zeros((2, 2), dtype=bool)
    
    with pytest.raises(ValueError):
        eDiffractionPattern(data, center, mask)

# Test mask property and setter
def test_mask_property_and_setter():
    data, center, mask = create_valid_diffraction_pattern()
    pattern = eDiffractionPattern(data, center, mask)
    
    # Test getting the mask
    assert np.array_equal(pattern.mask, mask)
    
    # Test setting a new valid mask
    new_mask = np.ones((10, 10), dtype=bool)
    pattern.mask = new_mask
    assert np.array_equal(pattern.mask, new_mask)
    
    # Test setting an invalid mask (wrong shape)
    invalid_mask = np.ones((5, 5), dtype=bool)
    with pytest.raises(ValueError):
        pattern.mask = invalid_mask
    
    # Test setting an invalid mask (wrong type)
    invalid_mask = [[True, False], [False, True]]
    with pytest.raises(TypeError):
        pattern.mask = invalid_mask

# Run the tests
if __name__ == "__main__":
    pytest.main()