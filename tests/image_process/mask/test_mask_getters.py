# File: tests/image_process/mask/test_mask_getters.py
import pytest
import numpy as np
from edp2pdf.image_process.mask.mask_getters import (
    MaskGetter,
    RecursiveMask,
    GaussianBlurTreshMask,
    AdaptiveTreshMask,
    MeanTreshMask,
    superpose_masks,
)

# Fixtures
@pytest.fixture
def sample_data():
    return np.random.rand(100, 100)

# Tests for MaskGetter and its subclasses
def test_mask_getter_abstract():
    with pytest.raises(TypeError):
        mask_getter = MaskGetter()

def test_recursive_mask(sample_data):
    recursive_mask = RecursiveMask()
    mask = recursive_mask.get_mask(sample_data)
    assert isinstance(mask, np.ndarray)
    assert mask.shape == sample_data.shape
    assert mask.dtype == np.uint8

    # Test invalid input
    with pytest.raises(TypeError):
        recursive_mask.get_mask("invalid_data")
    with pytest.raises(ValueError):
        recursive_mask.get_mask(np.array([]))

def test_gaussian_blur_treshg_mask(sample_data):
    gaussian_mask = GaussianBlurTreshMask(sigma=20, iterations=10)
    mask = gaussian_mask.get_mask(sample_data)
    assert isinstance(mask, np.ndarray)
    assert mask.shape == sample_data.shape
    assert mask.dtype == np.uint8

    # Test invalid sigma and iterations
    with pytest.raises(ValueError):
        GaussianBlurTreshMask(sigma=-1)
    with pytest.raises(ValueError):
        GaussianBlurTreshMask(iterations=0)

def test_adaptive_tresh_mask(sample_data):
    adaptive_mask = AdaptiveTreshMask(sector_size=101, c_param=0)
    mask = adaptive_mask.get_mask(sample_data)
    assert isinstance(mask, np.ndarray)
    assert mask.shape == sample_data.shape
    assert mask.dtype == np.uint8

    # Test invalid sector size
    with pytest.raises(ValueError):
        AdaptiveTreshMask(sector_size=100)  # Even number
    with pytest.raises(ValueError):
        AdaptiveTreshMask(sector_size=-1)  # Negative number

def test_mean_tresh_mask(sample_data):
    mean_mask = MeanTreshMask(constant=0.2)
    mask = mean_mask.get_mask(sample_data)
    assert isinstance(mask, np.ndarray)
    assert mask.shape == sample_data.shape
    assert mask.dtype == np.uint8

    # Test invalid constant
    with pytest.raises(ValueError):
        MeanTreshMask(constant=-0.1)

def test_superpose_masks(sample_data):
    mask_getters = [MeanTreshMask(), GaussianBlurTreshMask()]
    mask = superpose_masks(sample_data, mask_getters)
    assert isinstance(mask, np.ndarray)
    assert mask.shape == sample_data.shape
    assert mask.dtype == np.uint8

    # Test empty mask list
    with pytest.raises(ValueError):
        superpose_masks(sample_data, [])
    # Test invalid mask getter
    with pytest.raises(TypeError):
        superpose_masks(sample_data, ["invalid_mask_getter"])