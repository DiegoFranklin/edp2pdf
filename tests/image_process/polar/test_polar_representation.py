from edp2pdf.image_process.polar.polar_representation import PolarRepresentation
from edp2pdf.image_process.diffraction_pattern import eDiffractionPattern
from edp2pdf.image_process.polar.polar_transformation import CVPolarTransformation

import numpy as np
import pytest
import cv2
from unittest.mock import MagicMock
import logging


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


@pytest.fixture
def mock_edp():
    """Mock for eDiffractionPattern"""
    mock = MagicMock(spec=eDiffractionPattern)
    mock.data = np.ones((2048, 2048))
    mock.center = (1024, 1024)
    mock.mask = np.zeros((2048, 2048))
    mock.mask = cv2.circle(mock.mask, mock.center, 500, 1, -1)
    return mock

# ====================== Existing tests remain unchanged ======================

def test_initialization(mock_edp):
    edp = mock_edp
    polar_representation = PolarRepresentation(edp=edp)

    assert isinstance(polar_representation._polar_transformer, CVPolarTransformation)

    assert polar_representation._relative_radial_range[0] == 0
    assert polar_representation._relative_radial_range[1] == 1
    assert polar_representation._angular_range[0] == 0
    assert polar_representation._angular_range[1] == 359

    assert polar_representation._polar_image is None
    assert polar_representation._radius_space is None
    assert polar_representation._theta_space is None
    assert polar_representation._angular_mask is None


def test_invalid_initialization(mock_edp):
    """Test exceptions on invalid radial/angle values."""
    with pytest.raises(ValueError):
        PolarRepresentation(edp=mock_edp, radial_range=(-0.1, 1))
    with pytest.raises(ValueError):
        PolarRepresentation(edp=mock_edp, radial_range=(0, 1.1))
    with pytest.raises(ValueError):
        PolarRepresentation(edp=mock_edp, radial_range=(0.8, 0,7))


def test_radius_property(mock_edp):
    """Test radius property"""
    polar_representation = PolarRepresentation(edp=mock_edp)

    # Initially, _radius_space should be None
    assert polar_representation._radius_space is None

    # Accessing the 'radius' property should trigger the computation
    radius = polar_representation.radius

    # Check that the radius property now contains an array
    assert radius is not None
    assert isinstance(radius, np.ndarray)

    # Check that _radius_space is populated after accessing the property
    assert polar_representation._radius_space is not None

    # The radius should correspond to the full range of polar image's radial dimension
    assert radius[0] == 0
    assert radius[-1] == polar_representation._full_polar_image.shape[1] - 1


def test_full_attr_behavior(mock_edp):
    """Test full attribute behavior"""
    polar_representation = PolarRepresentation(edp=mock_edp)

    # Attributes should be None initially
    assert polar_representation._full_polar_image is None
    assert polar_representation._full_radius_space is None
    assert polar_representation._full_theta_space is None

    # Call computing methods and check if they're populated
    polar_representation._compute_full_polar_image()
    polar_representation._compute_full_radius_space()
    polar_representation._compute_full_theta_space()

    assert polar_representation._full_polar_image is not None
    assert polar_representation._full_radius_space is not None
    assert polar_representation._full_theta_space is not None

    # Check types
    assert isinstance(polar_representation._full_polar_image, np.ndarray)
    assert isinstance(polar_representation._full_radius_space, np.ndarray)
    assert isinstance(polar_representation._full_theta_space, np.ndarray)

    # Check dimensions and edge values consistency
    assert polar_representation._full_radius_space[0] == 0
    assert polar_representation._full_theta_space[0] == 0

    assert polar_representation._full_radius_space.shape[0] == polar_representation._full_polar_image.shape[1]
    assert polar_representation._full_theta_space.shape[0] == polar_representation._full_polar_image.shape[0]

    assert polar_representation._full_radius_space[-1] == polar_representation._full_radius_space.shape[0] - 1
    assert polar_representation._full_theta_space[-1] < 360


def test_radial_index_computation(mock_edp):
    """Test radial index behavior"""
    polar_representation = PolarRepresentation(edp=mock_edp)

    assert polar_representation._radial_indices[0] is None
    assert polar_representation._radial_indices[1] is None

    polar_representation._compute_radial_index()

    # Verify if the attributes are populated
    assert polar_representation._radial_indices[0] is not None
    assert polar_representation._radial_indices[1] is not None

    # Check types
    assert isinstance(polar_representation._radial_indices[0], int)
    assert isinstance(polar_representation._radial_indices[1], int)

    # Check they don't exceed bounds
    assert polar_representation._radial_indices[0] >= 0
    assert polar_representation._radial_indices[1] <= polar_representation._full_radius_space.shape[0]


def test_angular_index_computation(mock_edp):
    """Test angular index behavior"""
    polar_representation = PolarRepresentation(edp=mock_edp)

    assert polar_representation._angular_indices[0] is None
    assert polar_representation._angular_indices[1] is None

    polar_representation._compute_angular_index()

    # Verify if the attributes are populated
    assert polar_representation._angular_indices[0] is not None
    assert polar_representation._angular_indices[1] is not None

    # Check types
    assert isinstance(polar_representation._angular_indices[0], int)
    assert isinstance(polar_representation._angular_indices[1], int)

    # Check they don't exceed bounds
    assert polar_representation._angular_indices[0] >= 0
    assert polar_representation._angular_indices[1] < polar_representation._full_theta_space.shape[0]


def test_polar_image_computation(mock_edp):
    """Test polar image computation"""
    polar_representation = PolarRepresentation(edp=mock_edp)

    # Initially, the polar image should be None
    assert polar_representation._polar_image is None

    # After calling the computation method, it should be populated
    polar_representation._compute_polar_image()
    assert polar_representation._polar_image is not None
    assert isinstance(polar_representation._polar_image, np.ndarray)


def test_angular_mask_computation(mock_edp):
    """Test angular mask computation"""
    polar_representation = PolarRepresentation(edp=mock_edp)

    # Initially, the angular mask should be None
    assert polar_representation._angular_mask is None

    # After calling the computation method, it should be populated
    polar_representation._compute_angular_mask()
    assert polar_representation._angular_mask is not None
    assert isinstance(polar_representation._angular_mask, np.ndarray)


def test_radial_range_setter(mock_edp):
    """Test the radial_range setter behavior"""
    polar_representation = PolarRepresentation(edp=mock_edp)

    polar_representation.radial_range = (0.1, 0.9)
    assert polar_representation._relative_radial_range[0] == 0.1
    assert polar_representation._relative_radial_range[1] == 0.9

    with pytest.raises(ValueError):
        polar_representation.radial_range = (-0.1, 0.9)
    with pytest.raises(ValueError):
        polar_representation.radial_range = (0.9, 1.1)
    with pytest.raises(ValueError):
        polar_representation.radial_range = (0.9, 0.8)


def test_angular_range_setter(mock_edp):
    """Test the angular_range setter behavior"""
    polar_representation = PolarRepresentation(edp=mock_edp)

    polar_representation.angular_range = (10, 350)
    assert polar_representation._angular_range[0] == 10
    assert polar_representation._angular_range[1] == 350


def test_theta_property(mock_edp):
    """Test theta property"""
    polar_representation = PolarRepresentation(edp=mock_edp)

    # Initially, _theta_space should be None
    assert polar_representation._theta_space is None

    # Accessing the 'theta' property should trigger the computation
    theta = polar_representation.theta

    # Check that the theta property now contains an array
    assert theta is not None
    assert isinstance(theta, np.ndarray)

    # Check that _theta_space is populated after accessing the property
    assert polar_representation._theta_space is not None

    # The theta space should be between 0 and 360 degrees (non-inclusive)
    assert theta[0] == 0
    assert theta[-1] < 360


def test_polar_image_property(mock_edp):
    """Test polar_image property"""
    polar_representation = PolarRepresentation(edp=mock_edp)

    # Initially, _polar_image should be None
    assert polar_representation._polar_image is None

    # Accessing the 'polar_image' property should trigger the computation
    polar_image = polar_representation.polar_image

    # Check that the polar image property now contains an array
    assert polar_image is not None
    assert isinstance(polar_image, np.ndarray)

    # Check that _polar_image is populated after accessing the property
    assert polar_representation._polar_image is not None

    # The polar image shape should match the expected radial and angular dimensions
    assert polar_image.shape[0] == polar_representation.theta.shape[0]
    assert polar_image.shape[1] == polar_representation.radius.shape[0]

# Test radial range adjustment
def test_radial_range_adjustment_berfore_and_after(mock_edp):
    polar_representation = PolarRepresentation(edp=mock_edp)
    assert polar_representation.polar_image.shape == polar_representation.polar_mask.shape

    polar_representation.radial_range = (0.1, 0.8)
    assert polar_representation.polar_image.shape == polar_representation.polar_mask.shape

def test_radial_range_adjustment_after(mock_edp):
    polar_representation = PolarRepresentation(edp=mock_edp)

    polar_representation.radial_range = (0.1, 0.8)
    assert polar_representation.polar_image.shape == polar_representation.polar_mask.shape