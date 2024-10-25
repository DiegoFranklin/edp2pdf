from src.image_process.polar.polar_representation import PolarRepresentation
from src.image_process.diffraction_pattern import eDiffractionPattern
from src.image_process.polar.polar_transformation import CVPolarTransformation
from src.image_process.mask.angular_mask import MeanAngularMask

import numpy as np
import pytest
from unittest.mock import MagicMock
import logging


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


@pytest.fixture
def mock_edp():
    """Mock for eDiffractionPattern"""
    mock = MagicMock(spec=eDiffractionPattern)
    mock.data = np.ones((2048, 2048))
    mock.center = (1024, 1024)
    return mock

# ====================== Existing tests remain unchanged ======================

def test_initialization(mock_edp):
    edp = mock_edp
    polar_representation = PolarRepresentation(edp=edp)

    assert isinstance(polar_representation._polar_transformer, CVPolarTransformation)
    assert isinstance(polar_representation._angular_mask_getter, MeanAngularMask)

    assert polar_representation._relative_radial_start == 0
    assert polar_representation._relative_radial_end == 1
    assert polar_representation._start_angle == 0
    assert polar_representation._end_angle == 359

    assert polar_representation._polar_image is None
    assert polar_representation._radius_space is None
    assert polar_representation._theta_space is None
    assert polar_representation._angular_mask is None


def test_invalid_initialization(mock_edp):
    """Test exceptions on invalid radial/angle values."""
    with pytest.raises(ValueError):
        PolarRepresentation(edp=mock_edp, relative_radial_start=-0.1)
    with pytest.raises(ValueError):
        PolarRepresentation(edp=mock_edp, relative_radial_end=1.1)
    with pytest.raises(ValueError):
        PolarRepresentation(edp=mock_edp, relative_radial_start=0.8, relative_radial_end=0.7)


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


def test_invalid_initialization(mock_edp):
    """Test exceptions on invalid radial/angle values."""
    with pytest.raises(ValueError):
        PolarRepresentation(edp=mock_edp, relative_radial_start=-0.1)
    with pytest.raises(ValueError):
        PolarRepresentation(edp=mock_edp, relative_radial_end=1.1)
    with pytest.raises(ValueError):
        PolarRepresentation(edp=mock_edp, relative_radial_start=0.8, relative_radial_end=0.7)


def test_full_attr_behaviour(mock_edp):
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

    assert polar_representation._start_radial_index is None
    assert polar_representation._end_radial_index is None

    polar_representation._compute_radial_index()

    # Verify if the attributes are populated
    assert polar_representation._start_radial_index is not None
    assert polar_representation._end_radial_index is not None

    # Check types
    assert isinstance(polar_representation._start_radial_index, int)
    assert isinstance(polar_representation._end_radial_index, int)

    # Check they don't exceed bounds
    assert polar_representation._start_radial_index >= 0
    assert polar_representation._end_radial_index <= polar_representation._full_radius_space.shape[0]


def test_angular_index_computation(mock_edp):
    """Test angular index behavior"""
    polar_representation = PolarRepresentation(edp=mock_edp)

    assert polar_representation._start_angle_index is None
    assert polar_representation._end_angle_index is None

    polar_representation._compute_angular_index()

    # Verify if the attributes are populated
    assert polar_representation._start_angle_index is not None
    assert polar_representation._end_angle_index is not None

    # Check types
    assert isinstance(polar_representation._start_angle_index, int)
    assert isinstance(polar_representation._end_angle_index, int)

    # Check they don't exceed bounds
    assert polar_representation._start_angle_index >= 0
    assert polar_representation._end_angle_index < polar_representation._full_theta_space.shape[0]


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
    assert polar_representation._relative_radial_start == 0.1
    assert polar_representation._relative_radial_end == 0.9

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
    assert polar_representation._start_angle == 10
    assert polar_representation._end_angle == 350


def test_angular_mask_params(mock_edp):
    """Test setting the angular mask parameters"""
    polar_representation = PolarRepresentation(edp=mock_edp)

    polar_representation.set_angular_mask_params(cyclic_shift=10, angular_range_expansion=5)
    assert isinstance(polar_representation._angular_mask_getter, MeanAngularMask)


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

