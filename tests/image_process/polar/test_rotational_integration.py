import pytest
import numpy as np
import cv2
from edp2pdf.image_process.polar.rotational_integration import RotationalIntegration
from edp2pdf.image_process.polar.polar_representation import PolarRepresentation
from edp2pdf.image_process.diffraction_pattern import eDiffractionPattern

# Mock eDiffractionPattern for testing
class MockEDP(eDiffractionPattern):
    def __init__(self, data, center=None, mask=None):
        super().__init__(data, center, mask)

# Mock PolarRepresentation for testing
class MockPolarRepresentation(PolarRepresentation):
    def __init__(self, edp):
        super().__init__(edp)

@pytest.fixture
def setup_polar_data():
    """Fixture to set up mock polar data for tests."""
    data = np.random.rand(2048, 2048)
    mask = np.zeros_like(data)
    center = (data.shape[0] // 2, data.shape[1] // 2)
    mask = cv2.circle(mask, center, 500, 1, -1)

    edp = MockEDP(data=data, center=center, mask=mask)
    polar_representation = MockPolarRepresentation(edp)

    return polar_representation

def test_rotational_integration_initialization(setup_polar_data):
    """Test RotationalIntegration initialization."""
    polar_representation = setup_polar_data
    rotational_integration = RotationalIntegration(polar_representation)
    assert isinstance(rotational_integration, RotationalIntegration)

    # Test invalid polar_representation type
    with pytest.raises(TypeError):
        RotationalIntegration("invalid_polar_representation")

def test_get_rotational_integration_mean(setup_polar_data):
    """Test rotational integration using mean method."""
    polar_representation = setup_polar_data
    rotational_integration = RotationalIntegration(polar_representation)

    # Test mean integration without mask
    result = rotational_integration.get_rotational_integration(0, 359, method="mean", use_mask=False)
    assert isinstance(result, np.ndarray)
    assert result.shape == polar_representation.radius.shape

    # Test mean integration with mask
    result_masked = rotational_integration.get_rotational_integration(0, 359, method="mean", use_mask=True)
    assert isinstance(result_masked, np.ndarray)
    assert result_masked.shape == polar_representation.radius.shape

def test_get_rotational_integration_median(setup_polar_data):
    """Test rotational integration using median method."""
    polar_representation = setup_polar_data
    rotational_integration = RotationalIntegration(polar_representation)

    # Test median integration without mask
    result = rotational_integration.get_rotational_integration(0, 359, method="median", use_mask=False)
    assert isinstance(result, np.ndarray)
    assert result.shape == polar_representation.radius.shape

    # Test median integration with mask
    result_masked = rotational_integration.get_rotational_integration(0, 359, method="median", use_mask=True)
    assert isinstance(result_masked, np.ndarray)
    assert result_masked.shape == polar_representation.radius.shape

def test_get_rotational_integration_invalid_method(setup_polar_data):
    """Test rotational integration with an invalid method."""
    polar_representation = setup_polar_data
    rotational_integration = RotationalIntegration(polar_representation)

    # Test invalid method
    with pytest.raises(ValueError):
        rotational_integration.get_rotational_integration(0, 359, method="invalid_method")

def test_get_rotational_integration_with_mask(setup_polar_data):
    """Test rotational integration with mask enabled."""
    polar_representation = setup_polar_data
    rotational_integration = RotationalIntegration(polar_representation)

    # Test integration with mask
    result = rotational_integration.get_rotational_integration(0, 359, method="mean", use_mask=True)
    assert isinstance(result, np.ndarray)
    assert result.shape == polar_representation.radius.shape

    # Test integration with mask and median method
    result_median = rotational_integration.get_rotational_integration(0, 359, method="median", use_mask=True)
    assert isinstance(result_median, np.ndarray)
    assert result_median.shape == polar_representation.radius.shape

def test_get_rotational_integration_partial_range(setup_polar_data):
    """Test rotational integration over a partial angular range."""
    polar_representation = setup_polar_data
    rotational_integration = RotationalIntegration(polar_representation)

    # Test integration over a partial angular range
    result = rotational_integration.get_rotational_integration(45, 135, method="mean", use_mask=False)
    assert isinstance(result, np.ndarray)
    assert result.shape == polar_representation.radius.shape

    # Test integration over a partial angular range with mask
    result_masked = rotational_integration.get_rotational_integration(45, 135, method="mean", use_mask=True)
    assert isinstance(result_masked, np.ndarray)
    assert result_masked.shape == polar_representation.radius.shape

def test_get_rotational_integration_partial_radius(setup_polar_data):
    """Test rotational integration over a partial radius range."""
    polar_representation = setup_polar_data
    assert polar_representation.polar_image.shape == polar_representation.polar_mask.shape


    polar_representation.radial_range = (0.1, 0.9)
    rotational_integration = RotationalIntegration(polar_representation)

    # Test integration over a partial angular range
    result = rotational_integration.get_rotational_integration(0, 359, method="mean", use_mask=False)
    assert isinstance(result, np.ndarray)
    assert result.shape == polar_representation.radius.shape

    # Test integration over a partial angular range with mask
    result_masked = rotational_integration.get_rotational_integration(0, 359, method="mean", use_mask=True)
    assert isinstance(result_masked, np.ndarray)
    assert result_masked.shape == polar_representation.radius.shape



