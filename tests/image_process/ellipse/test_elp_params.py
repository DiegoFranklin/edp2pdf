import pytest
import numpy as np
from edp2pdf.image_process.ellipse.elp_params import EllipseParams
from edp2pdf.image_process.diffraction_pattern import eDiffractionPattern

# Mock eDiffractionPattern for testing
class MockEDP(eDiffractionPattern):
    def __init__(self, data):
        self.data = data
        self._mask = np.ones_like(data)
        self.center = (data.shape[0] // 2, data.shape[1] // 2)

@pytest.fixture
def setup_data():
    data = np.random.rand(100, 100)
    return data

def test_ellipse_params_initialization(setup_data):
    edp = MockEDP(setup_data)
    ellipse_params = EllipseParams(edp)
    assert isinstance(ellipse_params, EllipseParams)

    # Test invalid edp type
    with pytest.raises(TypeError):
        EllipseParams(setup_data)

def test_measure_ellipse_params(setup_data):
    edp = MockEDP(setup_data)
    ellipse_params = EllipseParams(edp)
    params = ellipse_params.measure_ellipse_params()

    # Test return type
    assert isinstance(params, dict)

    # Test required keys
    assert "axis_ratio" in params
    assert "orientation" in params

    # Test valid values
    assert isinstance(params["axis_ratio"], float)
    assert isinstance(params["orientation"], float)