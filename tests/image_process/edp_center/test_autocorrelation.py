import pytest
import numpy as np
from edp2pdf.image_process.edp_center.autocorrelation import AutoCorrelation

@pytest.fixture
def setup_data():
    data = np.random.rand(100, 100)
    mask = np.random.randint(0, 2, (100, 100))
    return data, mask

def test_initialization(setup_data):
    data, mask = setup_data

    # Test valid initialization
    ac = AutoCorrelation(data, mask)
    assert isinstance(ac, AutoCorrelation)

    # Test invalid data type
    with pytest.raises(TypeError):
        AutoCorrelation([1, 2, 3], mask)

    # Test invalid mask type
    with pytest.raises(TypeError):
        AutoCorrelation(data, [1, 2, 3])

    # Test shape mismatch
    with pytest.raises(ValueError):
        AutoCorrelation(data, np.random.rand(50, 50))

def test_compute(setup_data):
    data, mask = setup_data
    ac = AutoCorrelation(data, mask)
    center, autocorr = ac.compute()

    # Test return types
    assert isinstance(center, tuple)
    assert isinstance(autocorr, np.ndarray)

    # Test center coordinates are within bounds
    assert 0 <= center[0] < data.shape[0]
    assert 0 <= center[1] < data.shape[1]

    # Test autocorrelation shape
    assert autocorr.shape == data.shape