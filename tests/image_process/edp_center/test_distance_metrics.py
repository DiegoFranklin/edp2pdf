import pytest
import numpy as np
from edp2pdf.image_process.edp_center.center_optimization.distance_metrics import (
    manhattan,
    mean_squared_error,
    cosine,
    jaccard_distance,
    masked_metric_factory,
)

@pytest.fixture
def setup_arrays():
    a = np.random.rand(10, 10)
    b = np.random.rand(10, 10)
    mask = np.random.randint(0, 2, (10, 10))
    return a, b, mask

def test_manhattan(setup_arrays):
    a, b, mask = setup_arrays
    distance = manhattan(a, b, mask)

    # Test return type
    assert isinstance(distance, float)

    # Test shape mismatch
    with pytest.raises(ValueError):
        manhattan(a, b, np.random.rand(5, 5))

def test_mean_squared_error(setup_arrays):
    a, b, mask = setup_arrays
    mse = mean_squared_error(a, b, mask)

    # Test return type
    assert isinstance(mse, float)

    # Test shape mismatch
    with pytest.raises(ValueError):
        mean_squared_error(a, b, np.random.rand(5, 5))

def test_cosine(setup_arrays):
    a, b, mask = setup_arrays
    cos_dist = cosine(a, b, mask)

    # Test return type
    assert isinstance(cos_dist, float)

    # Test shape mismatch
    with pytest.raises(ValueError):
        cosine(a, b, np.random.rand(5, 5))

def test_jaccard_distance(setup_arrays):
    a, b, mask = setup_arrays
    jaccard_dist = jaccard_distance(a, b, mask)

    # Test return type
    assert isinstance(jaccard_dist, float)

    # Test shape mismatch
    with pytest.raises(ValueError):
        jaccard_distance(a, b, np.random.rand(5, 5))

def test_masked_metric_factory():
    # Test valid metric
    metric = masked_metric_factory("manhattan")
    assert callable(metric)

    # Test invalid metric
    with pytest.raises(ValueError):
        masked_metric_factory("invalid_metric")