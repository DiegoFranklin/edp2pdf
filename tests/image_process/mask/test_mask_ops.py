import pytest
import numpy as np
from skimage.morphology import disk
from edp2pdf.image_process.mask.mask_ops import (
    MaskOperation,
    Dilate,
    Erode,
    OCRoutine,
)

# Fixtures
@pytest.fixture
def binary_mask():
    """Fixture to create a binary mask for testing."""
    return np.random.randint(0, 2, (100, 100), dtype=np.uint8)

# Tests for MaskOperation (Abstract Base Class)
def test_mask_operation_abstract():
    """Test that MaskOperation cannot be instantiated directly."""
    with pytest.raises(TypeError):
        mask_op = MaskOperation()

# Tests for Dilate
def test_dilate_initialization():
    """Test initialization of the Dilate class."""
    dilate = Dilate(iterations=2, kernel_size=3)
    assert isinstance(dilate, Dilate)
    assert dilate.iterations == 2
    assert dilate._kernel.shape == disk(3).shape

    # Test invalid iterations
    with pytest.raises(ValueError):
        Dilate(iterations=0)
    # Test invalid kernel size
    with pytest.raises(ValueError):
        Dilate(kernel_size=-1)

def test_dilate_operate(binary_mask):
    """Test the operate method of the Dilate class."""
    dilate = Dilate(iterations=2, kernel_size=3)
    result = dilate.operate(binary_mask)
    assert isinstance(result, np.ndarray)
    assert result.shape == binary_mask.shape
    assert result.dtype == np.uint8

    # Test invalid input type
    with pytest.raises(TypeError):
        dilate.operate("invalid_mask")
    # Test invalid input dimensions
    with pytest.raises(ValueError):
        dilate.operate(np.random.rand(100, 100, 3))  # 3D array

# Tests for Erode
def test_erode_initialization():
    """Test initialization of the Erode class."""
    erode = Erode(iterations=2, kernel_size=3)
    assert isinstance(erode, Erode)
    assert erode.iterations == 2
    assert erode._kernel.shape == disk(3).shape

    # Test invalid iterations
    with pytest.raises(ValueError):
        Erode(iterations=0)
    # Test invalid kernel size
    with pytest.raises(ValueError):
        Erode(kernel_size=-1)

def test_erode_operate(binary_mask):
    """Test the operate method of the Erode class."""
    erode = Erode(iterations=2, kernel_size=3)
    result = erode.operate(binary_mask)
    assert isinstance(result, np.ndarray)
    assert result.shape == binary_mask.shape
    assert result.dtype == np.uint8

    # Test invalid input type
    with pytest.raises(TypeError):
        erode.operate("invalid_mask")
    # Test invalid input dimensions
    with pytest.raises(ValueError):
        erode.operate(np.random.rand(100, 100, 3))  # 3D array

# Tests for OCRoutine
def test_oc_routine_initialization():
    """Test initialization of the OCRoutine class."""
    dilate = Dilate(iterations=1)
    erode = Erode(iterations=1)
    oc_routine = OCRoutine(routine=[dilate, erode])
    assert isinstance(oc_routine, OCRoutine)
    assert len(oc_routine._routine) == 2

    # Test invalid routine (non-MaskOperation objects)
    with pytest.raises(TypeError):
        OCRoutine(routine=["invalid_operation"])

def test_oc_routine_execute_routine(binary_mask):
    """Test the execute_routine method of the OCRoutine class."""
    dilate = Dilate(iterations=1)
    erode = Erode(iterations=1)
    oc_routine = OCRoutine(routine=[dilate, erode])
    result = oc_routine.execute_routine(binary_mask)
    assert isinstance(result, np.ndarray)
    assert result.shape == binary_mask.shape
    assert result.dtype == np.uint8

    # Test invalid input type
    with pytest.raises(TypeError):
        oc_routine.execute_routine("invalid_mask")
    # Test invalid input dimensions
    with pytest.raises(ValueError):
        oc_routine.execute_routine(np.random.rand(100, 100, 3))  # 3D array

def test_oc_routine_empty_routine(binary_mask):
    """Test OCRoutine with an empty routine."""
    oc_routine = OCRoutine(routine=[])
    result = oc_routine.execute_routine(binary_mask)
    assert np.array_equal(result, binary_mask)  # No changes to the input mask