import pytest
import numpy as np
from edp2pdf.image_process.ellipse.elp_correction import (
    TransformationStep,
    Pad,
    ReversePad,
    AffineTransformation,
    Rotate,
    Scale,
    ComposedTransform,
    CorrectionPipeline,
    correct_ellipse,
)
from edp2pdf.image_process.diffraction_pattern import eDiffractionPattern

# Mock eDiffractionPattern for testing
class MockEDP(eDiffractionPattern):
    def __init__(self, data, center):
        self.data = data
        self.center = center

@pytest.fixture
def setup_data():
    data = np.random.rand(100, 100)
    center = (50, 50)
    return data, center

def test_transformation_step_abstract():
    with pytest.raises(TypeError):
        step = TransformationStep()

def test_pad_initialization(setup_data):
    data, center = setup_data
    pad_step = Pad(center)
    assert isinstance(pad_step, Pad)

    # Test invalid center type
    with pytest.raises(TypeError):
        Pad([50, 50])

def test_pad_execute(setup_data):
    data, center = setup_data
    pad_step = Pad(center)
    padded_data = pad_step.execute(data)

    # Test return type
    assert isinstance(padded_data, np.ndarray)

    # Test invalid input type
    with pytest.raises(TypeError):
        pad_step.execute([1, 2, 3])

    # Test invalid input shape
    with pytest.raises(ValueError):
        pad_step.execute(np.random.rand(100, 100, 3))

def test_reverse_pad_execute(setup_data):
    data, center = setup_data
    pad_step = Pad(center)
    padded_data = pad_step.execute(data)

    reverse_pad_step = ReversePad()
    recovered_data = reverse_pad_step.execute(padded_data)

    # Test return type
    assert isinstance(recovered_data, np.ndarray)

    # Test invalid input type
    with pytest.raises(TypeError):
        reverse_pad_step.execute([1, 2, 3])

    # Test invalid input shape
    with pytest.raises(ValueError):
        reverse_pad_step.execute(np.random.rand(100, 100, 3))

def test_affine_transformation_initialization():
    matrix = np.eye(3)
    affine_step = AffineTransformation(matrix)
    assert isinstance(affine_step, AffineTransformation)

    # Test invalid matrix type
    with pytest.raises(TypeError):
        AffineTransformation([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    # Test invalid matrix shape
    with pytest.raises(ValueError):
        AffineTransformation(np.eye(2))

def test_affine_transformation_execute(setup_data):
    data, _ = setup_data
    matrix = np.eye(3)
    affine_step = AffineTransformation(matrix)
    transformed_data = affine_step.execute(data)

    # Test return type
    assert isinstance(transformed_data, np.ndarray)

    # Test invalid input type
    with pytest.raises(TypeError):
        affine_step.execute([1, 2, 3])

    # Test invalid input shape
    with pytest.raises(ValueError):
        affine_step.execute(np.random.rand(100, 100, 3))

def test_rotate_initialization():
    rotate_step = Rotate(45)
    assert isinstance(rotate_step, Rotate)

    # Test invalid angle type
    with pytest.raises(TypeError):
        Rotate("45")

def test_scale_initialization():
    scale_step = Scale(1.5)
    assert isinstance(scale_step, Scale)

    # Test invalid axis_ratio type
    with pytest.raises(TypeError):
        Scale("1.5")

    # Test invalid axis_ratio value
    with pytest.raises(ValueError):
        Scale(-1.5)

def test_composed_transform_initialization():
    rotate_step = Rotate(45)
    scale_step = Scale(1.5)
    composed_step = ComposedTransform([rotate_step, scale_step])
    assert isinstance(composed_step, ComposedTransform)

    # Test invalid transformations type
    with pytest.raises(TypeError):
        ComposedTransform([rotate_step, "scale_step"])

def test_correction_pipeline(setup_data):
    data, center = setup_data
    pipeline = CorrectionPipeline()
    pipeline.add_step(Pad(center))
    pipeline.add_step(Rotate(45))
    pipeline.add_step(ReversePad())

    transformed_data = pipeline.execute(data)

    # Test return type
    assert isinstance(transformed_data, np.ndarray)

    # Test invalid step type
    with pytest.raises(TypeError):
        pipeline.add_step("invalid_step")

def test_correct_ellipse(setup_data):
    data, center = setup_data
    edp = MockEDP(data, center)
    ellipse_params = {"orientation": 45, "axis_ratio": 1.5}

    corrected_data = correct_ellipse(edp, ellipse_params)

    # Test return type
    assert isinstance(corrected_data, np.ndarray)

    # Test invalid edp type
    with pytest.raises(TypeError):
        correct_ellipse(data, ellipse_params)

    # Test invalid ellipse_params type
    with pytest.raises(TypeError):
        correct_ellipse(edp, [45, 1.5])

    # Test missing ellipse_params keys
    with pytest.raises(ValueError):
        correct_ellipse(edp, {"orientation": 45})

    # Test invalid ellipse_params values
    with pytest.raises(ValueError):
        correct_ellipse(edp, {"orientation": "45", "axis_ratio": 1.5})