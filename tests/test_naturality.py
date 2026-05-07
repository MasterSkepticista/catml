import numpy as np

from catml import eval_program, grad_program, shape_program, trace_function
from catml import linear, relu, sum_all


def test_primal_projection_commutes():
    rng = np.random.default_rng(0)
    dim = 3

    def model(x):
        return sum_all(relu(linear(x, dim, rng, tag="a")))

    x = rng.standard_normal((dim,))
    program = trace_function(model, x)
    y = eval_program(program, (x,))
    y2, _ = grad_program(program, (x,), np.array([1.0]))

    assert np.allclose(y, y2)


def test_shape_naturality_commutes():
    rng = np.random.default_rng(1)
    dim = 4

    def model(x):
        return sum_all(relu(linear(x, dim, rng, tag="a")))

    x = rng.standard_normal((dim,))
    program = trace_function(model, x)

    y = eval_program(program, (x,))
    shape_from_eval = _shape_of(y)
    shape_from_shape = shape_program(program, (x.shape,))

    assert shape_from_eval == shape_from_shape


def test_shape_naturality_product_commutes():
    rng = np.random.default_rng(2)
    dim = 5

    def model(x):
        left = relu(linear(x, dim, rng, tag="a"))
        right = sum_all(x)
        return (left, right)

    x = rng.standard_normal((dim,))
    program = trace_function(model, x)

    y = eval_program(program, (x,))
    shape_from_eval = _shape_of(y)
    shape_from_shape = shape_program(program, (x.shape,))

    assert shape_from_eval == shape_from_shape


def _shape_of(value):
    if isinstance(value, tuple):
        return tuple(_shape_of(item) for item in value)
    if hasattr(value, "shape"):
        return tuple(value.shape)
    raise TypeError("Cannot infer shape from value")
