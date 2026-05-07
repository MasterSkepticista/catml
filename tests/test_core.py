import numpy as np

from catml import param_count_program, trace_function
from catml import linear, relu, sum_all


def test_trace_builds_program():
    rng = np.random.default_rng(0)
    dim = 2

    def model(x):
        return sum_all(relu(linear(x, dim, rng, tag="a")))

    x = rng.standard_normal((dim,))
    program = trace_function(model, x)
    assert len(program.instructions) == 3


def test_param_count():
    rng = np.random.default_rng(1)

    def model(x):
        y = linear(x, 3, rng, tag="a")
        y = linear(y, 2, rng, tag="b")
        return sum_all(y)

    x = rng.standard_normal((2,))
    program = trace_function(model, x)
    params = param_count_program(program)
    assert params == 17
