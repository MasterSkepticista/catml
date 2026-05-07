import numpy as np

from catml.primitives import linear, relu, sigmoid
from catml.core import trace_function

def model(x):
    y = linear(x, 16, rng, tag="lin1")
    y = relu(y)
    y = linear(y, 1, rng, tag="lin2")
    return sigmoid(y)

rng = np.random.default_rng(42)
x0 = rng.normal(size=(100,))
program = trace_function(model, x0)

print(program)