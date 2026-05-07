import numpy as np

from catml.core import trace_function
from catml.interpreters import eval_program
from catml.primitives import linear, relu, sum_all
from catml.optimizers import fuse_linear_relu

def model(x):
  y = linear(x, 8, rng, tag="linear1")
  y = relu(y)
  y = linear(y, 1, rng, tag="linear2")
  return sum_all(y)

rng = np.random.default_rng(42)
x0 = rng.normal(size=(2,))
program = trace_function(model, x0)
print(program, eval_program(program, (x0,)))

# Fuse linear+relu patterns and re-evaluate.
fused_program = fuse_linear_relu(program)
print(fused_program, eval_program(fused_program, (x0,)))