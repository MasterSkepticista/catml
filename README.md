# CatML

CatML is a small categorical ML core built around the idea:

1. Trace a Python model into a tiny SSA-like program IR
2. Interpret that same program in different semantic domains

This leads to a framework where evaluation, shape propagation, parameter counting, reverse-mode autodiff, and diagrammatic training all operate over the same traced program object.

## Motivation

Most ML systems need more than forward execution. They also need shape reasoning, gradient propagation, parameter introspection, and higher-order transformations. CatML treats those as different interpretations of one shared syntax rather than as unrelated subsystems.

Example:

```text
        eval
  P ------------> V
  |               |
  | shape         | shape
  v               v
  S ============= S
         id
```

A traced program can be evaluated concretely in a value domain `V`, or abstractly in a shape domain `S`. CatML is built so these views line up.

## Capabilities

- Tiny traced IR with explicit SSA variables and instructions
- Primitive-based semantics: each op carries `eval`, `pullback`, `shape`, and parameter-count rules
- Functorial interpreters for:
  - numeric execution
  - shape propagation
  - parameter counting
  - reverse-mode autodiff via pullbacks
- Structured tuple inputs and outputs through tree flatten/unflatten
- Diagrammatic backprop over composed maps, not just ordinary scalar losses
- Enough primitives to express small MLPs and transformer blocks:
  `linear`, `attention`, `layer_norm`, `relu`, `sigmoid`, `sum_all`, `add`

## The Program IR

CatML traces Python once and records a compact program:

```python
@dataclass(frozen=True)
class Instr:
    prim: Any
    inputs: tuple[Var, ...]
    outputs: tuple[Var, ...]
    params: dict[str, Any] = field(default_factory=dict)
    tag: str | None = None

@dataclass(frozen=True)
class Program:
    inputs: tuple[Var, ...]
    input_treedefs: tuple[TreeDef, ...]
    instructions: tuple[Instr, ...]
    outputs: tuple[Var, ...]
    output_treedef: TreeDef
```

This keeps the core very small: a model becomes a linear list of primitive applications with explicit data dependencies and stored parameters.

## Example 1: Transformations

```python
import numpy as np

from catml import (
    eval_program,
    grad_program,
    linear,
    param_count_program,
    relu,
    shape_program,
    sigmoid,
    trace_function,
)

rng = np.random.default_rng(0)

def model(x):
    h = linear(x, 4, rng, tag="lin1")
    h = relu(h)
    y = linear(h, 1, rng, tag="lin2")
    return sigmoid(y)

x0 = np.zeros((2,))
program = trace_function(model, x0)

value = eval_program(program, (x0,))
shape = shape_program(program, (x0.shape,))
params = param_count_program(program)
primal, grads = grad_program(program, (x0,), np.ones((1,)))
```

The important point is that `program` is the shared object. The framework does not rebuild a separate representation for each analysis.

## Example 2: Diagrammatic Backprop

CatML also supports training objectives defined over diagrams of maps. The triangle example in [`examples/diagrammatic_backprop_triangle.py`](examples/diagrammatic_backprop_triangle.py) uses three learned morphisms:

```text
Evidence --f--> Syndrome --g--> Diagnosis
    \________________h________________/
```

and trains the direct path `h` to agree with the composed path `g ∘ f`.

```python
from catml import triangle_gradients

# ...
f_program = trace_function(
    lambda x: tiny_transformer(x, hidden_dim, latent_dim, rng, "evidence_syndrome"),
    example_evidence,
)
g_program = trace_function(
    lambda x: tiny_transformer(
        x, hidden_dim, diagnosis_dim, rng, "syndrome_diagnosis"
    ),
    example_latent,
)
h_program = trace_function(
    lambda x: tiny_transformer(
        x, hidden_dim, diagnosis_dim, rng, "evidence_diagnosis"
    ),
    example_evidence,
)
f_map, g_map, h_map = (
    DiagramMap("Evidence", "Syndrome", f_program, "f"),
    DiagramMap("Syndrome", "Diagnosis", g_program, "g"),
    DiagramMap("Evidence", "Diagnosis", h_program, "h"),
)
# ...

loss, h_grads, f_grads, g_grads = triangle_gradients(
    direct=h_map,
    left=f_map,
    right=g_map,
    x=sample.evidence,
)
```

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
python3 examples/xor.py
python3 examples/diagrammatic_backprop_triangle.py
```

## License
MIT