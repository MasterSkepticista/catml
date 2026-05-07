"""Primitive differentiable operations for tracing and interpretation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .core import Tracer, bind

EvalFn = Callable[[tuple[np.ndarray, ...], dict[str, np.ndarray]], np.ndarray]
PullbackFn = Callable[
    [tuple[np.ndarray, ...], dict[str, np.ndarray], np.ndarray, np.ndarray],
    tuple[tuple[np.ndarray, ...], dict[str, np.ndarray]],
]
ShapeFn = Callable[[list[tuple[int, ...]]], tuple[int, ...]]
ParamCountFn = Callable[[dict[str, np.ndarray]], int]


@dataclass(frozen=True)
class Primitive:
    """Primitive operation definition with evaluation and analysis rules.

    Attributes:
        name: Display name for the primitive.
        eval_fn: Numeric evaluator.
        pullback_fn: Reverse-mode pullback implementation.
        shape_fn: Shape propagation rule.
        param_count_fn: Parameter counting rule.
    """
    name: str
    eval_fn: EvalFn
    pullback_fn: PullbackFn
    shape_fn: ShapeFn
    param_count_fn: ParamCountFn

    def eval(self, args: tuple[np.ndarray, ...], params: dict[str, np.ndarray]) -> np.ndarray:
        return self.eval_fn(args, params)

    def pullback(
        self,
        args: tuple[np.ndarray, ...],
        params: dict[str, np.ndarray],
        out_grad: np.ndarray,
        out: np.ndarray,
    ) -> tuple[tuple[np.ndarray, ...], dict[str, np.ndarray]]:
        return self.pullback_fn(args, params, out_grad, out)

    def shape(self, input_shapes: list[tuple[int, ...]]) -> tuple[int, ...]:
        return self.shape_fn(input_shapes)

    def param_count(self, params: dict[str, np.ndarray]) -> int:
        return self.param_count_fn(params)


def add(x, y):
    """Add two values with gradient support.

    Args:
        x: Left operand.
        y: Right operand.

    Returns:
        Sum of `x` and `y`.
    """
    def eval_fn(args: tuple[np.ndarray, ...], params: dict[str, np.ndarray]) -> np.ndarray:
        return args[0] + args[1]

    def pullback_fn(
        args: tuple[np.ndarray, ...],
        params: dict[str, np.ndarray],
        out_grad: np.ndarray,
        out: np.ndarray,
    ) -> tuple[tuple[np.ndarray, ...], dict[str, np.ndarray]]:
        return (out_grad, out_grad), {}

    def shape_fn(input_shapes: list[tuple[int, ...]]) -> tuple[int, ...]:
        return input_shapes[0]

    prim = Primitive(
        name="add",
        eval_fn=eval_fn,
        pullback_fn=pullback_fn,
        shape_fn=shape_fn,
        param_count_fn=lambda params: 0,
    )
    return bind(prim, x, y)


def relu(x):
    """Apply ReLU activation.

    Args:
        x: Input value.

    Returns:
        ReLU-activated value.
    """
    def eval_fn(args: tuple[np.ndarray, ...], params: dict[str, np.ndarray]) -> np.ndarray:
        return np.maximum(0.0, args[0])

    def pullback_fn(
        args: tuple[np.ndarray, ...],
        params: dict[str, np.ndarray],
        out_grad: np.ndarray,
        out: np.ndarray,
    ) -> tuple[tuple[np.ndarray, ...], dict[str, np.ndarray]]:
        dx = out_grad * (args[0] > 0)
        return (dx,), {}

    def shape_fn(input_shapes: list[tuple[int, ...]]) -> tuple[int, ...]:
        return input_shapes[0]

    prim = Primitive(
        name="relu",
        eval_fn=eval_fn,
        pullback_fn=pullback_fn,
        shape_fn=shape_fn,
        param_count_fn=lambda params: 0,
    )
    return bind(prim, x)


def sigmoid(x):
    """Apply sigmoid activation.

    Args:
        x: Input value.

    Returns:
        Sigmoid-activated value.
    """
    def eval_fn(args: tuple[np.ndarray, ...], params: dict[str, np.ndarray]) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-args[0]))

    def pullback_fn(
        args: tuple[np.ndarray, ...],
        params: dict[str, np.ndarray],
        out_grad: np.ndarray,
        out: np.ndarray,
    ) -> tuple[tuple[np.ndarray, ...], dict[str, np.ndarray]]:
        dx = out_grad * out * (1.0 - out)
        return (dx,), {}

    def shape_fn(input_shapes: list[tuple[int, ...]]) -> tuple[int, ...]:
        return input_shapes[0]

    prim = Primitive(
        name="sigmoid",
        eval_fn=eval_fn,
        pullback_fn=pullback_fn,
        shape_fn=shape_fn,
        param_count_fn=lambda params: 0,
    )
    return bind(prim, x)


def sum_all(x):
    """Sum all elements of a vector into a scalar shape `(1,)`.

    Args:
        x: Input vector.

    Returns:
        Single-element array containing the sum.
    """
    def eval_fn(args: tuple[np.ndarray, ...], params: dict[str, np.ndarray]) -> np.ndarray:
        return np.array([np.sum(args[0])])

    def pullback_fn(
        args: tuple[np.ndarray, ...],
        params: dict[str, np.ndarray],
        out_grad: np.ndarray,
        out: np.ndarray,
    ) -> tuple[tuple[np.ndarray, ...], dict[str, np.ndarray]]:
        dx = np.ones_like(args[0]) * out_grad[0]
        return (dx,), {}

    def shape_fn(input_shapes: list[tuple[int, ...]]) -> tuple[int, ...]:
        return (1,)

    prim = Primitive(
        name="sum",
        eval_fn=eval_fn,
        pullback_fn=pullback_fn,
        shape_fn=shape_fn,
        param_count_fn=lambda params: 0,
    )
    return bind(prim, x)


def linear(x, out_dim: int, rng: np.random.Generator, tag: str | None = None):
    """Apply an affine transformation with freshly initialized parameters.

    Args:
        x: Input vector.
        out_dim: Output dimension.
        rng: NumPy random generator for parameter initialization.
        tag: Optional label for grouping parameter gradients.

    Returns:
        Output vector of shape `(out_dim,)`.

    Raises:
        ValueError: If the input shape cannot be inferred.
    """
    in_dim = _shape_of(x)[0]
    scale = 1.0 / np.sqrt(in_dim)
    params = {
        "W": rng.standard_normal((out_dim, in_dim)) * scale,
        "b": np.zeros((out_dim,)),
    }

    def eval_fn(args: tuple[np.ndarray, ...], params: dict[str, np.ndarray]) -> np.ndarray:
        return params["W"] @ args[0] + params["b"]

    def pullback_fn(
        args: tuple[np.ndarray, ...],
        params: dict[str, np.ndarray],
        out_grad: np.ndarray,
        out: np.ndarray,
    ) -> tuple[tuple[np.ndarray, ...], dict[str, np.ndarray]]:
        x_val = args[0]
        dW = np.outer(out_grad, x_val)
        db = out_grad
        dx = params["W"].T @ out_grad
        return (dx,), {"W": dW, "b": db}

    def shape_fn(input_shapes: list[tuple[int, ...]]) -> tuple[int, ...]:
        return (out_dim,)

    prim = Primitive(
        name="linear",
        eval_fn=eval_fn,
        pullback_fn=pullback_fn,
        shape_fn=shape_fn,
        param_count_fn=lambda params: int(sum(np.prod(p.shape) for p in params.values())),
    )
    return bind(prim, x, params=params, tag=tag)


def linear_relu(x, out_dim: int, rng: np.random.Generator, tag: str | None = None):
    """Apply an affine transformation followed by ReLU.

    Args:
        x: Input vector.
        out_dim: Output dimension.
        rng: NumPy random generator for parameter initialization.
        tag: Optional label for grouping parameter gradients.

    Returns:
        ReLU-activated output vector of shape `(out_dim,)`.

    Raises:
        ValueError: If the input shape cannot be inferred.
    """
    in_dim = _shape_of(x)[0]
    scale = 1.0 / np.sqrt(in_dim)
    params = {
        "W": rng.standard_normal((out_dim, in_dim)) * scale,
        "b": np.zeros((out_dim,)),
    }
    prim = linear_relu_prim(out_dim)
    return bind(prim, x, params=params, tag=tag)


def linear_relu_prim(out_dim: int) -> Primitive:
    """Construct a fused linear + ReLU primitive for a fixed output dimension."""
    def eval_fn(args: tuple[np.ndarray, ...], params: dict[str, np.ndarray]) -> np.ndarray:
        z = params["W"] @ args[0] + params["b"]
        return np.maximum(0.0, z)

    def pullback_fn(
        args: tuple[np.ndarray, ...],
        params: dict[str, np.ndarray],
        out_grad: np.ndarray,
        out: np.ndarray,
    ) -> tuple[tuple[np.ndarray, ...], dict[str, np.ndarray]]:
        x_val = args[0]
        dz = out_grad * (out > 0)
        dW = np.outer(dz, x_val)
        db = dz
        dx = params["W"].T @ dz
        return (dx,), {"W": dW, "b": db}

    def shape_fn(input_shapes: list[tuple[int, ...]]) -> tuple[int, ...]:
        return (out_dim,)

    return Primitive(
        name="linear_relu",
        eval_fn=eval_fn,
        pullback_fn=pullback_fn,
        shape_fn=shape_fn,
        param_count_fn=lambda params: int(sum(np.prod(p.shape) for p in params.values())),
    )


def _shape_of(x) -> tuple[int, ...]:
    if isinstance(x, Tracer) and x.shape is not None:
        return x.shape
    if hasattr(x, "shape"):
        return tuple(x.shape)
    raise ValueError("Cannot infer shape")
