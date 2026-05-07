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


def attention(q, k, v):
    """Apply scaled dot-product attention.

    Args:
        q: Query matrix of shape (seq_len, d_model).
        k: Key matrix of shape (seq_len, d_model).
        v: Value matrix of shape (seq_len, d_model).

    Returns:
        Attention output of shape (seq_len, d_model).
    """
    def eval_fn(args: tuple[np.ndarray, ...], params: dict[str, np.ndarray]) -> np.ndarray:
        q_val, k_val, v_val = args
        d_model = q_val.shape[-1]
        scale = 1.0 / np.sqrt(d_model)
        scores = (q_val @ k_val.swapaxes(-1, -2)) * scale
        scores_max = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        return weights @ v_val

    def pullback_fn(
        args: tuple[np.ndarray, ...],
        params: dict[str, np.ndarray],
        out_grad: np.ndarray,
        out: np.ndarray,
    ) -> tuple[tuple[np.ndarray, ...], dict[str, np.ndarray]]:
        q_val, k_val, v_val = args
        d_model = q_val.shape[-1]
        scale = 1.0 / np.sqrt(d_model)
        scores = (q_val @ k_val.swapaxes(-1, -2)) * scale
        scores_max = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

        dv = weights.swapaxes(-1, -2) @ out_grad
        dweights = out_grad @ v_val.swapaxes(-1, -2)
        dscores = dweights * weights
        dscores -= weights * np.sum(dweights * weights, axis=-1, keepdims=True)
        dscores *= scale
        dq = dscores @ k_val
        dk = dscores.swapaxes(-1, -2) @ q_val
        return (dq, dk, dv), {}

    def shape_fn(input_shapes: list[tuple[int, ...]]) -> tuple[int, ...]:
        return input_shapes[2]

    prim = Primitive(
        name="attention",
        eval_fn=eval_fn,
        pullback_fn=pullback_fn,
        shape_fn=shape_fn,
        param_count_fn=lambda params: 0,
    )
    return bind(prim, q, k, v)


def linear(x, out_dim: int, rng: np.random.Generator, tag: str | None = None):
    """Apply an affine transformation over the last dimension.

    Args:
        x: Input array with last dimension representing features.
        out_dim: Output feature dimension.
        rng: NumPy random generator for parameter initialization.
        tag: Optional label for grouping parameter gradients.

    Returns:
        Output array with the same leading dimensions and last dimension `out_dim`.

    Raises:
        ValueError: If the input shape cannot be inferred.
    """
    in_dim = _shape_of(x)[-1]
    scale = 1.0 / np.sqrt(in_dim)
    params = {
        "W": rng.standard_normal((out_dim, in_dim)) * scale,
        "b": np.zeros((out_dim,)),
    }

    def eval_fn(args: tuple[np.ndarray, ...], params: dict[str, np.ndarray]) -> np.ndarray:
        x_val = args[0]
        return x_val @ params["W"].T + params["b"]

    def pullback_fn(
        args: tuple[np.ndarray, ...],
        params: dict[str, np.ndarray],
        out_grad: np.ndarray,
        out: np.ndarray,
    ) -> tuple[tuple[np.ndarray, ...], dict[str, np.ndarray]]:
        x_val = args[0]
        x_flat = x_val.reshape(-1, x_val.shape[-1])
        grad_flat = out_grad.reshape(-1, out_grad.shape[-1])
        dW = grad_flat.T @ x_flat
        db = np.sum(grad_flat, axis=0)
        dx = out_grad @ params["W"]
        return (dx,), {"W": dW, "b": db}

    def shape_fn(input_shapes: list[tuple[int, ...]]) -> tuple[int, ...]:
        return (*input_shapes[0][:-1], out_dim)

    prim = Primitive(
        name="linear",
        eval_fn=eval_fn,
        pullback_fn=pullback_fn,
        shape_fn=shape_fn,
        param_count_fn=lambda params: int(sum(np.prod(p.shape) for p in params.values())),
    )
    return bind(prim, x, params=params, tag=tag)


def layer_norm(
    x,
    rng: np.random.Generator,
    eps: float = 1e-5,
    tag: str | None = None,
):
    """Apply layer normalization over the last dimension.

    Args:
        x: Input array.
        rng: NumPy random generator for parameter initialization.
        eps: Numerical stability constant.
        tag: Optional label for grouping parameter gradients.

    Returns:
        Layer-normalized output with the same shape as `x`.
    """
    in_dim = _shape_of(x)[-1]
    params = {
        "gamma": np.ones((in_dim,)),
        "beta": np.zeros((in_dim,)),
    }

    def eval_fn(args: tuple[np.ndarray, ...], params: dict[str, np.ndarray]) -> np.ndarray:
        x_val = args[0]
        mean = np.mean(x_val, axis=-1, keepdims=True)
        var = np.mean((x_val - mean) ** 2, axis=-1, keepdims=True)
        inv_std = 1.0 / np.sqrt(var + eps)
        x_hat = (x_val - mean) * inv_std
        return x_hat * params["gamma"] + params["beta"]

    def pullback_fn(
        args: tuple[np.ndarray, ...],
        params: dict[str, np.ndarray],
        out_grad: np.ndarray,
        out: np.ndarray,
    ) -> tuple[tuple[np.ndarray, ...], dict[str, np.ndarray]]:
        x_val = args[0]
        mean = np.mean(x_val, axis=-1, keepdims=True)
        var = np.mean((x_val - mean) ** 2, axis=-1, keepdims=True)
        inv_std = 1.0 / np.sqrt(var + eps)
        x_hat = (x_val - mean) * inv_std

        axes = tuple(range(x_val.ndim - 1))
        dgamma = np.sum(out_grad * x_hat, axis=axes)
        dbeta = np.sum(out_grad, axis=axes)

        dx_hat = out_grad * params["gamma"]
        dmean = np.mean(dx_hat, axis=-1, keepdims=True)
        dmean_xhat = np.mean(dx_hat * x_hat, axis=-1, keepdims=True)
        dx = inv_std * (dx_hat - dmean - x_hat * dmean_xhat)
        return (dx,), {"gamma": dgamma, "beta": dbeta}

    def shape_fn(input_shapes: list[tuple[int, ...]]) -> tuple[int, ...]:
        return input_shapes[0]

    prim = Primitive(
        name="layer_norm",
        eval_fn=eval_fn,
        pullback_fn=pullback_fn,
        shape_fn=shape_fn,
        param_count_fn=lambda params: int(sum(np.prod(p.shape) for p in params.values())),
    )
    return bind(prim, x, params=params, tag=tag)


def _shape_of(x) -> tuple[int, ...]:
    if isinstance(x, Tracer) and x.shape is not None:
        return x.shape
    if hasattr(x, "shape"):
        return tuple(x.shape)
    raise ValueError("Cannot infer shape")
