"""Diagrammatic backprop helpers built from traced programs."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import Program
from .interpreters import eval_program, pullback_program


@dataclass(frozen=True)
class DiagramMap:
    """A named program viewed as a morphism between representation spaces."""

    source: str
    target: str
    program: Program
    name: str


def collect_params(program: Program) -> dict[str, dict[str, np.ndarray]]:
    """Collect mutable parameter dicts grouped by instruction tag."""
    params_by_tag: dict[str, dict[str, np.ndarray]] = {}
    for instr in program.instructions:
        if instr.tag is None or not instr.params:
            continue
        params_by_tag[instr.tag] = instr.params
    return params_by_tag


def sgd_step(
    params_by_tag: dict[str, dict[str, np.ndarray]],
    grads_by_tag: dict[str, dict[str, np.ndarray]],
    lr: float,
) -> None:
    """Apply an in-place SGD step to a tagged parameter collection."""
    for tag, params in params_by_tag.items():
        grads = grads_by_tag.get(tag)
        if grads is None:
            continue
        for key, value in grads.items():
            params[key] = params[key] - lr * value


def add_grads(
    total: dict[str, dict[str, np.ndarray]],
    update: dict[str, dict[str, np.ndarray]],
) -> dict[str, dict[str, np.ndarray]]:
    """Accumulate parameter gradients across multiple losses."""
    for tag, grads in update.items():
        bucket = total.setdefault(tag, {})
        for key, value in grads.items():
            if key in bucket:
                bucket[key] = bucket[key] + value
            else:
                bucket[key] = value.copy()
    return total


def eval_map(map_: DiagramMap, x: np.ndarray) -> np.ndarray:
    """Evaluate a diagram edge under the standard numeric interpreter."""
    return eval_program(map_.program, (x,))


def eval_path(path: tuple[DiagramMap, ...], x: np.ndarray) -> np.ndarray:
    """Evaluate a compositional path of diagram edges."""
    value = x
    for map_ in path:
        value = eval_map(map_, value)
    return value


def triangle_loss(
    direct: DiagramMap,
    left: DiagramMap,
    right: DiagramMap,
    x: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return triangle energy and the two path outputs."""
    direct_y = eval_map(direct, x)
    composed_y = eval_path((left, right), x)
    residual = direct_y - composed_y
    return float(0.5 * np.sum(residual * residual)), direct_y, composed_y


def triangle_gradients(
    direct: DiagramMap,
    left: DiagramMap,
    right: DiagramMap,
    x: np.ndarray,
) -> tuple[float, dict[str, dict[str, np.ndarray]], dict[str, dict[str, np.ndarray]], dict[str, dict[str, np.ndarray]]]:
    """Backpropagate triangle energy through all three edges.

    The triangle is `direct ~= right o left`.
    """
    direct_y = eval_map(direct, x)
    left_y = eval_map(left, x)
    composed_y = eval_map(right, left_y)
    residual = direct_y - composed_y
    loss = float(0.5 * np.sum(residual * residual))

    _, _, direct_grads = pullback_program(direct.program, (x,), residual)
    _, right_input_grads, right_grads = pullback_program(right.program, (left_y,), -residual)
    left_out_grad = right_input_grads[0]
    _, _, left_grads = pullback_program(left.program, (x,), left_out_grad)
    return loss, direct_grads, left_grads, right_grads
