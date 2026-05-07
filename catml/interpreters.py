"""Program interpreters for evaluation and analysis."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .core import Program, TreeDef, flatten_tree, unflatten_tree


@dataclass(frozen=True)
class EvalInterpreter:
    """Interpreter that evaluates primitives numerically."""
    def eval(self, prim, inputs: tuple[np.ndarray, ...], params: dict[str, np.ndarray]):
        return prim.eval(inputs, params)


@dataclass(frozen=True)
class ShapeInterpreter:
    """Interpreter that propagates shapes through primitives."""
    def eval(self, prim, inputs: tuple[tuple[int, ...], ...], params: dict[str, np.ndarray]):
        return prim.shape(list(inputs))


def run_program(program: Program, inputs: tuple[Any, ...], interpreter) -> Any:
    """Run a program with the given interpreter.

    Args:
        program: Traced program IR.
        inputs: Inputs matching the program input structure.
        interpreter: Interpreter with an `eval` method.

    Returns:
        Program outputs reconstructed to the original output tree.

    Raises:
        ValueError: If input structure mismatches or interpreter outputs are invalid.
    """
    flat_inputs = _flatten_inputs(inputs, program.input_treedefs)
    env: dict[int, Any] = {}
    for var, value in zip(program.inputs, flat_inputs):
        env[var.ident] = value
    for instr in program.instructions:
        in_vals = tuple(env[var.ident] for var in instr.inputs)
        out_vals = interpreter.eval(instr.prim, in_vals, instr.params)
        if len(instr.outputs) == 1:
            env[instr.outputs[0].ident] = out_vals
            continue
        if not isinstance(out_vals, tuple) or len(out_vals) != len(instr.outputs):
            raise ValueError("Interpreter must return tuple output matching instr.outputs")
        for var, value in zip(instr.outputs, out_vals):
            env[var.ident] = value
    outputs = [env[var.ident] for var in program.outputs]
    return unflatten_tree(outputs, program.output_treedef)


def eval_program(program: Program, inputs: tuple[Any, ...]) -> Any:
    """Evaluate a program using numeric primitive implementations.

    Args:
        program: Traced program IR.
        inputs: Numeric inputs matching the program input structure.

    Returns:
        Numeric outputs reconstructed to the original output tree.
    """
    return run_program(program, inputs, EvalInterpreter())


def shape_program(program: Program, input_shapes: tuple[Any, ...]) -> Any:
    """Propagate shapes through a program.

    Args:
        program: Traced program IR.
        input_shapes: Shape tuples matching the program input structure.

    Returns:
        Output shapes reconstructed to the original output tree.
    """
    return run_program(program, input_shapes, ShapeInterpreter())


def param_count_program(program: Program) -> int:
    """Count parameters used by a program.

    Args:
        program: Traced program IR.

    Returns:
        Total parameter count across all instructions.
    """
    total = 0
    for instr in program.instructions:
        total += instr.prim.param_count(instr.params)
    return int(total)


def grad_program(program: Program, inputs: tuple[Any, ...], out_grad: Any) -> tuple[Any, dict[str, dict[str, np.ndarray]]]:
    """Run reverse-mode autodiff for a program.

    Args:
        program: Traced program IR.
        inputs: Numeric inputs matching the program input structure.
        out_grad: Output cotangent matching the program output structure.

    Returns:
        Tuple of (primal output, parameter gradients).

    Raises:
        ValueError: If input or output structures do not match the program.
    """
    flat_inputs = _flatten_inputs(inputs, program.input_treedefs)
    env: dict[int, Any] = {}
    for var, value in zip(program.inputs, flat_inputs):
        env[var.ident] = value
    contexts: list[tuple[Any, tuple[Any, ...], Any, tuple[int, ...]]] = []
    for instr in program.instructions:
        in_vals = tuple(env[var.ident] for var in instr.inputs)
        out_val = instr.prim.eval(in_vals, instr.params)
        env[instr.outputs[0].ident] = out_val
        contexts.append((instr, in_vals, out_val, tuple(var.ident for var in instr.inputs)))

    outputs = [env[var.ident] for var in program.outputs]
    primal = unflatten_tree(outputs, program.output_treedef)

    flat_out_grads = _flatten_output_grads(out_grad, program.output_treedef)
    grad_env: dict[int, Any] = {}
    for var, grad in zip(program.outputs, flat_out_grads):
        grad_env[var.ident] = grad

    param_grads: dict[str, dict[str, np.ndarray]] = {}
    for instr, in_vals, out_val, in_idents in reversed(contexts):
        out_grad_val = grad_env.get(instr.outputs[0].ident)
        if out_grad_val is None:
            continue
        in_grads, pgrads = instr.prim.pullback(in_vals, instr.params, out_grad_val, out_val)
        for ident, grad in zip(in_idents, in_grads):
            if ident in grad_env:
                grad_env[ident] = grad_env[ident] + grad
            else:
                grad_env[ident] = grad
        if instr.tag is not None and pgrads:
            if instr.tag not in param_grads:
                param_grads[instr.tag] = {k: v.copy() for k, v in pgrads.items()}
            else:
                for key, value in pgrads.items():
                    if key in param_grads[instr.tag]:
                        param_grads[instr.tag][key] = param_grads[instr.tag][key] + value
                    else:
                        param_grads[instr.tag][key] = value.copy()
    return primal, param_grads


def _flatten_inputs(inputs: tuple[Any, ...], treedefs: tuple[TreeDef, ...]) -> list[Any]:
    if len(inputs) != len(treedefs):
        raise ValueError("Input arity mismatch")
    flat: list[Any] = []
    for value, treedef in zip(inputs, treedefs):
        leaves, actual = flatten_tree(value)
        if actual != treedef:
            raise ValueError("Input structure does not match traced program")
        flat.extend(leaves)
    return flat


def _flatten_output_grads(out_grad: Any, treedef: TreeDef) -> list[Any]:
    leaves, actual = flatten_tree(out_grad)
    if actual != treedef:
        raise ValueError("Output gradient structure mismatch")
    return leaves
