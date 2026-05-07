"""IR optimization passes for traced programs."""
from __future__ import annotations

from typing import Dict

from .core import Instr, Program, Var
from .primitives import linear_relu_prim


def fuse_linear_relu(program: Program) -> Program:
    """Fuse linear + relu patterns into a single primitive.

    Args:
        program: Input traced program.

    Returns:
        New program with eligible linear+relu pairs fused.
    """
    use_counts = _use_counts(program)
    new_instructions: list[Instr] = []
    index = 0
    while index < len(program.instructions):
        instr = program.instructions[index]
        if _is_linear(instr) and index + 1 < len(program.instructions):
            next_instr = program.instructions[index + 1]
            if _is_relu(next_instr) and _can_fuse(instr, next_instr, use_counts, program):
                out_dim = int(instr.params["W"].shape[0])
                fused = Instr(
                    prim=linear_relu_prim(out_dim),
                    inputs=instr.inputs,
                    outputs=next_instr.outputs,
                    params=instr.params,
                    tag=instr.tag,
                )
                new_instructions.append(fused)
                index += 2
                continue
        new_instructions.append(instr)
        index += 1
    return Program(
        inputs=program.inputs,
        input_treedefs=program.input_treedefs,
        instructions=tuple(new_instructions),
        outputs=program.outputs,
        output_treedef=program.output_treedef,
    )


def _use_counts(program: Program) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for instr in program.instructions:
        for var in instr.inputs:
            counts[var.ident] = counts.get(var.ident, 0) + 1
    for var in program.outputs:
        counts[var.ident] = counts.get(var.ident, 0) + 1
    return counts


def _is_linear(instr: Instr) -> bool:
    return instr.prim.name == "linear"


def _is_relu(instr: Instr) -> bool:
    return instr.prim.name == "relu"


def _can_fuse(
    linear_instr: Instr,
    relu_instr: Instr,
    use_counts: Dict[int, int],
    program: Program,
) -> bool:
    if len(linear_instr.outputs) != 1 or len(relu_instr.inputs) != 1:
        return False
    linear_out = linear_instr.outputs[0]
    if relu_instr.inputs[0].ident != linear_out.ident:
        return False
    if use_counts.get(linear_out.ident, 0) != 1:
        return False
    if any(var.ident == linear_out.ident for var in program.outputs):
        return False
    if "W" not in linear_instr.params or "b" not in linear_instr.params:
        return False
    return True
