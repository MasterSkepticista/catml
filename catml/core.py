"""Tracing core and a tiny SSA program IR."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass(frozen=True)
class Aval:
    """Abstract value descriptor used during tracing.

    Attributes:
        shape: Tuple shape for the value.
    """
    shape: tuple[int, ...]


@dataclass(frozen=True)
class Var:
    """SSA variable identifier used in the program IR.

    Attributes:
        ident: Unique integer id for the variable.
    """
    ident: int


@dataclass(frozen=True)
class Instr:
    """Single SSA instruction in the program IR.

    Attributes:
        prim: Primitive to apply.
        inputs: Input variables.
        outputs: Output variables.
        params: Parameter dict for the primitive.
        tag: Optional label for grouping parameter gradients.
    """
    prim: Any
    inputs: tuple[Var, ...]
    outputs: tuple[Var, ...]
    params: dict[str, Any] = field(default_factory=dict)
    tag: str | None = None


@dataclass(frozen=True)
class TreeDef:
    """Tree structure descriptor for tuple-valued inputs or outputs.

    Attributes:
        kind: Node kind ("leaf" or "tuple").
        children: Child tree descriptors.
    """
    kind: str
    children: tuple["TreeDef", ...] = ()


@dataclass(frozen=True)
class Program:
    """Traced program in SSA form.

    Attributes:
        inputs: Flat input variables.
        input_treedefs: Tree descriptors for structured inputs.
        instructions: Linear instruction list.
        outputs: Flat output variables.
        output_treedef: Tree descriptor for structured outputs.
    """
    inputs: tuple[Var, ...]
    input_treedefs: tuple[TreeDef, ...]
    instructions: tuple[Instr, ...]
    outputs: tuple[Var, ...]
    output_treedef: TreeDef

    def __str__(self) -> str:
        return format_program(self)


class Trace:
    """Trace interface for defining primitive semantics."""
    def bind(self, prim: Any, args: tuple[Any, ...], params: dict[str, Any], tag: str | None):
        raise NotImplementedError


class ProgramTrace(Trace):
    """Trace that builds a `Program` IR by recording primitives."""
    def __init__(self) -> None:
        self._next_var = 0
        self.instructions: list[Instr] = []

    def new_var(self) -> Var:
        var = Var(self._next_var)
        self._next_var += 1
        return var

    def bind(self, prim: Any, args: tuple[Any, ...], params: dict[str, Any], tag: str | None):
        leaves, _ = flatten_tree(args)
        in_vars: list[Var] = []
        for leaf in leaves:
            if not isinstance(leaf, Tracer):
                raise TypeError("ProgramTrace expected Tracer arguments")
            in_vars.append(leaf.var)
        out_var = self.new_var()
        self.instructions.append(
            Instr(prim=prim, inputs=tuple(in_vars), outputs=(out_var,), params=params, tag=tag)
        )
        out_aval = infer_aval(prim, args)
        return Tracer(self, out_var, out_aval)


class Tracer:
    """Tracer proxy that records primitive applications."""
    def __init__(self, trace: Trace, var: Var, aval: Aval | None = None) -> None:
        self.trace = trace
        self.var = var
        self.aval = aval

    @property
    def shape(self) -> tuple[int, ...] | None:
        return None if self.aval is None else self.aval.shape

    def __add__(self, other: Any):
        from .primitives import add

        return add(self, other)

    def __radd__(self, other: Any):
        from .primitives import add

        return add(other, self)


_trace_stack: list[Trace] = []


class use_trace:
    """Context manager that sets the active trace."""
    def __init__(self, trace: Trace) -> None:
        self.trace = trace

    def __enter__(self):
        _trace_stack.append(self.trace)
        return self.trace

    def __exit__(self, exc_type, exc, tb):
        _trace_stack.pop()
        return False


def current_trace() -> Trace | None:
    return _trace_stack[-1] if _trace_stack else None


def bind(prim: Any, *args: Any, params: dict[str, Any] | None = None, tag: str | None = None):
    """Apply a primitive under the active trace or evaluate it eagerly.

    Args:
        prim: Primitive with `eval` and `shape` methods.
        *args: Inputs to the primitive.
        params: Parameter dict for the primitive.
        tag: Optional label to group parameter gradients.

    Returns:
        Tracer when tracing is active; otherwise the evaluated value.
    """
    params = params or {}
    trace = current_trace()
    if trace is None:
        return prim.eval(args, params)
    return trace.bind(prim, args, params, tag)


def trace_function(func, *example_inputs: Any) -> Program:
    """Trace a Python function into a `Program` using example inputs.

    Args:
        func: Python callable that uses primitives.
        *example_inputs: Example inputs used for tracing and shape inference.

    Returns:
        A `Program` IR capturing the traced computation.

    Raises:
        TypeError: If outputs are not tracer instances.
    """
    trace = ProgramTrace()
    input_vars: list[Var] = []
    input_treedefs: list[TreeDef] = []
    tracer_args: list[Any] = []
    with use_trace(trace):
        for arg in example_inputs:
            leaves, treedef = flatten_tree(arg)
            vars_for_arg = [trace.new_var() for _ in leaves]
            input_vars.extend(vars_for_arg)
            input_treedefs.append(treedef)
            tracer_leaves = [Tracer(trace, var, aval_from_value(leaf)) for var, leaf in zip(vars_for_arg, leaves)]
            tracer_args.append(unflatten_tree(tracer_leaves, treedef))
        outputs = func(*tracer_args)
    out_leaves, out_treedef = flatten_tree(outputs)
    out_vars = []
    for leaf in out_leaves:
        if not isinstance(leaf, Tracer):
            raise TypeError("trace_function outputs must be Tracer instances")
        out_vars.append(leaf.var)
    return Program(
        inputs=tuple(input_vars),
        input_treedefs=tuple(input_treedefs),
        instructions=tuple(trace.instructions),
        outputs=tuple(out_vars),
        output_treedef=out_treedef,
    )


def format_program(program: Program) -> str:
    """Format a program as a readable multiline string.

    Args:
        program: Traced program IR to format.

    Returns:
        Human-readable string representation of the program.
    """
    lines = ["Program("]
    lines.append(f"  inputs: {_format_vars(program.inputs)}")
    lines.append("  instructions:")
    for index, instr in enumerate(program.instructions):
        out_vars = _format_vars(instr.outputs)
        in_vars = _format_vars(instr.inputs)
        params_text = _format_params(instr.params)
        tag_text = f" tag={instr.tag}" if instr.tag else ""
        params_suffix = f" params={params_text}" if params_text else ""
        lines.append(
            f"    {index}: {out_vars} = {instr.prim.name}({in_vars}){params_suffix}{tag_text}"
        )
    lines.append(f"  outputs: {_format_vars(program.outputs)}")
    lines.append(")")
    return "\n".join(lines)


def _format_vars(vars: Iterable[Var]) -> str:
    return ", ".join(_var_name(var) for var in vars)


def _var_name(var: Var) -> str:
    return f"v{var.ident}"


def _format_params(params: dict[str, Any]) -> str:
    if not params:
        return ""
    parts = []
    for key in sorted(params.keys()):
        parts.append(f"{key}={_format_param_value(params[key])}")
    return ", ".join(parts)


def _format_param_value(value: Any) -> str:
    if hasattr(value, "shape"):
        return str(tuple(value.shape))
    return type(value).__name__


def flatten_tree(tree: Any) -> tuple[list[Any], TreeDef]:
    if isinstance(tree, tuple) and (not tree or all(isinstance(item, int) for item in tree)):
        return [tree], TreeDef(kind="leaf")
    if isinstance(tree, tuple):
        leaves: list[Any] = []
        children: list[TreeDef] = []
        for child in tree:
            child_leaves, child_def = flatten_tree(child)
            leaves.extend(child_leaves)
            children.append(child_def)
        return leaves, TreeDef(kind="tuple", children=tuple(children))
    return [tree], TreeDef(kind="leaf")


def unflatten_tree(leaves: Iterable[Any], treedef: TreeDef):
    leaves_iter = iter(leaves)
    value, _ = _unflatten_tree(leaves_iter, treedef)
    return value


def _unflatten_tree(leaves_iter: Iterable[Any], treedef: TreeDef):
    if treedef.kind == "leaf":
        return next(leaves_iter), leaves_iter
    items = []
    for child in treedef.children:
        item, leaves_iter = _unflatten_tree(leaves_iter, child)
        items.append(item)
    return tuple(items), leaves_iter


def aval_from_value(value: Any) -> Aval | None:
    if isinstance(value, Tracer):
        return value.aval
    if hasattr(value, "shape"):
        return Aval(shape=tuple(value.shape))
    return None


def infer_aval(prim: Any, args: tuple[Any, ...]) -> Aval | None:
    shapes: list[tuple[int, ...]] = []
    for arg in args:
        if isinstance(arg, Tracer) and arg.aval is not None:
            shapes.append(arg.aval.shape)
        elif hasattr(arg, "shape"):
            shapes.append(tuple(arg.shape))
        else:
            return None
    out_shape = prim.shape(shapes)
    return Aval(shape=out_shape)
