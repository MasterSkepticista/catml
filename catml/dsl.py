"""Small helpers for traced programs."""
from __future__ import annotations


def residual(block, x):
    """Apply a residual connection: block(x) + x.

    Args:
        block: Callable representing a residual block.
        x: Input value.

    Returns:
        Output of the residual block.
    """
    return block(x) + x
