"""Windowing constraint utilities for DTW.

The DTW algorithm supports windowing constraints via:
1. Built-in Sakoe-Chiba band (window parameter)
2. compose_constraints() for combining multiple constraint functions
3. set_cost_matrix() for arbitrary custom constraints (pre-masked matrix)
"""

from collections.abc import Callable

__all__ = ["compose_constraints"]


def compose_constraints(
    *constraint_fns: Callable[[int, int, int, int], bool],
) -> Callable[[int, int, int, int], bool]:
    """Compose multiple DTW constraint functions into a single constraint.

    Each constraint function takes (i, j, n, m) where:
    - i: row index in cost matrix (0 to n-1)
    - j: column index in cost matrix (0 to m-1)
    - n: total rows (length of trajectory x)
    - m: total columns (length of trajectory y)

    Returns True if the cell (i, j) is allowed, False if it should be masked.

    Parameters
    ----------
    *constraint_fns : Callable[[int, int, int, int], bool]
        Variable number of constraint functions to compose.
        All constraints must return True for a cell to be allowed.

    Returns
    -------
    Callable[[int, int, int, int], bool]
        Composed constraint function that returns True only if all
        input constraints return True.

    Examples
    --------
    >>> from zreg.algorithms.dtw.constraints import compose_constraints
    >>> # Sakoe-Chiba band constraint
    >>> def sakoe_chiba(i, j, n, m, window=2):
    ...     return abs(i - j) <= window
    >>> # Itakura parallelogram constraint
    >>> def itakura(i, j, n, m):
    ...     # Simplified Itakura parallelogram
    ...     return i >= j / 2 and i <= 2 * j + 1
    >>> # Combine them
    >>> combined = compose_constraints(
    ...     lambda i, j, n, m: sakoe_chiba(i, j, n, m, window=3),
    ...     itakura,
    ... )
    >>> # Use with set_cost_matrix(): mask cells where combined(i,j,n,m) is False
    """
    if not constraint_fns:
        # No constraints = all cells allowed
        return lambda i, j, n, m: True

    def composed(i: int, j: int, n: int, m: int) -> bool:
        return all(fn(i, j, n, m) for fn in constraint_fns)

    return composed
