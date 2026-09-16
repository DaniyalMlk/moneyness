"""Small numerical routines the calibration needs.

The library has no runtime dependencies and is not going to acquire any, so the
two pieces of machinery calibration rests on are written here: a direct solver
for the small symmetric system that the linear part of the fit reduces to, and
a derivative-free simplex search for the two parameters that remain nonlinear.

Neither is a general-purpose replacement for a real numerical library. Both are
chosen for the shape of the problem actually being solved. The linear system is
three by three, symmetric and positive definite, which is small enough that an
explicit factorisation is clearer and faster than anything iterative. The outer
search is over two parameters, on an objective that is continuous but whose
derivative is awkward to write down, which is the case Nelder-Mead was designed
for and one where its lack of convergence guarantees costs little: the search
space is two-dimensional and can be restarted from a different corner cheaply.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

__all__ = [
    "SimplexResult",
    "cholesky_solve",
    "nelder_mead",
    "project_onto_cone",
]


def cholesky_solve(matrix: Sequence[Sequence[float]], rhs: Sequence[float]) -> list[float]:
    """Solve ``A x = b`` for symmetric positive-definite ``A``.

    The matrix is factorised as ``L L^T`` and the two triangular systems are
    solved in turn. Cholesky rather than a general LU because the matrix here is
    always a Gram matrix — ``X^T X`` for a design matrix ``X`` — which is
    symmetric and, when the design has full column rank, positive definite. The
    factorisation is half the work of LU and, more usefully, it fails loudly
    exactly when that rank assumption breaks: a non-positive pivot means the
    columns are linearly dependent, which for the calibration means the quotes
    do not determine the parameters.

    Args:
        matrix: Square, symmetric, positive definite. Only the lower triangle is
            read, so the upper triangle need not be filled in exactly.
        rhs: Right-hand side, of matching length.

    Returns:
        The solution vector.

    Raises:
        ValueError: If the dimensions disagree, or the matrix is not positive
            definite.
    """
    n = len(matrix)
    if n == 0:
        raise ValueError("matrix must not be empty")
    if any(len(row) != n for row in matrix):
        raise ValueError("matrix must be square")
    if len(rhs) != n:
        raise ValueError(f"rhs has length {len(rhs)}, expected {n}")

    lower = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            total = matrix[i][j] - sum(lower[i][k] * lower[j][k] for k in range(j))
            if i == j:
                if total <= 0.0:
                    raise ValueError(
                        "matrix is not positive definite; "
                        f"pivot {i} is {total}, which means the columns are dependent"
                    )
                lower[i][j] = math.sqrt(total)
            else:
                lower[i][j] = total / lower[j][j]

    # Forward substitution, L y = b.
    y = [0.0] * n
    for i in range(n):
        y[i] = (rhs[i] - sum(lower[i][k] * y[k] for k in range(i))) / lower[i][i]

    # Back substitution, L^T x = y.
    x = [0.0] * n
    for i in reversed(range(n)):
        x[i] = (y[i] - sum(lower[k][i] * x[k] for k in range(i + 1, n))) / lower[i][i]
    return x


def project_onto_cone(d: float, c: float) -> tuple[float, float]:
    """Nearest point to ``(d, c)`` in the cone ``c >= |d|``.

    This is the constraint that keeps a calibrated slice non-negative
    everywhere, and it is a second-order cone rather than a box, so clamping
    each coordinate separately would not give the nearest feasible point.

    The cone is the intersection of the half-planes ``c >= d`` and ``c >= -d``,
    which splits the plane into three regions. Inside the cone, the point is its
    own projection. In the polar cone ``c <= -|d|``, every direction out of the
    point leads away from the cone and the projection collapses to the apex. In
    the two remaining wedges the nearest point lies on one face, and projecting
    onto that line gives the answer in closed form.

    Args:
        d: The coordinate constrained in absolute value.
        c: The coordinate constrained to dominate it.

    Returns:
        The projected ``(d, c)``.
    """
    if c >= abs(d):
        return d, c
    if c <= -abs(d):
        return 0.0, 0.0
    # One face. Projecting (d, c) onto the line c = d gives both coordinates
    # equal to their mean; onto c = -d, equal and opposite to the mean of
    # (-d, c). The sign of d picks the face.
    half = (abs(d) + c) / 2.0
    return math.copysign(half, d), half


@dataclass(frozen=True, slots=True)
class SimplexResult:
    """Where a simplex search stopped, and whether it got there honestly.

    Attributes:
        point: The best point found.
        value: The objective there.
        iterations: Iterations used.
        converged: True if the simplex collapsed below the tolerances, False if
            the iteration cap ran out first. A caller that cares about the
            difference should look, because the point is returned either way.
    """

    point: tuple[float, ...]
    value: float
    iterations: int
    converged: bool


def nelder_mead(
    objective: Callable[[tuple[float, ...]], float],
    start: Sequence[float],
    *,
    step: float = 0.1,
    tolerance: float = 1e-10,
    max_iterations: int = 2000,
) -> SimplexResult:
    """Minimise ``objective`` by the Nelder-Mead simplex method.

    A simplex of ``n + 1`` points crawls downhill by reflecting its worst vertex
    through the centroid of the others, then stretching further if that helped a
    lot, pulling back if it helped little, and shrinking the whole simplex
    towards the best vertex if it did not help at all.

    Two details matter for the use here. The initial simplex is built by
    perturbing each coordinate *relatively* where the coordinate is not near
    zero, because the parameters being searched over differ in scale and a fixed
    absolute step would be enormous for one and invisible for the other.
    Convergence is judged on both the spread of the simplex and the spread of
    the objective across it, since either alone can be small while the search is
    still moving: a flat valley shrinks the function values while the vertices
    are still far apart, and a steep narrow one does the reverse.

    Args:
        objective: The function to minimise. May return ``inf`` to reject a
            point, which is how constraints are expressed to this search.
        start: Initial point. Its length sets the dimension.
        step: Relative size of the initial simplex.
        tolerance: Convergence threshold, applied to both spreads.
        max_iterations: Cap on iterations.

    Returns:
        A :class:`SimplexResult`.

    Raises:
        ValueError: If ``start`` is empty or the objective is ``inf`` there.
    """
    n = len(start)
    if n == 0:
        raise ValueError("start must not be empty")

    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5

    def as_point(values: Sequence[float]) -> tuple[float, ...]:
        return tuple(float(v) for v in values)

    simplex = [as_point(start)]
    for i in range(n):
        shifted = list(start)
        # A relative step where the coordinate has a scale of its own, an
        # absolute one where it does not.
        shifted[i] += step * abs(shifted[i]) if abs(shifted[i]) > 1e-8 else step
        simplex.append(as_point(shifted))

    values = [objective(p) for p in simplex]
    if math.isinf(values[0]) or math.isnan(values[0]):
        raise ValueError(f"objective is not finite at the starting point {as_point(start)}")

    iterations = 0
    for iterations in range(1, max_iterations + 1):  # noqa: B007
        order = sorted(range(n + 1), key=lambda i: values[i])
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]

        spread = max(
            abs(simplex[i][j] - simplex[0][j]) for i in range(1, n + 1) for j in range(n)
        )
        value_spread = abs(values[-1] - values[0])
        if spread <= tolerance and value_spread <= tolerance:
            return SimplexResult(simplex[0], values[0], iterations, True)

        centroid = tuple(sum(p[j] for p in simplex[:-1]) / n for j in range(n))

        def blend(towards: tuple[float, ...], weight: float) -> tuple[float, ...]:
            return tuple(c + weight * (t - c) for c, t in zip(centroid, towards, strict=True))

        worst = simplex[-1]
        reflected = blend(worst, -alpha)
        reflected_value = objective(reflected)

        if reflected_value < values[0]:
            # Reflection beat the best point, so the direction is worth pushing.
            expanded = blend(worst, -gamma)
            expanded_value = objective(expanded)
            if expanded_value < reflected_value:
                simplex[-1], values[-1] = expanded, expanded_value
            else:
                simplex[-1], values[-1] = reflected, reflected_value
            continue

        if reflected_value < values[-2]:
            simplex[-1], values[-1] = reflected, reflected_value
            continue

        # Reflection was no better than the second worst point. Contract, on
        # whichever side of the centroid currently holds the better value.
        if reflected_value < values[-1]:
            contracted = blend(reflected, rho)
            contracted_value = objective(contracted)
            if contracted_value <= reflected_value:
                simplex[-1], values[-1] = contracted, contracted_value
                continue
        else:
            contracted = blend(worst, rho)
            contracted_value = objective(contracted)
            if contracted_value < values[-1]:
                simplex[-1], values[-1] = contracted, contracted_value
                continue

        best = simplex[0]
        simplex = [best] + [
            tuple(b + sigma * (p - b) for b, p in zip(best, point, strict=True))
            for point in simplex[1:]
        ]
        values = [values[0]] + [objective(p) for p in simplex[1:]]

    order = sorted(range(n + 1), key=lambda i: values[i])
    return SimplexResult(simplex[order[0]], values[order[0]], iterations, False)
