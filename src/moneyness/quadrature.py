"""Gauss-Legendre quadrature, adaptive, and over a semi-infinite range.

The transform pricing in :mod:`moneyness.heston` is an integral over the half
line with no closed form, and this package has no runtime dependencies, so the
integrator has to be here. Three decisions are worth recording.

**The nodes are computed, not tabulated.** A table fixes the order, and the
order is exactly the thing an adaptive rule needs to vary. Newton's method on
the Legendre polynomial converges in four or five iterations from the Chebyshev
starting point ``cos(pi (i - 1/4) / (n + 1/2))``, and the recurrence that
evaluates ``P_n`` gives ``P_n'`` from the same two terms, so the derivative is
free. Orders are cached, because a pricing loop asks for the same one every
time.

**The adaptive rule compares a panel against its own halves rather than against
a higher order on the same panel.** Both detect a panel that is not yet
resolved; bisection also fixes it, and the two halves are the work the next step
needs anyway. The error estimate is the difference between the two, which is
pessimistic for a smooth integrand — by roughly the ratio of the two rules'
error constants — and that is the right direction to be wrong in.

**The half line is cut, not mapped.** A substitution such as ``u = x / (1 - x)``
turns the tail into an endpoint singularity in disguise: the integrand is smooth
but its image is compressed into the last few per cent of the interval, and a
fixed rule there is guessing. Instead the range is extended by doubling until
the last panel contributes less than the tolerance, which for a decaying
integrand is a statement about the integrand rather than about the map. The
extension stops at a limit and says so, rather than looping, because an
integrand that does not decay is a caller error and should be reported as one.
"""

from __future__ import annotations

import math
from collections.abc import Callable

__all__ = [
    "QuadratureError",
    "adaptive_quad",
    "fixed_quad",
    "gauss_legendre",
    "semi_infinite_quad",
]

# Deeper than this and the panels are smaller than the rounding of their own
# endpoints, so the recursion cannot make progress.
_MAX_DEPTH = 40

# A budget on panels, not only on depth. Depth alone bounds the work at two to
# the depth, which is not a bound: an integrand that is rough everywhere —
# a transform evaluated through a cancelling expression, say — bisects on every
# branch and runs for hours rather than failing. The budget turns that into an
# error naming the tolerance, which is the thing the caller can act on.
_MAX_PANELS = 50_000

# The half line is cut here at the latest. Reaching it means the integrand did
# not decay, which is a statement about the caller's function.
_MAX_EXTENSIONS = 60


class QuadratureError(ValueError):
    """An integral that the rule declines to answer for."""


_CACHE: dict[int, tuple[tuple[float, ...], tuple[float, ...]]] = {}


def _legendre(order: int, z: float) -> tuple[float, float]:
    """``P_n(z)`` and ``P_n'(z)``, from one pass of Bonnet's recurrence.

    The derivative comes out of the same two terms the recurrence already
    carries, through ``(z^2 - 1) P_n' = n (z P_n - P_{n-1})``, so it is free.
    That identity is singular at the endpoints, which the Legendre roots never
    reach.
    """
    previous, current = 0.0, 1.0
    for j in range(1, order + 1):
        older, previous = previous, current
        current = ((2.0 * j - 1.0) * z * previous - (j - 1.0) * older) / j
    return current, order * (z * current - previous) / (z * z - 1.0)


def gauss_legendre(order: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Nodes and weights of the ``order``-point Gauss-Legendre rule on [-1, 1].

    The rule is exact for polynomials of degree ``2 * order - 1``. Nodes come
    back in increasing order and the weights sum to 2.

    Args:
        order: Number of points. At least one.

    Returns:
        A pair ``(nodes, weights)``, each of length ``order``.

    Raises:
        QuadratureError: If ``order`` is less than one.
    """
    if order < 1:
        raise QuadratureError(f"order must be at least 1, got {order}")
    cached = _CACHE.get(order)
    if cached is not None:
        return cached

    nodes = [0.0] * order
    weights = [0.0] * order
    # The rule is symmetric, so only the non-negative half is solved for.
    half = (order + 1) // 2
    for i in range(1, half + 1):
        # Chebyshev points interlace the Legendre roots closely enough that
        # Newton converges from them without safeguarding.
        z = math.cos(math.pi * (i - 0.25) / (order + 0.5))
        for _ in range(100):
            value, derivative = _legendre(order, z)
            step = value / derivative
            z -= step
            if abs(step) <= 1e-15:
                break
        # Re-evaluated at the converged point. Reusing the derivative from the
        # last iteration leaves it one Newton step stale, which is invisible in
        # the node and shows up in the weight, where the derivative is squared:
        # it costs about twenty ulps at order three.
        _, derivative = _legendre(order, z)
        index = order - i
        nodes[i - 1] = -z
        nodes[index] = z
        weight = 2.0 / ((1.0 - z * z) * derivative * derivative)
        weights[i - 1] = weight
        weights[index] = weight

    result = (tuple(nodes), tuple(weights))
    _CACHE[order] = result
    return result


def fixed_quad(f: Callable[[float], float], a: float, b: float, order: int = 20) -> float:
    """Integrate ``f`` over ``[a, b]`` with one Gauss-Legendre panel.

    Args:
        f: Integrand.
        a: Lower limit.
        b: Upper limit. May be below ``a``, which negates the result.
        order: Points in the rule.

    Returns:
        The estimated integral.
    """
    if a == b:
        return 0.0
    nodes, weights = gauss_legendre(order)
    half_width = 0.5 * (b - a)
    centre = 0.5 * (a + b)
    total = 0.0
    for node, weight in zip(nodes, weights, strict=True):
        total += weight * f(centre + half_width * node)
    return half_width * total


class _Budget:
    """A mutable count of panels, shared down one recursion."""

    __slots__ = ("left",)

    def __init__(self, panels: int) -> None:
        self.left = panels

    def spend(self) -> None:
        self.left -= 1
        if self.left < 0:
            raise QuadratureError(
                "the integrand did not resolve within the panel budget; it is "
                "either not smooth or the tolerance is below what double "
                "precision can deliver for it"
            )


def _adaptive(
    f: Callable[[float], float],
    a: float,
    b: float,
    whole: float,
    tol: float,
    order: int,
    depth: int,
    budget: _Budget,
) -> float:
    budget.spend()
    mid = 0.5 * (a + b)
    left = fixed_quad(f, a, mid, order)
    right = fixed_quad(f, mid, b, order)
    total = left + right
    if abs(total - whole) <= tol or depth >= _MAX_DEPTH:
        # Richardson is not available without knowing the rule's order of
        # accuracy on this integrand, so the refined value is returned as is.
        return total
    half_tol = 0.5 * tol
    return _adaptive(f, a, mid, left, half_tol, order, depth + 1, budget) + _adaptive(
        f, mid, b, right, half_tol, order, depth + 1, budget
    )


def adaptive_quad(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-12,
    order: int = 16,
    budget: _Budget | None = None,
) -> float:
    """Integrate ``f`` over ``[a, b]``, bisecting panels until they agree.

    Args:
        f: Integrand.
        a: Lower limit.
        b: Upper limit.
        tol: Absolute tolerance on the whole integral.
        order: Points in each panel's rule.
        budget: Panels the recursion may spend. A fresh one is made when this
            is omitted; the half-line rule passes its own so the whole walk
            shares a single allowance.

    Returns:
        The estimated integral.

    Raises:
        QuadratureError: If ``tol`` is not positive, a limit is not finite, or
            the panel budget runs out before the integrand resolves.
    """
    if not (tol > 0.0):
        raise QuadratureError(f"tol must be positive, got {tol}")
    for name, value in (("a", a), ("b", b)):
        if value != value or math.isinf(value):
            raise QuadratureError(f"{name} must be finite, got {value}")
    if a == b:
        return 0.0
    if budget is None:
        budget = _Budget(_MAX_PANELS)
    return _adaptive(f, a, b, fixed_quad(f, a, b, order), tol, order, 0, budget)


def semi_infinite_quad(
    f: Callable[[float], float],
    a: float = 0.0,
    tol: float = 1e-12,
    order: int = 16,
    initial_width: float = 1.0,
) -> float:
    """Integrate ``f`` from ``a`` to infinity.

    The range is covered by panels whose width doubles, each integrated to
    ``tol``, and the walk stops once a panel contributes less than ``tol``
    against a running total that has stopped moving. Two consecutive negligible
    panels are required, because an oscillatory integrand can cross zero over a
    single panel and look finished when it is not.

    Args:
        f: Integrand. Must decay.
        a: Lower limit.
        tol: Absolute tolerance.
        order: Points in each panel's rule.
        initial_width: Width of the first panel.

    Returns:
        The estimated integral.

    Raises:
        QuadratureError: If the integrand has not decayed by the last panel.
    """
    if not (initial_width > 0.0):
        raise QuadratureError(f"initial_width must be positive, got {initial_width}")
    total = 0.0
    lower = a
    width = initial_width
    quiet = 0
    # One budget for the whole half line, so a walk that bisects hard on every
    # panel is caught rather than repeating the same failure sixty times.
    budget = _Budget(_MAX_PANELS)
    for _ in range(_MAX_EXTENSIONS):
        upper = lower + width
        piece = adaptive_quad(f, lower, upper, tol=tol, order=order, budget=budget)
        total += piece
        if abs(piece) <= tol:
            quiet += 1
            if quiet >= 2:
                return total
        else:
            quiet = 0
        lower = upper
        width *= 2.0
    raise QuadratureError(
        f"integrand had not decayed by {lower}: the last panel contributed {piece}, "
        f"which is above the tolerance {tol}"
    )
