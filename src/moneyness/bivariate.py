"""The bivariate normal distribution function.

One function, built carefully, because everything that uses it inherits its
accuracy and because the place it is needed is the awkward one. The two-step
exercise boundary in :mod:`moneyness.american` evaluates it at correlations
close to minus one, where the joint density collapses onto a line and the
straightforward approaches stop working.

The construction starts from Sheppard's identity, which says that the
derivative of the distribution function with respect to the correlation is the
*density*:

    d/drho Phi2(a, b; rho) = phi2(a, b; rho)

Integrating from zero, where the variables are independent and the answer is a
product of marginals:

    Phi2(a, b; rho) = Phi(a) Phi(b) + int_0^rho phi2(a, b; t) dt

That is exact, but the integrand carries a ``1 / sqrt(1 - t^2)`` that blows up
as the correlation approaches one — which is the region of interest, so it
cannot simply be avoided. The substitution ``t = sin(theta)`` removes it
entirely: the Jacobian ``cos(theta)`` cancels the square root, leaving

    Phi2(a, b; rho) = Phi(a) Phi(b)
                      + 1/(2 pi) int_0^{asin(rho)}
                            exp(-(a^2 + b^2 - 2ab sin(theta)) / (2 cos^2(theta))) dtheta

and an integrand that is smooth, strictly positive, and bounded above by one
everywhere on a closed interval. The exponent's numerator tends to ``(a - b)^2``
as theta approaches a right angle while its denominator tends to zero, so the
integrand tends to zero whenever the arguments differ and to one when they do
not. Nothing is singular; the only difficulty left is that for correlations very
near one the integrand develops a boundary layer at the top of the range, and a
rule with its points spread evenly will miss it.

So the quadrature is a composite Gauss-Legendre rule over panels that are
graded towards that end. The nodes are computed from the Legendre polynomials at
import rather than transcribed from a table, which removes a class of error that
is invisible on inspection and hard to find afterwards.

The accuracy is measured rather than claimed: the tests check the result against
``mpmath`` evaluating the same double integral at fifty digits, over a grid that
includes correlations of plus and minus 0.9999 and arguments out to eight
standard deviations.
"""

from __future__ import annotations

import math
from itertools import pairwise

from .normal import norm_cdf

__all__ = ["norm_cdf2"]

# Nodes per panel. Twenty is comfortably past the point where adding more stops
# helping for an integrand this smooth; the panels do the remaining work.
_ORDER = 20

# Panels across the integration range. Graded towards the upper endpoint, where
# a near-unit correlation puts the boundary layer.
_PANELS = 12


def _legendre(order: int, x: float) -> tuple[float, float]:
    """The Legendre polynomial of the given order at ``x``, and its derivative.

    Built by the three-term recurrence, which is stable upwards and costs
    nothing at these orders.
    """
    previous, current = 1.0, x
    for n in range(2, order + 1):
        previous, current = current, ((2 * n - 1) * x * current - (n - 1) * previous) / n
    derivative = order * (x * current - previous) / (x * x - 1.0)
    return current, derivative


def _gauss_legendre(order: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Nodes and weights for the Gauss-Legendre rule of the given order on [-1, 1].

    Computed rather than transcribed. A table of twenty nodes to sixteen digits
    is forty numbers that all look equally plausible, and a single wrong digit
    produces a rule that is slightly wrong everywhere — accurate enough to look
    fine and never exactly right. Newton's method on the Legendre polynomial,
    started from the standard Chebyshev-like approximation, converges in three
    or four iterations and is checkable by the properties the rule must satisfy,
    which the tests assert.
    """
    nodes: list[float] = []
    weights: list[float] = []
    for i in range(1, order + 1):
        x = math.cos(math.pi * (i - 0.25) / (order + 0.5))
        for _ in range(100):
            value, derivative = _legendre(order, x)
            step = value / derivative
            x -= step
            if abs(step) < 1e-16:
                break
        _, derivative = _legendre(order, x)
        nodes.append(x)
        weights.append(2.0 / ((1.0 - x * x) * derivative * derivative))
    return tuple(nodes), tuple(weights)


_NODES, _WEIGHTS = _gauss_legendre(_ORDER)


def _panel_edges(upper: float) -> list[float]:
    """Panel boundaries on ``[0, upper]``, graded towards ``upper``.

    The grading is a square law on the fraction of the interval, measured from
    the far end. For a correlation near one the integrand is flat across most of
    the range and turns over sharply in the last few percent of it; an even
    split would put one panel on the part that matters and eleven on the part
    that does not.
    """
    if upper == 0.0:
        return [0.0]
    return [upper * (1.0 - (1.0 - i / _PANELS) ** 2) for i in range(_PANELS + 1)]


def norm_cdf2(a: float, b: float, rho: float) -> float:
    """``P(X <= a, Y <= b)`` for standard normals with correlation ``rho``.

    Args:
        a: First upper limit.
        b: Second upper limit.
        rho: Correlation, in ``[-1, 1]``. The closed endpoints are the degenerate
            cases where the two variables are equal or exactly opposed, and both
            have closed forms that are returned directly rather than approached
            by quadrature.

    Returns:
        The joint probability, in ``[0, 1]``.

    Raises:
        ValueError: If ``rho`` is outside ``[-1, 1]``, or any argument is NaN.
    """
    if a != a or b != b or rho != rho:
        raise ValueError("norm_cdf2 is undefined at NaN")
    if rho < -1.0 or rho > 1.0:
        raise ValueError(f"rho must lie in [-1, 1], got {rho}")

    # Infinite limits collapse the problem onto a marginal, and are worth
    # handling explicitly: they arise naturally wherever one of two conditions
    # is vacuous, and the quadrature below would otherwise be asked to integrate
    # exp(-inf).
    if a == -math.inf or b == -math.inf:
        return 0.0
    if a == math.inf:
        return norm_cdf(b)
    if b == math.inf:
        return norm_cdf(a)

    if rho == 1.0:
        # The variables are the same variable.
        return norm_cdf(min(a, b))
    if rho == -1.0:
        # Y = -X, so the event is -b <= X <= a, empty when a < -b.
        return max(0.0, norm_cdf(a) - norm_cdf(-b))

    product = norm_cdf(a) * norm_cdf(b)
    if rho == 0.0:
        return product

    upper = math.asin(rho)
    squares = a * a + b * b
    cross = 2.0 * a * b

    total = 0.0
    edges = _panel_edges(upper) if upper > 0.0 else _panel_edges(-upper)
    sign = 1.0 if upper > 0.0 else -1.0
    for lower_edge, upper_edge in pairwise(edges):
        half = (upper_edge - lower_edge) / 2.0
        middle = (upper_edge + lower_edge) / 2.0
        for node, weight in zip(_NODES, _WEIGHTS, strict=True):
            theta = sign * (middle + half * node)
            sine = math.sin(theta)
            cosine_squared = 1.0 - sine * sine
            if cosine_squared <= 0.0:
                continue
            exponent = -(squares - cross * sine) / (2.0 * cosine_squared)
            # Guard only against underflow to zero, which is the correct limit
            # here and which exp would otherwise raise on for large negative
            # arguments on some platforms.
            total += weight * half * (math.exp(exponent) if exponent > -700.0 else 0.0)

    result = product + sign * total / (2.0 * math.pi)
    # Quadrature can leave the result a rounding error outside the unit
    # interval, which is meaningless for a probability and awkward for callers
    # that go on to take a logarithm of it.
    return min(1.0, max(0.0, result))
