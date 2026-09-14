"""The standard normal density, distribution and quantile functions.

Everything in this package that touches a probability goes through this module,
so the accuracy of the option prices is bounded by the accuracy here.

Two choices are worth stating.

The distribution function is written on ``math.erfc`` rather than on ``erf``.
The two are equivalent in exact arithmetic, but ``0.5 * (1 + erf(x / sqrt(2)))``
loses the left tail to cancellation: for x = -8 the true value is near 6.2e-16,
while ``erf`` returns a number within rounding distance of -1 and the sum
collapses to zero. ``0.5 * erfc(-x / sqrt(2))`` evaluates the same quantity
without the subtraction and stays accurate until the result underflows near
x = -38. Deep-wing option prices depend on exactly that tail.

The quantile function is Acklam's rational approximation, which is accurate to
about 1.15e-9 relative, followed by one Halley step against ``norm_cdf``. The
refinement is cubically convergent, so a starting error of 1e-9 lands at full
double precision in the single step, and it costs one ``erfc`` evaluation.

One consequence of working in double precision is worth stating, because it
looks like an inaccuracy in ``norm_ppf`` and is not. Inverting a probability
near one is ill conditioned at the input, not in the algorithm. ``norm_cdf(6)``
is 0.999999999013…, and the neighbouring doubles are 1.1e-16 apart; that spacing
is a step of ``ulp(p) / phi(x)`` in x, or 1.8e-8 at x = 6, rising to 2.2e-2 at
x = 8. The information is destroyed when the probability is stored, before this
module is called, so no algorithm recovers it. Small probabilities carry their
full relative precision, so a caller wanting the far right tail should pass the
small probability and negate: ``-norm_ppf(q)`` rather than ``norm_ppf(1 - q)``.
"""

from __future__ import annotations

import math

__all__ = ["norm_cdf", "norm_pdf", "norm_ppf"]

_SQRT_2 = math.sqrt(2.0)
_SQRT_2PI = math.sqrt(2.0 * math.pi)
_INV_SQRT_2PI = 1.0 / _SQRT_2PI

# Beyond this the density underflows to zero in double precision.
_PDF_CUTOFF = 40.0

_A = (
    -3.969683028665376e01,
    2.209460984245205e02,
    -2.759285104469687e02,
    1.383577518672690e02,
    -3.066479806614716e01,
    2.506628277459239e00,
)
_B = (
    -5.447609879822406e01,
    1.615858368580409e02,
    -1.556989798598866e02,
    6.680131188771972e01,
    -1.328068155288572e01,
)
_C = (
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e00,
    -2.549732539343734e00,
    4.374664141464968e00,
    2.938163982698783e00,
)
_D = (
    7.784695709041462e-03,
    3.224671290700398e-01,
    2.445134137142996e00,
    3.754408661907416e00,
)

# Where Acklam switches from the tail branch to the central branch.
_P_LOW = 0.02425
_P_HIGH = 1.0 - _P_LOW


def norm_pdf(x: float) -> float:
    """The standard normal density.

    Guarded against overflow in ``exp`` for large negative arguments to the
    exponential; past forty standard deviations the density is zero in double
    precision and computing it is only a way to raise.
    """
    if x != x:  # NaN propagates rather than being compared into a branch.
        return x
    if x < -_PDF_CUTOFF or x > _PDF_CUTOFF:
        return 0.0
    return _INV_SQRT_2PI * math.exp(-0.5 * x * x)


def norm_cdf(x: float) -> float:
    """The standard normal distribution function, accurate in both tails."""
    if x != x:
        return x
    return 0.5 * math.erfc(-x / _SQRT_2)


def norm_ppf(p: float) -> float:
    """The standard normal quantile function.

    Raises:
        ValueError: if ``p`` lies outside [0, 1] or is not a number. The closed
            endpoints return infinities, which is the mathematically correct
            limit and lets callers use them as sentinels.
    """
    if p != p:
        raise ValueError("norm_ppf is undefined at NaN")
    if p < 0.0 or p > 1.0:
        raise ValueError(f"norm_ppf requires a probability in [0, 1], got {p}")
    if p == 0.0:
        return -math.inf
    if p == 1.0:
        return math.inf

    if p < _P_LOW:
        q = math.sqrt(-2.0 * math.log(p))
        x = (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / (
            (((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0
        )
    elif p > _P_HIGH:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = -(((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / (
            (((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0
        )
    else:
        q = p - 0.5
        r = q * q
        x = (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * q / (
            ((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1.0
        )

    # One Halley step. The density is the derivative of the residual, so the
    # correction is written in terms of exp(x^2/2) directly; for |x| large the
    # density underflows and the starting point is already at the accuracy the
    # tail can carry, so the step is skipped rather than dividing by zero.
    density = norm_pdf(x)
    if density > 0.0:
        residual = norm_cdf(x) - p
        step = residual / density
        x -= step / (1.0 + 0.5 * x * step)
    return x
