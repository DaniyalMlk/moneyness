"""High-precision reference implementations, used only by the tests.

The point of this module is independence. The library under test is written for
speed in double precision, with tail-safe rearrangements and limit cases handled
by branching. If the tests checked it against constants transcribed from a book,
they would be checking transcription. Instead they check it against the same
mathematics evaluated at fifty decimal digits by ``mpmath``, which shares no
code, no rearrangement and no branch structure with the implementation.

Agreement to a stated tolerance across a grid of inputs is then evidence about
the implementation rather than about the person who typed the expected values.
"""

from __future__ import annotations

from mpmath import erfc, exp, log, mp, mpf, sqrt

mp.dps = 50


def ref_norm_cdf(x: float) -> float:
    """The normal distribution function at fifty digits, rounded to double."""
    return float(erfc(-mpf(x) / sqrt(2)) / 2)


def ref_bsm(
    spot: float,
    strike: float,
    time: float,
    rate: float,
    vol: float,
    carry: float,
    is_call: bool,
) -> float:
    """The generalised Black-Scholes-Merton price at fifty digits."""
    s, k = mpf(spot), mpf(strike)
    t, r, v, b = mpf(time), mpf(rate), mpf(vol), mpf(carry)

    if t == 0 or v == 0 or s == 0 or k == 0:
        fwd = s * exp(b * t)
        sign = mpf(1) if is_call else mpf(-1)
        return float(max(exp(-r * t) * sign * (fwd - k), mpf(0)))

    sd = v * sqrt(t)
    d1 = (log(s / k) + (b + v**2 / 2) * t) / sd
    d2 = d1 - sd
    sign = mpf(1) if is_call else mpf(-1)

    def n(x: object) -> object:
        return erfc(-x / sqrt(2)) / 2

    return float(
        sign * (s * exp((b - r) * t) * n(sign * d1) - k * exp(-r * t) * n(sign * d2))
    )
