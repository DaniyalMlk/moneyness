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

from typing import Any

from mpmath import diff, erfc, exp, log, mp, mpf, pi, quad, sqrt

mp.dps = 50

PARAMETERS = ("spot", "strike", "time", "rate", "vol", "carry")


def ref_norm_cdf(x: float) -> float:
    """The normal distribution function at fifty digits, rounded to double."""
    return float(erfc(-mpf(x) / sqrt(2)) / 2)


def ref_log_moneyness(spot: float, strike: float) -> float:
    """``log(S / K)`` at fifty digits, rounded once to double."""
    return float(log(mpf(spot) / mpf(strike)))


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

    n_d1 = erfc(-(sign * d1) / sqrt(2)) / 2
    n_d2 = erfc(-(sign * d2) / sqrt(2)) / 2
    return float(sign * (s * exp((b - r) * t) * n_d1 - k * exp(-r * t) * n_d2))


def mp_bsm(
    spot: Any,
    strike: Any,
    time: Any,
    rate: Any,
    vol: Any,
    carry: Any,
    is_call: bool,
) -> Any:
    """The price, kept in mpmath's arbitrary precision rather than rounded.

    Separate from :func:`ref_bsm` because differentiating it numerically needs
    the extra digits: a central difference gives back roughly two thirds of the
    precision it is handed, so starting from fifty digits leaves well over
    thirty in the derivative, and the comparison against the analytic Greek is
    then limited by the analytic side rather than by this one.
    """
    s, k, t = mpf(spot), mpf(strike), mpf(time)
    r, v, b = mpf(rate), mpf(vol), mpf(carry)
    sign = mpf(1) if is_call else mpf(-1)

    if t == 0 or v == 0 or s == 0 or k == 0:
        return max(exp(-r * t) * sign * (s * exp(b * t) - k), mpf(0))

    sd = v * sqrt(t)
    d1 = (log(s / k) + (b + v**2 / 2) * t) / sd
    d2 = d1 - sd
    n_d1 = erfc(-(sign * d1) / sqrt(2)) / 2
    n_d2 = erfc(-(sign * d2) / sqrt(2)) / 2
    return sign * (s * exp((b - r) * t) * n_d1 - k * exp(-r * t) * n_d2)


def ref_derivative(
    params: dict[str, float],
    is_call: bool,
    wrt: tuple[str, ...],
    orders: tuple[int, ...],
) -> float:
    """A partial derivative of the price, taken numerically at high precision.

    ``wrt`` names the parameters to differentiate against and ``orders`` gives
    the order in each, so ``(("spot", "vol"), (1, 1))`` is vanna and
    ``(("spot",), (3,))`` is speed.

    Differentiating the reference rather than comparing against a second
    closed form is deliberate. A second closed form would be another chance to
    make the same algebra mistake twice; a numerical derivative of the price
    shares nothing with the analytic Greek except the price itself, so an error
    in the Greek cannot hide.
    """
    fixed = dict(params)

    def at(*values: Any) -> Any:
        local = dict(fixed)
        for name, value in zip(wrt, values, strict=True):
            local[name] = value
        return mp_bsm(
            local["spot"],
            local["strike"],
            local["time"],
            local["rate"],
            local["vol"],
            local["carry"],
            is_call,
        )

    point = tuple(mpf(fixed[name]) for name in wrt)
    if len(wrt) == 1:
        return float(diff(at, point[0], orders[0]))
    return float(diff(at, point, orders))


def ref_black(forward: float, strike: float, total_vol: float, is_call: bool) -> float:
    """The undiscounted Black price in total-volatility coordinates, at fifty digits."""
    f, k, w = mpf(forward), mpf(strike), mpf(total_vol)
    sign = mpf(1) if is_call else mpf(-1)
    if w <= 0:
        return float(max(sign * (f - k), mpf(0)))
    d1 = log(f / k) / w + w / 2
    d2 = d1 - w
    n_d1 = erfc(-(sign * d1) / sqrt(2)) / 2
    n_d2 = erfc(-(sign * d2) / sqrt(2)) / 2
    return float(sign * (f * n_d1 - k * n_d2))


def ref_black_vega(forward: float, strike: float, total_vol: float) -> float:
    """``F phi(d1)`` evaluated at fifty digits.

    Deliberately the closed form rather than a numerical derivative, and the
    exception to this module's usual rule. Vega decays like ``e^{-d1^2 / 2}``,
    so differentiating it numerically is severely ill conditioned in the wings:
    at a strike of 20 against a forward of 100 with a total volatility of 0.1
    the true value is 1.008e-55, and ``mpmath``'s ``diff`` returns it wrong by
    4.7e-4 relative even carrying fifty digits, while the implementation under
    test is accurate to 3.1e-14. A numerical oracle there would be measuring the
    oracle.

    The derivative relationship is not taken on trust because of this; it is
    asserted separately by :func:`ref_black_derivative`, over the range of
    inputs where numerical differentiation is trustworthy.
    """
    f, k, w = mpf(forward), mpf(strike), mpf(total_vol)
    d1 = log(f / k) / w + w / 2
    return float(f * exp(-(d1**2) / 2) / sqrt(2 * pi))


def ref_black_derivative(forward: float, strike: float, total_vol: float) -> float:
    """``d(black)/d(total_vol)`` by numerical differentiation at fifty digits.

    Trustworthy only where the price is not vanishingly small; used to confirm
    that :func:`ref_black_vega`'s closed form really is the derivative.
    """
    f, k = mpf(forward), mpf(strike)

    def at(w: Any) -> Any:
        d1 = log(f / k) / w + w / 2
        d2 = d1 - w
        return f * (erfc(-d1 / sqrt(2)) / 2) - k * (erfc(-d2 / sqrt(2)) / 2)

    return float(diff(at, mpf(total_vol)))


def ref_norm_cdf2(a: float, b: float, rho: float) -> float:
    """The bivariate normal distribution function at fifty digits.

    Reduced to a single integral over the first variable, with the second
    integrated out analytically:

        Phi2(a, b; rho) = int_{-inf}^{a} phi(x) Phi((b - rho x) / sqrt(1 - rho^2)) dx

    The subdivision points are not decoration. As the correlation approaches
    one in absolute value the inner distribution function becomes a step of
    width ``sqrt(1 - rho^2)`` centred at ``x = b / rho``, and an adaptive
    quadrature handed the whole half-line will step over it and return a
    confidently wrong answer. A first version of this function did exactly
    that: it disagreed with the implementation by one part in a hundred at
    ``rho = -0.9999``, and the implementation was right. Naming the transition
    is what makes this an oracle rather than a second opinion.
    """
    left, right, correlation = mpf(a), mpf(b), mpf(rho)
    if correlation <= -1:
        return max(0.0, ref_norm_cdf(a) - ref_norm_cdf(-b))
    if correlation >= 1:
        return ref_norm_cdf(min(a, b))

    spread = sqrt(1 - correlation * correlation)

    def integrand(x: Any) -> Any:
        inner = (right - correlation * x) / spread
        return exp(-x * x / 2) / sqrt(2 * pi) * erfc(-inner / sqrt(2)) / 2

    points = [mpf(-45)]
    if correlation != 0:
        transition = right / correlation
        points.extend(transition + step * spread for step in (-8, -4, -2, -1, 0, 1, 2, 4, 8))
    points.extend(mpf(x) for x in (-6, -3, -1, 0, 1, 3, 6))
    inside = sorted({p for p in points if mpf(-45) < p < left})
    return float(quad(integrand, [mpf(-45), *inside, left]))
