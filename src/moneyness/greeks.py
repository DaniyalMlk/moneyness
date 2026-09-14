"""Analytic sensitivities of the Black-Scholes-Merton price.

Every function here is a partial derivative of :func:`moneyness.bsm.price`, and
every one is checked in the test suite against a central difference of that same
function. The closed forms are the implementation; the finite differences are
the specification.

**Sign conventions**, since they are the usual source of confusion. Derivatives
with respect to calendar time are written as derivatives with respect to time to
expiry and then negated, which is what the market means by decay: ``theta`` is
``-dV/dT``, and it is negative for a long option, whose value falls as expiry
approaches. ``charm``, ``veta`` and ``colour`` follow the same convention, being
``-d(delta)/dT``, ``-d(vega)/dT`` and ``-d(gamma)/dT``.

**Scaling.** Everything is a mathematical derivative in the natural units of its
argument: vega is per unit of volatility, so per 1.00 rather than per
volatility point, and theta is per year rather than per day. Rescaling is the
caller's business, and doing it here would make the finite-difference checks
test the rescaling instead of the mathematics.

**Rates.** The discount rate and the cost of carry are separate arguments to the
model, so they get separate derivatives. :func:`rho` moves the discount rate and
holds the carry; :func:`rho_carry` moves the carry and holds the rate. For a
share paying no dividend the two are locked together at ``b = r``, and the
sensitivity to a parallel move is the sum — which is the textbook rho, and is
asserted as such in the tests.

One identity does most of the work below. At the same inputs

    S e^{(b-r)T} phi(d1) = K e^{-rT} phi(d2)

which is why the terms in ``phi`` cancel out of delta, dual delta and the carry
rho, and why gamma and vega can each be written in two equivalent ways. It is
verified directly in the test suite rather than assumed.
"""

from __future__ import annotations

import math

from .bsm import Inputs, OptionType, d1_d2, price
from .normal import norm_cdf, norm_pdf

__all__ = [
    "charm",
    "colour",
    "delta",
    "dual_delta",
    "dual_gamma",
    "forward_delta",
    "gamma",
    "rho",
    "rho_carry",
    "speed",
    "theta",
    "vanna",
    "vega",
    "veta",
    "volga",
    "zomma",
]


def _carry_discount(inputs: Inputs) -> float:
    """``e^{(b - r)T}``, the factor multiplying spot in the price."""
    return math.exp((inputs.b - inputs.rate) * inputs.time)


def _require_live(inputs: Inputs, name: str) -> None:
    """Reject the degenerate inputs, where the derivative is not defined.

    At zero volatility or zero time the price is a kinked function of spot: the
    derivative is zero on one side of the strike, one on the other, and does not
    exist at it. Returning a number would be inventing one, so these raise and
    leave the caller to decide what a delta means for an expired option.
    """
    if inputs.is_degenerate:
        raise ValueError(
            f"{name} is undefined when time, volatility, spot or strike is zero: "
            "the payoff is kinked at the strike and the derivative does not exist there"
        )


def delta(inputs: Inputs, option: OptionType) -> float:
    """``dV/dS``, the sensitivity to the underlying."""
    _require_live(inputs, "delta")
    d1, _ = d1_d2(inputs)
    sign = option.sign
    return sign * _carry_discount(inputs) * norm_cdf(sign * d1)


def gamma(inputs: Inputs) -> float:
    """``d2V/dS2``. The same for a call and a put, by put-call parity.

    Parity says the two prices differ by ``(F - K)e^{-rT}``, which is linear in
    spot; a linear function has no curvature, so every second and higher
    derivative in spot is shared. The tests assert this rather than assuming it.
    """
    _require_live(inputs, "gamma")
    d1, _ = d1_d2(inputs)
    return _carry_discount(inputs) * norm_pdf(d1) / (inputs.spot * inputs.std_dev)


def vega(inputs: Inputs) -> float:
    """``dV/dsigma``, per unit of volatility. Shared by calls and puts."""
    _require_live(inputs, "vega")
    d1, _ = d1_d2(inputs)
    return inputs.spot * _carry_discount(inputs) * norm_pdf(d1) * math.sqrt(inputs.time)


def theta(inputs: Inputs, option: OptionType) -> float:
    """``-dV/dT``, the decay per year.

    Negative for a long position in almost every case, the exception being a
    deep in-the-money European put, whose holder is waiting to be paid the
    strike and gains as that wait shortens.
    """
    _require_live(inputs, "theta")
    d1, d2 = d1_d2(inputs)
    sign = option.sign
    cd = _carry_discount(inputs)
    decay = inputs.spot * cd * norm_pdf(d1) * inputs.vol / (2.0 * math.sqrt(inputs.time))
    carry_term = (inputs.b - inputs.rate) * inputs.spot * cd * norm_cdf(sign * d1)
    discount_term = inputs.rate * inputs.strike * inputs.discount * norm_cdf(sign * d2)
    return -decay - sign * (carry_term + discount_term)


def rho(inputs: Inputs, option: OptionType) -> float:
    """``dV/dr`` with the cost of carry held fixed.

    With ``b`` independent of ``r`` the rate enters only through the two
    discount factors, both of which carry a factor ``e^{-rT}``, so the whole
    price scales and the derivative collapses to ``-T V``. That is exact rather
    than an approximation, and it is the right sensitivity for an option on a
    future, where the carry is structurally zero and does not follow the rate.
    """
    _require_live(inputs, "rho")
    return -inputs.time * price(inputs, option)


def rho_carry(inputs: Inputs, option: OptionType) -> float:
    """``dV/db`` with the discount rate held fixed.

    Equal to ``T S delta``. The terms in ``phi`` cancel through the identity in
    the module docstring, leaving only the direct dependence of the carried
    spot on ``b``.

    For a share paying a continuous dividend ``q`` the carry is ``r - q``, so the
    sensitivity to the dividend is the negative of this.
    """
    _require_live(inputs, "rho_carry")
    return inputs.time * inputs.spot * delta(inputs, option)


def dual_delta(inputs: Inputs, option: OptionType) -> float:
    """``dV/dK``, the sensitivity to the strike.

    Minus the discounted risk-neutral probability of finishing in the money,
    which is what makes it the quantity a strike ladder differentiates to
    recover the implied distribution.
    """
    _require_live(inputs, "dual_delta")
    _, d2 = d1_d2(inputs)
    sign = option.sign
    return -sign * inputs.discount * norm_cdf(sign * d2)


def dual_gamma(inputs: Inputs) -> float:
    """``d2V/dK2``, proportional to the risk-neutral density at the strike."""
    _require_live(inputs, "dual_gamma")
    _, d2 = d1_d2(inputs)
    return inputs.discount * norm_pdf(d2) / (inputs.strike * inputs.std_dev)


def vanna(inputs: Inputs) -> float:
    """``d2V/dS dsigma``: how delta moves when volatility does."""
    _require_live(inputs, "vanna")
    d1, d2 = d1_d2(inputs)
    return -_carry_discount(inputs) * norm_pdf(d1) * d2 / inputs.vol


def volga(inputs: Inputs) -> float:
    """``d2V/dsigma2``, sometimes called vomma.

    Zero where ``d1 d2 = 0`` and positive in both wings, which is the analytic
    statement of why a strangle is long convexity in volatility.
    """
    _require_live(inputs, "volga")
    d1, d2 = d1_d2(inputs)
    return vega(inputs) * d1 * d2 / inputs.vol


def charm(inputs: Inputs, option: OptionType) -> float:
    """``-d(delta)/dT``: the drift of delta as expiry approaches."""
    _require_live(inputs, "charm")
    d1, d2 = d1_d2(inputs)
    cd = _carry_discount(inputs)
    d1_by_time = inputs.b / inputs.std_dev - d2 / (2.0 * inputs.time)
    return -((inputs.b - inputs.rate) * delta(inputs, option) + cd * norm_pdf(d1) * d1_by_time)


def veta(inputs: Inputs) -> float:
    """``-d(vega)/dT``: how the volatility exposure decays."""
    _require_live(inputs, "veta")
    d1, d2 = d1_d2(inputs)
    bracket = (
        (inputs.b - inputs.rate)
        - inputs.b * d1 / inputs.std_dev
        + (1.0 + d1 * d2) / (2.0 * inputs.time)
    )
    return -vega(inputs) * bracket


def colour(inputs: Inputs) -> float:
    """``-d(gamma)/dT``: how curvature concentrates as expiry approaches."""
    _require_live(inputs, "colour")
    d1, d2 = d1_d2(inputs)
    bracket = (
        (inputs.b - inputs.rate)
        - inputs.b * d1 / inputs.std_dev
        + (d1 * d2 - 1.0) / (2.0 * inputs.time)
    )
    return -gamma(inputs) * bracket


def speed(inputs: Inputs) -> float:
    """``d3V/dS3``: how gamma changes with the underlying."""
    _require_live(inputs, "speed")
    d1, _ = d1_d2(inputs)
    return -gamma(inputs) / inputs.spot * (1.0 + d1 / inputs.std_dev)


def zomma(inputs: Inputs) -> float:
    """``d(gamma)/dsigma``: how curvature responds to volatility."""
    _require_live(inputs, "zomma")
    d1, d2 = d1_d2(inputs)
    return gamma(inputs) * (d1 * d2 - 1.0) / inputs.vol


def forward_delta(inputs: Inputs, option: OptionType) -> float:
    """``dV/dF``, the sensitivity to the forward rather than to the spot.

    The quoting convention on a currency desk, where the forward is the tradable
    quantity. Related to the spot delta by the chain rule through ``F = S e^{bT}``.
    """
    _require_live(inputs, "forward_delta")
    return delta(inputs, option) / math.exp(inputs.b * inputs.time)
