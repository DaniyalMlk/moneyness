"""A closed-form approximation to the American price.

The lattice in :mod:`moneyness.lattice` solves the optimal stopping problem
properly, and pays ``O(n^2)`` node evaluations for the privilege. That is fine
for a price and far too slow to sit inside a calibration loop, where the American
value is wanted thousands of times.

Bjerksund and Stensland's approximation replaces the exact exercise boundary —
a curve that has to be solved for — with a *flat* one, chosen so that the
resulting value is as large as a flat boundary can make it. Exercising the moment
the underlying first crosses a fixed level is a legitimate strategy: not the best
one, but one the holder could actually follow. The value of a legitimate but
suboptimal strategy is a lower bound on the value of the best one, so

    European <= Bjerksund-Stensland <= American

everywhere, with no tuning and no appeal to how accurate the formula happens to
be. That two-sided bracket is what makes this worth having as a check on the
lattice rather than merely as a fast path: the two methods share no code, no
discretisation and no idea, and the inequality between them is exact.

The trigger level ``I`` interpolates between two things that are known. As the
time to expiry goes to zero the level collapses to ``B_0``, the lowest price at
which immediate exercise beats holding a European option to expiry; as it grows
without bound the level rises to ``B_inf``, the perpetual American call's
boundary. The exponential blend between them is the approximation, and everything
else in the formula is the exact value of the flat-boundary strategy given that
level.

This is the 1993 single-boundary form. The 2002 refinement splits the time to
expiry in two and uses a boundary that steps once, which is materially more
accurate but needs a bivariate normal distribution function — a numerical
component in its own right rather than a few extra lines here.

Puts are not implemented twice. Under the McDonald-Schroder transformation an
American put is an American call on the reciprocal market::

    P(S, K, T, r, b, v) = C(K, S, T, r - b, -b, v)

Strike and spot swap, the discount rate becomes the cost of carry's complement,
and the carry changes sign. Routing puts through the call keeps a single exercise
rule in the codebase: a bug in the boundary logic cannot be present for one
option type and absent for the other, which is exactly the asymmetry that makes
such bugs survive testing.
"""

from __future__ import annotations

import math

from .bsm import Inputs, OptionType, intrinsic, price
from .normal import norm_cdf

__all__ = ["bjerksund_stensland", "trigger_price"]


def _phi(
    spot: float,
    time: float,
    gamma: float,
    barrier: float,
    trigger: float,
    rate: float,
    carry: float,
    vol: float,
) -> float:
    """The building block of the flat-boundary valuation.

    This is the discounted expectation of ``S_T^gamma`` on the paths that finish
    above ``barrier`` *and* never touch ``trigger`` on the way. The reflection
    term — the second half of the bracket — is what removes the paths that did
    touch it, which is what makes the whole construction a barrier calculation
    rather than a European one.
    """
    variance = vol * vol
    sd = vol * math.sqrt(time)
    lam = (-rate + gamma * carry + 0.5 * gamma * (gamma - 1.0) * variance) * time
    d = -(math.log(spot / barrier) + (carry + (gamma - 0.5) * variance) * time) / sd
    kappa = 2.0 * carry / variance + (2.0 * gamma - 1.0)
    reflected = d - 2.0 * math.log(trigger / spot) / sd
    # float ** float is typed as Any, so the product is pinned back to float
    # here rather than at every call site.
    powered: float = spot**gamma
    reflection: float = (trigger / spot) ** kappa
    return math.exp(lam) * powered * (norm_cdf(d) - reflection * norm_cdf(reflected))


def trigger_price(inputs: Inputs, option: OptionType) -> float:
    """The flat boundary the approximation exercises at.

    For a call this is the level the underlying must reach before the holder
    gives up the remaining time value; for a put it is the level it must fall to,
    obtained through the same transformation used for the price itself.

    Returns:
        The trigger level, or infinity for a call whose early exercise is
        worthless (``b >= r``) — the boundary is then never reached, which is the
        honest way to say the exercise region is empty. The put's mirror of that
        case returns zero.

    Raises:
        ValueError: on degenerate inputs, where no boundary is defined.
    """
    if inputs.is_degenerate:
        raise ValueError(
            "the exercise boundary is undefined when time, volatility, spot or "
            "strike is zero"
        )
    if option is OptionType.PUT:
        mirrored = Inputs(
            inputs.strike,
            inputs.spot,
            inputs.time,
            inputs.rate - inputs.b,
            inputs.vol,
            carry=-inputs.b,
        )
        level = trigger_price(mirrored, OptionType.CALL)
        # The transformation swaps spot and strike, so a call trigger that sits
        # at infinity corresponds to a put trigger at zero.
        return 0.0 if math.isinf(level) else inputs.spot * inputs.strike / level

    rate, carry, vol, strike = inputs.rate, inputs.b, inputs.vol, inputs.strike
    if carry >= rate:
        return math.inf

    variance = vol * vol
    beta = (0.5 - carry / variance) + math.sqrt(
        (carry / variance - 0.5) ** 2 + 2.0 * rate / variance
    )
    perpetual = beta / (beta - 1.0) * strike
    immediate = max(strike, rate / (rate - carry) * strike)
    decay = -(carry * inputs.time + 2.0 * vol * math.sqrt(inputs.time)) * (
        immediate / (perpetual - immediate)
    )
    return immediate + (perpetual - immediate) * (1.0 - math.exp(decay))


def bjerksund_stensland(inputs: Inputs, option: OptionType) -> float:
    """The Bjerksund-Stensland approximation to the American price.

    Guaranteed to lie between the European price and the true American price, by
    construction rather than by accident: it is the exact value of a flat-boundary
    exercise strategy, which is admissible but not optimal.

    Args:
        inputs: The option and its market.
        option: Call or put.
    """
    if inputs.is_degenerate:
        return max(intrinsic(inputs, option), option.sign * (inputs.spot - inputs.strike), 0.0)

    if option is OptionType.PUT:
        mirrored = Inputs(
            inputs.strike,
            inputs.spot,
            inputs.time,
            inputs.rate - inputs.b,
            inputs.vol,
            carry=-inputs.b,
        )
        return bjerksund_stensland(mirrored, OptionType.CALL)

    rate, carry, vol = inputs.rate, inputs.b, inputs.vol
    spot, strike, time = inputs.spot, inputs.strike, inputs.time

    if carry >= rate:
        # The forward never falls behind the discounted strike fast enough to
        # make waiting costly, so the American call is the European call exactly.
        return price(inputs, option)

    level = trigger_price(inputs, option)
    if spot >= level:
        return spot - strike

    variance = vol * vol
    beta = (0.5 - carry / variance) + math.sqrt(
        (carry / variance - 0.5) ** 2 + 2.0 * rate / variance
    )
    scale: float = level**-beta
    alpha = (level - strike) * scale

    def phi(gamma: float, barrier: float) -> float:
        return _phi(spot, time, gamma, barrier, level, rate, carry, vol)

    powered: float = spot**beta
    return (
        alpha * powered
        - alpha * phi(beta, level)
        + phi(1.0, level)
        - phi(1.0, strike)
        - strike * phi(0.0, level)
        + strike * phi(0.0, strike)
    )
