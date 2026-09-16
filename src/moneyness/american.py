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

That is the 1993 single-boundary form, :func:`bjerksund_stensland`. The 2002
refinement, :func:`bjerksund_stensland_2002`, lets the boundary *step once*
partway through the option's life instead of holding one level throughout. A
step is still a strategy the holder could follow, so the bracket above survives
intact — and since a flat boundary is the special case where the two levels
coincide, the two-step value can never be the worse of the two.

The cost is a bivariate normal distribution function. Pricing the joint event of
not having crossed the first level before the step *and* finishing in the money
is inherently two-dimensional, and the correlation between the two dates is
``sqrt(t1 / T)``, which the switch date puts close to one. That is why
:mod:`moneyness.bivariate` is built the way it is rather than by the first method
that works at moderate correlation.

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

from .bivariate import norm_cdf2
from .bsm import Inputs, OptionType, intrinsic, price
from .normal import norm_cdf

__all__ = ["bjerksund_stensland", "bjerksund_stensland_2002", "trigger_price"]

# The fraction of the option's life at which the boundary is allowed to step.
# Bjerksund and Stensland put it at (sqrt(5) - 1) / 2 of the way through, the
# reciprocal of the golden ratio. It is not fitted to anything: the boundary
# rises fastest early on, so the switch belongs before the midpoint, and this
# is the self-similar choice -- the first segment stands in the same ratio to
# the whole as the second does to the first.
_SWITCH = 0.5 * (math.sqrt(5.0) - 1.0)


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


def _psi(
    spot: float,
    time: float,
    gamma: float,
    barrier: float,
    upper: float,
    lower: float,
    switch: float,
    rate: float,
    carry: float,
    vol: float,
) -> float:
    """The two-date analogue of :func:`_phi`.

    Where ``_phi`` discounts ``S_T^gamma`` over the paths that finish above a
    barrier having never touched one trigger, this one handles two triggers in
    succession: the path must stay below ``lower`` up to the switch date, then
    below ``upper`` to expiry, and finish above ``barrier``.

    Two dates means a joint distribution, hence the bivariate normal. The
    correlation is ``sqrt(t1 / T)`` — the correlation of a Brownian motion with
    itself at two times — and the four terms are the inclusion-exclusion over
    the reflections that remove the paths which did touch a trigger, exactly as
    the single reflection term does in ``_phi``.
    """
    variance = vol * vol
    drift = carry + (gamma - 0.5) * variance
    near = vol * math.sqrt(switch)
    far = vol * math.sqrt(time)

    e1 = (math.log(spot / lower) + drift * switch) / near
    e2 = (math.log(upper * upper / (spot * lower)) + drift * switch) / near
    e3 = (math.log(spot / lower) - drift * switch) / near
    e4 = (math.log(upper * upper / (spot * lower)) - drift * switch) / near

    f1 = (math.log(spot / barrier) + drift * time) / far
    f2 = (math.log(upper * upper / (spot * barrier)) + drift * time) / far
    f3 = (math.log(lower * lower / (spot * barrier)) + drift * time) / far
    f4 = (math.log(spot * lower * lower / (barrier * upper * upper)) + drift * time) / far

    rho = math.sqrt(switch / time)
    lam = -rate + gamma * carry + 0.5 * gamma * (gamma - 1.0) * variance
    kappa = 2.0 * carry / variance + (2.0 * gamma - 1.0)

    powered: float = spot**gamma
    upper_ratio: float = (upper / spot) ** kappa
    lower_ratio: float = (lower / spot) ** kappa
    both_ratio: float = (lower / upper) ** kappa

    joint = (
        norm_cdf2(-e1, -f1, rho)
        - upper_ratio * norm_cdf2(-e2, -f2, rho)
        - lower_ratio * norm_cdf2(-e3, -f3, -rho)
        + both_ratio * norm_cdf2(-e4, -f4, -rho)
    )
    return math.exp(lam * time) * powered * joint


def _two_step_triggers(inputs: Inputs) -> tuple[float, float, float]:
    """The two exercise levels and the date the boundary steps between them.

    Both levels come from the same blend between the immediate-exercise floor
    and the perpetual boundary that the 1993 form uses, evaluated at the two
    dates. The decay constant differs from the 1993 one: it is scaled by
    ``X^2 / ((B_inf - B_0) B_0)`` rather than ``B_0 / (B_inf - B_0)``, which is
    what makes the earlier level sit below the later one instead of the two
    being the same curve sampled twice.
    """
    rate, carry, vol, strike, time = (
        inputs.rate,
        inputs.b,
        inputs.vol,
        inputs.strike,
        inputs.time,
    )
    variance = vol * vol
    beta = (0.5 - carry / variance) + math.sqrt(
        (carry / variance - 0.5) ** 2 + 2.0 * rate / variance
    )
    perpetual = beta / (beta - 1.0) * strike
    immediate = max(strike, rate / (rate - carry) * strike)
    switch = _SWITCH * time

    scale = strike * strike / ((perpetual - immediate) * immediate)

    def level(at: float) -> float:
        decay = -(carry * at + 2.0 * vol * math.sqrt(at)) * scale
        return immediate + (perpetual - immediate) * (1.0 - math.exp(decay))

    return level(switch), level(time), switch


def _two_step_value(inputs: Inputs, option: OptionType) -> float:
    """The 2002 formula itself, without the safeguard in the caller.

    Kept separate so that the safeguard is visible as a decision rather than
    buried in the algebra, and so the tests can measure how often it binds.
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
        return _two_step_value(mirrored, OptionType.CALL)

    rate, carry, vol = inputs.rate, inputs.b, inputs.vol
    spot, strike, time = inputs.spot, inputs.strike, inputs.time

    if carry >= rate:
        return price(inputs, option)

    lower, upper, switch = _two_step_triggers(inputs)
    if spot >= upper:
        return spot - strike

    variance = vol * vol
    beta = (0.5 - carry / variance) + math.sqrt(
        (carry / variance - 0.5) ** 2 + 2.0 * rate / variance
    )
    lower_scale: float = lower**-beta
    upper_scale: float = upper**-beta
    alpha_lower = (lower - strike) * lower_scale
    alpha_upper = (upper - strike) * upper_scale

    def phi(gamma: float, barrier: float, trigger: float) -> float:
        return _phi(spot, switch, gamma, barrier, trigger, rate, carry, vol)

    def psi(gamma: float, barrier: float) -> float:
        return _psi(spot, time, gamma, barrier, upper, lower, switch, rate, carry, vol)

    powered: float = spot**beta
    return (
        alpha_upper * powered
        - alpha_upper * phi(beta, upper, upper)
        + phi(1.0, upper, upper)
        - phi(1.0, lower, upper)
        - strike * phi(0.0, upper, upper)
        + strike * phi(0.0, lower, upper)
        + alpha_lower * phi(beta, lower, upper)
        - alpha_lower * psi(beta, lower)
        + psi(1.0, lower)
        - psi(1.0, strike)
        - strike * psi(0.0, lower)
        + strike * psi(0.0, strike)
    )


def bjerksund_stensland_2002(inputs: Inputs, option: OptionType) -> float:
    """The 2002 two-step approximation, as the sharpest admissible lower bound.

    The exercise boundary is held at one level until ``(sqrt(5) - 1) / 2`` of
    the way through the option's life and at a higher one thereafter, rather
    than flat throughout. Across a grid of two hundred and forty random markets
    this is closer to the lattice than the 1993 form on all but nine of them,
    and improves the mean absolute error by about a factor of 1.4.

    Those nine are the reason this function returns a maximum rather than the
    formula alone. It is tempting to argue that a two-step boundary cannot be
    worse than a flat one, since a flat boundary is the special case of the two
    levels coinciding — but that argument is about the *optimal* two-step
    boundary, and the levels here come from a closed-form heuristic that is not
    optimised for either. When the spot sits between the two levels, or the
    volatility is very high over a long maturity, the heuristic can place them
    badly enough that the result falls below the 1993 value and, in one market
    out of the two hundred and forty, below the European price — which no
    American value may do, since holding to expiry is always available.

    Each of the three candidates is the value of a strategy the holder could
    actually follow, so each is a lower bound on the American price, and the
    largest of them is both the sharpest bound available and still a bound. The
    bracket the module is built on therefore holds unconditionally:

        European <= max(European, 1993, 2002) <= American

    Taking the maximum is not a patch over a numerical problem — it is the
    correct way to combine several admissible strategies when none dominates
    the others everywhere. It binds on about four per cent of that grid, and
    the tests assert that it binds rather than sitting there unused.

    Args:
        inputs: The option and its market.
        option: Call or put.

    Returns:
        The approximate American price.
    """
    if inputs.is_degenerate:
        return max(intrinsic(inputs, option), option.sign * (inputs.spot - inputs.strike), 0.0)
    return max(
        price(inputs, option),
        bjerksund_stensland(inputs, option),
        _two_step_value(inputs, option),
    )
