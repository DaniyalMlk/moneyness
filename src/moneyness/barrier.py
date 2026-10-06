"""Options that depend on where the path went, not just where it ended.

Everything else in this package that has a formula prices off the terminal
distribution. A barrier option does not: it pays only if the spot did, or did
not, reach a level before expiry, so the law of the running extreme enters.
:func:`moneyness.monte_carlo.barrier` already simulates one. This module gives
the closed form, and the vocabulary is that module's
:class:`~moneyness.monte_carlo.Barrier` rather than a second one, so the two
take the same arguments and can be compared directly.

Under Black-Scholes the reflection principle gives the joint law of the
terminal value and the extreme, and the eight standard single-barrier
Europeans come out of four terms — :func:`vanilla_term`, :func:`beyond_term`,
:func:`reflected_term` and :func:`reflected_beyond_term`, named after what they
integrate rather than lettered A to D, because the eight combinations are
otherwise unreadable and a lettered term dropped into the wrong slot leaves no
trace.

**Parity here is a transcription check and not an independent one.** The ``in``
and the ``out`` forms are both written out rather than one being the other
subtracted from the vanilla, and ``in + out = vanilla`` then holds to
**7.1e-15** across all eight, at three strikes either side of the barrier. That
catches a term copied with the wrong sign, which is the likeliest error in a
formula of this shape. It cannot catch the table being the wrong table, because
the table is algebraically consistent with parity by construction.

**There are two independent checks, and they disagree with this module about
nothing.** :func:`moneyness.pde.price_pde` now takes a knock-out and reaches
these numbers by marching a grid backwards with a Dirichlet condition at the
barrier: across seven contracts it converges at **ratios of 4.00** under
doubling of both mesh dimensions, which is the order the scheme supports, so
the agreement is at the mesh's own accuracy rather than to a chosen tolerance.
And :func:`moneyness.monte_carlo.barrier` with its Brownian bridge reaches them
from paths, agreeing within **0.8 standard errors** on all four styles. A
formula, a grid and a simulation, with no derivation in common.

**Three cases are exactly zero or exactly the vanilla.** A knock-out with the
spot on the barrier is worth nothing and the knock-in is worth the vanilla —
bit for bit, because the vanilla is taken from :func:`moneyness.bsm.price`
rather than recomputed here. And an up-and-out call struck at or above its
barrier is worth **exactly 0.0**: paying needs ``S_T > K >= H``, which cannot
happen without crossing ``H``, so the contract does not exist. The down-and-out
put struck at or below its barrier is the mirror.

**An up-and-out call is not monotone in volatility**, which is the finding
worth carrying out of this module. Every option in :mod:`moneyness.bsm` is
increasing in volatility. This one peaks at a volatility of **7.51%** and falls
away on both sides: on a spot of 100, a strike of 100, a barrier of 120 and a
year to run, it is worth **3.435** at the peak and then **3.117, 1.923, 1.107,
0.662, 0.418 and 0.191** at 10%, 15%, 20%, 25%, 30% and 40%. From the peak to
40% it loses 94% of its value for *more* volatility, because the volatility
that pays for the optionality also pays for the knock-out. Vega is **+31.0** at
a 5% volatility and **-11.9** at 20%, so it changes sign inside the range
anybody quotes, and an implied-volatility solve has two roots or none at almost
every price. That is why none is offered.

**Discrete monitoring is worth an order of magnitude more than the shift that
represents it.** A barrier checked at the close rather than continuously is
harder to breach, so a knock-out is worth more.
:func:`monitoring_shift` applies the Broadie-Glasserman-Kou correction, moving
the level away from the spot by ``exp(0.5826 sigma sqrt(dt))`` and pricing the
continuous contract there. Simulating the genuinely discrete contract to say
what that is worth — the bridge turned off, which is a different contract and
not a worse estimate of this one — the up-and-out call above at a 20%
volatility is **+13.3%** over continuous monitoring for a daily close,
**+28.3%** weekly and **+54.7%** monthly, against barrier shifts of 0.737%,
1.63% and 3.42%. The price is levered roughly eighteen times to the level.

**And the correction's own accuracy is measured rather than assumed**, because
it is first order in ``sqrt(dt)`` and someone will reach for the monthly
figure. Against those simulations it is **-0.33%** off daily, **+0.92%**
weekly and **+6.82%** monthly. So it is excellent for a daily close, fine for a
weekly one, and at monthly monitoring it overstates the contract by seven per
cent — still far better than ignoring the effect, which would understate it by
thirty-five, but not a number to quote. Simulate that one.
"""

from __future__ import annotations

import math

from .bsm import Inputs, OptionType, price
from .monte_carlo import Barrier
from .normal import norm_cdf

__all__ = [
    "BARRIER_DIGITS",
    "BarrierError",
    "barrier_price",
    "beyond_term",
    "cannot_pay",
    "is_structurally_worthless",
    "is_touched",
    "monitoring_shift",
    "reflected_beyond_term",
    "reflected_term",
    "vanilla_term",
]

#: Decimal places a barrier is compared at when deciding whether the spot is
#: *on* it. A barrier is a contractual level quoted to a few decimals, so an
#: exact float comparison would make the structural cases below unreachable for
#: any caller who computed the level rather than typing it: ``0.8 * 125.0`` is
#: not ``100.0`` in binary.
BARRIER_DIGITS = 12


class BarrierError(ValueError):
    """A barrier contract that cannot be priced as written."""


def _reflection(style: Barrier) -> float:
    """``+1`` for a down barrier, ``-1`` for an up one: the reflection's direction."""
    return 1.0 if style.is_down else -1.0


# -- the four terms -----------------------------------------------------------
#
# Each is the Black-Scholes integral over a different region of the joint law of
# the terminal spot and the running extreme. They are named after what they
# integrate rather than lettered A to D, because the eight combinations below
# are otherwise unreadable and a lettered term dropped into the wrong slot
# leaves no trace.


def _parts(inputs: Inputs) -> tuple[float, float, float, float]:
    """``(mu, sigma sqrt(T), discounted forward, discounted strike)``."""
    variance = inputs.vol * inputs.vol
    mu = (inputs.b - 0.5 * variance) / variance
    spread = inputs.vol * math.sqrt(inputs.time)
    return (
        mu,
        spread,
        inputs.spot * math.exp((inputs.b - inputs.rate) * inputs.time),
        inputs.strike * math.exp(-inputs.rate * inputs.time),
    )


def vanilla_term(inputs: Inputs, option: OptionType) -> float:
    """The unrestricted Black-Scholes value. Haug's ``A``.

    The same number as :func:`moneyness.bsm.price`, reached through ``mu``
    rather than through ``d1`` directly, which is how the other three terms
    reach it too. The tests assert the two agree, and that is the only
    cross-check available on this file's algebra without leaving it.
    """
    mu, spread, forward, discounted = _parts(inputs)
    phi = option.sign
    first = math.log(inputs.spot / inputs.strike) / spread + (1.0 + mu) * spread
    return phi * forward * norm_cdf(phi * first) - phi * discounted * norm_cdf(
        phi * (first - spread)
    )


def beyond_term(inputs: Inputs, level: float, option: OptionType) -> float:
    """The same integral cut at the barrier rather than at the strike. Haug's ``B``."""
    mu, spread, forward, discounted = _parts(inputs)
    phi = option.sign
    second = math.log(inputs.spot / level) / spread + (1.0 + mu) * spread
    return phi * forward * norm_cdf(phi * second) - phi * discounted * norm_cdf(
        phi * (second - spread)
    )


def reflected_term(
    inputs: Inputs, level: float, option: OptionType, style: Barrier
) -> float:
    """The reflected path's contribution, cut at the strike. Haug's ``C``.

    The reflection is about the barrier, so the image of the spot is
    ``H**2 / S`` and the weight is a power of ``H / S`` set by the drift. This
    is the term that collapses when the spot sits on the barrier, which is what
    makes the structural cases exact rather than close.
    """
    mu, spread, forward, discounted = _parts(inputs)
    phi = option.sign
    eta = _reflection(style)
    ratio = level / inputs.spot
    first = (
        math.log(level * level / (inputs.spot * inputs.strike)) / spread
        + (1.0 + mu) * spread
    )
    return phi * forward * math.pow(ratio, 2.0 * (mu + 1.0)) * norm_cdf(
        eta * first
    ) - phi * discounted * math.pow(ratio, 2.0 * mu) * norm_cdf(eta * (first - spread))


def reflected_beyond_term(
    inputs: Inputs, level: float, option: OptionType, style: Barrier
) -> float:
    """The reflected path's contribution, cut at the barrier. Haug's ``D``."""
    mu, spread, forward, discounted = _parts(inputs)
    phi = option.sign
    eta = _reflection(style)
    ratio = level / inputs.spot
    second = math.log(level / inputs.spot) / spread + (1.0 + mu) * spread
    return phi * forward * math.pow(ratio, 2.0 * (mu + 1.0)) * norm_cdf(
        eta * second
    ) - phi * discounted * math.pow(ratio, 2.0 * mu) * norm_cdf(eta * (second - spread))


# -- the contract -------------------------------------------------------------


def _check_level(level: float) -> float:
    if not math.isfinite(level) or level <= 0.0:
        raise BarrierError(
            f"a barrier at {level!r} is not a spot level the price can reach"
        )
    return level


def is_touched(spot: float, level: float, style: Barrier) -> bool:
    """Whether ``spot`` is already at or past the barrier.

    Compared at :data:`BARRIER_DIGITS` rather than exactly, because a caller
    who computed the level — as a percentage of a reference, say — would
    otherwise miss the structural cases by one bit and get a near-zero price
    where an exact zero is correct.
    """
    here = round(spot, BARRIER_DIGITS)
    there = round(_check_level(level), BARRIER_DIGITS)
    return here <= there if style.is_down else here >= there


def cannot_pay(strike: float, level: float, style: Barrier, option: OptionType) -> bool:
    """Whether finishing in the money requires having breached the barrier.

    An up barrier at or below a call's strike is the case: paying needs
    ``S_T > K >= H``, which cannot happen without crossing ``H``. A down
    barrier at or above a put's strike is the mirror. It makes the knock-*out*
    worthless and the knock-*in* the whole vanilla, which is the same statement
    read from the other side.
    """
    _check_level(level)
    if option is OptionType.CALL:
        return not style.is_down and strike >= level
    return style.is_down and strike <= level


def is_structurally_worthless(
    strike: float, level: float, style: Barrier, option: OptionType
) -> bool:
    """Whether the contract is worth exactly nothing whatever the market does."""
    return style.is_knock_out and cannot_pay(strike, level, style, option)


def monitoring_shift(
    level: float, style: Barrier, volatility: float, monitorings_per_year: float
) -> float:
    """The barrier level at which the continuous formula prices a discrete contract.

    A barrier checked only at fixed times is harder to breach than one watched
    continuously, so a knock-out is worth more and a knock-in less. The
    Broadie-Glasserman-Kou correction pushes the level *away* from the spot by
    ``exp(beta sigma sqrt(dt))``, with ``beta`` 0.5826, and prices the
    continuous contract there.

    The shift is small and what it is worth is not. On the up-and-out call in
    the module docstring at a 20% volatility, a daily check moves the level
    0.737% and the price 12.92%; weekly 1.63% and 29.47%; monthly 3.42% and
    65.24%. The price is levered about seventeen times to the level.

    The correction is first order in ``sqrt(dt)``, so it is good for a daily
    check and rough for a monthly one. That is said here rather than left to be
    assumed, because the monthly figure is the one someone will reach for —
    and :func:`moneyness.monte_carlo.barrier` with ``bridge=False`` prices the
    discrete contract directly, which is what the tests check it against.

    Args:
        level: The contractual barrier.
        style: Which barrier, for the direction to move it in.
        volatility: The volatility to measure the spacing in.
        monitorings_per_year: Checks per year. 252 for a daily close.

    Returns:
        The shifted level.

    Raises:
        BarrierError: If the level, the volatility or the frequency is not one.
    """
    _check_level(level)
    if not math.isfinite(volatility) or volatility < 0.0:
        raise BarrierError(f"a volatility of {volatility!r} is not one")
    if not math.isfinite(monitorings_per_year) or monitorings_per_year <= 0.0:
        raise BarrierError(
            f"{monitorings_per_year!r} monitorings a year is not a frequency; pass "
            "252 for a daily close"
        )
    spacing = math.sqrt(1.0 / monitorings_per_year)
    return level * math.exp(0.5826 * volatility * spacing * -_reflection(style))


def barrier_price(
    inputs: Inputs, option: OptionType, level: float, style: Barrier
) -> float:
    """Price one continuously monitored single-barrier European.

    The argument order follows :func:`moneyness.monte_carlo.barrier`, which
    prices the same contract by simulation, so the two can be called with the
    same arguments and compared — which is what the tests do.

    No rebate. A rebate is a one-touch digital, it needs a convention for *when*
    it pays that differs between the knock-in and knock-out cases, and folding
    one in would break the ``in + out = vanilla`` identity this file is checked
    by. Not supported beats supported wrongly.

    Args:
        inputs: Spot, strike, time, rate, volatility and carry.
        option: Call or put.
        level: The barrier, as a spot level. Positive.
        style: Which barrier, and whether it knocks in or out.

    Returns:
        The present value.

    Raises:
        BarrierError: If the spot, strike or level is not positive.

    The degenerate cases are answered before the formula rather than inside it,
    because each has an exact value the terms would reach only to rounding — or
    not at all, since they divide by ``sigma sqrt(T)``.
    """
    _check_level(level)
    if not math.isfinite(inputs.spot) or inputs.spot <= 0.0:
        raise BarrierError(
            f"a spot of {inputs.spot!r} has no log, so no barrier can be priced "
            "against it"
        )
    if not math.isfinite(inputs.strike) or inputs.strike <= 0.0:
        raise BarrierError(f"a strike of {inputs.strike!r} cannot be reflected")

    vanilla = price(inputs, option)

    # Already touched: an out is dead and an in is a vanilla, exactly. Taking
    # the vanilla from bsm.price rather than from vanilla_term is what makes
    # this bit for bit rather than to the last two bits.
    if is_touched(inputs.spot, level, style):
        return 0.0 if style.is_knock_out else vanilla

    # Cannot pay without breaching, or cannot pay without having breached.
    if cannot_pay(inputs.strike, level, style, option):
        return 0.0 if style.is_knock_out else vanilla

    # No diffusion: the path is the forward, which is monotone in time, so it
    # reaches the barrier if and only if its terminal value is past it. No
    # running maximum is needed and the closed form has no answer here anyway.
    if inputs.is_degenerate:
        forward = inputs.spot * math.exp(inputs.b * inputs.time)
        crosses = forward <= level if style.is_down else forward >= level
        if style.is_knock_out:
            return 0.0 if crosses else vanilla
        return vanilla if crosses else 0.0

    return _combination(inputs, option, level, style)


def _combination(
    inputs: Inputs, option: OptionType, level: float, style: Barrier
) -> float:
    """The four terms, combined by side, knock and where the strike sits.

    Written out for the ``out`` forms as well as the ``in`` forms rather than
    subtracting one from the vanilla. The subtraction would make parity true by
    construction and so remove the only check this file has on its own
    transcription; written out, the eight rows have to be consistent with each
    other, and a sign copied wrongly fails parity by the size of the term that
    was wrong.
    """
    first = vanilla_term(inputs, option)
    beyond = beyond_term(inputs, level, option)
    reflected = reflected_term(inputs, level, option, style)
    reflected_beyond = reflected_beyond_term(inputs, level, option, style)
    above = inputs.strike > level

    if option is OptionType.CALL:
        if style.is_down:
            if not style.is_knock_out:
                return reflected if above else first - beyond + reflected_beyond
            return first - reflected if above else beyond - reflected_beyond
        if not style.is_knock_out:
            return first if above else beyond - reflected + reflected_beyond
        # An up-and-out call struck above the barrier is caught by cannot_pay.
        return first - beyond + reflected - reflected_beyond
    if style.is_down:
        if not style.is_knock_out:
            return beyond - reflected + reflected_beyond if above else first
        # A down-and-out put struck below the barrier is caught by cannot_pay.
        return first - beyond + reflected - reflected_beyond
    if not style.is_knock_out:
        return first - beyond + reflected_beyond if above else reflected
    return beyond - reflected_beyond if above else first - reflected
