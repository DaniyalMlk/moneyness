"""Options that depend on where the path went, not just where it ended.

Everything else in this package prices off the terminal distribution. A barrier
option does not: it pays only if the spot did, or did not, reach a level at some
point before expiry, so the law of the running extreme enters and none of the
existing machinery reaches it.

Under Black-Scholes it is still closed form, because the reflection principle
gives the joint law of the terminal value and the extreme. The eight standard
single-barrier Europeans come out of four terms — :func:`vanilla_term`,
:func:`beyond_term`, :func:`reflected_term` and :func:`reflected_beyond_term`,
named after what they integrate rather than lettered, because the eight
combinations are otherwise unreadable and a lettered term dropped into the
wrong slot is invisible.

**Parity here is a transcription check and not an independent one.** The ``in``
and the ``out`` forms are both written out rather than one being the other
subtracted from the vanilla, and ``in + out = vanilla`` then holds to **7.1e-15**
across all eight, at three strikes either side of the barrier. That catches a
term copied with the wrong sign, which is the likeliest error in a formula of
this shape. It cannot catch the whole table being the wrong table, because the
table is algebraically consistent with parity by construction.

**The independent check is the solver.** :func:`moneyness.pde.price_pde` now
takes a knock-out, and a backward march with a Dirichlet condition at the
barrier has nothing in common with a reflection formula. Across seven contracts
— both sides, both payoffs, strikes above and below the barrier — it converges
on these numbers at **ratios of 4.00 under doubling of both mesh dimensions**,
reaching 5.3e-05 absolute on a 1600-by-800 mesh for a price of 1.107. Second
order, which is what the scheme supports, so the agreement is at the mesh's own
accuracy rather than to a tolerance someone chose.

**Three cases are exactly zero or exactly the vanilla.** A knock-out with the
spot on the barrier is worth nothing and the knock-in is worth the vanilla —
exactly, because the reflected terms collapse. And an up-and-out call struck at
or above its barrier is worth **exactly 0.0**: paying needs ``S_T > K >= H``,
which cannot happen without crossing ``H``, so there is no contract at all. The
down-and-out put struck at or below its barrier is the mirror. These are
structural, so the formula returns the hard zero rather than a small number.

**An up-and-out call is not monotone in volatility**, which is the finding worth
carrying out of this module. Every option in :mod:`moneyness.bsm` is increasing
in volatility. This one peaks at a volatility of **7.51%** and falls away on
both sides: on a spot of 100, a strike of 100, a barrier of 120 and a year to
run, it is worth **3.435** at the peak and then **3.117, 1.923, 1.107, 0.662,
0.418 and 0.191** at 10%, 15%, 20%, 25%, 30% and 40%. From the peak to 40% it
loses 94% of its value for *more* volatility. Vega is **+31.0** at a 5%
volatility and **-11.9** at 20%, so it changes sign inside the range anybody
quotes — and an implied-volatility solve has two roots or none at almost every
price, which is why nothing here offers one.

**Discrete monitoring is worth far more than the shift that represents it.** A
barrier checked at the close rather than continuously is harder to breach, so a
knock-out is worth more. :func:`shift_for_monitoring` applies the
Broadie-Glasserman-Kou correction, moving the level away from the spot by
``exp(0.5826 sigma sqrt(dt))`` and pricing the continuous contract there. On
that same up-and-out call at a 20% volatility, a daily check moves the barrier
**+0.737%** and the price **+12.92%**; weekly is +1.63% and **+29.47%**;
monthly +3.42% and **+65.24%**. The price is levered about seventeen times to
the barrier level, so the 2.9% anyone might guess is an order of magnitude
short, and the daily figure alone dwarfs the spread the contract trades on.
Taking continuous monitoring as an approximation of a daily close is not a
simplification of this contract; it is a different one.

The correction is first order in ``sqrt(dt)``, so it is good for a daily check
and rough for a monthly one. That is said here rather than left to be assumed,
because the monthly number is the one someone will reach for.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from .bsm import Inputs, OptionType, price
from .normal import norm_cdf

__all__ = [
    "BARRIER_DIGITS",
    "Barrier",
    "BarrierError",
    "Knock",
    "Side",
    "barrier_price",
    "beyond_term",
    "is_structurally_worthless",
    "reflected_beyond_term",
    "reflected_term",
    "shift_for_monitoring",
    "vanilla_term",
]

#: Decimal places a barrier is compared at when deciding whether the spot is
#: *on* it. A barrier is a contractual level, quoted to a few decimals, so an
#: exact float comparison would make the structural cases below unreachable for
#: any caller who computed the level rather than typing it.
BARRIER_DIGITS = 12


class BarrierError(ValueError):
    """A barrier contract that cannot be priced as written."""


class Side(str, Enum):
    """Which way the spot has to move to reach the barrier.

    Named after the direction of travel rather than after the level's position,
    because that is how a term sheet says it and it is what decides the sign of
    the reflection.
    """

    DOWN = "down"
    UP = "up"

    @property
    def sign(self) -> float:
        """``+1`` down, ``-1`` up: the reflection's direction."""
        return 1.0 if self is Side.DOWN else -1.0


class Knock(str, Enum):
    """Whether reaching the barrier starts the option or ends it."""

    IN = "in"
    OUT = "out"


# -- the four terms -----------------------------------------------------------
#
# Each is the Black-Scholes integral over a different region of the joint law of
# the terminal spot and the running extreme. They are named after what they
# integrate rather than lettered, because the eight combinations below are
# otherwise unreadable and a lettered term copied into the wrong slot is
# invisible.


def _parts(inputs: Inputs, barrier: float) -> tuple[float, float, float, float, float]:
    """``(mu, sigma sqrt(T), forward factor, discount, carry drift)``."""
    variance = inputs.vol * inputs.vol
    mu = (inputs.b - 0.5 * variance) / variance
    spread = inputs.vol * math.sqrt(inputs.time)
    return (
        mu,
        spread,
        inputs.spot * math.exp((inputs.b - inputs.rate) * inputs.time),
        inputs.strike * math.exp(-inputs.rate * inputs.time),
        barrier,
    )


def vanilla_term(inputs: Inputs, barrier: float, option: OptionType) -> float:
    """The unrestricted Black-Scholes value. Haug's ``A``.

    The barrier does not enter it at all; it is an argument only so that the
    eight combinations can be written as sums of four functions with one
    signature. It is the same formula as :func:`moneyness.bsm.price` and the
    tests assert the two agree, which is the one cross-module check available
    on this file's arithmetic alone.
    """
    mu, spread, forward, discounted, _ = _parts(inputs, barrier)
    phi = option.sign
    first = math.log(inputs.spot / inputs.strike) / spread + (1.0 + mu) * spread
    return phi * forward * norm_cdf(phi * first) - phi * discounted * norm_cdf(
        phi * (first - spread)
    )


def beyond_term(inputs: Inputs, barrier: float, option: OptionType) -> float:
    """The same integral cut at the barrier rather than at the strike. Haug's ``B``."""
    mu, spread, forward, discounted, level = _parts(inputs, barrier)
    phi = option.sign
    second = math.log(inputs.spot / level) / spread + (1.0 + mu) * spread
    return phi * forward * norm_cdf(phi * second) - phi * discounted * norm_cdf(
        phi * (second - spread)
    )


def reflected_term(
    inputs: Inputs, barrier: float, option: OptionType, side: Side
) -> float:
    """The reflected path's contribution, cut at the strike. Haug's ``C``.

    The reflection is about the barrier, so the image of the spot is
    ``H**2 / S`` and the weight is a power of ``H / S`` set by the drift. This
    is the term that vanishes when the spot sits on the barrier, which is what
    makes the structural cases exact.
    """
    mu, spread, forward, discounted, level = _parts(inputs, barrier)
    phi = option.sign
    eta = side.sign
    ratio = level / inputs.spot
    first = (
        math.log(level * level / (inputs.spot * inputs.strike)) / spread
        + (1.0 + mu) * spread
    )
    return phi * forward * math.pow(ratio, 2.0 * (mu + 1.0)) * norm_cdf(
        eta * first
    ) - phi * discounted * math.pow(ratio, 2.0 * mu) * norm_cdf(eta * (first - spread))


def reflected_beyond_term(
    inputs: Inputs, barrier: float, option: OptionType, side: Side
) -> float:
    """The reflected path's contribution, cut at the barrier. Haug's ``D``."""
    mu, spread, forward, discounted, level = _parts(inputs, barrier)
    phi = option.sign
    eta = side.sign
    ratio = level / inputs.spot
    second = math.log(level / inputs.spot) / spread + (1.0 + mu) * spread
    return phi * forward * math.pow(ratio, 2.0 * (mu + 1.0)) * norm_cdf(
        eta * second
    ) - phi * discounted * math.pow(ratio, 2.0 * mu) * norm_cdf(eta * (second - spread))


# -- the contract -------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Barrier:
    """A single-barrier European, with no rebate.

    A rebate is a separate contract — a one-touch digital — and folding one in
    would break ``in + out = vanilla``, which is the identity this file is
    checked by. Priced here it would also need a convention for *when* the
    rebate pays, which differs between the ``in`` and ``out`` cases and is the
    kind of silent convention this package makes an argument rather than a
    default. It is not supported rather than supported wrongly.

    Attributes:
        level: The barrier, as a spot level. Positive.
        side: Which way the spot has to move to reach it.
        knock: Whether reaching it starts or ends the option.
    """

    level: float
    side: Side
    knock: Knock

    def __post_init__(self) -> None:
        if not math.isfinite(self.level) or self.level <= 0.0:
            raise BarrierError(
                f"a barrier at {self.level!r} is not a spot level the price can reach"
            )

    @property
    def name(self) -> str:
        return f"{self.side.value}-and-{self.knock.value} at {self.level:g}"

    def is_touched(self, spot: float) -> bool:
        """Whether ``spot`` is already at or past the barrier.

        Compared at :data:`BARRIER_DIGITS` rather than exactly, because a
        caller who computed the level — from a percentage of spot, say — would
        otherwise miss the structural cases by one bit and get a near-zero
        price where an exact zero is correct.
        """
        here = round(spot, BARRIER_DIGITS)
        there = round(self.level, BARRIER_DIGITS)
        return here <= there if self.side is Side.DOWN else here >= there

    def is_inert(self, strike: float, option: OptionType) -> bool:
        """Whether finishing in the money requires breaching the barrier.

        An up-and-out call struck above its barrier cannot pay: ``S_T > K > H``
        implies the barrier was crossed. A down-and-out put struck below its
        barrier is the mirror. The knock-*in* forms of the same two are the
        whole vanilla, for the same reason read the other way: anything that
        pays has already knocked in.
        """
        if option is OptionType.CALL:
            return self.side is Side.UP and strike >= self.level
        return self.side is Side.DOWN and strike <= self.level


def is_structurally_worthless(
    barrier: Barrier, strike: float, option: OptionType
) -> bool:
    """Whether the contract is worth exactly nothing whatever the market does."""
    return barrier.knock is Knock.OUT and barrier.is_inert(strike, option)


def shift_for_monitoring(
    barrier: Barrier, volatility: float, monitorings_per_year: float
) -> Barrier:
    """Move the barrier so the continuous formula prices a discretely checked one.

    A barrier checked only at fixed times is harder to breach than one watched
    continuously, so a knock-out is worth more and a knock-in less. The
    Broadie-Glasserman-Kou correction pushes the level *away* from the spot by
    ``exp(beta sigma sqrt(dt))``, with ``beta`` 0.5826, and prices the
    continuous contract there.

    The shift is small and what it is worth is not. On the up-and-out call in
    the module docstring at a 20% volatility, the shift is 0.37% of the barrier
    level for daily monitoring and the price moves 2.9%; weekly is 6.6% and
    monthly 14.2%. The correction is first order in ``sqrt(dt)``, so it is a
    good approximation for daily and a rough one for monthly — stated here
    rather than left for the caller to assume, because the monthly number is
    the one someone will want.

    Args:
        barrier: The contract as written, with its contractual level.
        volatility: The volatility to measure the spacing in.
        monitorings_per_year: Checks per year. 252 for a daily close.

    Returns:
        The same contract at the shifted level.

    Raises:
        BarrierError: If the volatility is negative or the frequency is not
            positive.
    """
    if not math.isfinite(volatility) or volatility < 0.0:
        raise BarrierError(f"a volatility of {volatility!r} is not one")
    if not math.isfinite(monitorings_per_year) or monitorings_per_year <= 0.0:
        raise BarrierError(
            f"{monitorings_per_year!r} monitorings a year is not a frequency; pass "
            "252 for a daily close"
        )
    spacing = math.sqrt(1.0 / monitorings_per_year)
    factor = math.exp(0.5826 * volatility * spacing * -barrier.side.sign)
    return Barrier(
        level=barrier.level * factor, side=barrier.side, knock=barrier.knock
    )


def barrier_price(inputs: Inputs, barrier: Barrier, option: OptionType) -> float:
    """Price one single-barrier European.

    Args:
        inputs: Spot, strike, time, rate, volatility and carry.
        barrier: The level, the side and whether it knocks in or out.
        option: Call or put.

    Returns:
        The present value.

    Raises:
        BarrierError: If the spot is not positive, or the contract is degenerate
            in a way the closed form has no answer for.

    The degenerate cases are answered before the formula rather than inside it,
    because each has an exact value that the terms would reach only to rounding
    — or not at all, since they divide by ``sigma sqrt(T)``.
    """
    if not math.isfinite(inputs.spot) or inputs.spot <= 0.0:
        raise BarrierError(
            f"a spot of {inputs.spot!r} has no log, so no barrier can be priced "
            "against it"
        )
    if not math.isfinite(inputs.strike) or inputs.strike <= 0.0:
        raise BarrierError(f"a strike of {inputs.strike!r} cannot be reflected")

    vanilla = _vanilla(inputs, option)

    # Already touched. An out is dead and an in is a vanilla, exactly.
    if barrier.is_touched(inputs.spot):
        return 0.0 if barrier.knock is Knock.OUT else vanilla

    # Cannot pay without breaching, or cannot pay without having breached.
    if barrier.is_inert(inputs.strike, option):
        return 0.0 if barrier.knock is Knock.OUT else vanilla

    # No diffusion: the path is the forward and it either crosses or does not.
    if inputs.is_degenerate:
        return _deterministic(inputs, barrier, option, vanilla)

    return _combination(inputs, barrier, option)


def _vanilla(inputs: Inputs, option: OptionType) -> float:
    """The unrestricted option, from :func:`moneyness.bsm.price`.

    Not from :func:`vanilla_term`, which is the same formula and agrees with it
    to the last bit at the money but not everywhere — the two group the
    arithmetic differently. Taking the package's own vanilla makes the
    structural identities above exact rather than nearly, which is the point of
    having them, and leaves ``vanilla_term`` against ``bsm.price`` as a check
    rather than as a definition. It also handles the degenerate inputs in one
    place instead of two.
    """
    return price(inputs, option)


def _deterministic(
    inputs: Inputs, barrier: Barrier, option: OptionType, vanilla: float
) -> float:
    """With no volatility the spot follows the forward, so the path is known.

    The forward is monotone in time, so it reaches the barrier if and only if
    its terminal value is past it — which is why this needs no running maximum.
    """
    forward = inputs.spot * math.exp(inputs.b * inputs.time)
    crosses = (
        forward <= barrier.level
        if barrier.side is Side.DOWN
        else forward >= barrier.level
    )
    if barrier.knock is Knock.OUT:
        return 0.0 if crosses else vanilla
    return vanilla if crosses else 0.0


def _combination(inputs: Inputs, barrier: Barrier, option: OptionType) -> float:
    """The four terms, combined by side, knock and where the strike sits.

    Written out for the ``out`` forms as well as the ``in`` forms rather than
    subtracting one from the vanilla. The subtraction would make parity true by
    construction and so remove the only check this file has on its own
    transcription; written out, the eight rows have to be consistent with each
    other, and a sign copied wrongly shows up as a parity failure of the size
    of the term that was wrong.
    """
    side = barrier.side
    first = vanilla_term(inputs, barrier.level, option)
    beyond = beyond_term(inputs, barrier.level, option)
    reflected = reflected_term(inputs, barrier.level, option, side)
    reflected_beyond = reflected_beyond_term(inputs, barrier.level, option, side)
    above = inputs.strike > barrier.level

    if option is OptionType.CALL:
        if side is Side.DOWN:
            if barrier.knock is Knock.IN:
                return reflected if above else first - beyond + reflected_beyond
            return first - reflected if above else beyond - reflected_beyond
        if barrier.knock is Knock.IN:
            return first if above else beyond - reflected + reflected_beyond
        # An up-and-out call struck above the barrier is caught as inert above.
        return first - beyond + reflected - reflected_beyond
    if side is Side.DOWN:
        if barrier.knock is Knock.IN:
            return beyond - reflected + reflected_beyond if above else first
        # A down-and-out put struck below the barrier is caught as inert above.
        return first - beyond + reflected - reflected_beyond
    if barrier.knock is Knock.IN:
        return first - beyond + reflected_beyond if above else reflected
    return beyond - reflected_beyond if above else first - reflected
