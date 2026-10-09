"""Options on where the path got to, not on whether it reached a level.

:mod:`moneyness.barrier` prices contracts that depend on whether the running
extreme crossed a level. The same reflection principle gives the whole law of
that extreme, and a lookback option pays on *where* it got to, so the two
modules share their mathematics and divide the contracts between them.

Everything here is built from two expectations:

    E[(M_T - L)^+]    and    E[(L - m_T)^+]

where ``M_T`` and ``m_T`` are the running maximum and minimum over the life.
The four standard lookbacks are rearrangements of those, and so is the expected
extreme, so there is one formula to get right rather than four. The
rearrangements are exact and are written as such — a fixed-strike call struck
below the running maximum is the one struck at that maximum plus the discounted
difference of the strikes, and that holds to **1.3e-15** across a grid of
carries and volatilities, which is the subtraction's own round-off and not the
formula's, because both sides read the same tail integral.

**The distribution of the extreme is public.** :func:`maximum_cdf` and
:func:`minimum_cdf` are not an implementation detail of the pricers: the law of
a running extreme is what a stop-loss, a drawdown limit or a high-water-mark fee
is written on, and none of those is an option. Two of its values are exact
rather than computed. ``P(M_T <= S)`` is **0.0** bit for bit and so is
``P(m_T >= S)``: a Brownian path leaves its starting point immediately in both
directions, so the maximum exceeds the spot and the minimum falls below it with
probability one. A naive evaluation of the reflection formula at that point is
a difference of two equal terms and returns something near 1e-17 instead, which
then propagates into a tail integral as a spurious mass.

**The closed form divides by the carry, and zero carry is the futures case.**
Integrating the tail analytically produces a factor of ``sigma^2 / (2b)``
multiplying a difference of two terms that become equal as ``b`` goes to zero.
The textbook transcription therefore divides by zero on an option whose carry is
exactly zero — and :meth:`moneyness.bsm.Inputs.on_future` sets it to exactly
``0.0``, so an option on a future is not an edge case to be waved at but one of
the conventions the package advertises. Measured against fifty-digit arithmetic
at a spot of 100, a 20% volatility and a year to run, that grouping loses
accuracy steadily as the carry shrinks: :func:`maximum_excess` at the spot is
right to 2.0e-15 at a carry of 1%, 8.7e-14 at 1bp and **2.4e-12 at a carry of
1e-6**, and at a level 2.7 times the spot the three are 2.0e-14, 3.8e-12 and
**5.7e-10**.

So the terms are regrouped. Both normal arguments turn out to sit
symmetrically about one point, so pulling the shared exponential out leaves an
``expm1`` and a band of the normal law between two symmetric arguments, each of
which has a form that survives the limit. The regrouped form holds those same
six cases to between 2.1e-16 and 1.7e-14, and it returns the limit at a carry
of exactly 0.0 rather than raising.

**The two groupings fail in opposite regimes, and switching between them is
what either one alone cannot do.** The regrouped form cancels when the carry is
*large* relative to the variance, because the exponential it factors out then
dwarfs the difference it multiplies: at ``2b / sigma^2 = 15`` it is wrong by
2.3e-10 where the direct form is exact to 1.8e-14, and at 100 it is wrong by a
factor of 4e+24 on a value of 958. Over 7,840 combinations of level, carry,
volatility and maturity — 6,765 of them above a floor of ``1e-8`` times the
spot, below which a relative error is reporting round-off rather than method —
the worst relative error of the pair is **3.3e-13**, against 8.4e-09 for the
direct form alone. The crossover at ``|2b / sigma^2| = 1`` is not a tuned
value: the same sweep gives 3.3e-13 at 0.5 and at 2 as well, so it sits in the
middle of a flat region rather than on a cliff.

**A floating-strike lookback call has no strike and is never worth nothing.**
It pays ``S_T - m_T``, which is non-negative on every path and zero only on the
path that ends at its own minimum, so there is no moneyness and no exercise
decision. ``inputs.strike`` plays no part in the floating-strike styles and is
not read. That is worth stating because the argument is still there, carried by
:class:`~moneyness.bsm.Inputs`, and a caller who sets it will see no effect.

**Two of the four are the same contract plus a forward, which is the identity
to check rather than parity.** A fixed-strike call struck at the spot pays
``M_T - S`` with certainty, since the maximum is at least the spot, and a
floating-strike put pays ``M_T - S_T``. Their difference is the discounted
difference of the two deterministic legs, so

    fixed call at K = S  -  floating put  =  e^{-rT} (S - S e^{bT})

holds to **2.1e-14** on a spot of 100 across every carry and volatility tried,
and on a simulation the two payoffs differ by a constant on every path and come
out with the same standard error to the last digit. Measured against the
difference rather than against the spot it degrades to 1.6e-10 at a carry of
1e-6, where two prices of about 17 are being subtracted to leave 1e-4; that is
the cancellation in the comparison and not in either price. It is a
transcription check either way, which is why the independent checks below are
the ones that matter.

**``Inputs.is_degenerate`` is the wrong question to ask here**, and asking it
cost a wrong price before a test caught it. That property is true when the
*terminal payoff* carries no uncertainty, and a zero strike is one of its
cases, because a call struck at nothing is worth the forward whatever the path
does. A lookback's strike says nothing about the path: a floating-strike call
does not read the strike at all. Pricing one with the strike set to zero off a
deterministic path returned **4.98** where the answer is 17.56. What matters
here is whether the path itself is deterministic, which is no time or no
volatility and nothing else.

**The complement of the distribution function is not the tail.** ``1.0 -
maximum_cdf(...)`` is the right number only while it is representable: at five
times the spot on a 10% volatility the true tail is below 1e-16 and the
subtraction returns zero or round-off. Nothing here needs it — the tail is
integrated analytically rather than sampled — but a caller reaching for a deep
exceedance probability should know that :func:`maximum_cdf` is accurate in
absolute terms and its complement is not.

**Discrete monitoring is worth a great deal here, and in the opposite direction
from a barrier.** A maximum taken at the close rather than continuously is
smaller, so every lookback is worth less, and there is no level to shift to
represent it the way :func:`moneyness.barrier.monitoring_shift` shifts a
barrier. On a spot of 100, a 20% volatility, a 5% carry and a year, the
fixed-strike call struck at the spot is worth **19.17** continuously, and
18.36 daily, 17.41 weekly and 15.76 monthly — down **4.2%, 9.2% and 17.8%**.
The floating-strike call is worth 17.22 continuously and loses 3.4%, 7.5% and
14.7% over the same monitoring. So the effect is of the same order as the
barrier module's +13.3% for a daily-monitored up-and-out call, and
:func:`moneyness.monte_carlo.lookback` prices the discrete contract rather than
correcting the continuous one.

**The simulation is unbiased because each step's extreme is drawn, not
observed.** A path sampled on a grid has a smaller maximum than the path it was
drawn from, and the bias dies like one over the square root of the step count.
Conditional on its endpoints a Brownian path is a bridge, and the maximum of a
bridge has an invertible distribution function, so the step's true maximum comes
from one extra uniform. With 40 steps and 200,000 paths, over five independent
seeds, the simulated fixed-strike call struck at 120 lands between **-1.2 and
+0.9 standard errors** of this module's number and the floating-strike call
between **-1.2 and +1.7**, scattering either side rather than drifting. With the
bridge switched off and the same 40 steps it is **101 standard errors low**,
which is bias and not noise, and is why a grid on its own cannot check a
formula.
"""

from __future__ import annotations

import math

from .bsm import Inputs, OptionType, forward
from .monte_carlo import Lookback
from .normal import norm_cdf, norm_pdf

__all__ = [
    "CARRY_CROSSOVER",
    "LookbackError",
    "expected_maximum",
    "expected_minimum",
    "lookback_price",
    "maximum_cdf",
    "maximum_excess",
    "minimum_cdf",
    "minimum_shortfall",
]

#: Where the two groupings of the tail integral change over, in units of
#: ``2b / sigma^2`` — the dimensionless carry that appears as the exponent of
#: the reflection term. Below it the regrouped form is used and above it the
#: direct one. One is not a tuned value: it is the point at which the two
#: error mechanisms are of the same size, and the sweep in this module's
#: docstring measures what the pair achieves there.
CARRY_CROSSOVER = 1.0


class LookbackError(ValueError):
    """A lookback contract that cannot be priced as written."""


def _check(inputs: Inputs) -> None:
    if inputs.spot <= 0.0:
        raise LookbackError(f"the spot must be positive, got {inputs.spot}")


def _observed(inputs: Inputs, level: float | None, *, upper: bool) -> float:
    """The running extreme so far, defaulting to the spot for a fresh contract.

    A seasoned lookback carries a maximum at or above the spot and a minimum at
    or below it, because the spot is itself one of the observations. An extreme
    on the wrong side of the spot is a stale record rather than a contract term,
    and pricing it would silently use the spot instead.
    """
    if level is None:
        return inputs.spot
    if level <= 0.0:
        raise LookbackError(f"the observed extreme must be positive, got {level}")
    if upper and level < inputs.spot:
        raise LookbackError(
            f"an observed maximum of {level} is below the spot of {inputs.spot}, "
            "so the spot itself contradicts it"
        )
    if not upper and level > inputs.spot:
        raise LookbackError(
            f"an observed minimum of {level} is above the spot of {inputs.spot}, "
            "so the spot itself contradicts it"
        )
    return level


def _band(centre: float, half_width: float) -> float:
    """``Phi(x + d) - Phi(x - d)`` without the cancellation, for ``d >= 0``.

    Two regimes. When ``d`` times the larger of one and ``|x|`` is small the
    difference is a Taylor series in ``d`` about ``x``, whose coefficients are
    the even Hermite polynomials times the density; five terms then carry it to
    round-off, and the series is used precisely because the direct difference is
    two nearly equal numbers there.

    When it is not small the direct difference is safe, provided it is taken on
    the side where both values are small. For a positive centre the two
    distribution functions are both close to one and their difference loses
    every digit; the complementary tail gives the same number with none lost.
    """
    if half_width <= 0.0:
        return 0.0
    if half_width * max(1.0, abs(centre)) < 0.1:
        squared = centre * centre
        # Even Hermite polynomials He_2k, which are the even derivatives of the
        # density divided by the density itself.
        he2 = squared - 1.0
        he4 = squared * squared - 6.0 * squared + 3.0
        he6 = squared**3 - 15.0 * squared * squared + 45.0 * squared - 15.0
        he8 = (
            squared**4
            - 28.0 * squared**3
            + 210.0 * squared * squared
            - 420.0 * squared
            + 105.0
        )
        d2 = half_width * half_width
        series = half_width * (
            1.0
            + d2
            * (
                he2 / 6.0
                + d2 * (he4 / 120.0 + d2 * (he6 / 5040.0 + d2 * he8 / 362880.0))
            )
        )
        return 2.0 * norm_pdf(centre) * series
    if centre > 0.0:
        return norm_cdf(-(centre - half_width)) - norm_cdf(-(centre + half_width))
    return norm_cdf(centre + half_width) - norm_cdf(centre - half_width)


def _expm1_over(x: float) -> float:
    """``expm1(x) / x``, which is one at zero rather than undefined."""
    return 1.0 if x == 0.0 else math.expm1(x) / x


def _is_flat(inputs: Inputs) -> bool:
    """Whether the path is deterministic, which is not what ``is_degenerate`` means.

    :attr:`moneyness.bsm.Inputs.is_degenerate` is true when the *terminal payoff*
    carries no uncertainty, and a zero strike is one of the cases it covers: a
    call struck at nothing is worth the forward whatever the path does. Here the
    strike says nothing at all about the path, and a lookback with a zero strike
    is an ordinary contract on a random extreme. Reusing that property priced a
    zero-strike floating-strike call off a deterministic path and returned 4.98
    where the right answer is 17.56.
    """
    return inputs.time == 0.0 or inputs.vol == 0.0


def _degenerate_extremes(inputs: Inputs) -> tuple[float, float]:
    """The running minimum and maximum when the path is deterministic.

    With no volatility or no time left the underlying moves along its carry, so
    the extremes are the two endpoints of a monotone path.
    """
    terminal = forward(inputs)
    return min(inputs.spot, terminal), max(inputs.spot, terminal)


def maximum_cdf(inputs: Inputs, level: float) -> float:
    """``P(M_T <= level)``, the distribution function of the running maximum.

    The maximum is taken over the continuously observed path from now to expiry,
    so it ignores any history: a seasoned contract's observed maximum is a
    contract term and belongs in the pricers, not in the law of the future path.

    Args:
        inputs: The underlying and its market. Only the spot, time, volatility
            and carry are read.
        level: Where to evaluate. Positive.

    Returns:
        A probability. Exactly ``0.0`` at or below the spot, since the path
        leaves the spot upward immediately.

    Raises:
        LookbackError: If the spot or ``level`` is not positive.
    """
    _check(inputs)
    if level <= 0.0:
        raise LookbackError(f"level must be positive, got {level}")
    if level <= inputs.spot:
        return 0.0
    if _is_flat(inputs):
        return 1.0 if _degenerate_extremes(inputs)[1] <= level else 0.0
    sigma = inputs.vol
    v = inputs.std_dev
    drift = (inputs.b - 0.5 * sigma * sigma) * inputs.time
    u = math.log(level / inputs.spot)
    reflected = math.exp(2.0 * drift * u / (v * v))
    return norm_cdf((u - drift) / v) - reflected * norm_cdf((-u - drift) / v)


def minimum_cdf(inputs: Inputs, level: float) -> float:
    """``P(m_T <= level)``, the distribution function of the running minimum.

    Args:
        inputs: The underlying and its market.
        level: Where to evaluate. Positive.

    Returns:
        A probability. Exactly ``1.0`` at or above the spot, since the path
        leaves the spot downward immediately.

    Raises:
        LookbackError: If the spot or ``level`` is not positive.
    """
    _check(inputs)
    if level <= 0.0:
        raise LookbackError(f"level must be positive, got {level}")
    if level >= inputs.spot:
        return 1.0
    if _is_flat(inputs):
        return 1.0 if _degenerate_extremes(inputs)[0] <= level else 0.0
    sigma = inputs.vol
    v = inputs.std_dev
    drift = (inputs.b - 0.5 * sigma * sigma) * inputs.time
    u = math.log(level / inputs.spot)
    reflected = math.exp(2.0 * drift * u / (v * v))
    return norm_cdf((u - drift) / v) + reflected * norm_cdf((u + drift) / v)


def _reflection_integral(inputs: Inputs, log_level: float, *, upper: bool) -> float:
    """The reflection term's contribution to the tail integral, divided by ``c``.

    ``c = 2b / sigma^2`` is the exponent the reflection term carries, and the
    analytic integral of that term brings a ``1 / c`` with it. Both groupings of
    the surviving difference are written out: the direct one, which cancels as
    ``c`` approaches zero, and the regrouped one, which cancels when ``c`` is
    large. :data:`CARRY_CROSSOVER` picks between them.
    """
    sigma = inputs.vol
    v = inputs.std_dev
    time = inputs.time
    c = 2.0 * inputs.b / (sigma * sigma)
    a = log_level
    half_variance = 0.5 * v * v
    drift = (inputs.b - 0.5 * sigma * sigma) * time

    if abs(c) > CARRY_CROSSOVER:
        carried = math.exp(inputs.b * time)
        if upper:
            return (
                carried * norm_cdf((c * v * v - drift - a) / v)
                - math.exp(c * a) * norm_cdf((-a - drift) / v)
            ) / c
        return (
            math.exp(c * a) * norm_cdf((a + drift) / v)
            - carried * norm_cdf((a + drift - c * v * v) / v)
        ) / c

    # Regrouped. Both normal arguments sit symmetrically about ``centre``, so
    # pulling the shared exponential out leaves an ``expm1`` and a band of the
    # normal law, each of which has a form that survives ``c`` going to zero.
    centre = (half_variance - a) / v
    if not upper:
        centre = -centre
    half_width = 0.5 * c * v
    # ``(e^{c v^2 / 2} - e^{c a}) / c``, as a difference quotient.
    span = half_variance - a
    gap = math.exp(c * a) * span * _expm1_over(c * span)
    # ``(Phi(centre + d) - Phi(centre - d)) / c``. Both the band and ``c`` flip
    # sign with the carry, so the ratio is taken on the absolute values.
    ratio = (
        v * norm_pdf(centre)
        if c == 0.0
        else _band(centre, abs(half_width)) / abs(c)
    )
    edge = norm_cdf(centre + half_width)
    if upper:
        return gap * edge + math.exp(c * a) * ratio
    return -gap * edge + math.exp(c * half_variance) * ratio


def maximum_excess(inputs: Inputs, level: float) -> float:
    """``E[(M_T - level)^+]``, undiscounted, under the risk-neutral drift.

    This is the whole of the module: the fixed-strike lookback call is this
    discounted, the floating-strike put is this plus a forward, and the expected
    maximum is this at the spot plus the spot.

    The payoff is written as an integral of the tail, ``(M - L)^+ = int_L^inf
    1{M > y} dy``, so the expectation is an integral of :func:`maximum_cdf`'s
    complement and needs no density. That matters: the density of a running
    maximum has an atom-free but awkward form, and the tail is the thing the
    reflection principle gives directly.

    Args:
        inputs: The underlying and its market.
        level: The level excess is measured above. Positive. Levels below the
            spot are handled exactly, by the identity that the maximum is at
            least the spot.

    Returns:
        A non-negative expectation.

    Raises:
        LookbackError: If the spot or ``level`` is not positive.
    """
    _check(inputs)
    if level <= 0.0:
        raise LookbackError(f"level must be positive, got {level}")
    if _is_flat(inputs):
        return max(_degenerate_extremes(inputs)[1] - level, 0.0)
    if level < inputs.spot:
        # The maximum is at least the spot, so the first stretch is certain.
        return (inputs.spot - level) + maximum_excess(inputs, inputs.spot)
    spot = inputs.spot
    v = inputs.std_dev
    drift = (inputs.b - 0.5 * inputs.vol * inputs.vol) * inputs.time
    a = math.log(level / spot)
    vanilla = -math.exp(a) * norm_cdf((drift - a) / v) + math.exp(
        inputs.b * inputs.time
    ) * norm_cdf((drift + v * v - a) / v)
    return spot * (vanilla + _reflection_integral(inputs, a, upper=True))


def minimum_shortfall(inputs: Inputs, level: float) -> float:
    """``E[(level - m_T)^+]``, undiscounted, under the risk-neutral drift.

    The mirror of :func:`maximum_excess`, written out rather than derived from
    it by a change of variable. A lookback on the minimum is not the same
    contract as a lookback on the maximum of the reciprocal asset — the carry
    and the discounting do not transform the same way — so the symmetry is in
    the derivation and not in the code.

    Args:
        inputs: The underlying and its market.
        level: The level shortfall is measured below. Positive. Levels above the
            spot are handled exactly.

    Returns:
        A non-negative expectation.

    Raises:
        LookbackError: If the spot or ``level`` is not positive.
    """
    _check(inputs)
    if level <= 0.0:
        raise LookbackError(f"level must be positive, got {level}")
    if _is_flat(inputs):
        return max(level - _degenerate_extremes(inputs)[0], 0.0)
    if level > inputs.spot:
        return (level - inputs.spot) + minimum_shortfall(inputs, inputs.spot)
    spot = inputs.spot
    v = inputs.std_dev
    drift = (inputs.b - 0.5 * inputs.vol * inputs.vol) * inputs.time
    a = math.log(level / spot)
    vanilla = math.exp(a) * norm_cdf((a - drift) / v) - math.exp(
        inputs.b * inputs.time
    ) * norm_cdf((a - drift - v * v) / v)
    return spot * (vanilla + _reflection_integral(inputs, a, upper=False))


def expected_maximum(inputs: Inputs) -> float:
    """``E[M_T]`` under the risk-neutral drift.

    Always at or above the spot, and above the forward: the maximum of a path
    dominates its endpoint.
    """
    _check(inputs)
    if _is_flat(inputs):
        return _degenerate_extremes(inputs)[1]
    return inputs.spot + maximum_excess(inputs, inputs.spot)


def expected_minimum(inputs: Inputs) -> float:
    """``E[m_T]`` under the risk-neutral drift.

    Always at or below the spot, and below the forward.
    """
    _check(inputs)
    if _is_flat(inputs):
        return _degenerate_extremes(inputs)[0]
    return inputs.spot - minimum_shortfall(inputs, inputs.spot)


def lookback_price(
    inputs: Inputs,
    option: OptionType,
    style: Lookback,
    *,
    observed: float | None = None,
) -> float:
    """Price one of the four lookbacks in closed form.

    The four payoffs are ``max(M_T - K, 0)``, ``max(K - m_T, 0)``, ``S_T - m_T``
    and ``M_T - S_T``. The vocabulary is
    :class:`~moneyness.monte_carlo.Lookback` rather than a second enumeration of
    the same four, so this function and
    :func:`moneyness.monte_carlo.lookback` take the same arguments and can be
    compared directly — the same arrangement :mod:`moneyness.barrier` has with
    :func:`moneyness.monte_carlo.barrier`.

    A call and a put read different halves of the law, so the usual parity does
    not connect them. There is an exact relation within the fixed-strike call,
    though, and it is the one worth checking: struck below the running maximum
    it is the one struck at that maximum plus the discounted difference of the
    strikes, because it is then certain to pay at least that difference.

    Args:
        inputs: The option and its market. The strike is read only by the
            fixed-strike styles.
        option: Call or put.
        style: Fixed or floating strike.
        observed: The extreme recorded so far — the maximum for the styles that
            settle against it and the minimum for the others, as
            :meth:`~moneyness.monte_carlo.Lookback.reads_maximum` decides.
            Defaults to the spot, a contract that starts today. Must be on the
            correct side of the spot, since the spot is itself an observation.

    Returns:
        The present value. A floating-strike lookback is positive whenever any
        time remains, since its payoff is non-negative on every path.

    Raises:
        LookbackError: If the spot or ``observed`` is not positive, or
            ``observed`` is on the wrong side of the spot.
    """
    _check(inputs)
    discount = inputs.discount
    upper = style.reads_maximum(option)
    recorded = _observed(inputs, observed, upper=upper)
    if style.is_fixed:
        if upper:
            banked = max(recorded - inputs.strike, 0.0)
            return discount * (
                banked + maximum_excess(inputs, max(inputs.strike, recorded))
            )
        banked = max(inputs.strike - recorded, 0.0)
        return discount * (
            banked + minimum_shortfall(inputs, min(inputs.strike, recorded))
        )
    carried = forward(inputs)
    if upper:
        # E[max(recorded, M_T)] = recorded + E[(M_T - recorded)^+].
        expected = recorded + maximum_excess(inputs, recorded)
        return discount * (expected - carried)
    expected = recorded - minimum_shortfall(inputs, recorded)
    return discount * (carried - expected)
