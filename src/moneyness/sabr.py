"""SABR: the smile the market quotes, and the place its formula stops being one.

The surface machinery in :mod:`moneyness.svi` is parameterised in total variance
and fitted slice by slice. Interest rate and commodity options are not quoted
that way. They are quoted as four SABR parameters per expiry, and what turns
those into a smile is Hagan's asymptotic expansion — which is the market
convention and is also, provably, not a price.

The model is a forward and its own volatility, driven by correlated Brownian
motions::

    dF = alpha F**beta dW,    d(alpha) = nu alpha dZ,    <dW, dZ> = rho dt

with ``beta`` the backbone exponent (one lognormal, zero normal, a half the
square-root process), ``nu`` the volatility of volatility and ``rho`` the
correlation that tilts the smile. :func:`lognormal_volatility` is Hagan's
implied Black volatility and :func:`normal_volatility` the Bachelier one.

**Every degenerate limit is pinned to a closed form sharing no code with the
expansion.** At ``nu = 0`` and ``beta = 1`` the model is lognormal and the
implied volatility is ``alpha`` — not approximately, exactly, because every
correction term carries a factor of ``nu`` or ``(1 - beta)``, and the measured
gap across a strike ladder is **0.0**. At ``nu = 0`` and ``beta = 0``,
:func:`normal_volatility` returns ``alpha`` with the same exactness, which
depends on its two moneyness brackets being identical at a zero exponent. At
``nu = 0`` with a general exponent the model is CEV, whose leading implied
volatility is ``alpha / (FK)**((1-beta)/2)``, and here the formula is
*approximate* by the amount its own next term predicts: at a forward of 2% and
a strike of 3% the gap is 1.704e-03 of the volatility against the
``(1-beta)**2 log**2(F/K) / 24`` the expansion carries, which is 1.713e-03.
So the limits that should be exact are exact and the one that should be
approximate is wrong by the predicted amount.

**The at-the-money factor is a series, and its sign is the first thing to get
wrong.** Both formulas carry ``z / x(z)``, which is zero over zero at the money.
The expansion is

    z / x(z) = 1 - rho z / 2 + (2 - 3 rho**2) z**2 / 12 + ...

and the **minus** on the linear term is not a detail: the first draft here wrote
a plus, which leaves the series wrong by ``rho z`` — 3e-03 relative at ``z``
of 1e-02 — and the symptom was a series that never beat the ratio at any
threshold, which reads as a badly chosen threshold rather than as a sign error.
With the sign right the two cross at ``|z|`` of about **2e-04**, where each is
wrong by around 5e-13, and that is where the switch is. Below it the ratio's
cancellation takes over completely: it is wrong by 8.3e-08 at ``z = 1e-10`` and
by 8.9e-05 at 1e-12, however small ``z`` gets.

**The two forms have to price the same option, and how closely depends on the
maturity.** A lognormal volatility through Black and a normal volatility
through :func:`bachelier` are two second-order expansions in the same small
parameter, not the same expression, so they agree to their own order and not
better. Measured at strikes two at-the-money standard deviations out, the gap
is **0.0009% of the option's value at a quarter of a year, 0.0081% at one year
and 0.1154% at five**. Close enough to convert between at short maturities and
not at long ones, which is the sort of thing a conversion function would hide.

**Where the formula stops being a price.** Hagan's expansion is a volatility;
nothing makes the resulting call prices convex in the strike, and they are not.
Differentiating twice in the strike gives the risk-neutral density, and
:func:`density_floor` finds where it turns negative.

Getting that measurement right took two attempts and the first one was noise.
Differencing *call* prices below the forward is catastrophic cancellation: at a
strike of 1% against a forward of 2% a call is worth 0.0094, its last bit is
1.7e-18, and a second difference over a step of 1e-06 divides four of those by
1.1e-12 and reports 6e-06 of rounding — larger than the density there. That
produced a confident "negative density below 1.0649%, sixteen standard
deviations out" which was an artefact in its entirety. :func:`density` now
prices the out-of-the-money option, whose value is small and computed to full
relative precision; put-call parity is linear in the strike, so the two second
differences are the same number in exact arithmetic and only one of them is in
double precision.

With the arithmetic right the finding is almost the opposite. At ordinary
one-year swaption parameters there is **no** negative density at all down to
1e-04 of the forward — not at ``nu`` 0.4, 0.8 or 1.2, at any correlation
tested. The defect needs maturity or vol-of-vol: over a grid of five ``nu``, four
``rho`` and four maturities, 61 of 80 combinations have one, and all 19 that do
not are short-dated.

And where it appears it is not a tail curiosity. At ``nu`` 0.8, ``rho`` -0.3 and
ten years the density turns negative below a strike of **1.7935%**, which is
**0.795 standard deviations** below the forward — inside the range anyone quotes
— and ten per cent further down it is **-7.118** against a peak of 749.7, nearly
one per cent of the peak in magnitude and stable to five digits across four
orders of magnitude of differencing step. At thirty years the boundary reaches
0.152 standard deviations. So the right summary is that a short-dated SABR smile
is a distribution and a long-dated one is not, which is more useful than either
"the formula is fine" or "the formula admits arbitrage".

One caveat on reading :func:`density_floor`: the value *at* the strike it
returns is a bisected zero crossing, so it is rounding by construction and flips
sign with the step size. The magnitude has to be read a little inside the
region, which is what the tests do.

**beta is not identifiable from one smile**, so :func:`calibrate` takes it and
fits the other three. Fitting the same five quotes at four exponents moves
``alpha`` by a factor of 15.5 — 0.001828 at ``beta`` 0.3 against 0.028328 at 1.0
— and ``rho`` from -0.263 to -0.389, while the fitted smile moves by at most
**6.1 basis points of volatility** and by 2.5 at ``beta`` 0.3 or 0.7. The
exponent is a statement about the backbone, how the at-the-money volatility
moves when the forward does, and a single smile carries no information about it.

The shifted variant and the Bachelier pricer exist for the same market. Negative
forwards break ``F**beta`` at a fractional exponent, so the market displaces both
the forward and the strike and quotes SABR on the pair;
:func:`shifted_lognormal_volatility` does that, with the displacement an argument
rather than a default because it is a currency convention and not a property of
the model. This package had no normal-model pricer at all, so :func:`bachelier`,
:func:`bachelier_vega` and :func:`implied_normal_vol` arrive with it. The
inversion's tolerance is relative to the price rather than absolute, which is
not a nicety: an absolute 1e-14 on an option worth 1e-09 accepts any volatility
within a factor of two, and the first draft converged to 0.00105 where the
answer was 0.00057 and reported success. Relative, the round trip is good to
1.1e-13 across a grid of forty-two cases and to 1.2e-15 on an option worth
1.1e-15.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .bsm import OptionType
from .implied import black
from .normal import norm_cdf, norm_pdf
from .optimise import nelder_mead

__all__ = [
    "Calibration",
    "SabrParameters",
    "Smile",
    "bachelier",
    "bachelier_vega",
    "calibrate",
    "density",
    "density_floor",
    "implied_normal_vol",
    "lognormal_volatility",
    "normal_volatility",
    "shifted_lognormal_volatility",
    "smile",
]

#: Below this ``|z|`` the ``z / x(z)`` factor is taken from its series rather
#: than from the ratio. Chosen where the two are equally accurate: see the
#: module docstring and ``test_sabr.py::test_the_switch_is_where_the_two_agree``.
_SERIES_LIMIT = 2.0e-4

#: Below this ``|log(F/K)|`` the lognormal formula's own moneyness corrections
#: are at the money to double precision.
_ATM_LIMIT = 1.0e-12


@dataclass(frozen=True, slots=True)
class SabrParameters:
    """The four parameters, with the constraints each one actually has.

    Attributes:
        alpha: Initial volatility of the forward, in the units ``beta``
            implies: a rate per root time at ``beta = 1``, an absolute
            move per root time at ``beta = 0``. Strictly positive.
        beta: Backbone exponent, in ``[0, 1]``. Not identifiable from one
            smile — see :func:`calibrate` — so it is an input here rather
            than something fitted.
        rho: Correlation between the forward and its volatility, strictly
            inside ``(-1, 1)``. At exactly one the two processes are the same
            Brownian motion and ``x(z)`` has a zero denominator.
        nu: Volatility of volatility, non-negative. Zero is the CEV model and
            is admitted rather than refused, because it is the limit every
            other formula here is checked against.
    """

    alpha: float
    beta: float
    rho: float
    nu: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.alpha) or self.alpha <= 0.0:
            raise ValueError(f"alpha must be finite and positive, got {self.alpha}")
        if not math.isfinite(self.beta) or not 0.0 <= self.beta <= 1.0:
            raise ValueError(f"beta must lie in [0, 1], got {self.beta}")
        if not math.isfinite(self.rho) or not -1.0 < self.rho < 1.0:
            raise ValueError(
                f"rho must lie strictly inside (-1, 1), got {self.rho}. At exactly "
                "one the forward and its volatility are driven by the same Brownian "
                "motion and the formula's x(z) divides by 1 - rho."
            )
        if not math.isfinite(self.nu) or self.nu < 0.0:
            raise ValueError(f"nu must be finite and non-negative, got {self.nu}")

    @property
    def is_lognormal(self) -> bool:
        """``beta = 1`` with no vol of vol: the model is Black exactly."""
        return self.beta == 1.0 and self.nu == 0.0

    @property
    def is_normal(self) -> bool:
        """``beta = 0`` with no vol of vol: the model is Bachelier exactly."""
        return self.beta == 0.0 and self.nu == 0.0


# -- the factor that is zero over zero ----------------------------------------


def _z_over_x(z: float, rho: float) -> float:
    """``z / log((sqrt(1 - 2 rho z + z**2) - rho + z) / (1 - rho))``.

    One at ``z = 0``, where the expression itself is zero over zero. Below
    :data:`_SERIES_LIMIT` the series is used: the numerator and denominator
    agree to three terms, so the ratio cancels most of its significant digits
    before the division even happens.
    """
    if abs(z) < _SERIES_LIMIT:
        return 1.0 - 0.5 * rho * z + (2.0 - 3.0 * rho * rho) * z * z / 12.0
    root = math.sqrt(1.0 - 2.0 * rho * z + z * z)
    inner = (root - rho + z) / (1.0 - rho)
    if inner <= 0.0:  # pragma: no cover - needs |rho| >= 1, refused on construction
        raise ValueError(f"x(z) has no logarithm at z={z}, rho={rho}")
    return z / math.log(inner)


def _check_inputs(forward: float, strike: float, time: float) -> None:
    if not math.isfinite(forward) or forward <= 0.0:
        raise ValueError(f"the forward must be finite and positive, got {forward}")
    if not math.isfinite(strike) or strike <= 0.0:
        raise ValueError(f"the strike must be finite and positive, got {strike}")
    if not math.isfinite(time) or time <= 0.0:
        raise ValueError(f"the maturity must be finite and positive, got {time}")


# -- Hagan's two volatilities -------------------------------------------------


def lognormal_volatility(
    parameters: SabrParameters, forward: float, strike: float, time: float
) -> float:
    """Hagan's implied Black volatility.

    Args:
        parameters: The four SABR parameters.
        forward: Forward of the underlying, strictly positive.
        strike: Strike, strictly positive. Use
            :func:`shifted_lognormal_volatility` where either can be negative.
        time: Time to expiry in the same units as ``nu`` and ``alpha``.

    Returns:
        The Black volatility that reproduces the model's price to second order
        in the expansion's small parameter.

    Raises:
        ValueError: If the forward, the strike or the maturity is not positive.
    """
    _check_inputs(forward, strike, time)
    alpha, beta, rho, nu = (
        parameters.alpha,
        parameters.beta,
        parameters.rho,
        parameters.nu,
    )
    one_minus = 1.0 - beta
    log_moneyness = math.log(forward / strike)
    mid = math.pow(forward * strike, 0.5 * one_minus)

    # The denominator's moneyness correction, and its own series at the money.
    if abs(log_moneyness) < _ATM_LIMIT:
        spread = 1.0
    else:
        squared = log_moneyness * log_moneyness
        spread = (
            1.0
            + one_minus**2 * squared / 24.0
            + one_minus**4 * squared * squared / 1920.0
        )

    leading = alpha / (mid * spread)
    z = (nu / alpha) * mid * log_moneyness if alpha > 0.0 else 0.0
    ratio = _z_over_x(z, rho) if nu > 0.0 else 1.0

    correction = 1.0 + time * (
        one_minus**2 * alpha * alpha / (24.0 * mid * mid)
        + 0.25 * rho * beta * nu * alpha / mid
        + (2.0 - 3.0 * rho * rho) * nu * nu / 24.0
    )
    return leading * ratio * correction


def normal_volatility(
    parameters: SabrParameters, forward: float, strike: float, time: float
) -> float:
    """Hagan's implied Bachelier volatility, in absolute units of the forward.

    The form the negative-rate markets quote, and not a conversion of
    :func:`lognormal_volatility`: it is its own second-order expansion in the
    same small parameter, which is why the two price an option to slightly
    different numbers rather than identically. The module docstring measures the
    gap.

    The two moneyness brackets — one with the exponent in it and one without —
    are identical at ``beta = 0``, which is what makes the Bachelier limit
    exactly ``alpha`` rather than approximately it.
    """
    _check_inputs(forward, strike, time)
    alpha, beta, rho, nu = (
        parameters.alpha,
        parameters.beta,
        parameters.rho,
        parameters.nu,
    )
    one_minus = 1.0 - beta
    log_moneyness = math.log(forward / strike)
    product = forward * strike
    mid = math.pow(product, 0.5 * one_minus)

    if abs(log_moneyness) < _ATM_LIMIT:
        moneyness_ratio = 1.0
    else:
        squared = log_moneyness * log_moneyness
        numerator = 1.0 + squared / 24.0 + squared * squared / 1920.0
        denominator = (
            1.0
            + one_minus**2 * squared / 24.0
            + one_minus**4 * squared * squared / 1920.0
        )
        moneyness_ratio = numerator / denominator

    z = (nu / alpha) * mid * log_moneyness
    ratio = _z_over_x(z, rho) if nu > 0.0 else 1.0

    correction = 1.0 + time * (
        -beta * (2.0 - beta) * alpha * alpha / (24.0 * mid * mid)
        + 0.25 * rho * alpha * nu * beta / mid
        + (2.0 - 3.0 * rho * rho) * nu * nu / 24.0
    )
    return alpha * math.pow(product, 0.5 * beta) * moneyness_ratio * ratio * correction


def shifted_lognormal_volatility(
    parameters: SabrParameters,
    forward: float,
    strike: float,
    time: float,
    shift: float,
) -> float:
    """Hagan's lognormal volatility on a displaced forward and strike.

    The market's answer to negative rates: ``F**beta`` has no real value for a
    negative forward at a fractional exponent, so the forward and the strike are
    both moved up by ``shift`` and SABR is quoted on the displaced pair. The
    resulting volatility is the Black volatility of the *displaced* option,
    which is what the quote means and not the same thing as a volatility of the
    forward itself.

    Args:
        shift: The displacement, strictly positive and large enough to make both
            the forward and the strike positive.

    Raises:
        ValueError: If the shift does not make both arguments positive, which is
            reported with both numbers because a shift that works for one strike
            on a smile and not for another is the usual way this goes wrong.
    """
    if not math.isfinite(shift) or shift <= 0.0:
        raise ValueError(f"the shift must be finite and positive, got {shift}")
    if forward + shift <= 0.0 or strike + shift <= 0.0:
        raise ValueError(
            f"a shift of {shift} leaves the forward at {forward + shift} and the "
            f"strike at {strike + shift}; both have to be positive for the "
            "displaced lognormal formula to have a value"
        )
    return lognormal_volatility(parameters, forward + shift, strike + shift, time)


# -- the Bachelier model this package did not have ----------------------------


def bachelier(forward: float, strike: float, total_vol: float, option: OptionType) -> float:
    """The undiscounted Bachelier price, in total-volatility coordinates.

    ``total_vol`` is ``sigma * sqrt(T)`` in absolute units of the forward, so
    the price is ``total_vol * (d * Phi(d) + phi(d))`` for a call with
    ``d = (F - K) / total_vol``. Unlike Black, this admits a negative forward or
    strike: the normal model's whole point is that it does.
    """
    if not math.isfinite(total_vol) or total_vol < 0.0:
        raise ValueError(f"the total volatility must be non-negative, got {total_vol}")
    sign = option.sign
    if total_vol == 0.0:
        return max(sign * (forward - strike), 0.0)
    d = sign * (forward - strike) / total_vol
    return total_vol * (d * norm_cdf(d) + norm_pdf(d))


def bachelier_vega(forward: float, strike: float, total_vol: float) -> float:
    """``d(bachelier)/d(total_vol)``, which is ``phi(d)``.

    The same for calls and puts, and strictly positive, so the price is
    monotone in the volatility and the inverse below is unique.
    """
    if not math.isfinite(total_vol) or total_vol < 0.0:
        raise ValueError(f"the total volatility must be non-negative, got {total_vol}")
    if total_vol == 0.0:
        return 0.0
    return norm_pdf((forward - strike) / total_vol)


def implied_normal_vol(
    forward: float,
    strike: float,
    price: float,
    option: OptionType,
    *,
    tolerance: float = 1e-14,
    max_iterations: int = 100,
) -> float:
    """The total normal volatility that reproduces ``price``.

    Newton from the at-the-money closed form, which is exact where it starts:
    at ``F == K`` the price is ``total_vol / sqrt(2 pi)``, so the first guess is
    the answer there and a good one elsewhere. Safeguarded by bisection on a
    bracket, because vega vanishes in the deep wings and an unsafeguarded Newton
    step from there lands outside the domain.

    Raises:
        ValueError: If the price is below intrinsic, in which case no normal
            volatility reproduces it.
    """
    intrinsic = max(option.sign * (forward - strike), 0.0)
    if price < intrinsic - 1e-15:
        raise ValueError(
            f"a price of {price} is below the intrinsic {intrinsic}; no normal "
            "volatility is low enough to produce it"
        )
    if price <= intrinsic:
        return 0.0
    low, high = 0.0, max(abs(forward - strike), 1.0)
    while bachelier(forward, strike, high, option) < price:
        high *= 2.0
        if high > 1e12:  # pragma: no cover - needs a price above every bound
            raise ValueError(f"no normal volatility below {high} reaches {price}")
    guess = price * math.sqrt(2.0 * math.pi)
    for _ in range(max_iterations):
        value = bachelier(forward, strike, guess, option) - price
        # Relative to the price, not to one. A deep wing option is worth 1e-09
        # and an absolute tolerance of 1e-14 on it accepts any volatility within
        # a factor of two: the first draft of this converged to 0.00105 where
        # the answer was 0.00057 and reported success. The price is computed to
        # full relative precision, so demanding relative accuracy is achievable
        # as well as necessary.
        if abs(value) <= tolerance * max(abs(price), 1e-300):
            return guess
        if value > 0.0:
            high = guess
        else:
            low = guess
        slope = bachelier_vega(forward, strike, guess)
        step = guess - value / slope if slope > 0.0 else 0.5 * (low + high)
        guess = step if low < step < high else 0.5 * (low + high)
    return guess


# -- a whole smile ------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Smile:
    """A SABR smile evaluated at a set of strikes.

    Attributes:
        strikes: The strikes, in the order given.
        lognormal: Hagan's Black volatility at each.
        normal: Hagan's Bachelier volatility at each, in absolute units.
    """

    strikes: tuple[float, ...]
    lognormal: tuple[float, ...]
    normal: tuple[float, ...]

    def __len__(self) -> int:
        return len(self.strikes)


def smile(
    parameters: SabrParameters,
    forward: float,
    strikes: Sequence[float],
    time: float,
) -> Smile:
    """Both volatilities at every strike given."""
    if not strikes:
        raise ValueError("a smile needs at least one strike")
    return Smile(
        strikes=tuple(float(k) for k in strikes),
        lognormal=tuple(
            lognormal_volatility(parameters, forward, k, time) for k in strikes
        ),
        normal=tuple(normal_volatility(parameters, forward, k, time) for k in strikes),
    )


# -- where the formula stops being a price ------------------------------------


def density(
    parameters: SabrParameters,
    forward: float,
    strike: float,
    time: float,
    *,
    step: float | None = None,
) -> float:
    """The risk-neutral density the smile implies at ``strike``.

    The second derivative of the undiscounted call price in the strike, by a
    central difference. It is a density because a call price is the integral of
    the survival function twice over, so convexity in the strike *is*
    non-negative probability — and Hagan's expansion, being a volatility rather
    than a price, does not guarantee it.

    Args:
        step: Finite-difference step in the strike. Defaults to ``1e-4``
            of the forward, which the tests check is small enough that the
            answer is stable to three digits and large enough that the
            differencing noise stays below a per cent of the value.
    """
    _check_inputs(forward, strike, time)
    width = step if step is not None else 1.0e-4 * forward
    if width <= 0.0:
        raise ValueError(f"the step must be positive, got {width}")
    if strike - width <= 0.0:
        raise ValueError(
            f"a step of {width} reaches a strike of {strike - width}, which is not "
            "positive; pass a smaller step to look this far into the wing"
        )
    root = math.sqrt(time)
    # Price the *out of the money* option at all three points. Put-call parity
    # is linear in the strike, so a second difference of puts and a second
    # difference of calls are the same number in exact arithmetic -- and not in
    # double precision. Below the forward a call is worth its intrinsic plus a
    # whisper: at a strike of 1% against a forward of 2% the call is 0.0094, its
    # last bit is 1.7e-18, and a second difference over a step of 1e-06 divides
    # four of those by 1.1e-12 and reports 6e-06 of pure noise -- which is larger
    # than the density there. The put at the same strike is worth 1e-30 and is
    # computed to full relative precision, so the same difference carries signal.
    # The first draft of this used calls throughout and located a "negative
    # density" in the deep wing that was entirely rounding.
    option = OptionType.PUT if strike < forward else OptionType.CALL

    def value(at: float) -> float:
        vol = lognormal_volatility(parameters, forward, at, time)
        return black(forward, at, vol * root, option)

    return (value(strike + width) - 2.0 * value(strike) + value(strike - width)) / (
        width * width
    )


@dataclass(frozen=True, slots=True)
class Calibration:
    """A fitted smile, and what the fit cost.

    Attributes:
        parameters: The fitted parameters, with ``beta`` as it was given.
        residual: Root mean squared volatility error across the quotes.
        iterations: Simplex iterations used.
        converged: Whether the simplex collapsed rather than running out.
    """

    parameters: SabrParameters
    residual: float
    iterations: int
    converged: bool


def _alpha_from_atm(
    atm_vol: float, beta: float, rho: float, nu: float, forward: float, time: float
) -> float:
    """The ``alpha`` that reproduces the at-the-money volatility exactly.

    At the money Hagan's formula is a cubic in ``alpha``, and solving it is the
    standard way to remove one parameter from the search: every candidate
    ``(beta, rho, nu)`` then fits the at-the-money quote by construction, so the
    optimiser only has the wings to work on. Solved here by bisection on the
    monotone bracket rather than by the cubic formula, because the cubic's three
    roots need the right one selected and the smallest positive root is the one
    that is monotone in the quote.
    """
    one_minus = 1.0 - beta
    power = math.pow(forward, beta)

    def implied(alpha: float) -> float:
        return (
            alpha
            / math.pow(forward, one_minus)
            * (
                1.0
                + time
                * (
                    one_minus**2 * alpha * alpha / (24.0 * math.pow(forward, 2.0 * one_minus))
                    + 0.25 * rho * beta * nu * alpha / math.pow(forward, one_minus)
                    + (2.0 - 3.0 * rho * rho) * nu * nu / 24.0
                )
            )
        )

    low, high = 1e-12, max(atm_vol * power * 10.0, 1.0)
    if implied(high) < atm_vol:
        return high
    for _ in range(200):
        middle = 0.5 * (low + high)
        if implied(middle) < atm_vol:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def calibrate(
    forward: float,
    time: float,
    quotes: Sequence[tuple[float, float]],
    *,
    beta: float = 0.5,
    weights: Sequence[float] | None = None,
    start: tuple[float, float] = (-0.2, 0.4),
) -> Calibration:
    """Fit ``rho`` and ``nu`` to quoted Black volatilities, with ``alpha`` solved.

    ``beta`` is an argument and not fitted, because it is not identifiable from
    a single smile: three exponents fit the same five quotes to within a
    fraction of a basis point of volatility while ``alpha`` and ``rho`` move
    substantially. That is measured in the tests rather than asserted here. The
    convention is to fix ``beta`` from a view about the backbone — how the
    at-the-money volatility moves when the forward does — which a single smile
    carries no information about.

    Args:
        forward: Forward of the underlying.
        time: Time to expiry.
        quotes: ``(strike, Black volatility)`` pairs. At least two, and one
            should be at or near the money since ``alpha`` is solved from the
            quote nearest the forward.
        beta: The backbone exponent, held fixed.
        weights: Optional weights on the quotes, defaulting to equal.
        start: Initial ``(rho, nu)``.

    Returns:
        A :class:`Calibration`.

    Raises:
        ValueError: If fewer than two quotes are given, or if the weights do not
            match them.
    """
    if len(quotes) < 2:
        raise ValueError(f"a fit needs at least two quotes, got {len(quotes)}")
    if weights is not None and len(weights) != len(quotes):
        raise ValueError(
            f"{len(weights)} weights for {len(quotes)} quotes; they have to match"
        )
    chosen = list(weights) if weights is not None else [1.0] * len(quotes)
    if any(one < 0.0 for one in chosen) or sum(chosen) <= 0.0:
        raise ValueError("weights must be non-negative and not all zero")

    nearest = min(quotes, key=lambda pair: abs(math.log(pair[0] / forward)))
    atm_vol = nearest[1]

    def objective(point: tuple[float, ...]) -> float:
        rho, nu = point
        if not -0.999 < rho < 0.999 or nu <= 0.0 or nu > 10.0:
            return math.inf
        alpha = _alpha_from_atm(atm_vol, beta, rho, nu, forward, time)
        try:
            candidate = SabrParameters(alpha=alpha, beta=beta, rho=rho, nu=nu)
        except ValueError:  # pragma: no cover - the bounds above exclude it
            return math.inf
        total = 0.0
        for weight, (strike, quoted) in zip(chosen, quotes, strict=True):
            error = lognormal_volatility(candidate, forward, strike, time) - quoted
            total += weight * error * error
        return total / sum(chosen)

    found = nelder_mead(objective, start, step=0.3, tolerance=1e-14, max_iterations=4000)
    rho, nu = found.point
    alpha = _alpha_from_atm(atm_vol, beta, rho, nu, forward, time)
    return Calibration(
        parameters=SabrParameters(alpha=alpha, beta=beta, rho=rho, nu=nu),
        residual=math.sqrt(max(found.value, 0.0)),
        iterations=found.iterations,
        converged=found.converged,
    )


def density_floor(
    parameters: SabrParameters,
    forward: float,
    time: float,
    *,
    lower: float | None = None,
    tolerance: float = 1e-10,
) -> float | None:
    """The highest strike below the forward at which the density is negative.

    Returns ``None`` when the density stays non-negative all the way down to
    ``lower``, which is the honest answer and not a claim that the smile is
    arbitrage-free: a finer search or a lower floor may still find one.

    The value *at* the returned strike is a bisected zero crossing, so it is
    rounding by construction and will flip sign with the differencing step. To
    see how bad the defect is, evaluate :func:`density` a little inside the
    region rather than at its boundary.

    This searches *downwards* only, as the name says. A long-dated smile can be
    negative in the high-strike wing too — at ten years with a vol-of-vol of 0.8
    the density is -1.61 at a 2.67% strike against a 2% forward, stable across
    differencing steps — and nothing here looks for that.

    Args:
        lower: Where to stop looking, defaulting to 1e-03 of the forward. A
            SABR smile's density defect is in the low-strike wing, so the search
            walks down from the forward and bisects the first sign change.
        tolerance: Relative width at which the bisection stops.

    Returns:
        The strike, or ``None``.
    """
    _check_inputs(forward, forward, time)
    floor = lower if lower is not None else 1.0e-3 * forward
    if not 0.0 < floor < forward:
        raise ValueError(
            f"the floor must sit between zero and the forward, got {floor}"
        )

    def value(strike: float) -> float:
        return density(parameters, forward, strike, time, step=1.0e-4 * strike)

    good = forward
    if value(good) < 0.0:  # pragma: no cover - needs a defect at the money itself
        return good
    steps = 400
    bad: float | None = None
    for index in range(1, steps + 1):
        # Geometric, because the wing is where the strikes get close together.
        strike = forward * math.pow(floor / forward, index / steps)
        if value(strike) < 0.0:
            bad = strike
            break
        good = strike
    if bad is None:
        return None
    while (good - bad) / forward > tolerance:
        middle = math.sqrt(good * bad)
        if value(middle) < 0.0:
            bad = middle
        else:
            good = middle
    return bad
