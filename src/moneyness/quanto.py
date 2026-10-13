"""Options on a foreign asset, where the exchange rate is part of the payoff.

An option on a share quoted abroad has two random quantities in it and four
common ways of writing the contract. They are not variants of one price; they
are four different claims, and the correlation between the share and the
exchange rate enters each of them differently — in one case not at all. Writing
``X`` for the exchange rate in **domestic units per one foreign unit**, ``S``
for the share's price in foreign currency, and a payoff in domestic currency
throughout:

======================================= ==================== ==================
contract                                payoff               correlation
======================================= ==================== ==================
:attr:`Linkage.FOREIGN_STRUCK`          ``X_T (S_T - K_f)+``  absent
:attr:`Linkage.QUANTO`                  ``X_fix (S_T - K_f)+`` lowers a call
:attr:`Linkage.COMPOSITE`               ``(S_T X_T - K_d)+``   raises a call
:attr:`Linkage.EQUITY_LINKED_FX`        ``S_T (X_T - K_x)+``   raises a call
======================================= ==================== ==================

**The foreign-struck option does not depend on the exchange rate's dynamics at
all.** Converting a foreign-currency payoff at whatever rate prevails is the
same as holding the foreign claim, so the domestic value is today's exchange
rate times the foreign-currency Black-Scholes price. Neither the exchange rate's
volatility nor the correlation appears, which is worth knowing because it is the
one case where a correlation estimate cannot be wrong, and worth asserting
because every other case here is sensitive to it.

**The quanto fixes the conversion rate, which costs a drift.** Fixing the rate
makes the payoff a foreign-currency quantity paid in domestic currency, and
under the domestic risk-neutral measure the share then drifts at
``r_f - q - rho sigma_S sigma_X`` rather than at ``r_f - q``. That correction
is the whole of the quanto, and it enters through the product ``rho sigma_X``
and nothing else.

**The composite option strikes in domestic currency, so the underlying is the
product.** ``S X`` is a domestic-currency tradeable with the share's own
dividend yield, and its volatility is
``sqrt(sigma_S^2 + sigma_X^2 + 2 rho sigma_S sigma_X)``. The correlation enters
with a plus where the quanto's enters with a minus, so one number moves the two
prices in opposite directions — which is the single most useful thing to know
about this pair.

**The equity-linked exchange rate option is an exchange option.** Its payoff is
``S_T X_T - K_x S_T`` when it pays, the difference between a domestic tradeable
and ``K_x`` quanto shares, and the ratio of those two is ``X``. So Margrabe
applies with the exchange rate's own volatility, and the correlation enters only
through the quanto leg.

**One quote cannot separate the exchange rate's volatility from the
correlation, and two can.** A quanto price fixes ``rho sigma_X``; a composite
price fixes ``sigma_X^2 + 2 rho sigma_S sigma_X``. Either alone leaves a curve
of indistinguishable pairs. Together they are a linear system in
``rho sigma_X`` and ``sigma_X^2``, so :func:`implied_fx_pair` inverts them in
closed form with no search — and refuses the pair when the implied variance
comes out negative or the implied correlation leaves ``[-1, 1]``, which an
inconsistent pair of quotes does rather than warning.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from .bsm import Inputs, OptionType
from .bsm import price as bsm_price
from .normal import norm_cdf

__all__ = [
    "CrossCurrency",
    "FxPair",
    "Linkage",
    "QuantoError",
    "composite_volatility",
    "domestic_forward",
    "exchange_forward",
    "implied_fx_pair",
    "parity_gap",
    "price",
    "quanto_drift",
    "quanto_forward",
]


class QuantoError(ValueError):
    """Inputs that do not describe a priceable cross-currency option."""


class Linkage(str, Enum):
    """How the exchange rate enters the payoff."""

    FOREIGN_STRUCK = "foreign struck"
    """``X_T max(S_T - K_f, 0)``, struck abroad and converted at maturity."""

    QUANTO = "quanto"
    """``X_fix max(S_T - K_f, 0)``, struck abroad and converted at a fixed rate."""

    COMPOSITE = "composite"
    """``max(S_T X_T - K_d, 0)``, struck in domestic currency on the product."""

    EQUITY_LINKED_FX = "equity linked fx"
    """``S_T max(X_T - K_x, 0)``, an exchange rate option with a share as notional."""


@dataclass(frozen=True, slots=True)
class CrossCurrency:
    """A foreign share, an exchange rate, and the two currencies' rates.

    The exchange rate is quoted as **domestic units per one foreign unit**, and
    the correlation is between ``log S`` and ``log X`` in that quotation. The
    direction matters: quoting the rate the other way up negates the
    correlation and leaves its magnitude alone, so a correlation lifted from a
    screen in the other convention has the right size and the wrong sign, and
    the quanto drift then moves by ``2 rho sigma_S sigma_X`` rather than
    vanishing. :meth:`inverted` performs the change of quotation properly.

    Attributes:
        spot: The share's price in foreign currency. Non-negative.
        exchange: The exchange rate today, domestic per foreign. Positive.
        strike: The strike, in whichever currency the linkage implies — foreign
            for a foreign-struck or quanto option, domestic for a composite one,
            and an exchange rate for an equity-linked one.
        time: Year fraction to expiry. Non-negative.
        domestic_rate: Continuously compounded domestic rate.
        foreign_rate: Continuously compounded foreign rate.
        dividend: The share's continuous dividend yield, in foreign currency.
        asset_vol: Annualised lognormal volatility of the share. Non-negative.
        exchange_vol: Annualised lognormal volatility of the exchange rate.
            Non-negative.
        correlation: Correlation of the two log returns, in ``[-1, 1]``.
    """

    spot: float
    exchange: float
    strike: float
    time: float
    domestic_rate: float
    foreign_rate: float
    dividend: float
    asset_vol: float
    exchange_vol: float
    correlation: float

    def __post_init__(self) -> None:
        for name in (
            "spot",
            "exchange",
            "strike",
            "time",
            "domestic_rate",
            "foreign_rate",
            "dividend",
            "asset_vol",
            "exchange_vol",
            "correlation",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise QuantoError(f"{name} must be finite, got {value!r}")
        for name in ("spot", "strike", "time", "asset_vol", "exchange_vol"):
            value = float(getattr(self, name))
            if value < 0.0:
                raise QuantoError(f"{name} must be non-negative, got {value!r}")
        if self.exchange <= 0.0:
            raise QuantoError(
                f"the exchange rate must be positive, got {self.exchange!r}; a "
                "quotation of zero has no reciprocal and no logarithm"
            )
        if not -1.0 <= self.correlation <= 1.0:
            raise QuantoError(f"correlation must lie in [-1, 1], got {self.correlation!r}")

    @property
    def total_asset_vol(self) -> float:
        """``sigma_S sqrt(T)``."""
        return self.asset_vol * math.sqrt(self.time)

    @property
    def total_exchange_vol(self) -> float:
        """``sigma_X sqrt(T)``."""
        return self.exchange_vol * math.sqrt(self.time)

    @property
    def domestic_discount(self) -> float:
        """``e^{-r_d T}``."""
        return math.exp(-self.domestic_rate * self.time)

    def inverted(self) -> CrossCurrency:
        """The same market quoted the other way up.

        The two currencies change roles, the exchange rate becomes its own
        reciprocal, the correlation changes sign, and the strike is left alone
        because its meaning depends on the linkage rather than on the quotation.
        Applying this twice is the identity, which is the only property worth
        asserting about it.
        """
        return CrossCurrency(
            spot=self.spot,
            exchange=1.0 / self.exchange,
            strike=self.strike,
            time=self.time,
            domestic_rate=self.foreign_rate,
            foreign_rate=self.domestic_rate,
            dividend=self.dividend,
            asset_vol=self.asset_vol,
            exchange_vol=self.exchange_vol,
            correlation=-self.correlation,
        )


def quanto_drift(state: CrossCurrency) -> float:
    """``-rho sigma_S sigma_X``: what fixing the conversion rate costs the drift.

    Negative for a positively correlated pair, which is why a quanto call on a
    share that rises with its own currency is worth less than the unquantoed
    one. The share and the exchange rate enter only as a product, so this is
    also the only channel through which either reaches a quanto price.
    """
    return -state.correlation * state.asset_vol * state.exchange_vol


def quanto_forward(state: CrossCurrency) -> float:
    """The share's forward under the domestic risk-neutral measure.

    ``S_0 exp((r_f - q - rho sigma_S sigma_X) T)``, in foreign currency units.
    """
    carry = state.foreign_rate - state.dividend + quanto_drift(state)
    return state.spot * math.exp(carry * state.time)


def exchange_forward(state: CrossCurrency) -> float:
    """The exchange rate's forward, ``X_0 exp((r_d - r_f) T)``."""
    return state.exchange * math.exp(
        (state.domestic_rate - state.foreign_rate) * state.time
    )


def domestic_forward(state: CrossCurrency) -> float:
    """The forward of the share's domestic-currency value, ``S X``.

    ``S_0 X_0 exp((r_d - q) T)``. The foreign rate cancels: holding the share
    abroad earns the foreign rate and funding the currency position costs it,
    so a domestic investor's carry on the pair is the domestic rate less the
    share's own yield, whatever the two countries' rates are.
    """
    carry = state.domestic_rate - state.dividend
    return state.spot * state.exchange * math.exp(carry * state.time)


def composite_volatility(state: CrossCurrency) -> float:
    """``sqrt(sigma_S^2 + sigma_X^2 + 2 rho sigma_S sigma_X)``.

    The volatility of the product, which is the underlying of a composite
    option. The correlation enters with a plus here and with a minus in
    :func:`quanto_drift`, which is the reason the two contracts respond to it in
    opposite directions. The expression under the root is a variance and cannot
    be negative for a correlation in ``[-1, 1]``, but it can round below zero at
    ``rho = -1`` with equal volatilities, where the product is exactly
    deterministic, so it is floored rather than trusted.
    """
    variance = (
        state.asset_vol**2
        + state.exchange_vol**2
        + 2.0 * state.correlation * state.asset_vol * state.exchange_vol
    )
    return math.sqrt(max(variance, 0.0))


def _margrabe(first: float, second: float, total_vol: float, option: OptionType) -> float:
    """``first N(d1) - second N(d2)`` for a call, on two present values.

    Both arguments are already discounted, so no rate appears. The degenerate
    cases are limits rather than errors: zero total volatility is the
    deterministic payoff and a zero present value on either side is the other
    one outright.
    """
    sign = option.sign
    if total_vol <= 0.0 or first <= 0.0 or second <= 0.0:
        return max(sign * (first - second), 0.0)
    log = math.log(first / second)
    d1 = (log + 0.5 * total_vol * total_vol) / total_vol
    d2 = d1 - total_vol
    return sign * (first * norm_cdf(sign * d1) - second * norm_cdf(sign * d2))


def price(
    state: CrossCurrency,
    linkage: Linkage,
    option: OptionType = OptionType.CALL,
    *,
    fixed_exchange: float | None = None,
) -> float:
    """The option's value today, in domestic currency.

    Args:
        state: The two assets and the two currencies' rates.
        linkage: How the exchange rate enters the payoff.
        option: Call or put.
        fixed_exchange: The contractual conversion rate for
            :attr:`Linkage.QUANTO`. Defaults to today's rate, which is the usual
            contract. Ignored by every other linkage, because they have no fixed
            rate in them.

    Returns:
        The present value in domestic currency.

    Raises:
        QuantoError: If a fixed exchange rate is given that is not positive.
    """
    if fixed_exchange is not None and (
        not math.isfinite(fixed_exchange) or fixed_exchange <= 0.0
    ):
        raise QuantoError(
            f"a fixed conversion rate must be positive and finite, got {fixed_exchange!r}"
        )

    if linkage is Linkage.FOREIGN_STRUCK:
        # The foreign-currency option, valued abroad, converted at today's rate.
        foreign = bsm_price(
            Inputs(
                spot=state.spot,
                strike=state.strike,
                time=state.time,
                rate=state.foreign_rate,
                vol=state.asset_vol,
                carry=state.foreign_rate - state.dividend,
            ),
            option,
        )
        return state.exchange * foreign

    if linkage is Linkage.QUANTO:
        rate = state.exchange if fixed_exchange is None else fixed_exchange
        quantoed = bsm_price(
            Inputs(
                spot=state.spot,
                strike=state.strike,
                time=state.time,
                rate=state.domestic_rate,
                vol=state.asset_vol,
                carry=state.foreign_rate - state.dividend + quanto_drift(state),
            ),
            option,
        )
        return rate * quantoed

    if linkage is Linkage.COMPOSITE:
        return bsm_price(
            Inputs(
                spot=state.spot * state.exchange,
                strike=state.strike,
                time=state.time,
                rate=state.domestic_rate,
                vol=composite_volatility(state),
                carry=state.domestic_rate - state.dividend,
            ),
            option,
        )

    # Equity-linked: an exchange option between the share's domestic value and
    # K_x quanto shares, whose ratio is the exchange rate.
    discount = state.domestic_discount
    asset = discount * domestic_forward(state)
    cash = discount * state.strike * quanto_forward(state)
    return _margrabe(asset, cash, state.total_exchange_vol, option)


def parity_gap(
    state: CrossCurrency, linkage: Linkage, *, fixed_exchange: float | None = None
) -> float:
    """``call - put`` less the forward value the contract implies.

    Zero for every linkage, which is worth having as one function because the
    forward differs in each: a foreign-struck option's parity is the foreign
    one scaled by today's exchange rate, a quanto's is written on the drifted
    forward, a composite's on the product, and an equity-linked option's on the
    difference between the product and the quanto share. Four parities and one
    number, which is a transcription check on all four.
    """
    call = price(state, linkage, OptionType.CALL, fixed_exchange=fixed_exchange)
    put = price(state, linkage, OptionType.PUT, fixed_exchange=fixed_exchange)
    discount = state.domestic_discount
    if linkage is Linkage.FOREIGN_STRUCK:
        foreign = math.exp(-state.foreign_rate * state.time)
        forward = state.spot * math.exp((state.foreign_rate - state.dividend) * state.time)
        return call - put - state.exchange * foreign * (forward - state.strike)
    if linkage is Linkage.QUANTO:
        rate = state.exchange if fixed_exchange is None else fixed_exchange
        return call - put - rate * discount * (quanto_forward(state) - state.strike)
    if linkage is Linkage.COMPOSITE:
        return call - put - discount * (domestic_forward(state) - state.strike)
    return (
        call
        - put
        - discount * (domestic_forward(state) - state.strike * quanto_forward(state))
    )


@dataclass(frozen=True, slots=True)
class FxPair:
    """An exchange rate volatility and a correlation, recovered together.

    Attributes:
        exchange_vol: The implied ``sigma_X``.
        correlation: The implied ``rho``.
        drift: The implied quanto drift, ``-rho sigma_S sigma_X``, which is what
            the quanto quote determines on its own.
        composite_vol: The implied volatility of the product, which is what the
            composite quote determines on its own.
    """

    exchange_vol: float
    correlation: float
    drift: float
    composite_vol: float


def implied_fx_pair(
    state: CrossCurrency, quanto_drift_implied: float, composite_vol_implied: float
) -> FxPair:
    """Recover ``(sigma_X, rho)`` from a quanto drift and a composite volatility.

    Each quote determines one function of the pair and neither determines both:
    a quanto fixes ``rho sigma_X`` and a composite fixes
    ``sigma_X^2 + 2 rho sigma_S sigma_X``. Substituting the first into the second
    leaves ``sigma_X^2`` alone, so the system is linear in
    ``(rho sigma_X, sigma_X^2)`` and inverts without a search. The closed form
    is also what makes the failure mode visible: an inconsistent pair of quotes
    produces a negative variance, which a two-dimensional search would instead
    report as a poor fit at some plausible-looking corner.

    Args:
        state: The market. Only ``asset_vol`` is read; the exchange rate
            volatility and correlation it carries are what this recovers and are
            ignored.
        quanto_drift_implied: The drift a quanto quote implies, as
            :func:`quanto_drift` would return it — negative for a positively
            correlated pair.
        composite_vol_implied: The total volatility a composite quote implies.

    Returns:
        The pair, with the two single-quote quantities alongside.

    Raises:
        QuantoError: If the implied variance is negative, if the asset
            volatility is zero — a share with no volatility cannot carry a
            correlation, so the quanto quote says nothing — or if the implied
            correlation leaves ``[-1, 1]``.
    """
    if not math.isfinite(quanto_drift_implied) or not math.isfinite(composite_vol_implied):
        raise QuantoError("both implied quantities must be finite")
    if composite_vol_implied < 0.0:
        raise QuantoError(
            f"a total volatility cannot be negative, got {composite_vol_implied!r}"
        )
    if state.asset_vol <= 0.0:
        raise QuantoError(
            "the share's volatility is zero, so the quanto drift is zero whatever "
            "the correlation is and the pair cannot be separated"
        )
    product = -quanto_drift_implied / state.asset_vol
    variance = composite_vol_implied**2 - state.asset_vol**2 - 2.0 * state.asset_vol * product
    if variance < 0.0:
        raise QuantoError(
            f"the two quotes imply an exchange rate variance of {variance!r}. The "
            "composite volatility is below what the share's own volatility and the "
            "quanto drift already account for, so no pair reproduces both"
        )
    exchange_vol = math.sqrt(variance)
    if exchange_vol == 0.0:
        if product != 0.0:
            raise QuantoError(
                "the quotes imply a zero exchange rate volatility and a non-zero "
                f"quanto drift of {quanto_drift_implied!r}, which no correlation gives"
            )
        correlation = 0.0
    else:
        correlation = product / exchange_vol
    # A perfectly correlated pair inverts to exactly one in exact arithmetic and
    # to one and a few bits in this one, because the correlation comes out as a
    # ratio of two quantities each of which has been through a square root. A
    # boundary case refused for round-off would be a worse failure than a
    # correlation reported as 1 + 4e-16, so the boundary is snapped and anything
    # beyond a billionth is still refused.
    if 1.0 < abs(correlation) <= 1.0 + 1e-9:
        correlation = math.copysign(1.0, correlation)
    if not -1.0 <= correlation <= 1.0:
        raise QuantoError(
            f"the two quotes imply a correlation of {correlation!r}, which is outside "
            "[-1, 1]; one of them is not consistent with the share's volatility"
        )
    return FxPair(
        exchange_vol=exchange_vol,
        correlation=correlation,
        drift=quanto_drift_implied,
        composite_vol=composite_vol_implied,
    )
