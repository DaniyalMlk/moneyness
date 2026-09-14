"""The generalised Black-Scholes-Merton price.

One formula covers the four conventions by moving the cost of carry rather than
by changing the code. Writing ``b`` for the carry and ``r`` for the discount
rate, the forward is ``S e^{bT}`` and the price discounts at ``r``:

=========================== ============= ==============================
convention                  carry         instrument
=========================== ============= ==============================
Black-Scholes (1973)        ``b = r``     a share paying no dividend
Merton (1973)               ``b = r - q`` a share paying a yield ``q``
Black (1976)                ``b = 0``     a future, margined at ``r``
Garman-Kohlhagen (1983)     ``b = r - f`` a currency earning ``f`` abroad
=========================== ============= ==============================

The degenerate inputs are handled as limits rather than rejected. At zero
volatility or zero remaining time the payoff is deterministic and the price is
the discounted intrinsic value on the forward, which is what the limit of the
formula gives as ``v sqrt(T)`` goes to zero from above. Both boundaries matter
in practice: an implied-volatility search brackets from zero upwards, and a
lattice reads the terminal layer at zero time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from .normal import norm_cdf

__all__ = [
    "Inputs",
    "OptionType",
    "d1_d2",
    "forward",
    "intrinsic",
    "log_moneyness",
    "parity_gap",
    "price",
]


class OptionType(str, Enum):
    """Which side of the strike the holder receives."""

    CALL = "call"
    PUT = "put"

    @property
    def sign(self) -> float:
        """+1 for a call, -1 for a put.

        The two payoffs differ only by this sign once written as
        ``max(sign * (F - K), 0)``, and carrying it explicitly removes every
        branch on option type from the pricing and Greek formulas.
        """
        return 1.0 if self is OptionType.CALL else -1.0


@dataclass(frozen=True, slots=True)
class Inputs:
    """A single option, with its market state.

    Attributes:
        spot: Price of the underlying now. Non-negative.
        strike: Exercise price. Non-negative.
        time: Year fraction to expiry. Non-negative.
        rate: Continuously compounded discount rate.
        vol: Annualised lognormal volatility. Non-negative.
        carry: Cost of carry. Defaults to ``rate``, the non-dividend share case.
    """

    spot: float
    strike: float
    time: float
    rate: float
    vol: float
    carry: float | None = None

    def __post_init__(self) -> None:
        if self.spot < 0.0:
            raise ValueError(f"spot must be non-negative, got {self.spot}")
        if self.strike < 0.0:
            raise ValueError(f"strike must be non-negative, got {self.strike}")
        if self.time < 0.0:
            raise ValueError(f"time must be non-negative, got {self.time}")
        if self.vol < 0.0:
            raise ValueError(f"vol must be non-negative, got {self.vol}")
        for name in ("spot", "strike", "time", "rate", "vol"):
            value = getattr(self, name)
            if value != value or math.isinf(value):
                raise ValueError(f"{name} must be finite, got {value}")

    @property
    def b(self) -> float:
        """The cost of carry actually in force."""
        return self.rate if self.carry is None else self.carry

    @classmethod
    def with_dividend(
        cls,
        spot: float,
        strike: float,
        time: float,
        rate: float,
        vol: float,
        dividend: float,
    ) -> Inputs:
        """Merton's case: a share paying a continuous yield."""
        return cls(spot, strike, time, rate, vol, carry=rate - dividend)

    @classmethod
    def on_future(
        cls, future: float, strike: float, time: float, rate: float, vol: float
    ) -> Inputs:
        """Black's case: an option on a future, with zero carry."""
        return cls(future, strike, time, rate, vol, carry=0.0)

    @property
    def is_degenerate(self) -> bool:
        """True when the terminal value carries no uncertainty.

        Either no time remains or the underlying does not move, in which case
        the standard deviation of terminal log-price is zero and the price is
        the discounted intrinsic on the forward.
        """
        return self.time == 0.0 or self.vol == 0.0 or self.spot == 0.0 or self.strike == 0.0

    @property
    def std_dev(self) -> float:
        """Standard deviation of terminal log-price, ``v sqrt(T)``."""
        return self.vol * math.sqrt(self.time)

    @property
    def discount(self) -> float:
        """``e^{-rT}``, the factor from expiry back to now."""
        return math.exp(-self.rate * self.time)


def forward(inputs: Inputs) -> float:
    """The forward price ``S e^{bT}`` the option is written on."""
    return inputs.spot * math.exp(inputs.b * inputs.time)


def log_moneyness(spot: float, strike: float) -> float:
    """``log(S / K)``, evaluated so that it stays accurate near the money.

    Forming the ratio first is the obvious way and the wrong one. When the spot
    is close to the strike, ``S / K`` rounds to a number near one and the
    absolute rounding error of that division, about 1.1e-16, becomes the whole
    of the answer: at ``S / K = 1 + 1e-12`` the naive form is wrong by 7e-5
    relative, while ``log1p((S - K) / K)`` is wrong by 1.3e-16.

    The difference matters more than the size of the log suggests. In ``d1`` the
    log is divided by ``v sqrt(T)``, so for a short-dated option that small
    denominator amplifies the error — and short-dated near-the-money options are
    the most heavily traded contracts on any book.

    The threshold is on the ratio rather than on the difference so that the
    branch is scale free; away from the money the direct form is already exact
    to a rounding and is marginally cheaper.
    """
    difference = spot - strike
    if abs(difference) < 0.5 * strike:
        return math.log1p(difference / strike)
    return math.log(spot / strike)


def d1_d2(inputs: Inputs) -> tuple[float, float]:
    """The two standardised log-moneyness arguments.

    Raises:
        ValueError: on degenerate inputs, where the pair is not defined. Callers
            that want a price should call :func:`price`, which takes the limit
            instead of asking for these.
    """
    if inputs.is_degenerate:
        raise ValueError(
            "d1 and d2 are undefined when time, volatility, spot or strike is zero; "
            "these cases are handled as limits in price()"
        )
    sd = inputs.std_dev
    drift = (inputs.b + 0.5 * inputs.vol**2) * inputs.time
    d1 = (log_moneyness(inputs.spot, inputs.strike) + drift) / sd
    return d1, d1 - sd


def intrinsic(inputs: Inputs, option: OptionType) -> float:
    """Discounted intrinsic value on the forward.

    This is the price when nothing is uncertain, and it is also the lower
    arbitrage bound on the price when something is.
    """
    return max(inputs.discount * option.sign * (forward(inputs) - inputs.strike), 0.0)


def price(inputs: Inputs, option: OptionType) -> float:
    """The generalised Black-Scholes-Merton price."""
    if inputs.is_degenerate:
        return intrinsic(inputs, option)

    d1, d2 = d1_d2(inputs)
    sign = option.sign
    carry_discount = math.exp((inputs.b - inputs.rate) * inputs.time)
    return sign * (
        inputs.spot * carry_discount * norm_cdf(sign * d1)
        - inputs.strike * inputs.discount * norm_cdf(sign * d2)
    )


def parity_gap(inputs: Inputs) -> float:
    """``C - P - (F - K) e^{-rT}``, which is zero for any consistent price pair.

    Returned rather than asserted so that tests and callers can see the size of
    the residual, which is a direct read on accumulated rounding error.
    """
    call = price(inputs, OptionType.CALL)
    put = price(inputs, OptionType.PUT)
    return call - put - inputs.discount * (forward(inputs) - inputs.strike)
