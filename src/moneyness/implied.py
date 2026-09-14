"""Recovering the volatility implied by a quoted price.

The inversion is done in *total volatility* ``w = sigma sqrt(T)`` on the forward,
not in ``sigma`` on the spot. Three things fall out of that choice.

The discount factor and the carry leave the problem entirely. Writing
``F = S e^{bT}``, every price is ``e^{-rT}`` times a function of ``F``, ``K`` and
``w`` alone, so the solver divides the quote by the discount factor once and
then works on a two-parameter problem. Nothing inside the iteration has to know
which of the four market conventions the quote came from.

The function being inverted is better behaved. In ``w`` the normalised price
rises from the intrinsic value at ``w = 0`` to the forward (for a call) as
``w`` grows without bound, monotonically and with a derivative ``F phi(d1)``
that has no other zeros. A bracket therefore always exists, and the only place
the derivative vanishes is at the ends.

And the maturity stops mattering to conditioning. A one-day option and a
five-year option with the same total volatility are the same problem; dividing
by ``sqrt(T)`` at the very end is where the short-dated case gets its large
number, rather than somewhere in the middle of an iteration.

The solver is a safeguarded Newton iteration: a Newton step when it stays inside
the bracket and makes progress, a bisection step when it does not. Newton alone
is not safe here — vega vanishes in both wings, and a step divided by a vanishing
derivative leaves the bracket entirely — while bisection alone converges in the
linear number of steps nobody wants. The combination keeps Newton's quadratic
rate in the middle and bisection's guarantee everywhere. Brent's method is also
provided, for callers who want a derivative-free path and for the tests to check
the primary solver against a second, independent root finder.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from .bsm import Inputs, OptionType
from .normal import norm_cdf, norm_pdf

__all__ = [
    "Bounds",
    "Method",
    "Quote",
    "Solution",
    "black",
    "black_vega",
    "bounds",
    "implied_vol",
    "solve",
]

# The price residual is driven below this multiple of the forward. It is a few
# times the double-precision resolution of the price itself, so asking for less
# would be asking the iteration to chase rounding.
_PRICE_TOLERANCE = 1e-14

# Total volatility above this is not a market quote. sqrt(T) times an annual
# volatility of 1000% at ten years is about 32, so 64 is far outside anything
# meaningful while still leaving the bracket search room to terminate.
_MAX_TOTAL_VOL = 64.0

_MAX_ITERATIONS = 100


class Method(str, Enum):
    """Which route produced the answer."""

    BOUND = "bound"
    """The quote sat exactly on the lower bound; the volatility is zero."""

    NEWTON = "newton"
    """Safeguarded Newton, taking Newton steps."""

    BRENT = "brent"
    """Brent's method on the bracket."""


@dataclass(frozen=True, slots=True)
class Bounds:
    """The range of prices a volatility can produce, for one option."""

    lower: float
    upper: float

    def contains(self, price: float, scale: float) -> bool:
        slack = scale * _PRICE_TOLERANCE
        return self.lower - slack <= price <= self.upper + slack


@dataclass(frozen=True, slots=True)
class Solution:
    """The recovered volatility, with the evidence that it is right."""

    vol: float
    """Annualised volatility."""

    total_vol: float
    """``sigma sqrt(T)``, the quantity actually solved for."""

    iterations: int
    method: Method

    residual: float
    """Model price at :attr:`vol` minus the quoted price.

    Returned rather than discarded so a caller can tell a converged answer from
    one that ran out of iterations without re-pricing the option.
    """


@dataclass(frozen=True, slots=True)
class Quote:
    """An option quote: every input to the model except the volatility."""

    spot: float
    strike: float
    time: float
    rate: float
    price: float
    carry: float | None = None

    def __post_init__(self) -> None:
        for name in ("spot", "strike", "time", "rate", "price"):
            value = getattr(self, name)
            if value != value or math.isinf(value):
                raise ValueError(f"{name} must be finite, got {value}")
        if self.spot <= 0.0:
            raise ValueError(f"spot must be positive, got {self.spot}")
        if self.strike <= 0.0:
            raise ValueError(f"strike must be positive, got {self.strike}")
        if self.time <= 0.0:
            raise ValueError(
                f"time must be positive, got {self.time}: an expired option has no "
                "implied volatility, since every volatility reproduces its price"
            )
        if self.price < 0.0:
            raise ValueError(f"price must be non-negative, got {self.price}")

    @property
    def b(self) -> float:
        return self.rate if self.carry is None else self.carry

    @property
    def forward(self) -> float:
        return self.spot * math.exp(self.b * self.time)

    @property
    def discount(self) -> float:
        return math.exp(-self.rate * self.time)

    def with_vol(self, vol: float) -> Inputs:
        """The full model inputs, at a given volatility."""
        return Inputs(self.spot, self.strike, self.time, self.rate, vol, carry=self.carry)


def black(forward: float, strike: float, total_vol: float, option: OptionType) -> float:
    """The undiscounted Black price in total-volatility coordinates.

    Equal to ``e^{rT}`` times the Black-Scholes-Merton price at
    ``sigma = total_vol / sqrt(T)``, which the tests assert directly.
    """
    sign = option.sign
    if total_vol <= 0.0:
        return max(sign * (forward - strike), 0.0)
    d1 = math.log(forward / strike) / total_vol + 0.5 * total_vol
    d2 = d1 - total_vol
    return sign * (forward * norm_cdf(sign * d1) - strike * norm_cdf(sign * d2))


def black_vega(forward: float, strike: float, total_vol: float) -> float:
    """``d(black)/d(total_vol)``, which is ``F phi(d1)``.

    Shared by calls and puts, and strictly positive for any finite positive
    total volatility — which is what makes the price monotone and the root
    unique.
    """
    if total_vol <= 0.0:
        return 0.0
    d1 = math.log(forward / strike) / total_vol + 0.5 * total_vol
    return forward * norm_pdf(d1)


def bounds(quote: Quote, option: OptionType) -> Bounds:
    """The prices attainable by some non-negative volatility.

    The lower bound is the discounted intrinsic on the forward, reached at zero
    volatility. The upper bound is approached but never attained as volatility
    grows without bound: a call converges to the discounted forward, a put to
    the discounted strike. A quote outside this range is not merely hard to
    invert, it is inconsistent with the model at any volatility.
    """
    discount = quote.discount
    forward = quote.forward
    lower = max(discount * option.sign * (forward - quote.strike), 0.0)
    upper = discount * (forward if option is OptionType.CALL else quote.strike)
    return Bounds(lower, upper)


def _bracket(quote: Quote, option: OptionType, target: float) -> tuple[float, float]:
    """A pair of total volatilities whose prices straddle the target.

    Grows the upper end geometrically. The lower end stays at zero because the
    price there is the intrinsic value, which the caller has already checked the
    target to be above.
    """
    forward, strike = quote.forward, quote.strike
    high = 1.0
    while high < _MAX_TOTAL_VOL:
        if black(forward, strike, high, option) >= target:
            return 0.0, high
        high *= 2.0
    if black(forward, strike, _MAX_TOTAL_VOL, option) < target:
        raise ValueError(
            f"price {quote.price} is not reachable below a total volatility of "
            f"{_MAX_TOTAL_VOL}; the quote is within rounding of the upper "
            "no-arbitrage bound"
        )
    return 0.0, _MAX_TOTAL_VOL


def _initial_guess(forward: float, strike: float, target: float) -> float:
    """A starting total volatility.

    At the money the Black price is very nearly ``F w / sqrt(2 pi)``, which
    inverts to ``w = target sqrt(2 pi) / F`` and is accurate to about a percent
    for the volatilities that occur in practice. Away from the money that
    underestimates, so the guess is floored at ``sqrt(2 |log(F / K)|)`` — the
    total volatility below which ``d1`` and ``d2`` have the same sign and the
    option has essentially no time value at all.

    It does not need to be better than this. The safeguard turns a poor guess
    into a few extra bisection steps rather than a failure, so effort spent on
    a sharper approximation buys iterations, not robustness.
    """
    at_the_money = target * math.sqrt(2.0 * math.pi) / forward
    log_moneyness = math.log(forward / strike)
    floor = math.sqrt(2.0 * abs(log_moneyness))
    return min(max(at_the_money, floor, 1e-8), _MAX_TOTAL_VOL)


def solve(quote: Quote, option: OptionType, method: Method = Method.NEWTON) -> Solution:
    """Recover the volatility implied by :attr:`Quote.price`.

    Args:
        quote: The option and its quoted price.
        option: Call or put.
        method: :attr:`Method.NEWTON` for safeguarded Newton, the default, or
            :attr:`Method.BRENT` for the derivative-free alternative.

    Raises:
        ValueError: if the quote lies outside the no-arbitrage bounds, or is so
            close to the upper bound that no finite volatility reaches it.
    """
    limits = bounds(quote, option)
    scale = max(quote.forward, quote.strike)
    if not limits.contains(quote.price, scale):
        raise ValueError(
            f"price {quote.price} is outside the no-arbitrage range "
            f"[{limits.lower}, {limits.upper}] for this {option.value}"
        )

    # At or below the lower bound the answer is zero volatility exactly. The
    # comparison is made with the same slack used to validate the quote, so a
    # price a rounding below intrinsic is treated as being on the bound rather
    # than rejected by one branch and accepted by the other.
    if quote.price <= limits.lower + scale * _PRICE_TOLERANCE:
        return Solution(0.0, 0.0, 0, Method.BOUND, limits.lower - quote.price)

    # The upper bound is a supremum, not a maximum: the price approaches it as
    # volatility grows without bound and never attains it. A quote within
    # rounding of it is therefore not invertible, and saying so is more useful
    # than returning whatever the largest volatility searched happened to be.
    if quote.price >= limits.upper - scale * _PRICE_TOLERANCE:
        raise ValueError(
            f"price {quote.price} is not reachable: it sits on the upper "
            f"no-arbitrage bound {limits.upper}, which no finite volatility attains"
        )

    target = quote.price / quote.discount
    low, high = _bracket(quote, option, target)
    if method is Method.BRENT:
        total_vol, iterations = _brent(quote, option, target, low, high)
    else:
        total_vol, iterations = _safeguarded_newton(quote, option, target, low, high)

    vol = total_vol / math.sqrt(quote.time)
    residual = quote.discount * black(quote.forward, quote.strike, total_vol, option) - quote.price
    return Solution(vol, total_vol, iterations, method, residual)


def implied_vol(quote: Quote, option: OptionType) -> float:
    """The implied volatility, for callers who want only the number."""
    return solve(quote, option).vol


def _safeguarded_newton(
    quote: Quote, option: OptionType, target: float, low: float, high: float
) -> tuple[float, int]:
    """Newton where it behaves, bisection where it does not.

    The bracket is maintained on every iteration from the sign of the residual,
    so a Newton step that would leave it is rejected in favour of the midpoint.
    That is what keeps the wings safe: vega there is small enough that a Newton
    step can be enormous, and without the bracket the iteration would wander
    into a region where the price underflows and never return.

    Convergence is judged on the *step in total volatility*, not on the price
    residual, and the difference is not cosmetic. A deep in-the-money option has
    almost no time value, so the price is nearly flat in volatility: at a strike
    of 20 against a spot of 100 the derivative is 2.8e-7, and a residual
    threshold of 1.1e-12 — already at the resolution of the price — is reached
    while the volatility is still wrong in its sixth digit. Iterating until the
    argument stops moving instead converges to the precision the quote can
    actually support, and costs one or two extra evaluations where the problem
    is well conditioned.
    """
    forward, strike = quote.forward, quote.strike
    step_relative = 4.0 * 2.220446049250313e-16
    step_absolute = 1e-15
    guess = _initial_guess(forward, strike, target)
    total_vol = min(max(guess, low), high)

    for iteration in range(1, _MAX_ITERATIONS + 1):
        residual = black(forward, strike, total_vol, option) - target
        if residual == 0.0:
            return total_vol, iteration

        # The price is increasing in total volatility, so the sign of the
        # residual says which side of the root we are on.
        if residual > 0.0:
            high = total_vol
        else:
            low = total_vol

        derivative = black_vega(forward, strike, total_vol)
        candidate = total_vol - residual / derivative if derivative > 0.0 else math.inf
        if not low < candidate < high:
            candidate = 0.5 * (low + high)

        step = candidate - total_vol
        total_vol = candidate
        if abs(step) <= step_absolute + step_relative * abs(total_vol):
            return total_vol, iteration
        if high - low <= step_absolute + step_relative * abs(total_vol):
            return total_vol, iteration

    raise ValueError(
        f"implied volatility did not converge in {_MAX_ITERATIONS} iterations "
        f"for price {quote.price}"
    )


def _brent(
    quote: Quote, option: OptionType, target: float, low: float, high: float
) -> tuple[float, int]:
    """Brent's method: inverse quadratic interpolation with a bisection guard.

    Included as an independent check on the Newton path rather than because the
    library needs two solvers. It uses no derivative, so if the two agree on a
    grid of quotes then an error in ``black_vega`` cannot be hiding inside a
    converged answer.

    Termination is on the *bracket width* rather than on the residual, which is
    the detail that matters here. A deep in-the-money option has almost no time
    value, so the price is nearly flat in total volatility near the root and a
    residual test either stops immediately at a badly wrong point or never
    reaches its threshold at all. The width test converges on the argument,
    where the problem is well posed, and it always terminates.
    """
    forward, strike = quote.forward, quote.strike
    x_tolerance = 1e-15
    x_relative = 4.0 * 2.220446049250313e-16  # four times double epsilon

    def f(w: float) -> float:
        return black(forward, strike, w, option) - target

    previous, current = low, high
    f_previous, f_current = f(previous), f(current)
    if f_previous * f_current > 0.0:
        raise ValueError("the bracket does not straddle the root")
    if f_previous == 0.0:
        return previous, 0
    if f_current == 0.0:
        return current, 0

    blocked, f_blocked = previous, f_previous
    step, previous_step = current - previous, current - previous

    for iteration in range(1, _MAX_ITERATIONS + 1):
        if f_previous * f_current < 0.0:
            blocked, f_blocked = previous, f_previous
            step = previous_step = current - previous
        if abs(f_blocked) < abs(f_current):
            previous, current, blocked = current, blocked, current
            f_previous, f_current, f_blocked = f_current, f_blocked, f_current

        delta = 0.5 * (x_tolerance + x_relative * abs(current))
        bisection_step = 0.5 * (blocked - current)
        if f_current == 0.0 or abs(bisection_step) < delta:
            return current, iteration

        if abs(previous_step) > delta and abs(f_current) < abs(f_previous):
            if previous == blocked:
                trial = -f_current * (current - previous) / (f_current - f_previous)
            else:
                slope_previous = (f_previous - f_current) / (previous - current)
                slope_blocked = (f_blocked - f_current) / (blocked - current)
                trial = (
                    -f_current
                    * (f_blocked * slope_blocked - f_previous * slope_previous)
                    / (slope_blocked * slope_previous * (f_blocked - f_previous))
                )
            if 2.0 * abs(trial) < min(abs(previous_step), 3.0 * abs(bisection_step) - delta):
                previous_step, step = step, trial
            else:
                previous_step = step = bisection_step
        else:
            previous_step = step = bisection_step

        previous, f_previous = current, f_current
        current += step if abs(step) > delta else (delta if bisection_step > 0.0 else -delta)
        f_current = f(current)

    raise ValueError(f"Brent did not converge in {_MAX_ITERATIONS} iterations")
