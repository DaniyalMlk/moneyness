"""Heston's (1993) stochastic volatility model, priced through its transform.

The rest of this package prices under one volatility, or reads one off a
surface that was fitted to quotes. Heston instead lets the variance move:

.. math::

    dS_t = (r - q) S_t\\, dt + \\sqrt{V_t}\\, S_t\\, dW^S_t

    dV_t = \\kappa (\\theta - V_t)\\, dt + \\sigma \\sqrt{V_t}\\, dW^V_t

with the two Brownian motions correlated at ``rho``. The variance mean-reverts
to ``theta`` at speed ``kappa`` and is itself volatile at ``sigma``. That gives
a smile with a shape rather than a level: ``rho`` tilts it, ``sigma`` deepens
it, and ``kappa`` controls how fast both wash out with maturity.

There is no formula for the price, but there is one for the characteristic
function of the log forward, and a European option is an integral against it.
Three things about the implementation are worth stating.

**The characteristic function is written in the branch-stable form.** Heston's
original grouping takes a complex logarithm whose argument circles the origin
as maturity grows; the principal branch then jumps by ``2 pi i``, the
exponential of it jumps, and the price is wrong by a visible amount past a
maturity that depends on the parameters. Albrecher and co-authors' observation
is that the *algebraically identical* grouping obtained by taking the other
square root keeps the logarithm's argument inside the unit disc around 1, where
it cannot wind. Both are implemented, selected by :class:`Branch`, because the
failure is worth being able to show: :func:`branch_discrepancy` measures it.

**Two independent pricing routes are provided.** :func:`lewis_price` evaluates
Lewis's single integral, which is symmetric in the wings and needs no separate
treatment of the two probabilities. :func:`gil_pelaez_price` evaluates Heston's
original pair. They are different integrands reached by different arguments, so
agreement between them is evidence, and the tests require it.

**The Gil-Pelaez integrand is evaluated as an imaginary part, not a real one.**
The textbook writes ``Re(e^{-iuk} phi(u) / (iu))``, which at small ``u`` is a
number of order one obtained by cancelling two of order ``1/u``. The same
quantity is ``Im(e^{-iuk} phi(u)) / u``, where the near-cancellation happens
inside a complex multiplication that resolves it exactly, and the division is
by a small number rather than of two large ones. The rewrite is free and it
buys back the digits the first form loses near the origin.
"""

from __future__ import annotations

import cmath
import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from .bsm import Inputs, OptionType
from .implied import Quote, implied_vol
from .quadrature import semi_infinite_quad

__all__ = [
    "Branch",
    "Contract",
    "Heston",
    "branch_discrepancy",
    "char_func",
    "gil_pelaez_price",
    "lewis_price",
    "price",
    "smile",
]


class Branch(str, Enum):
    """Which grouping of the characteristic function to evaluate.

    The two are equal in exact arithmetic. ``STABLE`` is the default and the
    only one that should be used for pricing; ``TEXTBOOK`` exists so the
    failure can be demonstrated rather than asserted.
    """

    STABLE = "stable"
    TEXTBOOK = "textbook"


@dataclass(frozen=True, slots=True)
class Heston:
    """The five parameters of the variance process.

    Attributes:
        v0: Variance now, in annualised units. Non-negative.
        kappa: Speed of mean reversion. Positive.
        theta: Long-run variance. Non-negative.
        sigma: Volatility of variance. Non-negative; zero is the degenerate
            deterministic-variance case and is handled as a limit.
        rho: Correlation between the price and variance shocks, in [-1, 1].
    """

    v0: float
    kappa: float
    theta: float
    sigma: float
    rho: float

    def __post_init__(self) -> None:
        for name in ("v0", "kappa", "theta", "sigma", "rho"):
            value = getattr(self, name)
            if value != value or math.isinf(value):
                raise ValueError(f"{name} must be finite, got {value}")
        if self.v0 < 0.0:
            raise ValueError(f"v0 must be non-negative, got {self.v0}")
        if self.theta < 0.0:
            raise ValueError(f"theta must be non-negative, got {self.theta}")
        if self.sigma < 0.0:
            raise ValueError(f"sigma must be non-negative, got {self.sigma}")
        if self.kappa <= 0.0:
            raise ValueError(f"kappa must be positive, got {self.kappa}")
        if not (-1.0 <= self.rho <= 1.0):
            raise ValueError(f"rho must be in [-1, 1], got {self.rho}")

    @property
    def feller(self) -> float:
        """``2 kappa theta - sigma^2``.

        Positive means the variance process cannot reach zero. The model is
        well defined either way — the transform does not care — but a negative
        value says the variance spends time at the origin, which matters to any
        simulation of it and to how the wings behave.
        """
        return 2.0 * self.kappa * self.theta - self.sigma * self.sigma

    @property
    def satisfies_feller(self) -> bool:
        """Whether the variance process stays strictly positive."""
        return self.feller > 0.0

    def expected_variance(self, time: float) -> float:
        """``E[V_t]``, which decays from ``v0`` to ``theta``."""
        _check_time(time)
        return self.theta + (self.v0 - self.theta) * math.exp(-self.kappa * time)

    def expected_integrated_variance(self, time: float) -> float:
        """``E[int_0^t V_s ds]``: the total variance the model expects.

        This is the variance a Black-Scholes price would need in order to agree
        with the Heston price when the variance is not actually random, and it
        is what the zero-``sigma`` limit reduces to.
        """
        _check_time(time)
        if time == 0.0:
            return 0.0
        decay = -math.expm1(-self.kappa * time) / self.kappa
        return self.theta * time + (self.v0 - self.theta) * decay

    def equivalent_vol(self, time: float) -> float:
        """The single volatility carrying the model's expected total variance."""
        _check_time(time)
        if time == 0.0:
            return 0.0
        return math.sqrt(self.expected_integrated_variance(time) / time)


@dataclass(frozen=True, slots=True)
class Contract:
    """An option and its market, with the volatility left to the model.

    This is :class:`moneyness.Inputs` minus ``vol``, because under Heston the
    volatility is not an input: it is generated by ``v0``, ``theta``, ``kappa``
    and ``sigma``. The carry convention is the same one the rest of the package
    uses, so an option on a future is ``carry=0`` here exactly as it is there.

    Attributes:
        spot: Price of the underlying now. Non-negative.
        strike: Exercise price. Non-negative.
        time: Year fraction to expiry. Non-negative.
        rate: Continuously compounded discount rate.
        carry: Cost of carry. Defaults to ``rate``.
    """

    spot: float
    strike: float
    time: float
    rate: float
    carry: float | None = None

    def __post_init__(self) -> None:
        if self.spot < 0.0:
            raise ValueError(f"spot must be non-negative, got {self.spot}")
        if self.strike < 0.0:
            raise ValueError(f"strike must be non-negative, got {self.strike}")
        if self.time < 0.0:
            raise ValueError(f"time must be non-negative, got {self.time}")
        for name in ("spot", "strike", "time", "rate"):
            value = getattr(self, name)
            if value != value or math.isinf(value):
                raise ValueError(f"{name} must be finite, got {value}")

    @property
    def b(self) -> float:
        """The cost of carry actually in force."""
        return self.rate if self.carry is None else self.carry

    @property
    def forward(self) -> float:
        """The forward price the option is struck against."""
        return self.spot * math.exp(self.b * self.time)

    @property
    def discount(self) -> float:
        """The discount factor to expiry."""
        return math.exp(-self.rate * self.time)

    def with_vol(self, vol: float) -> Inputs:
        """The same contract as :class:`moneyness.Inputs` at one volatility.

        Used to compare against the lognormal model on identical conventions.
        """
        return Inputs(self.spot, self.strike, self.time, self.rate, vol, carry=self.carry)

    def quote(self, price: float) -> Quote:
        """The same contract as a :class:`moneyness.Quote` at a given price.

        Used to read a Heston price back as an implied volatility.
        """
        return Quote(self.spot, self.strike, self.time, self.rate, price, carry=self.carry)


def _log1p(z: complex) -> complex:
    """``log(1 + z)`` for complex ``z``, accurate when ``z`` is small.

    ``cmath.log(1 + z)`` loses every digit of a small ``z`` to the addition.
    The correction is the usual one: the computed ``u = 1 + z`` has an exact
    ``u - 1``, so scaling ``log(u)`` by ``z / (u - 1)`` recovers the digits the
    addition rounded away. Both branch-stable groupings below need this,
    because their logarithm's argument sits within ``sigma^2`` of one.
    """
    u = 1.0 + z
    if u == 1.0:
        return z
    return z * (cmath.log(u) / (u - 1.0))


def _check_time(time: float) -> None:
    if time != time or math.isinf(time):
        raise ValueError(f"time must be finite, got {time}")
    if time < 0.0:
        raise ValueError(f"time must be non-negative, got {time}")


def char_func(
    model: Heston, u: complex, time: float, branch: Branch = Branch.STABLE
) -> complex:
    """``E[exp(i u X_t)]`` where ``X_t = log(S_t / F_t)``.

    The forward is divided out, so the process is a martingale in levels and
    the function satisfies ``phi(0) = 1`` and ``phi(-i) = 1`` exactly. The
    second of those is the martingale condition and is the sharpest single
    check on the algebra below.

    Args:
        model: The variance process.
        u: Transform argument. Complex, because the pricing integrals evaluate
            it off the real line.
        time: Year fraction.
        branch: Which grouping to use. See :class:`Branch`.

    Returns:
        The characteristic function's value.
    """
    _check_time(time)
    # ``iu + u^2`` is the combination the drift-free log price contributes; at
    # zero volatility of variance the whole transform collapses to
    # ``exp(-(iu + u^2) * total variance / 2)``, the lognormal answer.
    m = 1j * u + u * u
    if time == 0.0 or m == 0.0:
        return 1.0 + 0j
    if model.sigma == 0.0:
        return cmath.exp(-0.5 * m * model.expected_integrated_variance(time))

    kappa, theta, sigma, rho = model.kappa, model.theta, model.sigma, model.rho
    s2 = sigma * sigma
    a = kappa - rho * sigma * 1j * u
    d = cmath.sqrt(a * a + s2 * m)

    if branch is Branch.STABLE:
        # The minus root, but never by subtracting: ``a - d`` cancels to
        # nothing as sigma goes to zero, where the two agree to leading order,
        # and the difference is the whole answer. ``(a - d)(a + d) = -s2 m``
        # gives it without the subtraction, and takes the volatility of
        # variance out of the numerator, where it was going to be divided back
        # out anyway. Without this the transform is noise below sigma of about
        # 1e-4 and the integrand stops being smooth enough to integrate.
        root = -s2 * m / (a + d)
        g = root / (a + d)
        decay = cmath.exp(-d * time)
    else:
        # Heston's original grouping: the plus root, and the reciprocal ``g``,
        # written the way the 1993 paper writes it.
        gap = a - d
        if gap == 0.0:
            raise ValueError(
                "the textbook grouping is degenerate here, because a - d has "
                "cancelled to zero; use Branch.STABLE, which does not subtract"
            )
        root = a + d
        g = root / gap
        decay = cmath.exp(d * time)

    denominator = 1.0 - g * decay
    log_ratio = _log1p(-g * decay) - _log1p(-g)
    c = (kappa * theta / s2) * (root * time - 2.0 * log_ratio)
    dterm = (root / s2) * (1.0 - decay) / denominator
    return cmath.exp(c + dterm * model.v0)


def branch_discrepancy(model: Heston, u: float, time: float) -> float:
    """How far the textbook grouping has drifted from the stable one.

    Returns the modulus of the difference between the two characteristic
    functions at a real argument. In exact arithmetic it is zero. In double
    precision it is of order the rounding error until the textbook form's
    logarithm crosses its branch cut, at which point it jumps to order one and
    stays there.

    Args:
        model: The variance process.
        u: Real transform argument.
        time: Year fraction.

    Returns:
        A non-negative discrepancy.
    """
    stable = char_func(model, u, time, Branch.STABLE)
    textbook = char_func(model, u, time, Branch.TEXTBOOK)
    return abs(stable - textbook)


def _bounds(contract: Contract, option: OptionType) -> float | None:
    """The price when the integral is not needed, or ``None`` when it is."""
    forward = contract.forward
    discount = contract.discount
    if contract.time == 0.0:
        payoff = option.sign * (contract.spot - contract.strike)
        return max(payoff, 0.0)
    if contract.strike == 0.0:
        # A call on a zero strike is the forward; the put is worthless.
        return discount * forward if option is OptionType.CALL else 0.0
    if contract.spot == 0.0:
        return 0.0 if option is OptionType.CALL else discount * contract.strike
    return None


def _panel_width(model: Heston, time: float) -> float:
    """A first panel matched to where the integrand actually decays.

    The transform decays like ``exp(-u^2 V / 2)`` with ``V`` the expected total
    variance, so the natural scale is ``1 / sqrt(V)``. Starting the doubling
    there means a short-dated, low-variance option — whose integrand is broad —
    does not have its tail reached by sixty doublings of a panel of width one.
    """
    total = model.expected_integrated_variance(time)
    if total <= 0.0:
        return 1.0
    return min(max(1.0 / math.sqrt(total), 1e-3), 1e6)


def lewis_price(
    model: Heston,
    contract: Contract,
    option: OptionType = OptionType.CALL,
    tol: float = 1e-12,
) -> float:
    """Price a European option by Lewis's single integral.

    The value of a call is

    .. math::

        e^{-rT}\\left[F - \\frac{\\sqrt{FK}}{\\pi}
        \\int_0^\\infty \\Re\\!\\left(e^{-iuk}\\,\\phi(u - i/2)\\right)
        \\frac{du}{u^2 + 1/4}\\right]

    with ``k = log(K / F)``. The integrand is evaluated half a unit below the
    real axis, which is where the payoff's own transform and the model's share
    a strip of convergence; the ``u^2 + 1/4`` in the denominator is the payoff's
    contribution and is what makes the integrand decay even when the model's
    does not.

    Args:
        model: The variance process.
        contract: The option and its market.
        option: Call or put. The put comes from parity, which is exact.
        tol: Absolute tolerance passed to the quadrature.

    Returns:
        The option's present value.
    """
    bound = _bounds(contract, option)
    if bound is not None:
        return bound

    forward = contract.forward
    strike = contract.strike
    discount = contract.discount
    k = math.log(strike / forward)
    scale = math.sqrt(forward * strike) / math.pi

    def integrand(u: float) -> float:
        value = cmath.exp(-1j * u * k) * char_func(model, u - 0.5j, contract.time)
        return value.real / (u * u + 0.25)

    integral = semi_infinite_quad(
        integrand, tol=tol, initial_width=_panel_width(model, contract.time)
    )
    call = discount * (forward - scale * integral)
    if option is OptionType.CALL:
        return max(call, 0.0)
    return max(call - discount * (forward - strike), 0.0)


def gil_pelaez_price(
    model: Heston,
    contract: Contract,
    option: OptionType = OptionType.CALL,
    tol: float = 1e-12,
) -> float:
    """Price a European option by Heston's original pair of probabilities.

    ``C = e^{-rT}(F P_1 - K P_2)``, where ``P_2`` is the risk-neutral
    probability of finishing above the strike and ``P_1`` is the same
    probability under the measure that uses the share as numeraire. Both come
    from Gil-Pelaez inversion of the characteristic function, ``P_1`` from the
    function shifted by ``-i``.

    This is a different integrand reached by a different argument from
    :func:`lewis_price`, which is the point of keeping both.

    Args:
        model: The variance process.
        contract: The option and its market.
        option: Call or put.
        tol: Absolute tolerance passed to the quadrature.

    Returns:
        The option's present value.
    """
    bound = _bounds(contract, option)
    if bound is not None:
        return bound

    forward = contract.forward
    strike = contract.strike
    discount = contract.discount
    time = contract.time
    k = math.log(strike / forward)
    width = _panel_width(model, time)

    def tail(shift: complex) -> float:
        def integrand(u: float) -> float:
            # Im(w) / u rather than Re(w / (iu)): the same number, without
            # cancelling two quantities of order 1/u against each other.
            w = cmath.exp(-1j * u * k) * char_func(model, u + shift, time)
            return w.imag / u

        return 0.5 + semi_infinite_quad(integrand, tol=tol, initial_width=width) / math.pi

    # phi(-i) is one by construction, so the share-measure transform is just
    # the function shifted, with no normalising divide.
    p1 = tail(-1j)
    p2 = tail(0j)
    call = discount * (forward * p1 - strike * p2)
    if option is OptionType.CALL:
        return max(call, 0.0)
    return max(call - discount * (forward - strike), 0.0)


def price(
    model: Heston,
    contract: Contract,
    option: OptionType = OptionType.CALL,
    tol: float = 1e-12,
) -> float:
    """Price a European option. Lewis's form, which is the faster of the two."""
    return lewis_price(model, contract, option, tol)


@dataclass(frozen=True, slots=True)
class SmilePoint:
    """One strike's price and the lognormal volatility that reproduces it.

    Attributes:
        strike: The strike.
        price: The Heston price of the option named by ``option``.
        implied_vol: The Black-Scholes volatility matching that price, or
            ``None`` where the price sits on an arbitrage bound and no finite
            volatility reaches it.
        option: Which side was priced.
    """

    strike: float
    price: float
    implied_vol: float | None
    option: OptionType


def smile(
    model: Heston,
    contract: Contract,
    strikes: Sequence[float],
    tol: float = 1e-12,
) -> list[SmilePoint]:
    """Price a ladder of strikes and read each price back as a volatility.

    Out-of-the-money options are priced on their own side of the forward —
    puts below it, calls above — because that is where the price carries the
    information and where the implied-volatility inversion is best conditioned.

    Args:
        model: The variance process.
        contract: Supplies the spot, maturity and rates; its strike is ignored.
        strikes: The ladder. Must be non-empty and strictly positive.
        tol: Absolute tolerance passed to the quadrature.

    Returns:
        One :class:`SmilePoint` per strike, in the order given.

    Raises:
        ValueError: If the ladder is empty or holds a non-positive strike.
    """
    if not strikes:
        raise ValueError("strikes must not be empty")
    forward = contract.forward
    points: list[SmilePoint] = []
    for strike in strikes:
        if not (strike > 0.0):
            raise ValueError(f"strikes must be positive, got {strike}")
        option = OptionType.CALL if strike >= forward else OptionType.PUT
        one = Contract(contract.spot, strike, contract.time, contract.rate, contract.carry)
        value = price(model, one, option, tol)
        try:
            vol: float | None = implied_vol(one.quote(value), option)
        except ValueError:
            vol = None
        points.append(SmilePoint(strike, value, vol, option))
    return points
