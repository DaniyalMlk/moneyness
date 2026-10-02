"""Variance and volatility swaps: the one number a smile implies without a model.

Every other pricer in this package takes a view and returns a price. A variance
swap goes the other way. Its fair strike is the expectation of realised
variance, and for a continuous diffusion with no jumps that expectation is a
*static portfolio of the options already quoted* — no volatility input, no
calibration, no model. Writing

    g(S) = -log(S / B) + S / B - 1

for any reference level ``B``, the identity

    g(S) = int_0^B (K - S)^+ / K**2 dK + int_B^inf (S - K)^+ / K**2 dK

holds pointwise, so taking expectations and using ``E[S_T] = F`` gives

    E[integral of variance] = 2 (int of out-of-the-money price over K**2) / D

with ``D`` the discount factor, provided the split ``B`` is the forward. That is
:func:`fair_variance`, and the whole content of this module is how far that
identity survives contact with a finite ladder of strikes.

**Two exact targets, and it hits both.** Under a flat Black-Scholes smile the
fair variance is ``sigma**2`` with nothing approximate about it, and the
replication returns it to within two or three units in the last place across
volatilities from 10% to 80% and maturities from three months to five years.
Under Heston it is ``E[int V]/T``, which :meth:`moneyness.Heston.
expected_integrated_variance` already supplies in closed form, so a
*model-free* formula is being checked against a *model's* own answer. It
agrees to 1e-12 relative once the strike range is wide enough, and the
qualification is the interesting part.

**The strike range needed is not set by the diffusive scale.** The natural
rule of thumb is a few standard deviations of log-moneyness. Under flat
Black-Scholes that is right: at one year and a 20% volatility, a range of plus
and minus five standard deviations is already good to 4.4e-08 relative and six
to 1.2e-10. Under Heston at the same equivalent volatility it is not close. At
a vol-of-vol of 0.3 ten standard deviations leaves 5.7e-07; at 0.5, 5.3e-05;
at 0.8, **1.3e-03**, and reaching 1e-09 there takes *thirty* standard
deviations, which is log-moneyness of six — strikes from a quarter of a per
cent of the forward to four hundred times it. The log-contract weight is
``1/K**2``, which is exactly the weight that keeps a fat tail relevant, and
stochastic volatility supplies one.

**So the truncation is reported, never assumed.** :func:`truncation_error`
gives it in closed form for a lognormal terminal law. Outside the quoted range
the replicating portfolio no longer pays ``g``: it pays the *tangent* to ``g``
at the boundary, because beyond the last strike there is nothing left to buy.
The error is therefore the expected excess of ``g`` over that tangent beyond the
boundary, which is an elementary lognormal expectation. Measured against the
quadrature it is right to 1e-14 relative at one standard deviation and 2e-11 at
four; by six standard deviations the error itself is 4.6e-12 and the comparison
is meaningless rather than wrong, which is a reminder that a prediction can
only be checked while the thing predicted is above the rounding.

**The discrete sum needs a centring term, and it is not the one usually
quoted.** A desk sums over listed strikes and splits puts from calls at the
largest listed strike at or below the forward, ``K0``, not at the forward
itself. Splitting at ``B = K0`` replicates ``g`` centred on ``K0``, and the
identity above then returns, exactly,

    sigma**2 + (2/T) (F/K0 - 1 - log(F/K0))

so the sum overstates the fair variance and the overstatement is known in
closed form. The market convention subtracts ``(F/K0 - 1)**2 / T``, which is the
leading term of that expression and no more: it overstates the true correction
by 0.67% at a one per cent gap, 6.6% at ten per cent and 32% at a half.
:class:`Centring` offers both, and the measurement is that *either* restores
second-order convergence in the strike spacing — halving ratios of 4.00 against
an uncorrected sum whose ratios wander between 1.46 and 10.7, because the
uncorrected error is dominated by where the forward happens to fall relative to
the lattice rather than by the spacing. The exact form earns its place at a
coarse lattice; by the time the spacing is fine the gap to the forward is small
and the two corrections agree to the digits that matter.

**A volatility swap is not the square root of a variance swap, and the usual
correction overshoots.** Realised volatility is the square root of realised
variance and the square root is concave, so the fair volatility sits below
``sqrt(K_var)``. The second-order expansion is

    K_vol ~ sqrt(mu) (1 - Var(RV) / (8 mu**2))

which needs the variance of integrated variance.
:func:`integrated_variance_variance` supplies it in closed form for the
square-root process, derived by integrating the covariance
``Cov(V_s, V_t) = e^{-kappa(t-s)} Var(V_s)`` twice and checked against an
independent double quadrature of that same covariance to 1e-14 relative. The
expansion then *overstates* the discount: at one year the predicted gap is
0.0195 in volatility against a simulated 0.0153 at a vol-of-vol of 0.5 — a
ratio of 1.27 — and 0.0343 against 0.0245 at a vol-of-vol of 0.8, a ratio of
1.40. Those are 13 and 22 standard errors of the simulation, so the
overstatement is real and not sampling noise. The direction of the adjustment
is right and its size is not, which is worth knowing before quoting one.

**Why the range is an argument rather than an automatic walk.** The obvious
implementation integrates each wing to infinity with
:func:`moneyness.semi_infinite_quad`. It does not work here, and the reason is
the price function rather than the quadrature. A transform price has an
*absolute* accuracy floor: asking :func:`moneyness.heston.price` for 1e-13
returns 1.48e-12 at log-moneyness 2 and 3.02e-12 at log-moneyness 5, which is
not even monotone, and 1.381e-03 at log-moneyness 40 where the true price is
zero to hundreds of digits. The cost also grows, from 2.3ms at the money to
34ms there. Dividing by the strike makes all of it harmless to the integral —
1.4e-03 over 2.4e+17 is 6e-21 — but an unbounded walk looking for a panel that
contributes nothing is looking for a property the integrand does not have, and
it did not terminate in minutes. A finite width with its truncation reported is
both cheaper and more honest.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum

from .heston import Heston
from .normal import norm_cdf, norm_pdf
from .quadrature import adaptive_quad

__all__ = [
    "BadStrip",
    "Centring",
    "Replication",
    "Strip",
    "VarianceSwap",
    "VolatilitySwap",
    "fair_variance",
    "fair_variance_from_strip",
    "integrated_variance_variance",
    "truncation_error",
    "volatility_swap_strike",
]

#: Share of each wing, measured from its outer edge, whose contribution is
#: reported as the truncation diagnostic. A tenth is wide enough to carry a
#: signal and narrow enough that a well-chosen range leaves it negligible.
_EDGE_SHARE = 0.1


class BadStrip(ValueError):
    """A ladder of strikes and prices that does not describe a market."""


def _positive(name: str, value: float) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise BadStrip(f"{name} must be a positive finite number, got {value!r}")
    return float(value)


@dataclass(frozen=True, slots=True)
class Replication:
    """The continuous log-contract replication and what it left out.

    Attributes:
        fair_variance: Annualised fair variance of a variance swap.
        lower: Contribution of the put wing to the undiscounted integral.
        upper: Contribution of the call wing.
        width: Half-width of the log-moneyness range integrated, so the range
            of strikes is ``F e**-width`` to ``F e**width``.
        edge: Share of the whole integral contributed by the outermost tenth
            of the two wings together. Small means the range was wide enough
            for the integrand; it says nothing about what lies beyond it, for
            which see :func:`truncation_error`.
    """

    fair_variance: float
    lower: float
    upper: float
    width: float
    edge: float

    @property
    def fair_volatility(self) -> float:
        """``sqrt`` of the fair variance: the variance swap quoted in vol terms.

        Not the fair strike of a volatility swap, which is strictly lower; see
        :func:`volatility_swap_strike`.
        """
        return math.sqrt(self.fair_variance)


def fair_variance(
    otm: Callable[[float], float],
    forward: float,
    time: float,
    discount: float,
    *,
    width: float = 3.0,
    tol: float = 1e-13,
) -> Replication:
    """Fair variance by static replication of the log contract.

    The integral is taken in log-moneyness ``k = log(K/F)``, which turns
    ``O(K)/K**2 dK`` into ``O(F e**k)/(F e**k) dk`` and makes the two wings the
    same integrand with opposite signs on ``k``.

    Args:
        otm: Present value of the out-of-the-money option at a strike: a put
            below ``forward``, a call above it. Called at strikes spanning
            ``forward * exp(-width)`` to ``forward * exp(width)``.
        forward: The forward the swap is struck against. Positive.
        time: Year fraction to expiry. Positive.
        discount: Discount factor to expiry. Positive.
        width: Half-width of the log-moneyness range. Positive. The default of
            3.0 covers strikes from a twentieth of the forward to twenty times
            it, which is ample for a lognormal smile at ordinary volatilities
            and is *not* ample under stochastic volatility; the module
            docstring has the measurements.
        tol: Absolute tolerance on each wing's integral.

    Returns:
        A :class:`Replication`.

    Raises:
        BadStrip: If an argument is not positive and finite, or ``otm`` returns
            a negative price.
    """
    forward = _positive("forward", forward)
    time = _positive("time", time)
    discount = _positive("discount", discount)
    width = _positive("width", width)
    tol = _positive("tol", tol)

    def wing(sign: float) -> Callable[[float], float]:
        def integrand(x: float) -> float:
            strike = forward * math.exp(sign * x)
            value = otm(strike)
            if not math.isfinite(value) or value < 0.0:
                raise BadStrip(
                    f"the out-of-the-money price at strike {strike!r} must be a "
                    f"non-negative finite number, got {value!r}"
                )
            return value / strike

        return integrand

    edge_from = width * (1.0 - _EDGE_SHARE)
    total = 0.0
    outer = 0.0
    wings: list[float] = []
    for sign in (-1.0, 1.0):
        integrand = wing(sign)
        inner = adaptive_quad(integrand, 0.0, edge_from, tol)
        rim = adaptive_quad(integrand, edge_from, width, tol)
        wings.append(inner + rim)
        total += inner + rim
        outer += rim
    edge = abs(outer) / abs(total) if total != 0.0 else 0.0
    return Replication(
        fair_variance=2.0 / time * total / discount,
        lower=wings[0],
        upper=wings[1],
        width=width,
        edge=edge,
    )


def truncation_error(
    forward: float,
    time: float,
    vol: float,
    lower_strike: float,
    upper_strike: float,
) -> float:
    """What a finite strike range costs the fair variance, for a lognormal law.

    Beyond the last available strike the replicating portfolio stops tracking
    ``g(S) = -log(S/F) + S/F - 1`` and pays its tangent at the boundary
    instead, because there is nothing further out to buy. The shortfall is the
    expected excess of ``g`` over that tangent beyond each boundary, and ``g``
    is convex, so the result is always a shortfall.

    Args:
        forward: The forward. Positive.
        time: Year fraction to expiry. Positive.
        vol: Lognormal volatility of the terminal law. Positive.
        lower_strike: Lowest strike quoted. Positive and below ``forward``.
        upper_strike: Highest strike quoted. Above ``forward``.

    Returns:
        The signed error, ``truncated fair variance - true fair variance``,
        which is negative whenever the range is finite.

    Raises:
        BadStrip: If an argument is not positive and finite, or the strikes do
            not bracket the forward.
    """
    forward = _positive("forward", forward)
    time = _positive("time", time)
    vol = _positive("vol", vol)
    lower_strike = _positive("lower_strike", lower_strike)
    upper_strike = _positive("upper_strike", upper_strike)
    if not lower_strike < forward < upper_strike:
        raise BadStrip(
            f"the strike range must bracket the forward, got "
            f"[{lower_strike!r}, {upper_strike!r}] around {forward!r}"
        )

    sd = vol * math.sqrt(time)
    excess = 0.0
    for upper, boundary in ((True, upper_strike), (False, lower_strike)):
        z = (math.log(boundary / forward) + 0.5 * sd * sd) / sd
        if upper:
            prob = norm_cdf(-z)
            e_log = sd * norm_pdf(z) - 0.5 * sd * sd * prob
            e_spot = forward * norm_cdf(sd - z)
        else:
            prob = norm_cdf(z)
            e_log = -sd * norm_pdf(z) - 0.5 * sd * sd * prob
            e_spot = forward * norm_cdf(z - sd)
        # E[g(S) 1{beyond}], with g expressed through log(S/F) and S/F.
        e_g = -e_log + e_spot / forward - prob
        g_boundary = -math.log(boundary / forward) + boundary / forward - 1.0
        slope = 1.0 / forward - 1.0 / boundary
        excess += e_g - g_boundary * prob - slope * (e_spot - boundary * prob)
    return -2.0 / time * excess


class Centring(str, Enum):
    """Which correction to apply for a split that is not at the forward."""

    NONE = "none"
    """No correction. The sum then overstates the fair variance."""

    QUADRATIC = "quadratic"
    """``(F/K0 - 1)**2 / T``: the market convention, the leading term only."""

    EXACT = "exact"
    """``2 (F/K0 - 1 - log(F/K0)) / T``: the whole of the centring term."""


@dataclass(frozen=True, slots=True)
class Strip:
    """A ladder of listed strikes and the out-of-the-money prices against them.

    One price per strike: a put below the reference strike, a call above it.
    At the reference strike itself the market convention is the *average* of
    the put and the call, because both are at the money there and neither is
    the out-of-the-money one; the caller supplies that average, since only the
    caller knows both quotes.

    Attributes:
        strikes: Listed strikes, strictly increasing and positive. Must
            bracket the forward.
        prices: Present value of the out-of-the-money option at each strike.
            Non-negative, same length as ``strikes``.
        forward: The forward the swap is struck against.
        time: Year fraction to expiry.
        discount: Discount factor to expiry.
    """

    strikes: tuple[float, ...]
    prices: tuple[float, ...]
    forward: float
    time: float
    discount: float

    def __post_init__(self) -> None:
        if len(self.strikes) != len(self.prices):
            raise BadStrip(
                f"got {len(self.strikes)} strikes and {len(self.prices)} prices, "
                "which must match"
            )
        if len(self.strikes) < 3:
            raise BadStrip(
                f"a strip needs at least three strikes to have an interior, got "
                f"{len(self.strikes)}"
            )
        _positive("forward", self.forward)
        _positive("time", self.time)
        _positive("discount", self.discount)
        previous = 0.0
        for strike in self.strikes:
            _positive("strike", strike)
            if strike <= previous:
                raise BadStrip(
                    f"strikes must be strictly increasing, got {strike!r} "
                    f"after {previous!r}"
                )
            previous = strike
        for price in self.prices:
            if not math.isfinite(price) or price < 0.0:
                raise BadStrip(f"prices must be non-negative and finite, got {price!r}")
        if not self.strikes[0] < self.forward < self.strikes[-1]:
            raise BadStrip(
                f"the strikes must bracket the forward, got "
                f"[{self.strikes[0]!r}, {self.strikes[-1]!r}] around {self.forward!r}"
            )

    @classmethod
    def from_quotes(
        cls,
        quotes: Iterable[tuple[float, float]],
        forward: float,
        time: float,
        discount: float,
    ) -> Strip:
        """Build a strip from ``(strike, price)`` pairs, sorted by strike."""
        pairs = sorted(quotes)
        return cls(
            tuple(strike for strike, _ in pairs),
            tuple(price for _, price in pairs),
            forward,
            time,
            discount,
        )

    @property
    def reference(self) -> float:
        """``K0``: the largest listed strike at or below the forward."""
        return max(strike for strike in self.strikes if strike <= self.forward)

    @property
    def widths(self) -> tuple[float, ...]:
        """Half the gap to either neighbour; one-sided at the two ends."""
        strikes = self.strikes
        last = len(strikes) - 1
        return tuple(
            strikes[1] - strikes[0]
            if i == 0
            else strikes[last] - strikes[last - 1]
            if i == last
            else 0.5 * (strikes[i + 1] - strikes[i - 1])
            for i in range(len(strikes))
        )

    @property
    def log_range(self) -> tuple[float, float]:
        """Log-moneyness of the lowest and highest listed strike."""
        return (
            math.log(self.strikes[0] / self.forward),
            math.log(self.strikes[-1] / self.forward),
        )


@dataclass(frozen=True, slots=True)
class VarianceSwap:
    """Fair variance from a listed strip, and the centring it had to undo.

    Attributes:
        fair_variance: Annualised fair variance, after the correction.
        raw: The uncorrected sum, which overstates the fair variance whenever
            the forward does not sit on a listed strike.
        correction: What was subtracted from ``raw``.
        centring: Which correction was used.
        reference: The split strike ``K0``.
        gap: ``F/K0 - 1``, the relative distance from the split to the forward
            and the only thing the correction depends on.
        strikes: How many strikes the sum ran over.
    """

    fair_variance: float
    raw: float
    correction: float
    centring: Centring
    reference: float
    gap: float
    strikes: int

    @property
    def fair_volatility(self) -> float:
        """``sqrt`` of the fair variance. Not a volatility swap strike."""
        return math.sqrt(self.fair_variance)


def fair_variance_from_strip(
    strip: Strip, *, centring: Centring = Centring.EXACT
) -> VarianceSwap:
    """Fair variance from a listed strip, the way a desk computes it.

    The sum is ``(2/T) sum dK_i / K_i**2 * Q_i / D``, with ``Q_i`` the
    out-of-the-money price and ``dK_i`` half the gap to either neighbour. It
    splits puts from calls at ``K0``, the largest listed strike at or below the
    forward, because that is where the quotes change over, and splitting there
    rather than at the forward replicates the log contract centred on ``K0``.
    That costs exactly ``(2/T)(F/K0 - 1 - log(F/K0))``, which is what
    :attr:`Centring.EXACT` removes; :attr:`Centring.QUADRATIC` removes the
    market convention's leading term instead.

    Args:
        strip: The ladder of strikes and out-of-the-money prices.
        centring: Which correction to subtract.

    Returns:
        A :class:`VarianceSwap`.
    """
    reference = strip.reference
    total = 0.0
    for strike, price, width in zip(strip.strikes, strip.prices, strip.widths, strict=True):
        total += width / (strike * strike) * price
    raw = 2.0 / strip.time * total / strip.discount
    ratio = strip.forward / reference
    gap = ratio - 1.0
    if centring is Centring.QUADRATIC:
        correction = gap * gap / strip.time
    elif centring is Centring.EXACT:
        correction = 2.0 / strip.time * (gap - math.log(ratio))
    else:
        correction = 0.0
    return VarianceSwap(
        fair_variance=raw - correction,
        raw=raw,
        correction=correction,
        centring=centring,
        reference=reference,
        gap=gap,
        strikes=len(strip.strikes),
    )


def integrated_variance_variance(model: Heston, time: float) -> float:
    """``Var(int_0^T V_s ds)`` for the square-root variance process.

    The conditional mean of the process is affine, so for ``s < t``

        Cov(V_s, V_t) = e**(-kappa (t - s)) Var(V_s)

    and the variance of the integral is twice the double integral of that over
    the triangle ``0 < s < t < T``. Both ``Var(V_s)`` and the exponential are
    elementary, so the double integral is too:

        Var(I) = A c_A(T) + B c_B(T),   A = sigma**2 theta / (2 kappa),
                                        B = sigma**2 v0 / kappa

    with ``c_A`` and ``c_B`` as written below. At ``v0 = theta`` the two
    collapse to the stationary form
    ``sigma**2 theta (2 kappa T - 3 + 4 e**(-kappa T) - e**(-2 kappa T)) / (2 kappa**3)``.

    Args:
        model: The variance process.
        time: Horizon. Non-negative.

    Returns:
        The variance of integrated variance over ``[0, time]``.

    Raises:
        ValueError: If ``time`` is negative or not finite.
    """
    if not math.isfinite(time) or time < 0.0:
        raise ValueError(f"time must be a non-negative finite number, got {time!r}")
    if time == 0.0:
        return 0.0
    kappa = model.kappa
    a = model.sigma * model.sigma * model.theta / (2.0 * kappa)
    b = model.sigma * model.sigma * model.v0 / kappa
    decay = math.exp(-kappa * time)
    decay2 = decay * decay
    c_a = (
        2.0 * time / kappa
        + 4.0 * time * decay / kappa
        - 5.0 / kappa**2
        + 4.0 * decay / kappa**2
        + decay2 / kappa**2
    )
    c_b = -2.0 * time * decay / kappa + 1.0 / kappa**2 - decay2 / kappa**2
    return a * c_a + b * c_b


@dataclass(frozen=True, slots=True)
class VolatilitySwap:
    """A volatility swap strike and the convexity that separates it from a variance one.

    Attributes:
        strike: The second-order fair volatility.
        variance_strike: The fair variance over the same horizon.
        convexity: ``sqrt(variance_strike) - strike``, the discount the
            concavity of the square root implies. Non-negative.
        dispersion: ``Var(RV) / mean(RV)**2``, the squared coefficient of
            variation of realised variance, which is the only thing the
            second-order term depends on.
    """

    strike: float
    variance_strike: float
    convexity: float
    dispersion: float


def volatility_swap_strike(model: Heston, time: float) -> VolatilitySwap:
    """Second-order fair volatility under the square-root variance process.

    Realised volatility is the square root of realised variance, and the
    square root is concave, so ``E[sqrt(RV)] < sqrt(E[RV])`` strictly whenever
    ``RV`` is not degenerate. Expanding about the mean,

        E[sqrt(RV)] = sqrt(mu) (1 - c/8 + ...),   c = Var(RV) / mu**2

    with ``mu = E[int V]/T`` exact in the model and ``Var(RV)`` from
    :func:`integrated_variance_variance`.

    **The expansion overstates the discount, measurably.** Against 60,000
    simulated variance paths at one year, the predicted gap is 0.0195 in
    volatility against 0.0153 realised at a vol-of-vol of 0.5, 0.0343 against
    0.0245 at 0.8, and 0.0142 against 0.0114 at 0.3 — ratios of 1.27, 1.40 and
    1.25, and 13, 22 and 12 standard errors of the simulation respectively. So
    this is the right sign and about a quarter to a half too large, and the
    truncated series is the reason: the next term in the expansion is positive.

    Args:
        model: The variance process.
        time: Horizon. Positive.

    Returns:
        A :class:`VolatilitySwap`.

    Raises:
        ValueError: If ``time`` is not positive and finite.
    """
    if not math.isfinite(time) or time <= 0.0:
        raise ValueError(f"time must be a positive finite number, got {time!r}")
    mean = model.expected_integrated_variance(time) / time
    if mean <= 0.0:
        return VolatilitySwap(0.0, 0.0, 0.0, 0.0)
    dispersion = integrated_variance_variance(model, time) / (time * time) / (mean * mean)
    root = math.sqrt(mean)
    strike = root * (1.0 - dispersion / 8.0)
    return VolatilitySwap(
        strike=strike,
        variance_strike=mean,
        convexity=root - strike,
        dispersion=dispersion,
    )
