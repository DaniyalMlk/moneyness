"""Options on the difference between two correlated assets.

A crack spread, a crush spread, an equity outperformance option and a
location spread on a commodity are all written on ``S1 - S2``, and the
difference of two lognormals is not lognormal. There is no closed form, so
this module is the same shape of problem as :mod:`moneyness.asian`: an
approximation everybody quotes, and the question of how to know when it is
wrong.

Three routes, each of which checks the others.

**Exact, up to quadrature.** Condition on the second asset. Given the normal
variate ``z`` that drives ``S2``, the first asset is still lognormal — with
its forward shifted by ``rho`` and its variance reduced to ``v1^2 (1 - rho^2)``
— so the conditional payoff is a Black-76 call on ``S1`` struck at
``S2(z) + K``. The price is then a one-dimensional integral of that Black
price against a standard normal density, and :func:`spread_price` evaluates it
on Gauss-Legendre panels. Two things make it a price rather than a plausible
number:

* The integrand has structure, and the structure goes on panel edges. The
  conditional option passes through the money at the roots of
  ``F1(z) - S2(z) - K``, which is where the integrand is least polynomial and
  where — as ``|rho|`` approaches one, or either volatility approaches zero —
  it acquires an actual kink. Those roots are located by
  :func:`critical_states` and inserted as edges.
* The ceiling is derived, not chosen. A call pays at most ``S1``, so the mass
  outside ``[-w, w]`` is at most ``F1 [N(rho v1 sqrt(T) - w) + N(-rho v1
  sqrt(T) - w)]``, which :func:`truncation_bound` returns in closed form. The
  width is solved from the tolerance asked for, and a tolerance no width
  attains is refused rather than silently missed.

**Margrabe's exchange option**, :func:`exchange`, which is exact. At a zero
strike the spread option *is* an exchange option, so Margrabe is what the
quadrature has to reproduce, and it contains nothing of this module's. Two more
reductions do the same job elsewhere: with no volatility on the second asset
the spread option is an ordinary Black call struck at ``F2 + K``, and with
none on the *first* it is a Black put on the second struck at ``F1 - K``. The
quadrature reproduces all three to between 4e-14 and 1e-13 relative.

**Kirk's approximation**, :func:`kirk`, which is what desks use. It replaces
``S2 + K`` by a lognormal with the same mean and a volatility scaled by
``F2 / (F2 + K)``, and prices the result with Black-76. At ``K = 0`` that
scaling is one and Kirk collapses to Margrabe exactly, which is a transcription
check and nothing more. Away from zero its error is measured below.

**The bounds are what make an approximation reportable**, and the two sides
come from opposite directions.

*Lower*, by sub-replication. For any half-space ``A`` of the two-dimensional
Gaussian, ``max(S1 - S2 - K, 0) >= (S1 - S2 - K) 1_A`` path by path, and the
expectation of the right-hand side is three normal distribution functions.
Optimising over the direction and offset of ``A`` can only lose tightness,
never correctness — a point the tests exploit by feeding
:func:`half_space_value` arbitrary directions and checking every one stays
below the price.

*Upper*, by super-replication with vanilla options:
``max(S1 - S2 - K, 0) <= max(S1 - a, 0) + max(a - S2 - K, 0)`` for every
``a``, so two Black formulas bound the spread with **no correlation in them at
all**. That is also the limit of what they can do. A bound holding for every
coupling of the two marginals is the price under the worst one, and for
lognormals the worst coupling is attainable, so the optimised portfolio does
not merely approximate the upper extreme — it equals the spread price at a
correlation of ``-1``, measured at 4.2e-14 to 6.7e-14 relative across five
strikes. Two routes with nothing in common agreeing to machine precision, which
is worth more than either of them agreeing with a tolerance.

Measured, at ``F1 = 100``, ``F2 = 95``, one year, 30% and 25% volatility:

* **Kirk's error changes sign**, so no tolerance describes it. It reads low at
  high correlation and short strikes — ``-1.5e-05`` at ``rho = 0.9, K = 1`` and
  ``-6.9e-05`` at ``rho = 0.8, K = 5`` — and high everywhere else, reaching
  ``+1.5e-02`` at ``rho = 0.9, K = 20`` and ``+5.5e-03`` at ``rho = -0.8,
  K = 20``. What is monotone is the strike: at every correlation measured the
  error grows with it.
* **The lower bound is usually the better number, and not always.** Across 84
  points its gap below the price is a median **59 times** smaller than Kirk's
  error, reaching a factor of ten million at ``rho = -0.99``. But it loses at
  high correlation and far strikes — by 3.4x at ``rho = 0.8, K = 20`` and 2x at
  ``rho = 0.9, K = 5`` — so the case for it is not that it is always closer. It
  is that its error has a *sign*: the price is above it, always, and Kirk could
  be either side.
* **The bound's gap is not monotone in correlation and the endpoints are
  exact.** At ``rho = +/-1`` the pair is driven by one variate, the exercise
  region genuinely is a half-space, and the bound is the price to 4.8e-14. In
  between it rises to a peak at ``rho ~ +0.95`` — 1.0e-06, 5.9e-06, 2.4e-05,
  6.9e-05, 1.2e-04 relative at ``rho`` of 0.999, 0.0, 0.5, 0.8 and 0.95 for
  ``K = 5`` — and falls away to 3.1e-10 at ``rho = -0.99``. Asserting that the
  bound tightens as the correlation rises would have passed on any sweep that
  stopped at 0.9.
* **The upper bound is a correlation reading, not an accuracy.** Its gap above
  the price is 1.1e-02 at ``rho = -0.95`` and 9.7 — nearly a factor of ten — at
  ``rho = 0.9, K = 20``, because that is the distance from the worst case it
  prices. The bracket is 1.1% of the price wide at ``rho = -0.95`` and 484% at
  ``rho = 0.9``.
* **The kink has to go on a panel edge, and the cost of missing it does not
  announce itself.** With no volatility on the first asset the integrand is
  genuinely kinked at one point, and uniform panels at 24, 48, 96 and 192
  panels give relative errors of 9.6e-06, **1.8e-05**, 3.3e-06 and 1.2e-06 —
  not even monotone in the refinement, so a two-point convergence check would
  have reported the method diverging. With the root of ``F1(z) - S2(z) - K``
  inserted as an edge the same quadrature is exact to 4.5e-14, eight orders of
  magnitude better.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

from .normal import norm_cdf
from .quadrature import fixed_quad

__all__ = [
    "FLOOR_EPSILONS",
    "Pair",
    "SpreadBounds",
    "SpreadError",
    "accuracy_floor",
    "critical_states",
    "exchange",
    "half_space_bound",
    "half_space_value",
    "kirk",
    "spread_bounds",
    "spread_price",
    "truncation_bound",
    "vanilla_split",
    "vanilla_upper_bound",
]

_MAX_WIDTH = 40.0
_EXP_CEILING = 600.0

#: Multiples of machine epsilon at which :func:`spread_price` stops resolving
#: anything. The price is a sum of order a hundred panel integrals, each of the
#: payoff's own size, so the round-off floor scales with the payoff rather than
#: with the answer and is far above epsilon. Measured as 6.1e3 epsilons of
#: ``e^{-rT} F1`` over 729 configurations spanning three decades of forward,
#: volatilities from 5% to 90%, maturities from a month to ten years and
#: correlations of -0.9, 0 and 0.9, each checked against Margrabe; this is that
#: worst case rounded up. Truncation can be driven far below it -- the default
#: tolerance does -- which does not make the digits underneath it real.
FLOOR_EPSILONS = 1.0e4


class SpreadError(ValueError):
    """A spread option that cannot be priced as asked.

    Raised when Kirk's approximation is applied to a shifted forward that is
    not positive, and when :func:`spread_price` is asked for an accuracy its
    truncation bound cannot reach.
    """


def _safe_exp(x: float) -> float:
    """``exp`` with the argument clamped, for use inside a root bracket.

    The root searches evaluate the integrand's structure function far out in
    ``z``, where an exponent can overflow even though the quadrature never
    reaches there. Clamping keeps the bracketing monotone instead of handing
    the search an ``inf`` to compare.
    """
    return math.exp(min(x, _EXP_CEILING))


@dataclass(frozen=True, slots=True)
class Pair:
    """Two lognormal assets and the option's market state.

    The assets enter only through their forwards, so a carry rate for each is
    one parameter too many: :meth:`from_spots` applies them and this class
    does not carry them.

    Attributes:
        forward1: Forward of the asset received. Non-negative.
        forward2: Forward of the asset given up. Non-negative.
        time: Year fraction to expiry. Non-negative.
        rate: Continuously compounded discount rate.
        vol1: Annualised lognormal volatility of the first asset.
        vol2: Annualised lognormal volatility of the second.
        correlation: Correlation of the two log returns, in ``[-1, 1]``.
    """

    forward1: float
    forward2: float
    time: float
    rate: float
    vol1: float
    vol2: float
    correlation: float

    def __post_init__(self) -> None:
        for name in ("forward1", "forward2", "time", "vol1", "vol2"):
            value = float(getattr(self, name))
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative, got {value}")
        for name in ("forward1", "forward2", "time", "rate", "vol1", "vol2", "correlation"):
            value = float(getattr(self, name))
            if value != value or math.isinf(value):
                raise ValueError(f"{name} must be finite, got {value}")
        if not -1.0 <= self.correlation <= 1.0:
            raise ValueError(f"correlation must lie in [-1, 1], got {self.correlation}")

    @classmethod
    def from_spots(
        cls,
        spot1: float,
        spot2: float,
        time: float,
        rate: float,
        vol1: float,
        vol2: float,
        correlation: float,
        carry1: float = 0.0,
        carry2: float = 0.0,
    ) -> Pair:
        """Build a pair from spots and a cost of carry for each asset."""
        return cls(
            spot1 * math.exp(carry1 * time),
            spot2 * math.exp(carry2 * time),
            time,
            rate,
            vol1,
            vol2,
            correlation,
        )

    @property
    def discount(self) -> float:
        """``e^{-rT}``."""
        return math.exp(-self.rate * self.time)

    @property
    def total1(self) -> float:
        """Total volatility of the first asset, ``v1 sqrt(T)``."""
        return self.vol1 * math.sqrt(self.time)

    @property
    def total2(self) -> float:
        """Total volatility of the second asset, ``v2 sqrt(T)``."""
        return self.vol2 * math.sqrt(self.time)

    @property
    def spread_total(self) -> float:
        """Total volatility of the log ratio, ``sqrt(v1^2 - 2 rho v1 v2 + v2^2) sqrt(T)``.

        This is the volatility Margrabe's formula is written in, and the one
        that goes to zero when the two assets are the same asset.
        """
        variance = self.vol1**2 - 2.0 * self.correlation * self.vol1 * self.vol2 + self.vol2**2
        return math.sqrt(max(variance, 0.0) * self.time)

    @property
    def is_degenerate(self) -> bool:
        """True when the terminal spread carries no uncertainty."""
        return self.time == 0.0 or (self.vol1 == 0.0 and self.vol2 == 0.0)

    def swapped(self) -> Pair:
        """The same market with the two assets exchanged.

        ``max(S2 - S1, 0)`` is the exchange option on the swapped pair, which
        is how the put side of the bracket is built.
        """
        return Pair(
            self.forward2,
            self.forward1,
            self.time,
            self.rate,
            self.vol2,
            self.vol1,
            self.correlation,
        )


def _black(forward: float, strike: float, total_vol: float, sign: float) -> float:
    """Undiscounted Black-76 value, ``sign`` being +1 for a call.

    Written on the forward and the total volatility so that it can serve as the
    conditional inner price without rebuilding an :class:`~moneyness.bsm.Inputs`
    on every quadrature node.
    """
    if total_vol <= 0.0 or forward <= 0.0 or strike <= 0.0:
        return max(sign * (forward - strike), 0.0)
    d1 = (math.log(forward / strike) + 0.5 * total_vol**2) / total_vol
    d2 = d1 - total_vol
    return sign * (forward * norm_cdf(sign * d1) - strike * norm_cdf(sign * d2))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def exchange(pair: Pair) -> float:
    """Margrabe's price for ``max(S1 - S2, 0)``, exactly.

    The ratio of two lognormals is lognormal, so an option to exchange one for
    the other is a Black-76 call on that ratio with the second asset as the
    numeraire. Exact, and therefore the reference the quadrature has to
    reproduce at a zero strike.
    """
    total = pair.spread_total
    discount = pair.discount
    if pair.forward2 == 0.0:
        return discount * pair.forward1
    if pair.forward1 == 0.0:
        return 0.0
    if total == 0.0:
        return discount * max(pair.forward1 - pair.forward2, 0.0)
    d1 = (math.log(pair.forward1 / pair.forward2) + 0.5 * total**2) / total
    d2 = d1 - total
    return discount * (pair.forward1 * norm_cdf(d1) - pair.forward2 * norm_cdf(d2))


def kirk(pair: Pair, strike: float, *, call: bool = True) -> float:
    """Kirk's approximation to the spread option price.

    ``S2 + K`` is replaced by a lognormal with the same mean and with its
    volatility scaled by ``F2 / (F2 + K)``, the share of the shifted forward
    the random part accounts for; the spread is then an exchange option
    between two lognormals and Margrabe applies.

    Raises:
        SpreadError: if ``F2 + K`` is not positive, where the substitution has
            no lognormal to make.
    """
    shifted = pair.forward2 + strike
    if shifted <= 0.0:
        raise SpreadError(f"Kirk needs a positive shifted forward, got F2 + K = {shifted}")
    weight = pair.forward2 / shifted
    variance = (
        pair.vol1**2
        - 2.0 * pair.correlation * pair.vol1 * pair.vol2 * weight
        + (pair.vol2 * weight) ** 2
    )
    total = math.sqrt(max(variance, 0.0) * pair.time)
    sign = 1.0 if call else -1.0
    return pair.discount * _black(pair.forward1, shifted, total, sign)


def critical_states(pair: Pair, strike: float, width: float) -> tuple[float, ...]:
    """Values of ``z`` in ``[-width, width]`` where the conditional option is at the money.

    ``F1(z) - S2(z) - K`` is a difference of two exponentials less a constant,
    so it has at most two roots, and they are where the integrand of
    :func:`spread_price` is least well approximated by a polynomial. As
    ``|rho| -> 1`` or either volatility goes to zero the conditional variance
    vanishes and the integrand acquires a genuine kink there, which is the case
    that forces these onto panel edges rather than merely suggesting it.
    """
    shift = pair.correlation * pair.total1
    total2 = pair.total2
    log1 = math.log(pair.forward1) if pair.forward1 > 0.0 else -math.inf
    log2 = math.log(pair.forward2) if pair.forward2 > 0.0 else -math.inf

    def gap(z: float) -> float:
        first = _safe_exp(log1 + shift * z - 0.5 * shift**2) if log1 > -math.inf else 0.0
        second = _safe_exp(log2 + total2 * z - 0.5 * total2**2) if log2 > -math.inf else 0.0
        return first - second - strike

    nodes = 512
    roots: list[float] = []
    previous_z = -width
    previous = gap(previous_z)
    for index in range(1, nodes + 1):
        current_z = -width + 2.0 * width * index / nodes
        current = gap(current_z)
        if previous == 0.0:
            roots.append(previous_z)
        elif previous * current < 0.0:
            low, high = previous_z, current_z
            for _ in range(80):
                middle = 0.5 * (low + high)
                if gap(low) * gap(middle) <= 0.0:
                    high = middle
                else:
                    low = middle
            roots.append(0.5 * (low + high))
        previous_z, previous = current_z, current
    if strike < 0.0 and pair.forward2 > 0.0 and total2 > 0.0:
        # Where the shifted strike changes sign and the conditional option
        # stops being an option. Not a kink, but the branch switch is here.
        branch = (math.log(-strike / pair.forward2) + 0.5 * total2**2) / total2
        if -width < branch < width:
            roots.append(branch)
    return tuple(sorted(roots))


def truncation_bound(pair: Pair, strike: float, width: float, *, call: bool = True) -> float:
    """An upper bound on the part of the price outside ``|z| <= width``.

    A spread call pays at most ``S1``, so the discarded mass is at most
    ``F1`` times the normal mass outside the interval under the measure in
    which ``S1`` is the numeraire — a shift of ``rho v1 sqrt(T)``. A spread put
    pays at most ``S2 + max(K, 0)``, which bounds the same way with a shift of
    ``v2 sqrt(T)``.

    This is why :func:`spread_price` can state an accuracy: the bound is
    closed form, so the ceiling is solved from the tolerance rather than set
    to a width that looks generous.
    """
    discount = pair.discount
    if call:
        shift = pair.correlation * pair.total1
        tail = norm_cdf(shift - width) + norm_cdf(-shift - width)
        return discount * pair.forward1 * tail
    shift = pair.total2
    tail = norm_cdf(shift - width) + norm_cdf(-shift - width)
    constant = 2.0 * norm_cdf(-width) * max(strike, 0.0)
    return discount * (pair.forward2 * tail + constant)


def accuracy_floor(pair: Pair, strike: float, *, call: bool = True) -> float:
    """Below this absolute error :func:`spread_price` resolves nothing.

    Reported rather than enforced, because truncation and round-off are
    different errors and only the first of them answers to a tolerance. See
    :data:`FLOOR_EPSILONS` for how the constant was measured.
    """
    scale = pair.forward1 if call else pair.forward2 + max(strike, 0.0)
    return FLOOR_EPSILONS * 2.220446049250313e-16 * pair.discount * scale


def _solve_width(pair: Pair, strike: float, tolerance: float, *, call: bool) -> float:
    """Smallest width whose truncation bound is within ``tolerance``.

    The bound underflows to exactly zero by a width of 39 for any pair whose
    total volatility is well inside the ceiling, so a bracket exists and the
    bisection below always has somewhere to land. A pair whose total volatility
    approaches the ceiling itself is the case that does not: the measure change
    shifts the normal tail by that much, nothing under 40 standard deviations
    bounds what is left, and widening further cannot be checked because the
    tail has stopped being representable. That is refused rather than answered.
    """
    if truncation_bound(pair, strike, _MAX_WIDTH, call=call) > tolerance:
        raise SpreadError(
            f"truncation bound {truncation_bound(pair, strike, _MAX_WIDTH, call=call):.3e} "
            f"at the widest ceiling of {_MAX_WIDTH:g} exceeds the tolerance "
            f"{tolerance:.3e}; the total volatility is "
            f"{max(pair.total1, pair.total2):.3g}"
        )
    low, high = 0.0, _MAX_WIDTH
    for _ in range(60):
        middle = 0.5 * (low + high)
        if truncation_bound(pair, strike, middle, call=call) > tolerance:
            low = middle
        else:
            high = middle
    return high


def spread_price(
    pair: Pair,
    strike: float,
    *,
    call: bool = True,
    tolerance: float = 1e-12,
    order: int = 24,
    panels_per_unit: float = 3.0,
) -> float:
    """The spread option price, exact up to the quadrature.

    Conditioning on the second asset's driving variate leaves a Black-76 call
    on the first, so the price is one integral against a normal density. The
    ceiling comes from :func:`truncation_bound` and the tolerance; the panel
    edges are uniform over the interval with the roots of
    :func:`critical_states` inserted.

    Args:
        pair: The two assets.
        strike: ``K`` in ``max(S1 - S2 - K, 0)``. May be negative, where the
            conditional option is sometimes certain to be exercised.
        call: False for ``max(S2 + K - S1, 0)``.
        tolerance: Absolute accuracy asked of the truncation.
        order: Gauss-Legendre order on each panel.
        panels_per_unit: Panels per unit of ``z``. The panel count grows with
            the ceiling, because a wider interval at a fixed panel count makes
            the answer worse rather than better.

    Raises:
        SpreadError: if no ceiling meets ``tolerance``.
    """
    if tolerance <= 0.0:
        raise ValueError(f"tolerance must be positive, got {tolerance}")
    sign = 1.0 if call else -1.0
    discount = pair.discount
    if pair.is_degenerate:
        return discount * max(sign * (pair.forward1 - pair.forward2 - strike), 0.0)

    shift = pair.correlation * pair.total1
    total2 = pair.total2
    conditional = pair.total1 * math.sqrt(max(1.0 - pair.correlation**2, 0.0))

    def integrand(z: float) -> float:
        second = pair.forward2 * _safe_exp(total2 * z - 0.5 * total2**2)
        shifted = second + strike
        first = pair.forward1 * _safe_exp(shift * z - 0.5 * shift**2)
        if shifted <= 0.0:
            inner = (first - shifted) if call else 0.0
        else:
            inner = _black(first, shifted, conditional, sign)
        return inner * _norm_pdf(z)

    width = _solve_width(pair, strike, tolerance, call=call)
    count = max(8, math.ceil(2.0 * width * panels_per_unit))
    edges = [-width + 2.0 * width * index / count for index in range(count + 1)]
    edges.extend(critical_states(pair, strike, width))
    edges.sort()
    total = 0.0
    for left, right in itertools.pairwise(edges):
        if right - left <= 0.0:
            continue
        total += fixed_quad(integrand, left, right, order=order)
    return discount * total


def half_space_value(
    pair: Pair, strike: float, angle: float, offset: float, *, call: bool = True
) -> float:
    """Value of exercising on the half-space ``{n . Z >= offset}``.

    ``n = (cos(angle), sin(angle))`` is a direction in the two-dimensional
    Gaussian driving the pair. Because ``max(x, 0) >= x 1_A`` for any event,
    the result is a lower bound on the price for *every* angle and offset, and
    the three terms are normal distribution functions of shifted arguments —
    each asset's own measure change moves ``Z`` by its total volatility along
    its own loading.
    """
    root = math.sqrt(max(1.0 - pair.correlation**2, 0.0))
    load1 = math.cos(angle)
    load2 = pair.correlation * math.cos(angle) + root * math.sin(angle)
    first = pair.forward1 * norm_cdf(pair.total1 * load1 - offset)
    second = pair.forward2 * norm_cdf(pair.total2 * load2 - offset)
    constant = strike * norm_cdf(-offset)
    value = (first - second - constant) if call else (second + constant - first)
    return pair.discount * max(value, 0.0)


def half_space_bound(pair: Pair, strike: float, *, call: bool = True) -> float:
    """The best sub-replicating half-space, as a rigorous lower bound.

    A grid over direction and offset followed by a shrinking pattern search.
    The search cannot produce an invalid answer: every value it visits is
    already a lower bound, so a worse optimum is a looser bound and not a
    wrong one.
    """
    best_angle, best_offset = 0.0, 0.0
    best = -math.inf
    for index in range(72):
        angle = 2.0 * math.pi * index / 72
        for step in range(-30, 31):
            offset = step / 5.0
            value = half_space_value(pair, strike, angle, offset, call=call)
            if value > best:
                best, best_angle, best_offset = value, angle, offset
    angle_step, offset_step = 2.0 * math.pi / 72, 0.2
    for _ in range(60):
        improved = False
        for d_angle, d_offset in (
            (angle_step, 0.0),
            (-angle_step, 0.0),
            (0.0, offset_step),
            (0.0, -offset_step),
        ):
            candidate = half_space_value(
                pair, strike, best_angle + d_angle, best_offset + d_offset, call=call
            )
            if candidate > best:
                best = candidate
                best_angle += d_angle
                best_offset += d_offset
                improved = True
        if not improved:
            angle_step *= 0.5
            offset_step *= 0.5
    return max(best, 0.0)


def vanilla_split(pair: Pair, strike: float) -> float:
    """The level ``a`` at which the super-replicating vanillas are cheapest.

    For any ``a``, ``max(S1 - S2 - K, 0) <= max(S1 - a, 0) + max(a - S2 - K, 0)``
    path by path: splitting at ``a`` either hands the whole shortfall to one of
    the two legs or divides it exactly between them. The cheapest split solves
    ``d2(F1, a) = -d2(F2, a - K)``, whose two sides move in opposite directions
    in ``a``, so one bisection finds it to machine precision — which matters,
    because a direct search on the portfolio value tops out several orders
    short of that.
    """
    low = max(0.0, strike) + 1e-12
    total1, total2 = pair.total1, pair.total2

    def residual(level: float) -> float:
        first = (math.log(pair.forward1 / level) - 0.5 * total1**2) / total1
        second = (math.log(pair.forward2 / (level - strike)) - 0.5 * total2**2) / total2
        return first + second

    # The residual falls from ``+inf`` at the floor to ``-inf`` as the level
    # grows, so the bracket is a positive left end and a negative right one.
    high = max(low * 2.0, pair.forward1 + pair.forward2 + abs(strike) + 1.0)
    while residual(high) > 0.0:
        high *= 2.0
        if high > 1e18:
            raise SpreadError("no finite split level brackets the vanilla bound")
    while residual(low) < 0.0:
        low = max(0.0, strike) + (low - max(0.0, strike)) * 0.5
        if low - max(0.0, strike) < 1e-300:
            return low
    for _ in range(200):
        middle = 0.5 * (low + high)
        if middle in (low, high):
            break
        if residual(middle) > 0.0:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def vanilla_upper_bound(pair: Pair, strike: float, *, call: bool = True) -> float:
    """A rigorous upper bound built from two vanilla options.

    The super-replicating portfolio of :func:`vanilla_split` uses no
    correlation at all, which is exactly what it costs: a bound that holds for
    every coupling of the two marginals is the price under the worst one. For
    lognormals that worst coupling is attainable, so this bound is not merely
    loose away from it — it *equals* the spread price at a correlation of
    ``-1``, and the tests assert that rather than quoting a tolerance.
    """
    if pair.is_degenerate or pair.total1 == 0.0 or pair.total2 == 0.0:
        return _domination_upper(pair, strike, call=call)
    if not call:
        return vanilla_upper_bound(pair, strike, call=True) - pair.discount * (
            pair.forward1 - pair.forward2 - strike
        )
    level = vanilla_split(pair, strike)
    first = _black(pair.forward1, level, pair.total1, 1.0)
    second = _black(pair.forward2, level - strike, pair.total2, -1.0)
    return pair.discount * (first + second)


@dataclass(frozen=True, slots=True)
class SpreadBounds:
    """A rigorous interval the spread option price lies in.

    Attributes:
        lower: The best of the sub-replicating half-space, the discounted
            intrinsic on the forwards, and zero.
        upper: The best of the payoff dominations available.
    """

    lower: float
    upper: float

    @property
    def width(self) -> float:
        """``upper - lower``."""
        return self.upper - self.lower

    def contains(self, price: float, *, slack: float = 0.0) -> bool:
        """Whether ``price`` lies in the interval, with optional slack."""
        return self.lower - slack <= price <= self.upper + slack


def _domination_upper(pair: Pair, strike: float, *, call: bool) -> float:
    """The upper bound available from dominating the payoff outright.

    A spread call pays less than the exchange option, less than an ordinary
    call struck at ``K``, and less than the first asset. A spread put pays
    less than the reversed exchange option plus the strike, and less than the
    second asset plus the strike.
    """
    discount = pair.discount
    if call:
        uppers = [discount * pair.forward1]
        if strike >= 0.0:
            uppers.append(exchange(pair))
            if strike > 0.0:
                uppers.append(discount * _black(pair.forward1, strike, pair.total1, 1.0))
        else:
            uppers.append(exchange(pair) + discount * (-strike))
        return min(uppers)
    return min(
        discount * (pair.forward2 + max(strike, 0.0)),
        exchange(pair.swapped()) + discount * max(strike, 0.0),
    )


def spread_bounds(pair: Pair, strike: float, *, call: bool = True) -> SpreadBounds:
    """Bound the spread option price by inequalities on its payoff.

    Lower, from sub-replication: a half-space is one exercise strategy, and
    Jensen gives the discounted intrinsic on the forwards. Upper, the better of
    outright domination and the vanilla super-replication of
    :func:`vanilla_upper_bound`.
    """
    discount = pair.discount
    forward_gap = pair.forward1 - pair.forward2 - strike
    intrinsic = forward_gap if call else -forward_gap
    lower = max(
        0.0,
        discount * max(intrinsic, 0.0),
        half_space_bound(pair, strike, call=call),
    )
    upper = min(
        _domination_upper(pair, strike, call=call),
        vanilla_upper_bound(pair, strike, call=call),
    )
    return SpreadBounds(lower, upper)
