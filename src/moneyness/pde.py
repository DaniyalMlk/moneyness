"""Finite-difference pricing under a local volatility.

:meth:`moneyness.surface.Surface.local_vol` answers the question "what
volatility, as a function of spot and time, reproduces these option prices".
This module is the other half of that sentence: it takes such a function and
produces the prices, so that the identity can be checked against the thing it
claims rather than against a finite-difference derivative of itself.

**The coordinate that matters.** Gatheral's form of the Dupire identity is
written in log-moneyness measured from the *forward*, and the local volatility
it returns at ``(k, T)`` belongs to the spot level ``S`` with
``ln(S / F_T) = k``, not to ``ln(S / S_0) = k``. The two agree only when the
cost of carry is zero. No test that differentiates the surface can see the
difference, because both readings differentiate the same function; only a price
can, which is why :func:`dupire_local_vol` exists and does that mapping in one
place instead of leaving it to each caller.

**The front stub, which is not a detail.** A :class:`~moneyness.surface.Surface`
holds total variance flat outside its quoted maturities, so ``dw/dT`` is zero
before the first quote and the Dupire numerator with it. Read literally, the
surface therefore says the local variance is zero on ``(0, T_1]`` while the
implied variance at ``T_1`` is positive — which is not a quirk of the
interpolation but a genuine calendar arbitrage at the origin, since total
variance must vanish as maturity does. Nothing can reproduce such a surface,
and a solver handed it prices every option at its forward intrinsic.
:class:`FrontStub` is the choice about what to do instead, and
:attr:`FrontStub.LINEAR` — total variance ramping linearly from zero — is the
default because it is the only one of the two under which the front slice is
priceable at all.

**The scheme.** Backward equation in ``x = ln S``, marched in time to expiry
``tau = T - t``:

    dV/dtau = (sigma^2/2) d2V/dx2 + (b - sigma^2/2) dV/dx - r V

Central differences in space, and a theta-scheme in time started by a few fully
implicit steps. The implicit start is Rannacher's, and the reason to want it is
not the price but the second derivative: Crank-Nicolson does not damp the
high-frequency content a kinked payoff puts on the grid, so gamma near the
strike is unusable even where the price looks respectable. The price error is
where it ought to be — second order in both the spacing and the step, measured
at ratios of 4.00 and 3.96 under refinement — and gamma, on the same mesh, is
wrong by 132%. :attr:`Mesh.rannacher` has the numbers.

Both the spot and the strike are placed exactly on grid nodes. The strike has
to be, or the payoff kink falls between nodes and the convergence order drops
to one; the spot is a convenience that turns the price, delta and gamma into a
node read and three differences rather than an interpolation.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum

from .bsm import OptionType
from .lattice import Exercise
from .surface import Surface

__all__ = [
    "DupireLocalVol",
    "FrontStub",
    "LocalVolError",
    "LocalVolatility",
    "Mesh",
    "MeshPrice",
    "dupire_local_vol",
    "price_pde",
]

LocalVolatility = Callable[[float, float], float]
"""A local volatility: ``(spot, calendar time) -> annualised volatility``."""


class LocalVolError(ValueError):
    """A local volatility was asked for where it does not exist."""


class FrontStub(Enum):
    """What to do with the maturities before the surface's first quote.

    Attributes:
        LINEAR: Ramp total variance linearly from zero at ``T = 0`` up to the
            first quoted slice. What this makes constant across the stub is
            ``dw/dT``, the Dupire *numerator*, which is what the surface was
            missing and what the front slice needs in order to reprice. The
            local variance itself still moves, because Durrleman's denominator
            depends on the level of total variance and that level is ramping:
            on a realistic front slice it falls by about a quarter from the
            origin to the first quote. At the money and in the limit ``T -> 0``
            the denominator goes to one exactly, so the local variance there
            approaches the first slice's own at-the-money implied variance.
        FLAT: Take the surface exactly as written, flat in total variance below
            the first quote. The local variance is zero there, and every option
            expiring on or before the first quoted maturity is worth its
            forward intrinsic. Provided so that the cost of the literal reading
            can be measured rather than asserted.
    """

    LINEAR = "linear"
    FLAT = "flat"


@dataclass(frozen=True, slots=True)
class Mesh:
    """The discretisation, separately from the contract.

    Attributes:
        space_steps: Number of spatial intervals requested. The solver may use
            a few more, because the spacing is adjusted so that the strike
            lands on a node and the domain is then filled out to the requested
            width.
        time_steps: Number of time steps requested, distributed across the
            intervals between breakpoints in proportion to their length.
        width: Half-width of the spatial domain, in standard deviations of a
            probe of the local volatility. Six, because with the spacing held
            fixed and only the domain grown, a one-unit widening moves a
            year-long at-the-money call by 2.8e-04 at width two, 5.9e-09 at
            three, 6.1e-12 at four, and nothing distinguishable from rounding
            after that. Truncation is therefore spent by five, and six is one
            unit of margin on a cost that has already vanished. Note what the
            margin is paid in: ``space_steps`` nodes spread over a wider domain
            are further apart, and the ``h^2`` error that follows is the term
            that actually dominates — at width six it is 1.0e-04 on the same
            option, four orders of magnitude above the truncation it bought
            protection from. Widen the domain and the node count together, or
            the widening makes the answer worse.
        rannacher: How many of the first time steps are taken fully implicit
            rather than Crank-Nicolson. Two, and the reason is gamma rather
            than the price: on a 400-by-40 mesh, pure Crank-Nicolson prices a
            year-long at-the-money call to -3.4e-03 but returns a gamma of
            0.0449 against a true 0.0193, wrong by 132%. One implicit step
            gives the best price of any setting here, -1.3e-03, and still
            leaves gamma out by -5.6e-04; two cost the price 7.3e-04 and bring
            gamma to +2.2e-05, twenty-five times closer. Each further implicit
            step is a first-order step and costs the price about 1.3e-03
            without improving gamma, so beyond two there is nothing to buy.
        reference_vol: The volatility ``width`` is measured in, overriding the
            at-the-money probe. Pass it when the smile is steep enough that the
            at-the-money local volatility is not the right scale for the
            terminal distribution's spread.

    Raises:
        ValueError: If any field is out of range, or ``rannacher`` exceeds
            ``time_steps``.
    """

    space_steps: int = 200
    time_steps: int = 100
    width: float = 6.0
    rannacher: int = 2
    reference_vol: float | None = None

    def __post_init__(self) -> None:
        if self.reference_vol is not None and (
            not math.isfinite(self.reference_vol) or self.reference_vol <= 0.0
        ):
            raise ValueError(
                f"reference_vol must be finite and positive, got {self.reference_vol}"
            )
        if self.space_steps < 8:
            raise ValueError(f"space_steps must be at least 8, got {self.space_steps}")
        if self.time_steps < 1:
            raise ValueError(f"time_steps must be at least 1, got {self.time_steps}")
        if not math.isfinite(self.width) or self.width <= 0.0:
            raise ValueError(f"width must be finite and positive, got {self.width}")
        if self.rannacher < 0:
            raise ValueError(f"rannacher must be non-negative, got {self.rannacher}")
        if self.rannacher > self.time_steps:
            raise ValueError(
                f"rannacher ({self.rannacher}) cannot exceed time_steps ({self.time_steps})"
            )


@dataclass(frozen=True, slots=True)
class MeshPrice:
    """A valuation, with the grid that produced it and what the grid knows.

    Attributes:
        value: The option price.
        delta: ``dV/dS`` at the spot node, by central difference.
        gamma: ``d2V/dS2`` at the spot node. Read it as the headline diagnostic
            of the time scheme: it is the quantity a pure Crank-Nicolson march
            gets wrong near the strike.
        nodes: Spatial nodes used, boundaries included.
        steps: Time steps taken.
        spot_range: The spot values at the two Dirichlet boundaries.
        early_exercise_premium: ``value`` less the European value on the *same*
            mesh. Differencing on one grid cancels the discretisation error to
            leading order, which a subtraction from the closed form would not.
            Zero for a European valuation.
        boundary: For an American valuation, ``(time remaining, spot)`` pairs
            tracing the exercise boundary, one per time step, latest first.
            ``nan`` for the spot where no node exercised. Empty for a European
            valuation.
    """

    value: float
    delta: float
    gamma: float
    nodes: int
    steps: int
    spot_range: tuple[float, float]
    early_exercise_premium: float
    boundary: tuple[tuple[float, float], ...] = field(default=())


@dataclass(frozen=True, slots=True)
class DupireLocalVol:
    """The surface's local volatility, as a function of spot and time.

    Callable as ``vol(spot, time)``. Also carries :attr:`breakpoints`, the
    calendar times at which the local variance jumps, so that a solver can put
    time nodes there: the quantity is piecewise constant in time and a grid
    that straddles a jump averages across it at first order.

    The local variance is taken *left*-continuous in time, so that the interval
    ending at a quoted maturity governs that maturity itself. The alternative
    makes the last quoted maturity the one point where the surface's own slope
    is zero, and so the one maturity that cannot be priced.

    Attributes:
        surface: The surface read.
        spot: Spot now, which fixes the forwards the mapping is measured from.
        carry: Cost of carry, which fixes them too.
        front: What happens below the first quoted maturity.
        breakpoints: Calendar times where the local variance is discontinuous.
    """

    surface: Surface
    spot: float
    carry: float
    front: FrontStub
    breakpoints: tuple[float, ...]

    def __call__(self, spot: float, time: float) -> float:
        return math.sqrt(self.variance(spot, time))

    def variance(self, spot: float, time: float) -> float:
        """Local *variance* at ``(spot, time)``.

        Raises:
            LocalVolError: If ``spot`` is not positive, if ``time`` is negative
                or beyond the last quoted maturity, or if the Dupire identity
                has no admissible answer at this point.
        """
        if spot <= 0.0 or not math.isfinite(spot):
            raise LocalVolError(f"spot must be finite and positive, got {spot}")
        if time < 0.0 or not math.isfinite(time):
            raise LocalVolError(f"time must be finite and non-negative, got {time}")
        maturities = self.surface.maturities
        last = maturities[-1]
        if time > last:
            raise LocalVolError(
                f"time {time} is past the last quoted maturity {last}; the surface is "
                "flat in total variance beyond it, so its local variance there is zero "
                "and every option would be worth its forward intrinsic"
            )
        if time == 0.0:
            # The limit from above. The payoff does not use it, but a probe may.
            time = min(last, 1e-12)
        forward = self.spot * math.exp(self.carry * time)
        k = math.log(spot / forward)
        first = maturities[0]

        if time <= first:
            if self.front is FrontStub.FLAT:
                return 0.0
            scale = time / first
            w = scale * self.surface.total_variance(k, first)
            dw = scale * self.surface.d_total_variance(k, first)
            d2w = scale * self.surface.d2_total_variance(k, first)
            numerator = self.surface.total_variance(k, first) / first
        else:
            w = self.surface.total_variance(k, time)
            dw = self.surface.d_total_variance(k, time)
            d2w = self.surface.d2_total_variance(k, time)
            index = next(i for i in range(1, len(maturities)) if maturities[i] >= time)
            span = maturities[index] - maturities[index - 1]
            early = self.surface.total_variance(k, maturities[index - 1])
            late = self.surface.total_variance(k, maturities[index])
            numerator = (late - early) / span

        if w <= 0.0:
            raise LocalVolError(
                f"total variance is {w} at spot {spot}, time {time}; the identity is "
                "undefined there"
            )
        bracket = 1.0 - k * dw / (2.0 * w)
        g = bracket * bracket - (dw * dw / 4.0) * (1.0 / w + 0.25) + d2w / 2.0
        if g <= 0.0:
            raise LocalVolError(
                f"Durrleman's function is {g} at spot {spot}, time {time}: the slice "
                "there implies a negative density, so no local volatility exists"
            )
        if numerator < 0.0:
            raise LocalVolError(
                f"dw/dT is {numerator} at spot {spot}, time {time}: the quoted slices "
                "cross, so no local volatility exists"
            )
        return numerator / g


def dupire_local_vol(
    surface: Surface,
    spot: float,
    *,
    carry: float = 0.0,
    front: FrontStub = FrontStub.LINEAR,
) -> DupireLocalVol:
    """Build the local volatility a surface implies.

    Args:
        surface: The quoted surface.
        spot: Spot now. Positive.
        carry: Cost of carry, which sets the forwards the surface's
            log-moneyness is measured from.
        front: What to do below the first quoted maturity. See
            :class:`FrontStub`; the default is the only choice under which the
            front slice reprices.

    Returns:
        A :class:`DupireLocalVol`.

    Raises:
        ValueError: If ``spot`` is not positive, or ``carry`` is not finite.
    """
    if not math.isfinite(spot) or spot <= 0.0:
        raise ValueError(f"spot must be finite and positive, got {spot}")
    if not math.isfinite(carry):
        raise ValueError(f"carry must be finite, got {carry}")
    return DupireLocalVol(
        surface=surface,
        spot=float(spot),
        carry=float(carry),
        front=front,
        breakpoints=surface.maturities,
    )


def _payoff(spot: float, strike: float, option: OptionType) -> float:
    if option is OptionType.CALL:
        return max(spot - strike, 0.0)
    return max(strike - spot, 0.0)


def _reference_vol(
    vol: LocalVolatility, spot: float, strike: float, time: float, carry: float
) -> float:
    """A probe of the local volatility at the money, used only to size the domain.

    Sampled at the forward and at the strike over a handful of times, and
    nowhere else. That is a decision and not laziness: a local volatility rises
    steeply into the wings — on an SVI surface with a realistic skew it reaches
    2.1 at two units of log-moneyness against 0.21 at the money — so a probe
    that took the maximum over a wide net would size the domain from a
    volatility the process only sees once it is already out there, and the
    domain would swell by orders of magnitude for nothing. The cost of the
    narrow probe is that a steep smile does diffuse a little faster in the
    wings than the at-the-money reading admits, so the domain comes out
    slightly narrower in implied standard deviations than :attr:`Mesh.width`
    names. :attr:`Mesh.reference_vol` is the override for a caller who knows
    the wing level and wants the width measured against it.

    Deliberately forgiving: a probe that lands where the local volatility does
    not exist is dropped rather than raised, because a single bad sample is not
    yet a reason to refuse to price. The solve itself does not forgive, so a
    genuine defect inside the domain is still an error.
    """
    best = 0.0
    for i in range(6):
        when = time * i / 5.0
        for level in (spot * math.exp(carry * when), strike):
            try:
                sampled = vol(level, when)
            except (LocalVolError, ValueError):
                continue
            if math.isfinite(sampled) and sampled > best:
                best = sampled
    return best


def _space_nodes(
    spot: float, strike: float, time: float, carry: float, reference: float, mesh: Mesh
) -> list[float]:
    """Log-spot nodes, anchored so that the strike is exactly one of them.

    The domain is centred on the spot and reaches ``Mesh.width`` standard
    deviations either side; the nodes are then laid on a lattice whose origin
    is the strike, which puts the payoff kink on a node without constraining
    the spacing.

    An earlier version anchored on the spot and shrank the spacing until the
    strike landed on a node too. That is fine until the strike is near the
    spot, at which point the spacing it demands is the gap between them: a
    strike 1% from the spot produced 1781 nodes where 600 were asked for, and
    a strike half a percent away would have produced twice that. The spot does
    not need to be on a node — three nodes around it and a quadratic are
    enough for the price and both derivatives — and the strike does.
    """
    centre = math.log(spot)
    anchor = math.log(strike)
    span = mesh.width * reference * math.sqrt(time) + abs(carry) * time
    step = 2.0 * span / mesh.space_steps
    lower = min(math.ceil((centre - span - anchor) / step), -2)
    upper = max(math.floor((centre + span - anchor) / step), 2)
    return [anchor + i * step for i in range(lower, upper + 1)]


def _read(nodes: list[float], values: list[float], spot: float) -> tuple[float, float, float]:
    """Value, ``dV/dS`` and ``d2V/dS2`` at ``spot``, off the three nearest nodes.

    The quadratic through three equally spaced nodes is second-order accurate
    in the value and in the first derivative, and its second derivative is the
    usual central difference. At a spot that happens to fall on a node this is
    exactly the node value and the two central differences.
    """
    step = nodes[1] - nodes[0]
    centre = min(max(round((math.log(spot) - nodes[0]) / step), 1), len(nodes) - 2)
    offset = (math.log(spot) - nodes[centre]) / step
    low, mid, high = values[centre - 1], values[centre], values[centre + 1]
    curve = high - 2.0 * mid + low
    slope = 0.5 * (high - low)
    value = mid + offset * slope + 0.5 * offset * offset * curve
    first = (slope + offset * curve) / step
    second = curve / (step * step)
    return value, first / spot, (second - first) / (spot * spot)


def _time_nodes(time: float, steps: int, breakpoints: Sequence[float]) -> list[float]:
    """Times to expiry, ascending from zero, with every breakpoint on a node."""
    cuts = {0.0, time}
    for raw in breakpoints:
        value = float(raw)
        if 0.0 < value < time:
            cuts.add(time - value)
    edges = sorted(cuts)
    spans = [b - a for a, b in zip(edges[:-1], edges[1:], strict=True)]
    total = sum(spans)
    counts = [max(1, round(steps * s / total)) for s in spans]
    nodes = [0.0]
    for lower, upper, count in zip(edges[:-1], edges[1:], counts, strict=True):
        nodes.extend(lower + (upper - lower) * i / count for i in range(1, count + 1))
    nodes[-1] = time
    return nodes


def _operator(
    nodes: list[float], vol: LocalVolatility, when: float, rate: float, carry: float, step: float
) -> tuple[list[float], list[float], list[float]]:
    """The tridiagonal spatial operator at one time, on the interior nodes."""
    size = len(nodes)
    lower = [0.0] * size
    diag = [0.0] * size
    upper = [0.0] * size
    inv = 1.0 / step
    inv2 = inv * inv
    for j in range(1, size - 1):
        sigma = vol(math.exp(nodes[j]), when)
        if not math.isfinite(sigma) or sigma < 0.0:
            raise LocalVolError(
                f"local volatility is {sigma} at spot {math.exp(nodes[j])}, time {when}"
            )
        a = 0.5 * sigma * sigma
        c = carry - a
        lower[j] = a * inv2 - 0.5 * c * inv
        diag[j] = -2.0 * a * inv2 - rate
        upper[j] = a * inv2 + 0.5 * c * inv
    return lower, diag, upper


def _dirichlet(
    nodes: list[float],
    strike: float,
    tau: float,
    rate: float,
    carry: float,
    option: OptionType,
) -> tuple[float, float]:
    """Values at the two boundaries, from the deep in- and out-of-the-money limits.

    Far enough from the strike an option is worth its discounted forward
    intrinsic exactly, so these are not approximations of the boundary
    condition but the condition itself, up to the probability of the domain
    being reached at all. That probability is what :attr:`Mesh.width` buys, and
    its docstring has the measurement of how much of it is worth buying.
    """
    low = math.exp(nodes[0]) * math.exp((carry - rate) * tau)
    high = math.exp(nodes[-1]) * math.exp((carry - rate) * tau)
    discounted = strike * math.exp(-rate * tau)
    if option is OptionType.CALL:
        return 0.0, max(high - discounted, 0.0)
    return max(discounted - low, 0.0), 0.0


def _thomas(lower: list[float], diag: list[float], upper: list[float], rhs: list[float]) -> None:
    """Solve a tridiagonal system in place, overwriting ``rhs`` with the answer."""
    size = len(rhs)
    scratch = [0.0] * size
    pivot = diag[0]
    if pivot == 0.0:
        raise ZeroDivisionError("tridiagonal solve met a zero pivot on the first row")
    scratch[0] = upper[0] / pivot
    rhs[0] = rhs[0] / pivot
    for i in range(1, size):
        pivot = diag[i] - lower[i] * scratch[i - 1]
        if pivot == 0.0:
            raise ZeroDivisionError(f"tridiagonal solve met a zero pivot on row {i}")
        scratch[i] = upper[i] / pivot
        rhs[i] = (rhs[i] - lower[i] * rhs[i - 1]) / pivot
    for i in range(size - 2, -1, -1):
        rhs[i] -= scratch[i] * rhs[i + 1]


def _march(
    nodes: list[float],
    times: list[float],
    spots: list[float],
    payoffs: list[float],
    vol: LocalVolatility,
    strike: float,
    rate: float,
    carry: float,
    option: OptionType,
    exercise: Exercise,
    rannacher: int,
    expiry: float,
) -> tuple[list[float], list[tuple[float, float]]]:
    """March the payoff back to now, returning the values and any boundary."""
    size = len(nodes)
    step = nodes[1] - nodes[0]
    values = list(payoffs)
    boundary: list[tuple[float, float]] = []
    interior = range(1, size - 1)

    previous_tau = times[0]
    low, high = _dirichlet(nodes, strike, previous_tau, rate, carry, option)
    values[0], values[-1] = low, high

    for index in range(1, len(times)):
        tau = times[index]
        delta_tau = tau - previous_tau
        theta = 1.0 if index <= rannacher else 0.5
        # One operator per step, read at the step's midpoint and used on both
        # sides of the theta-average. Reading it at the two endpoints instead
        # is equally second order for a coefficient smooth in time and loses an
        # order for one that is not: a local variance that jumps at a quoted
        # maturity gets the mean of its two sides on the step that begins
        # there, which is wrong by O(1) in the coefficient and so by O(dtau) in
        # the answer, once, and that term never refines away. The midpoint is
        # strictly inside the step, so with the jumps on step boundaries it
        # always reads the value that actually governs the step.
        middle = expiry - 0.5 * (previous_tau + tau)
        n_low, n_diag, n_upper = _operator(nodes, vol, middle, rate, carry, step)
        low, high = _dirichlet(nodes, strike, tau, rate, carry, option)

        rhs = [0.0] * (size - 2)
        if theta < 1.0:
            for j in interior:
                rhs[j - 1] = values[j] + (1.0 - theta) * delta_tau * (
                    n_low[j] * values[j - 1] + n_diag[j] * values[j] + n_upper[j] * values[j + 1]
                )
        else:
            for j in interior:
                rhs[j - 1] = values[j]

        a = [0.0] * (size - 2)
        b = [0.0] * (size - 2)
        c = [0.0] * (size - 2)
        for j in interior:
            a[j - 1] = -theta * delta_tau * n_low[j]
            b[j - 1] = 1.0 - theta * delta_tau * n_diag[j]
            c[j - 1] = -theta * delta_tau * n_upper[j]
        rhs[0] += theta * delta_tau * n_low[1] * low
        rhs[-1] += theta * delta_tau * n_upper[size - 2] * high
        a[0] = 0.0
        c[-1] = 0.0
        _thomas(a, b, c, rhs)

        values[0] = low
        values[-1] = high
        for j in interior:
            values[j] = rhs[j - 1]

        if exercise is Exercise.AMERICAN:
            crossing = math.nan
            for j in range(size):
                if payoffs[j] > values[j]:
                    values[j] = payoffs[j]
                    if option is OptionType.PUT:
                        crossing = spots[j]
                    elif math.isnan(crossing):
                        crossing = spots[j]
            boundary.append((expiry - tau, crossing))

        previous_tau = tau

    boundary.reverse()
    return values, boundary


def price_pde(
    *,
    spot: float,
    strike: float,
    time: float,
    rate: float,
    vol: LocalVolatility,
    option: OptionType,
    carry: float | None = None,
    exercise: Exercise = Exercise.EUROPEAN,
    mesh: Mesh | None = None,
    breakpoints: Sequence[float] = (),
) -> MeshPrice:
    """Price one option under a local volatility.

    Args:
        spot: Spot now. Positive.
        strike: Exercise price. Positive.
        time: Year fraction to expiry. Non-negative.
        rate: Continuously compounded discount rate.
        vol: The local volatility, ``(spot, calendar time) -> volatility``.
        option: Call or put.
        carry: Cost of carry. Defaults to ``rate``.
        exercise: European or American.
        mesh: The discretisation. Defaults to :class:`Mesh`'s own defaults.
        breakpoints: Calendar times at which ``vol`` is discontinuous in time,
            forced onto the time grid. A :class:`DupireLocalVol` carries the
            right ones in :attr:`DupireLocalVol.breakpoints`. Pass them. The
            reason is not that they make the error smaller on any given mesh —
            on a 1.37-year round trip through a four-slice surface they do at
            two of four refinements and do not at the other two — it is that
            they make it *orderly*: aligned, the error falls at ratios of
            3.77, 4.11, 3.94 and 3.98 under doubling, and unaligned it goes
            1.93, 18.08, 6.54 and then changes sign, because which side of a
            jump each step reads depends on where the steps happen to fall. An
            error that is not monotone in the mesh cannot be extrapolated and
            should not be trusted at any single mesh either.

    Returns:
        A :class:`MeshPrice`.

    Raises:
        ValueError: If ``spot``, ``strike`` or ``time`` is out of range, or any
            input is not finite.
        LocalVolError: If ``vol`` has no answer somewhere on the grid.
    """
    if not math.isfinite(spot) or spot <= 0.0:
        raise ValueError(f"spot must be finite and positive, got {spot}")
    if not math.isfinite(strike) or strike <= 0.0:
        raise ValueError(f"strike must be finite and positive, got {strike}")
    if not math.isfinite(time) or time < 0.0:
        raise ValueError(f"time must be finite and non-negative, got {time}")
    if not math.isfinite(rate):
        raise ValueError(f"rate must be finite, got {rate}")
    b = rate if carry is None else carry
    if not math.isfinite(b):
        raise ValueError(f"carry must be finite, got {b}")
    grid = Mesh() if mesh is None else mesh

    if time == 0.0:
        value = _payoff(spot, strike, option)
        sign = 1.0 if option is OptionType.CALL else -1.0
        return MeshPrice(
            value=value,
            delta=sign if value > 0.0 else 0.0,
            gamma=0.0,
            nodes=0,
            steps=0,
            spot_range=(spot, spot),
            early_exercise_premium=0.0,
        )

    reference = (
        grid.reference_vol
        if grid.reference_vol is not None
        else _reference_vol(vol, spot, strike, time, b)
    )
    if reference <= 0.0:
        # No diffusion anywhere the probe could see. Central differencing on a
        # pure drift is unstable and there is nothing to gain from it: the
        # answer is the discounted intrinsic of the forward, and for an
        # American right, the better of that and exercising now.
        forward = spot * math.exp(b * time)
        european = math.exp(-rate * time) * _payoff(forward, strike, option)
        value = european
        if exercise is Exercise.AMERICAN:
            value = max(european, _payoff(spot, strike, option))
        return MeshPrice(
            value=value,
            delta=math.nan,
            gamma=math.nan,
            nodes=0,
            steps=0,
            spot_range=(spot, spot),
            early_exercise_premium=value - european,
            boundary=(),
        )

    nodes = _space_nodes(spot, strike, time, b, reference, grid)
    times = _time_nodes(time, grid.time_steps, breakpoints)
    spots = [math.exp(x) for x in nodes]
    payoffs = [_payoff(s, strike, option) for s in spots]

    values, boundary = _march(
        nodes,
        times,
        spots,
        payoffs,
        vol,
        strike,
        rate,
        b,
        option,
        exercise,
        grid.rannacher,
        time,
    )
    value, first, second = _read(nodes, values, spot)

    premium = 0.0
    if exercise is Exercise.AMERICAN:
        reference_values, _ = _march(
            nodes,
            times,
            spots,
            payoffs,
            vol,
            strike,
            rate,
            b,
            option,
            Exercise.EUROPEAN,
            grid.rannacher,
            time,
        )
        premium = value - _read(nodes, reference_values, spot)[0]

    return MeshPrice(
        value=value,
        delta=first,
        gamma=second,
        nodes=len(nodes),
        steps=len(times) - 1,
        spot_range=(spots[0], spots[-1]),
        early_exercise_premium=premium,
        boundary=tuple(boundary),
    )
