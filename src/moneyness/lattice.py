"""Lattice pricing, and the right to exercise early.

A European option is a formula. An American option is an optimal stopping
problem: at every instant the holder compares what the contract is worth if held
and what it is worth if exercised, and takes the larger. There is no closed form
for that comparison under lognormal dynamics, so the standard route is to
discretise time into layers, solve the comparison exactly on the discrete tree,
and let the layer count grow.

Three constructions are provided. All three place the same lognormal law on the
terminal node set as the layer count grows; they differ in how they split the
first two moments between the node spacing and the probabilities.

``CRR``
    Cox-Ross-Rubinstein. The spacing is symmetric in log-price, ``u = e^{v sqrt(dt)}``
    and ``d = 1/u``, and the whole of the drift is carried by the probability.
    Because ``u d = 1`` the node set recombines onto a fixed log-grid that does
    not move with the drift, which is what makes the boundary read-out below
    clean.

``JARROW_RUDD``
    The probability is pinned at one half and the drift is carried by the
    spacing instead. Equal weights make the layer a symmetric binomial, so the
    convergence to the normal limit is smoother — at the cost of a log-grid that
    shifts with the carry.

``TRINOMIAL``
    Boyle's three-branch layer, with an unchanged middle node. One extra degree
    of freedom buys a node set twice as fine in log-price for the same number of
    layers, and the middle branch means a node sits on the strike-crossing
    region at every layer rather than straddling it.

Every construction has a layer count below which it is not usable at all. The
drift over one layer is ``b dt`` and the spread is of order ``v sqrt(dt)``; when
the first outgrows the second, the risk-neutral probability leaves ``[0, 1]``
and the recursion is no longer an expectation. See :func:`min_steps`, which is
enforced rather than documented.

The error in a lattice price is ``O(1/n)`` in the layer count, but it is not a
smooth ``c/n``: it carries a large oscillating component, because what the
lattice gets wrong depends on where the strike falls relative to the terminal
node grid, and that position jumps around as ``n`` changes. Measured on a
European call at ``S=100, K=95, T=0.5, r=0.04, v=0.22``, ``n`` times the error
wanders over roughly ``[-0.5, +1.2]`` rather than settling on a constant.

That oscillation is what makes the obvious accuracy trick fail. Richardson
extrapolation assumes the error is smooth in ``1/n``; applied to a raw lattice it
amplifies the oscillation instead of cancelling it, and on the grid above it
makes the answer about twice *worse* rather than four times better.

The cure is to remove the cause. The oscillation comes from the kink in the
terminal payoff sitting between nodes, so :func:`price_lattice` can replace the
final layer with the closed-form European price one layer before expiry — the
Broadie-Detemple smoothing — which integrates the kink exactly instead of
sampling it. The error that remains is smooth, and :func:`richardson` then does
what it is supposed to do. The two are separable and both are exposed, but the
default is smoothing on, because a lattice without it is worse in every measured
case and cheaper in none.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from .bsm import Inputs, OptionType, intrinsic, price

__all__ = [
    "Exercise",
    "Lattice",
    "LatticePrice",
    "boundary",
    "min_steps",
    "price_lattice",
    "richardson",
]


class Lattice(str, Enum):
    """Which discretisation of the underlying to build."""

    CRR = "crr"
    """Cox-Ross-Rubinstein: symmetric log-spacing, drift in the probability."""

    JARROW_RUDD = "jarrow-rudd"
    """Jarrow-Rudd: equal probabilities, drift in the spacing."""

    TRINOMIAL = "trinomial"
    """Boyle's trinomial: three branches, with the middle node unchanged."""

    @property
    def branches(self) -> int:
        """How many children a node has."""
        return 3 if self is Lattice.TRINOMIAL else 2


class Exercise(str, Enum):
    """When the holder may exercise."""

    EUROPEAN = "european"
    """At expiry only."""

    AMERICAN = "american"
    """At any layer of the lattice, expiry included."""


@dataclass(frozen=True, slots=True)
class LatticePrice:
    """A lattice valuation, with the grid it was produced on.

    Attributes:
        value: The option price.
        steps: Number of layers used.
        lattice: Which construction produced it.
        exercise: Which exercise right was priced.
        early_exercise_premium: ``value`` less the European value on the *same*
            lattice. Differencing on one grid rather than against the closed form
            cancels the discretisation error to leading order, so this is a far
            cleaner read on the value of the early-exercise right than
            subtracting :func:`~moneyness.bsm.price` would be. Zero for a
            European valuation.
    """

    value: float
    steps: int
    lattice: Lattice
    exercise: Exercise
    early_exercise_premium: float


def min_steps(inputs: Inputs, lattice: Lattice) -> int:
    """Fewest layers for which the branch probabilities stay in ``[0, 1]``.

    The condition is the same in spirit for all three constructions: over one
    layer the deterministic drift must not outrun the stochastic spread. For
    Cox-Ross-Rubinstein the up-move must strictly exceed the forward growth,
    ``e^{v sqrt(dt)} > e^{b dt}``, which rearranges to ``dt < (v / b)^2`` and so
    to a lower bound ``n > T b^2 / v^2`` on the layer count.

    The trinomial layer spans ``v sqrt(2 dt)`` rather than ``v sqrt(dt)``, which
    relaxes the bound by a factor of two, but it is also the construction whose
    probabilities are squares of differences and therefore vanish fastest near
    the limit. The same bound is used for it, which is conservative in the right
    direction.

    Jarrow-Rudd puts the drift into the spacing and fixes both probabilities at
    one half, so it has no such constraint and the bound is 1.

    Returns:
        The smallest usable layer count, always at least 1.
    """
    if lattice is Lattice.JARROW_RUDD:
        return 1
    if inputs.vol == 0.0 or inputs.b == 0.0 or inputs.time == 0.0:
        return 1
    bound = inputs.time * inputs.b**2 / inputs.vol**2
    return max(1, math.floor(bound) + 1)


@dataclass(frozen=True, slots=True)
class _Layer:
    """The one-layer transition: node spacing and branch probabilities."""

    up: float
    down: float
    p_up: float
    p_mid: float
    p_down: float
    discount: float
    spacing: float
    """Ratio between neighbouring nodes *within* a layer.

    Not the same as the up-move. A binomial node at index ``j`` of layer ``n``
    is ``S u^j d^{n-j}``, so neighbours differ by ``u / d``. A trinomial layer
    sits on the grid ``S u^k`` for ``-n <= k <= n``, so neighbours differ by
    ``u`` alone. Conflating the two puts the trinomial nodes on a grid twice as
    coarse as the one its probabilities were derived for.
    """


def _layer(inputs: Inputs, lattice: Lattice, steps: int) -> _Layer:
    """Build the one-layer transition, or refuse if it is not a distribution."""
    dt = inputs.time / steps
    discount = math.exp(-inputs.rate * dt)
    growth = math.exp(inputs.b * dt)

    if lattice is Lattice.CRR:
        up = math.exp(inputs.vol * math.sqrt(dt))
        down = 1.0 / up
        p_up = (growth - down) / (up - down)
        p_mid = 0.0
        p_down = 1.0 - p_up
    elif lattice is Lattice.JARROW_RUDD:
        drift = (inputs.b - 0.5 * inputs.vol**2) * dt
        spread = inputs.vol * math.sqrt(dt)
        up = math.exp(drift + spread)
        down = math.exp(drift - spread)
        p_up = 0.5
        p_mid = 0.0
        p_down = 0.5
    else:
        half = math.exp(inputs.vol * math.sqrt(0.5 * dt))
        up = half * half
        down = 1.0 / up
        mid_growth = math.exp(0.5 * inputs.b * dt)
        spread = half - 1.0 / half
        p_up = ((mid_growth - 1.0 / half) / spread) ** 2
        p_down = ((half - mid_growth) / spread) ** 2
        p_mid = 1.0 - p_up - p_down

    spacing = up if lattice is Lattice.TRINOMIAL else up / down

    for name, p in (("up", p_up), ("middle", p_mid), ("down", p_down)):
        if not -1e-12 <= p <= 1.0 + 1e-12:
            raise ValueError(
                f"{lattice.value} lattice with {steps} layers gives a {name} probability "
                f"of {p:.6g}, which is not a probability; the drift outruns the spread "
                f"over one layer. Use at least {min_steps(inputs, lattice)} layers."
            )
    return _Layer(up, down, p_up, p_mid, p_down, discount, spacing)


def _terminal_prices(inputs: Inputs, layer: _Layer, steps: int, branches: int) -> list[float]:
    """Underlying prices on the final layer, lowest node first.

    Built by repeated multiplication from the lowest node rather than by raising
    the spacing to a power at each node: on a recombining lattice the whole layer
    is one geometric progression, so the sequential form costs a single multiply
    per node, where per-node exponentiation would let each node round away from
    that progression independently.
    """
    span = steps if branches == 2 else 2 * steps
    prices = [0.0] * (span + 1)
    node = inputs.spot * layer.down**steps
    for i in range(span + 1):
        prices[i] = node
        node *= layer.spacing
    return prices


def _smoothed_layer(
    inputs: Inputs, option: OptionType, prices: list[float], dt: float
) -> list[float]:
    """European values one layer before expiry, in closed form.

    This is the Broadie-Detemple smoothing. The terminal payoff has a kink at the
    strike, and a lattice can only sample it at nodes; whether a node lands near
    the kink or far from it changes the answer by ``O(1/n)`` and changes
    erratically with ``n``. Evaluating the layer before expiry in closed form
    integrates across the kink exactly, so the kink stops contributing error at
    all and what remains is smooth enough to extrapolate.
    """
    return [
        price(
            Inputs(node, inputs.strike, dt, inputs.rate, inputs.vol, carry=inputs.carry),
            option,
        )
        for node in prices
    ]


def price_lattice(
    inputs: Inputs,
    option: OptionType,
    *,
    steps: int = 512,
    lattice: Lattice = Lattice.CRR,
    exercise: Exercise = Exercise.AMERICAN,
    smooth: bool = True,
) -> LatticePrice:
    """Price on a lattice, with or without the right to exercise early.

    The backward induction carries two values per node at once: the price under
    the requested exercise right, and the European price on the same grid. The
    second costs one extra multiply-add per node and is what makes
    :attr:`LatticePrice.early_exercise_premium` meaningful, because both sides of
    the difference then carry the same discretisation error.

    Args:
        inputs: The option and its market.
        option: Call or put.
        steps: Number of layers. Must be positive and at least
            :func:`min_steps` for the chosen construction.
        lattice: Which construction to build.
        exercise: European or American.
        smooth: Apply Broadie-Detemple smoothing, valuing the layer before
            expiry in closed form rather than rolling back from the kinked
            terminal payoff. On by default: it costs one closed-form evaluation
            per node of a single layer, and it is what makes the remaining error
            smooth enough for :func:`richardson` to help rather than hurt. Turn
            it off to see the raw lattice.

    Raises:
        ValueError: if ``steps`` is not positive, or if the resulting layer has a
            branch probability outside ``[0, 1]``.
    """
    if steps < 1:
        raise ValueError(f"steps must be positive, got {steps}")

    if inputs.is_degenerate:
        value = intrinsic(inputs, option)
        if exercise is Exercise.AMERICAN:
            value = max(value, option.sign * (inputs.spot - inputs.strike), 0.0)
        return LatticePrice(value, steps, lattice, exercise, 0.0)

    layer = _layer(inputs, lattice, steps)
    branches = lattice.branches
    sign = option.sign
    strike = inputs.strike
    american = exercise is Exercise.AMERICAN
    dt = inputs.time / steps

    if smooth:
        # Start one layer early, with that layer valued in closed form.
        start = steps - 1
        prices = _terminal_prices(inputs, layer, start, branches)
        european = _smoothed_layer(inputs, option, prices, dt)
        if american:
            values = [max(v, sign * (s - strike)) for v, s in zip(european, prices, strict=True)]
        else:
            values = list(european)
    else:
        start = steps
        prices = _terminal_prices(inputs, layer, steps, branches)
        values = [max(sign * (s - strike), 0.0) for s in prices]
        european = list(values)

    pu, pm, pd, df = layer.p_up, layer.p_mid, layer.p_down, layer.discount
    shrink = 1 if branches == 2 else 2
    ratio = layer.spacing

    for step in range(start - 1, -1, -1):
        width = (step if branches == 2 else 2 * step) + 1
        node = inputs.spot * layer.down**step
        for i in range(width):
            if branches == 2:
                held = df * (pu * values[i + 1] + pd * values[i])
                held_eu = df * (pu * european[i + 1] + pd * european[i])
            else:
                held = df * (pu * values[i + 2] + pm * values[i + 1] + pd * values[i])
                held_eu = df * (
                    pu * european[i + 2] + pm * european[i + 1] + pd * european[i]
                )
            european[i] = held_eu
            if american:
                exercise_now = sign * (node - strike)
                values[i] = held if held > exercise_now else exercise_now
            else:
                values[i] = held
            node *= ratio
        del values[width : width + shrink]
        del european[width : width + shrink]

    premium = values[0] - european[0] if american else 0.0
    return LatticePrice(values[0], steps, lattice, exercise, premium)


def richardson(
    inputs: Inputs,
    option: OptionType,
    *,
    steps: int = 512,
    lattice: Lattice = Lattice.CRR,
    exercise: Exercise = Exercise.AMERICAN,
    smooth: bool = True,
) -> float:
    """Two-point Richardson extrapolation in the layer count.

    If the error were a smooth ``c / n``, two prices at ``n`` and ``2n`` layers
    would satisfy ``P(n) = P + c/n`` and ``P(2n) = P + c/2n``, and the
    combination ``2 P(2n) - P(n)`` would cancel ``c`` exactly.

    On a raw lattice the premise is false and so is the conclusion. The error
    carries a large component that oscillates with ``n`` rather than decaying
    smoothly, and this combination — which subtracts one price from twice
    another — amplifies it. Measured on a European call over a range of layer
    counts, extrapolating the raw lattice roughly *doubles* the error instead of
    quartering it.

    With ``smooth=True`` the premise holds, because the smoothing removes the
    non-smooth term at its source, and the extrapolation delivers. That is why
    smoothing defaults on here and why turning it off is a documented way to
    make this function worse.

    Args:
        steps: The *coarse* layer count. The fine lattice uses twice as many.
        smooth: Passed through to :func:`price_lattice`. Leave it on unless the
            point is to observe the failure described above.
    """
    coarse = price_lattice(
        inputs, option, steps=steps, lattice=lattice, exercise=exercise, smooth=smooth
    ).value
    fine = price_lattice(
        inputs, option, steps=2 * steps, lattice=lattice, exercise=exercise, smooth=smooth
    ).value
    return 2.0 * fine - coarse


def boundary(
    inputs: Inputs,
    option: OptionType,
    *,
    steps: int = 512,
    lattice: Lattice = Lattice.CRR,
    smooth: bool = True,
) -> list[tuple[float, float]]:
    """The early-exercise boundary, read off the lattice as it unwinds.

    At each layer the holder of an American put exercises at low underlying
    prices and holds at high ones, and the critical price separating the two is
    the boundary. The value function crosses its own intrinsic value exactly
    once per layer, so the boundary is found by taking the extreme node at which
    exercise is still optimal: the highest such node for a put, the lowest for a
    call.

    The boundary is reported on the node grid, so its resolution is the node
    spacing and it does not converge faster than that. It is a picture of where
    the exercise region lies, not a high-precision curve; for the latter the
    boundary would be interpolated between the straddling nodes, which is a
    refinement this does not attempt.

    One consequence is worth stating plainly, because it looks like a bug and is
    not. The true boundary is monotone in time, but the *binomial* read-out is
    not: it alternates between two values one node apart. A Cox-Ross-Rubinstein
    layer ``n`` holds the nodes ``S u^{2j-n}``, so the parity of the exponent
    flips with every layer and consecutive layers are sampling two interleaved
    grids. Each parity subsequence is monotone; their interleaving is not. The
    trinomial layer holds every integer exponent ``S u^k`` at every layer, has no
    such parity, and does come out monotone — which is the construction to use if
    the boundary itself is the object of interest rather than the price.

    Returns:
        ``(time, spot)`` pairs, ordered from now to expiry, holding one entry per
        layer at which any node exercises. Layers where exercise is nowhere
        optimal are omitted, so an American call under ``b >= r`` — which is
        never exercised early — returns only the expiry point.
    """
    if steps < 1:
        raise ValueError(f"steps must be positive, got {steps}")
    if inputs.is_degenerate:
        return []

    layer = _layer(inputs, lattice, steps)
    branches = lattice.branches
    sign = option.sign
    strike = inputs.strike
    dt = inputs.time / steps

    found: list[tuple[float, float]] = []
    if smooth:
        start = steps - 1
        prices = _terminal_prices(inputs, layer, start, branches)
        held_layer = _smoothed_layer(inputs, option, prices, dt)
        values = []
        first: float | None = None
        for value, node in zip(held_layer, prices, strict=True):
            exercise_now = sign * (node - strike)
            if exercise_now > value:
                values.append(exercise_now)
                if first is None or option is OptionType.PUT:
                    first = node
            else:
                values.append(value)
        if first is not None:
            found.append((start * dt, first))
    else:
        start = steps
        prices = _terminal_prices(inputs, layer, steps, branches)
        values = [max(sign * (s - strike), 0.0) for s in prices]

    pu, pm, pd, df = layer.p_up, layer.p_mid, layer.p_down, layer.discount
    shrink = 1 if branches == 2 else 2
    ratio = layer.spacing

    for step in range(start - 1, -1, -1):
        width = (step if branches == 2 else 2 * step) + 1
        node = inputs.spot * layer.down**step
        critical: float | None = None
        for i in range(width):
            if branches == 2:
                held = df * (pu * values[i + 1] + pd * values[i])
            else:
                held = df * (pu * values[i + 2] + pm * values[i + 1] + pd * values[i])
            exercise_now = sign * (node - strike)
            if exercise_now > held:
                values[i] = exercise_now
                # A put exercises below the boundary, so the last node to satisfy
                # this on an ascending sweep is the boundary; a call exercises
                # above it, so the first one is.
                if critical is None or option is OptionType.PUT:
                    critical = node
            else:
                values[i] = held
            node *= ratio
        del values[width : width + shrink]
        if critical is not None:
            found.append((step * dt, critical))

    found.reverse()
    # At expiry the exercise region is exactly the in-the-money region, so the
    # boundary is the strike regardless of the grid. Appending it rather than
    # inferring it from the terminal layer keeps the endpoint free of node-grid
    # error.
    found.append((inputs.time, strike))
    return found
