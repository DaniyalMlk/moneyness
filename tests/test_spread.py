"""Spread options: the reductions that are exact, and the errors that are not."""

from __future__ import annotations

import itertools
import math

import pytest

from moneyness.bsm import Inputs, OptionType, price
from moneyness.normal import norm_cdf
from moneyness.quadrature import fixed_quad
from moneyness.spread import (
    FLOOR_EPSILONS,
    AssetPair,
    SpreadBounds,
    SpreadError,
    accuracy_floor,
    critical_states,
    exchange,
    half_space_bound,
    half_space_value,
    kirk,
    spread_bounds,
    spread_price,
    truncation_bound,
    vanilla_split,
    vanilla_upper_bound,
)

BASE = AssetPair(
    forward1=100.0,
    forward2=95.0,
    time=1.0,
    rate=0.03,
    vol1=0.30,
    vol2=0.25,
    correlation=0.4,
)

CORRELATIONS = (-0.99, -0.95, -0.8, -0.5, -0.2, 0.0, 0.2, 0.5, 0.8, 0.9, 0.95, 0.99)
STRIKES = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0)


def at(correlation: float) -> AssetPair:
    """``BASE`` with another correlation."""
    return AssetPair(
        BASE.forward1,
        BASE.forward2,
        BASE.time,
        BASE.rate,
        BASE.vol1,
        BASE.vol2,
        correlation,
    )


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("forward1", -1.0),
        ("forward2", -1.0),
        ("time", -0.5),
        ("vol1", -0.1),
        ("vol2", -0.1),
    ],
)
def test_negative_inputs_rejected(field: str, value: float) -> None:
    kwargs = {
        "forward1": 100.0,
        "forward2": 95.0,
        "time": 1.0,
        "rate": 0.03,
        "vol1": 0.3,
        "vol2": 0.25,
        "correlation": 0.4,
    }
    kwargs[field] = value
    with pytest.raises(ValueError, match="non-negative"):
        AssetPair(**kwargs)


@pytest.mark.parametrize("correlation", [-1.5, 1.5, 2.0])
def test_correlation_outside_the_interval_rejected(correlation: float) -> None:
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        AssetPair(100.0, 95.0, 1.0, 0.03, 0.3, 0.25, correlation)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_inputs_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        AssetPair(100.0, 95.0, 1.0, value, 0.3, 0.25, 0.4)


def test_from_spots_applies_each_carry() -> None:
    pair = AssetPair.from_spots(100.0, 95.0, 2.0, 0.03, 0.3, 0.25, 0.4, carry1=0.05, carry2=-0.01)
    assert pair.forward1 == pytest.approx(100.0 * math.exp(0.10))
    assert pair.forward2 == pytest.approx(95.0 * math.exp(-0.02))


def test_swapped_exchanges_both_assets() -> None:
    other = BASE.swapped()
    assert (other.forward1, other.vol1) == (BASE.forward2, BASE.vol2)
    assert (other.forward2, other.vol2) == (BASE.forward1, BASE.vol1)
    assert other.correlation == BASE.correlation
    assert other.spread_total == pytest.approx(BASE.spread_total)


def test_tolerance_must_be_positive() -> None:
    with pytest.raises(ValueError, match="tolerance"):
        spread_price(BASE, 5.0, tolerance=0.0)


# --------------------------------------------------------------------------
# The three exact reductions
# --------------------------------------------------------------------------


@pytest.mark.parametrize("correlation", CORRELATIONS)
def test_zero_strike_is_margrabe(correlation: float) -> None:
    """At K = 0 the spread option is an exchange option, which is closed form.

    Nothing of the quadrature's is in Margrabe's formula, so this is the one
    check on the conditional decomposition that does not go through another
    approximation.
    """
    pair = at(correlation)
    assert spread_price(pair, 0.0) == pytest.approx(exchange(pair), rel=1e-12)


def test_no_volatility_on_the_second_asset_is_a_black_call() -> None:
    """S2 is then a constant, so the spread option is struck at F2 + K."""
    pair = AssetPair(100.0, 95.0, 1.0, 0.03, 0.30, 0.0, 0.0)
    reference = price(Inputs.on_future(100.0, 100.0, 1.0, 0.03, 0.30), OptionType.CALL)
    assert spread_price(pair, 5.0) == pytest.approx(reference, rel=1e-12)


def test_no_volatility_on_the_first_asset_is_a_black_put() -> None:
    """max(F1 - S2 - K, 0) is a put on the second asset struck at F1 - K.

    This is the reduction in which the integrand is genuinely kinked, so it
    checks the panel edges as well as the decomposition.
    """
    pair = AssetPair(100.0, 95.0, 1.0, 0.03, 0.0, 0.25, 0.0)
    reference = price(Inputs.on_future(95.0, 95.0, 1.0, 0.03, 0.25), OptionType.PUT)
    assert spread_price(pair, 5.0) == pytest.approx(reference, rel=1e-12)


@pytest.mark.parametrize("strike", [-5.0, 0.0, 5.0, 25.0])
@pytest.mark.parametrize("correlation", [-0.8, 0.0, 0.7])
def test_put_call_parity_on_the_spread(strike: float, correlation: float) -> None:
    pair = at(correlation)
    call = spread_price(pair, strike)
    put = spread_price(pair, strike, call=False)
    forward = pair.discount * (pair.forward1 - pair.forward2 - strike)
    assert call - put == pytest.approx(forward, abs=1e-11)


def test_kirk_collapses_to_margrabe_at_a_zero_strike() -> None:
    """The variance weight is one there, so the two formulas are the same one."""
    for correlation in CORRELATIONS:
        pair = at(correlation)
        assert kirk(pair, 0.0) == pytest.approx(exchange(pair), rel=1e-14)


@pytest.mark.parametrize("strike", [0.0, 5.0, 20.0])
def test_degenerate_pair_is_discounted_intrinsic(strike: float) -> None:
    flat = AssetPair(100.0, 95.0, 1.0, 0.03, 0.0, 0.0, 0.0)
    assert spread_price(flat, strike) == pytest.approx(
        flat.discount * max(100.0 - 95.0 - strike, 0.0)
    )
    expiry = AssetPair(100.0, 95.0, 0.0, 0.03, 0.3, 0.25, 0.4)
    assert spread_price(expiry, strike) == pytest.approx(max(5.0 - strike, 0.0))


def test_exchange_handles_a_worthless_or_free_second_asset() -> None:
    assert exchange(AssetPair(100.0, 0.0, 1.0, 0.03, 0.3, 0.25, 0.4)) == pytest.approx(
        math.exp(-0.03) * 100.0
    )
    assert exchange(AssetPair(0.0, 95.0, 1.0, 0.03, 0.3, 0.25, 0.4)) == 0.0
    assert exchange(AssetPair(100.0, 95.0, 1.0, 0.03, 0.0, 0.0, 0.0)) == pytest.approx(
        math.exp(-0.03) * 5.0
    )


# --------------------------------------------------------------------------
# The vanilla upper bound equals the price at the worst coupling
# --------------------------------------------------------------------------


@pytest.mark.parametrize("strike", [-5.0, 0.0, 1.0, 5.0, 20.0])
def test_vanilla_bound_is_the_price_at_perfect_negative_correlation(
    strike: float,
) -> None:
    """The bound holds for every coupling, so it is the price under the worst.

    For lognormals the counter-monotone coupling is attainable at
    ``rho = -1``, so this is an equality and not an inequality -- between two
    routes with nothing in common, one being two Black formulas and the other
    a conditional quadrature.
    """
    bound = vanilla_upper_bound(at(0.4), strike)
    worst = spread_price(at(-1.0), strike)
    assert bound == pytest.approx(worst, rel=1e-12)


@pytest.mark.parametrize("strike", [0.0, 5.0, 20.0])
def test_vanilla_bound_does_not_depend_on_the_correlation(strike: float) -> None:
    values = [vanilla_upper_bound(at(rho), strike) for rho in CORRELATIONS]
    assert max(values) - min(values) < 1e-12


@pytest.mark.parametrize("strike", [0.0, 5.0, 20.0])
def test_vanilla_bound_puts_also_equal_the_worst_coupling(strike: float) -> None:
    bound = vanilla_upper_bound(at(0.4), strike, call=False)
    worst = spread_price(at(-1.0), strike, call=False)
    assert bound == pytest.approx(worst, rel=1e-12)


def test_the_split_level_found_is_the_cheapest_one() -> None:
    """Super-replication holds at every level, so the optimum must be a minimum.

    The portfolio is built out of the library's own vanilla pricer rather than
    the module's private Black helper, so the comparison does not share code
    with what it is checking.
    """
    strike = 5.0
    pair = at(0.4)
    best = vanilla_upper_bound(pair, strike)
    for index in range(400):
        level = strike + 0.01 + 400.0 * index / 399
        portfolio = pair.discount * (
            price(
                Inputs.on_future(pair.forward1, level, pair.time, 0.0, pair.vol1),
                OptionType.CALL,
            )
            + price(
                Inputs.on_future(pair.forward2, level - strike, pair.time, 0.0, pair.vol2),
                OptionType.PUT,
            )
        )
        assert portfolio >= best - 1e-12


def test_the_split_level_solves_its_own_first_order_condition() -> None:
    """Bisection on the condition, not a search on the value.

    A direct search on the portfolio value is the imprecise side of this
    comparison: it is flat to second order at the optimum, so it cannot locate
    the level better than the square root of the tolerance it reaches.
    """
    strike = 5.0
    pair = at(0.4)
    level = vanilla_split(pair, strike)
    first = (math.log(pair.forward1 / level) - 0.5 * pair.total1**2) / pair.total1
    second = (math.log(pair.forward2 / (level - strike)) - 0.5 * pair.total2**2) / pair.total2
    assert first + second == pytest.approx(0.0, abs=1e-13)


def test_the_split_level_is_undefined_without_volatility_on_both_assets() -> None:
    """And the bracket always exists when there is some, so nothing guards it.

    The residual runs from ``+inf`` at the floor to ``-inf`` as the level
    grows whenever both total volatilities are positive, so the only case the
    bisection cannot handle is the one rejected here.
    """
    with pytest.raises(SpreadError, match="no volatility"):
        vanilla_split(AssetPair(100.0, 95.0, 1.0, 0.0, 0.0, 0.25, 0.0), 5.0)
    with pytest.raises(SpreadError, match="no volatility"):
        vanilla_split(AssetPair(100.0, 95.0, 1.0, 0.0, 0.3, 0.0, 0.0), 5.0)


# --------------------------------------------------------------------------
# The sub-replicating half-space
# --------------------------------------------------------------------------


@pytest.mark.parametrize("call", [True, False])
def test_every_half_space_is_a_lower_bound(call: bool) -> None:
    """Not just the optimised one: ``max(x, 0) >= x 1_A`` for any event A.

    This is the property that makes the search safe. A worse optimum is a
    looser bound and never a wrong one, so the optimiser cannot be the source
    of an error in the price it brackets. Swept over the whole parameter plane
    rather than only near the optimum, because the claim is about every
    half-space and not about the good ones.
    """
    strike = 5.0
    pair = at(0.4)
    exact = spread_price(pair, strike, call=call)
    best = -math.inf
    for index in range(97):
        angle = -10.0 + 20.0 * index / 96
        for step in range(81):
            offset = -8.0 + 16.0 * step / 80
            value = half_space_value(pair, strike, angle, offset, call=call)
            assert value <= exact + 1e-11
            best = max(best, value)
    assert best > 0.99 * exact


@pytest.mark.parametrize("strike", [0.0, 1.0, 5.0, 20.0])
@pytest.mark.parametrize("correlation", CORRELATIONS)
def test_half_space_bound_is_below_the_price(strike: float, correlation: float) -> None:
    pair = at(correlation)
    exact = spread_price(pair, strike)
    assert half_space_bound(pair, strike) <= exact + 1e-10


@pytest.mark.parametrize("correlation", [-1.0, 1.0])
@pytest.mark.parametrize("strike", [5.0, 20.0])
def test_the_bound_is_exact_at_perfect_correlation(correlation: float, strike: float) -> None:
    """One driving variate means the exercise region really is a half-space."""
    pair = at(correlation)
    assert half_space_bound(pair, strike) == pytest.approx(spread_price(pair, strike), rel=1e-11)


def test_the_bound_gap_peaks_inside_the_correlation_range() -> None:
    """Measured, because the obvious monotone story is false.

    The gap is zero at both ends -- a single variate at either -- and rises to
    a maximum in between. A sweep stopping at 0.9 would have shown it growing
    with the correlation throughout.
    """
    strike = 5.0
    sweep = (-0.99, -0.9, -0.5, 0.0, 0.5, 0.8, 0.9, 0.95, 0.99, 0.999)
    gaps = []
    for correlation in sweep:
        pair = at(correlation)
        exact = spread_price(pair, strike)
        gaps.append((exact - half_space_bound(pair, strike)) / exact)
    peak = gaps.index(max(gaps))
    assert 0.0 < sweep[peak] < 1.0
    assert sweep[peak] == 0.95
    assert gaps[peak] == pytest.approx(1.24e-04, rel=0.15)
    assert gaps[-1] < 0.05 * gaps[peak]
    assert gaps[0] < 0.01 * gaps[peak]
    rising = gaps[: peak + 1]
    assert rising == sorted(rising)


def test_bound_gap_at_named_correlations() -> None:
    """The figures quoted in the module docstring, at K = 5."""
    expected = {
        0.0: 5.9e-06,
        0.5: 2.4e-05,
        0.8: 6.9e-05,
        0.95: 1.24e-04,
        0.999: 1.0e-06,
    }
    for correlation, gap in expected.items():
        pair = at(correlation)
        exact = spread_price(pair, 5.0)
        measured = (exact - half_space_bound(pair, 5.0)) / exact
        assert measured == pytest.approx(gap, rel=0.1)


# --------------------------------------------------------------------------
# Kirk, measured rather than asserted to be small
# --------------------------------------------------------------------------


def test_kirks_error_changes_sign_across_the_grid() -> None:
    """So no single tolerance describes it, and no direction can be assumed."""
    signs = set()
    for correlation, strike in itertools.product((0.9, 0.8, 0.0, -0.8), (1.0, 5.0, 20.0)):
        pair = at(correlation)
        exact = spread_price(pair, strike)
        signs.add(math.copysign(1.0, kirk(pair, strike) - exact))
    assert signs == {-1.0, 1.0}


@pytest.mark.parametrize(
    ("correlation", "strike", "relative"),
    [
        (0.9, 1.0, -1.45e-05),
        (0.8, 5.0, -6.85e-05),
        (0.9, 20.0, +1.49e-02),
        (-0.8, 20.0, +5.48e-03),
        (0.0, 5.0, +2.08e-04),
    ],
)
def test_kirks_error_at_named_points(correlation: float, strike: float, relative: float) -> None:
    pair = at(correlation)
    exact = spread_price(pair, strike)
    assert (kirk(pair, strike) - exact) / exact == pytest.approx(relative, rel=0.05)


@pytest.mark.parametrize("correlation", [-0.99, -0.95, -0.8, -0.5, -0.2, 0.0])
def test_kirks_error_grows_with_the_strike_where_its_sign_is_fixed(
    correlation: float,
) -> None:
    """At every non-positive correlation Kirk reads high at every strike.

    The error then has one sign across the range and its size is monotone in
    the strike, which is the only monotonicity here that survives measurement.
    """
    pair = at(correlation)
    errors = []
    for strike in STRIKES:
        exact = spread_price(pair, strike)
        error = kirk(pair, strike) - exact
        assert error > 0.0
        errors.append(error / exact)
    assert errors == sorted(errors)


def test_the_sign_change_breaks_that_monotonicity_where_it_lands() -> None:
    """Asserting it at every correlation would have been wrong.

    Where the error crosses zero inside the strike range its magnitude dips
    through the crossing, so the ordering fails -- at two of the twelve
    correlations swept, and at neither of the ones a sweep of negative
    correlations would have reached.
    """
    broken = []
    for correlation in CORRELATIONS:
        pair = at(correlation)
        errors = [
            abs(kirk(pair, strike) - spread_price(pair, strike)) / spread_price(pair, strike)
            for strike in STRIKES
        ]
        if errors != sorted(errors):
            broken.append(correlation)
    assert broken == [0.5, 0.95]


def test_the_bound_usually_beats_kirk_and_not_always() -> None:
    """The case for the bound is its sign, not that it is always closer.

    Measured over the whole grid: a median factor of about sixty in the
    bound's favour, and eight points out of eighty-four where Kirk is closer.
    Claiming the bound always wins would be false.
    """
    ratios = []
    for correlation, strike in itertools.product(CORRELATIONS, STRIKES):
        pair = at(correlation)
        exact = spread_price(pair, strike)
        kirk_error = abs(kirk(pair, strike) - exact) / exact
        bound_gap = (exact - half_space_bound(pair, strike)) / exact
        ratios.append(kirk_error / max(bound_gap, 1e-18))
    ratios.sort()
    median = ratios[len(ratios) // 2]
    assert median == pytest.approx(59.0, rel=0.2)
    losses = sum(1 for ratio in ratios if ratio < 1.0)
    assert losses == 8
    assert max(ratios) > 1e6


# --------------------------------------------------------------------------
# Quadrature structure: the kink, the ceiling, the floor
# --------------------------------------------------------------------------


def _uniform_panels(pair: AssetPair, strike: float, width: float, count: int) -> float:
    """The same integrand on uniform panels, with no edge at the kink."""
    shift = pair.correlation * pair.total1
    total2 = pair.total2
    conditional = pair.total1 * math.sqrt(max(1.0 - pair.correlation**2, 0.0))

    def integrand(z: float) -> float:
        second = pair.forward2 * math.exp(total2 * z - 0.5 * total2**2)
        shifted = second + strike
        first = pair.forward1 * math.exp(shift * z - 0.5 * shift**2)
        if conditional <= 0.0 or shifted <= 0.0:
            inner = max(first - shifted, 0.0)
        else:
            d1 = (math.log(first / shifted) + 0.5 * conditional**2) / conditional
            inner = first * norm_cdf(d1) - shifted * norm_cdf(d1 - conditional)
        return inner * math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)

    edges = [-width + 2.0 * width * index / count for index in range(count + 1)]
    return pair.discount * sum(
        fixed_quad(integrand, left, right, order=24) for left, right in itertools.pairwise(edges)
    )


def test_missing_the_kink_costs_eight_orders_and_does_not_converge_cleanly() -> None:
    """Uniform panels on a kinked integrand are not merely slower.

    At 24, 48, 96 and 192 panels the relative error is 9.6e-06, 1.8e-05,
    3.3e-06 and 1.2e-06 -- it rises on the first refinement, so a two-point
    convergence check would have reported the method diverging. The edge at
    the kink makes the same quadrature exact to 4.5e-14.
    """
    pair = AssetPair(100.0, 95.0, 1.0, 0.03, 0.0, 0.25, 0.0)
    strike = 5.0
    reference = price(Inputs.on_future(95.0, 95.0, 1.0, 0.03, 0.25), OptionType.PUT)
    errors = [
        abs(_uniform_panels(pair, strike, 8.0, count) - reference) / reference
        for count in (24, 48, 96, 192)
    ]
    assert errors[1] > errors[0]
    assert min(errors) > 1e-7
    assert abs(spread_price(pair, strike) - reference) / reference < 1e-12
    assert min(errors) / (abs(spread_price(pair, strike) - reference) / reference) > 1e6


def test_critical_states_finds_the_conditional_money() -> None:
    pair = AssetPair(100.0, 95.0, 1.0, 0.03, 0.0, 0.25, 0.0)
    roots = critical_states(pair, 5.0, 8.0)
    assert len(roots) == 1
    z = roots[0]
    second = 95.0 * math.exp(pair.total2 * z - 0.5 * pair.total2**2)
    assert 100.0 - second - 5.0 == pytest.approx(0.0, abs=1e-9)


def test_a_negative_strike_adds_the_branch_switch_as_an_edge() -> None:
    """Below it the conditional option is certain to be exercised."""
    pair = at(0.4)
    roots = critical_states(pair, -40.0, 9.0)
    branch = (math.log(40.0 / pair.forward2) + 0.5 * pair.total2**2) / pair.total2
    assert any(abs(root - branch) < 1e-9 for root in roots)


@pytest.mark.parametrize("strike", [-20.0, -5.0])
def test_a_deeply_negative_strike_is_the_forward(strike: float) -> None:
    """Exercise is then certain, so the call is worth the forward spread."""
    pair = at(0.4)
    deep = -1e4
    assert spread_price(pair, deep) == pytest.approx(
        pair.discount * (pair.forward1 - pair.forward2 - deep), rel=1e-12
    )
    assert spread_price(pair, strike) > pair.discount * (pair.forward1 - pair.forward2 - strike)
    assert spread_price(pair, strike, call=False) > 0.0


@pytest.mark.parametrize("call", [True, False])
def test_the_truncation_bound_actually_bounds_the_discarded_mass(call: bool) -> None:
    """Compare a narrow ceiling against a wide one, at the same panel density."""
    pair = at(0.4)
    strike = 5.0
    wide = spread_price(pair, strike, call=call, tolerance=1e-14)
    for width in (2.0, 3.0, 4.0, 5.0):
        narrow = 0.0
        count = max(8, math.ceil(2.0 * width * 3.0))
        edges = [-width + 2.0 * width * index / count for index in range(count + 1)]
        narrow = sum(
            fixed_quad(lambda z: _integrand(pair, strike, z, call=call), left, right, order=24)
            for left, right in itertools.pairwise(edges)
        )
        discarded = wide - pair.discount * narrow
        assert 0.0 <= discarded <= truncation_bound(pair, strike, width, call=call)


def _integrand(pair: AssetPair, strike: float, z: float, *, call: bool) -> float:
    shift = pair.correlation * pair.total1
    total2 = pair.total2
    conditional = pair.total1 * math.sqrt(max(1.0 - pair.correlation**2, 0.0))
    sign = 1.0 if call else -1.0
    second = pair.forward2 * math.exp(total2 * z - 0.5 * total2**2)
    shifted = second + strike
    first = pair.forward1 * math.exp(shift * z - 0.5 * shift**2)
    if shifted <= 0.0:
        inner = (first - shifted) if call else 0.0
    elif conditional <= 0.0:
        inner = max(sign * (first - shifted), 0.0)
    else:
        d1 = (math.log(first / shifted) + 0.5 * conditional**2) / conditional
        inner = sign * (first * norm_cdf(sign * d1) - shifted * norm_cdf(sign * (d1 - conditional)))
    return inner * math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


@pytest.mark.parametrize("call", [True, False])
def test_the_truncation_bound_underflows_before_the_ceiling(call: bool) -> None:
    """Which is what guarantees the width bisection has a bracket.

    Swept rather than guarded: the property the design rests on is that the
    bound falls below anything a caller can ask for, for every pair whose
    total volatility is well under it. It is not identically zero: at a total
    volatility of 16 the worst case over the sweep is 7.6e-144, small enough
    for any tolerance and not small enough to describe as underflow.
    """
    worst = 0.0
    for forward, vol, time in itertools.product(
        (1.0, 100.0, 1e4), (0.05, 0.5, 1.5, 3.0), (0.1, 5.0, 30.0)
    ):
        pair = AssetPair(forward, forward * 0.95, time, 0.01, vol, vol * 0.8, 0.3)
        worst = max(worst, truncation_bound(pair, 5.0, 39.0, call=call))
    assert worst < 1e-140


def test_a_total_volatility_near_the_ceiling_is_refused() -> None:
    """The measure change shifts the tail by that much, so nothing bounds it.

    The guard is reachable with a representable pair rather than being an
    assertion about inputs that cannot occur.
    """
    pair = AssetPair(100.0, 95.0, 100.0, 0.0, 5.0, 0.3, 0.9)
    assert pair.total1 == pytest.approx(50.0)
    with pytest.raises(SpreadError, match="widest ceiling"):
        spread_price(pair, 5.0)


def test_the_accuracy_floor_is_above_the_measured_error_and_below_the_price() -> None:
    """Round-off scales with the payoff, not with the answer.

    The floor is reported rather than enforced, because truncation and
    round-off are different errors and only the first answers to a tolerance.
    """
    worst = 0.0
    for forward1, vol1, correlation, time in itertools.product(
        (1.0, 100.0, 5000.0), (0.05, 0.3, 0.9), (-0.9, 0.0, 0.9), (0.08, 1.0, 10.0)
    ):
        pair = AssetPair(forward1, forward1 * 0.95, time, 0.03, vol1, 0.25, correlation)
        error = abs(spread_price(pair, 0.0) - exchange(pair))
        worst = max(worst, error / accuracy_floor(pair, 0.0))
    assert worst < 1.0
    assert worst > 0.05
    assert FLOOR_EPSILONS == 1.0e4


# --------------------------------------------------------------------------
# Bounds as an object
# --------------------------------------------------------------------------


@pytest.mark.parametrize("call", [True, False])
@pytest.mark.parametrize("strike", [-5.0, 0.0, 5.0, 20.0])
@pytest.mark.parametrize("correlation", [-0.95, -0.4, 0.0, 0.6, 0.9])
def test_bounds_contain_the_price(correlation: float, strike: float, call: bool) -> None:
    pair = at(correlation)
    floor = accuracy_floor(pair, strike, call=call)
    bounds = spread_bounds(pair, strike, call=call)
    assert bounds.lower <= bounds.upper + floor
    assert bounds.contains(spread_price(pair, strike, call=call), slack=floor)


@pytest.mark.parametrize("call", [True, False])
@pytest.mark.parametrize("correlation", [-0.9, -0.3, 0.0, 0.5, 0.9])
def test_the_bracket_closes_at_a_zero_strike(correlation: float, call: bool) -> None:
    """Both sides are the exchange option there, so the interval is a point.

    ``S1 > S2`` is already a linear condition on the driving Gaussian, so the
    best sub-replicating half-space is the exercise region itself and the
    dominating exchange option is the same price. The residual width is the
    quadrature's own round-off, which is what the accuracy floor is for.
    """
    pair = at(correlation)
    bounds = spread_bounds(pair, 0.0, call=call)
    assert abs(bounds.width) < accuracy_floor(pair, 0.0, call=call)


def test_kirk_leaves_the_rigorous_bracket_at_eighteen_of_eighty_four_points() -> None:
    """The approximation desks quote is not arbitrage-free everywhere.

    Fifteen points sit **below** the sub-replicating lower bound, all at
    positive correlation and short strikes, by up to 1.1e-03 relative at
    ``rho = 0.99, K = 2``. Three sit **above** the super-replicating upper
    bound, at near-perfect negative correlation and far strikes, by up to
    1.9e-02 relative at ``rho = -0.99, K = 40`` -- which is above the price
    under *every* coupling of the two marginals, not merely above this one.
    """
    below, above = [], []
    for correlation, strike in itertools.product(CORRELATIONS, STRIKES):
        pair = at(correlation)
        exact = spread_price(pair, strike)
        bounds = spread_bounds(pair, strike)
        value = kirk(pair, strike)
        if value < bounds.lower - 1e-12:
            below.append(((bounds.lower - value) / exact, correlation, strike))
        elif value > bounds.upper + 1e-12:
            above.append(((value - bounds.upper) / exact, correlation, strike))
    assert len(below) == 15
    assert len(above) == 3
    assert all(correlation > 0.0 for _, correlation, _ in below)
    assert all(correlation < -0.9 for _, correlation, _ in above)
    assert max(below)[0] == pytest.approx(1.13e-03, rel=0.1)
    assert max(above)[0] == pytest.approx(1.94e-02, rel=0.1)


def test_the_bracket_width_reads_the_distance_from_the_worst_coupling() -> None:
    """It is 1% of the price near rho = -1 and several multiples of it near +1.

    The upper bound carries no correlation, so it cannot be tight anywhere
    except where the correlation is the one it prices.
    """
    widths = {}
    for correlation in (-0.95, -0.5, 0.0, 0.5, 0.9):
        pair = at(correlation)
        exact = spread_price(pair, 20.0)
        widths[correlation] = spread_bounds(pair, 20.0).width / exact
    assert widths[-0.95] < 0.02
    assert widths[0.9] > 4.0
    ordered = [widths[rho] for rho in (-0.95, -0.5, 0.0, 0.5, 0.9)]
    assert ordered == sorted(ordered)


def test_bounds_width_and_containment_helpers() -> None:
    bounds = SpreadBounds(1.0, 3.0)
    assert bounds.width == 2.0
    assert bounds.contains(2.0)
    assert not bounds.contains(0.9)
    assert bounds.contains(0.9, slack=0.2)


def test_kirk_refuses_a_non_positive_shifted_forward() -> None:
    with pytest.raises(SpreadError, match="shifted forward"):
        kirk(BASE, -BASE.forward2 - 1.0)


def test_degenerate_pairs_reach_the_vanilla_bound_without_a_split() -> None:
    flat = AssetPair(100.0, 95.0, 1.0, 0.03, 0.0, 0.0, 0.0)
    assert vanilla_upper_bound(flat, 1.0) >= spread_price(flat, 1.0) - 1e-12
    one_sided = AssetPair(100.0, 95.0, 1.0, 0.03, 0.3, 0.0, 0.0)
    assert vanilla_upper_bound(one_sided, 1.0) >= spread_price(one_sided, 1.0) - 1e-12


@pytest.mark.parametrize("correlation", [-0.98, -0.37, 0.0, 0.61, 0.98])
def test_price_lies_in_its_bracket_across_the_strike_range(correlation: float) -> None:
    pair = at(correlation)
    for index in range(19):
        strike = -30.0 + 90.0 * index / 18
        exact = spread_price(pair, strike)
        bounds = spread_bounds(pair, strike)
        assert bounds.lower - 1e-9 <= exact <= bounds.upper + 1e-9
