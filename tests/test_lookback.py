"""Lookback options and the law of the running extreme.

The checks here are deliberately of four different strengths, and saying which
is which is the point of this docstring.

**The structural cases owe nothing to the module's arithmetic.** ``P(M_T <= S)``
is zero and ``P(m_T >= S)`` is zero because a Brownian path leaves its starting
point in both directions immediately, and the assertions are on ``0.0`` and
``1.0`` exactly rather than within a tolerance. With no volatility or no time
the extremes are the two endpoints of a monotone path, which is also exact.

**The internal identities are transcription checks.** A fixed-strike call
struck below the running maximum differs from the one struck at that maximum by
the discounted difference of the strikes, and the fixed-strike call struck at
the spot differs from the floating-strike put by the discounted difference of
the deterministic legs. Both hold to ``0.0`` because the same function supplies
both sides. They catch a sign or a term copied wrongly in the rearrangements
and nothing at all about the tail integral they share.

**The oracle is fifty-digit integration of the tail law**, in
``reference.ref_extreme_expectation``. The layer-cake identity turns the
expectation into an integral of the tail, which ``mpmath`` evaluates
adaptively; the implementation instead integrates that tail analytically. The
two share the formula for the tail and nothing else, so agreement is evidence
about the analytic integration and about the double-precision rearrangement of
it.

**The independent check is the simulation**, because it does not use the
reflection principle at all. It draws paths, and for each step draws the true
extreme of the Brownian bridge between the two endpoints from its own
distribution function, so it is unbiased for the continuously monitored
contract. It is the only check here that would catch the tail law itself being
wrong, and it is checked across five seeds so that a near miss cannot be read
as agreement. The same simulation with the bridge switched off is a hundred
standard errors away, which is what the bias looks like and why the grid alone
proves nothing.
"""

from __future__ import annotations

import importlib
import itertools
import math

import pytest

from moneyness.bsm import Inputs, OptionType, price
from moneyness.monte_carlo import Lookback, Settings
from moneyness.monte_carlo import lookback as simulate

from .reference import (
    ref_extreme_expectation,
    ref_extreme_expectation_closed,
    ref_running_tail,
)

# The package re-exports ``monte_carlo.lookback`` under the bare name, the way
# it does for ``barrier`` and ``asian``, so ``moneyness.lookback`` is the
# simulator and the module has to be reached through the import system.
lookback = importlib.import_module("moneyness.lookback")

BOTH = (OptionType.CALL, OptionType.PUT)
STYLES = (Lookback.FIXED_STRIKE, Lookback.FLOATING_STRIKE)

#: Carries chosen to straddle the crossover between the two groupings of the
#: reflection integral, including zero exactly and both signs immediately
#: either side of it, since that is where the direct form loses its digits.
CARRIES = (
    0.5,
    0.3,
    0.15,
    0.08,
    0.05,
    0.02,
    0.01,
    1e-3,
    1e-6,
    0.0,
    -1e-6,
    -1e-3,
    -0.01,
    -0.05,
    -0.15,
    -0.3,
)
VOLS = (0.1, 0.2, 0.35, 0.6)
TIMES = (0.08, 0.5, 2.0, 5.0)

#: Below this fraction of the spot an expectation is small enough that a
#: relative comparison reports double-precision round-off in the oracle rather
#: than anything about the method. Reported rather than enforced: the
#: implementation is not asked to be accurate there, only not to be wrong.
RELATIVE_FLOOR = 1e-8


def market(spot: float, strike: float, time: float, vol: float, carry: float) -> Inputs:
    return Inputs(spot, strike, time, 0.03, vol, carry=carry)


# -- the law itself -----------------------------------------------------------


def test_maximum_cdf_is_exactly_zero_at_the_spot() -> None:
    """The maximum exceeds the starting point with probability one.

    The reflection formula at this point is a difference of two equal terms, so
    an implementation that evaluates it returns something near 1e-17 instead of
    nothing. That would be harmless on its own and is not harmless inside a
    tail integral.
    """
    for vol in VOLS:
        for time in TIMES:
            for carry in CARRIES:
                inputs = market(100.0, 100.0, time, vol, carry)
                assert lookback.maximum_cdf(inputs, 100.0) == 0.0
                assert lookback.minimum_cdf(inputs, 100.0) == 1.0


def test_extreme_distributions_are_monotone_and_reach_their_limits() -> None:
    inputs = market(100.0, 100.0, 1.0, 0.25, 0.04)
    levels = [100.0 * 1.05**k for k in range(1, 60)]
    values = [lookback.maximum_cdf(inputs, level) for level in levels]
    assert all(b >= a for a, b in itertools.pairwise(values))
    assert values[-1] == pytest.approx(1.0, abs=1e-12)
    lows = [100.0 / 1.05**k for k in range(1, 60)]
    falling = [lookback.minimum_cdf(inputs, level) for level in lows]
    assert all(b <= a for a, b in itertools.pairwise(falling))
    assert falling[-1] == pytest.approx(0.0, abs=1e-12)


def test_extreme_distributions_match_the_reference_tail() -> None:
    """The implementation's law against the same law at fifty digits.

    Not an independent check of the law — the oracle writes out the same
    reflection formula — but it does catch the double-precision arrangement of
    it, which is where the reflected term's exponent can overflow.
    """
    for carry in (0.3, 0.05, 0.0, -0.05, -0.3):
        for vol in (0.1, 0.35):
            inputs = market(100.0, 100.0, 1.5, vol, carry)
            for mult in (1.01, 1.1, 1.5, 2.5, 5.0):
                high = 100.0 * mult
                got = lookback.maximum_cdf(inputs, high)
                want = 1.0 - ref_running_tail(
                    100.0, carry, vol, 1.5, high, upper=True
                )
                assert got == pytest.approx(want, rel=1e-12, abs=1e-15)
                low = 100.0 / mult
                got = lookback.minimum_cdf(inputs, low)
                want = ref_running_tail(100.0, carry, vol, 1.5, low, upper=False)
                assert got == pytest.approx(want, rel=1e-12, abs=1e-15)


def test_extreme_distributions_reject_impossible_levels() -> None:
    inputs = market(100.0, 100.0, 1.0, 0.2, 0.05)
    with pytest.raises(lookback.LookbackError, match="level must be positive"):
        lookback.maximum_cdf(inputs, 0.0)
    with pytest.raises(lookback.LookbackError, match="level must be positive"):
        lookback.minimum_cdf(inputs, -1.0)
    with pytest.raises(lookback.LookbackError, match="spot must be positive"):
        lookback.maximum_cdf(market(0.0, 100.0, 1.0, 0.2, 0.05), 100.0)


# -- the analytic integral against fifty-digit integration --------------------


def test_tail_expectations_match_fifty_digit_integration() -> None:
    """The oracle integrates the tail; the implementation integrates it in closed form.

    Eight points rather than a grid, because the oracle's adaptive quadrature at
    fifty digits costs seconds per point. The grid lives in
    :func:`test_tail_expectations_survive_the_crossover`, where the oracle is
    the same closed form carried at fifty digits and costs nothing.
    """
    cases = (
        (0.05, 0.2, 1.0, 1.0),
        (0.05, 0.2, 1.0, 1.2),
        (0.0, 0.2, 1.0, 1.0),
        (-0.03, 0.35, 2.0, 1.1),
        (0.08, 0.15, 0.5, 1.0),
        (0.02, 0.4, 3.0, 1.3),
        (0.3, 0.1, 0.25, 1.0),
        (-1e-6, 0.2, 1.0, 1.05),
    )
    for carry, vol, time, mult in cases:
        inputs = market(100.0, 100.0, time, vol, carry)
        got = lookback.maximum_excess(inputs, 100.0 * mult)
        want = ref_extreme_expectation(
            100.0, carry, vol, time, 100.0 * mult, upper=True
        )
        assert got == pytest.approx(want, rel=1e-11)
        got = lookback.minimum_shortfall(inputs, 100.0 / mult)
        want = ref_extreme_expectation(
            100.0, carry, vol, time, 100.0 / mult, upper=False
        )
        assert got == pytest.approx(want, rel=1e-11)


def test_tail_expectations_survive_the_crossover() -> None:
    """A grid across the carry, including zero and both signs beside it.

    The direct grouping of the reflection integral loses digits as the carry
    approaches zero and the regrouped one loses them when the carry is large,
    so the sweep has to cover both sides of the switch rather than a
    neighbourhood of one.
    """
    worst = 0.0
    checked = 0
    for carry in CARRIES:
        for vol in VOLS:
            for time in TIMES:
                inputs = market(100.0, 100.0, time, vol, carry)
                for mult in (1.0, 1.1, 1.6, 2.72):
                    for upper in (True, False):
                        level = 100.0 * mult if upper else 100.0 / mult
                        want = ref_extreme_expectation_closed(
                            100.0, carry, vol, time, level, upper=upper
                        )
                        got = (
                            lookback.maximum_excess(inputs, level)
                            if upper
                            else lookback.minimum_shortfall(inputs, level)
                        )
                        if want <= RELATIVE_FLOOR * 100.0:
                            # Below the floor only the absolute size is
                            # meaningful, and it must still be small.
                            assert abs(got - want) < 1e-9
                            continue
                        checked += 1
                        worst = max(worst, abs(got - want) / want)
    assert checked > 1000
    assert worst < 1e-11, f"worst relative error {worst:.3e} over {checked} points"


def test_zero_carry_is_the_limit_of_small_carry() -> None:
    """The value at a carry of exactly zero is approached from both sides.

    The regrouped form reaches the limit rather than approaching it, so the
    check is that the limit it returns is the one the neighbourhood points
    converge to — a discontinuity of 1e-9 at zero would otherwise pass every
    other test here.
    """
    inputs = market(100.0, 100.0, 1.0, 0.2, 0.0)
    at_zero = lookback.maximum_excess(inputs, 110.0)
    for carry in (1e-4, 1e-6, 1e-8):
        for sign in (1.0, -1.0):
            near = lookback.maximum_excess(
                market(100.0, 100.0, 1.0, 0.2, sign * carry), 110.0
            )
            assert abs(near - at_zero) < 60.0 * carry


# -- degenerate and structural cases ------------------------------------------


@pytest.mark.parametrize("vol,time", [(0.0, 1.0), (0.2, 0.0), (0.0, 0.0)])
def test_degenerate_paths_have_deterministic_extremes(vol: float, time: float) -> None:
    for carry in (0.1, 0.0, -0.1):
        inputs = market(100.0, 100.0, time, vol, carry)
        terminal = 100.0 * math.exp(carry * time)
        assert lookback.expected_maximum(inputs) == pytest.approx(
            max(100.0, terminal), rel=1e-15
        )
        assert lookback.expected_minimum(inputs) == pytest.approx(
            min(100.0, terminal), rel=1e-15
        )
        assert lookback.maximum_excess(inputs, 1e6) == 0.0
        assert lookback.minimum_shortfall(inputs, 1e-6) == 0.0


def test_expected_extremes_bracket_the_spot_and_the_forward() -> None:
    """The maximum dominates both endpoints of the path, and the minimum is dominated."""
    for carry in CARRIES:
        for vol in VOLS:
            inputs = market(100.0, 100.0, 1.0, vol, carry)
            forward = 100.0 * math.exp(carry)
            high = lookback.expected_maximum(inputs)
            low = lookback.expected_minimum(inputs)
            assert high >= max(100.0, forward)
            assert low <= min(100.0, forward)
            assert high > low


def test_expected_maximum_is_the_tail_integral_at_the_spot() -> None:
    inputs = market(100.0, 100.0, 1.0, 0.3, 0.02)
    assert lookback.expected_maximum(inputs) == 100.0 + lookback.maximum_excess(
        inputs, 100.0
    )
    assert lookback.expected_minimum(inputs) == 100.0 - lookback.minimum_shortfall(
        inputs, 100.0
    )


# -- the exact rearrangements --------------------------------------------------


def test_fixed_strike_call_translates_exactly_below_the_recorded_maximum() -> None:
    """Struck below the maximum already recorded, the payoff is certain at least once.

    So the difference between two such strikes is the discounted difference of
    the strikes, with no approximation anywhere: both sides read the same
    ``maximum_excess`` call.
    """
    for recorded in (100.0, 115.0):
        inputs = market(100.0, recorded, 1.0, 0.25, 0.04)
        base = lookback.lookback_price(
            inputs, OptionType.CALL, Lookback.FIXED_STRIKE, observed=recorded
        )
        for strike in (0.5, 40.0, 90.0, recorded):
            lower = market(100.0, strike, 1.0, 0.25, 0.04)
            shifted = lookback.lookback_price(
                lower, OptionType.CALL, Lookback.FIXED_STRIKE, observed=recorded
            )
            assert shifted - base == pytest.approx(
                (recorded - strike) * inputs.discount, rel=1e-14, abs=1e-14
            )


def test_fixed_strike_put_translates_exactly_above_the_recorded_minimum() -> None:
    for recorded in (100.0, 85.0):
        inputs = market(100.0, recorded, 1.0, 0.25, 0.04)
        base = lookback.lookback_price(
            inputs, OptionType.PUT, Lookback.FIXED_STRIKE, observed=recorded
        )
        for strike in (recorded, 140.0, 400.0):
            higher = market(100.0, strike, 1.0, 0.25, 0.04)
            shifted = lookback.lookback_price(
                higher, OptionType.PUT, Lookback.FIXED_STRIKE, observed=recorded
            )
            assert shifted - base == pytest.approx(
                (strike - recorded) * inputs.discount, rel=1e-14, abs=1e-14
            )


def test_the_two_maximum_contracts_differ_by_a_forward() -> None:
    """A fixed-strike call at the spot and a floating-strike put are one contract.

    Both pay the maximum less a deterministic leg, so their difference is
    discounted and known in advance. This is the identity that replaces
    put-call parity here, and like parity in :mod:`moneyness.barrier` it is a
    transcription check: it cannot catch the tail integral being wrong, because
    both sides use the same one.
    """
    for carry in CARRIES:
        for vol in VOLS:
            inputs = market(100.0, 100.0, 1.0, vol, carry)
            call = lookback.lookback_price(
                inputs, OptionType.CALL, Lookback.FIXED_STRIKE
            )
            put = lookback.lookback_price(
                inputs, OptionType.PUT, Lookback.FLOATING_STRIKE
            )
            legs = inputs.discount * (100.0 * math.exp(carry) - 100.0)
            assert call - put == pytest.approx(legs, rel=1e-13, abs=1e-13)


def test_floating_strike_does_not_read_the_strike() -> None:
    for option in BOTH:
        base = lookback.lookback_price(
            market(100.0, 100.0, 1.0, 0.2, 0.05), option, Lookback.FLOATING_STRIKE
        )
        for strike in (0.0, 1.0, 1e6):
            other = lookback.lookback_price(
                market(100.0, strike, 1.0, 0.2, 0.05),
                option,
                Lookback.FLOATING_STRIKE,
            )
            assert other == base


# -- rigorous inequalities against the vanilla --------------------------------


def test_every_lookback_dominates_its_vanilla_pathwise() -> None:
    """Each payoff is at least the corresponding European's on every path.

    ``(M_T - K)^+ >= (S_T - K)^+`` because the maximum is at least the terminal
    value, and ``S_T - m_T >= (S_T - S)^+`` because the minimum is at most the
    spot and the difference is non-negative. These are inequalities on the
    payoffs, so they hold whatever the model does, and they are the only checks
    here that cannot be satisfied by a self-consistent wrong formula.
    """
    for carry in (0.08, 0.0, -0.08):
        for vol in VOLS:
            for strike in (70.0, 100.0, 130.0):
                inputs = market(100.0, strike, 1.0, vol, carry)
                for option in BOTH:
                    fixed = lookback.lookback_price(
                        inputs, option, Lookback.FIXED_STRIKE
                    )
                    assert fixed >= price(inputs, option)
                at_spot = market(100.0, 100.0, 1.0, vol, carry)
                for option in BOTH:
                    floating = lookback.lookback_price(
                        at_spot, option, Lookback.FLOATING_STRIKE
                    )
                    assert floating >= price(at_spot, option)
                    assert floating > 0.0


def test_lookbacks_rise_with_volatility_and_with_time() -> None:
    """Measured rather than assumed, since :mod:`moneyness.barrier` has a contract
    that does neither.

    A lookback has no cap to work against, so more volatility and more time can
    only widen the range the extreme is drawn from.
    """
    for style in STYLES:
        for option in BOTH:
            values = [
                lookback.lookback_price(
                    market(100.0, 100.0, 1.0, vol, 0.03), option, style
                )
                for vol in (0.05, 0.1, 0.2, 0.3, 0.5, 0.8)
            ]
            assert all(b > a for a, b in itertools.pairwise(values))
            over_time = [
                lookback.lookback_price(
                    market(100.0, 100.0, time, 0.25, 0.0), option, style
                )
                for time in (0.1, 0.25, 0.5, 1.0, 2.0, 5.0)
            ]
            assert all(b > a for a, b in itertools.pairwise(over_time))


def test_a_seasoned_contract_is_worth_at_least_a_fresh_one() -> None:
    """A recorded extreme already further out can only help the holder."""
    for option in BOTH:
        for style in STYLES:
            fresh = lookback.lookback_price(
                market(100.0, 100.0, 1.0, 0.25, 0.03), option, style
            )
            upper = style.reads_maximum(option)
            for step in (1.0, 1.1, 1.4, 2.0):
                recorded = 100.0 * step if upper else 100.0 / step
                seasoned = lookback.lookback_price(
                    market(100.0, 100.0, 1.0, 0.25, 0.03),
                    option,
                    style,
                    observed=recorded,
                )
                assert seasoned >= fresh - 1e-12


def test_a_recorded_extreme_on_the_wrong_side_is_refused() -> None:
    inputs = market(100.0, 100.0, 1.0, 0.2, 0.05)
    with pytest.raises(lookback.LookbackError, match="below the spot"):
        lookback.lookback_price(
            inputs, OptionType.CALL, Lookback.FIXED_STRIKE, observed=90.0
        )
    with pytest.raises(lookback.LookbackError, match="above the spot"):
        lookback.lookback_price(
            inputs, OptionType.PUT, Lookback.FIXED_STRIKE, observed=110.0
        )
    with pytest.raises(lookback.LookbackError, match="must be positive"):
        lookback.lookback_price(
            inputs, OptionType.CALL, Lookback.FIXED_STRIKE, observed=0.0
        )


def test_style_reads_the_side_it_settles_against() -> None:
    assert Lookback.FIXED_STRIKE.reads_maximum(OptionType.CALL)
    assert not Lookback.FIXED_STRIKE.reads_maximum(OptionType.PUT)
    assert not Lookback.FLOATING_STRIKE.reads_maximum(OptionType.CALL)
    assert Lookback.FLOATING_STRIKE.reads_maximum(OptionType.PUT)
    assert Lookback.FIXED_STRIKE.is_fixed
    assert not Lookback.FLOATING_STRIKE.is_fixed


# -- the simulation, which shares no mathematics with the formula -------------


def test_bridge_simulation_agrees_across_seeds() -> None:
    """Five seeds, each within two standard errors and scattering either side.

    One seed inside its interval is weak evidence: an estimator biased by half a
    standard error passes it most of the time. Requiring the signed deviations
    to take both signs is what makes this a check on the bias rather than on
    the variance.
    """
    cases = (
        (market(100.0, 120.0, 1.0, 0.2, 0.05), OptionType.CALL, Lookback.FIXED_STRIKE),
        (
            market(100.0, 100.0, 1.0, 0.2, 0.05),
            OptionType.CALL,
            Lookback.FLOATING_STRIKE,
        ),
        (market(100.0, 90.0, 1.0, 0.3, -0.02), OptionType.PUT, Lookback.FIXED_STRIKE),
    )
    for inputs, option, style in cases:
        exact = lookback.lookback_price(inputs, option, style)
        deviations = []
        for seed in (1, 2, 3, 4, 5):
            estimate = simulate(
                inputs, option, style, 40, Settings(paths=40_000, seed=seed)
            )
            deviations.append((estimate.value - exact) / estimate.standard_error)
        assert all(abs(z) < 3.0 for z in deviations), deviations
        assert min(deviations) < 0.0 < max(deviations), deviations


def test_dropping_the_bridge_is_a_bias_and_not_noise() -> None:
    """The grid-only estimator is many standard errors low, in one direction.

    An observed extreme is less extreme than the path's, so the error has a
    sign. At 40 steps it is two orders of magnitude outside the interval, which
    is why the bridge is on by default and why a simulation without it cannot
    be used to check a formula.
    """
    inputs = market(100.0, 120.0, 1.0, 0.2, 0.05)
    exact = lookback.lookback_price(inputs, OptionType.CALL, Lookback.FIXED_STRIKE)
    for seed in (1, 2, 3):
        grid = simulate(
            inputs,
            OptionType.CALL,
            Lookback.FIXED_STRIKE,
            40,
            Settings(paths=40_000, seed=seed),
            bridge=False,
        )
        assert (grid.value - exact) / grid.standard_error < -20.0


def test_discrete_monitoring_rises_towards_the_continuous_price() -> None:
    """More observations means a more extreme extreme, monotonically.

    The discretely monitored contract is a different contract, so this is not a
    convergence test on an estimator: each point is the right answer to its own
    question, and the sequence has to be increasing and bounded above by the
    continuous price.
    """
    inputs = market(100.0, 100.0, 1.0, 0.2, 0.05)
    exact = lookback.lookback_price(inputs, OptionType.CALL, Lookback.FIXED_STRIKE)
    values = [
        simulate(
            inputs,
            OptionType.CALL,
            Lookback.FIXED_STRIKE,
            steps,
            Settings(paths=60_000, seed=11),
            bridge=False,
        ).value
        for steps in (4, 12, 52, 252)
    ]
    assert all(b > a for a, b in itertools.pairwise(values))
    assert values[-1] < exact
    # And the gap at a monthly close is large enough that no caller should
    # reach for the continuous price instead: the measured figure is 17.8%.
    assert 0.10 < (exact - values[1]) / exact < 0.25


def test_simulation_refuses_what_the_formula_refuses() -> None:
    inputs = market(100.0, 100.0, 1.0, 0.2, 0.05)
    with pytest.raises(ValueError, match="steps must be at least 1"):
        simulate(inputs, OptionType.CALL, Lookback.FIXED_STRIKE, 0)
    with pytest.raises(ValueError, match="below the spot"):
        simulate(
            inputs, OptionType.CALL, Lookback.FIXED_STRIKE, 10, observed=80.0
        )
    with pytest.raises(ValueError, match="above the spot"):
        simulate(inputs, OptionType.PUT, Lookback.FIXED_STRIKE, 10, observed=120.0)
    with pytest.raises(ValueError, match="must be positive"):
        simulate(inputs, OptionType.CALL, Lookback.FIXED_STRIKE, 10, observed=-1.0)


def test_simulated_floating_payoff_is_never_negative() -> None:
    """The floating-strike payoff is non-negative path by path, so its estimate is too."""
    for option in BOTH:
        estimate = simulate(
            market(100.0, 100.0, 1.0, 0.45, -0.1),
            option,
            Lookback.FLOATING_STRIKE,
            25,
            Settings(paths=20_000, seed=3),
        )
        assert estimate.value > 0.0
