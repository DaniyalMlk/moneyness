"""Average-price options: two inequalities, one exact identity, one measured error.

The arithmetic average has no closed form, so nothing here can be checked against
a formula for the thing itself. What it can be checked against:

**Inequalities that hold path by path.** AM-GM puts the geometric price below an
arithmetic call and *above* an arithmetic put — which side it falls on depends on
the payoff, and getting that backwards is the easiest mistake available here.
Convexity puts the average of ordinary options above either. These are statements
about the payoff, not approximations, so a violation is a defect and not a
tolerance question.

**One exact identity.** Put-call parity for the average needs only the first
moment, which is a finite sum of forwards. It holds for the simulation, for
Curran's bound and for the moment-matched price alike, and it catches a sign error
in any of them.

**The existing simulation**, which shares no code with any of this and comes with
a standard error, so agreement can be stated in units of that error rather than
as a tolerance someone chose.

**Measurement, for the approximation.** Moment matching's error changes sign with
moneyness, so it cannot be quoted as a tolerance at all; the tests pin the
direction and the size separately on each side of the money.
"""

from __future__ import annotations

import math
from itertools import pairwise

import pytest

from moneyness.asian import (
    AsianBounds,
    average_moments,
    curran,
    monitoring_times,
    parity_difference,
    price_bounds,
    turnbull_wakeman,
)
from moneyness.bsm import Inputs, OptionType, price
from moneyness.monte_carlo import Settings, asian, geometric_asian

STEPS = 12


def market(
    spot: float = 100.0,
    strike: float = 100.0,
    time: float = 1.0,
    rate: float = 0.05,
    vol: float = 0.20,
    carry: float | None = None,
) -> Inputs:
    return Inputs(
        spot=spot, strike=strike, time=time, rate=rate, vol=vol, carry=carry
    )


# A grid wide enough to include the cases where moment matching fails.
GRID = [
    (vol, strike)
    for vol in (0.05, 0.10, 0.20, 0.40, 0.80)
    for strike in (80.0, 100.0, 120.0)
]


# -- the monitoring convention -------------------------------------------------


def test_monitoring_dates_exclude_the_spot_and_include_the_expiry() -> None:
    """Stated once so that every formula here and in monte_carlo agrees about it.

    "The average of the closing prices on the last m business days" means the
    expiry is one of the fixings and today is not, which is also the convention
    ``geometric_asian`` already uses. An off-by-one here would shift every price
    in the module by a fixing's worth of carry.
    """
    times = monitoring_times(1.0, 4)
    assert times == (0.25, 0.5, 0.75, 1.0)
    assert monitoring_times(2.0, 1) == (2.0,)
    assert len(monitoring_times(1.0, 250)) == 250


@pytest.mark.parametrize(
    "time,steps", [(0.0, 4), (-1.0, 4), (1.0, 0), (1.0, -1)]
)
def test_monitoring_dates_refuse_what_cannot_be_averaged(time: float, steps: int) -> None:
    with pytest.raises(ValueError):
        monitoring_times(time, steps)


# -- the moments, which are exact ----------------------------------------------


@pytest.mark.parametrize("vol", [0.05, 0.20, 0.80])
@pytest.mark.parametrize("carry", [None, 0.0, -0.03, 0.08])
def test_the_first_moment_is_the_average_of_forwards(
    vol: float, carry: float | None
) -> None:
    """``E[A]`` needs no distributional assumption at all, only ``E[S_t]``."""
    inputs = market(vol=vol, carry=carry)
    moments = average_moments(inputs, STEPS)
    expected = inputs.spot * math.fsum(
        math.exp(inputs.b * one) for one in monitoring_times(inputs.time, STEPS)
    ) / STEPS
    assert moments.first == pytest.approx(expected, rel=1e-14)


def test_one_fixing_reduces_to_the_terminal_price() -> None:
    """A single fixing at expiry is not an average, and every formula must know it.

    The strongest reduction available: with ``m = 1`` the "average" is ``S_T``, so
    the moments are the lognormal's own, moment matching is exact rather than
    approximate, and the Asian price is the vanilla one.
    """
    inputs = market()
    moments = average_moments(inputs, 1)
    assert moments.first == pytest.approx(inputs.spot * math.exp(inputs.b), rel=1e-14)
    assert moments.second == pytest.approx(
        inputs.spot**2 * math.exp(2.0 * inputs.b + inputs.vol**2), rel=1e-14
    )
    assert moments.log_variance == pytest.approx(inputs.vol**2 * inputs.time, rel=1e-14)
    for option in OptionType:
        vanilla = price(inputs, option)
        assert turnbull_wakeman(inputs, option, 1) == pytest.approx(vanilla, rel=1e-12)
        assert geometric_asian(inputs, option, 1) == pytest.approx(vanilla, rel=1e-12)
        assert curran(inputs, option, 1) == pytest.approx(vanilla, rel=1e-6)


def test_the_second_moment_exceeds_the_square_of_the_first() -> None:
    """Jensen, so the matched log-variance is non-negative and the fit is real."""
    for vol, _ in GRID:
        moments = average_moments(market(vol=vol), STEPS)
        assert moments.second > moments.first**2
        assert moments.variance > 0.0
        assert moments.log_variance > 0.0
        assert moments.effective_vol == pytest.approx(
            math.sqrt(moments.log_variance), rel=1e-15
        )


def test_averaging_destroys_variance_and_the_moments_say_how_much() -> None:
    """The reason an Asian is cheaper than a vanilla, as a number.

    Twelve monthly fixings leave the average with a matched log-variance of
    0.01526 against the terminal price's 0.04 at 20% volatility over a year — 38%
    of it, so an effective volatility of 12.35%.
    """
    inputs = market()
    monthly = average_moments(inputs, 12).log_variance
    terminal = inputs.vol**2 * inputs.time
    assert monthly == pytest.approx(0.01526, abs=5e-6)
    assert monthly / terminal == pytest.approx(0.3814, abs=1e-3)
    assert math.sqrt(monthly / inputs.time) == pytest.approx(0.1235, abs=1e-4)
    ratios = [
        average_moments(inputs, steps).log_variance / terminal
        for steps in (12, 60, 300, 1500)
    ]
    for coarser, finer in pairwise(ratios):
        assert finer < coarser


def test_dense_fixings_approach_a_third_at_the_rate_predicted() -> None:
    """Against a derived prediction, not against a limit it is heading towards.

    The continuous result is ``v**2 T / 3``, and the discrete ratio approaches it
    as ``1/3 + 1/(2m)`` — the geometric average's own ratio ``(m+1)(2m+1)/(6m**2)``
    expanded, which the arithmetic one shares at small volatility. At three
    thousand fixings that predicts 0.333500 and the measurement is **0.333503**.
    Asserting "tends to a third" instead would pass on a formula with the wrong
    first-order term in it.

    The textbook third is a *zero-carry* statement, which is the second thing this
    pins down. At a 5% cost of carry the dense limit sits at 0.3377 rather than
    0.3333, because ``exp(b t)`` weights the later and more variable part of the
    path more heavily. An earlier version of this test asserted a third with a
    carry of 5% in force and failed by eight times its own tolerance.
    """
    for vol in (0.01, 0.20):
        inputs = market(vol=vol, carry=0.0)
        terminal = vol * vol * inputs.time
        for steps in (300, 3000):
            ratio = average_moments(inputs, steps).log_variance / terminal
            predicted = 1.0 / 3.0 + 1.0 / (2.0 * steps)
            assert ratio == pytest.approx(predicted, abs=2e-3 * vol / 0.01 * 0.01 + 2e-3)
    precise = average_moments(market(vol=0.01, carry=0.0), 3000).log_variance / 1e-4
    assert precise == pytest.approx(1.0 / 3.0 + 1.0 / 6000.0, abs=5e-6)
    # Carry shifts the limit, and by a measured amount.
    with_carry = average_moments(market(vol=0.01, carry=0.05), 3000).log_variance / 1e-4
    assert with_carry - precise == pytest.approx(0.00418, abs=5e-5)


@pytest.mark.parametrize(
    "inputs,steps",
    [
        (market(time=0.0), 12),
        (market(spot=0.0), 12),
        (market(vol=0.0), 12),
        (market(), 0),
    ],
)
def test_moments_refuse_a_degenerate_average(inputs: Inputs, steps: int) -> None:
    with pytest.raises(ValueError):
        average_moments(inputs, steps)


def test_the_zero_volatility_refusal_says_what_to_do_instead() -> None:
    with pytest.raises(ValueError, match="worth its intrinsic value"):
        average_moments(market(vol=0.0), STEPS)


# -- parity, which is exact ----------------------------------------------------


@pytest.mark.parametrize("vol,strike", GRID)
def test_parity_holds_for_every_route(vol: float, strike: float) -> None:
    """Needs only the first moment, so it holds without a model of the average.

    Which makes it the one check that applies to the approximation, the bound and
    the simulation alike — and the one that would catch a call and a put built
    from inconsistent conventions.
    """
    inputs = market(vol=vol, strike=strike)
    gap = parity_difference(inputs, STEPS)
    moments = average_moments(inputs, STEPS)
    assert gap == pytest.approx(
        math.exp(-inputs.rate * inputs.time) * (moments.first - strike), rel=1e-14
    )
    matched = turnbull_wakeman(inputs, OptionType.CALL, STEPS) - turnbull_wakeman(
        inputs, OptionType.PUT, STEPS
    )
    assert matched == pytest.approx(gap, abs=1e-11)
    bounded = curran(inputs, OptionType.CALL, STEPS) - curran(
        inputs, OptionType.PUT, STEPS
    )
    assert bounded == pytest.approx(gap, abs=1e-11)


def test_parity_difference_is_not_the_bsm_residual_of_the_same_reading() -> None:
    """Two similar names for quantities that are not the same kind of thing.

    ``bsm.parity_gap`` is a residual and is zero for any correct price pair.
    This is the difference itself, which is a price. Asserting they differ keeps
    the distinction from being lost to a later refactor that merges them.
    """
    inputs = market(strike=80.0)
    assert abs(parity_difference(inputs, STEPS)) > 1.0


# -- the inequalities ----------------------------------------------------------


@pytest.mark.parametrize("vol,strike", GRID)
@pytest.mark.parametrize("option", list(OptionType))
def test_the_bounds_are_ordered(vol: float, strike: float, option: OptionType) -> None:
    bounds = price_bounds(market(vol=vol, strike=strike), option, STEPS)
    assert bounds.lower <= bounds.upper
    assert bounds.width >= 0.0
    assert bounds.lower <= bounds.midpoint <= bounds.upper
    assert bounds.brackets(bounds.midpoint)
    assert not bounds.brackets(bounds.upper + 1.0)
    assert bounds.brackets(bounds.upper + 1.0, tolerance=1.5)


@pytest.mark.parametrize("vol,strike", GRID)
def test_the_geometric_price_changes_sides_with_the_payoff(
    vol: float, strike: float
) -> None:
    """AM-GM gives an inequality on the payoff, and the payoff flips it.

    ``G <= A`` path by path, so ``max(G - K, 0) <= max(A - K, 0)`` — the geometric
    call is cheaper — while ``max(K - G, 0) >= max(K - A, 0)`` — the geometric put
    is dearer. Measured at the money with 20% volatility and twelve fixings: a
    geometric call of 5.9402 under a true 6.1571, and a geometric put of 3.6517
    over a true 3.5355.

    Treating the geometric price as a lower bound for both sides would have looked
    right on every call that was tried.
    """
    inputs = market(vol=vol, strike=strike)
    call_bounds = price_bounds(inputs, OptionType.CALL, STEPS)
    put_bounds = price_bounds(inputs, OptionType.PUT, STEPS)
    # The geometric call is below the arithmetic one, for which Curran's bound is
    # the best available witness from below -- so allow its own small gap.
    assert call_bounds.geometric <= call_bounds.lower + 1e-9
    # The geometric put is above the arithmetic one, so it *is* an upper bound,
    # and is taken as one.
    assert put_bounds.upper <= put_bounds.geometric + 1e-12
    assert put_bounds.geometric >= put_bounds.lower


def test_the_geometric_bound_is_the_tighter_one_for_a_put() -> None:
    """So it is used, and the convexity bound is wide enough that it matters.

    Convexity allows the fixings to be independent; averaging destroys far more
    variance than that, so the convexity interval is nearly six times wider than
    the geometric one at the money. 0.681 against 0.118.
    """
    inputs = market()
    call_bounds = price_bounds(inputs, OptionType.CALL, STEPS)
    put_bounds = price_bounds(inputs, OptionType.PUT, STEPS)
    assert put_bounds.width < 0.2
    assert call_bounds.width > 0.6
    assert call_bounds.width / put_bounds.width > 4.0


@pytest.mark.parametrize("vol,strike", GRID)
@pytest.mark.parametrize("option", list(OptionType))
def test_the_convexity_bound_is_the_average_of_vanillas(
    vol: float, strike: float, option: OptionType
) -> None:
    """Reconstructed rather than trusted, including the discounting from each expiry.

    An Asian pays at the final fixing, later than all but the last of the
    vanillas the bound is built from, so each has to be carried forward from its
    own expiry to the Asian's. Dropping that factor would make the bound too high
    and so still a bound, which is exactly why it needs checking.
    """
    inputs = market(vol=vol, strike=strike)
    expected = math.fsum(
        math.exp(-inputs.rate * (inputs.time - one))
        * price(
            Inputs(
                spot=inputs.spot,
                strike=inputs.strike,
                time=one,
                rate=inputs.rate,
                vol=inputs.vol,
                carry=inputs.b,
            ),
            option,
        )
        for one in monitoring_times(inputs.time, STEPS)
    ) / STEPS
    bounds = price_bounds(inputs, option, STEPS)
    if option is OptionType.CALL:
        assert bounds.upper == pytest.approx(expected, rel=1e-14)
    else:
        assert bounds.upper == pytest.approx(min(expected, bounds.geometric), rel=1e-14)


# -- against the simulation, which shares no code ------------------------------


@pytest.mark.parametrize("option", list(OptionType))
@pytest.mark.parametrize("vol", [0.10, 0.20, 0.40])
def test_the_bounds_bracket_the_simulation(option: OptionType, vol: float) -> None:
    """Stated in standard errors, not in a tolerance someone picked."""
    inputs = market(vol=vol)
    bounds = price_bounds(inputs, option, STEPS)
    estimate = asian(inputs, option, STEPS, Settings(paths=200_000, seed=11))
    assert bounds.brackets(estimate.value, tolerance=4.0 * estimate.standard_error)
    assert bounds.lower < estimate.value + 4.0 * estimate.standard_error


@pytest.mark.parametrize("vol", [0.10, 0.20, 0.40, 0.80])
def test_currans_bound_is_never_above_the_price(vol: float) -> None:
    """The direction is the claim a bound makes, so it is tested on its own.

    Cheap enough to run across the volatility range, which the size measurement
    below is not.
    """
    inputs = market(vol=vol)
    bound = curran(inputs, OptionType.CALL, STEPS)
    estimate = asian(inputs, OptionType.CALL, STEPS, Settings(paths=200_000, seed=5))
    assert bound < estimate.value + 3.0 * estimate.standard_error


@pytest.mark.parametrize("vol,gap", [(0.40, 3.2e-04), (0.80, 1.5e-03)])
def test_currans_shortfall_is_measured_where_it_clears_the_noise(
    vol: float, gap: float
) -> None:
    """Only where the gap is resolvable, which at low volatility it is not.

    Against two million paths the relative shortfall is 6.0e-05 at 20%
    volatility, 3.2e-04 at 40% and 1.5e-03 at 80%, roughly quadrupling per
    doubling — but in units of that run's own standard error those are 2.1, 5.1
    and 8.8. At 10% volatility the shortfall came out at **0.3 standard errors**,
    which is not a measurement of anything, and an earlier version of this test
    asserted a figure of 5.1e-06 taken from exactly that single draw.

    So the size is pinned only at 40% and 80%, where the simulation can see it,
    and the direction is tested separately across the whole range.
    """
    inputs = market(vol=vol)
    bound = curran(inputs, OptionType.CALL, STEPS)
    estimate = asian(inputs, OptionType.CALL, STEPS, Settings(paths=1_000_000, seed=5))
    shortfall = estimate.value - bound
    assert shortfall > 3.0 * estimate.standard_error
    assert shortfall / bound == pytest.approx(gap, rel=0.4)


def test_currans_bound_beats_the_geometric_one_it_has_to_beat() -> None:
    """Otherwise conditioning on the geometric average bought nothing.

    It buys a great deal: at the money with 20% volatility the geometric bound is
    0.2154 below the price and Curran's is 0.0004 below it, a factor of about 600.
    """
    inputs = market()
    bounds = price_bounds(inputs, OptionType.CALL, STEPS)
    estimate = asian(inputs, OptionType.CALL, STEPS, Settings(paths=400_000, seed=5))
    geometric_error = estimate.value - bounds.geometric
    curran_error = estimate.value - bounds.lower
    assert curran_error > 0.0
    assert geometric_error / curran_error > 100.0


def test_the_moments_reproduce_the_simulations_own_average() -> None:
    """The one check on the simulation that does not pass through an approximation.

    ``E[A]`` is a finite sum of forwards and the simulation has to find it, so
    this catches a drift convention or a monitoring-date off-by-one in either.
    """
    inputs = market()
    moments = average_moments(inputs, STEPS)
    # A zero-strike call on a non-negative average *is* the discounted average.
    at_zero = market(strike=0.0)
    estimate = asian(at_zero, OptionType.CALL, STEPS, Settings(paths=200_000, seed=19))
    discounted = math.exp(-inputs.rate * inputs.time) * moments.first
    assert estimate.value == pytest.approx(
        discounted, abs=4.0 * estimate.standard_error
    )


def test_a_zero_strike_call_is_the_discounted_average_in_closed_form() -> None:
    """No root to solve and no bound to be loose about, so Curran returns it exactly."""
    inputs = market(strike=0.0)
    moments = average_moments(inputs, STEPS)
    discounted = math.exp(-inputs.rate * inputs.time) * moments.first
    assert curran(inputs, OptionType.CALL, STEPS) == pytest.approx(discounted, rel=1e-14)
    assert curran(inputs, OptionType.PUT, STEPS) == pytest.approx(0.0, abs=1e-14)


# -- the approximation, measured on both sides of the money --------------------


@pytest.mark.parametrize(
    "vol,error",
    [
        (0.05, 6.1e-04),
        (0.10, 1.4e-03),
        (0.20, 3.0e-03),
        (0.40, 8.5e-03),
        (0.80, 3.0e-02),
    ],
)
def test_moment_matching_reads_high_at_the_money(vol: float, error: float) -> None:
    """High, and two to three times worse per doubling of the volatility.

    Not an order of magnitude per doubling, which is the guess the shape of the
    problem invites. The reference is Curran's bound, whose own shortfall is an
    order of magnitude smaller than the number being measured at every point here.
    """
    inputs = market(vol=vol)
    bound = curran(inputs, OptionType.CALL, STEPS)
    matched = turnbull_wakeman(inputs, OptionType.CALL, STEPS)
    assert matched > bound
    assert (matched - bound) / bound == pytest.approx(error, rel=0.25)


@pytest.mark.parametrize(
    "vol,shortfall", [(0.10, 0.092), (0.20, 0.034), (0.30, 0.018), (0.40, 0.0079)]
)
def test_moment_matching_breaks_the_lower_bound_out_of_the_money(
    vol: float, shortfall: float
) -> None:
    """Below a rigorous lower bound, which is a defect in the method and not noise.

    A call struck at 120 against a spot of 100 comes out **below Curran's bound**
    by 9.2% at 10% volatility, 3.4% at 20%, 1.8% at 30% and 0.79% at 40%. A sum of
    lognormals has a fatter right tail than the lognormal fitted to its first two
    moments, so the far out-of-the-money call is underpriced.

    Note which way that runs with volatility: the violation *shrinks* as the
    volatility rises, and somewhere between 40% and 60% it crosses over — at 60%
    the same option is 0.9% above the bound and at 80% it is 2.7% above. So the
    error changes sign across the strike *and* across the volatility, and the two
    are not independent. An earlier version of this test had the 40% figure as
    7.9% from misreading -7.86e-03 as a percentage; it is 0.79%, and the trend it
    implied was backwards.

    This is why no tolerance is quoted for moment matching anywhere in the module,
    and why the bounds exist — without one, nothing here could tell this price is
    wrong.
    """
    inputs = market(strike=120.0, vol=vol)
    bound = curran(inputs, OptionType.CALL, STEPS)
    matched = turnbull_wakeman(inputs, OptionType.CALL, STEPS)
    assert matched < bound
    assert (bound - matched) / bound == pytest.approx(shortfall, rel=0.2)
    assert not price_bounds(inputs, OptionType.CALL, STEPS).brackets(matched)


def test_the_out_of_the_money_violation_crosses_over_at_high_volatility() -> None:
    """Measured, because the trend at moderate volatility points the other way.

    Shrinking violations as the volatility rises invite the reading that moment
    matching gets better. It does not; it crosses. At 120 strike the relative gap
    to Curran's bound runs -9.2%, -3.4%, -1.8%, -0.79%, +0.9%, +2.7% as the
    volatility goes 10%, 20%, 30%, 40%, 60%, 80%.
    """
    gaps = []
    for vol in (0.10, 0.20, 0.30, 0.40, 0.60, 0.80):
        inputs = market(strike=120.0, vol=vol)
        bound = curran(inputs, OptionType.CALL, STEPS)
        gaps.append(
            (turnbull_wakeman(inputs, OptionType.CALL, STEPS) - bound) / bound
        )
    for lower, higher in pairwise(gaps):
        assert higher > lower
    assert gaps[0] < -0.05
    assert gaps[-1] > 0.02
    assert any(one < 0.0 for one in gaps) and any(one > 0.0 for one in gaps)


def test_matching_is_exact_as_the_volatility_vanishes() -> None:
    """The limit the method is built for, approached rather than asserted at a point."""
    errors = []
    for vol in (0.08, 0.04, 0.02, 0.01):
        inputs = market(vol=vol)
        bound = curran(inputs, OptionType.CALL, STEPS)
        matched = turnbull_wakeman(inputs, OptionType.CALL, STEPS)
        errors.append(abs(matched - bound) / bound)
    for coarser, finer in pairwise(errors):
        assert finer < coarser
    assert errors[-1] < 1e-5


# -- the shape of the answers --------------------------------------------------


@pytest.mark.parametrize("option", list(OptionType))
def test_more_fixings_cost_less_for_a_call_and_the_average_settles(
    option: OptionType,
) -> None:
    """Averaging more often destroys more variance, so a call gets cheaper.

    And the effect saturates: going from one fixing to twelve does far more than
    going from twelve to three hundred, because the variance ratio is heading for
    a third rather than to zero.
    """
    inputs = market()
    prices = [curran(inputs, option, steps) for steps in (1, 2, 4, 12, 60, 300)]
    if option is OptionType.CALL:
        for coarser, finer in pairwise(prices):
            assert finer < coarser
    assert abs(prices[-1] - prices[-2]) < abs(prices[1] - prices[0])


def test_a_dividend_paying_underlying_lowers_the_average() -> None:
    """Carry enters only through ``E[S_t]``, so it moves the moments and nothing else."""
    plain = average_moments(market(), STEPS)
    paying = average_moments(market(carry=0.0), STEPS)
    assert paying.first < plain.first
    assert curran(market(carry=0.0), OptionType.CALL, STEPS) < curran(
        market(), OptionType.CALL, STEPS
    )


def test_bounds_hold_with_a_negative_carry_and_a_long_average() -> None:
    """A combination none of the formulas was derived with in mind."""
    inputs = market(time=3.0, rate=0.02, carry=-0.05, vol=0.35, strike=90.0)
    for option in OptionType:
        bounds = price_bounds(inputs, option, 36)
        estimate = asian(inputs, option, 36, Settings(paths=100_000, seed=23))
        assert bounds.brackets(estimate.value, tolerance=4.0 * estimate.standard_error)


def test_every_price_is_finite_across_the_grid() -> None:
    """Non-finite numbers must not reach a caller, including through a bound."""
    for vol, strike in GRID:
        for option in OptionType:
            inputs = market(vol=vol, strike=strike)
            bounds: AsianBounds = price_bounds(inputs, option, STEPS)
            for value in (
                bounds.lower,
                bounds.upper,
                bounds.geometric,
                turnbull_wakeman(inputs, option, STEPS),
            ):
                assert math.isfinite(value)
                assert value >= -1e-12
            # Not a price: call minus put is negative wherever the strike is
            # above the average's forward, which is most of this grid.
            assert math.isfinite(parity_difference(inputs, STEPS))
