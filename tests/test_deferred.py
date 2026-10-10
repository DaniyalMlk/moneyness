"""Choosers and compound options.

**Nothing here is checked against a number this module produced.** Four oracles
carry the file and each one is independent of the formula it validates.

*The chooser's decomposition.* Put-call parity at the decision date turns
``max(C, P)`` into a call plus a scaled put on a shifted strike, both priced by
:func:`moneyness.bsm.price`. The closed form is Rubinstein's four normal
integrals; the decomposition is two existing vanilla prices. They agree to
1.8e-14 at worst across the parameter sweep.

*The chooser's two limits.* With the decision at maturity it is a straddle; with
the decision now it is the larger of the two vanillas, to the last bit.

*The compound pairs' parity.* ``max(C - K1, 0) - max(K1 - C, 0) == C - K1``
pathwise, so the outer call less the outer put is the inner option less the
discounted premium, exactly. This validates both members of a pair at once —
four bivariate arguments and two correlation signs each — and is the reason the
put-side formulas can be trusted at all.

*A simulation sharing no derivation.* Antithetic paths to the decision date with
the inner option priced by Black-Scholes at each. Checked over five seeds per
pair with the requirement that the signed deviations take both signs, because
one seed inside its interval does not distinguish unbiased from biased by half a
standard error.

The rest is the degenerate cases, which are trades rather than errors, and the
refusals.
"""

from __future__ import annotations

import math
import random
from itertools import pairwise

import pytest

from moneyness.bsm import Inputs, OptionType, price
from moneyness.deferred import (
    BadDeferral,
    ChooserLegs,
    Compound,
    CompoundValue,
    chooser_legs,
    chooser_price,
    compound_parity_gap,
    compound_price,
    critical_spot,
    straddle_price,
)

#: Spot, strike, decision, maturity, rate, volatility, carry.
CHOOSERS = [
    (100.0, 100.0, 0.25, 1.0, 0.05, 0.20, 0.05),
    (100.0, 110.0, 0.50, 2.0, 0.03, 0.35, 0.00),
    (80.0, 100.0, 0.10, 0.5, 0.08, 0.15, 0.02),
    (120.0, 100.0, 0.75, 1.5, 0.01, 0.60, -0.03),
    (100.0, 80.0, 0.05, 3.0, 0.06, 0.25, 0.06),
    (95.0, 100.0, 1.50, 2.0, 0.00, 0.30, 0.01),
]

#: Spot, premium, inner strike, decision, inner maturity, rate, volatility, carry.
COMPOUNDS = [
    (100.0, 5.0, 100.0, 0.50, 1.0, 0.05, 0.25, 0.05),
    (90.0, 10.0, 100.0, 0.25, 2.0, 0.03, 0.40, 0.00),
    (100.0, 1.0, 120.0, 0.75, 1.5, 0.08, 0.20, 0.04),
    (100.0, 15.0, 100.0, 0.50, 1.0, 0.05, 0.25, 0.05),
    (110.0, 3.0, 100.0, 0.10, 0.5, 0.02, 0.30, -0.01),
]

BASE = Inputs(100.0, 100.0, 1.0, 0.05, 0.25, carry=0.05)


def chooser_inputs(case: tuple[float, ...]) -> tuple[Inputs, float]:
    spot, strike, decision, maturity, rate, vol, carry = case
    return Inputs(spot, strike, maturity, rate, vol, carry=carry), decision


def compound_inputs(case: tuple[float, ...]) -> tuple[Inputs, float, float]:
    spot, premium, strike, decision, maturity, rate, vol, carry = case
    return Inputs(spot, strike, maturity, rate, vol, carry=carry), premium, decision


# -- the chooser's decomposition ---------------------------------------------


@pytest.mark.parametrize("case", CHOOSERS)
def test_the_closed_form_agrees_with_the_vanilla_pair(case: tuple[float, ...]) -> None:
    """Two derivations of the same number, neither from the other."""
    inputs, decision = chooser_inputs(case)
    assert chooser_price(inputs, decision) == pytest.approx(
        chooser_legs(inputs, decision).total, abs=1e-12, rel=1e-13
    )


@pytest.mark.parametrize("case", CHOOSERS)
def test_the_put_leg_is_struck_where_the_parity_bracket_vanishes(
    case: tuple[float, ...],
) -> None:
    """The shifted strike is not a fudge: it is where ``P - C`` crosses zero.

    Checked by pricing the two vanillas at the decision date's forward of the
    shifted strike and asking for the bracket itself, which has to be zero
    there.
    """
    inputs, decision = chooser_inputs(case)
    legs = chooser_legs(inputs, decision)
    remaining = inputs.time - decision
    at_boundary = Inputs(
        legs.shifted_strike,
        inputs.strike,
        remaining,
        inputs.rate,
        inputs.vol,
        carry=inputs.carry,
    )
    gap = price(at_boundary, OptionType.PUT) - price(at_boundary, OptionType.CALL)
    assert gap == pytest.approx(0.0, abs=1e-11)


@pytest.mark.parametrize("case", CHOOSERS)
def test_a_chooser_sits_between_the_better_vanilla_and_the_straddle(
    case: tuple[float, ...],
) -> None:
    inputs, decision = chooser_inputs(case)
    value = chooser_price(inputs, decision)
    best = max(price(inputs, OptionType.CALL), price(inputs, OptionType.PUT))
    assert best <= value <= straddle_price(inputs) + 1e-12
    assert value >= price(inputs, OptionType.CALL)
    assert value >= price(inputs, OptionType.PUT)


def test_a_decision_at_maturity_is_a_straddle() -> None:
    """Choosing when the payoffs are known is taking the larger of the two,
    which is what a straddle pays."""
    assert chooser_price(BASE, BASE.time) == pytest.approx(
        straddle_price(BASE), abs=1e-13
    )
    # And the limit is approached rather than only hit at the endpoint.
    near = chooser_price(BASE, BASE.time - 1e-10)
    assert abs(near - straddle_price(BASE)) < 1e-9


def test_a_decision_now_is_the_larger_of_the_two_vanillas() -> None:
    """Exactly, not approximately: the closed form's `y` divides by sqrt(0), so
    this is a branch rather than a limit, and the branch has to be right."""
    for case in CHOOSERS:
        inputs, _ = chooser_inputs(case)
        expected = max(price(inputs, OptionType.CALL), price(inputs, OptionType.PUT))
        assert chooser_price(inputs, 0.0) == expected


def test_the_chooser_rises_with_the_decision_date() -> None:
    """More time to decide is worth more, monotonically between the two limits."""
    values = [chooser_price(BASE, decision) for decision in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)]
    for earlier, later in pairwise(values):
        assert later > earlier
    assert values[-1] == pytest.approx(straddle_price(BASE), abs=1e-13)


def test_the_decomposition_is_reported_and_adds_up() -> None:
    legs = chooser_legs(BASE, 0.5)
    assert isinstance(legs, ChooserLegs)
    assert legs.total == pytest.approx(legs.call + legs.scale * legs.put, abs=0.0)
    assert legs.call > 0.0
    assert legs.put > 0.0
    # Zero carry leaves the strike unshifted and the scale at the discount.
    flat = Inputs(100.0, 100.0, 1.0, 0.05, 0.25, carry=0.0)
    unshifted = chooser_legs(flat, 0.5)
    assert unshifted.shifted_strike == pytest.approx(100.0, abs=1e-13)
    assert unshifted.scale == pytest.approx(math.exp(-0.05 * 0.5), abs=1e-15)


# -- the compound pairs' parity ----------------------------------------------


@pytest.mark.parametrize("case", COMPOUNDS)
@pytest.mark.parametrize("inner", [OptionType.CALL, OptionType.PUT])
def test_each_compound_pair_satisfies_its_parity(
    case: tuple[float, ...], inner: OptionType
) -> None:
    """The strongest check in the file and the cheapest.

    Pathwise the two outer payoffs differ by ``C - K1`` whatever the spot does,
    so the discounted difference is the inner option less the discounted
    premium. Nothing from this module appears on the right-hand side, and the
    identity constrains both members of the pair at once.
    """
    inputs, premium, decision = compound_inputs(case)
    compound = Compound(
        outer=OptionType.CALL, inner=inner, premium=premium, decision=decision
    )
    assert compound_parity_gap(inputs, compound) == pytest.approx(0.0, abs=1e-13)


@pytest.mark.parametrize("case", COMPOUNDS)
@pytest.mark.parametrize("inner", [OptionType.CALL, OptionType.PUT])
def test_the_right_to_decline_is_the_opposite_outer_option(
    case: tuple[float, ...], inner: OptionType
) -> None:
    """Which is the parity read as a statement about the trade.

    A compound option is a forward purchase of the inner option plus an option
    on the premium, and the second term is the other member of the pair. Worth
    asserting separately from the parity because it is the sentence a reader
    takes away.
    """
    inputs, premium, decision = compound_inputs(case)
    bought = compound_price(
        inputs, Compound(OptionType.CALL, inner, premium, decision)
    )
    sold = compound_price(inputs, Compound(OptionType.PUT, inner, premium, decision))
    assert bought.deferral_value == pytest.approx(sold.value, abs=1e-13)
    assert bought.forward_purchase == pytest.approx(
        bought.inner - premium * bought.discount, abs=1e-15
    )
    assert bought.value > bought.forward_purchase


@pytest.mark.parametrize("case", COMPOUNDS)
@pytest.mark.parametrize("inner", [OptionType.CALL, OptionType.PUT])
def test_the_critical_spot_reprices_the_inner_option_to_the_premium(
    case: tuple[float, ...], inner: OptionType
) -> None:
    """The bisection's own check, which the solve does not perform on itself."""
    inputs, premium, decision = compound_inputs(case)
    result = compound_price(
        inputs, Compound(OptionType.CALL, inner, premium, decision)
    )
    assert result.critical_spot is not None
    at_boundary = Inputs(
        result.critical_spot,
        inputs.strike,
        inputs.time - decision,
        inputs.rate,
        inputs.vol,
        carry=inputs.carry,
    )
    assert price(at_boundary, inner) == pytest.approx(premium, abs=1e-9, rel=1e-10)


@pytest.mark.parametrize("case", COMPOUNDS)
def test_the_exercise_boundary_lies_the_right_side_for_each_pair(
    case: tuple[float, ...],
) -> None:
    """A call inside rises with the spot and a put inside falls, so the four
    combinations exercise on opposite sides. The property names which, and the
    probability has to agree with it."""
    inputs, premium, decision = compound_inputs(case)
    for outer in (OptionType.CALL, OptionType.PUT):
        for inner in (OptionType.CALL, OptionType.PUT):
            compound = Compound(outer, inner, premium, decision)
            result = compound_price(inputs, compound)
            assert result.critical_spot is not None
            assert result.exercise_probability is not None
            assert 0.0 < result.exercise_probability < 1.0
            opposite = compound_price(
                inputs,
                Compound(
                    OptionType.PUT if outer is OptionType.CALL else OptionType.CALL,
                    inner,
                    premium,
                    decision,
                ),
            )
            assert opposite.exercise_probability is not None
            # The two outer options are exercised on complementary events.
            assert result.exercise_probability + opposite.exercise_probability == (
                pytest.approx(1.0, abs=1e-12)
            )
            # And `exercises_above` is tested by behaviour rather than by
            # restating its definition: raising the spot has to make exercise
            # more likely exactly when the exercise region is above the
            # boundary. Lifting the spot by 1% is enough on every case here.
            higher = compound_price(
                Inputs(
                    inputs.spot * 1.01,
                    inputs.strike,
                    inputs.time,
                    inputs.rate,
                    inputs.vol,
                    carry=inputs.carry,
                ),
                compound,
            )
            assert higher.exercise_probability is not None
            rose = higher.exercise_probability > result.exercise_probability
            assert rose is compound.exercises_above


@pytest.mark.parametrize("case", COMPOUNDS)
def test_the_correlation_is_the_square_root_of_the_time_ratio(
    case: tuple[float, ...],
) -> None:
    inputs, premium, decision = compound_inputs(case)
    result = compound_price(
        inputs, Compound(OptionType.CALL, OptionType.CALL, premium, decision)
    )
    assert result.correlation == pytest.approx(
        math.sqrt(decision / inputs.time), abs=1e-15
    )
    assert 0.0 < result.correlation < 1.0


# -- the simulation ----------------------------------------------------------


def simulate(inputs: Inputs, compound: Compound, *, paths: int, seed: int) -> tuple[float, float]:
    """The compound payoff by simulation, with the inner option priced exactly.

    Only the spot at the decision date is drawn; the inner option is then a
    closed form at that spot, so the estimator has no discretisation error and
    its only disagreement with :func:`compound_price` is sampling. Antithetic
    pairs, because the payoff is monotone in the draw and the variance reduction
    is free.
    """
    generator = random.Random(seed)
    drift = (inputs.b - 0.5 * inputs.vol**2) * compound.decision
    spread = inputs.vol * math.sqrt(compound.decision)
    remaining = inputs.time - compound.decision
    discount = math.exp(-inputs.rate * compound.decision)
    total = 0.0
    squares = 0.0
    count = 0
    for _ in range(paths // 2):
        draw = generator.gauss(0.0, 1.0)
        for signed in (draw, -draw):
            spot = inputs.spot * math.exp(drift + spread * signed)
            at_decision = Inputs(
                spot, inputs.strike, remaining, inputs.rate, inputs.vol, carry=inputs.carry
            )
            value = price(at_decision, compound.inner)
            payoff = (
                max(value - compound.premium, 0.0)
                if compound.outer is OptionType.CALL
                else max(compound.premium - value, 0.0)
            ) * discount
            total += payoff
            squares += payoff * payoff
            count += 1
    mean = total / count
    return mean, math.sqrt(max(squares / count - mean * mean, 0.0) / count)


@pytest.mark.parametrize("outer", [OptionType.CALL, OptionType.PUT])
@pytest.mark.parametrize("inner", [OptionType.CALL, OptionType.PUT])
def test_a_simulation_sharing_no_derivation_agrees_over_five_seeds(
    outer: OptionType, inner: OptionType
) -> None:
    """And the signed deviations have to take both signs.

    One seed inside its interval cannot tell an unbiased estimator from one
    biased by half a standard error, so the test requires the five deviations to
    straddle zero as well as to be small.
    """
    compound = Compound(outer=outer, inner=inner, premium=5.0, decision=0.5)
    closed = compound_price(BASE, compound).value
    deviations = []
    for seed in (1, 2, 3, 4, 5):
        mean, error = simulate(BASE, compound, paths=60_000, seed=seed)
        assert error > 0.0
        deviations.append((mean - closed) / error)
    assert all(abs(one) < 3.5 for one in deviations)
    assert any(one > 0.0 for one in deviations)
    assert any(one < 0.0 for one in deviations)


# -- degenerate cases that are trades ----------------------------------------


def test_a_zero_premium_makes_the_purchase_certain() -> None:
    """So the outer call is the inner option and the outer put is worthless.

    Returned directly rather than through a bisection, since the critical spot
    is zero and the normal arguments are infinite.
    """
    free = compound_price(BASE, Compound(OptionType.CALL, OptionType.CALL, 0.0, 0.5))
    assert free.value == pytest.approx(price(BASE, OptionType.CALL), abs=0.0)
    assert free.critical_spot is None
    assert free.exercise_probability == 1.0
    sold = compound_price(BASE, Compound(OptionType.PUT, OptionType.CALL, 0.0, 0.5))
    assert sold.value == 0.0
    assert sold.exercise_probability == 0.0


def test_the_zero_premium_limit_is_approached_linearly() -> None:
    """The branch agrees with the formula it replaces, rather than only being
    defensible on its own."""
    inner = price(BASE, OptionType.CALL)
    gaps = []
    for premium in (1e-3, 1e-6, 1e-9):
        value = compound_price(
            BASE, Compound(OptionType.CALL, OptionType.CALL, premium, 0.5)
        ).value
        assert value < inner
        gaps.append((inner - value) / premium)
    # The shortfall is the premium times the exercise probability, which tends
    # to one, so the ratio is flat in the premium and close to it.
    for ratio in gaps:
        assert 0.9 < ratio < 1.0
    assert gaps[-1] == pytest.approx(gaps[0], rel=1e-3)


def test_a_premium_above_a_puts_ceiling_is_never_paid() -> None:
    """A put is bounded by its discounted strike, so a large enough premium
    settles the decision before the spot moves. The outer call is then worth
    nothing and the outer put the whole parity residual."""
    never = compound_price(BASE, Compound(OptionType.CALL, OptionType.PUT, 200.0, 0.5))
    assert never.value == 0.0
    assert never.critical_spot is None
    assert never.exercise_probability == 0.0
    always = compound_price(BASE, Compound(OptionType.PUT, OptionType.PUT, 200.0, 0.5))
    assert always.exercise_probability == 1.0
    assert always.value == pytest.approx(
        200.0 * math.exp(-0.05 * 0.5) - price(BASE, OptionType.PUT), abs=1e-12
    )
    assert always.value == pytest.approx(187.603, abs=5e-4)
    # And the parity still holds across the degenerate branch.
    assert compound_parity_gap(
        BASE, Compound(OptionType.CALL, OptionType.PUT, 200.0, 0.5)
    ) == pytest.approx(0.0, abs=1e-12)


def test_a_decision_now_settles_against_a_known_price() -> None:
    inner = price(BASE, OptionType.CALL)
    bought = compound_price(BASE, Compound(OptionType.CALL, OptionType.CALL, 5.0, 0.0))
    assert bought.value == pytest.approx(inner - 5.0, abs=1e-13)
    sold = compound_price(BASE, Compound(OptionType.PUT, OptionType.CALL, 5.0, 0.0))
    assert sold.value == 0.0
    # A premium above the known price flips which of them is worth anything.
    dear = compound_price(BASE, Compound(OptionType.CALL, OptionType.CALL, 50.0, 0.0))
    assert dear.value == 0.0
    assert compound_price(
        BASE, Compound(OptionType.PUT, OptionType.CALL, 50.0, 0.0)
    ).value == pytest.approx(50.0 - inner, abs=1e-13)


def test_a_decision_at_maturity_leaves_an_option_on_an_intrinsic() -> None:
    """An outer call reduces to one vanilla on a shifted strike."""
    on_call = compound_price(BASE, Compound(OptionType.CALL, OptionType.CALL, 5.0, 1.0))
    shifted = Inputs(100.0, 105.0, 1.0, 0.05, 0.25, carry=0.05)
    assert on_call.value == pytest.approx(price(shifted, OptionType.CALL), abs=1e-13)

    on_put = compound_price(BASE, Compound(OptionType.CALL, OptionType.PUT, 5.0, 1.0))
    lowered = Inputs(100.0, 95.0, 1.0, 0.05, 0.25, carry=0.05)
    assert on_put.value == pytest.approx(price(lowered, OptionType.PUT), abs=1e-13)


def test_an_outer_put_at_maturity_is_a_vertical_spread_not_a_vanilla() -> None:
    """The one case where the payoff algebra does not reduce to one option.

    Capping ``max(K1 - (S - K)^+, 0)`` truncates at both ends: flat at the
    premium below the inner strike and zero above ``K + K1``. Writing it as the
    single put struck at ``K + K1`` overpays by the whole flat region — 9.881
    against 2.422, four times the price.
    """
    value = compound_price(
        BASE, Compound(OptionType.PUT, OptionType.CALL, 5.0, 1.0)
    ).value
    shifted = Inputs(100.0, 105.0, 1.0, 0.05, 0.25, carry=0.05)
    spread = price(shifted, OptionType.PUT) - price(BASE, OptionType.PUT)
    assert value == pytest.approx(spread, abs=1e-13)
    assert value == pytest.approx(2.422, abs=5e-4)
    assert price(shifted, OptionType.PUT) == pytest.approx(9.881, abs=5e-4)
    # The parity holds here too, which is what rules out the single vanilla.
    assert compound_parity_gap(
        BASE, Compound(OptionType.CALL, OptionType.CALL, 5.0, 1.0)
    ) == pytest.approx(0.0, abs=1e-12)


def test_a_premium_above_a_puts_strike_at_maturity_is_never_paid() -> None:
    """The intrinsic of a put is bounded by its strike, so the boundary runs
    off the bottom and the branch has to notice rather than price a negative
    strike."""
    never = compound_price(BASE, Compound(OptionType.CALL, OptionType.PUT, 150.0, 1.0))
    assert never.value == 0.0
    always = compound_price(BASE, Compound(OptionType.PUT, OptionType.PUT, 150.0, 1.0))
    assert always.value == pytest.approx(
        150.0 * math.exp(-0.05) - price(BASE, OptionType.PUT), abs=1e-12
    )


# -- what the optionality is worth -------------------------------------------


def test_the_worked_figures_in_the_docstring() -> None:
    """Pinned, because they are the example a reader checks the module against."""
    inner = price(BASE, OptionType.CALL)
    result = compound_price(BASE, Compound(OptionType.CALL, OptionType.CALL, 5.0, 0.5))
    assert inner == pytest.approx(12.336, abs=5e-4)
    assert result.value == pytest.approx(8.381, abs=5e-4)
    assert result.forward_purchase == pytest.approx(7.459, abs=5e-4)
    assert result.deferral_value == pytest.approx(0.922, abs=5e-4)
    assert isinstance(result, CompoundValue)


def test_a_dearer_premium_buys_less_and_defers_more() -> None:
    values = []
    deferrals = []
    for premium in (1.0, 5.0, 10.0, 20.0, 40.0):
        result = compound_price(
            BASE, Compound(OptionType.CALL, OptionType.CALL, premium, 0.5)
        )
        values.append(result.value)
        deferrals.append(result.deferral_value)
    for dearer, cheaper in pairwise(values):
        assert cheaper < dearer
    for smaller, larger in pairwise(deferrals):
        assert larger > smaller


def test_the_compound_option_never_exceeds_the_inner_option() -> None:
    """An option to buy a thing is worth less than the thing, whatever the
    premium, and that is not something the formula enforces structurally."""
    for case in COMPOUNDS:
        inputs, premium, decision = compound_inputs(case)
        for inner in (OptionType.CALL, OptionType.PUT):
            result = compound_price(
                inputs, Compound(OptionType.CALL, inner, premium, decision)
            )
            assert 0.0 < result.value < result.inner


# -- refusals ----------------------------------------------------------------


def test_a_decision_after_the_maturity_is_refused() -> None:
    with pytest.raises(BadDeferral, match="after the maturity"):
        chooser_price(BASE, 1.5)
    with pytest.raises(BadDeferral, match="after the maturity"):
        compound_price(BASE, Compound(OptionType.CALL, OptionType.CALL, 5.0, 1.5))


def test_a_negative_decision_date_or_premium_is_refused() -> None:
    with pytest.raises(BadDeferral, match="non-negative year fraction"):
        chooser_price(BASE, -0.1)
    with pytest.raises(BadDeferral, match="non-negative"):
        Compound(OptionType.CALL, OptionType.CALL, -1.0, 0.5)
    with pytest.raises(BadDeferral, match="non-negative year fraction"):
        Compound(OptionType.CALL, OptionType.CALL, 5.0, -0.5)


def test_a_non_finite_parameter_is_refused() -> None:
    for bad in (math.inf, math.nan):
        with pytest.raises(BadDeferral, match="must be finite"):
            Compound(OptionType.CALL, OptionType.CALL, bad, 0.5)
        with pytest.raises(BadDeferral, match="must be finite"):
            Compound(OptionType.CALL, OptionType.CALL, 5.0, bad)


def test_a_contract_with_no_uncertainty_left_is_refused() -> None:
    """Rather than priced as a vanilla under another name, which would hide the
    fact that the decision was never deferred at all."""
    for inputs in (
        Inputs(100.0, 100.0, 1.0, 0.05, 0.0, carry=0.05),
        Inputs(0.0, 100.0, 1.0, 0.05, 0.25, carry=0.05),
        Inputs(100.0, 0.0, 1.0, 0.05, 0.25, carry=0.05),
    ):
        with pytest.raises(BadDeferral, match="Price that instead"):
            chooser_price(inputs, 0.5)
        with pytest.raises(BadDeferral, match="Price that instead"):
            compound_price(inputs, Compound(OptionType.CALL, OptionType.CALL, 5.0, 0.5))


def test_a_critical_spot_outside_the_options_range_is_refused() -> None:
    inner = Inputs(100.0, 100.0, 0.5, 0.05, 0.25, carry=0.05)
    with pytest.raises(BadDeferral, match="positive premium"):
        critical_spot(inner, 0.0, OptionType.CALL)
    with pytest.raises(BadDeferral, match="worth at most"):
        critical_spot(inner, 500.0, OptionType.PUT)
    with pytest.raises(BadDeferral, match="time left"):
        critical_spot(
            Inputs(100.0, 100.0, 0.0, 0.05, 0.25, carry=0.05), 5.0, OptionType.CALL
        )
    # A call has no ceiling, so any positive premium has a boundary.
    assert critical_spot(inner, 500.0, OptionType.CALL) > 500.0


def test_everything_returned_is_a_finite_number() -> None:
    """A price is a payload, and a non-finite one must never reach it."""
    for case in COMPOUNDS:
        inputs, premium, decision = compound_inputs(case)
        for outer in (OptionType.CALL, OptionType.PUT):
            for inner in (OptionType.CALL, OptionType.PUT):
                result = compound_price(inputs, Compound(outer, inner, premium, decision))
                for value in (
                    result.value,
                    result.inner,
                    result.correlation,
                    result.discount,
                    result.deferral_value,
                ):
                    assert math.isfinite(value)
    for choice in CHOOSERS:
        chooser, decision = chooser_inputs(choice)
        assert math.isfinite(chooser_price(chooser, decision))
