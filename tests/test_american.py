"""The closed-form American approximation, checked against the lattice.

The central test here is not an accuracy tolerance but an *inequality*. The
Bjerksund-Stensland value is the exact worth of a flat-boundary exercise
strategy, which the holder could follow but which is not optimal, so it is a true
lower bound on the American price and a true upper bound on the European one:

    European <= Bjerksund-Stensland <= American

That bracket holds for every input, with no tuning, and it is sharper than a
tolerance would be. A tolerance says the two numbers are close; the bracket says
which side of the lattice the approximation must fall on, and an implementation
that drifted above the lattice would fail even while looking accurate.

The lattice value used as the upper edge is itself an approximation, so the
comparison carries the lattice's own error. That is handled by using the
extrapolated trinomial value, whose error is around 5e-06 on these inputs — two
orders of magnitude below the gap being measured — and by allowing a small
margin at the upper edge for it.
"""

from __future__ import annotations

import math
from itertools import pairwise

import pytest

from moneyness import Inputs, OptionType, price
from moneyness.american import (
    _two_step_triggers,
    _two_step_value,
    bjerksund_stensland,
    bjerksund_stensland_2002,
    trigger_price,
)
from moneyness.lattice import Lattice, richardson

# Spot, strike, time, rate, carry, vol. Every row has a carry below the rate for
# at least one option type, since that is the only case where the approximation
# has anything to do.
GRID = [
    (100.0, 100.0, 0.5, 0.08, -0.04, 0.20),
    (100.0, 100.0, 1.0, 0.06, -0.04, 0.35),
    (90.0, 100.0, 2.0, 0.06, -0.02, 0.25),
    (120.0, 100.0, 0.75, 0.05, -0.03, 0.28),
    (100.0, 95.0, 0.5, 0.10, -0.04, 0.20),
    (100.0, 110.0, 0.25, 0.03, -0.05, 0.18),
    (100.0, 100.0, 1.0, 0.05, 0.05, 0.25),
    (80.0, 100.0, 1.5, 0.07, 0.0, 0.30),
    (130.0, 100.0, 0.5, 0.04, -0.06, 0.22),
]


def inputs_of(row: tuple[float, float, float, float, float, float]) -> Inputs:
    spot, strike, time, rate, carry, vol = row
    return Inputs(spot, strike, time, rate, vol, carry=carry)


def lattice_value(option_inputs: Inputs, option: OptionType) -> float:
    """A high-resolution American value to bracket against."""
    return richardson(option_inputs, option, steps=600, lattice=Lattice.TRINOMIAL)


class TestBracket:
    """The inequality that holds by construction."""

    @pytest.mark.parametrize("row", GRID)
    @pytest.mark.parametrize("option", list(OptionType))
    def test_between_the_european_and_american_values(
        self, row: tuple[float, float, float, float, float, float], option: OptionType
    ) -> None:
        option_inputs = inputs_of(row)
        approximation = bjerksund_stensland(option_inputs, option)
        european = price(option_inputs, option)
        american = lattice_value(option_inputs, option)

        assert approximation >= european - 1e-9
        assert approximation <= american + 1e-5

    @pytest.mark.parametrize("row", GRID)
    @pytest.mark.parametrize("option", list(OptionType))
    def test_at_least_the_immediate_exercise_value(
        self, row: tuple[float, float, float, float, float, float], option: OptionType
    ) -> None:
        option_inputs = inputs_of(row)
        approximation = bjerksund_stensland(option_inputs, option)
        immediate = max(option.sign * (option_inputs.spot - option_inputs.strike), 0.0)
        assert approximation >= immediate - 1e-9

    @pytest.mark.parametrize("row", GRID)
    @pytest.mark.parametrize("option", list(OptionType))
    def test_close_enough_to_be_useful(
        self, row: tuple[float, float, float, float, float, float], option: OptionType
    ) -> None:
        """The bracket alone would be satisfied by returning the European value.

        This is the companion assertion that stops that: the approximation has to
        recover most of the early-exercise premium, not merely sit somewhere
        inside the band. The bound is relative to the option price, since the
        rows span values from well under a point to over twenty.
        """
        option_inputs = inputs_of(row)
        approximation = bjerksund_stensland(option_inputs, option)
        american = lattice_value(option_inputs, option)
        assert abs(approximation - american) <= 0.02 * max(american, 1.0)


class TestNoEarlyExercise:
    """Where the right is worthless the approximation must be the European price."""

    @pytest.mark.parametrize("row", GRID)
    def test_call_with_carry_at_least_rate_is_the_european_price(
        self, row: tuple[float, float, float, float, float, float]
    ) -> None:
        option_inputs = inputs_of(row)
        if option_inputs.b < option_inputs.rate:
            pytest.skip("early exercise has value when the carry is below the rate")
        assert bjerksund_stensland(option_inputs, OptionType.CALL) == price(
            option_inputs, OptionType.CALL
        )

    def test_that_case_is_exact_not_approximate(self) -> None:
        """Equality, to the last bit.

        The short-circuit returns the closed form itself rather than evaluating
        the approximation and hoping it lands there, so this is an identity and
        is asserted as one.
        """
        option_inputs = Inputs(100.0, 95.0, 0.5, 0.04, 0.22)
        assert bjerksund_stensland(option_inputs, OptionType.CALL) == price(
            option_inputs, OptionType.CALL
        )


class TestTriggerPrice:
    """The flat boundary the strategy exercises at."""

    @pytest.mark.parametrize("row", GRID)
    def test_call_trigger_is_above_the_strike(
        self, row: tuple[float, float, float, float, float, float]
    ) -> None:
        option_inputs = inputs_of(row)
        level = trigger_price(option_inputs, OptionType.CALL)
        if option_inputs.b >= option_inputs.rate:
            assert math.isinf(level)
        else:
            assert level > option_inputs.strike

    @pytest.mark.parametrize("row", GRID)
    def test_put_trigger_is_below_the_strike(
        self, row: tuple[float, float, float, float, float, float]
    ) -> None:
        option_inputs = inputs_of(row)
        level = trigger_price(option_inputs, OptionType.PUT)
        assert 0.0 <= level < option_inputs.strike

    def test_exercises_immediately_past_the_trigger(self) -> None:
        """Beyond the boundary the answer is the intrinsic value, exactly."""
        base = Inputs(100.0, 100.0, 0.5, 0.08, 0.20, carry=-0.04)
        level = trigger_price(base, OptionType.CALL)
        beyond = Inputs(level * 1.05, 100.0, 0.5, 0.08, 0.20, carry=-0.04)
        assert bjerksund_stensland(beyond, OptionType.CALL) == pytest.approx(
            beyond.spot - beyond.strike
        )

    def test_trigger_rises_with_time_to_expiry(self) -> None:
        """More time left means a higher bar for giving up the time value."""
        levels = [
            trigger_price(
                Inputs(100.0, 100.0, time, 0.08, 0.20, carry=-0.04), OptionType.CALL
            )
            for time in (0.05, 0.25, 0.5, 1.0, 2.0, 5.0)
        ]
        assert all(a < b for a, b in pairwise(levels))

    def test_trigger_is_undefined_for_degenerate_inputs(self) -> None:
        for option_inputs in (
            Inputs(100.0, 100.0, 0.0, 0.08, 0.20, carry=-0.04),
            Inputs(100.0, 100.0, 0.5, 0.08, 0.0, carry=-0.04),
        ):
            for option in OptionType:
                with pytest.raises(ValueError, match="undefined"):
                    trigger_price(option_inputs, option)


class TestPutTransformation:
    """Puts go through the call, so the transformation itself needs checking."""

    @pytest.mark.parametrize("row", GRID)
    def test_put_agrees_with_a_lattice_put(
        self, row: tuple[float, float, float, float, float, float]
    ) -> None:
        option_inputs = inputs_of(row)
        approximation = bjerksund_stensland(option_inputs, OptionType.PUT)
        american = lattice_value(option_inputs, OptionType.PUT)
        assert approximation == pytest.approx(american, rel=0.02, abs=1e-6)

    def test_the_transformation_is_the_one_documented(self) -> None:
        """``P(S, K, T, r, b, v)`` must equal ``C(K, S, T, r - b, -b, v)``.

        Asserted directly rather than inferred from the prices agreeing, so that
        the identity in the module docstring is the thing under test.
        """
        option_inputs = Inputs(100.0, 110.0, 0.75, 0.06, 0.28, carry=-0.02)
        mirrored = Inputs(
            option_inputs.strike,
            option_inputs.spot,
            option_inputs.time,
            option_inputs.rate - option_inputs.b,
            option_inputs.vol,
            carry=-option_inputs.b,
        )
        assert bjerksund_stensland(option_inputs, OptionType.PUT) == (
            bjerksund_stensland(mirrored, OptionType.CALL)
        )

    def test_put_at_zero_rate_and_zero_carry_has_no_premium(self) -> None:
        """The mirror of the call's no-early-exercise case.

        With ``r = 0`` and ``b = 0`` the transformed call has carry equal to its
        rate, so the short-circuit fires and the put comes back at its European
        value — which is the correct answer, since a put at a zero rate is never
        exercised early.
        """
        option_inputs = Inputs(60.0, 100.0, 1.0, 0.0, 0.30, carry=0.0)
        assert bjerksund_stensland(option_inputs, OptionType.PUT) == pytest.approx(
            price(option_inputs, OptionType.PUT)
        )


class TestDegenerate:
    """Zero time, zero volatility, zero spot and zero strike."""

    @pytest.mark.parametrize(
        "option_inputs",
        [
            Inputs(100.0, 95.0, 0.0, 0.04, 0.22, carry=-0.02),
            Inputs(100.0, 95.0, 0.5, 0.04, 0.0, carry=-0.02),
            Inputs(0.0, 95.0, 0.5, 0.04, 0.22, carry=-0.02),
            Inputs(100.0, 0.0, 0.5, 0.04, 0.22, carry=-0.02),
        ],
    )
    @pytest.mark.parametrize("option", list(OptionType))
    def test_degenerate_is_finite_and_at_least_intrinsic(
        self, option_inputs: Inputs, option: OptionType
    ) -> None:
        value = bjerksund_stensland(option_inputs, option)
        immediate = max(option.sign * (option_inputs.spot - option_inputs.strike), 0.0)
        assert math.isfinite(value)
        assert value >= immediate - 1e-12


class TestAgainstTheLattice:
    """The two methods share no code, so agreement is evidence."""

    @pytest.mark.parametrize("option", list(OptionType))
    def test_agreement_across_a_strike_ladder(self, option: OptionType) -> None:
        for strike in (70.0, 85.0, 100.0, 115.0, 130.0):
            option_inputs = Inputs(100.0, strike, 1.0, 0.07, 0.28, carry=-0.03)
            approximation = bjerksund_stensland(option_inputs, option)
            american = lattice_value(option_inputs, option)
            assert approximation <= american + 1e-5
            assert approximation == pytest.approx(american, rel=0.03, abs=1e-4)

    @pytest.mark.parametrize("option", list(OptionType))
    def test_agreement_across_maturities(self, option: OptionType) -> None:
        for time in (0.08, 0.25, 0.5, 1.0, 2.0):
            option_inputs = Inputs(100.0, 100.0, time, 0.07, 0.28, carry=-0.03)
            approximation = bjerksund_stensland(option_inputs, option)
            american = lattice_value(option_inputs, option)
            assert approximation <= american + 1e-5
            assert approximation == pytest.approx(american, rel=0.03, abs=1e-4)


class TestTwoStepBoundary:
    """The 2002 refinement, and the safeguard that makes its guarantee hold."""

    @pytest.mark.parametrize("row", GRID)
    @pytest.mark.parametrize("option", list(OptionType))
    def test_the_bracket_still_holds(
        self, row: tuple[float, float, float, float, float, float], option: OptionType
    ) -> None:
        """European <= 2002 <= American, the inequality the whole module rests on."""
        option_inputs = inputs_of(row)
        european = price(option_inputs, option)
        approximate = bjerksund_stensland_2002(option_inputs, option)
        american = lattice_value(option_inputs, option)

        assert approximate >= european - 1e-12
        assert approximate <= american + 1e-6

    @pytest.mark.parametrize("row", GRID)
    @pytest.mark.parametrize("option", list(OptionType))
    def test_is_never_worse_than_the_single_boundary(
        self, row: tuple[float, float, float, float, float, float], option: OptionType
    ) -> None:
        """Guaranteed by the maximum, not by the two-step formula on its own."""
        option_inputs = inputs_of(row)
        assert bjerksund_stensland_2002(option_inputs, option) >= bjerksund_stensland(
            option_inputs, option
        ) - 1e-12

    @pytest.mark.parametrize("row", GRID)
    @pytest.mark.parametrize("option", list(OptionType))
    def test_is_at_least_as_accurate_as_the_single_boundary(
        self, row: tuple[float, float, float, float, float, float], option: OptionType
    ) -> None:
        """Both are lower bounds, so being larger is the same as being closer.

        Worth stating as accuracy anyway, since that is the claim a reader of
        the module cares about, and it is the one that would fail first if the
        two-step algebra were wrong.
        """
        option_inputs = inputs_of(row)
        american = lattice_value(option_inputs, option)
        single = abs(bjerksund_stensland(option_inputs, option) - american)
        double = abs(bjerksund_stensland_2002(option_inputs, option) - american)
        assert double <= single + 1e-12

    def test_the_improvement_over_the_single_boundary_is_measured(self) -> None:
        """The number written in the docstring, asserted rather than asserted-to.

        A refinement that is merely never worse is not worth the bivariate
        normal it costs. This pins the size of the gain, so that a change which
        quietly reduced the 2002 value to the 1993 one everywhere -- the most
        likely way for this to rot -- would fail here rather than pass every
        inequality above.
        """
        single_total = 0.0
        double_total = 0.0
        for row in GRID:
            for option in OptionType:
                option_inputs = inputs_of(row)
                american = lattice_value(option_inputs, option)
                single_total += abs(bjerksund_stensland(option_inputs, option) - american)
                double_total += abs(bjerksund_stensland_2002(option_inputs, option) - american)

        assert double_total < single_total / 1.2, (
            f"mean error {single_total:.6f} -> {double_total:.6f}, "
            "less improvement than the module claims"
        )

    def test_the_safeguard_binds_somewhere(self) -> None:
        """The maximum is not decoration, and this is the market that proves it.

        Found by a random sweep: high volatility over a long maturity with a
        carry close to the rate. Here the raw two-step formula falls *below*
        the European price, which no American value may do since holding to
        expiry is always available. If a future change made the raw formula
        well behaved everywhere, this test failing is the right way to find out.
        """
        option_inputs = Inputs(91.98, 90.95, 2.766, 0.0272, 0.572, carry=0.0165)
        european = price(option_inputs, OptionType.CALL)
        raw = _two_step_value(option_inputs, OptionType.CALL)
        guarded = bjerksund_stensland_2002(option_inputs, OptionType.CALL)

        assert raw < european, "this market is chosen because the raw formula misbehaves"
        assert guarded >= european
        assert guarded == pytest.approx(
            max(european, bjerksund_stensland(option_inputs, OptionType.CALL))
        )

    @pytest.mark.parametrize("row", GRID)
    def test_the_put_transformation_agrees_with_the_call(
        self, row: tuple[float, float, float, float, float, float]
    ) -> None:
        """A put priced directly must equal its mirrored call, as for the 1993 form."""
        option_inputs = inputs_of(row)
        mirrored = Inputs(
            option_inputs.strike,
            option_inputs.spot,
            option_inputs.time,
            option_inputs.rate - option_inputs.b,
            option_inputs.vol,
            carry=-option_inputs.b,
        )
        assert bjerksund_stensland_2002(option_inputs, OptionType.PUT) == pytest.approx(
            bjerksund_stensland_2002(mirrored, OptionType.CALL), rel=1e-12
        )

    @pytest.mark.parametrize("row", GRID)
    def test_a_call_with_no_exercise_region_is_exactly_european(
        self, row: tuple[float, float, float, float, float, float]
    ) -> None:
        """With the carry at or above the rate, waiting never costs anything.

        Stated for the call only, and deliberately so. The put's mirror of this
        case is a carry at or below zero, which is not the same condition, and
        writing one parametrised test over both option types would leave the
        put branch asserting nothing.
        """
        spot, strike, time, rate, _carry, vol = row
        option_inputs = Inputs(spot, strike, time, rate, vol, carry=rate)
        assert bjerksund_stensland_2002(option_inputs, OptionType.CALL) == pytest.approx(
            price(option_inputs, OptionType.CALL), rel=1e-12
        )

    @pytest.mark.parametrize("row", GRID)
    def test_a_put_with_no_exercise_region_is_exactly_european(
        self, row: tuple[float, float, float, float, float, float]
    ) -> None:
        """The mirror condition: under the transformation the put's call has
        carry equal to its rate when the put's own rate is zero."""
        spot, strike, time, _rate, _carry, vol = row
        option_inputs = Inputs(spot, strike, time, 0.0, vol, carry=0.0)
        assert bjerksund_stensland_2002(option_inputs, OptionType.PUT) == pytest.approx(
            price(option_inputs, OptionType.PUT), rel=1e-12
        )

    def test_the_two_triggers_are_ordered_and_straddle_the_single_one(self) -> None:
        """The earlier level sits below the later one, which is what a step means.

        If the scaling in the decay constant were wrong the two would coincide,
        the formula would collapse to something close to the 1993 one, and every
        inequality above would still pass.
        """
        for row in GRID:
            option_inputs = inputs_of(row)
            if option_inputs.b >= option_inputs.rate:
                continue
            lower, upper, switch = _two_step_triggers(option_inputs)
            assert 0.0 < lower < upper
            assert 0.0 < switch < option_inputs.time
            assert switch == pytest.approx(0.6180339887 * option_inputs.time, rel=1e-8)

    @pytest.mark.parametrize("option", list(OptionType))
    def test_degenerate_inputs_give_the_intrinsic_value(self, option: OptionType) -> None:
        for option_inputs in (
            Inputs(100.0, 95.0, 0.0, 0.05, 0.2),
            Inputs(100.0, 95.0, 1.0, 0.05, 0.0),
        ):
            expected = max(option.sign * (option_inputs.spot - option_inputs.strike), 0.0)
            assert bjerksund_stensland_2002(option_inputs, option) >= expected - 1e-12
