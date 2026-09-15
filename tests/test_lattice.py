"""Lattice pricing, American exercise and the early-exercise boundary.

An American price has no closed form, so the tests here cannot check it against
one. What they can do is check the properties that any correct American price
must have, and check them on a grid rather than at a point:

* where theory says the early-exercise right is worthless, the lattice must say
  so *exactly*, not approximately;
* where it is worth something, the price must respect every arbitrage bound;
* three independent constructions must agree with each other;
* the European limit must converge to the closed form at the documented rate.

Together these pin the induction down tightly. The first in particular is a
sharp test: an American call under ``b >= r`` must equal the European call to the
last bit, and almost any indexing error in the backward induction breaks it.
"""

from __future__ import annotations

import math
from itertools import pairwise
from statistics import mean

import pytest

from moneyness import Inputs, OptionType, price
from moneyness.lattice import (
    Exercise,
    Lattice,
    LatticePrice,
    boundary,
    min_steps,
    price_lattice,
    richardson,
)

LATTICES = list(Lattice)

# Spot, strike, time, rate, vol, carry. Chosen to cover both sides of the money,
# short and long maturities, and the sign changes of the carry that decide
# whether early exercise has any value at all.
GRID = [
    (100.0, 95.0, 0.5, 0.04, 0.22, None),
    (100.0, 100.0, 1.0, 0.05, 0.30, None),
    (100.0, 110.0, 0.25, 0.03, 0.18, None),
    (90.0, 100.0, 2.0, 0.06, 0.25, None),
    (100.0, 95.0, 0.5, 0.10, 0.20, -0.04),
    (100.0, 100.0, 1.0, 0.05, 0.35, 0.0),
    (120.0, 100.0, 0.75, 0.04, 0.28, 0.01),
]


def inputs_of(row: tuple[float, float, float, float, float, float | None]) -> Inputs:
    spot, strike, time, rate, vol, carry = row
    return Inputs(spot, strike, time, rate, vol, carry=carry)


class TestEuropeanConvergence:
    """The lattice must reproduce the closed form in the European case."""

    @pytest.mark.parametrize("lattice", LATTICES)
    @pytest.mark.parametrize("option", list(OptionType))
    def test_converges_to_closed_form(self, lattice: Lattice, option: OptionType) -> None:
        option_inputs = inputs_of(GRID[0])
        exact = price(option_inputs, option)
        got = price_lattice(
            option_inputs,
            option,
            steps=400,
            lattice=lattice,
            exercise=Exercise.EUROPEAN,
        ).value
        assert got == pytest.approx(exact, abs=5e-3)

    @pytest.mark.parametrize("lattice", LATTICES)
    def test_error_decays_like_one_over_n(self, lattice: Lattice) -> None:
        """The rate, not just the endpoint.

        The raw lattice error oscillates, so no single pair of layer counts shows
        the rate cleanly and asserting a monotone decrease would produce a test
        that fails for correct code. What is stable is the envelope: ``n`` times
        the error stays bounded as ``n`` grows, which is exactly the statement
        that the error is ``O(1/n)``. A quantity decaying more slowly would send
        this product to infinity.
        """
        option_inputs = inputs_of(GRID[0])
        exact = price(option_inputs, OptionType.CALL)
        scaled = []
        for n in (100, 200, 400, 800):
            got = price_lattice(
                option_inputs,
                OptionType.CALL,
                steps=n,
                lattice=lattice,
                exercise=Exercise.EUROPEAN,
                smooth=False,
            ).value
            scaled.append(n * abs(got - exact))
        assert max(scaled) < 5.0
        assert scaled[-1] < 5.0 * scaled[0] + 1.0

    @pytest.mark.parametrize("lattice", LATTICES)
    def test_smoothing_beats_the_raw_lattice(self, lattice: Lattice) -> None:
        option_inputs = inputs_of(GRID[0])
        exact = price(option_inputs, OptionType.CALL)
        raw: list[float] = []
        smoothed: list[float] = []
        for n in range(60, 121, 12):
            for target, flag in ((raw, False), (smoothed, True)):
                got = price_lattice(
                    option_inputs,
                    OptionType.CALL,
                    steps=n,
                    lattice=lattice,
                    exercise=Exercise.EUROPEAN,
                    smooth=flag,
                ).value
                target.append(abs(got - exact))
        assert mean(smoothed) < mean(raw)


class TestRichardson:
    """Extrapolation helps only once the error it assumes is the error there is."""

    @pytest.mark.parametrize("lattice", LATTICES)
    def test_extrapolation_beats_the_fine_lattice_when_smoothed(
        self, lattice: Lattice
    ) -> None:
        option_inputs = inputs_of(GRID[0])
        exact = price(option_inputs, OptionType.CALL)
        fine: list[float] = []
        extrapolated: list[float] = []
        for n in range(60, 121, 12):
            fine.append(
                abs(
                    price_lattice(
                        option_inputs,
                        OptionType.CALL,
                        steps=2 * n,
                        lattice=lattice,
                        exercise=Exercise.EUROPEAN,
                    ).value
                    - exact
                )
            )
            extrapolated.append(
                abs(
                    richardson(
                        option_inputs,
                        OptionType.CALL,
                        steps=n,
                        lattice=lattice,
                        exercise=Exercise.EUROPEAN,
                    )
                    - exact
                )
            )
        assert mean(extrapolated) < 0.2 * mean(fine)

    @pytest.mark.parametrize("lattice", LATTICES)
    def test_extrapolating_the_raw_lattice_is_worse(self, lattice: Lattice) -> None:
        """The documented failure, asserted so it stays documented.

        This is not a defect being enshrined. It is the reason smoothing is on by
        default, and if it ever stopped being true the default would be worth
        revisiting — so the claim is checked rather than left in prose.
        """
        option_inputs = inputs_of(GRID[0])
        exact = price(option_inputs, OptionType.CALL)
        fine: list[float] = []
        extrapolated: list[float] = []
        for n in range(60, 121, 12):
            fine.append(
                abs(
                    price_lattice(
                        option_inputs,
                        OptionType.CALL,
                        steps=2 * n,
                        lattice=lattice,
                        exercise=Exercise.EUROPEAN,
                        smooth=False,
                    ).value
                    - exact
                )
            )
            extrapolated.append(
                abs(
                    richardson(
                        option_inputs,
                        OptionType.CALL,
                        steps=n,
                        lattice=lattice,
                        exercise=Exercise.EUROPEAN,
                        smooth=False,
                    )
                    - exact
                )
            )
        assert mean(extrapolated) > mean(fine)


class TestNoEarlyExercise:
    """Where the right is worthless, the lattice must price it at exactly zero."""

    @pytest.mark.parametrize("lattice", LATTICES)
    @pytest.mark.parametrize("row", GRID)
    def test_american_call_equals_european_when_carry_at_least_rate(
        self, lattice: Lattice, row: tuple[float, float, float, float, float, float | None]
    ) -> None:
        """The classic result, and a sharp test of the induction.

        With ``b >= r`` the forward never falls below the spot fast enough to make
        waiting costly, so exercising an American call early is never optimal and
        its value equals the European one. On a lattice this must hold *node by
        node*, so the two backward inductions produce bitwise identical numbers —
        not merely close ones. An off-by-one in the child indices, a probability
        attached to the wrong branch or a node price off by one grid step would
        all show up here as a non-zero premium.
        """
        option_inputs = inputs_of(row)
        if option_inputs.b < option_inputs.rate:
            pytest.skip("early exercise can have value when the carry is below the rate")
        result = price_lattice(
            option_inputs, OptionType.CALL, steps=200, lattice=lattice
        )
        assert result.early_exercise_premium == 0.0

    @pytest.mark.parametrize("lattice", LATTICES)
    def test_american_put_at_zero_rate_has_no_premium(self, lattice: Lattice) -> None:
        """At ``r = 0`` the put's early-exercise right is worthless too.

        The whole reason to exercise a put early is to receive the strike sooner
        and earn interest on it. Set the rate to zero and that motive disappears,
        so however deep in the money the option goes, waiting is never worse —
        and the premium must be exactly zero, not merely small.

        The case is worth stating separately from the call because it is the
        mirror image and because the obvious intuition points the wrong way: a
        deep in-the-money put with a falling underlying *looks* like it should be
        exercised. Holding the strike sooner is the only thing exercise buys, and
        at a zero rate that is worth nothing.
        """
        option_inputs = Inputs(60.0, 100.0, 1.0, 0.0, 0.30, carry=-0.05)
        result = price_lattice(option_inputs, OptionType.PUT, steps=200, lattice=lattice)
        assert result.early_exercise_premium == 0.0

    @pytest.mark.parametrize("lattice", LATTICES)
    def test_american_put_has_a_premium_once_the_rate_is_positive(
        self, lattice: Lattice
    ) -> None:
        """The same option at a positive rate does carry a premium.

        Paired with the test above, this is what shows the zero is a real
        consequence of ``r = 0`` rather than an exercise rule that never fires.
        """
        option_inputs = Inputs(60.0, 100.0, 1.0, 0.08, 0.30, carry=0.03)
        result = price_lattice(option_inputs, OptionType.PUT, steps=200, lattice=lattice)
        assert result.early_exercise_premium > 1e-3


class TestArbitrageBounds:
    """American prices must sit inside the bounds no-arbitrage forces on them."""

    @pytest.mark.parametrize("lattice", LATTICES)
    @pytest.mark.parametrize("row", GRID)
    @pytest.mark.parametrize("option", list(OptionType))
    def test_dominates_european_and_intrinsic(
        self,
        lattice: Lattice,
        row: tuple[float, float, float, float, float, float | None],
        option: OptionType,
    ) -> None:
        option_inputs = inputs_of(row)
        american = price_lattice(option_inputs, option, steps=200, lattice=lattice)
        european = price_lattice(
            option_inputs, option, steps=200, lattice=lattice, exercise=Exercise.EUROPEAN
        )
        immediate = max(option.sign * (option_inputs.spot - option_inputs.strike), 0.0)

        assert american.value >= european.value - 1e-12
        assert american.value >= immediate - 1e-12
        assert american.early_exercise_premium >= -1e-12

    @pytest.mark.parametrize("row", GRID)
    def test_american_parity_inequality(
        self, row: tuple[float, float, float, float, float, float | None]
    ) -> None:
        """American options obey a parity *band*, not an identity.

        Early exercise breaks the exact European relation and leaves the two-sided
        bound ``S e^{(b-r)T} - K <= C - P <= S - K e^{-rT}``. The two sides are
        not symmetric: the lower edge carries the spot forward by the dividend
        yield, the upper edge does not. Writing ``S e^{-qT}`` on both — the
        plausible-looking mistake — gives a band too tight to hold, and a
        dividend-paying case walks straight out of it. Checking
        the band rather than an equality is the point: an implementation that
        priced the American pair as if they were European would satisfy a parity
        identity and fail here on the lower edge.
        """
        option_inputs = inputs_of(row)
        call = price_lattice(option_inputs, OptionType.CALL, steps=300).value
        put = price_lattice(option_inputs, OptionType.PUT, steps=300).value
        carried = option_inputs.spot * math.exp(
            (option_inputs.b - option_inputs.rate) * option_inputs.time
        )
        lower = carried - option_inputs.strike
        upper = option_inputs.spot - option_inputs.strike * option_inputs.discount
        assert lower - 1e-6 <= call - put <= upper + 1e-6

    @pytest.mark.parametrize("lattice", LATTICES)
    def test_monotone_in_strike(self, lattice: Lattice) -> None:
        """A call falls and a put rises as the strike goes up. No exceptions."""
        calls: list[float] = []
        puts: list[float] = []
        for strike in (80.0, 90.0, 100.0, 110.0, 120.0):
            option_inputs = Inputs(100.0, strike, 1.0, 0.05, 0.25, carry=0.0)
            calls.append(price_lattice(option_inputs, OptionType.CALL, steps=150,
                                       lattice=lattice).value)
            puts.append(price_lattice(option_inputs, OptionType.PUT, steps=150,
                                      lattice=lattice).value)
        assert all(a >= b - 1e-12 for a, b in pairwise(calls))
        assert all(a <= b + 1e-12 for a, b in pairwise(puts))


class TestConstructionsAgree:
    """Three different discretisations, one answer."""

    @pytest.mark.parametrize("row", GRID)
    @pytest.mark.parametrize("option", list(OptionType))
    def test_lattices_agree_on_american_prices(
        self,
        row: tuple[float, float, float, float, float, float | None],
        option: OptionType,
    ) -> None:
        option_inputs = inputs_of(row)
        values = [
            richardson(option_inputs, option, steps=200, lattice=lattice)
            for lattice in LATTICES
        ]
        # Measured worst case over this grid is 3.0e-3, on the two-year
        # deep-in-the-money put where the early-exercise premium is largest and
        # the three constructions resolve the boundary differently. That is 2e-4
        # relative to a price near 14, so the bound is set just above the
        # observed spread rather than at a round number chosen for comfort.
        assert max(values) - min(values) < 5e-3


class TestStability:
    """A layer too coarse for the drift is refused, not silently priced."""

    def test_refuses_a_layer_that_is_not_a_distribution(self) -> None:
        # A large carry against a small volatility: over one layer the forward
        # grows past the up-node, and the risk-neutral probability leaves [0, 1].
        option_inputs = Inputs(100.0, 100.0, 5.0, 0.02, 0.05, carry=0.60)
        with pytest.raises(ValueError, match="not a probability"):
            price_lattice(option_inputs, OptionType.CALL, steps=2, lattice=Lattice.CRR)

    def test_min_steps_is_enough(self) -> None:
        option_inputs = Inputs(100.0, 100.0, 5.0, 0.02, 0.05, carry=0.60)
        for lattice in LATTICES:
            floor = min_steps(option_inputs, lattice)
            result = price_lattice(
                option_inputs, OptionType.CALL, steps=floor, lattice=lattice
            )
            assert math.isfinite(result.value)

    def test_min_steps_is_tight_for_crr(self) -> None:
        """One layer below the stated minimum must actually fail.

        A bound that is merely sufficient would be satisfied by returning a huge
        number, so the test checks that the floor is where the behaviour changes.
        """
        option_inputs = Inputs(100.0, 100.0, 5.0, 0.02, 0.05, carry=0.60)
        floor = min_steps(option_inputs, Lattice.CRR)
        assert floor > 1
        with pytest.raises(ValueError, match="not a probability"):
            price_lattice(
                option_inputs, OptionType.CALL, steps=floor - 1, lattice=Lattice.CRR
            )

    def test_rejects_non_positive_steps(self) -> None:
        option_inputs = inputs_of(GRID[0])
        for bad in (0, -1):
            with pytest.raises(ValueError, match="steps must be positive"):
                price_lattice(option_inputs, OptionType.CALL, steps=bad)
            with pytest.raises(ValueError, match="steps must be positive"):
                boundary(option_inputs, OptionType.PUT, steps=bad)


class TestDegenerate:
    """Zero time, zero volatility and zero spot still have to return something."""

    @pytest.mark.parametrize(
        "option_inputs",
        [
            Inputs(100.0, 95.0, 0.0, 0.04, 0.22),
            Inputs(100.0, 95.0, 0.5, 0.04, 0.0),
            Inputs(0.0, 95.0, 0.5, 0.04, 0.22),
            Inputs(100.0, 0.0, 0.5, 0.04, 0.22),
        ],
    )
    @pytest.mark.parametrize("option", list(OptionType))
    def test_degenerate_matches_intrinsic(
        self, option_inputs: Inputs, option: OptionType
    ) -> None:
        result = price_lattice(option_inputs, option, steps=50)
        assert math.isfinite(result.value)
        assert result.value >= max(
            option.sign * (option_inputs.spot - option_inputs.strike), 0.0
        ) - 1e-12

    def test_boundary_of_a_degenerate_option_is_empty(self) -> None:
        assert boundary(Inputs(100.0, 95.0, 0.0, 0.04, 0.22), OptionType.PUT) == []


class TestBoundary:
    """Where the exercise region begins."""

    def test_ends_at_the_strike(self) -> None:
        option_inputs = Inputs(100.0, 100.0, 1.0, 0.06, 0.25, carry=0.0)
        curve = boundary(option_inputs, OptionType.PUT, steps=120)
        assert curve[-1] == (option_inputs.time, option_inputs.strike)

    def test_put_boundary_lies_below_the_strike(self) -> None:
        option_inputs = Inputs(100.0, 100.0, 1.0, 0.06, 0.25, carry=0.0)
        curve = boundary(option_inputs, OptionType.PUT, steps=120)
        assert len(curve) > 10
        for _, critical in curve[:-1]:
            assert critical < option_inputs.strike

    def test_trinomial_boundary_is_monotone(self) -> None:
        """The trinomial grid has no parity, so the read-out rises cleanly."""
        option_inputs = Inputs(100.0, 100.0, 1.0, 0.06, 0.25, carry=0.0)
        curve = boundary(
            option_inputs, OptionType.PUT, steps=120, lattice=Lattice.TRINOMIAL
        )
        levels = [s for _, s in curve]
        assert all(a <= b + 1e-9 for a, b in pairwise(levels))

    def test_crr_boundary_is_monotone_within_each_parity(self) -> None:
        """The documented sawtooth, pinned down rather than tolerated.

        Consecutive Cox-Ross-Rubinstein layers sample two interleaved node grids,
        so the boundary alternates between them. Because ``u d = 1`` the grid
        itself does not move with the layer index, so splitting by parity
        recovers two monotone sequences. Asserting that is what distinguishes a
        grid artefact of known shape from a genuine failure of the exercise rule.
        """
        option_inputs = Inputs(100.0, 100.0, 1.0, 0.06, 0.25, carry=0.0)
        curve = boundary(option_inputs, OptionType.PUT, steps=120, lattice=Lattice.CRR)
        levels = [s for _, s in curve[:-1]]
        assert len(levels) > 20
        for parity in (0, 1):
            side = levels[parity::2]
            assert all(a <= b + 1e-9 for a, b in pairwise(side))

    def test_jarrow_rudd_boundary_tracks_crr_within_a_node(self) -> None:
        """Jarrow-Rudd samples the same curve on a grid that slides.

        The parity split above does *not* rescue Jarrow-Rudd, and it should not
        be expected to. Its node set is ``S e^{n mu + (2j - n) sigma}``: the whole
        grid translates by ``mu`` per layer, where the Cox-Ross-Rubinstein grid
        stands still. Over these inputs ``mu`` is negative, so the grid drifts
        down while the true boundary rises, and within a parity class the slide
        wins — the sampled boundary edges downward by a few hundredths.

        What must still hold is that the two constructions trace the same
        underlying curve. Compared at the layers they share they agree to 0.82 of
        one node spacing, which is the most that can be asked of two read-outs
        quantised onto different grids.
        """
        option_inputs = Inputs(100.0, 100.0, 1.0, 0.06, 0.25, carry=0.0)
        steps = 120
        spacing = 2.0 * option_inputs.vol * math.sqrt(option_inputs.time / steps)
        crr = dict(
            boundary(option_inputs, OptionType.PUT, steps=steps, lattice=Lattice.CRR)
        )
        jr = dict(
            boundary(
                option_inputs, OptionType.PUT, steps=steps, lattice=Lattice.JARROW_RUDD
            )
        )
        shared = sorted(set(crr) & set(jr))
        assert len(shared) > 50
        worst = max(abs(math.log(crr[t] / jr[t])) for t in shared)
        assert worst < spacing

    def test_call_with_high_carry_is_never_exercised_early(self) -> None:
        """Only the expiry point survives when the exercise region is empty."""
        option_inputs = Inputs(100.0, 100.0, 1.0, 0.05, 0.25)
        curve = boundary(option_inputs, OptionType.CALL, steps=120)
        assert curve == [(option_inputs.time, option_inputs.strike)]

    def test_call_on_a_dividend_payer_is_exercised_early(self) -> None:
        """With the carry below the rate the call does have an exercise region."""
        option_inputs = Inputs(100.0, 100.0, 1.0, 0.05, 0.25, carry=-0.05)
        curve = boundary(option_inputs, OptionType.CALL, steps=120)
        assert len(curve) > 10
        for _, critical in curve[:-1]:
            assert critical > option_inputs.strike


class TestResultShape:
    """The returned record describes the grid it came from."""

    def test_reports_its_own_grid(self) -> None:
        option_inputs = inputs_of(GRID[0])
        result = price_lattice(
            option_inputs,
            OptionType.PUT,
            steps=64,
            lattice=Lattice.TRINOMIAL,
            exercise=Exercise.AMERICAN,
        )
        assert isinstance(result, LatticePrice)
        assert result.steps == 64
        assert result.lattice is Lattice.TRINOMIAL
        assert result.exercise is Exercise.AMERICAN

    def test_european_valuation_has_no_premium(self) -> None:
        option_inputs = inputs_of(GRID[0])
        result = price_lattice(
            option_inputs, OptionType.PUT, steps=64, exercise=Exercise.EUROPEAN
        )
        assert result.early_exercise_premium == 0.0
