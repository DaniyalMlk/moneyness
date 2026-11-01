"""The options nothing had ever chosen, each exercised by a test that can fail.

``tests/test_options.py`` asks mechanically whether every documented option is
supplied somewhere. These are the answers for the three it found unsupplied.
Each one checks what the option *governs*: a call that passes an argument and
asserts nothing would satisfy the gate and establish nothing, which is the
failure mode a gate like that invites.
"""

from __future__ import annotations

import itertools
import math

import pytest

from moneyness.quadrature import QuadratureError, _Budget, adaptive_quad, semi_infinite_quad
from moneyness.spread import AssetPair, spread_price
from moneyness.svi import _VERTICES, SVI, calibrate

#: An ordinary pair: two assets of similar size, moderate volatilities, a
#: correlation nothing special happens at.
SMOOTH = AssetPair(100.0, 96.0, 1.0, 0.03, 0.25, 0.30, 0.4)

#: The corner the panel density is for. At a correlation of 0.999 the
#: conditional volatility is a thirtieth of the unconditional one, so the
#: conditional option is nearly its own intrinsic value and the integrand
#: nearly carries a kink.
NEARLY_INTRINSIC = AssetPair(100.0, 96.0, 1.0, 0.03, 0.05, 0.90, 0.999)

#: Quotes for a slice, and a vertex inside their range.
QUOTES = (-0.6, -0.45, -0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3, 0.45, 0.6)
TRUTH = SVI(a=0.02, b=0.40, rho=-0.6, m=0.05, s=0.18)


class TestThePanelDensity:
    """``spread_price(panels_per_unit=)``: panels per unit of the driving variate."""

    @pytest.mark.parametrize("density", [0.0, -1.0, -1e-30, math.inf, math.nan])
    def test_a_density_that_is_not_positive_and_finite_is_refused(self, density: float) -> None:
        with pytest.raises(ValueError, match="panels_per_unit must be positive"):
            spread_price(SMOOTH, 4.0, panels_per_unit=density)

    def test_the_guard_beside_it_is_unchanged(self) -> None:
        with pytest.raises(ValueError, match="tolerance must be positive"):
            spread_price(SMOOTH, 4.0, tolerance=0.0)

    def test_the_price_converges_as_the_density_rises(self) -> None:
        # The claim "exact up to the quadrature" means the sequence has to
        # settle, and it does: measured against a dense reference on the hard
        # pair, the error falls from 2.2e-07 at one panel per unit to 1.1e-15 at
        # eight.
        #
        # Strictly falling, and over a range chosen by measurement. Below one
        # panel per unit the eight-panel floor makes successive densities the
        # same computation, and above eight the error is double-precision noise
        # that rises and falls at the sixteenth digit -- so a non-strict
        # comparison over a wider range passes against a density that is capped
        # part way, where the capped end of the sequence is a run of equal
        # numbers. That is a break this test missed before the range was pinned
        # down.
        reference = spread_price(NEARLY_INTRINSIC, 40.0, panels_per_unit=600.0)
        errors = [
            abs(spread_price(NEARLY_INTRINSIC, 40.0, panels_per_unit=density) - reference)
            / reference
            for density in (1.0, 2.0, 4.0, 8.0)
        ]
        for coarser, finer in itertools.pairwise(errors):
            assert finer < coarser, errors
        assert errors[0] > 1e-8, (
            "the coarsest density must be visibly wrong, or this proves nothing"
        )
        assert errors[-1] < 1e-11

    def test_the_kink_is_on_a_panel_edge_and_that_is_why_this_works(self) -> None:
        # Not an option, but the reason the density can be as low as it is: the
        # roots of `critical_states` are inserted as panel edges, so the
        # integrand is smooth inside every panel. The suite checks that those
        # roots are found; nothing checked that they are used, and this is what
        # a coarse density buys with them.
        #
        # Measured both ways. At one panel per unit, with the roots on the
        # edges, the hard pair prices to 2.2e-07 and 5.3e-08 relative; with the
        # insertion removed the same two prices come out at 5.7e-05 and
        # 9.7e-05, so a bound of 1e-06 separates them by about two orders of
        # magnitude.
        for strike in (40.0, 0.0):
            reference = spread_price(NEARLY_INTRINSIC, strike, panels_per_unit=600.0)
            coarse = spread_price(NEARLY_INTRINSIC, strike, panels_per_unit=1.0)
            assert abs(coarse - reference) / reference < 1e-6, strike
        # And the other half: on a smooth pair the insertion changes nothing, so
        # the claim above is about the kink rather than about panels in general.
        smooth_reference = spread_price(SMOOTH, 4.0, panels_per_unit=600.0)
        smooth_coarse = spread_price(SMOOTH, 4.0, panels_per_unit=1.0)
        assert abs(smooth_coarse - smooth_reference) / smooth_reference < 1e-14

    def test_the_density_is_what_moves_it_and_not_the_pair(self) -> None:
        # The spread between a coarse and a dense run, on both pairs. On the
        # smooth one the floor of eight panels already integrates to machine
        # precision, so the parameter looks inert; on the hard one it does not.
        # Asserting both is what distinguishes a density that works from one
        # that is ignored -- a test on the smooth pair alone would pass either
        # way.
        def spread(pair: AssetPair, strike: float) -> float:
            dense = spread_price(pair, strike, panels_per_unit=600.0)
            coarse = spread_price(pair, strike, panels_per_unit=0.25)
            return abs(dense - coarse) / dense

        assert spread(SMOOTH, 4.0) < 1e-14
        assert spread(NEARLY_INTRINSIC, 40.0) > 1e-7

    def test_the_floor_is_the_reason_the_smooth_case_cannot_see_it(self) -> None:
        # Eight panels is the floor, so every density below 8 / (2 * width) is
        # the same computation. Asserted as an equality because it is the same
        # arithmetic, not an approximation to it.
        exact = spread_price(SMOOTH, 4.0, panels_per_unit=1e-9)
        for tiny in (1e-12, 1e-10, 1e-8):
            assert spread_price(SMOOTH, 4.0, panels_per_unit=tiny) == exact


class TestTheRestartCount:
    """``calibrate(restarts=)``: how many of the four starting vertices to try."""

    @staticmethod
    def quotes() -> tuple[list[float], list[float]]:
        ks = list(QUOTES)
        return ks, [TRUTH.total_variance(k) for k in ks]

    @pytest.mark.parametrize("restarts", [0, -1, -100])
    def test_a_count_below_one_is_refused_rather_than_silently_becoming_one(
        self, restarts: int
    ) -> None:
        ks, ws = self.quotes()
        with pytest.raises(ValueError, match="restarts must be at least one"):
            calibrate(ks, ws, restarts=restarts)

    def test_more_starts_never_fit_worse(self) -> None:
        # The objective is a minimum over the starts tried, so it is monotone
        # non-increasing in the count by construction. Worth asserting because
        # it is the property a reader relies on when turning the knob up, and a
        # search that overwrote its best rather than comparing would break it.
        ks, ws = self.quotes()
        errors = [calibrate(ks, ws, restarts=r).rmse for r in range(1, _VERTICES + 1)]
        assert errors == sorted(errors, reverse=True), errors
        # And a spread: if every count gave the same number this would hold
        # vacuously. One start here fits measurably worse than four.
        assert errors[0] > errors[-1]

    @pytest.mark.parametrize("restarts", [_VERTICES + 1, 10, 1000])
    def test_asking_for_more_than_there_are_is_the_same_search(self, restarts: int) -> None:
        # The documented cap, asserted as exact agreement rather than as
        # closeness: the same vertices in the same order produce the same
        # floating-point result, and anything else would mean the extra counts
        # were doing something undocumented.
        ks, ws = self.quotes()
        capped = calibrate(ks, ws, restarts=_VERTICES)
        more = calibrate(ks, ws, restarts=restarts)
        assert more.rmse == capped.rmse
        assert more.max_error == capped.max_error
        assert more.slice_ == capped.slice_
        assert more.iterations == capped.iterations

    def test_one_start_is_not_the_same_search_as_four(self) -> None:
        # Otherwise the test above would pass against a function that ignored
        # the count entirely.
        ks, ws = self.quotes()
        assert calibrate(ks, ws, restarts=1).rmse != calibrate(ks, ws, restarts=_VERTICES).rmse

    def test_every_count_still_recovers_the_slice_it_was_given(self) -> None:
        # The fit is what the count is for, so the count is not allowed to
        # change the answer, only how hard the search looked for it.
        ks, ws = self.quotes()
        for restarts in range(1, _VERTICES + 1):
            fitted = calibrate(ks, ws, restarts=restarts).slice_
            assert fitted.m == pytest.approx(TRUTH.m, abs=1e-6)
            assert fitted.s == pytest.approx(TRUTH.s, abs=1e-6)
            assert fitted.b == pytest.approx(TRUTH.b, abs=1e-6)
            assert fitted.rho == pytest.approx(TRUTH.rho, abs=1e-6)


class TestThePanelBudget:
    """``adaptive_quad(budget=)``: the panel allowance, shared down one walk."""

    def test_two_integrals_sharing_one_allowance_spend_from_the_same_count(self) -> None:
        # Which is the whole reason the parameter exists: the half-line rule
        # walks outwards over many panels and a per-call allowance would let it
        # spend without limit in total.
        budget = _Budget(400)
        first = adaptive_quad(math.sin, 0.0, 1.0, budget=budget)
        after_one = budget.left
        second = adaptive_quad(math.sin, 1.0, 2.0, budget=budget)
        assert first == pytest.approx(1.0 - math.cos(1.0), abs=1e-12)
        assert second == pytest.approx(math.cos(1.0) - math.cos(2.0), abs=1e-12)
        assert after_one < 400, "the first integral spent nothing"
        assert budget.left < after_one, "the second integral spent from a fresh allowance"

    def test_an_allowance_too_small_for_the_integrand_is_refused(self) -> None:
        # An oscillation the rule has to bisect its way into, against an
        # allowance of one panel. The error says what to do about it, which is
        # the difference between a refusal and a crash.
        def wiggly(x: float) -> float:
            return math.sin(400.0 * x) * math.exp(-x)

        with pytest.raises(QuadratureError, match="panel budget"):
            adaptive_quad(wiggly, 0.0, 10.0, budget=_Budget(1))

    def test_a_fresh_allowance_is_made_when_none_is_given(self) -> None:
        # The documented default, and the reason the two calls below do not
        # interfere: each gets its own count.
        for _ in range(3):
            assert adaptive_quad(math.cos, 0.0, 1.0) == pytest.approx(math.sin(1.0), abs=1e-12)

    def test_the_half_line_rule_shares_one_allowance_across_its_panels(self) -> None:
        # The behaviour the parameter was added for, from the outside: a
        # decaying integrand over the half-line resolves, and the same
        # integrand with the allowance of a single panel does not.
        value = semi_infinite_quad(lambda x: math.exp(-x), 0.0)
        assert value == pytest.approx(1.0, abs=1e-12)
        with pytest.raises(QuadratureError):
            adaptive_quad(
                lambda x: math.sin(400.0 * x) * math.exp(-x), 0.0, 40.0, budget=_Budget(2)
            )
