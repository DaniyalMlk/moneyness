"""The surface: interpolation, the two arbitrage conditions, and local volatility.

The centrepiece is :class:`TestDupire`. The library computes local volatility
from the surface analytically, as ``(dw/dT) / g``. The test computes it from
option *prices*, by finite differences, using the original Dupire formula. The
two share no algebra — one differentiates total variance in log-moneyness, the
other differentiates call prices in strike and maturity — so agreement between
them is evidence about the identity rather than about either implementation.
"""

from __future__ import annotations

import math
import random

import pytest

from moneyness.bsm import Inputs, OptionType, price
from moneyness.surface import Surface
from moneyness.svi import SVI, Butterfly, durrleman

# Three ordered slices: total variance rises with maturity everywhere, the skew
# flattens as it should, and each is individually admissible. This is the shape
# of a real equity surface.
NEAR = SVI(a=0.02, b=0.15, rho=-0.35, m=0.0, s=0.12)
MID = SVI(a=0.05, b=0.22, rho=-0.30, m=0.0, s=0.18)
FAR = SVI(a=0.10, b=0.30, rho=-0.25, m=0.0, s=0.25)

SURFACE = Surface([(0.25, NEAR), (1.0, MID), (2.0, FAR)])

KS = [-0.8, -0.45, -0.2, 0.0, 0.15, 0.35, 0.6]
TIMES = [0.4, 0.75, 1.25, 1.8]


def _flat(sigma: float, maturities: tuple[float, ...] = (0.25, 1.0, 2.0)) -> Surface:
    """A surface with one constant volatility everywhere.

    Total variance is ``sigma^2 T``, which as a slice is ``a = sigma^2 T`` with
    ``b = 0``. Every derivative in ``k`` vanishes, ``g`` is identically one, and
    local volatility must come back as exactly ``sigma`` — the one case where
    implied and local volatility coincide, and therefore the one case where the
    Dupire implementation has a right answer known to the last bit.
    """
    return Surface(
        [(t, SVI(a=sigma * sigma * t, b=0.0, rho=0.0, m=0.0, s=1.0)) for t in maturities]
    )


class TestConstruction:
    def test_sorts_slices_by_maturity(self) -> None:
        surface = Surface([(2.0, FAR), (0.25, NEAR), (1.0, MID)])
        assert surface.maturities == (0.25, 1.0, 2.0)
        assert surface.slices == (NEAR, MID, FAR)

    def test_a_single_slice_is_a_valid_surface(self) -> None:
        surface = Surface([(1.0, MID)])
        assert surface.calendar() == []
        assert surface.total_variance(0.1, 5.0) == MID.total_variance(0.1)

    @pytest.mark.parametrize(
        ("slices", "match"),
        [
            ([], "at least one slice"),
            ([(0.0, MID)], "finite and positive"),
            ([(-1.0, MID)], "finite and positive"),
            ([(math.inf, MID)], "finite and positive"),
            ([(1.0, MID), (1.0, FAR)], "share the maturity"),
        ],
    )
    def test_rejects_a_malformed_surface(
        self, slices: list[tuple[float, SVI]], match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            Surface(slices)

    def test_repr_says_what_it_holds(self) -> None:
        assert "3 slices" in repr(SURFACE)


class TestInterpolation:
    @pytest.mark.parametrize("k", KS)
    def test_reproduces_the_quoted_slices_exactly(self, k: float) -> None:
        for time, slice_ in zip(SURFACE.maturities, SURFACE.slices, strict=True):
            assert SURFACE.total_variance(k, time) == pytest.approx(
                slice_.total_variance(k), abs=1e-15
            )

    @pytest.mark.parametrize("k", KS)
    def test_the_midpoint_is_the_average_of_its_neighbours(self, k: float) -> None:
        """Linear in total variance, which is the whole basis of the calendar argument."""
        expected = (NEAR.total_variance(k) + MID.total_variance(k)) / 2.0
        assert SURFACE.total_variance(k, 0.625) == pytest.approx(expected, rel=1e-14)

    @pytest.mark.parametrize("k", KS)
    def test_extrapolates_flat_in_total_variance(self, k: float) -> None:
        """Beyond the quoted range, total variance holds and volatility decays.

        The consequence worth asserting is the second half: implied volatility
        falls like ``1 / sqrt(T)`` rather than staying put, which is the
        conservative direction to be wrong in.
        """
        assert SURFACE.total_variance(k, 0.01) == pytest.approx(NEAR.total_variance(k))
        assert SURFACE.total_variance(k, 50.0) == pytest.approx(FAR.total_variance(k))
        assert SURFACE.volatility(k, 8.0) == pytest.approx(
            SURFACE.volatility(k, 2.0) * math.sqrt(2.0 / 8.0), rel=1e-12
        )

    @pytest.mark.parametrize("k", KS)
    def test_total_variance_never_falls_with_maturity(self, k: float) -> None:
        """The property linear interpolation is chosen to guarantee."""
        previous = -math.inf
        for i in range(400):
            time = 0.05 + i * 3.0 / 400
            current = SURFACE.total_variance(k, time)
            assert current >= previous - 1e-14
            previous = current

    @pytest.mark.parametrize("k", KS)
    @pytest.mark.parametrize("time", TIMES)
    def test_the_k_derivatives_match_numerical_ones(self, k: float, time: float) -> None:
        h = 1e-5
        first = (
            SURFACE.total_variance(k + h, time) - SURFACE.total_variance(k - h, time)
        ) / (2.0 * h)
        assert SURFACE.d_total_variance(k, time) == pytest.approx(first, abs=1e-7, rel=1e-6)

        h2 = 1e-4
        second = (
            SURFACE.total_variance(k + h2, time)
            - 2.0 * SURFACE.total_variance(k, time)
            + SURFACE.total_variance(k - h2, time)
        ) / (h2 * h2)
        assert SURFACE.d2_total_variance(k, time) == pytest.approx(second, abs=1e-5, rel=1e-4)

    @pytest.mark.parametrize("k", KS)
    @pytest.mark.parametrize("time", TIMES)
    def test_the_maturity_derivative_matches_a_numerical_one(
        self, k: float, time: float
    ) -> None:
        """Checked inside a bracket, where the piecewise-linear surface is differentiable.

        At a quoted maturity the slope genuinely jumps, so a central difference
        there straddles two different linear pieces and would be comparing
        against a quantity that does not exist. The test stays away from the
        knots on purpose rather than smoothing over them.
        """
        h = 1e-6
        numerical = (
            SURFACE.total_variance(k, time + h) - SURFACE.total_variance(k, time - h)
        ) / (2.0 * h)
        assert SURFACE.dw_dt(k, time) == pytest.approx(numerical, abs=1e-6, rel=1e-6)

    @pytest.mark.parametrize("k", KS)
    def test_the_maturity_derivative_vanishes_outside_the_quoted_range(self, k: float) -> None:
        assert SURFACE.dw_dt(k, 0.05) == 0.0
        assert SURFACE.dw_dt(k, 20.0) == 0.0

    @pytest.mark.parametrize("time", [0.0, -1.0])
    def test_rejects_a_non_positive_maturity(self, time: float) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            SURFACE.volatility(0.0, time)


class TestCalendar:
    def test_ordered_slices_show_no_crossing(self) -> None:
        report = SURFACE.calendar()
        assert len(report) == 2
        assert all(entry.free for entry in report)
        assert all(entry.worst < 0.0 for entry in report)
        assert [(e.earlier, e.later) for e in report] == [(0.25, 1.0), (1.0, 2.0)]

    def test_crossing_slices_are_caught_with_the_size_of_the_crossing(self) -> None:
        """A later slice placed below an earlier one is a certain profit, and is reported."""
        crossed = Surface([(0.5, MID), (1.5, NEAR)])
        entry = crossed.calendar()[0]
        assert not entry.free
        assert entry.worst > 0.0
        gap = MID.total_variance(entry.at) - NEAR.total_variance(entry.at)
        assert entry.worst == pytest.approx(gap, rel=1e-12)

    def test_a_partial_crossing_is_found_even_where_most_of_the_slice_is_fine(self) -> None:
        """The wings cross while the middle does not, which a spot check would miss."""
        steep_early = SVI(a=0.02, b=0.55, rho=-0.1, m=0.0, s=0.02)
        shallow_late = SVI(a=0.09, b=0.08, rho=-0.1, m=0.0, s=0.2)
        surface = Surface([(0.5, steep_early), (1.0, shallow_late)])

        assert surface.total_variance(0.0, 0.5) < surface.total_variance(0.0, 1.0)
        entry = surface.calendar()[0]
        assert not entry.free
        assert abs(entry.at) > 0.5

    @pytest.mark.parametrize(("wing", "points"), [(0.0, 401), (-1.0, 401), (5.0, 2)])
    def test_rejects_a_nonsensical_scan(self, wing: float, points: int) -> None:
        with pytest.raises(ValueError):
            SURFACE.calendar(wing=wing, points=points)


class TestButterfly:
    def test_checks_the_gaps_as_well_as_the_quoted_maturities(self) -> None:
        report = SURFACE.butterfly()
        times = [t for t, _, _ in report]
        assert times == sorted(times)
        assert 0.625 in times and 1.5 in times
        assert set(SURFACE.maturities) <= set(times)

    def test_an_admissible_surface_is_admissible_between_maturities_too(self) -> None:
        assert all(worst >= 0.0 for _, worst, _ in SURFACE.butterfly())

    def test_agrees_with_the_slice_scan_at_a_quoted_maturity(self) -> None:
        """The surface and the slice must say the same thing where they overlap."""
        for time, slice_ in zip(SURFACE.maturities, SURFACE.slices, strict=True):
            from_surface = next(
                w for t, w, _ in SURFACE.butterfly(maturities=[time]) if t == time
            )
            expected = Butterfly.scan(slice_, points=401).worst
            assert from_surface == pytest.approx(expected, rel=1e-12)

    def test_a_bad_slice_is_reported_at_its_own_maturity(self) -> None:
        bad = SVI(a=0.01, b=1.5, rho=0.5, m=0.0, s=0.05)
        surface = Surface([(1.0, bad)])
        _, worst, at = surface.butterfly()[0]
        assert worst < 0.0
        assert durrleman(bad, at) == pytest.approx(worst, rel=1e-12)

    def test_interpolation_is_not_assumed_to_preserve_the_condition(self) -> None:
        """A randomised hunt for an interpolated violation, kept as a regression.

        No counterexample has been found — a much larger sweep than this one
        also came up empty — so the surface may well preserve the condition in
        general. It is not proved here, which is why the code recomputes ``g``
        on interpolated slices instead of assuming it, and why this test exists:
        if a future change to the interpolation breaks the property, this is
        what notices.
        """
        rng = random.Random(11)
        examined = 0
        for _ in range(4000):
            try:
                early = SVI(
                    a=rng.uniform(0.001, 0.15),
                    b=rng.uniform(0.0, 1.2),
                    rho=rng.uniform(-0.95, 0.95),
                    m=rng.uniform(-0.3, 0.3),
                    s=rng.uniform(0.02, 0.6),
                )
                late = SVI(
                    a=early.a + rng.uniform(0.0, 0.3),
                    b=rng.uniform(0.0, 1.2),
                    rho=rng.uniform(-0.95, 0.95),
                    m=rng.uniform(-0.3, 0.3),
                    s=rng.uniform(0.02, 0.6),
                )
            except ValueError:
                continue
            if not Butterfly.scan(early, points=121).free:
                continue
            if not Butterfly.scan(late, points=121).free:
                continue
            surface = Surface([(1.0, early), (2.0, late)])
            if not surface.calendar(points=121)[0].free:
                continue

            examined += 1
            _, worst, at = surface.butterfly(points=121, maturities=[1.5])[0]
            assert worst >= 0.0, (
                f"interpolating {early} and {late} gives g={worst} at k={at}"
            )

        assert examined > 100, f"only {examined} admissible pairs were examined"

    @pytest.mark.parametrize(("wing", "points"), [(0.0, 401), (-1.0, 401), (5.0, 2)])
    def test_rejects_a_nonsensical_scan(self, wing: float, points: int) -> None:
        with pytest.raises(ValueError):
            SURFACE.butterfly(wing=wing, points=points)

    def test_rejects_a_non_positive_requested_maturity(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            SURFACE.butterfly(maturities=[1.0, 0.0])


class TestDupire:
    @pytest.mark.parametrize("k", KS)
    @pytest.mark.parametrize("time", TIMES)
    def test_a_flat_surface_gives_back_its_own_volatility(self, k: float, time: float) -> None:
        """Where implied volatility is constant, local volatility equals it exactly.

        Not approximately: ``g`` is identically one and ``dw/dT`` is identically
        ``sigma^2``, so the identity reduces to arithmetic with no rounding in
        it. Anything other than an exact match is a bug.
        """
        sigma = 0.27
        result = _flat(sigma).local_vol(k, time)
        assert result.admissible
        assert result.durrleman == 1.0
        assert result.volatility == pytest.approx(sigma, abs=1e-15)

    @pytest.mark.parametrize("k", [-0.4, -0.15, 0.0, 0.2, 0.45])
    @pytest.mark.parametrize("time", [0.5, 0.75, 1.4])
    def test_matches_dupire_computed_from_option_prices(self, k: float, time: float) -> None:
        """The independent check, and the reason to trust the identity.

        The library evaluates ``(dw/dT) / g`` on the surface. This test instead
        prices calls off the same surface and applies the original Dupire
        formula to them,

            sigma_loc^2 = C_T / (K^2 C_KK / 2)

        at zero rate and zero carry, where the two drift terms drop out. The
        only thing the two paths share is the surface itself; the differentiation
        happens in different variables, of different functions, one analytically
        and one numerically.

        The tolerance is set by the finite differences, not by the library. A
        second-order central difference on a smooth function carries a relative
        error around ``1e-8`` at these step sizes, and that is what is seen.
        """
        spot = 100.0

        def call(strike: float, maturity: float) -> float:
            log_moneyness = math.log(strike / spot)
            vol = SURFACE.volatility(log_moneyness, maturity)
            return price(Inputs(spot, strike, maturity, 0.0, vol, carry=0.0), OptionType.CALL)

        strike = spot * math.exp(k)
        h_t, h_k = 1e-4, strike * 1e-4
        c_t = (call(strike, time + h_t) - call(strike, time - h_t)) / (2.0 * h_t)
        c_kk = (
            call(strike + h_k, time) - 2.0 * call(strike, time) + call(strike - h_k, time)
        ) / (h_k * h_k)

        from_prices = math.sqrt(c_t / (0.5 * strike * strike * c_kk))
        from_surface = SURFACE.local_vol(k, time).volatility
        assert from_surface == pytest.approx(from_prices, rel=1e-6)

    @pytest.mark.parametrize("k", KS)
    @pytest.mark.parametrize("time", TIMES)
    def test_the_denominator_is_durrlemans_function(self, k: float, time: float) -> None:
        """Stated in the module docstring, asserted here rather than left as a remark."""
        result = SURFACE.local_vol(k, time)
        assert result.durrleman == pytest.approx(SURFACE.durrleman(k, time), rel=1e-15)
        assert result.dw_dt == pytest.approx(SURFACE.dw_dt(k, time), rel=1e-15)
        if result.admissible:
            assert result.variance == pytest.approx(result.dw_dt / result.durrleman, rel=1e-14)
            assert result.volatility == pytest.approx(math.sqrt(result.variance), rel=1e-15)

    def test_a_calendar_crossing_makes_the_identity_inadmissible(self) -> None:
        """A negative numerator, reported as such rather than as a square root of one."""
        crossed = Surface([(0.5, MID), (1.5, NEAR)])
        entry = crossed.calendar()[0]
        result = crossed.local_vol(entry.at, 1.0)

        assert not result.admissible
        assert math.isnan(result.variance)
        assert math.isnan(result.volatility)
        assert result.dw_dt < 0.0

    def test_a_density_defect_makes_the_identity_inadmissible(self) -> None:
        """A non-positive denominator, from a slice that implies negative probability."""
        bad = SVI(a=0.01, b=1.5, rho=0.5, m=0.0, s=0.05)
        worse = SVI(a=0.02, b=1.6, rho=0.5, m=0.0, s=0.05)
        surface = Surface([(1.0, bad), (2.0, worse)])
        _, worst, at = surface.butterfly(maturities=[1.5])[0]
        assert worst < 0.0

        result = surface.local_vol(at, 1.5)
        assert not result.admissible
        assert result.durrleman < 0.0
        assert math.isnan(result.volatility)

    @pytest.mark.parametrize("time", [0.0, -1.0])
    def test_rejects_a_non_positive_maturity(self, time: float) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            SURFACE.local_vol(0.0, time)

    @pytest.mark.parametrize("k", KS)
    @pytest.mark.parametrize("time", TIMES)
    def test_local_volatility_is_real_wherever_the_surface_is_admissible(
        self, k: float, time: float
    ) -> None:
        """The claim the module docstring makes about existence, checked on the grid."""
        assert all(entry.free for entry in SURFACE.calendar())
        assert all(worst >= 0.0 for _, worst, _ in SURFACE.butterfly())
        assert SURFACE.local_vol(k, time).admissible


class TestDurrlemanOnTheSurface:
    @pytest.mark.parametrize("k", KS)
    def test_agrees_with_the_slice_at_a_quoted_maturity(self, k: float) -> None:
        for time, slice_ in zip(SURFACE.maturities, SURFACE.slices, strict=True):
            assert SURFACE.durrleman(k, time) == pytest.approx(durrleman(slice_, k), rel=1e-14)

    @pytest.mark.parametrize("k", KS)
    def test_is_not_the_interpolation_of_the_endpoint_values(self, k: float) -> None:
        """``g`` is nonlinear in ``w``, which is exactly why it has to be recomputed.

        If this ever became an equality, the interpolation would have silently
        stopped being what the rest of the module assumes.
        """
        blended = (durrleman(NEAR, k) + durrleman(MID, k)) / 2.0
        assert SURFACE.durrleman(k, 0.625) != pytest.approx(blended, rel=1e-12)

    def test_undefined_where_total_variance_vanishes(self) -> None:
        b, s = 0.4, 0.1
        touching = SVI(a=-b * s, b=b, rho=0.0, m=0.0, s=s)
        with pytest.raises(ValueError, match="undefined"):
            Surface([(1.0, touching)]).durrleman(0.0, 1.0)
