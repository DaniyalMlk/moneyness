"""SVI slices: the parametrisation, the density it implies, and the fit.

The organising idea is that almost nothing here is checked against a transcribed
number. The derivatives are checked against numerical differentiation of the
function they claim to differentiate, the density is checked against the two
integrals any risk-neutral density must satisfy, and the calibration is checked
by recovering slices it was never told the parameters of.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence

import pytest

from moneyness.svi import SVI, Butterfly, calibrate, density, durrleman

# A well-behaved equity-style slice: downward skew, rounded vertex, wings well
# inside Lee's bound. Used wherever a test needs "an ordinary slice".
ORDINARY = SVI(a=0.04, b=0.4, rho=-0.4, m=0.05, s=0.1)

# Slices spanning the shapes the parametrisation is meant to cover: nearly flat,
# strongly skewed, sharply peaked, far off centre, and symmetric.
SHAPES = [
    SVI(a=0.04, b=0.4, rho=-0.4, m=0.05, s=0.1),
    SVI(a=0.09, b=0.2, rho=-0.3, m=0.0, s=0.3),
    SVI(a=0.02, b=0.15, rho=-0.75, m=-0.1, s=0.05),
    SVI(a=0.16, b=0.05, rho=0.5, m=0.4, s=0.6),
    SVI(a=0.01, b=0.3, rho=0.0, m=0.0, s=0.2),
    SVI(a=0.25, b=0.0, rho=0.0, m=0.0, s=1.0),
]

GRID = [-2.0, -1.2, -0.7, -0.35, -0.1, 0.0, 0.1, 0.25, 0.5, 0.9, 1.5]


def _simpson(
    f: Callable[[float], float], lo: float, hi: float, intervals: int
) -> float:
    """Composite Simpson, used only to integrate densities in the tests."""
    if intervals % 2:
        intervals += 1
    h = (hi - lo) / intervals
    total: float = f(lo) + f(hi)
    for i in range(1, intervals):
        total += (4.0 if i % 2 else 2.0) * f(lo + i * h)
    return total * h / 3.0


class TestDomain:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"b": -0.1}, "non-negative"),
            ({"rho": 1.0}, "strictly inside"),
            ({"rho": -1.0}, "strictly inside"),
            ({"rho": 1.5}, "strictly inside"),
            ({"s": 0.0}, "must be positive"),
            ({"s": -0.2}, "must be positive"),
            ({"a": math.nan}, "must be finite"),
            ({"b": math.inf}, "must be finite"),
        ],
    )
    def test_rejects_parameters_outside_the_domain(
        self, kwargs: dict[str, float], match: str
    ) -> None:
        base: dict[str, float] = {"a": 0.04, "b": 0.4, "rho": -0.4, "m": 0.0, "s": 0.1}
        with pytest.raises(ValueError, match=match):
            SVI(**{**base, **kwargs})

    def test_rejects_a_slice_that_dips_below_zero_variance(self) -> None:
        """The constraint that is not obvious from any single parameter.

        Each of ``a``, ``b``, ``rho`` and ``s`` is individually admissible here;
        it is their combination that puts part of the slice at negative total
        variance, which is not a mispriced option but a meaningless one.
        """
        with pytest.raises(ValueError, match="negative"):
            SVI(a=-0.5, b=0.4, rho=-0.4, m=0.0, s=0.1)

    def test_accepts_a_slice_that_just_touches_zero(self) -> None:
        """The boundary is inclusive, because zero total variance is attainable."""
        b, rho, s = 0.4, -0.4, 0.1
        a = -b * s * math.sqrt(1.0 - rho * rho)
        slice_ = SVI(a=a, b=b, rho=rho, m=0.0, s=s)
        assert slice_.minimum == pytest.approx(0.0, abs=1e-15)

    @pytest.mark.parametrize("slice_", SHAPES)
    def test_the_minimum_is_actually_the_minimum(self, slice_: SVI) -> None:
        """``minimum`` claims a closed form; a sweep has to agree with it."""
        swept = min(slice_.total_variance(-8.0 + i * 16.0 / 4000) for i in range(4001))
        assert swept >= slice_.minimum - 1e-12
        assert slice_.minimum == pytest.approx(min(swept, slice_.minimum), abs=1e-9)


class TestDerivatives:
    @pytest.mark.parametrize("slice_", SHAPES)
    @pytest.mark.parametrize("k", GRID)
    def test_first_derivative_matches_a_numerical_one(self, slice_: SVI, k: float) -> None:
        h = 1e-5
        numerical = (slice_.total_variance(k + h) - slice_.total_variance(k - h)) / (2.0 * h)
        assert slice_.d_total_variance(k) == pytest.approx(numerical, abs=1e-7, rel=1e-6)

    @pytest.mark.parametrize("slice_", SHAPES)
    @pytest.mark.parametrize("k", GRID)
    def test_second_derivative_matches_a_numerical_one(self, slice_: SVI, k: float) -> None:
        h = 1e-4
        numerical = (
            slice_.total_variance(k + h)
            - 2.0 * slice_.total_variance(k)
            + slice_.total_variance(k - h)
        ) / (h * h)
        assert slice_.d2_total_variance(k) == pytest.approx(numerical, abs=1e-5, rel=1e-4)

    @pytest.mark.parametrize("slice_", SHAPES)
    def test_the_slice_is_convex(self, slice_: SVI) -> None:
        """Convexity in log-moneyness is structural, not a property of these parameters."""
        assert all(slice_.d2_total_variance(k) >= 0.0 for k in GRID)

    @pytest.mark.parametrize("slice_", SHAPES)
    def test_the_slope_stays_between_the_asymptotes(self, slice_: SVI) -> None:
        left, right = slice_.wing_slopes
        for k in GRID:
            assert left - 1e-12 <= slice_.d_total_variance(k) <= right + 1e-12

    def test_the_wings_approach_their_asymptotic_slopes(self) -> None:
        left, right = ORDINARY.wing_slopes
        assert ORDINARY.d_total_variance(-1e6) == pytest.approx(left, rel=1e-9)
        assert ORDINARY.d_total_variance(1e6) == pytest.approx(right, rel=1e-9)


class TestVolatility:
    def test_converts_total_variance_to_a_volatility(self) -> None:
        assert ORDINARY.volatility(0.0, 4.0) == pytest.approx(
            math.sqrt(ORDINARY.total_variance(0.0) / 4.0)
        )

    @pytest.mark.parametrize("time", [0.0, -1.0])
    def test_rejects_a_non_positive_maturity(self, time: float) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            ORDINARY.volatility(0.0, time)


class TestDurrlemanAndDensity:
    @pytest.mark.parametrize("slice_", SHAPES)
    def test_the_density_integrates_to_one(self, slice_: SVI) -> None:
        """The demanding check on the analytic derivatives.

        An error in ``w'`` or ``w''`` propagates straight into ``g`` and so into
        the mass, with no cancellation to hide behind.

        The range has to be chosen rather than fixed. Total variance grows
        linearly in the wings with slope ``b(1 +- rho)``, so the density decays
        like ``exp(-|k| / (2 slope))`` — the steeper the wing, the fatter the
        tail and the further out the mass sits. Integrating a steep slice over a
        range picked for a flat one loses mass and looks like a bug; this is
        what a first run of this test actually did, at the default range.
        """
        widest = max(abs(x) for x in slice_.wing_slopes)
        reach = max(40.0, 120.0 * widest)
        mass = _simpson(lambda k: density(slice_, k), -reach, reach, 40000)
        assert mass == pytest.approx(1.0, abs=1e-8)

    @pytest.mark.parametrize("slice_", SHAPES)
    def test_the_density_prices_the_forward_correctly(self, slice_: SVI) -> None:
        """``E[e^k] = 1``: the forward is a martingale under the implied measure.

        Independent of the mass check, and sensitive to different errors — this
        one weights the right wing heavily, where the previous one barely looks.
        """
        widest = max(abs(x) for x in slice_.wing_slopes)
        reach = max(40.0, 120.0 * widest)
        forward = _simpson(lambda k: math.exp(k) * density(slice_, k), -reach, reach, 40000)
        assert forward == pytest.approx(1.0, abs=1e-7)

    def test_a_flat_slice_gives_the_lognormal_density(self) -> None:
        """Where ``w`` is constant, ``g`` is one and the density is textbook."""
        w = 0.25
        flat = SVI(a=w, b=0.0, rho=0.0, m=0.0, s=1.0)
        for k in GRID:
            assert durrleman(flat, k) == pytest.approx(1.0, abs=1e-14)
            d_minus = -k / math.sqrt(w) - math.sqrt(w) / 2.0
            expected = math.exp(-d_minus * d_minus / 2.0) / math.sqrt(2.0 * math.pi * w)
            assert density(flat, k) == pytest.approx(expected, rel=1e-14)

    @pytest.mark.parametrize("slice_", SHAPES)
    @pytest.mark.parametrize("k", GRID)
    def test_the_density_carries_the_sign_of_durrleman(self, slice_: SVI, k: float) -> None:
        """The claim that makes ``g`` the butterfly condition rather than a proxy."""
        g = durrleman(slice_, k)
        p = density(slice_, k)
        assert math.copysign(1.0, g) == math.copysign(1.0, p) or abs(g) < 1e-300

    def test_undefined_where_total_variance_vanishes(self) -> None:
        """A slice may legally touch zero, and there the density has a pole.

        Built by putting the minimum exactly at zero, which with ``rho = 0``
        sits at ``k = m``.
        """
        b, s = 0.4, 0.1
        touching = SVI(a=-b * s, b=b, rho=0.0, m=0.0, s=s)
        assert touching.total_variance(0.0) == 0.0

        with pytest.raises(ValueError, match="undefined"):
            durrleman(touching, 0.0)
        with pytest.raises(ValueError, match="undefined"):
            density(touching, 0.0)

    def test_a_sharp_vertex_does_not_underflow(self) -> None:
        """Regression: the second derivative used to cube ``r`` and divide by zero.

        ``s`` here is legal — positive and finite — but small enough that ``r^3``
        underflows while ``b (s/r)^2 / r`` stays perfectly representable. The
        grouping in the implementation is the whole fix.
        """
        sharp = SVI(a=0.04, b=0.4, rho=0.0, m=0.0, s=1e-300)
        assert math.isfinite(sharp.d2_total_variance(0.0))
        assert sharp.d2_total_variance(0.0) > 0.0
        assert math.isfinite(durrleman(sharp, 0.0))


class TestButterflyScan:
    @pytest.mark.parametrize("slice_", SHAPES)
    def test_ordinary_slices_are_admissible(self, slice_: SVI) -> None:
        assert Butterfly.scan(slice_).free

    @pytest.mark.parametrize(
        "slice_",
        [
            SVI(a=0.01, b=1.5, rho=0.5, m=0.0, s=0.05),
            SVI(a=0.02, b=0.9, rho=-0.8, m=0.0, s=0.03),
            SVI(a=0.005, b=1.2, rho=0.0, m=0.0, s=0.02),
        ],
    )
    def test_steep_slices_are_caught(self, slice_: SVI) -> None:
        """Steep wings on a low level is the classic way to imply negative probability."""
        report = Butterfly.scan(slice_)
        assert not report.free
        assert report.worst < 0.0
        assert durrleman(slice_, report.at) == pytest.approx(report.worst, rel=1e-12)

    def test_a_violation_shows_up_as_negative_density(self) -> None:
        bad = SVI(a=0.01, b=1.5, rho=0.5, m=0.0, s=0.05)
        report = Butterfly.scan(bad)
        assert density(bad, report.at) < 0.0

    @pytest.mark.parametrize(("wing", "points"), [(0.0, 601), (-1.0, 601), (5.0, 2)])
    def test_rejects_a_nonsensical_scan(self, wing: float, points: int) -> None:
        with pytest.raises(ValueError):
            Butterfly.scan(ORDINARY, wing=wing, points=points)


class TestCalibration:
    @pytest.mark.parametrize("slice_", SHAPES[:5])
    def test_recovers_a_slice_it_generated_the_quotes_from(self, slice_: SVI) -> None:
        """The strongest available check: the answer is known exactly.

        Quotes are generated from a slice and the fit is asked to find it. There
        is no noise, so a correct implementation should land on the parameters
        to near machine precision, and anything less points at the search rather
        than at the data.
        """
        ks = [-0.6, -0.4, -0.25, -0.1, 0.0, 0.1, 0.2, 0.35, 0.5, 0.7]
        quotes = [slice_.total_variance(k) for k in ks]
        fit = calibrate(ks, quotes)

        assert fit.converged
        assert fit.rmse < 1e-10
        assert fit.max_error < 1e-9
        for k in ks:
            assert fit.slice_.total_variance(k) == pytest.approx(
                slice_.total_variance(k), abs=1e-9
            )

    def test_the_fitted_function_matches_off_the_quoted_points(self) -> None:
        """Fitting the quotes is easy; reproducing the function between them is the test."""
        ks = [-0.6, -0.4, -0.25, -0.1, 0.0, 0.1, 0.2, 0.35, 0.5, 0.7]
        fit = calibrate(ks, [ORDINARY.total_variance(k) for k in ks])
        for k in (-0.52, -0.33, -0.05, 0.14, 0.41, 0.63):
            assert fit.slice_.total_variance(k) == pytest.approx(
                ORDINARY.total_variance(k), abs=1e-8
            )

    @pytest.mark.parametrize("seed", range(12))
    def test_fits_noisy_quotes_without_leaving_the_domain(self, seed: int) -> None:
        """Real quotes are not on any slice, and the fit must stay admissible anyway.

        The interesting failure is not a large residual, it is a fit that
        achieves a small residual by leaving the region where the parameters
        mean anything — negative ``b``, ``|rho|`` past one, or a slice dipping
        below zero variance. The constructor enforces all three, so reaching
        this assertion at all is most of the test.
        """
        rng = random.Random(seed)
        ks = [-0.7 + i * 0.1 for i in range(15)]
        quotes = [
            ORDINARY.total_variance(k) * (1.0 + rng.uniform(-0.03, 0.03)) for k in ks
        ]
        fit = calibrate(ks, quotes)

        assert fit.slice_.b >= 0.0
        assert -1.0 < fit.slice_.rho < 1.0
        assert fit.slice_.s > 0.0
        assert fit.slice_.minimum >= 0.0
        assert fit.rmse < 0.01

    def test_weights_move_the_fit_towards_the_weighted_quotes(self) -> None:
        """A weight of zero on a quote must make the fit ignore it."""
        ks = [-0.6, -0.4, -0.25, -0.1, 0.0, 0.1, 0.2, 0.35, 0.5, 0.7]
        quotes = [ORDINARY.total_variance(k) for k in ks]
        spoiled = list(quotes)
        spoiled[0] = quotes[0] * 3.0

        ignored = calibrate(ks, spoiled, weight=[0.0] + [1.0] * (len(ks) - 1))
        for k, target in zip(ks[1:], quotes[1:], strict=True):
            assert ignored.slice_.total_variance(k) == pytest.approx(target, abs=1e-7)

        counted = calibrate(ks, spoiled)
        assert counted.slice_.total_variance(ks[0]) > ignored.slice_.total_variance(ks[0])

    def test_a_flat_set_of_quotes_gives_a_flat_slice(self) -> None:
        """``b`` collapses and ``rho`` stops being identified; the fit must not thrash."""
        ks = [-0.5, -0.25, -0.1, 0.0, 0.1, 0.3, 0.6]
        fit = calibrate(ks, [0.09] * len(ks))
        assert fit.max_error < 1e-8
        assert fit.slice_.b == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.parametrize(
        ("ks", "quotes", "match"),
        [
            ([0.0, 0.1, 0.2], [0.04] * 3, "at least five"),
            ([0.0, 0.1, 0.2, 0.3], [0.04] * 4, "at least five"),
            ([0.0, 0.1, 0.2, 0.3, 0.4], [0.04] * 4, "5 abscissae and 4"),
            ([0.0, 0.1, 0.2, 0.3, 0.4], [0.04, 0.04, 0.0, 0.04, 0.04], "must be positive"),
            ([0.0, 0.0, 0.0, 0.0, 0.0, 0.1], [0.04] * 6, "distinct"),
        ],
    )
    def test_rejects_quotes_it_cannot_fit(
        self, ks: Sequence[float], quotes: Sequence[float], match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            calibrate(ks, quotes)

    @pytest.mark.parametrize(
        ("weight", "match"),
        [
            ([1.0, 1.0], "2 weights"),
            ([1.0, 1.0, -1.0, 1.0, 1.0, 1.0], "non-negative"),
            ([0.0] * 6, "at least one weight"),
        ],
    )
    def test_rejects_bad_weights(self, weight: list[float], match: str) -> None:
        ks = [-0.4, -0.2, 0.0, 0.2, 0.4, 0.6]
        with pytest.raises(ValueError, match=match):
            calibrate(ks, [ORDINARY.total_variance(k) for k in ks], weight=weight)
