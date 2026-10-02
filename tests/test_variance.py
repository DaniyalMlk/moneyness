"""Variance and volatility swaps: two exact targets and three error terms.

The log-contract replication is model-free, so it can be checked against
answers that are *exact* rather than against itself. Under a flat
Black-Scholes smile the fair variance is ``sigma**2``, with nothing
approximate about it. Under Heston it is ``E[int V]/T``, which the model
supplies in closed form. Both are computed here from option *prices* through
the replication, which shares no algebra with either closed form.

Everything else in this file is about what a finite ladder of strikes costs:
the truncation at the ends of the range, the spacing between strikes, and the
centring term that arises because the quotes change over at a listed strike
and not at the forward. Each has a prediction, and each prediction is checked
against the measurement rather than quoted.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable

import pytest

from moneyness.bsm import Inputs, OptionType
from moneyness.bsm import price as bs_price
from moneyness.heston import Contract, Heston
from moneyness.heston import price as heston_price
from moneyness.heston_mc import variance_path
from moneyness.quadrature import adaptive_quad
from moneyness.variance import (
    BadStrip,
    Centring,
    Strip,
    fair_variance,
    fair_variance_from_strip,
    integrated_variance_variance,
    truncation_error,
    volatility_swap_strike,
)

FORWARD = 100.0
RATE = 0.03

# Three variance processes: one starting at its own long-run level, one above
# it and one below, so the mean-reversion term is exercised in both directions
# and is absent in the first.
STATIONARY = Heston(v0=0.04, theta=0.04, kappa=1.5, sigma=0.5, rho=-0.7)
ABOVE = Heston(v0=0.09, theta=0.04, kappa=2.0, sigma=0.8, rho=-0.5)
BELOW = Heston(v0=0.01, theta=0.06, kappa=0.5, sigma=0.3, rho=0.3)
MODELS = (STATIONARY, ABOVE, BELOW)


def flat_otm(
    forward: float, time: float, rate: float, vol: float
) -> Callable[[float], float]:
    """Out-of-the-money Black-Scholes prices on a forward, as the replication wants."""

    def otm(strike: float) -> float:
        option = OptionType.CALL if strike >= forward else OptionType.PUT
        return bs_price(Inputs.on_future(forward, strike, time, rate, vol), option)

    return otm


def heston_otm(
    model: Heston, contract: Contract, tol: float = 1e-13
) -> Callable[[float], float]:
    def otm(strike: float) -> float:
        option = OptionType.CALL if strike >= contract.forward else OptionType.PUT
        one = Contract(contract.spot, strike, contract.time, contract.rate, contract.carry)
        return heston_price(model, one, option, tol)

    return otm


class TestFlatExact:
    """A flat smile's fair variance is its own variance, to the last bit."""

    @pytest.mark.parametrize("vol", [0.1, 0.2, 0.4, 0.8])
    @pytest.mark.parametrize("time", [0.25, 1.0, 5.0])
    def test_recovers_sigma_squared(self, vol: float, time: float) -> None:
        """Eight standard deviations of log-moneyness, and a tolerance in them.

        The width has to be set in standard deviations rather than in
        log-moneyness, because a flat 20% at a quarter of a year and a flat
        80% at five years differ by a factor of eighteen in how far out the
        strikes have to reach. The quadrature tolerance has to scale too: it
        is absolute, and the integral it bounds is ``sigma**2 T / 2``, which
        spans four orders of magnitude over this grid.
        """
        discount = math.exp(-RATE * time)
        sd = vol * math.sqrt(time)
        result = fair_variance(
            flat_otm(FORWARD, time, RATE, vol),
            FORWARD,
            time,
            discount,
            width=8.0 * sd,
            tol=1e-15 * vol * vol * time,
        )
        assert result.fair_variance == pytest.approx(vol * vol, rel=1e-13)

    @pytest.mark.parametrize("vol", [0.15, 0.3])
    def test_fair_volatility_is_the_square_root(self, vol: float) -> None:
        discount = math.exp(-RATE)
        result = fair_variance(flat_otm(FORWARD, 1.0, RATE, vol), FORWARD, 1.0, discount)
        assert result.fair_volatility == pytest.approx(math.sqrt(result.fair_variance))
        assert result.fair_volatility == pytest.approx(vol, rel=1e-12)

    def test_wings_are_symmetric_in_log_strike(self) -> None:
        """At the money the put and call wings carry nearly the same weight.

        Not exactly the same: the lognormal law is skewed in the strike even
        with a flat volatility, so the put wing is the heavier one. The point
        is that neither wing is negligible, which is what justifies
        integrating both rather than doubling one.
        """
        discount = math.exp(-RATE)
        result = fair_variance(flat_otm(FORWARD, 1.0, RATE, 0.2), FORWARD, 1.0, discount)
        assert result.lower > result.upper > 0.0
        assert result.lower / result.upper < 1.3

    def test_the_put_wing_is_the_expensive_one_to_lose(self) -> None:
        """At equal log-distance, cutting the low strikes costs more.

        ``g(S) = -log(S/F) + S/F - 1`` grows logarithmically below the forward
        and linearly above it, so the convexity the portfolio gives up by
        stopping is larger on the low side. Measured at a flat 20% over one
        year, the ratio is 1.32 at one standard deviation and 2.32 at four, so
        the preference is not only real but strengthens the further out the
        range already reaches. A desk one strike short should buy the low one.
        """
        ratios = []
        for n in (1, 2, 3, 4):
            distance = n * 0.2
            low = truncation_error(
                FORWARD, 1.0, 0.2, FORWARD * math.exp(-distance), FORWARD * math.exp(30.0)
            )
            high = truncation_error(
                FORWARD, 1.0, 0.2, FORWARD * math.exp(-30.0), FORWARD * math.exp(distance)
            )
            ratios.append(low / high)
        assert all(r > 1.0 for r in ratios)
        assert ratios == sorted(ratios)
        assert ratios[0] == pytest.approx(1.324, abs=0.002)
        assert ratios[-1] == pytest.approx(2.320, abs=0.002)

    def test_edge_share_is_negligible_on_a_wide_range(self) -> None:
        discount = math.exp(-RATE)
        wide = fair_variance(
            flat_otm(FORWARD, 1.0, RATE, 0.2), FORWARD, 1.0, discount, width=3.0
        )
        narrow = fair_variance(
            flat_otm(FORWARD, 1.0, RATE, 0.2), FORWARD, 1.0, discount, width=0.5
        )
        assert wide.edge < 1e-20
        assert narrow.edge > 1e-3


class TestHestonExact:
    """The model-free formula against the model's own closed form."""

    @pytest.mark.parametrize("model", MODELS)
    @pytest.mark.parametrize("time", [0.25, 1.0, 5.0])
    def test_recovers_expected_integrated_variance(self, model: Heston, time: float) -> None:
        contract = Contract(FORWARD, FORWARD, time, 0.02, carry=0.0)
        # Six units of log-moneyness: strikes from a quarter of a per cent of
        # the forward to four hundred times it. The module docstring explains
        # why a Gaussian rule of thumb is not enough here.
        wide = fair_variance(
            heston_otm(model, contract),
            contract.forward,
            time,
            contract.discount,
            width=6.0,
            tol=1e-12,
        )
        narrow = fair_variance(
            heston_otm(model, contract),
            contract.forward,
            time,
            contract.discount,
            width=4.0,
            tol=1e-12,
        )
        target = model.expected_integrated_variance(time) / time
        # Six units of log-moneyness is ample at a quarter of a year and not
        # quite ample at five, where the terminal law has had four times as
        # long to spread. The residual is truncation, which is a shortfall, so
        # the narrower range is always the lower number and always below the
        # closed form. The wider one is not asserted to be below it: where six
        # units is ample the residual is rounding rather than truncation and
        # lands a few parts in 1e14 either side, which is the floor and not a
        # bias.
        assert wide.fair_variance == pytest.approx(target, rel=1e-4)
        narrow_error = narrow.fair_variance / target - 1.0
        wide_error = wide.fair_variance / target - 1.0
        # The floor is the quadrature's own tolerance, which is absolute: 1e-12
        # against an integral of ``fair * T / 2``, so between 1e-11 and 1e-9
        # relative over this grid. Below that nothing about truncation is
        # resolved.
        if abs(narrow_error) > 1e-8:
            # Truncation is resolved: it is a shortfall, and widening the
            # range reduces it.
            assert narrow_error < 0.0
            assert narrow_error < wide_error
        else:
            # Four units of log-moneyness was already ample, so both numbers
            # are at the rounding floor and neither ordering nor sign means
            # anything there.
            assert abs(wide_error) < 1e-8
        if time <= 1.0:
            assert wide.fair_variance == pytest.approx(target, rel=1e-8)

    def test_a_gaussian_width_is_not_enough(self) -> None:
        """The error at ten equivalent standard deviations, by vol-of-vol.

        Same equivalent volatility in all three, so the same strike range in
        standard deviations; only the vol-of-vol differs. The residual grows
        by three orders of magnitude across it, which is the whole argument
        for making the width explicit.
        """
        time = 1.0
        contract = Contract(FORWARD, FORWARD, time, 0.02, carry=0.0)
        errors = []
        for sigma in (0.3, 0.5, 0.8):
            model = Heston(v0=0.04, theta=0.04, kappa=1.5, sigma=sigma, rho=-0.7)
            width = 10.0 * model.equivalent_vol(time) * math.sqrt(time)
            result = fair_variance(
                heston_otm(model, contract),
                contract.forward,
                time,
                contract.discount,
                width=width,
                tol=1e-12,
            )
            target = model.expected_integrated_variance(time) / time
            errors.append(abs(result.fair_variance / target - 1.0))
        # Monotone in vol-of-vol, and the span is three orders of magnitude.
        assert errors[0] < errors[1] < errors[2]
        assert errors[2] / errors[0] > 1e3
        # A flat smile at the same volatility and the same width is at the floor.
        flat = fair_variance(
            flat_otm(FORWARD, time, RATE, 0.2), FORWARD, time, math.exp(-RATE), width=2.0
        )
        assert abs(flat.fair_variance / 0.04 - 1.0) < 1e-14


class TestTruncation:
    """The closed-form cost of a finite strike range, against the measurement."""

    @pytest.mark.parametrize("sd_width", [1.0, 1.5, 2.0, 3.0, 4.0])
    def test_prediction_matches_measurement(self, sd_width: float) -> None:
        time, vol = 1.0, 0.2
        discount = math.exp(-RATE * time)
        width = sd_width * vol * math.sqrt(time)
        measured = (
            fair_variance(
                flat_otm(FORWARD, time, RATE, vol), FORWARD, time, discount, width=width
            ).fair_variance
            - vol * vol
        )
        predicted = truncation_error(
            FORWARD, time, vol, FORWARD * math.exp(-width), FORWARD * math.exp(width)
        )
        assert measured == pytest.approx(predicted, rel=1e-9)

    def test_beyond_six_standard_deviations_the_check_is_meaningless(self) -> None:
        """The error falls below the rounding of the thing it is an error in.

        At six standard deviations the truncation is 4.6e-12 against a fair
        variance of 0.04, so the measurement is 1e-10 relative and carries
        about five digits. By eight it is 1.9e-18 and the measured value is
        pure rounding — a prediction can only be checked while what it
        predicts is above the floor.
        """
        time, vol = 1.0, 0.2
        discount = math.exp(-RATE * time)
        at_six = truncation_error(
            FORWARD,
            time,
            vol,
            FORWARD * math.exp(-6.0 * vol),
            FORWARD * math.exp(6.0 * vol),
        )
        at_eight = truncation_error(
            FORWARD,
            time,
            vol,
            FORWARD * math.exp(-8.0 * vol),
            FORWARD * math.exp(8.0 * vol),
        )
        assert -1e-11 < at_six < -1e-12
        assert abs(at_eight) < 1e-17
        measured_eight = (
            fair_variance(
                flat_otm(FORWARD, time, RATE, vol),
                FORWARD,
                time,
                discount,
                width=8.0 * vol,
            ).fair_variance
            - vol * vol
        )
        # Both are noise at this point, and they do not even share a sign
        # reliably; all that can be asserted is that both are rounding.
        assert abs(measured_eight) < 1e-15

    def test_it_is_always_a_shortfall(self) -> None:
        """``g`` is convex, so the tangent is below it and the error is negative."""
        for width in (0.1, 0.5, 1.0, 2.0):
            for vol in (0.1, 0.3, 0.6):
                error = truncation_error(
                    FORWARD,
                    1.0,
                    vol,
                    FORWARD * math.exp(-width),
                    FORWARD * math.exp(width),
                )
                assert error < 0.0

    def test_the_nearer_cut_dominates(self) -> None:
        """Whichever wing stops closer to the money sets the error."""
        near_below = truncation_error(FORWARD, 1.0, 0.2, FORWARD * 0.7, FORWARD * 10.0)
        near_above = truncation_error(FORWARD, 1.0, 0.2, FORWARD * 0.01, FORWARD * 1.3)
        # 1.3 is 0.262 in log-moneyness and 0.7 is -0.357, so the call side is
        # the nearer cut here and the larger error, notwithstanding that the
        # put side is dearer at equal distance.
        assert abs(near_above) > abs(near_below)

    def test_rejects_a_range_that_misses_the_forward(self) -> None:
        with pytest.raises(BadStrip, match="bracket the forward"):
            truncation_error(FORWARD, 1.0, 0.2, FORWARD * 1.1, FORWARD * 2.0)
        with pytest.raises(BadStrip, match="bracket the forward"):
            truncation_error(FORWARD, 1.0, 0.2, FORWARD * 0.5, FORWARD * 0.9)


class TestReplicationValidation:
    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"forward": 0.0}, "forward"),
            ({"forward": -1.0}, "forward"),
            ({"time": 0.0}, "time"),
            ({"discount": 0.0}, "discount"),
            ({"width": 0.0}, "width"),
            ({"tol": 0.0}, "tol"),
            ({"forward": math.inf}, "forward"),
        ],
    )
    def test_rejects_bad_arguments(self, kwargs: dict[str, float], message: str) -> None:
        call = {
            "forward": FORWARD,
            "time": 1.0,
            "discount": 0.97,
            "width": 2.0,
            "tol": 1e-12,
        }
        call.update(kwargs)
        with pytest.raises(BadStrip, match=message):
            fair_variance(
                flat_otm(FORWARD, 1.0, RATE, 0.2),
                call["forward"],
                call["time"],
                call["discount"],
                width=call["width"],
                tol=call["tol"],
            )

    def test_rejects_a_negative_price(self) -> None:
        with pytest.raises(BadStrip, match="non-negative"):
            fair_variance(lambda _strike: -1.0, FORWARD, 1.0, 0.97, width=1.0)

    def test_rejects_a_non_finite_price(self) -> None:
        with pytest.raises(BadStrip, match="non-negative"):
            fair_variance(lambda _strike: math.nan, FORWARD, 1.0, 0.97, width=1.0)


def flat_strip(
    forward: float, time: float, rate: float, vol: float, step: float
) -> Strip:
    """A lattice of strikes at a flat volatility, quoted the way a desk sees it.

    Strikes sit on a multiple of ``step`` regardless of where the forward is,
    which is the point: the forward almost never coincides with a listed
    strike, and the gap between it and the largest strike below it is what the
    centring correction exists for.
    """
    discount = math.exp(-rate * time)
    strikes: list[float] = []
    strike = step * math.ceil(forward * 0.1 / step)
    while strike <= forward * 10.0:
        strikes.append(round(strike, 12))
        strike += step
    reference = max(k for k in strikes if k <= forward)
    quotes: list[tuple[float, float]] = []
    for k in strikes:
        put = bs_price(Inputs.on_future(forward, k, time, rate, vol), OptionType.PUT)
        call = bs_price(Inputs.on_future(forward, k, time, rate, vol), OptionType.CALL)
        if k < reference:
            quotes.append((k, put))
        elif k > reference:
            quotes.append((k, call))
        else:
            quotes.append((k, 0.5 * (put + call)))
    return Strip.from_quotes(quotes, forward, time, discount)


class TestCentring:
    """The sum over listed strikes, and the term that makes it converge."""

    # A forward that is not a round number, so it never lands on a lattice
    # point for any of the spacings below.
    OFF_LATTICE = 102.37
    SPACINGS = (8.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.125)

    def _errors(self, centring: Centring) -> list[float]:
        return [
            fair_variance_from_strip(
                flat_strip(self.OFF_LATTICE, 1.0, RATE, 0.2, step), centring=centring
            ).fair_variance
            - 0.04
            for step in self.SPACINGS
        ]

    @pytest.mark.parametrize("centring", [Centring.QUADRATIC, Centring.EXACT])
    def test_correction_gives_second_order_convergence(self, centring: Centring) -> None:
        """Halving the spacing quarters the error, which is the rectangle rule's own order."""
        errors = self._errors(centring)
        ratios = [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]
        # The first two spacings are coarse enough that the asymptotic order
        # has not set in; from 2.0 down it is clean.
        for ratio in ratios[2:]:
            assert ratio == pytest.approx(4.0, abs=0.05)

    def test_uncorrected_sum_has_no_order_at_all(self) -> None:
        """Its error is set by where the forward falls, not by the spacing.

        Halving ratios of 6.64, 10.71, 2.65, 1.71, 7.23 and 1.46 against the
        corrected sum's 4.00. The uncorrected error carries the centring term
        ``(F/K0 - 1)**2 / T``, and ``K0`` jumps about as the lattice changes,
        so the sequence is not even monotone in the spacing.
        """
        errors = self._errors(Centring.NONE)
        ratios = [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]
        assert max(ratios) > 6.0
        assert min(ratios) < 2.0
        # And it is always worse than the corrected sum at the same spacing.
        corrected = self._errors(Centring.EXACT)
        for raw, fixed in zip(errors[2:], corrected[2:], strict=True):
            assert abs(raw) > abs(fixed)

    def test_the_market_convention_is_the_leading_term_only(self) -> None:
        """``(F/K0-1)**2`` against ``2(F/K0-1-log(F/K0))``, measured.

        The convention overstates the true centring term by 0.67% at a gap of
        one per cent, 6.6% at ten per cent and 32% at a half. Both agree to
        third order, which is why a fine lattice cannot tell them apart.
        """
        for gap, expected in (
            (0.001, 1.000667),
            (0.01, 1.006661),
            (0.1, 1.066139),
            (0.5, 1.322263),
        ):
            ratio = 1.0 + gap
            quadratic = gap * gap
            exact = 2.0 * (gap - math.log(ratio))
            assert quadratic / exact == pytest.approx(expected, rel=1e-6)

    def test_the_sum_overstates_before_correction(self) -> None:
        strip = flat_strip(self.OFF_LATTICE, 1.0, RATE, 0.2, 5.0)
        result = fair_variance_from_strip(strip, centring=Centring.EXACT)
        assert result.raw > result.fair_variance
        assert result.correction > 0.0
        assert result.gap == pytest.approx(self.OFF_LATTICE / result.reference - 1.0)
        assert result.reference <= self.OFF_LATTICE
        assert result.strikes == len(strip.strikes)
        assert result.fair_volatility == pytest.approx(math.sqrt(result.fair_variance))

    def test_a_forward_on_the_lattice_needs_no_correction(self) -> None:
        strip = flat_strip(100.0, 1.0, RATE, 0.2, 5.0)
        assert strip.reference == 100.0
        for centring in Centring:
            result = fair_variance_from_strip(strip, centring=centring)
            assert result.correction == 0.0
            assert result.gap == 0.0

    def test_strip_geometry(self) -> None:
        strip = Strip.from_quotes(
            [(90.0, 1.0), (100.0, 2.0), (115.0, 1.5), (130.0, 0.5)], 103.0, 1.0, 0.97
        )
        assert strip.reference == 100.0
        assert strip.widths == (10.0, 12.5, 15.0, 15.0)
        low, high = strip.log_range
        assert low == pytest.approx(math.log(90.0 / 103.0))
        assert high == pytest.approx(math.log(130.0 / 103.0))

    def test_quotes_are_sorted_on_the_way_in(self) -> None:
        strip = Strip.from_quotes(
            [(130.0, 0.5), (90.0, 1.0), (115.0, 1.5), (100.0, 2.0)], 103.0, 1.0, 0.97
        )
        assert strip.strikes == (90.0, 100.0, 115.0, 130.0)
        assert strip.prices == (1.0, 2.0, 1.5, 0.5)


class TestStripValidation:
    def test_rejects_mismatched_lengths(self) -> None:
        with pytest.raises(BadStrip, match="must match"):
            Strip((90.0, 100.0, 110.0), (1.0, 2.0), 100.0, 1.0, 0.97)

    def test_rejects_too_few_strikes(self) -> None:
        with pytest.raises(BadStrip, match="at least three"):
            Strip((90.0, 110.0), (1.0, 1.0), 100.0, 1.0, 0.97)

    def test_rejects_unsorted_or_repeated_strikes(self) -> None:
        with pytest.raises(BadStrip, match="strictly increasing"):
            Strip((90.0, 110.0, 100.0), (1.0, 1.0, 1.0), 100.0, 1.0, 0.97)
        with pytest.raises(BadStrip, match="strictly increasing"):
            Strip((90.0, 100.0, 100.0, 110.0), (1.0, 1.0, 1.0, 1.0), 100.0, 1.0, 0.97)

    def test_rejects_a_negative_price(self) -> None:
        with pytest.raises(BadStrip, match="non-negative"):
            Strip((90.0, 100.0, 110.0), (1.0, -2.0, 1.0), 100.0, 1.0, 0.97)

    def test_rejects_strikes_that_do_not_bracket_the_forward(self) -> None:
        with pytest.raises(BadStrip, match="bracket the forward"):
            Strip((110.0, 120.0, 130.0), (1.0, 1.0, 1.0), 100.0, 1.0, 0.97)
        with pytest.raises(BadStrip, match="bracket the forward"):
            Strip((70.0, 80.0, 90.0), (1.0, 1.0, 1.0), 100.0, 1.0, 0.97)

    def test_rejects_a_bad_market(self) -> None:
        for kwargs in ({"forward": -1.0}, {"time": 0.0}, {"discount": 0.0}):
            market = {"forward": 100.0, "time": 1.0, "discount": 0.97}
            market.update(kwargs)
            with pytest.raises(BadStrip):
                Strip(
                    (90.0, 100.0, 110.0),
                    (1.0, 1.0, 1.0),
                    market["forward"],
                    market["time"],
                    market["discount"],
                )


class TestIntegratedVarianceVariance:
    """The closed form against the covariance it was integrated from."""

    @staticmethod
    def by_quadrature(model: Heston, time: float) -> float:
        """``2 int int e^{-kappa(t-s)} Var(V_s) ds dt``, evaluated numerically.

        Shares the covariance structure with the closed form and none of the
        algebra, which is the only part that could be wrong.
        """
        kappa = model.kappa
        sigma_sq = model.sigma * model.sigma

        def var_v(s: float) -> float:
            return (sigma_sq * model.theta / (2.0 * kappa)) * (
                1.0 - math.exp(-kappa * s)
            ) ** 2 + (sigma_sq * model.v0 / kappa) * (
                math.exp(-kappa * s) - math.exp(-2.0 * kappa * s)
            )

        def inner(t: float) -> float:
            return adaptive_quad(
                lambda s: math.exp(-kappa * (t - s)) * var_v(s), 0.0, t, 1e-14
            )

        return 2.0 * adaptive_quad(inner, 0.0, time, 1e-13)

    @pytest.mark.parametrize("model", MODELS)
    @pytest.mark.parametrize("time", [0.25, 1.0, 5.0])
    def test_against_double_quadrature(self, model: Heston, time: float) -> None:
        closed = integrated_variance_variance(model, time)
        assert closed == pytest.approx(self.by_quadrature(model, time), rel=1e-11)

    def test_stationary_case_collapses(self) -> None:
        """At ``v0 = theta`` the two coefficients combine into the known form."""
        model = Heston(v0=0.04, theta=0.04, kappa=1.3, sigma=0.45, rho=-0.6)
        for time in (0.1, 0.5, 2.0, 7.0):
            kappa = model.kappa
            expected = (
                model.sigma**2
                * model.theta
                * (
                    2.0 * kappa * time
                    - 3.0
                    + 4.0 * math.exp(-kappa * time)
                    - math.exp(-2.0 * kappa * time)
                )
                / (2.0 * kappa**3)
            )
            assert integrated_variance_variance(model, time) == pytest.approx(
                expected, rel=1e-13
            )

    def test_zero_vol_of_vol_leaves_no_variance(self) -> None:
        model = Heston(v0=0.04, theta=0.05, kappa=1.0, sigma=0.0, rho=0.0)
        for time in (0.0, 0.5, 3.0):
            assert integrated_variance_variance(model, time) == 0.0

    def test_grows_with_the_horizon(self) -> None:
        for model in MODELS:
            values = [integrated_variance_variance(model, t) for t in (0.25, 1.0, 4.0, 10.0)]
            assert values == sorted(values)
            assert all(v > 0.0 for v in values)

    def test_rejects_a_negative_horizon(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            integrated_variance_variance(STATIONARY, -1.0)


class TestVolatilitySwap:
    """The convexity gap, and how much the second-order term overstates it."""

    @staticmethod
    def simulate(
        model: Heston, time: float, paths: int, steps: int, seed: int
    ) -> tuple[float, float]:
        """``E[sqrt(RV)]`` and its standard error, from simulated variance paths."""
        rng = random.Random(seed)
        step = time / steps
        total = 0.0
        total_sq = 0.0
        for _ in range(paths):
            path = variance_path(model, time, steps, [rng.random() for _ in range(steps)])
            previous = model.v0
            integral = 0.0
            for value in path:
                integral += 0.5 * (previous + value) * step
                previous = value
            root = math.sqrt(integral / time)
            total += root
            total_sq += root * root
        mean = total / paths
        return mean, math.sqrt(max(total_sq / paths - mean * mean, 0.0) / paths)

    def test_strike_is_below_the_square_root_of_fair_variance(self) -> None:
        for model in MODELS:
            for time in (0.25, 1.0, 5.0):
                result = volatility_swap_strike(model, time)
                assert result.convexity > 0.0
                assert result.strike < math.sqrt(result.variance_strike)
                assert result.strike + result.convexity == pytest.approx(
                    math.sqrt(result.variance_strike)
                )
                assert result.variance_strike == pytest.approx(
                    model.expected_integrated_variance(time) / time
                )

    def test_dispersion_is_the_squared_coefficient_of_variation(self) -> None:
        for model in MODELS:
            time = 1.5
            result = volatility_swap_strike(model, time)
            mean = model.expected_integrated_variance(time) / time
            expected = integrated_variance_variance(model, time) / (time * time) / mean**2
            assert result.dispersion == pytest.approx(expected, rel=1e-14)

    def test_no_vol_of_vol_means_no_convexity(self) -> None:
        model = Heston(v0=0.04, theta=0.04, kappa=1.0, sigma=0.0, rho=0.0)
        result = volatility_swap_strike(model, 1.0)
        assert result.convexity == 0.0
        assert result.strike == pytest.approx(0.2, rel=1e-14)

    def test_second_order_overstates_the_discount(self) -> None:
        """Against simulation, and the overstatement is many standard errors.

        Twelve thousand paths of eighty steps each. The step count is not the
        binding constraint: at 25,000 paths the simulated mean moves by less
        than one standard error between 50 and 800 steps, because the
        trapezoid of an exact-marginal path has no bias in the first moment
        and the second is matched to within sampling error too. What moves the
        answer is the truncated series.
        """
        for seed, model in enumerate(MODELS):
            time = 1.0
            predicted = volatility_swap_strike(model, time)
            simulated, error = self.simulate(model, time, 12_000, 80, 4000 + seed)
            root = math.sqrt(predicted.variance_strike)
            # Right sign: both sit below the square root of fair variance.
            assert simulated < root
            # Wrong size: the predicted discount is the larger one, by enough
            # that sampling cannot account for it.
            assert predicted.strike < simulated
            assert (simulated - predicted.strike) / error > 3.0
            ratio = predicted.convexity / (root - simulated)
            assert 1.1 < ratio < 1.6

    def test_rejects_a_zero_horizon(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            volatility_swap_strike(STATIONARY, 0.0)
