"""Heston's model: the transform, its branch, and the two pricing routes."""

from __future__ import annotations

import math
from itertools import pairwise
from typing import Any

import pytest
from mpmath import mp

from moneyness import price as bs_price
from moneyness.bsm import OptionType
from moneyness.heston import (
    Branch,
    Contract,
    Heston,
    branch_discrepancy,
    char_func,
    gil_pelaez_price,
    lewis_price,
    price,
    smile,
)
from moneyness.quadrature import semi_infinite_quad

# A parameter set with a pronounced negative skew and a Feller condition that
# fails, which is the usual situation when a model is fitted to equity index
# quotes. Reusing one set across the file keeps the numbers below comparable.
REFERENCE = Heston(v0=0.04, kappa=1.5768, theta=0.04, sigma=0.5751, rho=-0.5711)

STRIKES = (50.0, 70.0, 80.0, 95.0, 100.0, 105.0, 125.0, 160.0, 200.0)
MATURITIES = (0.02, 0.05, 0.25, 1.0, 3.0, 10.0)


class TestParameters:
    def test_feller_condition_is_reported_not_enforced(self) -> None:
        assert REFERENCE.feller == pytest.approx(
            2.0 * 1.5768 * 0.04 - 0.5751**2, abs=1e-15
        )
        assert not REFERENCE.satisfies_feller
        assert Heston(0.04, 4.0, 0.04, 0.3, -0.5).satisfies_feller

    def test_expected_variance_decays_from_v0_towards_theta(self) -> None:
        model = Heston(0.09, 2.0, 0.04, 0.3, -0.5)
        assert model.expected_variance(0.0) == pytest.approx(0.09, abs=1e-15)
        assert model.expected_variance(100.0) == pytest.approx(0.04, abs=1e-12)
        assert model.expected_variance(1.0) == pytest.approx(
            0.04 + 0.05 * math.exp(-2.0), abs=1e-15
        )

    def test_integrated_variance_is_the_integral_of_the_expectation(self) -> None:
        # Checked against a quadrature of E[V_s] rather than against the same
        # algebra written twice.
        model = Heston(0.09, 2.0, 0.04, 0.3, -0.5)
        mp.dps = 40
        want = float(mp.quad(lambda s: 0.04 + 0.05 * mp.e ** (-2 * s), [0, 1.7]))
        assert model.expected_integrated_variance(1.7) == pytest.approx(want, abs=1e-13)

    def test_integrated_variance_is_zero_at_zero_time(self) -> None:
        assert REFERENCE.expected_integrated_variance(0.0) == 0.0
        assert REFERENCE.equivalent_vol(0.0) == 0.0

    def test_equivalent_vol_is_the_root_mean_variance(self) -> None:
        flat = Heston(0.04, 2.0, 0.04, 0.3, 0.0)
        assert flat.equivalent_vol(2.5) == pytest.approx(0.2, abs=1e-15)

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("v0", -1e-9, "v0 must be non-negative"),
            ("theta", -1e-9, "theta must be non-negative"),
            ("sigma", -1e-9, "sigma must be non-negative"),
            ("kappa", 0.0, "kappa must be positive"),
            ("kappa", -1.0, "kappa must be positive"),
            ("rho", 1.0000001, r"rho must be in \[-1, 1\]"),
            ("rho", -1.0000001, r"rho must be in \[-1, 1\]"),
        ],
    )
    def test_a_parameter_outside_its_range_is_refused(
        self, field: str, value: float, message: str
    ) -> None:
        kwargs = {"v0": 0.04, "kappa": 1.5, "theta": 0.04, "sigma": 0.5, "rho": -0.5}
        kwargs[field] = value
        with pytest.raises(ValueError, match=message):
            Heston(**kwargs)

    @pytest.mark.parametrize("field", ["v0", "kappa", "theta", "sigma", "rho"])
    @pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
    def test_a_non_finite_parameter_is_refused(self, field: str, value: float) -> None:
        kwargs = {"v0": 0.04, "kappa": 1.5, "theta": 0.04, "sigma": 0.5, "rho": -0.5}
        kwargs[field] = value
        with pytest.raises(ValueError, match="must be finite"):
            Heston(**kwargs)

    def test_the_boundary_correlations_are_allowed(self) -> None:
        assert Heston(0.04, 1.5, 0.04, 0.5, -1.0).rho == -1.0
        assert Heston(0.04, 1.5, 0.04, 0.5, 1.0).rho == 1.0


class TestContract:
    def test_the_forward_and_discount_follow_the_carry_convention(self) -> None:
        c = Contract(100.0, 90.0, 2.0, 0.05, carry=0.02)
        assert c.forward == pytest.approx(100.0 * math.exp(0.04), abs=1e-13)
        assert c.discount == pytest.approx(math.exp(-0.10), abs=1e-15)

    def test_the_carry_defaults_to_the_rate(self) -> None:
        c = Contract(100.0, 90.0, 2.0, 0.05)
        assert c.b == 0.05
        assert c.forward == pytest.approx(100.0 * math.exp(0.10), abs=1e-12)

    def test_it_converts_to_the_lognormal_inputs_unchanged(self) -> None:
        c = Contract(100.0, 90.0, 2.0, 0.05, carry=0.02)
        inputs = c.with_vol(0.3)
        assert (inputs.spot, inputs.strike, inputs.time, inputs.rate, inputs.vol) == (
            100.0,
            90.0,
            2.0,
            0.05,
            0.3,
        )
        assert inputs.b == 0.02

    def test_it_converts_to_a_quote_carrying_a_price(self) -> None:
        quote = Contract(100.0, 90.0, 2.0, 0.05, carry=0.02).quote(13.5)
        assert quote.price == 13.5
        assert quote.b == 0.02

    @pytest.mark.parametrize(
        ("field", "value"),
        [("spot", -1.0), ("strike", -1.0), ("time", -1.0)],
    )
    def test_a_negative_input_is_refused(self, field: str, value: float) -> None:
        kwargs = {"spot": 100.0, "strike": 100.0, "time": 1.0, "rate": 0.02}
        kwargs[field] = value
        with pytest.raises(ValueError, match="non-negative"):
            Contract(**kwargs)

    def test_a_non_finite_input_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must be finite"):
            Contract(100.0, 100.0, 1.0, math.nan)


class TestCharacteristicFunction:
    def test_it_is_one_at_zero(self) -> None:
        for time in MATURITIES:
            assert char_func(REFERENCE, 0.0, time) == 1.0 + 0j

    @pytest.mark.parametrize("time", MATURITIES)
    def test_the_martingale_condition_holds_at_minus_i(self, time: float) -> None:
        # phi(-i) = E[exp(X)] = 1 because the forward has been divided out.
        # Worth being precise about what this does and does not show: at u = -i
        # the combination iu + u^2 is exactly zero, and the transform is then
        # structurally one in this parameterisation rather than one by
        # arithmetic. It is a check that the grouping is the right shape, not
        # a check on the values it produces.
        assert char_func(REFERENCE, -1j, time) == pytest.approx(1.0 + 0j, abs=1e-13)

    @pytest.mark.parametrize("epsilon", [1e-3, 1e-5, 1e-7])
    def test_it_approaches_the_martingale_point_continuously(self, epsilon: float) -> None:
        # Just off -i the shortcut does not fire and the full expression runs.
        # Its value has to arrive at one linearly, which is the check the
        # martingale condition itself cannot give.
        value = char_func(REFERENCE, -1j * (1.0 + epsilon), 3.0)
        assert value.imag == pytest.approx(0.0, abs=1e-14)
        assert (value.real - 1.0) / epsilon == pytest.approx(0.05146, rel=1e-3)

    @pytest.mark.parametrize("time", [0.0, 0.0])
    def test_zero_time_is_the_degenerate_transform(self, time: float) -> None:
        assert char_func(REFERENCE, 3.7, time) == 1.0 + 0j

    @pytest.mark.parametrize("time", MATURITIES)
    @pytest.mark.parametrize("u", [0.1, 1.0, 5.0, 20.0, 100.0])
    def test_the_modulus_never_exceeds_one_on_the_real_line(
        self, time: float, u: float
    ) -> None:
        assert abs(char_func(REFERENCE, u, time)) <= 1.0 + 1e-14

    @pytest.mark.parametrize("time", MATURITIES)
    @pytest.mark.parametrize("u", [0.1, 1.0, 5.0, 20.0, 100.0])
    @pytest.mark.parametrize("shift", [-0.5j, -1j])
    def test_the_modulus_never_exceeds_one_on_the_pricing_lines(
        self, time: float, u: float, shift: complex
    ) -> None:
        # Both pricing routes evaluate off the real axis. |phi(u - i/2)| and
        # |phi(u - i)| are bounded by E[e^{X/2}] and E[e^X], both of which are
        # at most one, so a value above one is a branch that has gone wrong.
        assert abs(char_func(REFERENCE, u + shift, time)) <= 1.0 + 1e-14

    def test_it_is_continuous_across_the_strip_the_pricing_uses(self) -> None:
        # A branch cut crossed between the real axis and -i would show up as a
        # step far larger than the neighbouring ones.
        for u in (0.5, 1.0, 3.0, 10.0):
            values = [char_func(REFERENCE, complex(u, -j / 40.0), 4.0) for j in range(41)]
            steps = [abs(b - a) for a, b in pairwise(values)]
            assert max(steps) < 8.0 * (sum(steps) / len(steps))

    @pytest.mark.parametrize("time", [0.25, 1.0, 7.0])
    def test_zero_volatility_of_variance_is_the_lognormal_transform(
        self, time: float
    ) -> None:
        model = Heston(0.05, 1.5, 0.04, 0.0, -0.5)
        total = model.expected_integrated_variance(time)
        for u in (0.3, 1.0, 4.0):
            m = 1j * u + u * u
            want = complex(math.exp(-0.5 * (m * total).real), 0.0) * complex(
                math.cos(-0.5 * (m * total).imag), math.sin(-0.5 * (m * total).imag)
            )
            assert char_func(model, u, time) == pytest.approx(want, abs=1e-14)

    @pytest.mark.parametrize("time", [0.1, 1.0, 5.0])
    def test_the_first_moment_is_minus_half_the_integrated_variance(
        self, time: float
    ) -> None:
        # E[X] = -E[int V]/2, because X is the log of a martingale. Read off
        # the transform by a central difference, which touches none of the
        # code that computes the integrated variance.
        h = 1e-5
        derivative = (char_func(REFERENCE, h, time) - char_func(REFERENCE, -h, time)) / (
            2.0 * h
        )
        got = (derivative / 1j).real
        assert got == pytest.approx(
            -0.5 * REFERENCE.expected_integrated_variance(time), abs=1e-9
        )

    def test_the_variance_of_the_log_price_collapses_to_the_total_variance(self) -> None:
        # With no volatility of variance the log price is normal with variance
        # equal to the integrated variance, which the transform must reproduce.
        model = Heston(0.05, 1.5, 0.04, 0.0, -0.5)
        h = 1e-4
        second = (
            char_func(model, h, 2.0) - 2.0 * char_func(model, 0.0, 2.0) + char_func(model, -h, 2.0)
        ) / (h * h)
        mean = ((char_func(model, h, 2.0) - char_func(model, -h, 2.0)) / (2.0 * h) / 1j).real
        variance = -second.real - mean * mean
        assert variance == pytest.approx(model.expected_integrated_variance(2.0), abs=1e-7)

    def test_a_negative_maturity_is_refused(self) -> None:
        with pytest.raises(ValueError, match="time must be non-negative"):
            char_func(REFERENCE, 1.0, -0.5)


class TestBranch:
    """The textbook grouping fails; this measures where."""

    @pytest.mark.parametrize("time", [0.25, 1.0, 2.0])
    def test_the_two_groupings_agree_before_the_cut_is_crossed(self, time: float) -> None:
        for u in (0.3, 1.0, 3.0):
            assert branch_discrepancy(REFERENCE, u, time) < 1e-12

    def test_the_textbook_grouping_breaks_at_a_long_maturity(self) -> None:
        # For these parameters and u = 1 the complex logarithm's argument
        # completes a circuit of the origin between 8.5 and 9 years, and the
        # principal branch jumps by 2 pi i. The price error that follows is not
        # small: the transform is out by order one.
        assert branch_discrepancy(REFERENCE, 1.0, 8.5) < 1e-12
        assert branch_discrepancy(REFERENCE, 1.0, 9.0) > 1.0

    def test_the_stable_grouping_does_not_break_anywhere_along_the_way(self) -> None:
        # Sampled finely across the maturity where the textbook form fails: the
        # stable one stays on the martingale condition throughout.
        for j in range(1, 401):
            time = j / 20.0
            assert char_func(REFERENCE, -1j, time, Branch.STABLE) == pytest.approx(
                1.0 + 0j, abs=1e-11
            )

    def test_the_maturity_it_survives_to_falls_as_the_frequency_rises(self) -> None:
        # This is why the failure is easy to miss. The logarithm's argument
        # circles the origin at a rate that grows with u, so the textbook form
        # breaks in the *tail* of the pricing integrand while the low
        # frequencies that carry most of the price are still fine. The price
        # then looks plausible and is wrong.
        def first_failure(u: float) -> float:
            for j in range(1, 2401):
                time = j / 40.0
                if branch_discrepancy(REFERENCE, u, time) > 1e-6:
                    return time
            raise AssertionError(f"the textbook grouping never failed at u = {u}")

        at_one = first_failure(1.0)
        at_three = first_failure(3.0)
        at_ten = first_failure(10.0)
        assert at_one == pytest.approx(8.65, abs=0.05)
        assert at_three == pytest.approx(2.875, abs=0.05)
        assert at_ten == pytest.approx(1.55, abs=0.05)
        assert at_ten < at_three < at_one

    def test_the_textbook_grouping_is_degenerate_at_zero_volatility_of_variance(
        self,
    ) -> None:
        # a - d cancels exactly there, which is the subtraction the stable
        # grouping avoids. Saying so beats dividing by zero.
        model = Heston(0.04, 1.5, 0.04, 1e-300, -0.5)
        with pytest.raises(ValueError, match="degenerate"):
            char_func(model, 1.0, 1.0, Branch.TEXTBOOK)


class TestZeroVolatilityOfVariance:
    """The one case with an exact answer from outside the model."""

    @pytest.mark.parametrize("strike", STRIKES)
    @pytest.mark.parametrize("option", list(OptionType))
    def test_it_is_exactly_the_lognormal_price(
        self, strike: float, option: OptionType
    ) -> None:
        model = Heston(0.05, 1.5, 0.04, 0.0, -0.5)
        contract = Contract(100.0, strike, 2.0, 0.02, carry=0.01)
        want = bs_price(contract.with_vol(model.equivalent_vol(2.0)), option)
        assert lewis_price(model, contract, option) == pytest.approx(want, abs=1e-12)
        assert gil_pelaez_price(model, contract, option) == pytest.approx(want, abs=1e-12)

    def test_the_smile_it_produces_is_flat(self) -> None:
        model = Heston(0.05, 1.5, 0.04, 0.0, 0.0)
        contract = Contract(100.0, 100.0, 2.0, 0.02, carry=0.0)
        points = smile(model, contract, [60.0, 80.0, 100.0, 130.0, 170.0])
        flat = model.equivalent_vol(2.0)
        for point in points:
            assert point.implied_vol is not None
            assert point.implied_vol == pytest.approx(flat, abs=1e-9)

    def test_the_error_is_first_order_in_sigma_when_the_shocks_are_correlated(
        self,
    ) -> None:
        # Halving the volatility of variance halves the gap, because the
        # leading correction is the skew term and that is linear in rho sigma.
        contract = Contract(100.0, 100.0, 1.0, 0.03, carry=0.01)
        ratios = []
        for sigma in (1e-5, 2e-5, 4e-5, 8e-5):
            model = Heston(0.05, 1.5, 0.04, sigma, -0.5)
            reference = bs_price(contract.with_vol(model.equivalent_vol(1.0)), OptionType.CALL)
            ratios.append((lewis_price(model, contract) - reference) / sigma)
        assert all(abs(r - ratios[0]) < 0.01 * abs(ratios[0]) for r in ratios)
        assert ratios[0] == pytest.approx(-0.18779, abs=2e-4)

    def test_the_error_is_second_order_in_sigma_when_they_are_not(self) -> None:
        # With no correlation the skew term is absent and the gap is quadratic,
        # so the same halving cuts it by four.
        contract = Contract(100.0, 100.0, 1.0, 0.03, carry=0.01)
        ratios = []
        for sigma in (1e-5, 2e-5, 4e-5, 8e-5):
            model = Heston(0.05, 1.5, 0.04, sigma, 0.0)
            reference = bs_price(contract.with_vol(model.equivalent_vol(1.0)), OptionType.CALL)
            ratios.append((lewis_price(model, contract) - reference) / (sigma * sigma))
        assert all(abs(r - ratios[0]) < 0.01 * abs(ratios[0]) for r in ratios)
        assert ratios[0] == pytest.approx(-2.95778, abs=2e-4)


class TestTheTwoRoutesAgree:
    @pytest.mark.parametrize("time", MATURITIES)
    @pytest.mark.parametrize("strike", STRIKES)
    @pytest.mark.parametrize("option", list(OptionType))
    def test_lewis_and_gil_pelaez_give_the_same_price(
        self, time: float, strike: float, option: OptionType
    ) -> None:
        contract = Contract(100.0, strike, time, 0.02, carry=0.0)
        one = lewis_price(REFERENCE, contract, option)
        two = gil_pelaez_price(REFERENCE, contract, option)
        assert one == pytest.approx(two, abs=2e-11)

    @pytest.mark.parametrize("rho", [-0.9, -0.3, 0.0, 0.4, 0.85])
    def test_they_agree_across_the_correlation(self, rho: float) -> None:
        model = Heston(0.06, 2.2, 0.03, 0.7, rho)
        for strike in (70.0, 100.0, 140.0):
            contract = Contract(100.0, strike, 1.5, 0.01, carry=0.0)
            assert lewis_price(model, contract) == pytest.approx(
                gil_pelaez_price(model, contract), abs=2e-11
            )

    def test_price_is_the_lewis_route(self) -> None:
        contract = Contract(100.0, 115.0, 1.0, 0.02, carry=0.0)
        assert price(REFERENCE, contract) == lewis_price(REFERENCE, contract)


class TestAgainstThirtyDigits:
    """The same integrals, re-derived in mpmath, sharing no code."""

    @staticmethod
    def _phi(model: Heston, u: Any, time: float) -> Any:
        z = mp.mpc(u)
        m = 1j * z + z * z
        a = model.kappa - model.rho * model.sigma * 1j * z
        d = mp.sqrt(a * a + model.sigma**2 * m)
        root = -(model.sigma**2) * m / (a + d)
        g = root / (a + d)
        decay = mp.e ** (-d * time)
        c = (model.kappa * model.theta / model.sigma**2) * (
            root * time - 2 * mp.log((1 - g * decay) / (1 - g))
        )
        return mp.e ** (c + (root / model.sigma**2) * (1 - decay) / (1 - g * decay) * model.v0)

    @pytest.mark.parametrize("strike", [70.0, 100.0, 140.0])
    @pytest.mark.parametrize("time", [0.25, 2.0])
    def test_the_lewis_integral_at_thirty_digits(self, strike: float, time: float) -> None:
        mp.dps = 30
        forward, rate = 100.0, 0.02
        k = math.log(strike / forward)
        half = mp.mpf(0.5)

        def integrand(u: Any) -> Any:
            shifted = self._phi(REFERENCE, u - half * 1j, time)
            return mp.re(mp.e ** (-1j * u * k) * shifted) / (u * u + mp.mpf(0.25))

        integral = mp.quad(integrand, [0, 1, 5, 20, 100, mp.inf])
        want = float(
            mp.e ** (-rate * time) * (forward - mp.sqrt(forward * strike) / mp.pi * integral)
        )
        contract = Contract(forward, strike, time, rate, carry=0.0)
        assert lewis_price(REFERENCE, contract) == pytest.approx(want, abs=1e-11)


class TestBoundsAndShape:
    @pytest.mark.parametrize("strike", STRIKES)
    @pytest.mark.parametrize("time", [0.25, 1.0, 5.0])
    def test_put_call_parity(self, strike: float, time: float) -> None:
        contract = Contract(100.0, strike, time, 0.03, carry=0.01)
        call = price(REFERENCE, contract, OptionType.CALL)
        put = price(REFERENCE, contract, OptionType.PUT)
        want = contract.discount * (contract.forward - strike)
        assert call - put == pytest.approx(want, abs=1e-10)

    @pytest.mark.parametrize("time", [0.1, 1.0, 5.0])
    def test_a_call_sits_between_its_arbitrage_bounds(self, time: float) -> None:
        for strike in STRIKES:
            contract = Contract(100.0, strike, time, 0.03, carry=0.01)
            call = price(REFERENCE, contract, OptionType.CALL)
            floor = contract.discount * max(contract.forward - strike, 0.0)
            assert floor - 1e-12 <= call <= contract.discount * contract.forward + 1e-12

    @pytest.mark.parametrize("time", [0.1, 1.0, 5.0])
    def test_a_call_falls_with_the_strike(self, time: float) -> None:
        values = [
            price(REFERENCE, Contract(100.0, k, time, 0.03, carry=0.01), OptionType.CALL)
            for k in STRIKES
        ]
        assert all(b < a for a, b in pairwise(values))

    @pytest.mark.parametrize("time", [0.1, 1.0, 5.0])
    def test_the_butterfly_is_non_negative(self, time: float) -> None:
        # Convexity in the strike. A negative butterfly is an arbitrage and is
        # the usual first symptom of a transform that has been mis-integrated.
        step = 4.0
        for centre in (70.0, 85.0, 100.0, 115.0, 140.0):
            wing_low, middle, wing_high = (
                price(
                    REFERENCE,
                    Contract(100.0, centre + offset, time, 0.03, carry=0.01),
                    OptionType.CALL,
                )
                for offset in (-step, 0.0, step)
            )
            assert wing_low - 2.0 * middle + wing_high >= -1e-10

    def test_an_expired_option_is_its_intrinsic_value(self) -> None:
        assert price(REFERENCE, Contract(100.0, 90.0, 0.0, 0.03), OptionType.CALL) == 10.0
        assert price(REFERENCE, Contract(100.0, 90.0, 0.0, 0.03), OptionType.PUT) == 0.0
        assert price(REFERENCE, Contract(80.0, 90.0, 0.0, 0.03), OptionType.PUT) == 10.0

    def test_a_zero_strike_call_is_the_discounted_forward(self) -> None:
        contract = Contract(100.0, 0.0, 2.0, 0.03, carry=0.01)
        assert price(REFERENCE, contract, OptionType.CALL) == pytest.approx(
            contract.discount * contract.forward, abs=1e-14
        )
        assert price(REFERENCE, contract, OptionType.PUT) == 0.0

    def test_a_worthless_underlying_leaves_only_the_put(self) -> None:
        contract = Contract(0.0, 90.0, 2.0, 0.03, carry=0.01)
        assert price(REFERENCE, contract, OptionType.CALL) == 0.0
        assert price(REFERENCE, contract, OptionType.PUT) == pytest.approx(
            contract.discount * 90.0, abs=1e-14
        )

    def test_a_deep_out_of_the_money_call_is_small_and_positive(self) -> None:
        contract = Contract(100.0, 400.0, 0.25, 0.0, carry=0.0)
        value = price(REFERENCE, contract, OptionType.CALL)
        assert 0.0 <= value < 1e-3

    def test_prices_rise_with_the_initial_variance(self) -> None:
        contract = Contract(100.0, 100.0, 1.0, 0.02, carry=0.0)
        values = [
            price(Heston(v0, 1.5768, 0.04, 0.5751, -0.5711), contract)
            for v0 in (0.01, 0.02, 0.04, 0.08, 0.16)
        ]
        assert all(b > a for a, b in pairwise(values))


class TestSmile:
    def test_a_negative_correlation_tilts_the_smile_down(self) -> None:
        contract = Contract(100.0, 100.0, 1.0, 0.0, carry=0.0)
        points = smile(REFERENCE, contract, [80.0, 100.0, 125.0])
        vols = [p.implied_vol for p in points]
        assert all(v is not None for v in vols)
        assert vols[0] > vols[1] > vols[2]  # type: ignore[operator]

    def test_a_positive_correlation_tilts_it_the_other_way(self) -> None:
        model = Heston(0.04, 1.5768, 0.04, 0.5751, 0.5711)
        contract = Contract(100.0, 100.0, 1.0, 0.0, carry=0.0)
        vols = [p.implied_vol for p in smile(model, contract, [80.0, 100.0, 125.0])]
        assert all(v is not None for v in vols)
        assert vols[0] < vols[1] < vols[2]  # type: ignore[operator]

    def test_the_skew_flattens_with_maturity(self) -> None:
        # Mean reversion pulls the variance back to its long-run level, so the
        # correlation has proportionally less to act on as the horizon grows.
        slopes = []
        for time in (0.25, 1.0, 5.0):
            contract = Contract(100.0, 100.0, time, 0.0, carry=0.0)
            low, high = smile(REFERENCE, contract, [90.0, 110.0])
            assert low.implied_vol is not None and high.implied_vol is not None
            slopes.append(abs(high.implied_vol - low.implied_vol))
        assert slopes[0] > slopes[1] > slopes[2]

    def test_each_point_reprices_to_the_price_it_came_from(self) -> None:
        contract = Contract(100.0, 100.0, 1.0, 0.01, carry=0.0)
        for point in smile(REFERENCE, contract, [70.0, 90.0, 110.0, 150.0]):
            assert point.implied_vol is not None
            one = Contract(100.0, point.strike, 1.0, 0.01, carry=0.0)
            assert bs_price(one.with_vol(point.implied_vol), point.option) == pytest.approx(
                point.price, abs=1e-10
            )

    def test_strikes_are_priced_on_the_out_of_the_money_side(self) -> None:
        contract = Contract(100.0, 100.0, 1.0, 0.0, carry=0.05)
        forward = contract.forward
        for point in smile(REFERENCE, contract, [90.0, 100.0, 105.0, 130.0]):
            expected = OptionType.CALL if point.strike >= forward else OptionType.PUT
            assert point.option is expected

    def test_an_empty_ladder_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            smile(REFERENCE, Contract(100.0, 100.0, 1.0, 0.0), [])

    @pytest.mark.parametrize("strike", [0.0, -5.0])
    def test_a_non_positive_strike_is_refused(self, strike: float) -> None:
        with pytest.raises(ValueError, match="strikes must be positive"):
            smile(REFERENCE, Contract(100.0, 100.0, 1.0, 0.0), [90.0, strike])


class TestTheFiguresInTheReadme:
    """Every number the documentation states, recomputed here.

    A figure in prose is a claim, and a claim that nothing evaluates drifts
    away from the code the first time the code changes.
    """

    def test_the_smile_in_the_usage_example(self) -> None:
        contract = Contract(spot=100.0, strike=100.0, time=1.0, rate=0.0, carry=0.0)
        vols = [p.implied_vol for p in smile(REFERENCE, contract, [80.0, 100.0, 125.0])]
        assert vols[0] == pytest.approx(0.228815, abs=1e-6)
        assert vols[1] == pytest.approx(0.174341, abs=1e-6)
        assert vols[2] == pytest.approx(0.151918, abs=1e-6)

    def test_the_two_hundred_strike_call_the_sign_error_mispriced(self) -> None:
        # The true value, the lognormal value it should sit below, and the
        # value the reversed sign produced. The third is recomputed here by
        # reversing the sign deliberately, so the comparison is live.
        contract = Contract(100.0, 200.0, 1.0, 0.02, carry=0.0)
        true = lewis_price(REFERENCE, contract, OptionType.CALL)
        lognormal = bs_price(contract.with_vol(0.20), OptionType.CALL)
        assert true == pytest.approx(0.0012692, abs=1e-7)
        assert lognormal == pytest.approx(0.0018489, abs=1e-7)
        assert true < lognormal, "a negative correlation thins the upside tail"

        forward, strike = contract.forward, contract.strike
        k = math.log(strike / forward)

        def reversed_sign(u: float) -> float:
            value = complex(math.cos(u * k), math.sin(u * k)) * char_func(
                REFERENCE, u - 0.5j, contract.time
            )
            return value.real / (u * u + 0.25)

        integral = semi_infinite_quad(reversed_sign, tol=1e-12, initial_width=5.0)
        mirrored = contract.discount * (
            forward - math.sqrt(forward * strike) / math.pi * integral
        )
        assert mirrored == pytest.approx(0.2478, abs=5e-4)
        assert mirrored / lognormal == pytest.approx(134.0, abs=1.0)

    def test_lewis_is_the_faster_route(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Asserted as a count of transform evaluations rather than as a wall
        # clock, which is not reproducible on shared hardware. Gil-Pelaez runs
        # two integrals where Lewis runs one, and that is where the 1.45 in
        # the README comes from.
        import moneyness.heston as module

        calls = [0]
        real = module.char_func

        def counted(
            model: Heston, u: complex, time: float, branch: Branch = Branch.STABLE
        ) -> complex:
            calls[0] += 1
            return real(model, u, time, branch)

        monkeypatch.setattr(module, "char_func", counted)
        contract = Contract(100.0, 110.0, 1.0, 0.02, carry=0.0)
        calls[0] = 0
        lewis_price(REFERENCE, contract)
        one = calls[0]
        calls[0] = 0
        gil_pelaez_price(REFERENCE, contract)
        two = calls[0]
        assert two > one
        assert two / one == pytest.approx(1.45, abs=0.35)
