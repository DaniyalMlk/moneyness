"""Gauss-Legendre nodes, adaptive panels and the half line."""

from __future__ import annotations

import math

import pytest
from mpmath import mp

from moneyness.quadrature import (
    QuadratureError,
    adaptive_quad,
    fixed_quad,
    gauss_legendre,
    semi_infinite_quad,
)


class TestNodes:
    def test_two_point_rule_is_the_textbook_one(self) -> None:
        nodes, weights = gauss_legendre(2)
        root = 1.0 / math.sqrt(3.0)
        assert nodes == pytest.approx((-root, root), abs=1e-15)
        assert weights == pytest.approx((1.0, 1.0), abs=1e-15)

    def test_three_point_rule_is_the_textbook_one(self) -> None:
        nodes, weights = gauss_legendre(3)
        root = math.sqrt(0.6)
        assert nodes == pytest.approx((-root, 0.0, root), abs=1e-15)
        assert weights == pytest.approx((5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0), abs=1e-15)

    def test_one_point_rule_is_the_midpoint(self) -> None:
        nodes, weights = gauss_legendre(1)
        assert nodes == (0.0,)
        assert weights == pytest.approx((2.0,), abs=1e-15)

    @pytest.mark.parametrize("order", [1, 2, 3, 5, 8, 16, 20, 33, 64])
    def test_weights_sum_to_the_length_of_the_interval(self, order: int) -> None:
        _, weights = gauss_legendre(order)
        assert sum(weights) == pytest.approx(2.0, abs=1e-14)

    @pytest.mark.parametrize("order", [2, 3, 7, 16, 41])
    def test_nodes_are_increasing_and_inside_the_interval(self, order: int) -> None:
        nodes, _ = gauss_legendre(order)
        assert list(nodes) == sorted(nodes)
        assert all(-1.0 < node < 1.0 for node in nodes)

    @pytest.mark.parametrize("order", [2, 3, 7, 16, 41])
    def test_nodes_are_symmetric_about_zero(self, order: int) -> None:
        nodes, weights = gauss_legendre(order)
        for i in range(order):
            assert nodes[i] == pytest.approx(-nodes[order - 1 - i], abs=1e-15)
            assert weights[i] == pytest.approx(weights[order - 1 - i], abs=1e-15)

    @pytest.mark.parametrize("order", [1, 2, 3, 4, 5, 6, 7, 8, 12])
    def test_exact_on_polynomials_up_to_degree_two_n_minus_one(self, order: int) -> None:
        # The defining property of the rule. Degree 2n-1 is exact; 2n is not,
        # and the second half of this test is what makes the first meaningful.
        for degree in range(2 * order):
            got = fixed_quad(lambda x, d=degree: x**d, -1.0, 1.0, order)  # type: ignore[misc]
            want = 0.0 if degree % 2 else 2.0 / (degree + 1.0)
            assert got == pytest.approx(want, abs=1e-13)

    def test_not_exact_one_degree_beyond(self) -> None:
        order = 3
        degree = 2 * order
        got = fixed_quad(lambda x: x**degree, -1.0, 1.0, order)
        want = 2.0 / (degree + 1.0)
        assert abs(got - want) > 1e-4

    def test_the_cache_returns_the_same_object(self) -> None:
        assert gauss_legendre(19) is gauss_legendre(19)

    @pytest.mark.parametrize("order", [0, -1, -100])
    def test_a_non_positive_order_is_refused(self, order: int) -> None:
        with pytest.raises(QuadratureError, match="at least 1"):
            gauss_legendre(order)


class TestFixedPanel:
    def test_exponential_over_the_unit_interval(self) -> None:
        assert fixed_quad(math.exp, 0.0, 1.0, 20) == pytest.approx(math.e - 1.0, abs=1e-14)

    def test_an_empty_interval_is_zero(self) -> None:
        assert fixed_quad(math.exp, 2.0, 2.0, 20) == 0.0

    def test_reversing_the_limits_negates(self) -> None:
        forward = fixed_quad(math.sin, 0.0, 1.3, 20)
        backward = fixed_quad(math.sin, 1.3, 0.0, 20)
        assert forward == pytest.approx(-backward, abs=1e-15)


class TestAdaptive:
    def test_a_smooth_integrand(self) -> None:
        got = adaptive_quad(math.sin, 0.0, math.pi)
        assert got == pytest.approx(2.0, abs=1e-13)

    def test_a_narrow_peak_that_one_panel_would_miss(self) -> None:
        # A Lorentzian a thousandth of the interval wide. One twenty-point
        # panel across [-1, 1] puts no node inside it.
        def peak(x: float) -> float:
            return 1e-3 / (x * x + 1e-6)

        want = 2.0 * math.atan(1e3)
        single = fixed_quad(peak, -1.0, 1.0, 20)
        assert abs(single - want) > 0.5
        assert adaptive_quad(peak, -1.0, 1.0, tol=1e-11) == pytest.approx(want, abs=1e-9)

    def test_an_oscillatory_integrand(self) -> None:
        got = adaptive_quad(lambda x: math.sin(40.0 * x), 0.0, math.pi, tol=1e-13)
        want = (1.0 - math.cos(40.0 * math.pi)) / 40.0
        assert got == pytest.approx(want, abs=1e-12)

    def test_an_empty_interval_is_zero(self) -> None:
        assert adaptive_quad(math.exp, 1.0, 1.0) == 0.0

    @pytest.mark.parametrize("tol", [0.0, -1e-9])
    def test_a_non_positive_tolerance_is_refused(self, tol: float) -> None:
        with pytest.raises(QuadratureError, match="tol must be positive"):
            adaptive_quad(math.exp, 0.0, 1.0, tol=tol)

    @pytest.mark.parametrize("limit", [math.inf, -math.inf, math.nan])
    def test_a_non_finite_limit_is_refused(self, limit: float) -> None:
        with pytest.raises(QuadratureError, match="must be finite"):
            adaptive_quad(math.exp, 0.0, limit)


class TestHalfLine:
    def test_a_decaying_exponential(self) -> None:
        assert semi_infinite_quad(lambda x: math.exp(-x)) == pytest.approx(1.0, abs=1e-11)

    def test_a_gaussian(self) -> None:
        got = semi_infinite_quad(lambda x: math.exp(-x * x))
        assert got == pytest.approx(0.5 * math.sqrt(math.pi), abs=1e-11)

    def test_an_algebraic_tail(self) -> None:
        # Decays like 1/x^2, which is slow enough that the doubling has to run
        # a long way before a panel is negligible.
        got = semi_infinite_quad(lambda x: 1.0 / (1.0 + x * x), tol=1e-10)
        assert got == pytest.approx(0.5 * math.pi, abs=1e-8)

    def test_a_shifted_lower_limit(self) -> None:
        got = semi_infinite_quad(lambda x: math.exp(-x), a=2.0)
        assert got == pytest.approx(math.exp(-2.0), abs=1e-11)

    def test_an_integrand_that_crosses_zero_before_it_decays(self) -> None:
        # exp(-x) sin(x) integrates to 1/2 over the half line and changes sign
        # every pi, so a single quiet panel is not evidence of convergence.
        got = semi_infinite_quad(lambda x: math.exp(-x) * math.sin(x))
        assert got == pytest.approx(0.5, abs=1e-11)

    def test_an_integrand_that_does_not_decay_is_reported(self) -> None:
        with pytest.raises(QuadratureError, match="had not decayed"):
            semi_infinite_quad(lambda x: 1.0 / (1.0 + x))

    def test_a_non_positive_initial_width_is_refused(self) -> None:
        with pytest.raises(QuadratureError, match="initial_width must be positive"):
            semi_infinite_quad(math.exp, initial_width=0.0)


class TestAgainstFiftyDigits:
    """The same integrals evaluated by mpmath, which shares no code."""

    def test_a_trigonometric_integral(self) -> None:
        mp.dps = 50
        want = float(mp.quad(lambda x: mp.cos(x) * mp.exp(-x), [0, mp.inf]))
        got = semi_infinite_quad(lambda x: math.cos(x) * math.exp(-x))
        assert got == pytest.approx(want, abs=1e-11)

    def test_a_rational_integrand_on_a_finite_range(self) -> None:
        mp.dps = 50
        want = float(mp.quad(lambda x: 1 / (1 + x**4), [0, 3]))
        got = adaptive_quad(lambda x: 1.0 / (1.0 + x**4), 0.0, 3.0)
        assert got == pytest.approx(want, abs=1e-13)

    def test_a_logarithmic_integrand(self) -> None:
        mp.dps = 50
        want = float(mp.quad(lambda x: mp.log(x) ** 2, [1, 4]))
        got = adaptive_quad(lambda x: math.log(x) ** 2, 1.0, 4.0)
        assert got == pytest.approx(want, abs=1e-13)
