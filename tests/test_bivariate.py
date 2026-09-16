"""The bivariate normal distribution function.

Checked three ways. Against identities that hold exactly and can be written
down without computing anything — the marginals, independence, the degenerate
correlations, the reflection symmetries. Against the one closed form there is,
at the origin. And against ``mpmath`` evaluating the same integral at fifty
digits over a grid that deliberately includes the region this function exists
to serve: correlations within one part in ten thousand of minus one.
"""

from __future__ import annotations

import itertools
import math

import pytest

from moneyness.bivariate import _gauss_legendre, norm_cdf2
from moneyness.normal import norm_cdf

from .reference import ref_norm_cdf2

ARGUMENTS = [-8.0, -3.0, -1.5, -0.5, 0.0, 0.5, 1.5, 3.0, 8.0]
CORRELATIONS = [-0.9999, -0.99, -0.9, -0.5, -0.1, 0.1, 0.5, 0.9, 0.99, 0.9999]


class TestQuadratureRule:
    """The nodes are computed, so the computation has to be checked.

    A Gauss-Legendre rule of order n integrates every polynomial up to degree
    2n - 1 exactly, and that single property pins the nodes and weights
    completely. Checking it is strictly stronger than comparing against a
    transcribed table, and it cannot be satisfied by a rule with a typo in it.
    """

    @pytest.mark.parametrize("order", [2, 5, 12, 20])
    def test_weights_sum_to_the_length_of_the_interval(self, order: int) -> None:
        _, weights = _gauss_legendre(order)
        assert math.fsum(weights) == pytest.approx(2.0, abs=1e-14)

    @pytest.mark.parametrize("order", [2, 5, 12, 20])
    def test_nodes_are_symmetric_and_inside_the_interval(self, order: int) -> None:
        nodes, _ = _gauss_legendre(order)
        ordered = sorted(nodes)
        assert all(-1.0 < x < 1.0 for x in ordered)
        for low, high in zip(ordered, reversed(ordered), strict=True):
            assert low == pytest.approx(-high, abs=1e-14)

    @pytest.mark.parametrize("order", [2, 5, 12, 20])
    def test_integrates_polynomials_exactly_to_the_expected_degree(self, order: int) -> None:
        """Exact to degree 2n - 1, and only to there.

        The second half matters as much as the first: a rule that was somehow
        exact beyond its degree would not be a Gauss rule, and the failure at
        degree 2n is what shows the nodes are where they should be rather than
        merely in a plausible place.
        """
        nodes, weights = _gauss_legendre(order)
        for degree in range(2 * order):
            quadrature = math.fsum(w * x**degree for x, w in zip(nodes, weights, strict=True))
            exact = 0.0 if degree % 2 else 2.0 / (degree + 1)
            assert quadrature == pytest.approx(exact, abs=1e-12)

        degree = 2 * order
        quadrature = math.fsum(w * x**degree for x, w in zip(nodes, weights, strict=True))
        assert quadrature != pytest.approx(2.0 / (degree + 1), abs=1e-12)


class TestIdentities:
    @pytest.mark.parametrize("rho", CORRELATIONS)
    def test_the_value_at_the_origin_has_a_closed_form(self, rho: float) -> None:
        """``Phi2(0, 0; rho) = 1/4 + arcsin(rho) / (2 pi)``.

        A direct consequence of the construction: at the origin the integrand
        is identically one, so the integral is just the length of the range.
        """
        expected = 0.25 + math.asin(rho) / (2.0 * math.pi)
        assert norm_cdf2(0.0, 0.0, rho) == pytest.approx(expected, abs=1e-15)

    @pytest.mark.parametrize("a", ARGUMENTS)
    @pytest.mark.parametrize("b", ARGUMENTS)
    def test_independence_gives_the_product_of_the_marginals(self, a: float, b: float) -> None:
        assert norm_cdf2(a, b, 0.0) == pytest.approx(norm_cdf(a) * norm_cdf(b), rel=1e-15)

    @pytest.mark.parametrize("a", ARGUMENTS)
    @pytest.mark.parametrize("rho", CORRELATIONS)
    def test_an_infinite_limit_collapses_onto_a_marginal(self, a: float, rho: float) -> None:
        assert norm_cdf2(a, math.inf, rho) == pytest.approx(norm_cdf(a), rel=1e-15)
        assert norm_cdf2(math.inf, a, rho) == pytest.approx(norm_cdf(a), rel=1e-15)
        assert norm_cdf2(a, -math.inf, rho) == 0.0
        assert norm_cdf2(-math.inf, a, rho) == 0.0

    @pytest.mark.parametrize("a", ARGUMENTS)
    @pytest.mark.parametrize("b", ARGUMENTS)
    def test_perfect_correlation_is_the_smaller_marginal(self, a: float, b: float) -> None:
        assert norm_cdf2(a, b, 1.0) == pytest.approx(norm_cdf(min(a, b)), rel=1e-15)

    @pytest.mark.parametrize("a", ARGUMENTS)
    @pytest.mark.parametrize("b", ARGUMENTS)
    def test_perfect_anticorrelation_is_an_interval(self, a: float, b: float) -> None:
        """With ``Y = -X`` the event is ``-b <= X <= a``, and empty when ``a < -b``."""
        expected = max(0.0, norm_cdf(a) - norm_cdf(-b))
        assert norm_cdf2(a, b, -1.0) == pytest.approx(expected, abs=1e-15)

    @pytest.mark.parametrize("a", ARGUMENTS)
    @pytest.mark.parametrize("b", ARGUMENTS)
    @pytest.mark.parametrize("rho", CORRELATIONS)
    def test_is_symmetric_in_its_arguments(self, a: float, b: float, rho: float) -> None:
        assert norm_cdf2(a, b, rho) == pytest.approx(norm_cdf2(b, a, rho), abs=1e-15)

    @pytest.mark.parametrize("a", ARGUMENTS)
    @pytest.mark.parametrize("b", ARGUMENTS)
    @pytest.mark.parametrize("rho", CORRELATIONS)
    def test_reflecting_one_variable_flips_the_correlation(
        self, a: float, b: float, rho: float
    ) -> None:
        """``P(X <= a, Y <= b) + P(X <= a, -Y <= -b) = P(X <= a)``.

        Reflecting ``Y`` negates the correlation, and the two events partition
        the marginal. This ties together opposite signs of ``rho``, which the
        implementation reaches by different branches of the same code.
        """
        combined = norm_cdf2(a, b, rho) + norm_cdf2(a, -b, -rho)
        assert combined == pytest.approx(norm_cdf(a), abs=1e-14)

    @pytest.mark.parametrize("b", ARGUMENTS)
    @pytest.mark.parametrize("rho", CORRELATIONS)
    def test_is_monotone_in_each_argument(self, b: float, rho: float) -> None:
        values = [norm_cdf2(a, b, rho) for a in sorted(ARGUMENTS)]
        # Up to a rounding error rather than exactly: where the function
        # saturates, two neighbouring values are the same number and may differ
        # in the last bit, which is not a failure of monotonicity.
        for lower, higher in itertools.pairwise(values):
            assert higher >= lower - 1e-15

    @pytest.mark.parametrize("a", ARGUMENTS)
    @pytest.mark.parametrize("b", ARGUMENTS)
    def test_is_monotone_in_the_correlation(self, a: float, b: float) -> None:
        """Sheppard's identity says the derivative in ``rho`` is a density, so
        the function must increase with it."""
        values = [norm_cdf2(a, b, rho) for rho in sorted(CORRELATIONS)]
        for lower, higher in itertools.pairwise(values):
            assert higher >= lower - 1e-15


class TestAgainstHighPrecision:
    @pytest.mark.parametrize("rho", CORRELATIONS)
    def test_matches_fifty_digit_integration_across_the_grid(self, rho: float) -> None:
        """The check that actually establishes the accuracy.

        The oracle integrates the second variable out analytically and the first
        numerically, which shares no algebra with the implementation's route
        through Sheppard's identity.

        A first version of the oracle was handed the whole half-line and no
        subdivision points, and disagreed here by one part in a hundred at
        ``rho = -0.9999``. The implementation was right: near a unit
        correlation the oracle's integrand is a step of width
        ``sqrt(1 - rho^2)``, and the adaptive quadrature stepped over it. The
        lesson is in ``reference.py``, where the transition is now named.
        """
        worst = 0.0
        for a, b in itertools.product(ARGUMENTS, ARGUMENTS):
            worst = max(worst, abs(norm_cdf2(a, b, rho) - ref_norm_cdf2(a, b, rho)))
        assert worst < 1e-14, f"worst absolute error {worst:.3e} at rho={rho}"

    @pytest.mark.parametrize(
        ("a", "b", "rho"),
        [
            (-6.0, -6.0, 0.9),
            (-6.0, -6.0, -0.9),
            (-4.0, 4.0, 0.99),
            (5.0, -5.0, -0.99),
            (0.0, -7.0, 0.5),
        ],
    )
    def test_is_accurate_in_the_joint_tail(self, a: float, b: float, rho: float) -> None:
        """Where the probability is tiny, absolute agreement is not enough.

        A joint tail probability near ``1e-9`` is reproduced to sixteen digits
        by anything that gets the leading term right, so the assertion here is
        relative.
        """
        expected = ref_norm_cdf2(a, b, rho)
        assert norm_cdf2(a, b, rho) == pytest.approx(expected, rel=1e-9)


class TestDomain:
    @pytest.mark.parametrize("rho", [-1.5, 1.5, 1.0000001, -1.0000001])
    def test_rejects_a_correlation_outside_the_unit_interval(self, rho: float) -> None:
        with pytest.raises(ValueError, match=r"rho must lie in \[-1, 1\]"):
            norm_cdf2(0.0, 0.0, rho)

    @pytest.mark.parametrize(
        "args", [(math.nan, 0.0, 0.5), (0.0, math.nan, 0.5), (0.0, 0.0, math.nan)]
    )
    def test_rejects_nan(self, args: tuple[float, float, float]) -> None:
        with pytest.raises(ValueError, match="undefined at NaN"):
            norm_cdf2(*args)

    @pytest.mark.parametrize("a", ARGUMENTS)
    @pytest.mark.parametrize("b", ARGUMENTS)
    @pytest.mark.parametrize("rho", CORRELATIONS)
    def test_the_result_is_always_a_probability(
        self, a: float, b: float, rho: float
    ) -> None:
        """Quadrature can leave a rounding error outside the unit interval, which
        is meaningless for a probability and awkward for a caller that goes on
        to take its logarithm."""
        assert 0.0 <= norm_cdf2(a, b, rho) <= 1.0
