"""Tests for the normal density, distribution and quantile functions."""

from __future__ import annotations

import math

import pytest

from moneyness.normal import norm_cdf, norm_pdf, norm_ppf

from .reference import ref_norm_cdf

# A grid that spends most of its points where option pricing actually lands,
# and still reaches far enough out to catch a tail written on erf by mistake.
GRID = [
    -40.0, -20.0, -12.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5,
    -0.25, -0.1, -1e-8, 0.0, 1e-8, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0,
    4.0, 6.0, 8.0, 12.0, 20.0, 40.0,
]


def test_pdf_at_zero() -> None:
    assert norm_pdf(0.0) == pytest.approx(1.0 / math.sqrt(2.0 * math.pi), rel=0, abs=1e-17)


def test_pdf_is_even() -> None:
    for x in GRID:
        assert norm_pdf(x) == norm_pdf(-x)


def test_pdf_underflows_rather_than_raising() -> None:
    assert norm_pdf(1e3) == 0.0
    assert norm_pdf(-1e3) == 0.0


def test_pdf_integrates_against_its_own_derivative() -> None:
    # phi'(x) = -x phi(x). Checking the analytic derivative against a central
    # difference confirms the density and the constant in front of it together.
    h = 1e-5
    for x in (-2.0, -0.5, 0.3, 1.7):
        numeric = (norm_pdf(x + h) - norm_pdf(x - h)) / (2 * h)
        assert numeric == pytest.approx(-x * norm_pdf(x), rel=1e-8, abs=1e-12)


def test_cdf_matches_high_precision_reference() -> None:
    for x in GRID:
        expected = ref_norm_cdf(x)
        got = norm_cdf(x)
        # Relative accuracy is the demanding claim: it must hold where the
        # value itself is 1e-89, which an erf-based implementation cannot do.
        assert got == pytest.approx(expected, rel=1e-13, abs=1e-300)


def test_cdf_left_tail_is_not_lost_to_cancellation() -> None:
    # The specific failure this guards against: 0.5 * (1 + erf(x / sqrt(2)))
    # returns exactly 0.0 here, whereas the true value is representable.
    assert norm_cdf(-8.0) > 0.0
    assert norm_cdf(-8.0) == pytest.approx(6.2209605742717841e-16, rel=1e-13)
    assert norm_cdf(-20.0) == pytest.approx(2.7536241186062337e-89, rel=1e-13)


def test_cdf_is_a_distribution_function() -> None:
    previous = 0.0
    for x in GRID:
        value = norm_cdf(x)
        assert 0.0 <= value <= 1.0
        assert value >= previous
        previous = value
    assert norm_cdf(0.0) == 0.5


def test_cdf_reflection() -> None:
    for x in GRID:
        assert norm_cdf(x) + norm_cdf(-x) == pytest.approx(1.0, rel=0, abs=1e-15)


def test_ppf_known_quantiles() -> None:
    assert norm_ppf(0.975) == pytest.approx(1.9599639845400543, rel=1e-14)
    assert norm_ppf(0.5) == pytest.approx(0.0, abs=1e-15)
    assert norm_ppf(1e-10) == pytest.approx(-6.3613409024040562, rel=1e-12)


def test_ppf_inverts_cdf_through_the_lower_tail() -> None:
    # Round-tripping is the strongest single statement about the quantile
    # function: it exercises all three branches of the rational approximation
    # and the Halley refinement that follows them.
    #
    # It is asserted through the lower tail because that is where the round trip
    # is well conditioned. For x below zero the probability is small and carries
    # its full relative precision as a double, so nothing is lost on the way in
    # and the inverse is exact to the last bit or two.
    for x in GRID:
        if x > 0.0:
            continue
        p = norm_cdf(x)
        if p > 0.0:
            assert norm_ppf(p) == pytest.approx(x, rel=1e-14, abs=1e-14)


def test_ppf_upper_tail_is_limited_only_by_the_conditioning_of_p() -> None:
    # Above zero the round trip cannot be exact, and the reason is not the
    # implementation. norm_cdf(6) is 0.999999999013..., whose neighbouring
    # doubles are 1.1e-16 apart; that spacing corresponds to a step of
    # ulp(p) / phi(x) in x, which is 1.8e-8 at x = 6. The information is gone
    # when p is stored, before norm_ppf is called, so no algorithm recovers it.
    #
    # Asserting against that bound states the real guarantee, and would catch a
    # genuine regression in the upper branch, which a loose tolerance would not.
    for x in GRID:
        if x <= 0.0:
            continue
        p = norm_cdf(x)
        density = norm_pdf(x)
        if p >= 1.0 or density == 0.0:
            continue
        conditioning_limit = math.ulp(p) / density
        assert abs(norm_ppf(p) - x) <= max(conditioning_limit, 1e-14)


def test_ppf_reaches_the_far_tail_through_the_small_probability() -> None:
    # The practical consequence of the above: a caller wanting six or eight
    # standard deviations must ask for the small probability, not one minus it.
    for x in (3.0, 4.0, 6.0, 8.0, 12.0):
        assert norm_ppf(norm_cdf(-x)) == pytest.approx(-x, rel=1e-14, abs=1e-14)


def test_ppf_endpoints_are_infinite() -> None:
    assert norm_ppf(0.0) == -math.inf
    assert norm_ppf(1.0) == math.inf


@pytest.mark.parametrize("bad", [-1e-16, 1.0000000000000002, -1.0, 2.0])
def test_ppf_rejects_non_probabilities(bad: float) -> None:
    with pytest.raises(ValueError, match="probability"):
        norm_ppf(bad)


def test_ppf_rejects_nan() -> None:
    with pytest.raises(ValueError, match="NaN"):
        norm_ppf(math.nan)


def test_nan_propagates_without_raising() -> None:
    assert math.isnan(norm_cdf(math.nan))
    assert math.isnan(norm_pdf(math.nan))
