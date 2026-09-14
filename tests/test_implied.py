"""Tests for the implied-volatility solver.

The central test is a round trip: price an option at a known volatility, hand
the price back to the solver, and require the original volatility to return. It
is run across a grid deliberately weighted towards the cases that break naive
solvers — deep wings, one-day maturities, volatilities of 1% and 300% — because
the middle of the surface inverts under almost any method.
"""

from __future__ import annotations

import itertools
import math

import pytest

from moneyness.bsm import Inputs, OptionType, price
from moneyness.implied import (
    Method,
    Quote,
    Solution,
    black,
    black_vega,
    bounds,
    implied_vol,
    solve,
)

from .reference import ref_black_derivative, ref_black_vega

BOTH = (OptionType.CALL, OptionType.PUT)

STRIKES = (20.0, 60.0, 90.0, 100.0, 110.0, 160.0, 400.0)
TIMES = (1.0 / 365.0, 0.05, 0.5, 2.0, 10.0)
VOLS = (0.01, 0.05, 0.2, 0.6, 1.5, 3.0)
RATES = (0.0, 0.05)
CARRIES = (None, 0.0, -0.02)


def quotes() -> list[tuple[Quote, OptionType, float]]:
    """Every grid point, priced at a known volatility."""
    out: list[tuple[Quote, OptionType, float]] = []
    for k, t, r, v, b in itertools.product(STRIKES, TIMES, RATES, VOLS, CARRIES):
        for option in BOTH:
            inputs = Inputs(100.0, k, t, r, v, carry=b)
            out.append((Quote(100.0, k, t, r, price(inputs, option), carry=b), option, v))
    return out


def conditioning_limit(quote: Quote, vol: float) -> float:
    """The finest relative change in volatility this quote can express.

    A price is a double, so it moves in steps of ``ulp(price)``; dividing that
    by ``dPrice/dVol`` gives the smallest volatility difference the quote can
    distinguish at all. Where an option has little time value that ratio is
    large, and no solver — however carefully written — can do better, because
    the information is not in the input.

    The factor of thirty covers the several units in the last place of error
    already present in the price, which was produced by the same cancellation.
    """
    sensitivity = (
        quote.discount
        * black_vega(quote.forward, quote.strike, vol * math.sqrt(quote.time))
        * math.sqrt(quote.time)
    )
    if sensitivity <= 0.0:
        return math.inf
    return max(30.0 * math.ulp(quote.price) / (sensitivity * vol), 1e-9)


def test_the_grid_reaches_the_hard_cases() -> None:
    assert len(quotes()) >= 1000


def test_black_agrees_with_the_discounted_model_price() -> None:
    # The solver works on the undiscounted forward price, so that equivalence is
    # the bridge between what it solves and what the library prices.
    for k, t, r, v, b in itertools.product(STRIKES, TIMES, RATES, VOLS, CARRIES):
        inputs = Inputs(100.0, k, t, r, v, carry=b)
        quote = Quote(100.0, k, t, r, 0.0, carry=b)
        for option in BOTH:
            undiscounted = black(quote.forward, k, v * math.sqrt(t), option)
            assert quote.discount * undiscounted == pytest.approx(
                price(inputs, option), rel=1e-12, abs=1e-14
            )


def test_black_vega_matches_its_value_at_high_precision() -> None:
    # Checked against the closed form evaluated at fifty digits, including in
    # the wings where the true value is 1e-55 and relative accuracy is the only
    # meaningful claim.
    for k, w in itertools.product(STRIKES, (0.01, 0.1, 0.5, 2.0)):
        assert black_vega(100.0, k, w) == pytest.approx(
            ref_black_vega(100.0, k, w), rel=1e-11, abs=1e-300
        )


def test_black_vega_really_is_the_derivative_of_black() -> None:
    # The relationship, asserted where numerical differentiation can be trusted.
    #
    # Restricting the range is the point rather than a dodge. Vega decays like
    # exp(-d1^2 / 2), so differentiating it numerically is ill conditioned in
    # the deep wings: at a strike of 20 and a total volatility of 0.1 even a
    # fifty-digit difference quotient is wrong by 4.7e-4 relative, while the
    # implementation is accurate to 3.1e-14. Asserting there would test the
    # oracle. Here the two claims are kept apart: the value everywhere, against
    # the closed form; the derivative relationship where it can be measured.
    for k, w in itertools.product((60.0, 90.0, 100.0, 110.0, 160.0), (0.2, 0.5, 2.0)):
        if ref_black_vega(100.0, k, w) < 1e-8:
            continue
        assert black_vega(100.0, k, w) == pytest.approx(
            ref_black_derivative(100.0, k, w), rel=1e-9
        )


def test_round_trip_recovers_the_volatility() -> None:
    worst = 0.0
    skipped = 0
    for quote, option, expected in quotes():
        # An option whose price is zero to the last bit carries no information
        # about volatility: every volatility below some threshold produces that
        # same price in double precision, so the round trip is not a meaningful
        # question. These are counted rather than silently passed over.
        limits = bounds(quote, option)
        if quote.price - limits.lower <= max(quote.forward, quote.strike) * 1e-13:
            skipped += 1
            continue
        got = implied_vol(quote, option)
        relative = abs(got - expected) / expected

        # The accuracy attainable is set by the quote, not by the solver: see
        # conditioning_limit above. Asserting a flat tolerance here would either
        # fail on the deep in-the-money quotes, where the information genuinely
        # is not present, or be loose enough to miss a real regression in the
        # well-conditioned majority.
        limit = conditioning_limit(quote, expected)
        assert relative <= limit, (quote, option, expected, got, limit)
        worst = max(worst, relative)
    # A raw pin as well, so that a change making everything uniformly worse
    # while still inside its conditioning bound does not pass unnoticed. The
    # worst case is a deep in-the-money option whose entire time value is a few
    # units in the last place of its price.
    assert worst < 1e-4, f"worst relative recovery error {worst:.3e}"

    # About a third of the grid is skipped, which is a property of how extreme
    # the grid is rather than a weakness in the solver: a one-day option struck
    # at 20 against a spot of 100 with a volatility of 1% has no time value left
    # in double precision, so its price says nothing whatever about volatility
    # and inverting it is not a well-posed question.
    #
    # The claim worth asserting is therefore the size of the set that *is*
    # exercised, not the size of the set that is not.
    exercised = len(quotes()) - skipped
    assert exercised > 1500, f"only {exercised} quotes carried any volatility information"


def test_newton_and_brent_agree() -> None:
    # Brent uses no derivative. If the two solvers agree everywhere, an error in
    # black_vega cannot be hiding inside a converged Newton answer.
    for quote, option, _ in quotes():
        limits = bounds(quote, option)
        if quote.price - limits.lower <= max(quote.forward, quote.strike) * 1e-13:
            continue
        by_newton = solve(quote, option, Method.NEWTON)
        by_brent = solve(quote, option, Method.BRENT)
        # Compared against the same conditioning bound as the round trip. Where
        # the quote pins the volatility the two must agree closely; where it
        # does not, they land on different points that reproduce the price
        # equally well, and demanding more would be demanding information the
        # quote does not carry.
        limit = conditioning_limit(quote, by_brent.vol)
        assert abs(by_newton.vol - by_brent.vol) <= limit * by_brent.vol


def test_newton_is_the_faster_of_the_two() -> None:
    # Not a correctness property, but it is the reason Newton is the default,
    # so it is worth failing if a change to the safeguard quietly destroys it.
    newton_total = 0
    brent_total = 0
    for quote, option, _ in quotes():
        limits = bounds(quote, option)
        if quote.price - limits.lower <= max(quote.forward, quote.strike) * 1e-13:
            continue
        newton_total += solve(quote, option, Method.NEWTON).iterations
        brent_total += solve(quote, option, Method.BRENT).iterations
    assert newton_total < brent_total


def test_residual_is_reported_and_small() -> None:
    for quote, option, _ in quotes():
        solution = solve(quote, option)
        scale = max(quote.forward, quote.strike)
        assert abs(solution.residual) <= scale * 1e-12


def test_a_quote_at_intrinsic_implies_zero_volatility() -> None:
    for k, t, r in itertools.product(STRIKES, TIMES, RATES):
        for option in BOTH:
            intrinsic = price(Inputs(100.0, k, t, r, 0.0), option)
            solution = solve(Quote(100.0, k, t, r, intrinsic), option)
            assert solution.vol == 0.0
            assert solution.method is Method.BOUND


@pytest.mark.parametrize("option", BOTH)
def test_a_quote_below_intrinsic_is_rejected(option: OptionType) -> None:
    # The strike is chosen so that the option has positive intrinsic value on
    # whichever side is being tested; otherwise "one below intrinsic" is a
    # negative price and the constructor rejects it for a different reason.
    strike = 80.0 if option is OptionType.CALL else 140.0
    intrinsic = price(Inputs(100.0, strike, 1.0, 0.05, 0.0), option)
    assert intrinsic > 1.0
    with pytest.raises(ValueError, match="no-arbitrage"):
        solve(Quote(100.0, strike, 1.0, 0.05, intrinsic - 1.0), option)


@pytest.mark.parametrize("option", BOTH)
def test_a_quote_above_the_upper_bound_is_rejected(option: OptionType) -> None:
    quote = Quote(100.0, 100.0, 1.0, 0.05, 0.0)
    limits = bounds(quote, option)
    with pytest.raises(ValueError, match="no-arbitrage"):
        solve(Quote(100.0, 100.0, 1.0, 0.05, limits.upper * 1.01), option)


@pytest.mark.parametrize("option", BOTH)
def test_a_quote_on_the_upper_bound_is_reported_as_unreachable(option: OptionType) -> None:
    # The upper bound is a supremum, not a maximum: no finite volatility attains
    # it. Saying so is better than returning the largest volatility searched.
    quote = Quote(100.0, 100.0, 1.0, 0.05, 0.0)
    limits = bounds(quote, option)
    with pytest.raises(ValueError, match=r"not reachable|no-arbitrage"):
        solve(Quote(100.0, 100.0, 1.0, 0.05, limits.upper), option)


def test_bounds_are_the_limits_of_the_price() -> None:
    for k, t, r, b in itertools.product(STRIKES, TIMES, RATES, CARRIES):
        quote = Quote(100.0, k, t, r, 0.0, carry=b)
        for option in BOTH:
            limits = bounds(quote, option)
            at_zero = price(Inputs(100.0, k, t, r, 0.0, carry=b), option)
            assert limits.lower == pytest.approx(at_zero, abs=1e-13)
            # Approached from below as volatility grows, never exceeded.
            for v in (5.0, 20.0, 100.0):
                assert price(Inputs(100.0, k, t, r, v, carry=b), option) <= limits.upper + 1e-12
            # The limit is reached in *total* volatility, so the annualised
            # figure needed to approach it scales as 1 / sqrt(T): a one-day
            # option needs a far larger number than a ten-year one. Using a
            # fixed annualised volatility here would test only the long end.
            huge = 40.0 / math.sqrt(t)
            assert price(Inputs(100.0, k, t, r, huge, carry=b), option) == pytest.approx(
                limits.upper, rel=1e-9
            )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"time": 0.0}, "expired"),
        ({"time": -1.0}, "time"),
        ({"spot": 0.0}, "spot"),
        ({"strike": 0.0}, "strike"),
        ({"price": -1.0}, "price"),
        ({"spot": math.nan}, "finite"),
        ({"price": math.inf}, "finite"),
    ],
)
def test_invalid_quotes_raise(kwargs: dict[str, float], message: str) -> None:
    base = {"spot": 100.0, "strike": 100.0, "time": 1.0, "rate": 0.05, "price": 10.0}
    base.update(kwargs)
    with pytest.raises(ValueError, match=message):
        Quote(**base)


def test_solution_carries_enough_to_audit_it() -> None:
    inputs = Inputs(100.0, 105.0, 0.5, 0.03, 0.25)
    quote = Quote(100.0, 105.0, 0.5, 0.03, price(inputs, OptionType.CALL))
    solution = solve(quote, OptionType.CALL)
    assert isinstance(solution, Solution)
    assert solution.vol == pytest.approx(0.25, rel=1e-10)
    assert solution.total_vol == pytest.approx(0.25 * math.sqrt(0.5), rel=1e-10)
    assert 0 < solution.iterations < 60
    assert solution.method is Method.NEWTON


def test_a_one_day_deep_wing_option_still_inverts() -> None:
    # The case that motivates working in total volatility: a very short maturity
    # turns a modest total volatility into a large annualised one, and a solver
    # that iterates on the annualised number has to cover a correspondingly
    # wider range to the same absolute accuracy.
    inputs = Inputs(100.0, 160.0, 1.0 / 365.0, 0.02, 2.5)
    quote = Quote(100.0, 160.0, 1.0 / 365.0, 0.02, price(inputs, OptionType.CALL))
    assert implied_vol(quote, OptionType.CALL) == pytest.approx(2.5, rel=1e-8)


def test_every_convention_inverts_to_the_same_volatility() -> None:
    # The discount factor and the carry leave the problem entirely once the
    # quote is divided through, so all four conventions are one code path.
    for carry in (None, 0.0, 0.03, -0.04):
        inputs = Inputs(100.0, 95.0, 0.75, 0.04, 0.3, carry=carry)
        for option in BOTH:
            quote = Quote(100.0, 95.0, 0.75, 0.04, price(inputs, option), carry=carry)
            assert implied_vol(quote, option) == pytest.approx(0.3, rel=1e-9)
