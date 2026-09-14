"""Tests for the generalised Black-Scholes-Merton price."""

from __future__ import annotations

import itertools
import math
from itertools import pairwise

import pytest

from moneyness.bsm import (
    Inputs,
    OptionType,
    d1_d2,
    forward,
    intrinsic,
    log_moneyness,
    parity_gap,
    price,
)

from .reference import ref_bsm, ref_log_moneyness

BOTH = (OptionType.CALL, OptionType.PUT)

# Chosen to span the wings rather than to cluster at the money: a strike of 250
# against a spot of 100 at six months is far enough out that a tail error in the
# distribution function shows up as a relative error in the price.
SPOTS = (100.0,)
STRIKES = (25.0, 60.0, 90.0, 100.0, 110.0, 160.0, 250.0)
TIMES = (1.0 / 365.0, 0.08, 0.5, 1.0, 5.0)
RATES = (-0.005, 0.0, 0.045)
VOLS = (0.02, 0.12, 0.35, 0.9)
CARRIES = (None, 0.0, 0.02, -0.03)


def grid() -> list[Inputs]:
    return [
        Inputs(s, k, t, r, v, carry=b)
        for s, k, t, r, v, b in itertools.product(SPOTS, STRIKES, TIMES, RATES, VOLS, CARRIES)
    ]


def test_grid_is_large_enough_to_mean_something() -> None:
    assert len(grid()) >= 300


def test_price_matches_high_precision_reference() -> None:
    worst = 0.0
    for inputs in grid():
        for option in BOTH:
            expected = ref_bsm(
                inputs.spot,
                inputs.strike,
                inputs.time,
                inputs.rate,
                inputs.vol,
                inputs.b,
                option is OptionType.CALL,
            )
            got = price(inputs, option)
            # The claim is relative accuracy of 1e-11, with an absolute term
            # that only matters for prices so small they are zero in any
            # practical sense.
            #
            # 1e-11 is short of the 1e-15 the distribution function itself
            # achieves, and the gap is real rather than slack. d2 is computed in
            # double precision and carries a rounding error of order d2 * eps,
            # which the tail amplifies by phi(d2) / N(-d2): at a strike of 25
            # against a spot of 100 that factor is about two, turning 1e-15 in
            # d2 into 3e-15 in N(-d2). The subtraction forming the price costs
            # another digit and a half. The loss is in the conditioning of the
            # formula, not in the quality of its evaluation, so tightening the
            # tolerance would not find a bug — it would only fail.
            tolerance = 1e-11 * abs(expected) + 1e-22
            assert abs(got - expected) <= tolerance, (inputs, option, got, expected)
            if abs(expected) > inputs.spot * 1e-12:
                worst = max(worst, abs(got - expected) / abs(expected))
    # Pinned so that a regression in the wings shows up as a failure rather than
    # being absorbed by a tolerance chosen with room to spare.
    assert worst < 5e-12, f"worst relative error {worst:.3e}"


def test_put_call_parity_is_an_identity() -> None:
    for inputs in grid():
        assert abs(parity_gap(inputs)) <= 1e-12 * inputs.spot


def test_price_is_bounded_by_no_arbitrage() -> None:
    for inputs in grid():
        carry_discount = math.exp((inputs.b - inputs.rate) * inputs.time)
        call = price(inputs, OptionType.CALL)
        put = price(inputs, OptionType.PUT)
        # A call is worth at least its discounted intrinsic and never more than
        # the carried spot; a put never more than the discounted strike.
        assert call >= intrinsic(inputs, OptionType.CALL) - 1e-12
        assert call <= inputs.spot * carry_discount + 1e-12
        assert put >= intrinsic(inputs, OptionType.PUT) - 1e-12
        assert put <= inputs.strike * inputs.discount + 1e-12
        assert call >= 0.0
        assert put >= 0.0


def test_call_falls_and_put_rises_with_strike() -> None:
    for t, r, v in itertools.product(TIMES, RATES, VOLS):
        calls = [price(Inputs(100.0, k, t, r, v), OptionType.CALL) for k in STRIKES]
        puts = [price(Inputs(100.0, k, t, r, v), OptionType.PUT) for k in STRIKES]
        assert all(a >= b for a, b in pairwise(calls))
        assert all(a <= b for a, b in pairwise(puts))


def test_value_rises_with_volatility_and_with_time() -> None:
    for k in STRIKES:
        by_vol = [price(Inputs(100.0, k, 1.0, 0.03, v), OptionType.CALL) for v in VOLS]
        assert all(a <= b for a, b in pairwise(by_vol))
        # Longer-dated is worth more once carry is removed, which is the clean
        # statement; with carry in play the comparison is not monotone in general.
        by_time = [price(Inputs(100.0, k, t, 0.0, 0.3, carry=0.0), OptionType.CALL) for t in TIMES]
        assert all(a <= b + 1e-12 for a, b in pairwise(by_time))


@pytest.mark.parametrize("option", BOTH)
def test_zero_time_is_intrinsic(option: OptionType) -> None:
    for k in STRIKES:
        inputs = Inputs(100.0, k, 0.0, 0.04, 0.3)
        assert price(inputs, option) == max(option.sign * (100.0 - k), 0.0)


@pytest.mark.parametrize("option", BOTH)
def test_zero_volatility_is_discounted_intrinsic_on_the_forward(option: OptionType) -> None:
    for k in STRIKES:
        inputs = Inputs(100.0, k, 0.75, 0.04, 0.0, carry=0.01)
        expected = max(math.exp(-0.04 * 0.75) * option.sign * (forward(inputs) - k), 0.0)
        assert price(inputs, option) == pytest.approx(expected, rel=1e-15, abs=1e-15)


@pytest.mark.parametrize("option", BOTH)
def test_price_is_continuous_as_volatility_vanishes(option: OptionType) -> None:
    # The degenerate branch must be the limit of the general one, not merely a
    # plausible value: approaching zero volatility has to converge to it.
    for k in (60.0, 100.0, 160.0):
        limit = price(Inputs(100.0, k, 0.75, 0.04, 0.0, carry=0.01), option)
        for v in (1e-3, 1e-4, 1e-5, 1e-6):
            near = price(Inputs(100.0, k, 0.75, 0.04, v, carry=0.01), option)
            assert abs(near - limit) < 0.02


def test_zero_spot_and_zero_strike() -> None:
    worthless_call = Inputs(0.0, 100.0, 1.0, 0.05, 0.3)
    assert price(worthless_call, OptionType.CALL) == 0.0
    assert price(worthless_call, OptionType.PUT) == pytest.approx(100.0 * math.exp(-0.05))

    free_call = Inputs(100.0, 0.0, 1.0, 0.05, 0.3, carry=0.0)
    assert price(free_call, OptionType.CALL) == pytest.approx(100.0 * math.exp(-0.05))
    assert price(free_call, OptionType.PUT) == 0.0


def test_conventions_agree_with_their_explicit_carry() -> None:
    plain = Inputs(100.0, 95.0, 0.5, 0.04, 0.22)
    assert plain.b == 0.04
    assert price(plain, OptionType.CALL) == price(
        Inputs(100.0, 95.0, 0.5, 0.04, 0.22, carry=0.04), OptionType.CALL
    )

    merton = Inputs.with_dividend(100.0, 95.0, 0.5, 0.04, 0.22, dividend=0.015)
    assert merton.b == pytest.approx(0.025)

    black = Inputs.on_future(100.0, 95.0, 0.5, 0.04, 0.22)
    assert black.b == 0.0
    # Black's model discounts the forward intrinsic; the forward equals the
    # quoted future, which is the defining property of zero carry.
    assert forward(black) == pytest.approx(100.0)


def test_d1_d2_are_consistent_with_each_other() -> None:
    for inputs in grid():
        d1, d2 = d1_d2(inputs)
        assert d1 - d2 == pytest.approx(inputs.std_dev, rel=1e-14)


def test_d1_d2_refuses_degenerate_inputs() -> None:
    for inputs in (
        Inputs(100.0, 100.0, 0.0, 0.03, 0.2),
        Inputs(100.0, 100.0, 1.0, 0.03, 0.0),
        Inputs(0.0, 100.0, 1.0, 0.03, 0.2),
        Inputs(100.0, 0.0, 1.0, 0.03, 0.2),
    ):
        assert inputs.is_degenerate
        with pytest.raises(ValueError, match="undefined"):
            d1_d2(inputs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"spot": -1.0}, "spot"),
        ({"strike": -1.0}, "strike"),
        ({"time": -0.5}, "time"),
        ({"vol": -0.2}, "vol"),
        ({"spot": math.nan}, "finite"),
        ({"rate": math.inf}, "finite"),
        ({"vol": math.nan}, "vol"),
    ],
)
def test_invalid_inputs_raise(kwargs: dict[str, float], message: str) -> None:
    base = {"spot": 100.0, "strike": 100.0, "time": 1.0, "rate": 0.03, "vol": 0.2}
    base.update(kwargs)
    with pytest.raises(ValueError, match=message):
        Inputs(**base)


def test_inputs_are_immutable() -> None:
    # The ignore is the point of the test rather than a wart on it: mypy rejects
    # the assignment statically because the dataclass is frozen, and the runtime
    # raises for the same reason. Both halves of that guarantee are asserted.
    inputs = Inputs(100.0, 100.0, 1.0, 0.03, 0.2)
    with pytest.raises((AttributeError, TypeError)):
        inputs.spot = 105.0  # type: ignore[misc]


def test_log_moneyness_stays_accurate_near_the_money() -> None:
    # The naive log(S / K) degrades as the spot approaches the strike, because
    # the ratio rounds to a number near one and the rounding is the whole of the
    # answer. This asserts the accuracy that the log1p form actually delivers;
    # the naive form is wrong by 7e-5 relative at the tightest point below and
    # would fail this by nine orders of magnitude.
    for offset in (1e-1, 1e-3, 1e-6, 1e-9, 1e-12):
        for direction in (1.0, -1.0):
            strike = 100.0
            spot = strike * (1.0 + direction * offset)
            expected = ref_log_moneyness(spot, strike)
            assert log_moneyness(spot, strike) == pytest.approx(expected, rel=1e-14)


def test_log_moneyness_matches_the_direct_form_away_from_the_money() -> None:
    for spot, strike in ((100.0, 25.0), (100.0, 250.0), (100.0, 160.0), (5.0, 100.0)):
        assert log_moneyness(spot, strike) == pytest.approx(
            ref_log_moneyness(spot, strike), rel=1e-15
        )


def test_log_moneyness_is_exactly_zero_at_the_money() -> None:
    assert log_moneyness(100.0, 100.0) == 0.0


def test_short_dated_at_the_money_price_is_accurate() -> None:
    # Where the log1p form earns its place: d1 divides the log by v*sqrt(T), so
    # a one-day option magnifies any error in it by a factor of about fifty.
    for offset in (0.0, 1e-9, 1e-6, 1e-3):
        inputs = Inputs(100.0 + offset, 100.0, 1.0 / 365.0, 0.03, 0.2)
        for option in BOTH:
            expected = ref_bsm(
                inputs.spot, inputs.strike, inputs.time, inputs.rate, inputs.vol, inputs.b,
                option is OptionType.CALL,
            )
            assert price(inputs, option) == pytest.approx(expected, rel=1e-13)
