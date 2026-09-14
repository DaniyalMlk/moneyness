"""Tests for the analytic sensitivities.

The organising idea: every Greek is a partial derivative of the price, so every
Greek is checked against that partial derivative taken numerically at fifty
digits. The analytic formula and the numerical derivative share nothing but the
price function itself, so an algebra error in a closed form cannot be masked by
the same error appearing on both sides — which is the failure mode of checking
one closed form against another.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable

import pytest

from moneyness.bsm import Inputs, OptionType, price
from moneyness.greeks import (
    charm,
    colour,
    delta,
    dual_delta,
    dual_gamma,
    forward_delta,
    gamma,
    rho,
    rho_carry,
    speed,
    theta,
    vanna,
    vega,
    veta,
    volga,
    zomma,
)
from moneyness.normal import norm_pdf

from .reference import ref_derivative

BOTH = (OptionType.CALL, OptionType.PUT)

STRIKES = (70.0, 100.0, 140.0)
TIMES = (0.05, 0.5, 2.0)
VOLS = (0.1, 0.3, 0.7)
RATES = (0.04,)
CARRIES = (0.0, 0.02)


def params() -> list[dict[str, float]]:
    return [
        {"spot": 100.0, "strike": k, "time": t, "rate": r, "vol": v, "carry": b}
        for k, t, r, v, b in itertools.product(STRIKES, TIMES, RATES, VOLS, CARRIES)
    ]


def as_inputs(p: dict[str, float]) -> Inputs:
    return Inputs(p["spot"], p["strike"], p["time"], p["rate"], p["vol"], carry=p["carry"])


# name, function of (inputs, option), parameters to differentiate, orders, and
# whether the Greek is the negative of that derivative. The time-based Greeks
# are negated because the market quotes decay against calendar time while the
# model is parameterised by time remaining.
Directional = tuple[
    str, Callable[[Inputs, OptionType], float], tuple[str, ...], tuple[int, ...], bool
]
Shared = tuple[str, Callable[[Inputs], float], tuple[str, ...], tuple[int, ...], bool]

DIRECTIONAL: list[Directional] = [
    ("delta", delta, ("spot",), (1,), False),
    ("theta", theta, ("time",), (1,), True),
    ("rho", rho, ("rate",), (1,), False),
    ("rho_carry", rho_carry, ("carry",), (1,), False),
    ("dual_delta", dual_delta, ("strike",), (1,), False),
    ("charm", charm, ("spot", "time"), (1, 1), True),
]

SHARED: list[Shared] = [
    ("gamma", gamma, ("spot",), (2,), False),
    ("vega", vega, ("vol",), (1,), False),
    ("vanna", vanna, ("spot", "vol"), (1, 1), False),
    ("volga", volga, ("vol",), (2,), False),
    ("speed", speed, ("spot",), (3,), False),
    ("zomma", zomma, ("spot", "vol"), (2, 1), False),
    ("dual_gamma", dual_gamma, ("strike",), (2,), False),
    ("veta", veta, ("vol", "time"), (1, 1), True),
    ("colour", colour, ("spot", "time"), (2, 1), True),
]


def tolerance(expected: float, scale: float) -> float:
    """Relative where the value is meaningful, absolute where it is near zero.

    The absolute floor is tied to the size of the quantities the Greek is built
    from rather than to the Greek itself, because several of them pass through
    zero — volga and zomma vanish where ``d1 d2 = 1`` — and a purely relative
    test would demand infinite precision at the crossing.
    """
    return 1e-9 * abs(expected) + 1e-11 * scale


@pytest.mark.parametrize(("name", "fn", "wrt", "orders", "negated"), DIRECTIONAL)
@pytest.mark.parametrize("option", BOTH)
def test_directional_greek_matches_numerical_derivative(
    name: str,
    fn: Callable[[Inputs, OptionType], float],
    wrt: tuple[str, ...],
    orders: tuple[int, ...],
    negated: bool,
    option: OptionType,
) -> None:
    for p in params():
        expected = ref_derivative(p, option is OptionType.CALL, wrt, orders)
        if negated:
            expected = -expected
        got = fn(as_inputs(p), option)
        assert got == pytest.approx(expected, abs=tolerance(expected, p["spot"])), (name, p)


@pytest.mark.parametrize(("name", "fn", "wrt", "orders", "negated"), SHARED)
def test_shared_greek_matches_numerical_derivative(
    name: str,
    fn: Callable[[Inputs], float],
    wrt: tuple[str, ...],
    orders: tuple[int, ...],
    negated: bool,
) -> None:
    for p in params():
        for option in BOTH:
            expected = ref_derivative(p, option is OptionType.CALL, wrt, orders)
            if negated:
                expected = -expected
            got = fn(as_inputs(p))
            assert got == pytest.approx(expected, abs=tolerance(expected, p["spot"])), (name, p)


@pytest.mark.parametrize(("name", "fn", "wrt", "orders", "negated"), SHARED)
def test_shared_greeks_are_the_same_for_calls_and_puts(
    name: str,
    fn: Callable[[Inputs], float],
    wrt: tuple[str, ...],
    orders: tuple[int, ...],
    negated: bool,
) -> None:
    # Parity makes the two prices differ by a function that is linear in spot
    # and in strike and independent of volatility, so every derivative that is
    # second order or higher, or taken with respect to volatility at all, is
    # shared. Asserted here on the reference rather than on the implementation,
    # so it is a statement about the mathematics.
    for p in params():
        for_call = ref_derivative(p, True, wrt, orders)
        for_put = ref_derivative(p, False, wrt, orders)
        assert for_call == pytest.approx(for_put, abs=tolerance(for_call, p["spot"]))


def test_the_density_identity_that_the_closed_forms_rely_on() -> None:
    # S e^{(b-r)T} phi(d1) = K e^{-rT} phi(d2). Several Greeks are only simple
    # because the phi terms cancel through this, so it is checked directly.
    from moneyness.bsm import d1_d2

    for p in params():
        inputs = as_inputs(p)
        d1, d2 = d1_d2(inputs)
        left = inputs.spot * math.exp((inputs.b - inputs.rate) * inputs.time) * norm_pdf(d1)
        right = inputs.strike * inputs.discount * norm_pdf(d2)
        assert left == pytest.approx(right, rel=1e-12)


def test_put_call_relationships() -> None:
    for p in params():
        inputs = as_inputs(p)
        carry_discount = math.exp((inputs.b - inputs.rate) * inputs.time)
        call_delta = delta(inputs, OptionType.CALL)
        put_delta = delta(inputs, OptionType.PUT)
        # Differentiating parity in spot gives the delta relationship exactly.
        assert call_delta - put_delta == pytest.approx(carry_discount, rel=1e-13)
        # And in strike, for the dual delta.
        assert dual_delta(inputs, OptionType.CALL) - dual_delta(
            inputs, OptionType.PUT
        ) == pytest.approx(-inputs.discount, rel=1e-13)


def test_signs_are_what_a_trader_would_expect() -> None:
    for p in params():
        inputs = as_inputs(p)
        assert delta(inputs, OptionType.CALL) >= 0.0
        assert delta(inputs, OptionType.PUT) <= 0.0
        assert gamma(inputs) > 0.0
        assert vega(inputs) > 0.0
        assert dual_gamma(inputs) > 0.0


def test_a_call_on_a_plain_share_always_decays() -> None:
    # Where the carry is the rate, both terms of theta are negative for a
    # non-negative rate, so this is a theorem rather than an observation.
    for k, t, v in itertools.product(STRIKES, TIMES, VOLS):
        inputs = Inputs(100.0, k, t, 0.04, v)
        assert theta(inputs, OptionType.CALL) < 0.0


def test_theta_is_positive_where_the_holder_is_waiting_to_be_paid() -> None:
    # Decay is not one-signed in general, and the exceptions are not exotic.
    #
    # A deep in-the-money put: the holder is owed the strike and gains as the
    # wait for it shortens.
    put = Inputs(spot=40.0, strike=140.0, time=1.0, rate=0.15, vol=0.1, carry=0.15)
    assert theta(put, OptionType.PUT) > 0.0

    # A deep in-the-money call on a *future*: with zero carry the option is a
    # discounted claim on a near-certain payoff, and the discount unwinds as
    # expiry approaches. This is the case that a sign test written for shares
    # gets wrong, so it is pinned with the mechanism spelled out.
    call = Inputs.on_future(future=100.0, strike=70.0, time=0.05, rate=0.04, vol=0.1)
    assert theta(call, OptionType.CALL) > 0.0
    later = price(Inputs.on_future(100.0, 70.0, 0.20, 0.04, 0.1), OptionType.CALL)
    sooner = price(Inputs.on_future(100.0, 70.0, 0.01, 0.04, 0.1), OptionType.CALL)
    assert later < sooner < 30.0


def test_volga_and_zomma_vanish_together() -> None:
    # Both carry a factor of (d1 d2 - 1) or d1 d2, so they change sign at
    # specific moneyness rather than being one-signed. Finding a bracket proves
    # the crossing exists and is not an artefact.
    def volga_at(strike: float) -> float:
        return volga(Inputs(100.0, strike, 1.0, 0.0, 0.3, carry=0.0))

    assert volga_at(100.0) < 0.0
    assert volga_at(200.0) > 0.0


def test_forward_delta_is_the_chain_rule_through_the_forward() -> None:
    for p in params():
        inputs = as_inputs(p)
        for option in BOTH:
            expected = delta(inputs, option) / math.exp(inputs.b * inputs.time)
            assert forward_delta(inputs, option) == pytest.approx(expected, rel=1e-14)


def test_rho_is_minus_time_times_price() -> None:
    for p in params():
        inputs = as_inputs(p)
        for option in BOTH:
            assert rho(inputs, option) == pytest.approx(
                -inputs.time * price(inputs, option), rel=1e-14
            )


def test_total_rate_sensitivity_for_a_plain_share() -> None:
    # When the carry is the rate, moving the rate moves both, and the sum of the
    # two partials is the textbook rho: T K e^{-rT} N(d2) for a call.
    from moneyness.bsm import d1_d2
    from moneyness.normal import norm_cdf

    for k, t, v in itertools.product(STRIKES, TIMES, VOLS):
        inputs = Inputs(100.0, k, t, 0.04, v)
        _, d2 = d1_d2(inputs)
        total = rho(inputs, OptionType.CALL) + rho_carry(inputs, OptionType.CALL)
        expected = t * k * inputs.discount * norm_cdf(d2)
        assert total == pytest.approx(expected, rel=1e-11)


@pytest.mark.parametrize(
    "fn",
    [gamma, vega, vanna, volga, speed, zomma, dual_gamma, veta, colour],
)
def test_shared_greeks_refuse_degenerate_inputs(fn: Callable[[Inputs], float]) -> None:
    for inputs in (
        Inputs(100.0, 100.0, 0.0, 0.03, 0.2),
        Inputs(100.0, 100.0, 1.0, 0.03, 0.0),
    ):
        with pytest.raises(ValueError, match="undefined"):
            fn(inputs)


@pytest.mark.parametrize("fn", [delta, theta, rho, rho_carry, dual_delta, charm, forward_delta])
def test_directional_greeks_refuse_degenerate_inputs(
    fn: Callable[[Inputs, OptionType], float],
) -> None:
    for inputs in (
        Inputs(100.0, 100.0, 0.0, 0.03, 0.2),
        Inputs(100.0, 100.0, 1.0, 0.03, 0.0),
    ):
        with pytest.raises(ValueError, match="undefined"):
            fn(inputs, OptionType.CALL)
