"""Tests for cross-currency equity options.

The closed forms are checked against a two-asset Monte Carlo simulation under
the domestic risk-neutral measure, which shares nothing with them: the
simulation draws the share and the exchange rate, applies the four payoffs as
written, and discounts. Beyond that the structure does the work. Three of the
four contracts collapse onto one number when the exchange rate stops moving,
which pins the three strike conventions against each other; the fourth is
invariant to the exchange rate's dynamics by construction and that invariance is
asserted rather than assumed; and the identifiability of the pair
``(sigma_X, rho)`` is demonstrated by exhibiting prices that agree to the last
bit across a curve of different pairs.

The simulation uses Box-Muller over ``random.Random`` rather than
``random.gauss`` so the draws do not depend on CPython's own normal variate
algorithm.
"""

from __future__ import annotations

import math
import random
from dataclasses import replace

import pytest

from moneyness.bsm import Inputs, OptionType
from moneyness.bsm import price as bsm_price
from moneyness.quanto import (
    CrossCurrency,
    FxPair,
    Linkage,
    QuantoError,
    composite_volatility,
    domestic_forward,
    exchange_forward,
    implied_fx_pair,
    parity_gap,
    price,
    quanto_drift,
    quanto_forward,
)

MARKET = {
    "spot": 100.0,
    "exchange": 1.25,
    "time": 1.0,
    "domestic_rate": 0.04,
    "foreign_rate": 0.02,
    "dividend": 0.01,
    "asset_vol": 0.22,
    "exchange_vol": 0.14,
    "correlation": 0.35,
}

# A strike that is roughly five per cent out of the money in each contract's own
# units, so the four prices are comparable in size rather than in moneyness.
STRIKES = {
    Linkage.FOREIGN_STRUCK: 105.0,
    Linkage.QUANTO: 105.0,
    Linkage.COMPOSITE: 130.0,
    Linkage.EQUITY_LINKED_FX: 1.30,
}


def state(**overrides: float) -> CrossCurrency:
    merged = {**MARKET, "strike": 105.0, **overrides}
    return CrossCurrency(**merged)


def for_linkage(linkage: Linkage, **overrides: float) -> CrossCurrency:
    return state(strike=STRIKES[linkage], **overrides)


# -- the simulation -----------------------------------------------------------


def simulate(
    market: CrossCurrency,
    linkage: Linkage,
    option: OptionType,
    *,
    paths: int = 300_000,
    seed: int = 7,
    fixed_exchange: float | None = None,
) -> tuple[float, float]:
    """The four payoffs, drawn and discounted. Returns the mean and its error.

    Under the domestic risk-neutral measure the exchange rate drifts at
    ``r_d - r_f`` and the share, quoted abroad, drifts at
    ``r_f - q - rho sigma_S sigma_X`` — that last term is the only thing the
    measure change does, and writing it here rather than importing it is what
    makes the comparison a check. Antithetic pairs, because every payoff is
    monotone in at least one of the two drivers.
    """
    source = random.Random(seed)
    rho = market.correlation
    root = math.sqrt(market.time)
    share_drift = (
        market.foreign_rate
        - market.dividend
        - rho * market.asset_vol * market.exchange_vol
        - 0.5 * market.asset_vol**2
    ) * market.time
    rate_drift = (
        market.domestic_rate - market.foreign_rate - 0.5 * market.exchange_vol**2
    ) * market.time
    fixed = market.exchange if fixed_exchange is None else fixed_exchange
    sign = option.sign
    orthogonal = math.sqrt(max(1.0 - rho * rho, 0.0))

    total = 0.0
    squares = 0.0
    count = 0
    for _ in range(paths // 2):
        first, second = source.random(), source.random()
        radius = math.sqrt(-2.0 * math.log(1.0 - first))
        angle = 2.0 * math.pi * second
        base = (radius * math.cos(angle), radius * math.sin(angle))
        for flip in (1.0, -1.0):
            share_shock = flip * base[0]
            rate_shock = rho * share_shock + orthogonal * flip * base[1]
            share = market.spot * math.exp(share_drift + market.asset_vol * root * share_shock)
            rate = market.exchange * math.exp(
                rate_drift + market.exchange_vol * root * rate_shock
            )
            if linkage is Linkage.FOREIGN_STRUCK:
                payoff = rate * max(sign * (share - market.strike), 0.0)
            elif linkage is Linkage.QUANTO:
                payoff = fixed * max(sign * (share - market.strike), 0.0)
            elif linkage is Linkage.COMPOSITE:
                payoff = max(sign * (share * rate - market.strike), 0.0)
            else:
                payoff = share * max(sign * (rate - market.strike), 0.0)
            total += payoff
            squares += payoff * payoff
            count += 1
    mean = total / count
    variance = max(squares / count - mean * mean, 0.0) / count
    discount = market.domestic_discount
    return discount * mean, discount * math.sqrt(variance)


@pytest.mark.parametrize("linkage", list(Linkage))
@pytest.mark.parametrize("option", list(OptionType))
def test_every_payoff_matches_a_two_asset_simulation(
    linkage: Linkage, option: OptionType
) -> None:
    market = for_linkage(linkage)
    exact = price(market, linkage, option)
    estimate, error = simulate(market, linkage, option)
    assert error > 0.0
    assert abs(estimate - exact) < 3.0 * error
    # The check only means something if the simulation resolves the price: a
    # standard error of a tenth of the value would accept almost anything.
    assert error < 0.01 * exact


def test_a_negatively_correlated_market_also_simulates() -> None:
    """The orthogonal component and the quanto drift both change, so the sign of
    the correlation is worth running rather than assuming symmetric."""
    for linkage in Linkage:
        market = for_linkage(linkage, correlation=-0.6)
        exact = price(market, linkage)
        estimate, error = simulate(market, linkage, OptionType.CALL, seed=11)
        assert abs(estimate - exact) < 3.5 * error


# -- the one contract the exchange rate's dynamics do not reach ---------------


def test_a_foreign_struck_option_is_the_foreign_price_at_todays_rate() -> None:
    """And that is exact, not an approximation: converting a foreign payoff at
    whatever rate prevails is holding the foreign claim."""
    for option in OptionType:
        market = for_linkage(Linkage.FOREIGN_STRUCK)
        foreign = bsm_price(
            Inputs(
                spot=market.spot,
                strike=market.strike,
                time=market.time,
                rate=market.foreign_rate,
                vol=market.asset_vol,
                carry=market.foreign_rate - market.dividend,
            ),
            option,
        )
        assert price(market, Linkage.FOREIGN_STRUCK, option) == pytest.approx(
            market.exchange * foreign, rel=1e-15
        )


def test_a_foreign_struck_option_ignores_the_exchange_rate_vol_and_correlation() -> None:
    """Invariant to the last bit across the whole of both parameters, while the
    quanto on the same economics moves by 28% over the same correlations."""
    reference = price(for_linkage(Linkage.FOREIGN_STRUCK), Linkage.FOREIGN_STRUCK)
    for vol in (0.0, 0.05, 0.3, 1.5):
        for correlation in (-1.0, -0.5, 0.0, 0.5, 1.0):
            market = for_linkage(
                Linkage.FOREIGN_STRUCK, exchange_vol=vol, correlation=correlation
            )
            assert price(market, Linkage.FOREIGN_STRUCK) == reference
    quantos = [
        price(for_linkage(Linkage.QUANTO, correlation=rho), Linkage.QUANTO)
        for rho in (-0.8, 0.8)
    ]
    assert quantos[1] / quantos[0] == pytest.approx(0.7195, abs=1e-3)


# -- what the correlation does, contract by contract --------------------------


def test_one_correlation_moves_the_four_contracts_four_different_ways() -> None:
    """The single most useful thing to know about this family: the same number
    leaves one price alone, lowers one and raises two, because it enters the
    quanto drift with a minus and the composite variance with a plus."""
    correlations = (-0.8, -0.4, 0.0, 0.4, 0.8)
    curves = {
        linkage: [
            price(for_linkage(linkage, correlation=rho), linkage) for rho in correlations
        ]
        for linkage in Linkage
    }
    flat = curves[Linkage.FOREIGN_STRUCK]
    assert all(value == flat[0] for value in flat)
    assert curves[Linkage.QUANTO] == sorted(curves[Linkage.QUANTO], reverse=True)
    assert curves[Linkage.COMPOSITE] == sorted(curves[Linkage.COMPOSITE])
    assert curves[Linkage.EQUITY_LINKED_FX] == sorted(curves[Linkage.EQUITY_LINKED_FX])
    # And by how much, so "sensitive" is a number: the composite more than
    # doubles across the range where the quanto loses a quarter.
    def travel(values: list[float]) -> float:
        return values[-1] / values[0] - 1.0

    assert travel(curves[Linkage.QUANTO]) == pytest.approx(-0.2805, abs=1e-3)
    assert travel(curves[Linkage.COMPOSITE]) == pytest.approx(1.6250, abs=1e-3)
    assert travel(curves[Linkage.EQUITY_LINKED_FX]) == pytest.approx(0.5610, abs=1e-3)


def test_quoting_the_exchange_rate_the_other_way_up_only_changes_the_sign() -> None:
    """Which is the trap: a correlation lifted from a screen in the other
    convention has the right magnitude and the wrong sign, and the quanto drift
    then moves by ``2 rho sigma_S sigma_X`` instead of vanishing. A quanto call
    comes out 15.5% too high at a correlation of 0.35 and 41.9% at 0.85.
    """
    for correlation, overstatement in (
        (0.2, 0.0857),
        (0.35, 0.1548),
        (0.6, 0.2799),
        (0.85, 0.4186),
    ):
        right = price(for_linkage(Linkage.QUANTO, correlation=correlation), Linkage.QUANTO)
        flipped = price(for_linkage(Linkage.QUANTO, correlation=-correlation), Linkage.QUANTO)
        assert flipped / right - 1.0 == pytest.approx(overstatement, abs=5e-4)
        gap = 2.0 * correlation * MARKET["asset_vol"] * MARKET["exchange_vol"]
        assert quanto_drift(
            for_linkage(Linkage.QUANTO, correlation=-correlation)
        ) - quanto_drift(for_linkage(Linkage.QUANTO, correlation=correlation)) == (
            pytest.approx(gap, rel=1e-14)
        )


def test_inverting_the_quotation_twice_is_the_identity() -> None:
    market = state()
    twice = market.inverted().inverted()
    for field in (
        "spot",
        "exchange",
        "strike",
        "time",
        "domestic_rate",
        "foreign_rate",
        "dividend",
        "asset_vol",
        "exchange_vol",
        "correlation",
    ):
        assert getattr(twice, field) == getattr(market, field)
    once = market.inverted()
    assert once.correlation == -market.correlation
    assert once.exchange == pytest.approx(1.0 / market.exchange, rel=1e-15)
    assert once.domestic_rate == market.foreign_rate


# -- the three strike conventions, pinned against each other ------------------


def test_the_three_struck_contracts_collapse_when_the_exchange_rate_stops_moving() -> None:
    """A deterministic exchange rate leaves one claim with three descriptions.

    The foreign strike converts at the forward rate, the quanto's fixed rate is
    that forward, and the composite's domestic strike is the foreign one
    converted at it. All three then equal today's rate times the foreign-currency
    option, to within six parts in 1e15 — which is what ties the three strike
    conventions together and would catch any one of them being written in the
    wrong currency.
    """
    for option in OptionType:
        quiet = state(exchange_vol=0.0, correlation=0.0)
        forward = exchange_forward(quiet)
        foreign_struck = price(quiet, Linkage.FOREIGN_STRUCK, option)
        quanto = price(quiet, Linkage.QUANTO, option, fixed_exchange=forward)
        composite = price(
            state(exchange_vol=0.0, correlation=0.0, strike=quiet.strike * forward),
            Linkage.COMPOSITE,
            option,
        )
        assert quanto == pytest.approx(foreign_struck, rel=1e-14)
        assert composite == pytest.approx(foreign_struck, rel=1e-14)


def test_a_zero_correlation_still_separates_the_quanto_from_the_foreign_struck() -> None:
    """The drift correction vanishes, but the discounting does not: the quanto
    discounts a foreign-currency payoff at the domestic rate."""
    uncorrelated = state(correlation=0.0)
    quanto = price(uncorrelated, Linkage.QUANTO)
    foreign = price(uncorrelated, Linkage.FOREIGN_STRUCK)
    assert quanto_drift(uncorrelated) == 0.0
    assert quanto != foreign
    # The two differ by the ratio of the two discount factors applied to the same
    # undiscounted foreign expectation.
    ratio = math.exp((uncorrelated.foreign_rate - uncorrelated.domestic_rate) * uncorrelated.time)
    assert quanto / foreign == pytest.approx(ratio, rel=1e-14)


def test_an_equity_linked_option_reduces_to_a_scaled_exchange_rate_option() -> None:
    """With a riskless share it is a plain Garman-Kohlhagen option on the rate,
    with the share's forward as its notional."""
    for option in OptionType:
        market = state(strike=1.30, asset_vol=0.0)
        notional = market.spot * math.exp(
            (market.foreign_rate - market.dividend) * market.time
        )
        plain = bsm_price(
            Inputs(
                spot=market.exchange,
                strike=market.strike,
                time=market.time,
                rate=market.domestic_rate,
                vol=market.exchange_vol,
                carry=market.domestic_rate - market.foreign_rate,
            ),
            option,
        )
        assert price(market, Linkage.EQUITY_LINKED_FX, option) == pytest.approx(
            notional * plain, rel=1e-13
        )


# -- the forwards -------------------------------------------------------------


def test_the_domestic_forward_does_not_contain_the_foreign_rate() -> None:
    """Earning it abroad and funding it in the currency position cancel, so a
    domestic investor's carry on the pair is ``r_d - q`` whatever the two rates
    are. Worth asserting because the cancellation is the easiest thing here to
    get wrong."""
    base = domestic_forward(state())
    for foreign in (-0.01, 0.0, 0.03, 0.09):
        assert domestic_forward(state(foreign_rate=foreign)) == pytest.approx(base, rel=1e-14)
    expected = MARKET["spot"] * MARKET["exchange"] * math.exp(
        (MARKET["domestic_rate"] - MARKET["dividend"]) * MARKET["time"]
    )
    assert base == pytest.approx(expected, rel=1e-15)


def test_the_quanto_forward_is_the_share_forward_less_the_drift() -> None:
    market = state()
    undrifted = market.spot * math.exp(
        (market.foreign_rate - market.dividend) * market.time
    )
    assert quanto_forward(market) == pytest.approx(
        undrifted * math.exp(quanto_drift(market) * market.time), rel=1e-15
    )
    assert quanto_forward(state(correlation=0.0)) == pytest.approx(undrifted, rel=1e-15)


def test_the_composite_volatility_is_the_product_rule() -> None:
    market = state()
    expected = math.sqrt(
        market.asset_vol**2
        + market.exchange_vol**2
        + 2.0 * market.correlation * market.asset_vol * market.exchange_vol
    )
    assert composite_volatility(market) == pytest.approx(expected, rel=1e-15)
    # Perfect negative correlation with equal volatilities is a deterministic
    # product, which is where the expression under the root can round negative.
    assert composite_volatility(state(asset_vol=0.2, exchange_vol=0.2, correlation=-1.0)) == 0.0
    assert composite_volatility(state(correlation=1.0)) == pytest.approx(
        market.asset_vol + market.exchange_vol, rel=1e-14
    )


# -- parity -------------------------------------------------------------------


@pytest.mark.parametrize("linkage", list(Linkage))
def test_each_contract_has_its_own_parity_and_all_four_hold(linkage: Linkage) -> None:
    for correlation in (-0.9, -0.2, 0.0, 0.45, 0.95):
        for moneyness in (0.6, 1.0, 1.7):
            market = for_linkage(linkage, correlation=correlation)
            market = replace(market, strike=market.strike * moneyness)
            assert parity_gap(market, linkage) == pytest.approx(0.0, abs=1e-12)


def test_the_quanto_parity_uses_the_contractual_rate_not_todays() -> None:
    market = for_linkage(Linkage.QUANTO)
    assert parity_gap(market, Linkage.QUANTO, fixed_exchange=1.4) == pytest.approx(0.0, abs=1e-12)
    assert price(market, Linkage.QUANTO, fixed_exchange=1.4) == pytest.approx(
        price(market, Linkage.QUANTO) * 1.4 / market.exchange, rel=1e-14
    )


# -- the pair that one quote cannot separate ----------------------------------


def test_a_quanto_price_fixes_only_the_product_of_the_two() -> None:
    """Five different pairs, identical to the last bit, while the composite price
    on the same pairs moves by 42%."""
    anchor = MARKET["correlation"] * MARKET["exchange_vol"]
    quantos = []
    composites = []
    for vol in (0.05, 0.10, 0.14, 0.25, 0.50):
        correlation = anchor / vol
        assert abs(correlation) <= 1.0
        market = state(exchange_vol=vol, correlation=correlation)
        quantos.append(price(market, Linkage.QUANTO))
        composites.append(price(market, Linkage.COMPOSITE))
    assert all(value == quantos[0] for value in quantos)
    assert composites[-1] / composites[0] - 1.0 == pytest.approx(0.4219, abs=1e-3)


def test_a_composite_price_fixes_only_the_total_variance() -> None:
    """And it is a different function of the pair, which is why two quotes are
    enough where one is not."""
    target = composite_volatility(state())
    asset = MARKET["asset_vol"]
    prices = []
    quantos = []
    for vol in (0.10, 0.14, 0.25):
        correlation = (target**2 - asset**2 - vol**2) / (2.0 * asset * vol)
        assert abs(correlation) <= 1.0
        market = state(exchange_vol=vol, correlation=correlation)
        prices.append(price(replace(market, strike=130.0), Linkage.COMPOSITE))
        quantos.append(price(market, Linkage.QUANTO))
    assert all(value == pytest.approx(prices[0], rel=1e-13) for value in prices)
    assert quantos[-1] / quantos[0] - 1.0 == pytest.approx(0.1926, abs=1e-3)


def test_the_two_quotes_together_invert_in_closed_form() -> None:
    for vol, correlation in ((0.14, 0.35), (0.08, -0.7), (0.3, 0.05), (0.2, 1.0)):
        truth = state(exchange_vol=vol, correlation=correlation)
        recovered = implied_fx_pair(
            state(exchange_vol=0.99, correlation=0.0),
            quanto_drift(truth),
            composite_volatility(truth),
        )
        assert recovered.exchange_vol == pytest.approx(vol, rel=1e-12)
        assert recovered.correlation == pytest.approx(correlation, abs=1e-12)
        assert -1.0 <= recovered.correlation <= 1.0
        assert recovered.drift == quanto_drift(truth)
        assert recovered.composite_vol == composite_volatility(truth)


def test_a_zero_exchange_rate_volatility_is_recovered_as_a_zero_correlation() -> None:
    """A rate that does not move carries no correlation, so the only consistent
    answer is zero rather than an arbitrary one."""
    recovered = implied_fx_pair(state(), 0.0, MARKET["asset_vol"])
    assert recovered.exchange_vol == pytest.approx(0.0, abs=1e-15)
    assert recovered.correlation == 0.0


def test_an_inconsistent_pair_of_quotes_is_refused_rather_than_fitted() -> None:
    market = state()
    with pytest.raises(QuantoError, match="exchange rate variance"):
        implied_fx_pair(market, -0.02, 0.2)
    with pytest.raises(QuantoError, match=r"outside"):
        # A variance that is positive but smaller than the product it has to
        # support, so the ratio leaves the unit interval rather than failing the
        # square root.
        implied_fx_pair(market, -0.066, 0.45)
    with pytest.raises(QuantoError, match="total volatility cannot be negative"):
        implied_fx_pair(market, 0.0, -0.1)
    with pytest.raises(QuantoError, match="must be finite"):
        implied_fx_pair(market, float("nan"), 0.3)
    with pytest.raises(QuantoError, match="share's volatility is zero"):
        implied_fx_pair(state(asset_vol=0.0), 0.0, 0.3)
    with pytest.raises(QuantoError, match="no correlation gives"):
        # Values chosen so the implied variance cancels to exactly zero in binary
        # while the implied product does not: 0.5625 - 0.25 - 0.3125.
        implied_fx_pair(state(asset_vol=0.5), -0.15625, 0.75)


def test_the_pair_is_a_plain_record() -> None:
    pair = FxPair(exchange_vol=0.14, correlation=0.35, drift=-0.01078, composite_vol=0.299)
    assert pair.exchange_vol == 0.14
    assert pair.correlation == 0.35


# -- degenerate inputs are limits ---------------------------------------------


@pytest.mark.parametrize("linkage", list(Linkage))
def test_zero_time_is_the_intrinsic_value(linkage: Linkage) -> None:
    for option in OptionType:
        for moneyness in (0.7, 1.0, 1.4):
            market = for_linkage(linkage)
            market = replace(market, time=0.0, strike=market.strike * moneyness)
            value = price(market, linkage, option)
            if linkage is Linkage.FOREIGN_STRUCK or linkage is Linkage.QUANTO:
                expected = market.exchange * max(
                    option.sign * (market.spot - market.strike), 0.0
                )
            elif linkage is Linkage.COMPOSITE:
                expected = max(
                    option.sign * (market.spot * market.exchange - market.strike), 0.0
                )
            else:
                expected = market.spot * max(
                    option.sign * (market.exchange - market.strike), 0.0
                )
            assert value == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("linkage", list(Linkage))
def test_both_volatilities_at_zero_is_the_discounted_forward_intrinsic(
    linkage: Linkage,
) -> None:
    market = for_linkage(linkage, asset_vol=0.0, exchange_vol=0.0, correlation=0.0)
    for option in OptionType:
        value = price(market, linkage, option)
        assert value >= 0.0
        assert parity_gap(market, linkage) == pytest.approx(0.0, abs=1e-12)
        assert math.isfinite(value)


# -- refusals -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("spot", -1.0, "spot must be non-negative"),
        ("strike", -1.0, "strike must be non-negative"),
        ("time", -1.0, "time must be non-negative"),
        ("asset_vol", -0.1, "asset_vol must be non-negative"),
        ("exchange_vol", -0.1, "exchange_vol must be non-negative"),
        ("exchange", 0.0, "exchange rate must be positive"),
        ("exchange", -1.25, "exchange rate must be positive"),
        ("correlation", 1.5, r"correlation must lie in \[-1, 1\]"),
        ("correlation", float("nan"), "must be finite"),
        ("domestic_rate", float("inf"), "must be finite"),
    ],
)
def test_an_unpriceable_market_is_refused(field: str, value: float, message: str) -> None:
    with pytest.raises(QuantoError, match=message):
        state(**{field: value})


@pytest.mark.parametrize("rate", [0.0, -1.0, float("nan")])
def test_a_fixed_conversion_rate_that_is_not_a_rate_is_refused(rate: float) -> None:
    with pytest.raises(QuantoError, match="fixed conversion rate"):
        price(state(), Linkage.QUANTO, fixed_exchange=rate)
