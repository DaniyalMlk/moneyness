"""Single-barrier Europeans.

The checks here fall into three kinds, and they are not equally strong.

**Parity is a transcription check.** The ``in`` and ``out`` forms are both
written out from the table, so ``in + out = vanilla`` holds algebraically for
all eight and asserting it catches a term copied with the wrong sign — the
likeliest error in a formula of this shape — and nothing else. It cannot catch
the table being the wrong table.

**The exact cases are structural.** A knock-out with the spot on the barrier is
worth nothing and the knock-in is worth the vanilla, bit for bit. An up-and-out
call struck at or above its barrier is worth exactly 0.0, because paying needs
``S_T > K >= H`` and that cannot happen without crossing ``H``. These have
known answers that owe nothing to this module's arithmetic.

**The solver is the independent check**, and it is the one that would catch a
wrong table. :mod:`moneyness.pde` marches a grid backwards with a Dirichlet
condition at the barrier; there is no reflection principle anywhere in it. Those
tests live in ``test_pde.py`` beside the rest of the solver's.

The remainder is the measured behaviour — the volatility non-monotonicity above
all, since it is the one property here that contradicts everything else in the
package — and the refusals.
"""

from __future__ import annotations

import itertools
import math

import pytest

from moneyness.barrier import (
    BarrierError,
    barrier_price,
    beyond_term,
    cannot_pay,
    is_structurally_worthless,
    is_touched,
    monitoring_shift,
    reflected_beyond_term,
    reflected_term,
    vanilla_term,
)
from moneyness.bsm import Inputs, OptionType, price
from moneyness.monte_carlo import Barrier, Settings
from moneyness.monte_carlo import barrier as simulate

SPOT = 100.0
TIME = 1.0
RATE = 0.05
CARRY = 0.02
VOL = 0.20

DOWN_LEVELS = (80.0, 85.0, 95.0, 99.0)
UP_LEVELS = (101.0, 105.0, 115.0, 120.0)
STRIKES = (85.0, 90.0, 100.0, 110.0, 125.0)
VOL_LADDER = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40)


def at(
    strike: float = 100.0, vol: float = VOL, spot: float = SPOT, time: float = TIME
) -> Inputs:
    return Inputs(spot=spot, strike=strike, time=time, rate=RATE, vol=vol, carry=CARRY)


def levels(style: Barrier) -> tuple[float, ...]:
    return DOWN_LEVELS if style.is_down else UP_LEVELS


def pairs() -> list[tuple[Barrier, Barrier]]:
    """The knock-out and knock-in of each side, for the parity identity."""
    return [
        (Barrier.DOWN_AND_OUT, Barrier.DOWN_AND_IN),
        (Barrier.UP_AND_OUT, Barrier.UP_AND_IN),
    ]


def contracts() -> list[tuple[Barrier, float, float, OptionType]]:
    """Every style, level and strike combination that is a live contract."""
    made = []
    for style, option in itertools.product(Barrier, OptionType):
        for level, strike in itertools.product(levels(style), STRIKES):
            made.append((style, level, strike, option))
    return made


# -- the vanilla term, against the package's own vanilla ----------------------


@pytest.mark.parametrize("strike", STRIKES)
@pytest.mark.parametrize("option", list(OptionType))
def test_the_vanilla_term_is_the_package_vanilla(
    strike: float, option: OptionType
) -> None:
    """Same formula, grouped differently, so they agree to the last bit or close.

    A cross-check on this file's algebra: the drift enters the four terms
    through ``mu`` rather than through ``d1`` directly, and a mistake in that
    rearrangement would show up here and nowhere else, because every other
    term shares it.
    """
    inputs = at(strike)
    assert vanilla_term(inputs, option) == pytest.approx(
        price(inputs, option), rel=1e-15
    )


# -- parity -------------------------------------------------------------------


@pytest.mark.parametrize(("style", "level", "strike", "option"), contracts())
def test_in_plus_out_is_the_vanilla(
    style: Barrier, level: float, strike: float, option: OptionType
) -> None:
    inputs = at(strike)
    out, knocked_in = (
        (style, Barrier.DOWN_AND_IN if style.is_down else Barrier.UP_AND_IN)
        if style.is_knock_out
        else (Barrier.DOWN_AND_OUT if style.is_down else Barrier.UP_AND_OUT, style)
    )
    total = barrier_price(inputs, option, level, out) + barrier_price(
        inputs, option, level, knocked_in
    )
    assert total == pytest.approx(price(inputs, option), rel=1e-13, abs=1e-14)


@pytest.mark.parametrize(("style", "level", "strike", "option"), contracts())
def test_every_half_is_non_negative_and_bounded_by_the_vanilla(
    style: Barrier, level: float, strike: float, option: OptionType
) -> None:
    """Each is a sub-portfolio of the vanilla, so neither can exceed it.

    Worth asserting separately from parity: two errors that cancel in the sum
    pass parity, and one of them going negative or above the vanilla does not
    pass this.
    """
    inputs = at(strike)
    value = barrier_price(inputs, option, level, style)
    assert -1e-14 <= value <= price(inputs, option) + 1e-14


# -- the two independent checks -----------------------------------------------


@pytest.mark.parametrize(
    ("style", "level", "option"),
    [
        (Barrier.UP_AND_OUT, 120.0, OptionType.CALL),
        (Barrier.DOWN_AND_OUT, 85.0, OptionType.CALL),
        (Barrier.UP_AND_IN, 120.0, OptionType.CALL),
        (Barrier.DOWN_AND_IN, 85.0, OptionType.CALL),
        (Barrier.UP_AND_OUT, 115.0, OptionType.PUT),
        (Barrier.DOWN_AND_OUT, 85.0, OptionType.PUT),
    ],
)
def test_the_simulation_agrees_with_the_closed_form(
    style: Barrier, level: float, option: OptionType
) -> None:
    """A formula and a simulation, with no derivation in common.

    :func:`moneyness.monte_carlo.barrier` with its Brownian bridge prices the
    continuously monitored contract from paths. The agreement is asserted as a
    z-score rather than as a tolerance, because the standard error is what
    says how close the two should be — a fixed tolerance is either loose enough
    to pass a real disagreement or tight enough to depend on the seed.

    The seed is fixed, so this is deterministic; it is a check on the formula,
    not a sampling experiment.
    """
    inputs = at()
    exact = barrier_price(inputs, option, level, style)
    estimate = simulate(
        inputs,
        option,
        level,
        style,
        steps=100,
        settings=Settings(paths=20_000, seed=7),
        bridge=True,
    )
    assert estimate.standard_error > 0.0
    assert abs(estimate.value - exact) < 4.0 * estimate.standard_error


# -- the structural cases -----------------------------------------------------


@pytest.mark.parametrize("option", list(OptionType))
@pytest.mark.parametrize("style", list(Barrier))
def test_the_spot_on_the_barrier_is_exact(style: Barrier, option: OptionType) -> None:
    """The out is dead and the in is the vanilla, bit for bit.

    Bit for bit because the vanilla comes from :func:`moneyness.bsm.price`
    rather than being recomputed out of the four terms, which is the reason it
    is taken from there.
    """
    inputs = at(strike=105.0 if option is OptionType.PUT else 95.0)
    assert is_touched(SPOT, 100.0, style)
    value = barrier_price(inputs, option, 100.0, style)
    if style.is_knock_out:
        assert value == 0.0
    else:
        assert value == price(inputs, option)


def test_a_computed_barrier_level_still_hits_the_structural_case() -> None:
    """A level reached by arithmetic must not miss the exact answer by a bit.

    ``0.8 * 125.0`` need not be ``100.0`` in binary. A caller who writes the
    barrier as a percentage of a reference and gets a near-zero price where
    zero is correct has been failed by a float comparison, so the comparison is
    made at a fixed number of decimals instead.
    """
    level = 0.8 * 125.0
    assert is_touched(100.0, level, Barrier.UP_AND_OUT)
    assert barrier_price(at(strike=95.0), OptionType.CALL, level, Barrier.UP_AND_OUT) == 0.0


@pytest.mark.parametrize("level", [100.0, 105.0, 120.0])
def test_an_up_and_out_call_struck_above_its_barrier_is_worthless(
    level: float,
) -> None:
    """Exactly 0.0: paying needs ``S_T > K >= H``, which crosses ``H``."""
    inputs = at(strike=level + 5.0, spot=level - 10.0)
    assert is_structurally_worthless(
        inputs.strike, level, Barrier.UP_AND_OUT, OptionType.CALL
    )
    assert barrier_price(inputs, OptionType.CALL, level, Barrier.UP_AND_OUT) == 0.0
    # The knock-in is then the whole vanilla, the same statement read back.
    assert barrier_price(
        inputs, OptionType.CALL, level, Barrier.UP_AND_IN
    ) == price(inputs, OptionType.CALL)


@pytest.mark.parametrize("level", [80.0, 95.0])
def test_a_down_and_out_put_struck_below_its_barrier_is_worthless(
    level: float,
) -> None:
    inputs = at(strike=level - 5.0, spot=level + 10.0)
    assert is_structurally_worthless(
        inputs.strike, level, Barrier.DOWN_AND_OUT, OptionType.PUT
    )
    assert barrier_price(inputs, OptionType.PUT, level, Barrier.DOWN_AND_OUT) == 0.0


def test_the_live_contracts_are_not_structurally_worthless() -> None:
    """The exemption is narrow, and claiming it widely would zero real contracts."""
    live = [
        (85.0, Barrier.DOWN_AND_OUT, OptionType.CALL),
        (120.0, Barrier.UP_AND_OUT, OptionType.CALL),
        (85.0, Barrier.DOWN_AND_OUT, OptionType.PUT),
        (120.0, Barrier.UP_AND_OUT, OptionType.PUT),
    ]
    for level, style, option in live:
        assert not is_structurally_worthless(100.0, level, style, option)
        assert barrier_price(at(100.0), option, level, style) > 0.0


# -- limits -------------------------------------------------------------------


def test_a_distant_barrier_recovers_the_vanilla() -> None:
    """And the convergence is fast, which is worth knowing before widening one.

    An up barrier at 150 still costs the call 24% of its value; at 200, 0.58%;
    at 300, 8.7e-07; and by 500 the knock-out is the vanilla to 3.7e-14.
    """
    inputs = at()
    vanilla = price(inputs, OptionType.CALL)
    gaps = [
        abs(
            barrier_price(inputs, OptionType.CALL, level, Barrier.UP_AND_OUT) / vanilla
            - 1.0
        )
        for level in (150.0, 200.0, 300.0, 500.0)
    ]
    assert gaps == sorted(gaps, reverse=True)
    assert gaps[0] == pytest.approx(0.2379, abs=5e-4)
    assert gaps[1] == pytest.approx(0.00584, abs=5e-5)
    assert gaps[-1] < 1e-12


@pytest.mark.parametrize("style", list(Barrier))
@pytest.mark.parametrize("option", list(OptionType))
def test_zero_volatility_follows_the_forward(
    style: Barrier, option: OptionType
) -> None:
    """With no diffusion the path is the forward, which is monotone in time.

    So it reaches the barrier if and only if its terminal value is past it, and
    no running maximum is needed. The forward here is 102.02, inside both
    levels, so nothing is reached.
    """
    inputs = at(strike=100.0, vol=0.0)
    forward = SPOT * math.exp(CARRY * TIME)
    assert 95.0 < forward < 115.0
    level = 95.0 if style.is_down else 115.0
    value = barrier_price(inputs, option, level, style)
    if style.is_knock_out:
        assert value == pytest.approx(price(inputs, option), rel=1e-15)
    else:
        assert value == 0.0


def test_zero_volatility_with_the_forward_past_the_barrier_knocks_out() -> None:
    """A large carry walks the forward through the level, deterministically."""
    inputs = Inputs(spot=100.0, strike=100.0, time=1.0, rate=0.05, vol=0.0, carry=0.40)
    assert 100.0 * math.exp(0.40) > 120.0
    assert barrier_price(inputs, OptionType.CALL, 120.0, Barrier.UP_AND_OUT) == 0.0
    assert barrier_price(
        inputs, OptionType.CALL, 120.0, Barrier.UP_AND_IN
    ) == pytest.approx(price(inputs, OptionType.CALL), rel=1e-15)


def test_zero_time_is_the_payoff() -> None:
    inputs = at(strike=95.0, time=0.0)
    assert barrier_price(
        inputs, OptionType.CALL, 120.0, Barrier.UP_AND_OUT
    ) == pytest.approx(5.0, rel=1e-15)


# -- the measured behaviour ---------------------------------------------------


def test_an_up_and_out_call_is_not_monotone_in_volatility() -> None:
    """The finding this module exists to record.

    Every option in :mod:`moneyness.bsm` is increasing in volatility. This one
    peaks at 7.51% and loses 94% of its value between there and 40%, because
    the volatility that pays for the optionality also pays for the knock-out.
    """
    values = [
        barrier_price(at(vol=vol), OptionType.CALL, 120.0, Barrier.UP_AND_OUT)
        for vol in VOL_LADDER
    ]
    assert values == pytest.approx([3.117, 1.923, 1.107, 0.662, 0.418, 0.191], abs=5e-3)
    assert values == sorted(values, reverse=True)
    # While the vanilla strictly rises across the same ladder.
    vanillas = [price(at(vol=vol), OptionType.CALL) for vol in VOL_LADDER]
    assert vanillas == sorted(vanillas)
    # And it is worth more at a volatility below all of them, so there is a peak.
    peak = barrier_price(at(vol=0.0751), OptionType.CALL, 120.0, Barrier.UP_AND_OUT)
    assert peak > values[0]
    assert peak == pytest.approx(3.435, abs=5e-3)


def test_the_vega_of_an_up_and_out_call_changes_sign() -> None:
    """Which is why no implied-volatility solve is offered for one.

    A bisection needs a monotone relation between price and volatility. Here
    there are two roots or none at almost every price, so a solve would return
    whichever root the bracket happened to contain and report success.
    """

    def vega(vol: float) -> float:
        step = 1e-3
        high = barrier_price(
            at(vol=vol + step), OptionType.CALL, 120.0, Barrier.UP_AND_OUT
        )
        low = barrier_price(
            at(vol=vol - step), OptionType.CALL, 120.0, Barrier.UP_AND_OUT
        )
        return (high - low) / (2.0 * step)

    assert vega(0.05) == pytest.approx(31.0, abs=1.0)
    assert vega(0.20) == pytest.approx(-11.9, abs=1.0)
    assert vega(0.05) > 0.0 > vega(0.20)


@pytest.mark.parametrize("option", list(OptionType))
def test_a_knock_out_is_worth_less_the_nearer_its_barrier(option: OptionType) -> None:
    """Monotone in the level, which is the one monotonicity that does hold."""
    values = [
        barrier_price(at(), option, level, Barrier.UP_AND_OUT)
        for level in (101.0, 105.0, 115.0, 130.0, 200.0)
    ]
    assert values == sorted(values)


def test_the_monitoring_shift_moves_away_from_the_spot_on_both_sides() -> None:
    """Up barriers go up and down barriers go down: harder to breach, either way."""
    up = monitoring_shift(120.0, Barrier.UP_AND_OUT, VOL, 252.0)
    down = monitoring_shift(85.0, Barrier.DOWN_AND_OUT, VOL, 252.0)
    assert up > 120.0
    assert down < 85.0
    # Symmetric in log terms, since the correction is a factor.
    assert math.log(up / 120.0) == pytest.approx(-math.log(down / 85.0), rel=1e-12)
    assert monitoring_shift(120.0, Barrier.UP_AND_OUT, 0.0, 252.0) == 120.0


def test_the_monitoring_shift_is_small_and_what_it_is_worth_is_not() -> None:
    """The price is levered about eighteen times to the barrier level.

    So a daily close against continuous monitoring is a 13% difference in price
    from a 0.74% difference in the level — far more than the spread the
    contract trades on, and the reason this correction exists at all.
    """
    inputs = at()
    continuous = barrier_price(inputs, OptionType.CALL, 120.0, Barrier.UP_AND_OUT)
    shifts = []
    shares = []
    for frequency in (252.0, 52.0, 12.0):
        level = monitoring_shift(120.0, Barrier.UP_AND_OUT, VOL, frequency)
        shifts.append(level / 120.0 - 1.0)
        shares.append(
            barrier_price(inputs, OptionType.CALL, level, Barrier.UP_AND_OUT)
            / continuous
            - 1.0
        )
    assert shifts == pytest.approx([0.00737, 0.01629, 0.03421], abs=5e-5)
    assert shares == pytest.approx([0.1292, 0.2947, 0.6524], abs=5e-4)
    assert shares == sorted(shares)
    assert shares[0] / shifts[0] == pytest.approx(17.5, abs=1.0)


def test_discrete_monitoring_cuts_a_knock_in() -> None:
    """The level moves the same way, so the in is worth less rather than more."""
    inputs = at()
    continuous = barrier_price(inputs, OptionType.CALL, 120.0, Barrier.UP_AND_IN)
    level = monitoring_shift(120.0, Barrier.UP_AND_IN, VOL, 252.0)
    assert (
        barrier_price(inputs, OptionType.CALL, level, Barrier.UP_AND_IN) < continuous
    )


@pytest.mark.parametrize(
    ("frequency", "steps", "error"),
    [(52.0, 52, 0.0092), (12.0, 12, 0.0682)],
)
def test_the_monitoring_correction_is_measured_against_the_real_contract(
    frequency: float, steps: int, error: float
) -> None:
    """How good the correction is, simulated rather than assumed.

    :func:`moneyness.monte_carlo.barrier` with ``bridge=False`` prices the
    genuinely discrete contract — a different contract, not a worse estimate of
    the continuous one — so it says what the correction is actually worth. It
    is -0.33% off for a daily close, +0.92% weekly and **+6.82%** monthly: good
    daily, fine weekly, and at monthly monitoring it overstates the contract by
    seven per cent. Still far better than ignoring the effect, which would
    understate the monthly contract by thirty-five, but not a figure to quote.

    The daily case is left out of the suite rather than asserted loosely: 252
    steps at enough paths to resolve a third of a per cent costs more than the
    check is worth, and the two here carry the same claim.
    """
    inputs = at()
    level = monitoring_shift(120.0, Barrier.UP_AND_OUT, VOL, frequency)
    corrected = barrier_price(inputs, OptionType.CALL, level, Barrier.UP_AND_OUT)
    truth = simulate(
        inputs,
        OptionType.CALL,
        120.0,
        Barrier.UP_AND_OUT,
        steps=steps,
        settings=Settings(paths=60_000, seed=11),
        bridge=False,
    )
    assert corrected / truth.value - 1.0 == pytest.approx(error, abs=0.02)
    # And the discrete contract is dearer than the continuous one, which is the
    # whole direction of the effect.
    assert truth.value > barrier_price(
        inputs, OptionType.CALL, 120.0, Barrier.UP_AND_OUT
    )


# -- the terms themselves -----------------------------------------------------


def test_the_reflected_terms_vanish_as_the_barrier_recedes() -> None:
    """The mechanism behind the distant-barrier limit above."""
    inputs = at()
    values = [
        abs(reflected_term(inputs, level, OptionType.CALL, Barrier.UP_AND_OUT))
        for level in (150.0, 300.0, 1000.0)
    ]
    assert values == sorted(values, reverse=True)
    assert values[-1] < 1e-10


def test_a_barrier_at_the_strike_collapses_the_pairs_of_terms() -> None:
    """``beyond_term`` cut at the strike *is* ``vanilla_term``, by construction.

    Asserted because the two differ only in which level they cut at, and a
    confusion between them is a mistake parity would not see: it would move
    ``in`` and ``out`` by the same amount in opposite directions.
    """
    inputs = at(strike=95.0)
    for option in OptionType:
        assert beyond_term(inputs, 95.0, option) == pytest.approx(
            vanilla_term(inputs, option), rel=1e-15
        )
        assert reflected_beyond_term(
            inputs, 95.0, option, Barrier.DOWN_AND_OUT
        ) == pytest.approx(
            reflected_term(inputs, 95.0, option, Barrier.DOWN_AND_OUT), rel=1e-15
        )


# -- refusals -----------------------------------------------------------------


@pytest.mark.parametrize("level", [0.0, -1.0, math.inf, math.nan])
def test_a_barrier_that_is_not_a_level_is_refused(level: float) -> None:
    with pytest.raises(BarrierError, match="not a spot level"):
        barrier_price(at(), OptionType.CALL, level, Barrier.UP_AND_OUT)
    with pytest.raises(BarrierError, match="not a spot level"):
        is_touched(100.0, level, Barrier.UP_AND_OUT)
    with pytest.raises(BarrierError, match="not a spot level"):
        cannot_pay(100.0, level, Barrier.UP_AND_OUT, OptionType.CALL)


def test_a_spot_or_strike_that_cannot_be_reflected_is_refused() -> None:
    with pytest.raises(BarrierError, match="has no log"):
        barrier_price(
            Inputs(spot=0.0, strike=100.0, time=1.0, rate=0.05, vol=0.2),
            OptionType.CALL,
            120.0,
            Barrier.UP_AND_OUT,
        )
    with pytest.raises(BarrierError, match="cannot be reflected"):
        barrier_price(
            Inputs(spot=100.0, strike=0.0, time=1.0, rate=0.05, vol=0.2),
            OptionType.CALL,
            120.0,
            Barrier.UP_AND_OUT,
        )


@pytest.mark.parametrize(
    ("volatility", "frequency", "message"),
    [
        (-0.1, 252.0, "is not one"),
        (0.2, 0.0, "not a frequency"),
        (0.2, -12.0, "not a frequency"),
        (math.nan, 252.0, "is not one"),
    ],
)
def test_the_monitoring_shift_refuses_what_it_cannot_use(
    volatility: float, frequency: float, message: str
) -> None:
    with pytest.raises(BarrierError, match=message):
        monitoring_shift(120.0, Barrier.UP_AND_OUT, volatility, frequency)
