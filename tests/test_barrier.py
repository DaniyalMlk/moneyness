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
    Barrier,
    BarrierError,
    Knock,
    Side,
    barrier_price,
    beyond_term,
    is_structurally_worthless,
    reflected_beyond_term,
    reflected_term,
    shift_for_monitoring,
    vanilla_term,
)
from moneyness.bsm import Inputs, OptionType, price

SPOT = 100.0
TIME = 1.0
RATE = 0.05
CARRY = 0.02
VOL = 0.20

DOWN_LEVELS = (80.0, 85.0, 95.0, 99.0)
UP_LEVELS = (101.0, 105.0, 115.0, 120.0)
STRIKES = (85.0, 90.0, 100.0, 110.0, 125.0)


def at(
    strike: float = 100.0, vol: float = VOL, spot: float = SPOT, time: float = TIME
) -> Inputs:
    return Inputs(spot=spot, strike=strike, time=time, rate=RATE, vol=vol, carry=CARRY)


def contracts() -> list[tuple[Side, float, float, OptionType]]:
    """Every side, level and strike combination that is a live contract."""
    made = []
    for side, option in itertools.product(Side, OptionType):
        levels = DOWN_LEVELS if side is Side.DOWN else UP_LEVELS
        for level, strike in itertools.product(levels, STRIKES):
            made.append((side, level, strike, option))
    return made


# -- the vanilla term, against the package's own vanilla ----------------------


@pytest.mark.parametrize("strike", STRIKES)
@pytest.mark.parametrize("option", list(OptionType))
def test_the_vanilla_term_is_the_package_vanilla(
    strike: float, option: OptionType
) -> None:
    """Same formula, grouped differently, so they agree to the last bit or close.

    It is a cross-module check on this file's arithmetic — the drift enters the
    terms here through ``mu`` rather than through ``d1`` directly, and a
    mistake in that algebra would show up here and nowhere else.
    """
    inputs = at(strike)
    assert vanilla_term(inputs, 1.0, option) == pytest.approx(
        price(inputs, option), rel=1e-15
    )


# -- parity -------------------------------------------------------------------


@pytest.mark.parametrize(("side", "level", "strike", "option"), contracts())
def test_in_plus_out_is_the_vanilla(
    side: Side, level: float, strike: float, option: OptionType
) -> None:
    inputs = at(strike)
    knocked_in = barrier_price(inputs, Barrier(level, side, Knock.IN), option)
    knocked_out = barrier_price(inputs, Barrier(level, side, Knock.OUT), option)
    assert knocked_in + knocked_out == pytest.approx(
        price(inputs, option), rel=1e-13, abs=1e-14
    )


@pytest.mark.parametrize(("side", "level", "strike", "option"), contracts())
def test_both_halves_are_non_negative_and_bounded_by_the_vanilla(
    side: Side, level: float, strike: float, option: OptionType
) -> None:
    """Each is a sub-portfolio of the vanilla, so neither can exceed it.

    Worth asserting separately from parity: a pair of errors that cancel in the
    sum would pass parity, and one of them going negative or above the vanilla
    would not pass this.
    """
    inputs = at(strike)
    vanilla = price(inputs, option)
    for knock in Knock:
        value = barrier_price(inputs, Barrier(level, side, knock), option)
        assert -1e-14 <= value <= vanilla + 1e-14


# -- the structural cases -----------------------------------------------------


@pytest.mark.parametrize("option", list(OptionType))
@pytest.mark.parametrize("side", list(Side))
def test_the_spot_on_the_barrier_is_exact(side: Side, option: OptionType) -> None:
    """The out is dead and the in is the vanilla, bit for bit.

    Bit for bit because the vanilla comes from :func:`moneyness.bsm.price`
    rather than being recomputed out of the terms, which is the reason it is
    taken from there.
    """
    level = 100.0
    inputs = at(strike=105.0 if option is OptionType.PUT else 95.0)
    assert barrier_price(inputs, Barrier(level, side, Knock.OUT), option) == 0.0
    assert barrier_price(inputs, Barrier(level, side, Knock.IN), option) == price(
        inputs, option
    )


def test_a_computed_barrier_level_still_hits_the_structural_case() -> None:
    """A level arrived at by arithmetic must not miss the exact answer by a bit.

    ``0.8 * 125.0`` is not ``100.0`` in binary. A caller who writes the barrier
    as a percentage of a reference and gets a near-zero price where zero is
    correct has been failed by a float comparison, so the comparison is made at
    a fixed number of decimals.
    """
    level = 0.8 * 125.0
    barrier = Barrier(level, Side.UP, Knock.OUT)
    assert barrier.is_touched(100.0)
    assert barrier_price(at(strike=95.0), barrier, OptionType.CALL) == 0.0


@pytest.mark.parametrize("level", [100.0, 105.0, 120.0])
def test_an_up_and_out_call_struck_above_its_barrier_is_worthless(
    level: float,
) -> None:
    """Exactly 0.0: paying needs ``S_T > K >= H``, which crosses ``H``."""
    inputs = at(strike=level + 5.0, spot=level - 10.0)
    barrier = Barrier(level, Side.UP, Knock.OUT)
    assert is_structurally_worthless(barrier, inputs.strike, OptionType.CALL)
    assert barrier_price(inputs, barrier, OptionType.CALL) == 0.0
    # And the knock-in is then the whole vanilla, for the same reason read back.
    assert barrier_price(
        inputs, Barrier(level, Side.UP, Knock.IN), OptionType.CALL
    ) == price(inputs, OptionType.CALL)


@pytest.mark.parametrize("level", [80.0, 95.0])
def test_a_down_and_out_put_struck_below_its_barrier_is_worthless(
    level: float,
) -> None:
    inputs = at(strike=level - 5.0, spot=level + 10.0)
    barrier = Barrier(level, Side.DOWN, Knock.OUT)
    assert is_structurally_worthless(barrier, inputs.strike, OptionType.PUT)
    assert barrier_price(inputs, barrier, OptionType.PUT) == 0.0


def test_the_other_six_are_not_structurally_worthless() -> None:
    """The exemption is narrow, and claiming it widely would zero live contracts."""
    live = [
        (Barrier(85.0, Side.DOWN, Knock.OUT), 100.0, OptionType.CALL),
        (Barrier(120.0, Side.UP, Knock.OUT), 100.0, OptionType.CALL),
        (Barrier(85.0, Side.DOWN, Knock.OUT), 100.0, OptionType.PUT),
        (Barrier(120.0, Side.UP, Knock.OUT), 100.0, OptionType.PUT),
    ]
    for barrier, strike, option in live:
        assert not is_structurally_worthless(barrier, strike, option)
        assert barrier_price(at(strike), barrier, option) > 0.0


# -- limits -------------------------------------------------------------------


def test_a_distant_barrier_recovers_the_vanilla() -> None:
    """And the convergence is fast, which is worth knowing before widening one.

    An up barrier at 150 still costs the call 24% of its value; at 200 it costs
    0.58%; at 300, 8.7e-07; and by 500 the knock-out is the vanilla to 3.7e-14.
    """
    inputs = at()
    vanilla = price(inputs, OptionType.CALL)
    gaps = [
        abs(
            barrier_price(inputs, Barrier(level, Side.UP, Knock.OUT), OptionType.CALL)
            / vanilla
            - 1.0
        )
        for level in (150.0, 200.0, 300.0, 500.0)
    ]
    assert gaps == sorted(gaps, reverse=True)
    assert gaps[0] == pytest.approx(0.2379, abs=5e-4)
    assert gaps[1] == pytest.approx(0.00584, abs=5e-5)
    assert gaps[-1] < 1e-12


@pytest.mark.parametrize(("side", "level"), [(Side.DOWN, 85.0), (Side.UP, 120.0)])
@pytest.mark.parametrize("option", list(OptionType))
def test_zero_volatility_follows_the_forward(
    side: Side, level: float, option: OptionType
) -> None:
    """With no diffusion the path is the forward, which is monotone in time.

    So it reaches the barrier if and only if its terminal value is past it, and
    the knock-out is the discounted intrinsic or nothing, with no running
    maximum needed. The forward here is 102.02, so neither barrier is reached.
    """
    inputs = at(strike=100.0, vol=0.0)
    forward = SPOT * math.exp(CARRY * TIME)
    assert 85.0 < forward < 120.0
    out = barrier_price(inputs, Barrier(level, side, Knock.OUT), option)
    assert out == pytest.approx(price(inputs, option), rel=1e-15)
    assert barrier_price(inputs, Barrier(level, side, Knock.IN), option) == 0.0


def test_zero_volatility_with_the_forward_past_the_barrier_knocks_out() -> None:
    """A large carry walks the forward through the level, deterministically."""
    inputs = Inputs(spot=100.0, strike=100.0, time=1.0, rate=0.05, vol=0.0, carry=0.40)
    forward = 100.0 * math.exp(0.40)
    assert forward > 120.0
    assert (
        barrier_price(inputs, Barrier(120.0, Side.UP, Knock.OUT), OptionType.CALL)
        == 0.0
    )
    assert barrier_price(
        inputs, Barrier(120.0, Side.UP, Knock.IN), OptionType.CALL
    ) == pytest.approx(price(inputs, OptionType.CALL), rel=1e-15)


def test_zero_time_is_the_payoff() -> None:
    inputs = at(strike=95.0, time=0.0)
    assert barrier_price(
        inputs, Barrier(120.0, Side.UP, Knock.OUT), OptionType.CALL
    ) == pytest.approx(5.0, rel=1e-15)


# -- the measured behaviour ---------------------------------------------------


def test_an_up_and_out_call_is_not_monotone_in_volatility() -> None:
    """The finding this module exists to record.

    Every option in :mod:`moneyness.bsm` is increasing in volatility. This one
    peaks at 7.51% and loses 94% of its value between there and 40%, because
    the volatility that pays for the optionality also pays for the knock-out.
    """
    barrier = Barrier(120.0, Side.UP, Knock.OUT)
    values = [
        barrier_price(at(vol=vol), barrier, OptionType.CALL)
        for vol in (0.10, 0.15, 0.20, 0.25, 0.30, 0.40)
    ]
    assert values == pytest.approx([3.117, 1.923, 1.107, 0.662, 0.418, 0.191], abs=5e-3)
    # Strictly falling across that range, while the vanilla strictly rises.
    assert values == sorted(values, reverse=True)
    vanillas = [
        price(at(vol=vol), OptionType.CALL)
        for vol in (0.10, 0.15, 0.20, 0.25, 0.30, 0.40)
    ]
    assert vanillas == sorted(vanillas)
    # And it is higher at a volatility below all of them, so there is a peak.
    assert barrier_price(at(vol=0.0751), barrier, OptionType.CALL) > values[0]
    assert barrier_price(at(vol=0.0751), barrier, OptionType.CALL) == pytest.approx(
        3.435, abs=5e-3
    )


def test_the_vega_of_an_up_and_out_call_changes_sign() -> None:
    """Which is why no implied-volatility solve is offered for one.

    A bisection needs a monotone relation between price and volatility. Here
    there are two roots or none at almost every price, so a solve would return
    whichever root the bracket happened to contain and report success.
    """
    barrier = Barrier(120.0, Side.UP, Knock.OUT)

    def vega(vol: float) -> float:
        step = 1e-3
        high = barrier_price(at(vol=vol + step), barrier, OptionType.CALL)
        low = barrier_price(at(vol=vol - step), barrier, OptionType.CALL)
        return (high - low) / (2.0 * step)

    assert vega(0.05) == pytest.approx(31.0, abs=1.0)
    assert vega(0.20) == pytest.approx(-11.9, abs=1.0)
    assert vega(0.05) > 0.0 > vega(0.20)


@pytest.mark.parametrize("option", list(OptionType))
def test_a_knock_out_is_worth_less_the_nearer_its_barrier(option: OptionType) -> None:
    """Monotone in the level, which is the one monotonicity that does hold."""
    values = [
        barrier_price(at(), Barrier(level, Side.UP, Knock.OUT), option)
        for level in (101.0, 105.0, 115.0, 130.0, 200.0)
    ]
    assert values == sorted(values)


def test_discrete_monitoring_is_worth_more_than_the_shift_suggests() -> None:
    """The price is levered about seventeen times to the barrier level.

    So a daily close against continuous monitoring is a 12.9% difference in
    price from a 0.74% difference in the level — an order of magnitude more
    than the correction's own size would suggest, and far more than the spread
    the contract trades on.
    """
    barrier = Barrier(120.0, Side.UP, Knock.OUT)
    inputs = at()
    continuous = barrier_price(inputs, barrier, OptionType.CALL)
    shares = []
    shifts = []
    for frequency in (252.0, 52.0, 12.0):
        moved = shift_for_monitoring(barrier, VOL, frequency)
        assert moved.side is barrier.side
        assert moved.knock is barrier.knock
        shifts.append(moved.level / barrier.level - 1.0)
        shares.append(
            barrier_price(inputs, moved, OptionType.CALL) / continuous - 1.0
        )
    assert shifts == pytest.approx([0.00737, 0.01629, 0.03421], abs=5e-5)
    assert shares == pytest.approx([0.1292, 0.2947, 0.6524], abs=5e-4)
    # Monotone in the spacing, and the leverage is roughly constant.
    assert shares == sorted(shares)
    assert shares[0] / shifts[0] == pytest.approx(17.5, abs=1.0)


def test_the_monitoring_shift_moves_away_from_the_spot_on_both_sides() -> None:
    """Up barriers go up and down barriers go down: harder to breach, either way."""
    up = shift_for_monitoring(Barrier(120.0, Side.UP, Knock.OUT), VOL, 252.0)
    down = shift_for_monitoring(Barrier(85.0, Side.DOWN, Knock.OUT), VOL, 252.0)
    assert up.level > 120.0
    assert down.level < 85.0
    # Symmetric in log terms, since the correction is a factor.
    assert math.log(up.level / 120.0) == pytest.approx(
        -math.log(down.level / 85.0), rel=1e-12
    )


def test_a_monitoring_shift_at_zero_volatility_does_nothing() -> None:
    barrier = Barrier(120.0, Side.UP, Knock.OUT)
    assert shift_for_monitoring(barrier, 0.0, 252.0).level == 120.0


def test_discrete_monitoring_cuts_a_knock_in() -> None:
    """The level moves the same way, so the in is worth less rather than more."""
    barrier = Barrier(120.0, Side.UP, Knock.IN)
    inputs = at()
    continuous = barrier_price(inputs, barrier, OptionType.CALL)
    daily = barrier_price(
        inputs, shift_for_monitoring(barrier, VOL, 252.0), OptionType.CALL
    )
    assert daily < continuous


# -- the terms themselves -----------------------------------------------------


def test_the_reflected_terms_vanish_as_the_barrier_recedes() -> None:
    """Which is the mechanism behind the distant-barrier limit above."""
    inputs = at()
    values = [
        abs(reflected_term(inputs, level, OptionType.CALL, Side.UP))
        for level in (150.0, 300.0, 1000.0)
    ]
    assert values == sorted(values, reverse=True)
    assert values[-1] < 1e-10


def test_the_beyond_terms_are_the_vanilla_when_cut_at_the_strike() -> None:
    """``beyond_term`` at the strike *is* ``vanilla_term``, by construction.

    Asserted because the two differ only in which level they cut at, and a
    confusion between them is the kind of mistake parity would not see — it
    would move ``in`` and ``out`` by the same amount in opposite directions.
    """
    inputs = at(strike=95.0)
    for option in OptionType:
        assert beyond_term(inputs, 95.0, option) == pytest.approx(
            vanilla_term(inputs, 1.0, option), rel=1e-15
        )
        assert reflected_beyond_term(
            inputs, 95.0, option, Side.DOWN
        ) == pytest.approx(
            reflected_term(inputs, 95.0, option, Side.DOWN), rel=1e-15
        )


# -- refusals -----------------------------------------------------------------


@pytest.mark.parametrize("level", [0.0, -1.0, math.inf, math.nan])
def test_a_barrier_that_is_not_a_level_is_refused(level: float) -> None:
    with pytest.raises(BarrierError, match="not a spot level"):
        Barrier(level, Side.UP, Knock.OUT)


def test_a_spot_or_strike_that_cannot_be_reflected_is_refused() -> None:
    with pytest.raises(BarrierError, match="has no log"):
        barrier_price(
            Inputs(spot=0.0, strike=100.0, time=1.0, rate=0.05, vol=0.2),
            Barrier(120.0, Side.UP, Knock.OUT),
            OptionType.CALL,
        )
    with pytest.raises(BarrierError, match="cannot be reflected"):
        barrier_price(
            Inputs(spot=100.0, strike=0.0, time=1.0, rate=0.05, vol=0.2),
            Barrier(120.0, Side.UP, Knock.OUT),
            OptionType.CALL,
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
        shift_for_monitoring(Barrier(120.0, Side.UP, Knock.OUT), volatility, frequency)


def test_a_barrier_reports_itself() -> None:
    assert Barrier(120.0, Side.UP, Knock.OUT).name == "up-and-out at 120"
    assert Barrier(85.5, Side.DOWN, Knock.IN).name == "down-and-in at 85.5"
    assert Side.DOWN.sign == 1.0
    assert Side.UP.sign == -1.0
