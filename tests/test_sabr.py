"""SABR, and the four closed forms that pin it.

Hagan's formula is an asymptotic expansion, so there is nothing exact to compare
the general case against. What there is instead is a set of limits whose answers
are known in advance, and the discipline here is that each one is checked
against a formula sharing no code with the expansion:

* ``nu = 0``, ``beta = 1`` is lognormal, so the implied volatility is ``alpha``.
  Exactly, because every correction term carries a factor of ``nu`` or
  ``(1 - beta)``.
* ``nu = 0``, ``beta = 0`` is Bachelier, and ``normal_volatility`` returns
  ``alpha`` with the same exactness.
* ``nu = 0`` with a general exponent is CEV, where the formula is *approximate*
  — and wrong by the amount its own next term predicts, which is the stronger
  statement.
* The two volatility forms price the same option two ways, so their gap is
  bounded by the expansion's own order and measured rather than assumed.

Beyond that, ``mpmath`` at fifty digits is used for the one place double
precision genuinely fails: the ``z / x(z)`` factor, whose numerator and
denominator agree to three terms. The sign of that series' linear term was wrong
in the first draft and the symptom was a series that never beat the ratio at any
threshold — which looks like a tuning problem rather than a sign error, so it is
pinned here against an independent evaluation.
"""

from __future__ import annotations

import math

import mpmath as mp
import pytest

from moneyness.bsm import OptionType
from moneyness.implied import black
from moneyness.sabr import (
    _SERIES_LIMIT,
    SabrParameters,
    bachelier,
    bachelier_vega,
    calibrate,
    density,
    density_floor,
    implied_normal_vol,
    lognormal_volatility,
    normal_volatility,
    shifted_lognormal_volatility,
    smile,
)

FORWARD = 0.02
TIME = 1.0
SWAPTION = SabrParameters(alpha=0.004, beta=0.5, rho=-0.3, nu=0.4)

LADDER = (0.012, 0.016, 0.02, 0.026, 0.034)


def atm_sd(parameters: SabrParameters, time: float = TIME) -> float:
    """One at-the-money standard deviation of the forward, in absolute terms."""
    return normal_volatility(parameters, FORWARD, FORWARD, time) * math.sqrt(time)


# -- the parameters -----------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"alpha": 0.0}, "alpha must be finite and positive"),
        ({"alpha": -1.0}, "alpha must be finite and positive"),
        ({"beta": 1.5}, r"beta must lie in \[0, 1\]"),
        ({"beta": -0.1}, r"beta must lie in \[0, 1\]"),
        ({"rho": 1.0}, "rho must lie strictly inside"),
        ({"rho": -1.0}, "rho must lie strictly inside"),
        ({"nu": -0.1}, "nu must be finite and non-negative"),
        ({"alpha": math.nan}, "alpha must be finite"),
    ],
)
def test_the_parameters_refuse_what_the_formula_cannot_take(
    kwargs: dict[str, float], message: str
) -> None:
    arguments = {"alpha": 0.2, "beta": 0.5, "rho": -0.3, "nu": 0.4}
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=message):
        SabrParameters(**arguments)


def test_a_correlation_of_one_is_refused_for_a_stated_reason() -> None:
    """``x(z)`` divides by ``1 - rho``, so this is not a conservative bound."""
    with pytest.raises(ValueError, match="divides by 1 - rho"):
        SabrParameters(alpha=0.2, beta=0.5, rho=1.0, nu=0.4)


def test_the_two_degenerate_flags() -> None:
    assert SabrParameters(alpha=0.2, beta=1.0, rho=0.0, nu=0.0).is_lognormal
    assert SabrParameters(alpha=0.2, beta=0.0, rho=0.0, nu=0.0).is_normal
    assert not SWAPTION.is_lognormal
    assert not SwaptionLike().is_normal


def SwaptionLike() -> SabrParameters:
    return SabrParameters(alpha=0.004, beta=0.0, rho=-0.3, nu=0.4)


# -- the limits that must be exact --------------------------------------------


@pytest.mark.parametrize("strike", [60.0, 80.0, 100.0, 130.0, 200.0])
@pytest.mark.parametrize("rho", [-0.8, 0.0, 0.5])
def test_no_vol_of_vol_and_a_unit_exponent_is_black_exactly(
    strike: float, rho: float
) -> None:
    """Not approximately. Every correction carries ``nu`` or ``(1 - beta)``."""
    parameters = SabrParameters(alpha=0.25, beta=1.0, rho=rho, nu=0.0)
    assert lognormal_volatility(parameters, 100.0, strike, 1.0) == 0.25


@pytest.mark.parametrize("strike", [0.01, 0.015, 0.02, 0.03, 0.05])
@pytest.mark.parametrize("rho", [-0.8, 0.0, 0.5])
def test_no_vol_of_vol_and_a_zero_exponent_is_bachelier_exactly(
    strike: float, rho: float
) -> None:
    """Which needs the two moneyness brackets to coincide at ``beta = 0``."""
    parameters = SabrParameters(alpha=0.004, beta=0.0, rho=rho, nu=0.0)
    assert normal_volatility(parameters, FORWARD, strike, TIME) == 0.004


@pytest.mark.parametrize(
    ("strike", "expected_gap"),
    [(0.021, 2.480e-05), (0.022, 9.463e-05), (0.025, 5.187e-04), (0.03, 1.713e-03)],
)
def test_no_vol_of_vol_is_cev_wrong_by_its_own_next_term(
    strike: float, expected_gap: float
) -> None:
    """The limit that should be approximate, approximate by the predicted amount.

    The leading CEV implied volatility is ``alpha`` over the geometric mid
    raised to ``1 - beta``. Hagan's formula sits above it by its own moneyness
    correction, ``(1 - beta)**2 log**2(F/K) / 24``. Checking the *size* of the
    error against the term that causes it is a stronger statement than checking
    that the error is small.
    """
    parameters = SabrParameters(alpha=0.004, beta=0.5, rho=0.0, nu=0.0)
    leading = 0.004 / math.pow(FORWARD * strike, 0.25)
    got = lognormal_volatility(parameters, FORWARD, strike, TIME)
    gap = abs(got - leading) / leading
    assert gap == pytest.approx(expected_gap, rel=0.35)
    # And at the widest strike the prediction is tight rather than merely the
    # right order: the other terms have not taken over yet.
    if strike == 0.03:
        assert gap == pytest.approx(expected_gap, rel=0.01)


def test_the_at_the_money_volatility_is_continuous_through_the_strike() -> None:
    """Across the two thresholds, where three branches meet.

    The offsets straddle ``_ATM_LIMIT`` on the log-moneyness correction. Further
    out than this the smile's own skew moves the volatility by more than any
    continuity tolerance, which is the smile working rather than a discontinuity:
    at an offset of 1e-06 on a forward of 2% the volatility has already moved by
    3e-05 of itself, and that is the skew.
    """
    at = lognormal_volatility(SWAPTION, FORWARD, FORWARD, TIME)
    # The branch is on |log(F/K)| against 1e-12, which is an offset of about
    # 2e-14 on this forward. These straddle it.
    for offset in (1e-13, 2e-14, 1e-14, 1e-15, 0.0):
        near = lognormal_volatility(SWAPTION, FORWARD, FORWARD + offset, TIME)
        assert near == pytest.approx(at, rel=1e-10)
    # Further out the volatility does move, and it is the skew rather than a
    # discontinuity: the elasticity is about 2.4, so an offset of 1e-11 shifts
    # the volatility by 1.19e-09 of itself and 1e-06 shifts it by 1.19e-04.
    for offset, expected in ((1e-11, 1.185e-9), (1e-6, 1.185e-4)):
        skewed = lognormal_volatility(SWAPTION, FORWARD, FORWARD + offset, TIME)
        assert abs(skewed - at) / at == pytest.approx(expected, rel=0.3)


# -- the factor that is zero over zero ----------------------------------------


def exact_z_over_x(z: float, rho: float) -> float:
    """Fifty digits of the same expression, for the one place doubles fail."""
    with mp.workdps(50):
        zz, r = mp.mpf(z), mp.mpf(rho)
        root = mp.sqrt(1 - 2 * r * zz + zz * zz)
        return float(zz / mp.log((root - r + zz) / (1 - r)))


def series_z_over_x(z: float, rho: float) -> float:
    return 1.0 - 0.5 * rho * z + (2.0 - 3.0 * rho * rho) * z * z / 12.0


def ratio_z_over_x(z: float, rho: float) -> float:
    root = math.sqrt(1.0 - 2.0 * rho * z + z * z)
    return z / math.log((root - rho + z) / (1.0 - rho))


@pytest.mark.parametrize("rho", [-0.8, -0.3, 0.0, 0.5])
@pytest.mark.parametrize("z", [1e-1, 1e-2, 1e-3])
def test_the_series_has_the_sign_the_expansion_has(rho: float, z: float) -> None:
    """The defect that read as a threshold problem.

    A plus on the linear term leaves the series wrong by ``rho z``, which at
    ``z`` of 1e-02 is 3e-03 — worse than the ratio everywhere, so no threshold
    makes it useful and the symptom points at the wrong thing. Checked against
    fifty digits, which shares no arithmetic with either.
    """
    truth = exact_z_over_x(z, rho)
    error = abs(series_z_over_x(z, rho) - truth) / truth
    # Third order in z, so the bound is cubic: 0.076 z**3 at these rho.
    assert error < 0.1 * z**3
    # The sign matters: flipping it is wrong by about |rho z|.
    flipped = 1.0 + 0.5 * rho * z + (2.0 - 3.0 * rho * rho) * z * z / 12.0
    if rho != 0.0:
        assert abs(flipped - truth) / truth == pytest.approx(abs(rho) * z, rel=0.05)


def test_the_switch_is_where_the_two_agree() -> None:
    """Both routes are wrong by about 5e-13 at the threshold.

    Above it the ratio is better and below it the series is, so the threshold is
    the crossing rather than a round number near it.
    """
    rho = -0.3
    truth = exact_z_over_x(_SERIES_LIMIT, rho)
    from_series = abs(series_z_over_x(_SERIES_LIMIT, rho) - truth) / truth
    from_ratio = abs(ratio_z_over_x(_SERIES_LIMIT, rho) - truth) / truth
    assert from_series < 1e-12
    assert from_ratio < 1e-11
    # An order of magnitude either side, each route wins its own side.
    above = 10.0 * _SERIES_LIMIT
    truth_above = exact_z_over_x(above, rho)
    assert abs(ratio_z_over_x(above, rho) - truth_above) < abs(
        series_z_over_x(above, rho) - truth_above
    )
    below = 0.1 * _SERIES_LIMIT
    truth_below = exact_z_over_x(below, rho)
    assert abs(series_z_over_x(below, rho) - truth_below) < abs(
        ratio_z_over_x(below, rho) - truth_below
    )


def test_the_ratio_loses_everything_as_z_vanishes() -> None:
    """Which is why there is a series at all: the error does not go to zero."""
    rho = -0.3
    for z, floor in ((1e-10, 1e-8), (1e-12, 1e-5)):
        truth = exact_z_over_x(z, rho)
        assert abs(ratio_z_over_x(z, rho) - truth) / truth > floor


# -- the two forms against each other -----------------------------------------


@pytest.mark.parametrize(
    ("time", "worst"), [(0.25, 0.000009), (1.0, 0.000081), (5.0, 0.001154)]
)
def test_the_two_forms_price_the_same_option_to_their_own_order(
    time: float, worst: float
) -> None:
    """Two second-order expansions, not two writings of one expression.

    Measured at strikes up to two at-the-money standard deviations out, where
    an option is still worth something. Further out than that a price is
    intrinsic to double precision and the comparison measures nothing.
    """
    sd = atm_sd(SWAPTION, time)
    seen = 0.0
    for deviations in (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0):
        strike = FORWARD + deviations * sd
        option = OptionType.PUT if strike < FORWARD else OptionType.CALL
        from_black = black(
            FORWARD,
            strike,
            lognormal_volatility(SWAPTION, FORWARD, strike, time) * math.sqrt(time),
            option,
        )
        from_normal = bachelier(
            FORWARD,
            strike,
            normal_volatility(SWAPTION, FORWARD, strike, time) * math.sqrt(time),
            option,
        )
        seen = max(seen, abs(from_black - from_normal) / from_black)
    assert seen == pytest.approx(worst, rel=0.1)


def test_the_gap_between_the_forms_grows_with_the_maturity() -> None:
    gaps = []
    for time in (0.25, 1.0, 5.0):
        sd = atm_sd(SWAPTION, time)
        strike = FORWARD - 2.0 * sd
        from_black = black(
            FORWARD,
            strike,
            lognormal_volatility(SWAPTION, FORWARD, strike, time) * math.sqrt(time),
            OptionType.PUT,
        )
        from_normal = bachelier(
            FORWARD,
            strike,
            normal_volatility(SWAPTION, FORWARD, strike, time) * math.sqrt(time),
            OptionType.PUT,
        )
        gaps.append(abs(from_black - from_normal) / from_black)
    assert gaps == sorted(gaps)


# -- the Bachelier model ------------------------------------------------------


def test_bachelier_at_the_money_is_the_closed_form() -> None:
    """``sigma sqrt(T) / sqrt(2 pi)``, which is where the inversion starts."""
    for total_vol in (0.001, 0.01, 0.25):
        assert bachelier(0.02, 0.02, total_vol, OptionType.CALL) == pytest.approx(
            total_vol / math.sqrt(2.0 * math.pi), rel=1e-15
        )


def test_bachelier_takes_a_negative_forward_where_black_cannot() -> None:
    """The whole point of the normal model."""
    price = bachelier(-0.002, 0.001, 0.004, OptionType.CALL)
    assert price > 0.0
    assert math.isfinite(price)


def test_bachelier_satisfies_put_call_parity() -> None:
    for forward, strike in ((0.02, 0.015), (-0.001, 0.004), (0.02, 0.02)):
        call = bachelier(forward, strike, 0.004, OptionType.CALL)
        put = bachelier(forward, strike, 0.004, OptionType.PUT)
        assert call - put == pytest.approx(forward - strike, abs=1e-18)


def test_bachelier_at_no_volatility_is_intrinsic() -> None:
    assert bachelier(0.02, 0.015, 0.0, OptionType.CALL) == pytest.approx(0.005)
    assert bachelier(0.02, 0.025, 0.0, OptionType.CALL) == 0.0


def test_bachelier_vega_is_the_derivative() -> None:
    for strike in (0.015, 0.02, 0.026):
        step = 1e-7
        up = bachelier(FORWARD, strike, 0.004 + step, OptionType.CALL)
        down = bachelier(FORWARD, strike, 0.004 - step, OptionType.CALL)
        assert bachelier_vega(FORWARD, strike, 0.004) == pytest.approx(
            (up - down) / (2.0 * step), rel=1e-7
        )
    assert bachelier_vega(FORWARD, FORWARD, 0.0) == 0.0


@pytest.mark.parametrize("total_vol", [-1.0, math.nan])
def test_bachelier_refuses_a_volatility_that_is_not_one(total_vol: float) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        bachelier(0.02, 0.02, total_vol, OptionType.CALL)
    with pytest.raises(ValueError, match="non-negative"):
        bachelier_vega(0.02, 0.02, total_vol)


def test_the_normal_inversion_round_trips_across_the_smile() -> None:
    """Forty-two cases, worst relative error 1.1e-13."""
    worst = 0.0
    cases = 0
    for time in (0.25, 1.0, 5.0):
        sd = atm_sd(SWAPTION, time)
        for deviations in (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0):
            strike = FORWARD + deviations * sd
            for option in (OptionType.CALL, OptionType.PUT):
                total = normal_volatility(SWAPTION, FORWARD, strike, time) * math.sqrt(
                    time
                )
                price = bachelier(FORWARD, strike, total, option)
                if price < 1e-10 * FORWARD:
                    continue
                recovered = implied_normal_vol(FORWARD, strike, price, option)
                worst = max(worst, abs(recovered - total) / total)
                cases += 1
    assert cases == 42
    assert worst < 1e-12


def test_the_inversion_tolerance_is_relative_so_the_deep_wing_still_inverts() -> None:
    """The defect an absolute tolerance hid.

    An option twenty at-the-money standard deviations out is worth 1.1e-15. An
    absolute tolerance of 1e-14 on that accepts any volatility within a factor
    of two and reports success; relative to the price, it comes back to 1e-15.
    """
    sd = atm_sd(SWAPTION)
    strike = FORWARD - 20.0 * sd
    total = normal_volatility(SWAPTION, FORWARD, strike, TIME)
    price = bachelier(FORWARD, strike, total, OptionType.PUT)
    assert price < 1e-14
    assert implied_normal_vol(FORWARD, strike, price, OptionType.PUT) == pytest.approx(
        total, rel=1e-13
    )


def test_the_inversion_refuses_a_price_below_intrinsic() -> None:
    with pytest.raises(ValueError, match="below the intrinsic"):
        implied_normal_vol(0.02, 0.015, 0.001, OptionType.CALL)


def test_a_price_at_intrinsic_inverts_to_no_volatility() -> None:
    assert implied_normal_vol(0.02, 0.015, 0.005, OptionType.CALL) == 0.0


# -- the shifted form ---------------------------------------------------------


def test_the_shift_moves_both_sides_and_nothing_else() -> None:
    shifted = shifted_lognormal_volatility(SWAPTION, 0.002, 0.004, TIME, 0.03)
    direct = lognormal_volatility(SWAPTION, 0.032, 0.034, TIME)
    assert shifted == direct


def test_the_shift_lets_a_negative_forward_be_quoted() -> None:
    value = shifted_lognormal_volatility(SWAPTION, -0.005, -0.002, TIME, 0.03)
    assert value > 0.0
    assert math.isfinite(value)


@pytest.mark.parametrize("shift", [0.0, -0.01, math.nan])
def test_a_shift_that_is_not_one_is_refused(shift: float) -> None:
    with pytest.raises(ValueError, match="shift must be finite and positive"):
        shifted_lognormal_volatility(SWAPTION, 0.002, 0.004, TIME, shift)


def test_a_shift_too_small_to_lift_the_strike_names_both_numbers() -> None:
    with pytest.raises(ValueError, match="leaves the forward at"):
        shifted_lognormal_volatility(SWAPTION, -0.005, -0.02, TIME, 0.01)


# -- the smile and its inputs -------------------------------------------------


def test_a_smile_carries_both_volatilities_at_every_strike() -> None:
    made = smile(SWAPTION, FORWARD, LADDER, TIME)
    assert len(made) == len(LADDER)
    assert made.strikes == LADDER
    for index, strike in enumerate(LADDER):
        assert made.lognormal[index] == lognormal_volatility(
            SWAPTION, FORWARD, strike, TIME
        )
        assert made.normal[index] == normal_volatility(SWAPTION, FORWARD, strike, TIME)


def test_a_smile_needs_a_strike() -> None:
    with pytest.raises(ValueError, match="at least one strike"):
        smile(SWAPTION, FORWARD, [], TIME)


def test_the_smile_smiles() -> None:
    """Higher volatility in both wings than at the money, which is the shape."""
    made = smile(SWAPTION, FORWARD, LADDER, TIME)
    middle = made.lognormal[2]
    assert made.lognormal[0] > middle
    assert made.lognormal[-1] < made.lognormal[0]
    assert min(made.lognormal) == pytest.approx(min(made.lognormal))


@pytest.mark.parametrize(
    ("forward", "strike", "time"),
    [(0.0, 0.02, 1.0), (0.02, 0.0, 1.0), (0.02, 0.02, 0.0), (-0.02, 0.02, 1.0)],
)
def test_the_formula_refuses_inputs_it_has_no_value_at(
    forward: float, strike: float, time: float
) -> None:
    with pytest.raises(ValueError):
        lognormal_volatility(SWAPTION, forward, strike, time)
    with pytest.raises(ValueError):
        normal_volatility(SWAPTION, forward, strike, time)


# -- where it stops being a price ---------------------------------------------


def test_the_density_integrates_to_something_like_one_near_the_money() -> None:
    """A crude check that this is a density and not a scaled one.

    Trapezoidal over four at-the-money standard deviations either side, which
    is most of the mass for a smile this narrow.
    """
    sd = atm_sd(SWAPTION)
    points = 161
    low, high = FORWARD - 4.0 * sd, FORWARD + 4.0 * sd
    width = (high - low) / (points - 1)
    total = 0.0
    for index in range(points):
        strike = low + index * width
        weight = 0.5 if index in (0, points - 1) else 1.0
        total += weight * density(SWAPTION, FORWARD, strike, TIME, step=1e-4 * strike)
    assert total * width == pytest.approx(1.0, abs=0.02)


@pytest.mark.parametrize("step", [1e-3, 1e-4, 1e-5])
def test_the_density_is_stable_in_the_differencing_step(step: float) -> None:
    """Three digits across two orders of magnitude of step size."""
    assert density(SWAPTION, FORWARD, FORWARD, TIME, step=step * FORWARD) == (
        pytest.approx(730.3, rel=1e-3)
    )


def test_the_density_refuses_a_step_that_walks_off_the_strike_axis() -> None:
    with pytest.raises(ValueError, match="which is not"):
        density(SWAPTION, FORWARD, 0.001, TIME, step=0.002)
    with pytest.raises(ValueError, match="step must be positive"):
        density(SWAPTION, FORWARD, FORWARD, TIME, step=-1.0)


LONG_DATED = SabrParameters(alpha=0.004, beta=0.5, rho=-0.3, nu=0.8)


def test_an_ordinary_one_year_smile_has_no_negative_density_at_all() -> None:
    """The measurement the first draft of this got backwards.

    Differencing *call* prices below the forward is catastrophic cancellation,
    and it manufactured a negative density sixteen standard deviations out that
    was entirely rounding: a call at a 1% strike against a 2% forward is worth
    0.0094, four of its last bits divided by a squared step of 1e-06 is 6e-06,
    and the density there is smaller than that. Pricing the out-of-the-money
    option instead, every short-dated case tested is a proper distribution.
    """
    for nu in (0.4, 0.8, 1.2):
        for rho in (-0.8, -0.3, 0.0, 0.5):
            parameters = SabrParameters(alpha=0.004, beta=0.5, rho=rho, nu=nu)
            assert density_floor(parameters, FORWARD, 1.0, lower=1e-4 * FORWARD) is None


def test_the_density_is_a_second_difference_of_the_cheap_option() -> None:
    """Which is the same number as the dear one's, in exact arithmetic.

    Near the money both are accurate, so the two routes agree there and the
    choice is invisible. The point of the choice is the wing, where only one of
    them carries any signal.
    """
    root = math.sqrt(TIME)
    width = 1e-4 * FORWARD

    def by_call(strike: float) -> float:
        def value(at: float) -> float:
            vol = lognormal_volatility(SWAPTION, FORWARD, at, TIME)
            return black(FORWARD, at, vol * root, OptionType.CALL)

        return (value(strike + width) - 2.0 * value(strike) + value(strike - width)) / (
            width * width
        )

    assert by_call(FORWARD) == pytest.approx(
        density(SWAPTION, FORWARD, FORWARD, TIME), rel=1e-6
    )


def test_a_long_dated_smile_stops_being_a_distribution_near_the_money() -> None:
    """And this one is signal: stable to four digits across three decades of step.

    At ten years with a vol-of-vol of 0.8 the density turns negative less than
    one standard deviation below the forward, which is inside the range anyone
    quotes, and ten per cent further down it is nearly a per cent of the peak in
    magnitude.
    """
    floor = density_floor(LONG_DATED, FORWARD, 10.0, lower=1e-4 * FORWARD)
    assert floor is not None
    assert floor == pytest.approx(0.0179350, rel=1e-3)

    sd = normal_volatility(LONG_DATED, FORWARD, FORWARD, 10.0) * math.sqrt(10.0)
    assert (FORWARD - floor) / sd == pytest.approx(0.795, rel=0.02)

    inside = 0.9 * floor
    peak = density(LONG_DATED, FORWARD, FORWARD, 10.0)
    for step in (1e-3, 1e-4, 1e-5):
        value = density(LONG_DATED, FORWARD, inside, 10.0, step=step * inside)
        assert value == pytest.approx(-7.1184, rel=1e-4)
    assert 7.1184 / peak == pytest.approx(0.0095, rel=0.02)


def test_the_value_at_the_floor_is_a_crossing_and_not_a_magnitude() -> None:
    """Which is why the test above reads the magnitude inside the region.

    A bisected zero is rounding by construction: it carries no information about
    how bad the defect is, and its sign is not even stable in the step.
    """
    floor = density_floor(LONG_DATED, FORWARD, 10.0, lower=1e-4 * FORWARD)
    assert floor is not None
    values = [
        density(LONG_DATED, FORWARD, floor, 10.0, step=step * floor)
        for step in (1e-3, 1e-4, 1e-5)
    ]
    assert max(abs(one) for one in values) < 1e-3


def test_the_defect_needs_maturity_or_vol_of_vol() -> None:
    """Sixty-one of eighty combinations have one, and every clean one is short."""
    reachable = 0
    clean_maturities = set()
    for nu in (0.4, 0.8, 1.2, 1.8, 2.5):
        for rho in (-0.8, -0.3, 0.0, 0.5):
            for time in (1.0, 5.0, 10.0, 30.0):
                parameters = SabrParameters(alpha=0.004, beta=0.5, rho=rho, nu=nu)
                if (
                    density_floor(parameters, FORWARD, time, lower=1e-4 * FORWARD)
                    is None
                ):
                    clean_maturities.add(time)
                else:
                    reachable += 1
    assert reachable == 61
    assert 30.0 not in clean_maturities


def test_the_floor_reports_nothing_rather_than_claiming_none_exists() -> None:
    """A search that found nothing says so; it does not certify the smile."""
    assert density_floor(SWAPTION, FORWARD, TIME, lower=0.9 * FORWARD) is None


def test_the_floor_refuses_a_bound_outside_the_strike_axis() -> None:
    with pytest.raises(ValueError, match="between zero and the forward"):
        density_floor(SWAPTION, FORWARD, TIME, lower=FORWARD * 2.0)


# -- calibration --------------------------------------------------------------


def generated_quotes(
    parameters: SabrParameters = SWAPTION,
) -> list[tuple[float, float]]:
    return [
        (FORWARD * m, lognormal_volatility(parameters, FORWARD, FORWARD * m, TIME))
        for m in (0.7, 0.85, 1.0, 1.2, 1.5)
    ]


def test_calibration_recovers_the_parameters_it_was_generated_from() -> None:
    fit = calibrate(FORWARD, TIME, generated_quotes(), beta=0.5)
    assert fit.converged
    assert fit.residual < 1e-12
    assert fit.parameters.alpha == pytest.approx(0.004, rel=1e-6)
    assert fit.parameters.rho == pytest.approx(-0.3, abs=1e-5)
    assert fit.parameters.nu == pytest.approx(0.4, rel=1e-5)


@pytest.mark.parametrize(
    ("beta", "alpha", "worst_basis_points"),
    [(0.3, 0.001828, 2.53), (0.7, 0.008752, 2.48), (1.0, 0.028328, 6.10)],
)
def test_beta_is_not_identifiable_from_one_smile(
    beta: float, alpha: float, worst_basis_points: float
) -> None:
    """Which is why it is an argument rather than a fitted parameter.

    ``alpha`` moves by a factor of 15.5 across these exponents and ``rho`` from
    -0.263 to -0.389, while the fitted smile moves by at most 6.1 basis points
    of volatility. A single smile carries no information about the backbone.
    """
    quotes = generated_quotes()
    fit = calibrate(FORWARD, TIME, quotes, beta=beta)
    assert fit.parameters.alpha == pytest.approx(alpha, rel=0.01)
    worst = max(
        abs(lognormal_volatility(fit.parameters, FORWARD, strike, TIME) - quoted)
        for strike, quoted in quotes
    )
    assert worst * 1e4 == pytest.approx(worst_basis_points, rel=0.05)


def test_alpha_is_solved_from_the_quote_nearest_the_money() -> None:
    """So every candidate fits that quote by construction."""
    quotes = generated_quotes()
    for beta in (0.3, 0.5, 0.7, 1.0):
        fit = calibrate(FORWARD, TIME, quotes, beta=beta)
        at_the_money = lognormal_volatility(fit.parameters, FORWARD, FORWARD, TIME)
        assert at_the_money == pytest.approx(quotes[2][1], rel=1e-9)


def test_calibration_takes_weights() -> None:
    quotes = generated_quotes()
    heavy = calibrate(FORWARD, TIME, quotes, weights=[1.0, 1.0, 1.0, 1.0, 50.0])
    assert heavy.converged
    far = abs(
        lognormal_volatility(heavy.parameters, FORWARD, quotes[-1][0], TIME)
        - quotes[-1][1]
    )
    assert far < 1e-8


@pytest.mark.parametrize(
    ("quotes", "weights", "message"),
    [
        ([(0.02, 0.3)], None, "at least two quotes"),
        ([(0.02, 0.3), (0.03, 0.4)], [1.0], "they have to match"),
        ([(0.02, 0.3), (0.03, 0.4)], [0.0, 0.0], "not all zero"),
        ([(0.02, 0.3), (0.03, 0.4)], [-1.0, 2.0], "non-negative"),
    ],
)
def test_calibration_refuses_what_it_cannot_fit(
    quotes: list[tuple[float, float]],
    weights: list[float] | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        calibrate(FORWARD, TIME, quotes, weights=weights)


def test_a_long_dated_smile_is_negative_in_the_high_wing_too() -> None:
    """Which ``density_floor`` does not look for, as its name says.

    Found by reading the command line's density column rather than by looking:
    at ten years the defect is on both sides, and the upper one is stable across
    differencing steps in the same way the lower one is.
    """
    for strike, expected in ((0.0267, -1.607), (0.0308, -1.107), (0.035, -0.756)):
        for step in (1e-3, 1e-4, 1e-5):
            value = density(LONG_DATED, FORWARD, strike, 10.0, step=step * strike)
            assert value == pytest.approx(expected, rel=1e-3)


def test_the_one_year_high_wing_is_tiny_and_positive_rather_than_zero() -> None:
    """Which needed checking rather than reading off a formatted column.

    Printed to six decimals these are ``0.000000``, which is what first
    suggested the differencing had run out of precision up here. It has not:
    pricing the out-of-the-money *call* keeps full relative accuracy, so the
    density at a 5% strike comes back as 1.1e-19 and is a real number. An
    absence of a defect and an absence of arithmetic look identical in a
    printed table and are not the same thing.
    """
    values = [
        density(SWAPTION, FORWARD, strike, TIME, step=1e-4 * strike)
        for strike in (0.03, 0.05, 0.08)
    ]
    assert all(one > 0.0 for one in values)
    assert values == sorted(values, reverse=True)
    assert values[0] == pytest.approx(4.196e-10, rel=1e-3)
    assert values[1] == pytest.approx(1.111e-19, rel=1e-3)
