"""The local-volatility solver, and the round trip that justifies it.

Two groups of tests matter more than the rest.

:class:`TestDupireRoundTrip` is the one the module exists for. It takes an
arbitrage-free surface, asks it for its own local volatility, prices a ladder of
calls on a grid under that volatility, inverts the prices back to implied
volatility, and compares them with the surface that produced them. Nothing in
that loop shares algebra with anything else in it: the Dupire identity
differentiates total variance in log-moneyness, the solver differences a partial
differential equation in log-spot, and the inversion is a root find on the Black
formula. Agreement is therefore evidence about the identity and not about any
one implementation of it.

:class:`TestCoordinate` is the test that a derivative of the surface cannot
replace. Gatheral's ``sigma_loc(k, T)`` is indexed by log-moneyness measured
from the *forward*, and reading it as measured from the spot differentiates the
same function, satisfies every self-consistency check, and prices wrong the
moment the carry is non-zero. The test holds the deliberate mistake alongside
the correct mapping and measures the gap.
"""

from __future__ import annotations

import math
from itertools import pairwise
from typing import ClassVar

import pytest

from moneyness.barrier import barrier_price
from moneyness.bsm import Inputs, OptionType, price
from moneyness.greeks import delta as bs_delta
from moneyness.greeks import gamma as bs_gamma
from moneyness.implied import Quote, implied_vol
from moneyness.lattice import Exercise, Lattice, richardson
from moneyness.monte_carlo import Barrier
from moneyness.pde import (
    DupireLocalVol,
    FrontStub,
    LocalVolatility,
    LocalVolError,
    Mesh,
    MeshPrice,
    _space_nodes,
    dupire_local_vol,
    price_pde,
)
from moneyness.surface import Surface
from moneyness.svi import SVI

# Four ordered, individually admissible slices with the shape of an equity
# surface: total variance rising in maturity, a negative skew that flattens.
SLICES = [
    (0.25, SVI(a=0.010, b=0.10, rho=-0.35, m=0.00, s=0.12)),
    (0.50, SVI(a=0.022, b=0.12, rho=-0.33, m=0.01, s=0.14)),
    (1.00, SVI(a=0.046, b=0.15, rho=-0.30, m=0.02, s=0.16)),
    (2.00, SVI(a=0.098, b=0.19, rho=-0.28, m=0.03, s=0.18)),
]
SURFACE = Surface(SLICES)
ONE_SLICE = Surface([(1.00, SVI(a=0.046, b=0.15, rho=-0.30, m=0.02, s=0.16))])

SPOT = 100.0
RATE = 0.03
CARRY = 0.01


def _flat(sigma: float) -> LocalVolatility:
    """A constant local volatility, as a function of spot and time."""

    def volatility(spot: float, time: float) -> float:
        return sigma

    return volatility


def _closed_form(
    strike: float,
    time: float,
    sigma: float,
    option: OptionType,
    *,
    rate: float = RATE,
    carry: float = CARRY,
) -> float:
    return price(Inputs(SPOT, strike, time, rate, sigma, carry=carry), option)


def _round_trip(
    surface: Surface,
    time: float,
    k: float,
    *,
    mesh: Mesh,
    front: FrontStub = FrontStub.LINEAR,
    align: bool = True,
    carry: float = CARRY,
) -> float:
    """Implied volatility recovered from a solved price, less the surface's own."""
    local = dupire_local_vol(surface, SPOT, carry=carry, front=front)
    forward = SPOT * math.exp(carry * time)
    strike = forward * math.exp(k)
    solved = price_pde(
        spot=SPOT,
        strike=strike,
        time=time,
        rate=RATE,
        vol=local,
        option=OptionType.CALL,
        carry=carry,
        mesh=mesh,
        breakpoints=local.breakpoints if align else (),
    )
    recovered = implied_vol(
        Quote(spot=SPOT, strike=strike, time=time, rate=RATE, price=solved.value, carry=carry),
        OptionType.CALL,
    )
    return recovered - surface.volatility(k, time)


class TestEuropeanAgainstClosedForm:
    """Constant volatility is the one case with an answer known to the last bit."""

    @pytest.mark.parametrize(
        ("strike", "time", "sigma", "option"),
        [
            (100.0, 1.0, 0.20, OptionType.CALL),
            (90.0, 1.0, 0.20, OptionType.PUT),
            (120.0, 0.5, 0.35, OptionType.CALL),
            (100.0, 2.0, 0.15, OptionType.PUT),
            (140.0, 0.25, 0.25, OptionType.CALL),
            (70.0, 3.0, 0.30, OptionType.PUT),
        ],
    )
    def test_matches_black_scholes(
        self, strike: float, time: float, sigma: float, option: OptionType
    ) -> None:
        solved = price_pde(
            spot=SPOT,
            strike=strike,
            time=time,
            rate=RATE,
            vol=_flat(sigma),
            option=option,
            carry=CARRY,
            mesh=Mesh(space_steps=400, time_steps=200),
        )
        exact = _closed_form(strike, time, sigma, option)
        assert solved.value == pytest.approx(exact, abs=5e-3)
        assert math.isfinite(solved.delta)
        assert math.isfinite(solved.gamma)
        assert solved.spot_range[0] < SPOT < solved.spot_range[1]
        assert abs(solved.nodes - 401) <= 2
        assert solved.early_exercise_premium == 0.0
        assert solved.boundary == ()

    def test_put_call_parity_on_the_solver_s_own_output(self) -> None:
        """Parity is an identity, so it holds on the discrete solution too.

        Both legs carry the same boundary and the same spacing, so most of the
        discretisation error is common to them and cancels. The tolerance is
        therefore far tighter than either leg's own error against the closed
        form, which is the point of checking it.
        """
        mesh = Mesh(space_steps=400, time_steps=200)
        for strike, time in ((100.0, 1.0), (85.0, 0.5), (115.0, 2.0)):
            call = price_pde(
                spot=SPOT,
                strike=strike,
                time=time,
                rate=RATE,
                vol=_flat(0.25),
                option=OptionType.CALL,
                carry=CARRY,
                mesh=mesh,
            ).value
            put = price_pde(
                spot=SPOT,
                strike=strike,
                time=time,
                rate=RATE,
                vol=_flat(0.25),
                option=OptionType.PUT,
                carry=CARRY,
                mesh=mesh,
            ).value
            forward = SPOT * math.exp(CARRY * time) - strike
            assert call - put == pytest.approx(math.exp(-RATE * time) * forward, abs=5e-5)

    def test_a_time_varying_volatility_prices_at_its_root_mean_square(self) -> None:
        """A volatility that depends only on time has a closed form too.

        The lognormal price depends on the volatility only through the
        integrated variance, so a piecewise-constant term structure must price
        exactly as a constant volatility equal to its root mean square. That
        exercises the time-dependent path through the operator against an
        answer that is still exact, which a surface cannot do.
        """

        def volatility(spot: float, time: float) -> float:
            return 0.15 if time < 0.4 else 0.35

        total = 0.15 * 0.15 * 0.4 + 0.35 * 0.35 * 0.6
        equivalent = math.sqrt(total / 1.0)
        solved = price_pde(
            spot=SPOT,
            strike=105.0,
            time=1.0,
            rate=RATE,
            vol=volatility,
            option=OptionType.CALL,
            carry=CARRY,
            mesh=Mesh(space_steps=600, time_steps=300),
            breakpoints=(0.4,),
        )
        assert solved.value == pytest.approx(
            _closed_form(105.0, 1.0, equivalent, OptionType.CALL), abs=3e-3
        )


class TestConvergence:
    """The rate, measured. Both directions are second order."""

    def test_second_order_in_the_spacing(self) -> None:
        exact = _closed_form(100.0, 1.0, 0.20, OptionType.CALL)
        errors = []
        for nodes in (100, 200, 400):
            solved = price_pde(
                spot=SPOT,
                strike=100.0,
                time=1.0,
                rate=RATE,
                vol=_flat(0.20),
                option=OptionType.CALL,
                carry=CARRY,
                mesh=Mesh(space_steps=nodes, time_steps=2000),
            )
            errors.append(abs(solved.value - exact))
        for coarse, fine in pairwise(errors):
            assert 3.5 < coarse / fine < 4.6

    def test_second_order_in_the_step(self) -> None:
        """Differenced, because the spatial error is a floor this cannot cross.

        At a fixed spacing the error tends to the spatial error rather than to
        zero, so the raw ratio of errors decays towards one and says nothing
        about the time scheme. Differencing successive refinements removes the
        constant and leaves the term being measured.
        """
        values = []
        for steps in (20, 40, 80, 160):
            solved = price_pde(
                spot=SPOT,
                strike=100.0,
                time=1.0,
                rate=RATE,
                vol=_flat(0.20),
                option=OptionType.CALL,
                carry=CARRY,
                mesh=Mesh(space_steps=800, time_steps=steps),
            )
            values.append(solved.value)
        gaps = [abs(b - a) for a, b in pairwise(values)]
        for coarse, fine in pairwise(gaps):
            assert 3.4 < coarse / fine < 4.6

    def test_the_strike_sits_on_a_node(self) -> None:
        """Which is what buys the second-order rate on a kinked payoff."""
        solved = price_pde(
            spot=SPOT,
            strike=103.7,
            time=1.0,
            rate=RATE,
            vol=_flat(0.2),
            option=OptionType.CALL,
            carry=CARRY,
            mesh=Mesh(space_steps=200, time_steps=100),
        )
        low, high = solved.spot_range
        step = (math.log(high) - math.log(low)) / (solved.nodes - 1)
        offset = (math.log(103.7) - math.log(low)) / step
        assert offset == pytest.approx(round(offset), abs=1e-9)

    def test_a_strike_beside_the_spot_does_not_blow_up_the_grid(self) -> None:
        """The regression that forced the anchor off the spot.

        Pinning both the spot and the strike to nodes means the spacing divides
        the gap between them, and a strike one percent away demands a hundredth
        of the domain per node. The node count then has nothing to do with what
        was asked for.
        """
        for strike in (100.0, 100.5, 100.05, 100.005):
            solved = price_pde(
                spot=SPOT,
                strike=strike,
                time=1.0,
                rate=RATE,
                vol=_flat(0.2),
                option=OptionType.CALL,
                carry=CARRY,
                mesh=Mesh(space_steps=200, time_steps=50),
            )
            assert solved.nodes < 260


class TestRannacher:
    """The implicit start earns its keep in the second derivative, not the price."""

    def test_crank_nicolson_alone_ruins_gamma(self) -> None:
        inputs = Inputs(SPOT, 100.0, 1.0, RATE, 0.20, carry=CARRY)
        truth = bs_gamma(inputs)

        def at(rannacher: int) -> MeshPrice:
            return price_pde(
                spot=SPOT,
                strike=100.0,
                time=1.0,
                rate=RATE,
                vol=_flat(0.20),
                option=OptionType.CALL,
                carry=CARRY,
                mesh=Mesh(space_steps=400, time_steps=40, rannacher=rannacher),
            )

        naive = at(0)
        started = at(2)
        # The documented measurement: wrong by more than a factor of two
        # without the implicit steps, and within a tenth of a percent with them.
        assert naive.gamma > 2.0 * truth
        assert abs(started.gamma / truth - 1.0) < 1e-2
        # And the price is barely affected either way, which is exactly why the
        # price is the wrong thing to judge the time scheme on.
        exact = _closed_form(100.0, 1.0, 0.20, OptionType.CALL)
        assert abs(naive.value - exact) < 5e-3
        assert abs(started.value - exact) < 5e-3

    def test_more_implicit_steps_cost_the_price(self) -> None:
        """Each is a first-order step, so they are not free."""
        exact = _closed_form(100.0, 1.0, 0.20, OptionType.CALL)
        errors = []
        for count in (2, 4, 8):
            solved = price_pde(
                spot=SPOT,
                strike=100.0,
                time=1.0,
                rate=RATE,
                vol=_flat(0.20),
                option=OptionType.CALL,
                carry=CARRY,
                mesh=Mesh(space_steps=400, time_steps=40, rannacher=count),
            )
            errors.append(abs(solved.value - exact))
        assert errors[0] < errors[1] < errors[2]


class TestTruncation:
    """What the domain width buys, measured with the spacing held fixed."""

    def test_truncation_is_spent_by_five_standard_deviations(self) -> None:
        """Widening past five moves nothing a double can resolve.

        The node count is grown with the width so that the spacing is
        identical in every case; otherwise this measures the spacing and not
        the boundary. The carry is zero for the same reason, since the domain
        carries a drift allowance that does not scale with the width.
        """
        prices = {}
        for width in (2.0, 3.0, 4.0, 5.0, 6.0, 8.0):
            nodes = round(400 * width / 2.0)
            prices[width] = price_pde(
                spot=SPOT,
                strike=100.0,
                time=1.0,
                rate=0.0,
                vol=_flat(0.20),
                option=OptionType.CALL,
                carry=0.0,
                mesh=Mesh(space_steps=nodes, time_steps=600, width=width),
            ).value
        assert abs(prices[3.0] - prices[2.0]) > 1e-5
        assert abs(prices[4.0] - prices[3.0]) < 1e-7
        assert abs(prices[6.0] - prices[5.0]) < 1e-9
        assert abs(prices[8.0] - prices[6.0]) < 1e-9

    def test_a_wider_domain_at_a_fixed_node_count_is_worse(self) -> None:
        """The trade the default has to be chosen against.

        Truncation falls with the width and the spacing error rises with it,
        and past the point where truncation is resolved only the second term is
        still moving. Widening without adding nodes therefore makes the answer
        monotonically worse, which is the opposite of what a safety margin is
        supposed to do.
        """
        exact = _closed_form(100.0, 1.0, 0.20, OptionType.CALL, rate=0.0, carry=0.0)
        errors = []
        for width in (4.0, 6.0, 8.0, 12.0):
            solved = price_pde(
                spot=SPOT,
                strike=100.0,
                time=1.0,
                rate=0.0,
                vol=_flat(0.20),
                option=OptionType.CALL,
                carry=0.0,
                mesh=Mesh(space_steps=400, time_steps=400, width=width),
            )
            errors.append(abs(solved.value - exact))
        assert errors == sorted(errors)

    def test_reference_vol_overrides_the_probe(self) -> None:
        narrow = price_pde(
            spot=SPOT,
            strike=100.0,
            time=1.0,
            rate=RATE,
            vol=_flat(0.20),
            option=OptionType.CALL,
            carry=CARRY,
            mesh=Mesh(space_steps=200, time_steps=100, reference_vol=0.10),
        )
        wide = price_pde(
            spot=SPOT,
            strike=100.0,
            time=1.0,
            rate=RATE,
            vol=_flat(0.20),
            option=OptionType.CALL,
            carry=CARRY,
            mesh=Mesh(space_steps=200, time_steps=100, reference_vol=0.40),
        )
        assert wide.spot_range[1] > narrow.spot_range[1]
        assert wide.spot_range[0] < narrow.spot_range[0]


class TestGreeks:
    """Delta and gamma off the grid, against the closed forms."""

    @pytest.mark.parametrize("strike", [85.0, 100.0, 118.0])
    def test_against_the_closed_forms(self, strike: float) -> None:
        inputs = Inputs(SPOT, strike, 1.0, RATE, 0.25, carry=CARRY)
        solved = price_pde(
            spot=SPOT,
            strike=strike,
            time=1.0,
            rate=RATE,
            vol=_flat(0.25),
            option=OptionType.CALL,
            carry=CARRY,
            mesh=Mesh(space_steps=600, time_steps=300),
        )
        assert solved.delta == pytest.approx(bs_delta(inputs, OptionType.CALL), abs=2e-4)
        assert solved.gamma == pytest.approx(bs_gamma(inputs), abs=2e-4)


class TestDupireRoundTrip:
    """The centrepiece: the surface's own local volatility reprices the surface."""

    @pytest.mark.parametrize("time", [0.25, 0.5, 1.0, 2.0])
    @pytest.mark.parametrize("k", [-0.3, -0.1, 0.0, 0.1, 0.3])
    def test_the_surface_is_recovered(self, time: float, k: float) -> None:
        gap = _round_trip(SURFACE, time, k, mesh=Mesh(space_steps=400, time_steps=200))
        assert abs(gap) < 1.5e-3

    def test_second_order_in_the_mesh_through_a_single_slice(self) -> None:
        """Where the local volatility is smooth in time, the rate is clean.

        A one-slice surface has no interior jump in ``dw/dT``, so the only
        structure in time is the front stub, across which the numerator is
        constant. This is the case that shows the round trip is exact in the
        limit rather than merely close.
        """
        errors = []
        for nodes in (200, 400, 800):
            mesh = Mesh(space_steps=nodes, time_steps=nodes // 2)
            errors.append(abs(_round_trip(ONE_SLICE, 1.0, 0.0, mesh=mesh)))
        for coarse, fine in pairwise(errors):
            assert 3.3 < coarse / fine < 4.7
        assert errors[-1] < 2e-5

    def test_second_order_in_the_mesh_through_four_slices(self) -> None:
        """And where it is not smooth, provided the jumps are on step boundaries.

        This is the regression for reading the coefficient at the step
        midpoint. Reading it at the two endpoints and averaging gave ratios of
        1.6 to 1.9 here instead of four.
        """
        errors = []
        for nodes in (400, 800, 1600):
            mesh = Mesh(space_steps=nodes, time_steps=nodes // 2)
            errors.append(abs(_round_trip(SURFACE, 1.0, 0.0, mesh=mesh)))
        for coarse, fine in pairwise(errors):
            assert 3.3 < coarse / fine < 4.7

    def test_a_flat_surface_recovers_its_own_volatility(self) -> None:
        """The one case where implied and local volatility coincide exactly.

        Total variance ``sigma^2 T`` with no skew makes ``g`` identically one
        and ``dw/dT`` identically ``sigma^2``, so the local volatility is
        ``sigma`` everywhere and the recovered implied volatility must be too.
        """
        flat = Surface(
            [(t, SVI(a=0.09 * t, b=0.0, rho=0.0, m=0.0, s=1.0)) for t in (0.25, 1.0, 2.0)]
        )
        for time in (0.25, 1.0, 1.5):
            for k in (-0.2, 0.0, 0.2):
                gap = _round_trip(flat, time, k, mesh=Mesh(space_steps=400, time_steps=200))
                assert abs(gap) < 1e-3


class TestCoordinate:
    """Forward log-moneyness, not spot log-moneyness. No derivative test sees this."""

    def test_reading_the_moneyness_off_the_spot_tilts_the_smile(self) -> None:
        carry = 0.08
        correct = dupire_local_vol(SURFACE, SPOT, carry=carry)

        def mistaken(spot: float, time: float) -> float:
            """``k`` measured from the spot rather than the forward."""
            safe = max(time, 1e-12)
            return correct(spot * math.exp(carry * safe), safe)

        mesh = Mesh(space_steps=400, time_steps=200)
        for k, floor in ((-0.2, 1.0e-2), (0.2, 5.0e-3)):
            good = _round_trip(SURFACE, 1.0, k, mesh=mesh, carry=carry)
            forward = SPOT * math.exp(carry)
            strike = forward * math.exp(k)
            solved = price_pde(
                spot=SPOT,
                strike=strike,
                time=1.0,
                rate=RATE,
                vol=mistaken,
                option=OptionType.CALL,
                carry=carry,
                mesh=mesh,
                breakpoints=correct.breakpoints,
            )
            recovered = implied_vol(
                Quote(
                    spot=SPOT, strike=strike, time=1.0, rate=RATE, price=solved.value, carry=carry
                ),
                OptionType.CALL,
            )
            bad = recovered - SURFACE.volatility(k, 1.0)
            assert abs(good) < 5e-4
            assert abs(bad) > floor
            assert abs(bad) > 20.0 * abs(good)
        # And the two wings are wrong in opposite directions, which is what
        # makes it a tilt rather than a level error a recalibration would hide.
        left = _round_trip(SURFACE, 1.0, -0.2, mesh=mesh, carry=carry)
        assert left < 0.0


class TestFrontStub:
    """What the surface says literally below its first quote, and why it cannot stand."""

    def test_the_literal_reading_prices_the_front_slice_at_intrinsic(self) -> None:
        """Flat total variance below the first quote means no local variance at all.

        That is a calendar arbitrage at the origin rather than a quirk of the
        extrapolation: total variance has to vanish as maturity does, and a
        surface holding it at its first quoted level all the way down does not.
        Nothing can reproduce such a surface, and the solver handed it returns
        the discounted forward intrinsic.
        """
        local = dupire_local_vol(SURFACE, SPOT, carry=CARRY, front=FrontStub.FLAT)
        assert local.variance(SPOT, 0.1) == 0.0
        assert local.variance(SPOT, 0.25) == 0.0
        forward = SPOT * math.exp(CARRY * 0.25)
        solved = price_pde(
            spot=SPOT,
            strike=forward,
            time=0.25,
            rate=RATE,
            vol=local,
            option=OptionType.CALL,
            carry=CARRY,
            mesh=Mesh(space_steps=200, time_steps=100),
            breakpoints=local.breakpoints,
        )
        assert solved.value == pytest.approx(0.0, abs=1e-12)

    def test_and_the_damage_outlives_the_front_slice(self) -> None:
        """Variance missed before the first quote is never made up later."""
        mesh = Mesh(space_steps=300, time_steps=150)
        for time, floor in ((0.25, 0.25), (1.0, 0.04), (2.0, 0.02)):
            gap = _round_trip(SURFACE, time, 0.0, mesh=mesh, front=FrontStub.FLAT)
            assert gap < -floor
            assert abs(_round_trip(SURFACE, time, 0.0, mesh=mesh)) < 2e-3

    def test_the_ramp_is_the_default(self) -> None:
        local = dupire_local_vol(SURFACE, SPOT, carry=CARRY)
        assert local.front is FrontStub.LINEAR
        # What is constant across the stub is dw/dT, not the local variance:
        # Durrleman's denominator depends on the level of total variance, and
        # that level is ramping. The variance falls by about a quarter.
        early = local.variance(SPOT, 0.01)
        late = local.variance(SPOT, 0.25)
        assert 0.0 < late < early
        assert late / early == pytest.approx(0.73, abs=0.05)
        # At the money the denominator goes to one in the limit, so the local
        # variance approaches the front slice's own at-the-money implied
        # variance. That is an exact limit and worth asserting as one.
        first = SURFACE.maturities[0]
        limit = SURFACE.total_variance(0.0, first) / first
        for tiny in (1e-4, 1e-5, 1e-6):
            forward = SPOT * math.exp(CARRY * tiny)
            assert local.variance(forward, tiny) == pytest.approx(limit, rel=2e-4)


class TestBreakpoints:
    """Aligning the time grid to the jumps buys order, not a smaller error."""

    def test_aligned_refinement_is_orderly_and_unaligned_is_not(self) -> None:
        # 1.37 years: the jumps at 0.25, 0.5 and 1.0 fall between steps of any
        # uniform grid used here, which is the case the alignment is for.
        aligned = []
        loose = []
        for nodes in (200, 400, 800, 1600):
            mesh = Mesh(space_steps=nodes, time_steps=nodes // 2)
            aligned.append(abs(_round_trip(SURFACE, 1.37, 0.0, mesh=mesh)))
            loose.append(abs(_round_trip(SURFACE, 1.37, 0.0, mesh=mesh, align=False)))
        ordered = [coarse / fine for coarse, fine in pairwise(aligned)]
        assert all(3.3 < ratio < 4.7 for ratio in ordered)
        assert aligned == sorted(aligned, reverse=True)
        # The unaligned sequence can happen to be smaller, and here at the two
        # finest meshes it is. What it is not is orderly: which side of each
        # jump a step reads depends on where the steps fall, so the ratios
        # scatter instead of settling at four, and a sequence like that can
        # neither be extrapolated nor trusted at any single mesh.
        scattered = [coarse / fine for coarse, fine in pairwise(loose)]
        assert any(ratio < 2.5 or ratio > 6.0 for ratio in scattered)

    def test_breakpoints_outside_the_horizon_are_ignored(self) -> None:
        plain = price_pde(
            spot=SPOT,
            strike=100.0,
            time=0.5,
            rate=RATE,
            vol=_flat(0.2),
            option=OptionType.CALL,
            carry=CARRY,
            mesh=Mesh(space_steps=100, time_steps=50),
        )
        padded = price_pde(
            spot=SPOT,
            strike=100.0,
            time=0.5,
            rate=RATE,
            vol=_flat(0.2),
            option=OptionType.CALL,
            carry=CARRY,
            mesh=Mesh(space_steps=100, time_steps=50),
            breakpoints=(-1.0, 0.0, 0.5, 2.0, 7.0),
        )
        assert plain.value == padded.value
        assert plain.steps == padded.steps


class TestAmerican:
    """Against the extrapolated lattice already in the package."""

    @pytest.mark.parametrize(
        ("strike", "time", "rate", "carry", "sigma", "option"),
        [
            (100.0, 1.0, 0.05, 0.0, 0.20, OptionType.PUT),
            (110.0, 0.5, 0.08, 0.0, 0.30, OptionType.PUT),
            (95.0, 1.0, 0.03, -0.04, 0.25, OptionType.CALL),
        ],
    )
    def test_agrees_with_the_lattice(
        self,
        strike: float,
        time: float,
        rate: float,
        carry: float,
        sigma: float,
        option: OptionType,
    ) -> None:
        inputs = Inputs(SPOT, strike, time, rate, sigma, carry=carry)
        extrapolated = richardson(inputs, option, steps=600, lattice=Lattice.CRR)
        solved = price_pde(
            spot=SPOT,
            strike=strike,
            time=time,
            rate=rate,
            vol=_flat(sigma),
            option=option,
            carry=carry,
            exercise=Exercise.AMERICAN,
            mesh=Mesh(space_steps=800, time_steps=400),
        )
        assert solved.value == pytest.approx(extrapolated, rel=4e-4)
        assert solved.value > price(inputs, option) - 1e-9
        assert solved.early_exercise_premium > 0.0
        assert solved.early_exercise_premium == pytest.approx(
            extrapolated - price(inputs, option), rel=1e-2
        )

    def test_the_premium_is_differenced_on_one_mesh(self) -> None:
        """Which is what makes it smaller than either leg's own error."""
        inputs = Inputs(SPOT, 100.0, 1.0, 0.05, 0.20, carry=0.0)

        def at(exercise: Exercise) -> MeshPrice:
            return price_pde(
                spot=SPOT,
                strike=100.0,
                time=1.0,
                rate=0.05,
                vol=_flat(0.20),
                option=OptionType.PUT,
                carry=0.0,
                exercise=exercise,
                mesh=Mesh(space_steps=300, time_steps=150),
            )

        american = at(Exercise.AMERICAN)
        european = at(Exercise.EUROPEAN)
        assert american.early_exercise_premium == pytest.approx(
            american.value - european.value, abs=1e-12
        )
        truth = richardson(inputs, OptionType.PUT, steps=600) - price(inputs, OptionType.PUT)
        assert abs(american.early_exercise_premium - truth) < abs(
            european.value - price(inputs, OptionType.PUT)
        ) + 1e-3

    def test_the_exercise_boundary_rises_to_the_strike_for_a_put(self) -> None:
        solved = price_pde(
            spot=SPOT,
            strike=100.0,
            time=1.0,
            rate=0.08,
            vol=_flat(0.20),
            option=OptionType.PUT,
            carry=0.0,
            exercise=Exercise.AMERICAN,
            mesh=Mesh(space_steps=400, time_steps=200),
        )
        assert len(solved.boundary) == solved.steps
        times = [t for t, _ in solved.boundary]
        assert times == sorted(times)
        levels = [s for _, s in solved.boundary if not math.isnan(s)]
        assert len(levels) == len(solved.boundary)
        assert levels == sorted(levels)
        # The boundary leaves the strike like sqrt(tau log tau) near expiry,
        # so the last step before it is resolved slowly: 93.1, 94.2, 95.6 and
        # 96.6 at 200, 400, 800 and 1600 nodes. Close to the strike and below
        # it is all a grid can say here.
        assert 0.9 * 100.0 < levels[-1] < 100.0
        assert levels[0] < levels[-1]

    def test_a_call_with_no_carry_advantage_is_never_exercised_early(self) -> None:
        """Carry equal to the rate on a call: the premium must vanish."""
        solved = price_pde(
            spot=SPOT,
            strike=100.0,
            time=1.0,
            rate=0.05,
            vol=_flat(0.20),
            option=OptionType.CALL,
            exercise=Exercise.AMERICAN,
            mesh=Mesh(space_steps=300, time_steps=150),
        )
        assert solved.early_exercise_premium == pytest.approx(0.0, abs=1e-10)
        assert all(math.isnan(level) for _, level in solved.boundary)

    def test_under_a_local_volatility(self) -> None:
        """The two features together, which is the case neither covers alone."""
        local = dupire_local_vol(SURFACE, SPOT, carry=0.0, front=FrontStub.LINEAR)
        european = price_pde(
            spot=SPOT,
            strike=110.0,
            time=1.0,
            rate=0.06,
            vol=local,
            option=OptionType.PUT,
            carry=0.0,
            mesh=Mesh(space_steps=400, time_steps=200),
            breakpoints=local.breakpoints,
        )
        american = price_pde(
            spot=SPOT,
            strike=110.0,
            time=1.0,
            rate=0.06,
            vol=local,
            option=OptionType.PUT,
            carry=0.0,
            exercise=Exercise.AMERICAN,
            mesh=Mesh(space_steps=400, time_steps=200),
            breakpoints=local.breakpoints,
        )
        assert american.value > european.value
        assert american.value >= 10.0 - 1e-9
        assert american.early_exercise_premium > 0.0


class TestDegenerate:
    """Zero time, zero volatility, and the places the grid cannot go."""

    def test_zero_time_is_the_payoff(self) -> None:
        solved = price_pde(
            spot=SPOT,
            strike=90.0,
            time=0.0,
            rate=RATE,
            vol=_flat(0.2),
            option=OptionType.CALL,
            carry=CARRY,
        )
        assert solved.value == 10.0
        assert solved.delta == 1.0
        assert solved.gamma == 0.0
        assert solved.steps == 0
        assert solved.nodes == 0

    def test_zero_time_out_of_the_money(self) -> None:
        solved = price_pde(
            spot=SPOT,
            strike=110.0,
            time=0.0,
            rate=RATE,
            vol=_flat(0.2),
            option=OptionType.CALL,
            carry=CARRY,
        )
        assert solved.value == 0.0
        assert solved.delta == 0.0

    def test_zero_volatility_is_the_discounted_forward_intrinsic(self) -> None:
        """Central differencing a pure drift is unstable and pointless.

        With no diffusion the terminal value is certain, so the answer is
        arithmetic and the grid is not built at all. The alternative is a
        scheme that oscillates its way to a number it already knows.
        """
        solved = price_pde(
            spot=SPOT,
            strike=100.0,
            time=1.0,
            rate=0.05,
            vol=_flat(0.0),
            option=OptionType.CALL,
            carry=0.03,
        )
        forward = SPOT * math.exp(0.03)
        assert solved.value == pytest.approx(math.exp(-0.05) * (forward - 100.0))
        assert solved.nodes == 0
        assert math.isnan(solved.delta)

    def test_zero_volatility_american_takes_the_better_of_now_and_then(self) -> None:
        solved = price_pde(
            spot=SPOT,
            strike=130.0,
            time=1.0,
            rate=0.05,
            vol=_flat(0.0),
            option=OptionType.PUT,
            carry=0.0,
            exercise=Exercise.AMERICAN,
        )
        assert solved.value == pytest.approx(30.0)
        assert solved.early_exercise_premium > 0.0


class TestValidation:
    """What gets refused, and with what said about it."""

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("space_steps", 7),
            ("time_steps", 0),
            ("width", 0.0),
            ("width", -1.0),
            ("rannacher", -1),
            ("reference_vol", 0.0),
            ("reference_vol", -0.2),
        ],
    )
    def test_mesh_rejects_bad_fields(self, field: str, value: float) -> None:
        with pytest.raises(ValueError, match=field):
            Mesh(**{field: value})  # type: ignore[arg-type]

    def test_rannacher_cannot_exceed_the_step_count(self) -> None:
        with pytest.raises(ValueError, match="cannot exceed"):
            Mesh(time_steps=3, rannacher=4)

    @pytest.mark.parametrize(
        ("field", "value"),
        [("spot", 0.0), ("spot", -1.0), ("strike", 0.0), ("time", -0.5)],
    )
    def test_price_rejects_bad_inputs(self, field: str, value: float) -> None:
        kwargs: dict[str, object] = {
            "spot": SPOT,
            "strike": 100.0,
            "time": 1.0,
            "rate": RATE,
            "vol": _flat(0.2),
            "option": OptionType.CALL,
        }
        kwargs[field] = value
        with pytest.raises(ValueError, match=field):
            price_pde(**kwargs)  # type: ignore[arg-type]

    def test_non_finite_rate_and_carry_are_refused(self) -> None:
        with pytest.raises(ValueError, match="rate"):
            price_pde(
                spot=SPOT,
                strike=100.0,
                time=1.0,
                rate=math.inf,
                vol=_flat(0.2),
                option=OptionType.CALL,
            )
        with pytest.raises(ValueError, match="carry"):
            price_pde(
                spot=SPOT,
                strike=100.0,
                time=1.0,
                rate=RATE,
                vol=_flat(0.2),
                option=OptionType.CALL,
                carry=math.nan,
            )

    def test_a_volatility_that_returns_nonsense_is_caught(self) -> None:
        def broken(spot: float, time: float) -> float:
            return math.nan if time > 0.3 else 0.2

        with pytest.raises(LocalVolError, match="local volatility is nan"):
            price_pde(
                spot=SPOT,
                strike=100.0,
                time=1.0,
                rate=RATE,
                vol=broken,
                option=OptionType.CALL,
                mesh=Mesh(space_steps=50, time_steps=20, reference_vol=0.2),
            )


class TestDupireBridge:
    """The bridge's own refusals and bookkeeping."""

    def test_past_the_last_quote_is_refused(self) -> None:
        local = dupire_local_vol(SURFACE, SPOT, carry=CARRY)
        with pytest.raises(LocalVolError, match="past the last quoted maturity"):
            local.variance(SPOT, 2.5)

    def test_the_last_quote_itself_is_priceable(self) -> None:
        """Left-continuity in time is what makes it so.

        The surface's own ``dw/dT`` is zero at and beyond its last maturity,
        so a bridge reading the slope to the right would make the last quoted
        expiry the one expiry it cannot price.
        """
        local = dupire_local_vol(SURFACE, SPOT, carry=CARRY)
        assert local.variance(SPOT, 2.0) > 0.0
        assert SURFACE.dw_dt(0.0, 2.0) == 0.0

    def test_bad_arguments(self) -> None:
        local = dupire_local_vol(SURFACE, SPOT, carry=CARRY)
        with pytest.raises(LocalVolError, match="spot"):
            local.variance(0.0, 1.0)
        with pytest.raises(LocalVolError, match="time"):
            local.variance(SPOT, -0.1)
        with pytest.raises(ValueError, match="spot"):
            dupire_local_vol(SURFACE, 0.0)
        with pytest.raises(ValueError, match="carry"):
            dupire_local_vol(SURFACE, SPOT, carry=math.inf)

    def test_the_volatility_is_the_root_of_the_variance(self) -> None:
        local = dupire_local_vol(SURFACE, SPOT, carry=CARRY)
        for spot in (70.0, 100.0, 140.0):
            assert local(spot, 0.8) == pytest.approx(math.sqrt(local.variance(spot, 0.8)))

    def test_breakpoints_are_the_quoted_maturities(self) -> None:
        local = dupire_local_vol(SURFACE, SPOT, carry=CARRY)
        assert local.breakpoints == SURFACE.maturities

    def test_a_butterfly_defect_is_named_as_one(self) -> None:
        """The two failure modes have different remedies, so they are reported apart."""
        bad = Surface([(1.0, SVI(a=0.04, b=0.9, rho=-0.95, m=0.0, s=0.05))])
        local = dupire_local_vol(bad, SPOT, carry=0.0)
        worst = min(g for _, g, _ in bad.butterfly(maturities=[1.0]))
        assert worst < 0.0
        found = False
        for offset in (-0.4, -0.2, 0.2, 0.4):
            try:
                local.variance(SPOT * math.exp(offset), 1.0)
            except LocalVolError as error:
                if "negative density" in str(error):
                    found = True
        assert found

    def test_a_calendar_crossing_is_named_as_one(self) -> None:
        crossing = Surface(
            [
                (1.0, SVI(a=0.09, b=0.0, rho=0.0, m=0.0, s=1.0)),
                (2.0, SVI(a=0.04, b=0.0, rho=0.0, m=0.0, s=1.0)),
            ]
        )
        local = dupire_local_vol(crossing, SPOT, carry=0.0)
        with pytest.raises(LocalVolError, match="slices cross"):
            local.variance(SPOT, 1.5)

    def test_the_bridge_is_frozen(self) -> None:
        local = dupire_local_vol(SURFACE, SPOT, carry=CARRY)
        assert isinstance(local, DupireLocalVol)
        with pytest.raises((AttributeError, TypeError)):
            local.spot = 1.0  # type: ignore[misc]


class TestBarrier:
    """A knock-out in the solver, against the closed form in :mod:`moneyness.barrier`.

    This is the independent check on both files. A backward march with a
    Dirichlet condition at the barrier and a reflection-principle formula share
    no derivation, so agreement between them is evidence about each, in a way
    that the closed form's own parity identity is not — parity holds
    algebraically for the table whether or not the table is right.

    The agreement is asserted as a convergence *order* rather than as a
    tolerance. Second order is what the scheme supports, so the two should
    differ by whatever the mesh costs and no more, and a ratio near four under
    doubling says that is what is happening. A tolerance alone would pass just
    as well if the solver were converging to a slightly wrong number.
    """

    CONTRACTS: ClassVar[list[tuple[Barrier, float, float, OptionType]]] = [
        (Barrier.DOWN_AND_OUT, 85.0, 100.0, OptionType.CALL),
        (Barrier.DOWN_AND_OUT, 95.0, 100.0, OptionType.CALL),
        (Barrier.UP_AND_OUT, 120.0, 100.0, OptionType.CALL),
        (Barrier.UP_AND_OUT, 120.0, 110.0, OptionType.CALL),
        (Barrier.DOWN_AND_OUT, 85.0, 100.0, OptionType.PUT),
        (Barrier.UP_AND_OUT, 115.0, 100.0, OptionType.PUT),
        (Barrier.UP_AND_OUT, 105.0, 90.0, OptionType.PUT),
    ]

    @staticmethod
    def _solve(
        style: Barrier,
        level: float,
        strike: float,
        option: OptionType,
        space: int,
        steps: int,
    ) -> MeshPrice:
        return price_pde(
            spot=SPOT,
            strike=strike,
            time=1.0,
            rate=0.05,
            vol=_flat(0.20),
            option=option,
            carry=0.02,
            mesh=Mesh(space_steps=space, time_steps=steps),
            barrier_level=level,
            barrier_style=style,
        )

    @staticmethod
    def _exact(
        style: Barrier, level: float, strike: float, option: OptionType
    ) -> float:
        return barrier_price(
            Inputs(spot=SPOT, strike=strike, time=1.0, rate=0.05, vol=0.20, carry=0.02),
            option,
            level,
            style,
        )

    @pytest.mark.parametrize(("style", "level", "strike", "option"), CONTRACTS)
    def test_the_solver_converges_on_the_closed_form_at_second_order(
        self, style: Barrier, level: float, strike: float, option: OptionType
    ) -> None:
        exact = self._exact(style, level, strike, option)
        errors = [
            self._solve(style, level, strike, option, space, steps).value - exact
            for space, steps in ((200, 100), (400, 200), (800, 400), (1600, 800))
        ]
        # Same sign throughout: an error that changes sign is not converging in
        # the sense a ratio describes, and this one does not.
        assert all(one < 0.0 for one in errors)
        ratios = [abs(errors[i] / errors[i + 1]) for i in range(3)]
        for ratio in ratios:
            assert 3.4 < ratio < 4.7
        assert abs(errors[-1]) < 1e-4

    @pytest.mark.parametrize(("style", "level", "strike", "option"), CONTRACTS)
    def test_the_barrier_lands_on_a_node(
        self, style: Barrier, level: float, strike: float, option: OptionType
    ) -> None:
        """And the knocked-out edge of the domain is the barrier itself."""
        solved = self._solve(style, level, strike, option, 800, 400)
        assert solved.strike_on_node
        edge = solved.spot_range[0] if style.is_down else solved.spot_range[1]
        assert edge == pytest.approx(level, rel=1e-13)

    def test_displacing_the_barrier_by_half_a_node_dominates_the_mesh_error(
        self,
    ) -> None:
        """The measurement that decides the node layout.

        A barrier between nodes is a barrier moved, which is first order in the
        spacing, against the scheme's second-order error. So the displacement
        does not merely add to the error, it takes it over — and by more as the
        mesh refines, which is the signature of the lower order winning.
        """
        exact = self._exact(Barrier.UP_AND_OUT, 120.0, 100.0, OptionType.CALL)
        span = 6.0 * 0.20 + 0.02
        factors = []
        for space in (200, 400, 800, 1600):
            spacing = 2.0 * span / space
            on_node = self._solve(
                Barrier.UP_AND_OUT, 120.0, 100.0, OptionType.CALL, space, space // 2
            ).value
            displaced = price_pde(
                spot=SPOT,
                strike=100.0,
                time=1.0,
                rate=0.05,
                vol=_flat(0.20),
                option=OptionType.CALL,
                carry=0.02,
                mesh=Mesh(space_steps=space, time_steps=space // 2),
                barrier_level=120.0 * math.exp(0.5 * spacing),
                barrier_style=Barrier.UP_AND_OUT,
            ).value
            factors.append(abs((displaced - on_node) / (on_node - exact)))
        assert factors == pytest.approx([35.0, 69.0, 137.0, 273.0], rel=0.2)
        assert factors == sorted(factors)

    def test_a_barrier_too_close_to_the_strike_gives_up_the_kink(self) -> None:
        """And says so, rather than building the mesh the spacing would demand.

        The node count stays within twice the request and the solver still
        converges — the kink's second-order cost is real but it is the smaller
        of the two, which is the whole reason the barrier gets the node.
        """
        exact = barrier_price(
            Inputs(spot=SPOT, strike=85.0, time=1.0, rate=0.05, vol=0.20, carry=0.02),
            OptionType.CALL,
            84.95,
            Barrier.DOWN_AND_OUT,
        )
        errors = []
        for space in (400, 800):
            solved = price_pde(
                spot=SPOT,
                strike=85.0,
                time=1.0,
                rate=0.05,
                vol=_flat(0.20),
                option=OptionType.CALL,
                carry=0.02,
                mesh=Mesh(space_steps=space, time_steps=space // 2),
                barrier_level=84.95,
                barrier_style=Barrier.DOWN_AND_OUT,
            )
            assert not solved.strike_on_node
            assert solved.nodes <= 2 * space + 4
            # The barrier is still exact; it is the strike that was given up.
            assert solved.spot_range[0] == pytest.approx(84.95, rel=1e-13)
            errors.append(abs(solved.value - exact))
        assert errors[0] / errors[1] > 2.0

    def test_a_barrier_a_few_basis_points_from_the_strike_does_not_explode(
        self,
    ) -> None:
        """The guard that had to be tightened before it did anything.

        At eight times the requested node count this built 2442 nodes for a
        request of 400 — the same collapse the strike anchoring exists to
        avoid, under a cap meant to prevent it.
        """
        solved = price_pde(
            spot=SPOT,
            strike=100.0,
            time=1.0,
            rate=0.05,
            vol=_flat(0.20),
            option=OptionType.CALL,
            carry=0.02,
            mesh=Mesh(space_steps=400, time_steps=200),
            barrier_level=100.05,
            barrier_style=Barrier.UP_AND_OUT,
        )
        assert not solved.strike_on_node
        assert solved.nodes <= 804

    def test_no_barrier_keeps_the_strike_on_a_node(self) -> None:
        """The existing layout is untouched when there is no barrier to place."""
        solved = price_pde(
            spot=SPOT,
            strike=101.0,
            time=1.0,
            rate=RATE,
            vol=_flat(0.20),
            option=OptionType.CALL,
        )
        assert solved.strike_on_node
        assert solved.spot_range[0] < 101.0 < solved.spot_range[1]

    @pytest.mark.parametrize("style", [Barrier.DOWN_AND_OUT, Barrier.UP_AND_OUT])
    def test_a_touched_barrier_reports_zero_without_a_grid(
        self, style: Barrier
    ) -> None:
        """nan derivatives, because the contract has ceased to exist.

        A delta of zero would say the value is flat in the spot, which is a
        different and false statement.
        """
        solved = price_pde(
            spot=100.0,
            strike=95.0 if not style.is_down else 105.0,
            time=1.0,
            rate=RATE,
            vol=_flat(0.20),
            option=OptionType.CALL if not style.is_down else OptionType.PUT,
            barrier_level=100.0,
            barrier_style=style,
        )
        assert solved.value == 0.0
        assert solved.nodes == 0
        assert math.isnan(solved.delta)
        assert math.isnan(solved.gamma)

    def test_a_structurally_worthless_barrier_reports_zero(self) -> None:
        solved = price_pde(
            spot=100.0,
            strike=125.0,
            time=1.0,
            rate=RATE,
            vol=_flat(0.20),
            option=OptionType.CALL,
            barrier_level=120.0,
            barrier_style=Barrier.UP_AND_OUT,
        )
        assert solved.value == 0.0
        assert solved.nodes == 0

    def test_a_knock_in_is_refused_rather_than_differenced(self) -> None:
        """It is the vanilla less the knock-out, and the closed form says so exactly.

        Computing it here would mean subtracting two valuations on different
        domains, which is the one subtraction this module avoids everywhere
        else — the early-exercise premium is differenced on a single mesh for
        the same reason.
        """
        with pytest.raises(ValueError, match="takes a knock-out"):
            price_pde(
                spot=SPOT,
                strike=100.0,
                time=1.0,
                rate=RATE,
                vol=_flat(0.20),
                option=OptionType.CALL,
                barrier_level=120.0,
                barrier_style=Barrier.UP_AND_IN,
            )

    def test_an_american_knock_out_is_worth_more_than_the_european(self) -> None:
        """And the early-exercise premium is still differenced on one mesh.

        The knocked-out node must not be resurrected by the payoff floor, which
        is what the interior-only sweep is for: exercising at the barrier is
        exercising a contract that no longer exists.
        """
        def solve(exercise: Exercise) -> MeshPrice:
            return price_pde(
                spot=SPOT,
                strike=105.0,
                time=1.0,
                rate=0.05,
                vol=_flat(0.20),
                option=OptionType.PUT,
                carry=0.02,
                mesh=Mesh(space_steps=400, time_steps=200),
                barrier_level=130.0,
                barrier_style=Barrier.UP_AND_OUT,
                exercise=exercise,
            )

        european = solve(Exercise.EUROPEAN)
        american = solve(Exercise.AMERICAN)
        assert american.value > european.value
        assert american.early_exercise_premium > 0.0
        assert american.early_exercise_premium == pytest.approx(
            american.value - european.value, rel=1e-9
        )
        # The barrier node never exercises: it is worth nothing, not intrinsic.
        assert american.spot_range[1] == pytest.approx(130.0, rel=1e-13)

    def test_a_distant_barrier_recovers_the_unbarriered_solve(self) -> None:
        """Both on the same scheme, so this is the solver against itself.

        Worth having separately from the closed-form comparison: it isolates
        the barrier plumbing from the formula, so a mistake in one does not
        look like agreement with the other.
        """
        plain = price_pde(
            spot=SPOT,
            strike=100.0,
            time=1.0,
            rate=0.05,
            vol=_flat(0.20),
            option=OptionType.CALL,
            carry=0.02,
            mesh=Mesh(space_steps=800, time_steps=400),
        )
        far = price_pde(
            spot=SPOT,
            strike=100.0,
            time=1.0,
            rate=0.05,
            vol=_flat(0.20),
            option=OptionType.CALL,
            carry=0.02,
            mesh=Mesh(space_steps=800, time_steps=400),
            barrier_level=1000.0,
            barrier_style=Barrier.UP_AND_OUT,
        )
        assert far.value == pytest.approx(plain.value, rel=2e-4)

    @pytest.mark.parametrize("width", [1.0, 3.0, 6.0, 12.0])
    @pytest.mark.parametrize("space", [8, 50, 200, 800])
    def test_the_domain_always_contains_the_spot(
        self, width: float, space: int
    ) -> None:
        """The property the -1183.8 came from violating.

        Sizing the barriered domain from the barrier rather than from the spot
        left a barrier at ten times the spot outside a grid that did not
        contain the spot, and the quadratic read extrapolated off the nearest
        three nodes to -1183.8 for a call worth 8.65. A wrong answer with a
        plausible shape is the worst outcome available here, so the containment
        is swept rather than spot-checked: both sides, barriers from 1% away to
        ten times the spot, four widths and four node counts, with the barrier
        asserted to be the edge in every one.
        """
        for style, levels in (
            (Barrier.DOWN_AND_OUT, (99.0, 85.0, 50.0, 10.0)),
            (Barrier.UP_AND_OUT, (101.0, 120.0, 200.0, 1000.0)),
        ):
            for level in levels:
                for strike in (85.0, 100.0, 115.0):
                    nodes, _ = _space_nodes(
                        SPOT,
                        strike,
                        1.0,
                        0.02,
                        0.20,
                        Mesh(space_steps=space, time_steps=10, width=width),
                        (level, style),
                    )
                    assert nodes[0] < math.log(SPOT) < nodes[-1]
                    edge = nodes[0] if style.is_down else nodes[-1]
                    assert edge == pytest.approx(math.log(level), rel=1e-13)
                    assert nodes == sorted(nodes)

    def test_a_barriered_solve_under_a_real_local_volatility_runs(self) -> None:
        """The barrier is orthogonal to the coefficient, so it has to compose.

        No closed form to check against here — that is the point. What is
        asserted is that it is bounded by the unbarriered solve on the same
        mesh and strictly below it, which a barrier that was being ignored
        would fail.
        """
        local = dupire_local_vol(SURFACE, SPOT, carry=CARRY)

        def solve(level: float | None, style: Barrier | None) -> MeshPrice:
            return price_pde(
                spot=SPOT,
                strike=100.0,
                time=1.0,
                rate=RATE,
                vol=local,
                option=OptionType.CALL,
                carry=CARRY,
                mesh=Mesh(space_steps=400, time_steps=200),
                breakpoints=local.breakpoints,
                barrier_level=level,
                barrier_style=style,
            )

        plain = solve(None, None)
        knocked = solve(130.0, Barrier.UP_AND_OUT)
        assert 0.0 < knocked.value < plain.value
