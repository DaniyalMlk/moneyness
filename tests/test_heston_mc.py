"""Simulating the variance process, and what the simulation says about the transform."""

from __future__ import annotations

import math
import random

import pytest

from moneyness.bsm import OptionType
from moneyness.heston import Contract, Heston
from moneyness.heston import price as transform_price
from moneyness.heston_mc import (
    MartingaleCorrectionError,
    Scheme,
    _coefficients,
    _fit,
    _replicate,
    asian,
    barrier,
    conditional_mean,
    conditional_variance,
    european,
    paths,
    variance_path,
)
from moneyness.monte_carlo import Barrier, Settings, _summarise
from moneyness.normal import norm_ppf

REFERENCE = Heston(v0=0.04, kappa=1.5768, theta=0.04, sigma=0.5751, rho=-0.5711)

# The Feller condition holds comfortably here, so the variance never
# approaches zero and the quadratic proposal is used throughout. Kept
# alongside the reference set, where it fails, so both branches are exercised.
FELLER = Heston(v0=0.04, kappa=6.0, theta=0.04, sigma=0.4, rho=-0.3)

# Deliberately extreme: a low reversion level with a large volatility of
# variance drives the ratio past the switching level and selects the
# exponential proposal, which is the branch with an atom at zero.
SPIKY = Heston(v0=0.002, kappa=0.5, theta=0.002, sigma=0.9, rho=-0.8)


def uniforms(count: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    return [rng.random() for _ in range(count)]


class TestConditionalMoments:
    """The two closed forms the whole scheme is fitted to."""

    @pytest.mark.parametrize("model", [REFERENCE, FELLER, SPIKY])
    @pytest.mark.parametrize("step", [0.001, 0.05, 1.0, 20.0])
    def test_the_mean_decays_towards_the_long_run_level(
        self, model: Heston, step: float
    ) -> None:
        got = conditional_mean(model, 0.09, step)
        want = model.theta + (0.09 - model.theta) * math.exp(-model.kappa * step)
        assert got == pytest.approx(want, rel=1e-14)

    def test_a_zero_step_leaves_both_moments_where_they_were(self) -> None:
        assert conditional_mean(REFERENCE, 0.07, 0.0) == pytest.approx(0.07, abs=1e-15)
        assert conditional_variance(REFERENCE, 0.07, 0.0) == pytest.approx(0.0, abs=1e-18)

    @pytest.mark.parametrize("model", [REFERENCE, FELLER, SPIKY])
    def test_the_variance_tends_to_the_stationary_one(self, model: Heston) -> None:
        # Far out in time the conditional variance forgets where it started
        # and settles at sigma^2 theta / (2 kappa).
        got = conditional_variance(model, 0.09, 500.0)
        want = model.sigma**2 * model.theta / (2.0 * model.kappa)
        assert got == pytest.approx(want, rel=1e-9)

    def test_a_zero_variance_still_has_a_positive_conditional_variance(self) -> None:
        # The process can leave zero, which is exactly what an Euler step that
        # absorbs at zero gets wrong.
        assert conditional_variance(REFERENCE, 0.0, 0.05) > 0.0

    def test_euler_cannot_leave_the_origin_and_that_is_the_whole_problem(self) -> None:
        # Started at exactly zero, a full-truncation Euler step has no
        # diffusion at all: the volatility of the increment is sqrt(V), and V
        # is zero. The path has to be lifted off the origin by the drift alone
        # before any randomness enters, so the simulated variance comes back
        # far below the true one. This is the degeneracy the scheme in this
        # module exists to avoid, and it is worst exactly where a fitted model
        # spends its time -- a failed Feller condition means the process
        # visits zero.
        model, step, inner, draws = SPIKY, 0.02, 400, 4000
        rng = random.Random(4242)
        inner_step = step / inner
        finals = []
        for _ in range(draws):
            level = 0.0
            for _ in range(inner):
                floor = max(level, 0.0)
                level += model.kappa * (model.theta - floor) * inner_step + (
                    model.sigma * math.sqrt(floor * inner_step) * norm_ppf(rng.random())
                )
            finals.append(max(level, 0.0))
        mean = math.fsum(finals) / draws
        spread = math.fsum((x - mean) ** 2 for x in finals) / (draws - 1)
        truth = conditional_variance(model, 0.0, step)
        assert spread < 0.75 * truth

        # The scheme here has no such trouble: its proposal is fitted to the
        # conditional moments whatever the starting point, including zero.
        proposal = _fit(model, 0.0, step)
        n = 200_000
        values = [proposal.draw((i + 0.5) / n) for i in range(n)]
        fitted_mean = math.fsum(values) / n
        fitted = math.fsum(v * v for v in values) / n - fitted_mean * fitted_mean
        assert fitted == pytest.approx(truth, rel=3e-2)

    @pytest.mark.parametrize("bad", [-1e-9, -1.0])
    def test_a_negative_variance_is_refused(self, bad: float) -> None:
        with pytest.raises(ValueError, match="variance must be non-negative"):
            conditional_mean(REFERENCE, bad, 1.0)
        with pytest.raises(ValueError, match="variance must be non-negative"):
            conditional_variance(REFERENCE, bad, 1.0)

    def test_a_negative_step_is_refused(self) -> None:
        with pytest.raises(ValueError, match="step must be non-negative"):
            conditional_variance(REFERENCE, 0.04, -1.0)

    @pytest.mark.parametrize("model", [REFERENCE, FELLER, SPIKY])
    @pytest.mark.parametrize("variance", [0.001, 0.04, 0.5])
    def test_the_moments_agree_with_a_direct_simulation_of_one_step(
        self, model: Heston, variance: float
    ) -> None:
        # An Euler simulation of one *very* short step, fine enough that its
        # own error is below the tolerance. This is the closed forms checked
        # against the process they describe, rather than against themselves.
        # Started away from zero, for the reason the next test gives.
        step, inner, draws = 0.02, 400, 4000
        rng = random.Random(4242)
        inner_step = step / inner
        finals = []
        for _ in range(draws):
            level = variance
            for _ in range(inner):
                shock = norm_ppf(rng.random())
                floor = max(level, 0.0)
                level += model.kappa * (model.theta - floor) * inner_step + (
                    model.sigma * math.sqrt(floor * inner_step) * shock
                )
            finals.append(max(level, 0.0))
        mean = math.fsum(finals) / draws
        spread = math.fsum((x - mean) ** 2 for x in finals) / (draws - 1)
        tolerance = 4.0 * math.sqrt(spread / draws)
        assert mean == pytest.approx(conditional_mean(model, variance, step), abs=tolerance)
        assert spread == pytest.approx(
            conditional_variance(model, variance, step), rel=0.25
        )


class TestTheProposals:
    def test_both_branches_are_reachable(self) -> None:
        assert _fit(REFERENCE, 0.04, 0.05).kind is Scheme.QUADRATIC
        assert _fit(SPIKY, 0.0, 1.0).kind is Scheme.EXPONENTIAL

    @pytest.mark.parametrize("model", [REFERENCE, FELLER, SPIKY])
    @pytest.mark.parametrize("variance", [0.0, 0.002, 0.04, 0.3])
    @pytest.mark.parametrize("step", [0.01, 0.25, 2.0])
    def test_each_proposal_is_exact_on_the_two_moments_it_was_fitted_to(
        self, model: Heston, variance: float, step: float
    ) -> None:
        # The defining property. Integrated over the uniform by a fine
        # midpoint rule rather than sampled, so there is no simulation error
        # in the check at all.
        proposal = _fit(model, variance, step)
        n = 200_000
        values = [proposal.draw((i + 0.5) / n) for i in range(n)]
        mean = math.fsum(values) / n
        second = math.fsum(v * v for v in values) / n
        assert mean == pytest.approx(conditional_mean(model, variance, step), rel=2e-3)
        assert second - mean * mean == pytest.approx(
            conditional_variance(model, variance, step), rel=3e-2
        )

    @pytest.mark.parametrize("model", [REFERENCE, FELLER, SPIKY])
    def test_no_draw_is_ever_negative(self, model: Heston) -> None:
        for variance in (0.0, 0.001, 0.04, 0.5):
            proposal = _fit(model, variance, 0.1)
            assert all(proposal.draw(u / 1000.0) >= 0.0 for u in range(1000))

    def test_the_exponential_branch_has_an_atom_at_zero(self) -> None:
        proposal = _fit(SPIKY, 0.0, 1.0)
        assert proposal.kind is Scheme.EXPONENTIAL
        assert proposal.p > 0.0
        assert proposal.draw(proposal.p * 0.5) == 0.0
        assert proposal.draw(min(proposal.p + 0.1, 0.999)) > 0.0

    def test_a_process_pinned_at_zero_stays_there(self) -> None:
        pinned = Heston(0.0, 1.5, 0.0, 0.5, -0.5)
        assert variance_path(pinned, 1.0, 10, uniforms(10, 1)) == [0.0] * 10


class TestVariancePaths:
    def test_the_path_starts_from_v0_and_has_one_value_per_step(self) -> None:
        path = variance_path(REFERENCE, 2.0, 7, uniforms(7, 3))
        assert len(path) == 7
        assert all(v >= 0.0 for v in path)

    @pytest.mark.parametrize("model", [REFERENCE, FELLER, SPIKY])
    def test_the_terminal_moments_survive_many_steps(self, model: Heston) -> None:
        # One step is exact on the moments by construction. Composing forty of
        # them is not, and this is where an error in the recursion would show.
        horizon, steps, draws = 1.0, 40, 8000
        rng = random.Random(97)
        finals = [
            variance_path(model, horizon, steps, [rng.random() for _ in range(steps)])[-1]
            for _ in range(draws)
        ]
        mean = math.fsum(finals) / draws
        spread = math.fsum((x - mean) ** 2 for x in finals) / (draws - 1)
        error = math.sqrt(spread / draws)
        assert mean == pytest.approx(conditional_mean(model, model.v0, horizon), abs=4 * error)
        assert spread == pytest.approx(
            conditional_variance(model, model.v0, horizon), rel=0.12
        )

    def test_the_wrong_number_of_uniforms_is_refused(self) -> None:
        with pytest.raises(ValueError, match="expected 5 uniforms"):
            variance_path(REFERENCE, 1.0, 5, uniforms(4, 1))

    def test_a_non_positive_step_count_is_refused(self) -> None:
        with pytest.raises(ValueError, match="steps must be at least 1"):
            variance_path(REFERENCE, 1.0, 0, [])


class TestPaths:
    def test_the_log_price_starts_at_zero_and_matches_the_step_count(self) -> None:
        prices, variances = paths(REFERENCE, 1.0, 12, uniforms(24, 5))
        assert len(prices) == 12
        assert len(variances) == 12
        assert all(math.isfinite(x) for x in prices)

    def test_it_is_reproducible(self) -> None:
        draw = uniforms(24, 5)
        assert paths(REFERENCE, 1.0, 12, draw) == paths(REFERENCE, 1.0, 12, draw)

    def test_zero_volatility_of_variance_gives_a_deterministic_variance(self) -> None:
        flat = Heston(0.05, 1.5, 0.04, 0.0, -0.5)
        _, variances = paths(flat, 1.0, 8, uniforms(16, 2))
        expected = 0.05
        for got in variances:
            expected = conditional_mean(flat, expected, 1.0 / 8)
            assert got == pytest.approx(expected, rel=1e-12)

    def test_the_wrong_number_of_uniforms_is_refused(self) -> None:
        with pytest.raises(ValueError, match="expected 24 uniforms"):
            paths(REFERENCE, 1.0, 12, uniforms(12, 5))


class TestTheMartingaleCorrection:
    def test_the_defect_it_removes_vanishes_at_the_long_run_variance(self) -> None:
        # The uncorrected drift constant is chosen so that it does. Away from
        # theta it does not, which is the next test.
        defect = _defect(REFERENCE, REFERENCE.theta, 0.05)
        assert abs(defect) < 1e-6

    @pytest.mark.parametrize(
        ("variance", "step", "basis_points"),
        [
            (0.0025, 0.25, 2.058),
            (0.2500, 0.25, -12.751),
            (0.0025, 0.05, 0.020),
            (0.2500, 0.05, -0.123),
        ],
    )
    def test_the_defect_is_what_the_documentation_says_it_is(
        self, variance: float, step: float, basis_points: float
    ) -> None:
        assert 1e4 * _defect(REFERENCE, variance, step) == pytest.approx(
            basis_points, abs=5e-3
        )

    def test_it_changes_sign_across_the_long_run_variance(self) -> None:
        assert _defect(REFERENCE, 0.01, 0.25) > 0.0
        assert _defect(REFERENCE, 0.09, 0.25) < 0.0

    def test_it_falls_faster_than_the_step_does(self) -> None:
        # Third order per step. Dividing the step by five cuts the defect by
        # about a hundred and twenty-five, not by five.
        coarse = abs(_defect(REFERENCE, 0.25, 0.25))
        fine = abs(_defect(REFERENCE, 0.25, 0.05))
        assert 80.0 < coarse / fine < 180.0

    def test_the_correction_makes_one_step_exactly_a_martingale(self) -> None:
        # Integrated over the uniform rather than sampled: the expectation of
        # the one-step price ratio, with the price shock taken analytically.
        model, step, n = REFERENCE, 0.2, 100_000
        coefficients = _coefficients(model, step)
        argument = coefficients.mgf_argument
        for variance in (0.005, 0.04, 0.2):
            proposal = _fit(model, variance, step)
            drift = -proposal.log_mgf(argument) - (
                coefficients.k1 + 0.5 * coefficients.k3
            ) * variance
            total = 0.0
            for i in range(n):
                nxt = proposal.draw((i + 0.5) / n)
                total += math.exp(
                    drift
                    + coefficients.k1 * variance
                    + coefficients.k2 * nxt
                    + 0.5 * (coefficients.k3 * variance + coefficients.k4 * nxt)
                )
            assert total / n == pytest.approx(1.0, rel=2e-3)

    @pytest.mark.parametrize(
        "model",
        [
            Heston(v0=0.01, kappa=2.0, theta=0.25, sigma=2.0, rho=0.9),
            Heston(v0=0.01, kappa=2.0, theta=1.0, sigma=2.0, rho=0.99),
        ],
    )
    def test_a_divergent_correction_is_reported_rather_than_dropped(
        self, model: Heston
    ) -> None:
        # It takes a positive correlation, a large volatility of variance and
        # a step of years rather than months before the moment generating
        # function stops being finite -- the two parameter sets here are the
        # exponential and the quadratic branch respectively, both at a
        # five-year step. On anything resembling a fitted model with a
        # sensible grid it never fires, which is why the correction is the
        # default. When it does fire it says so rather than quietly running
        # without the correction and returning a number that looks fine.
        with pytest.raises(MartingaleCorrectionError, match="diverges"):
            paths(model, 5.0, 1, uniforms(2, 1), martingale=True)
        assert len(paths(model, 5.0, 1, uniforms(2, 1), martingale=False)[0]) == 1

    def test_a_fitted_model_on_a_sensible_grid_never_reaches_that(self) -> None:
        # The claim above, swept rather than asserted.
        for model in (REFERENCE, FELLER, SPIKY):
            for steps in (1, 4, 12, 52):
                paths(model, 2.0, steps, uniforms(2 * steps, steps), martingale=True)


def _defect(model: Heston, variance: float, step: float) -> float:
    """The log of the one-step price ratio's expectation, uncorrected."""
    coefficients = _coefficients(model, step)
    proposal = _fit(model, variance, step)
    return (
        coefficients.k0
        + (coefficients.k1 + 0.5 * coefficients.k3) * variance
        + proposal.log_mgf(coefficients.mgf_argument)
    )


def _euler_european(
    model: Heston, contract: Contract, steps: int, settings: Settings
) -> float:
    """Full-truncation Euler, for comparison. The naive scheme, written naively."""
    step = contract.time / steps
    rho = model.rho
    orthogonal = math.sqrt(1.0 - rho * rho)
    forward, discount, strike = contract.forward, contract.discount, contract.strike

    def payoff(draw: list[float]) -> tuple[float, float]:
        variance, log_price = model.v0, 0.0
        for index in range(steps):
            first = norm_ppf(draw[2 * index])
            second = norm_ppf(draw[2 * index + 1])
            correlated = rho * first + orthogonal * second
            floor = max(variance, 0.0)
            root = math.sqrt(floor * step)
            log_price += -0.5 * floor * step + root * first
            variance += model.kappa * (model.theta - floor) * step + (
                model.sigma * root * correlated
            )
        terminal = forward * math.exp(log_price)
        return discount * max(terminal - strike, 0.0), discount * terminal

    values, controls = _replicate(settings, 2 * steps, payoff)  # type: ignore[arg-type]
    return _summarise(values, controls, discount * forward, "terminal", settings.paths).value


class TestAgainstTheTransform:
    """The simulation and the transform share no code. Agreement is evidence."""

    SETTINGS = Settings(paths=20_000, seed=11, antithetic=True, control=True)

    @pytest.mark.parametrize("strike", [80.0, 100.0, 125.0])
    @pytest.mark.parametrize("option", list(OptionType))
    def test_european_prices_agree_within_the_reported_error(
        self, strike: float, option: OptionType
    ) -> None:
        contract = Contract(100.0, strike, 1.0, 0.02, carry=0.0)
        estimate = european(REFERENCE, contract, option, steps=16, settings=self.SETTINGS)
        exact = transform_price(REFERENCE, contract, option)
        assert abs(estimate.value - exact) < 4.0 * estimate.standard_error

    @pytest.mark.parametrize("model", [REFERENCE, FELLER, SPIKY])
    def test_it_agrees_whether_or_not_the_feller_condition_holds(
        self, model: Heston
    ) -> None:
        contract = Contract(100.0, 100.0, 1.0, 0.01, carry=0.0)
        estimate = european(model, contract, steps=16, settings=self.SETTINGS)
        exact = transform_price(model, contract)
        assert abs(estimate.value - exact) < 4.0 * estimate.standard_error

    def test_four_steps_is_already_inside_the_noise(self) -> None:
        contract = Contract(100.0, 100.0, 1.0, 0.02, carry=0.0)
        estimate = european(REFERENCE, contract, steps=4, settings=self.SETTINGS)
        exact = transform_price(REFERENCE, contract)
        assert abs(estimate.value - exact) < 3.0 * estimate.standard_error

    def test_full_truncation_euler_is_not(self) -> None:
        # The comparison the module docstring rests on. At four steps a year
        # the naive scheme is out by an amount many times its own noise, and
        # in the same run the scheme here is not.
        contract = Contract(100.0, 100.0, 1.0, 0.02, carry=0.0)
        exact = transform_price(REFERENCE, contract)
        naive = _euler_european(REFERENCE, contract, 4, self.SETTINGS)
        good = european(REFERENCE, contract, steps=4, settings=self.SETTINGS)
        assert naive - exact > 0.5
        assert abs(good.value - exact) < 3.0 * good.standard_error

    def test_the_euler_bias_halves_with_each_doubling(self) -> None:
        contract = Contract(100.0, 100.0, 1.0, 0.02, carry=0.0)
        exact = transform_price(REFERENCE, contract)
        errors = [
            _euler_european(REFERENCE, contract, steps, self.SETTINGS) - exact
            for steps in (4, 8, 16)
        ]
        assert errors[0] > errors[1] > errors[2] > 0.0
        # Roughly first order, so quadrupling the steps should cut the bias by
        # about four. The bound is loose in both directions because each term
        # carries its own Monte Carlo noise; what it rules out is a bias that
        # is not falling, or one falling fast enough to be mistaken for the
        # scheme in this module.
        assert 2.5 < errors[0] / errors[2] < 8.0

    def test_a_deterministic_variance_reduces_to_the_lognormal_price(self) -> None:
        from moneyness import price as bs_price

        flat = Heston(0.05, 1.5, 0.04, 0.0, -0.5)
        contract = Contract(100.0, 105.0, 1.0, 0.02, carry=0.01)
        estimate = european(flat, contract, steps=32, settings=self.SETTINGS)
        exact = bs_price(contract.with_vol(flat.equivalent_vol(1.0)), OptionType.CALL)
        assert abs(estimate.value - exact) < 4.0 * estimate.standard_error


class TestPathDependentPayoffs:
    SETTINGS = Settings(paths=8_000, seed=5, antithetic=True, control=True)

    def test_an_asian_call_is_worth_less_than_the_european(self) -> None:
        # Averaging cuts the variance of the terminal quantity, so the option
        # on the average is cheaper than the option on the endpoint.
        contract = Contract(100.0, 100.0, 1.0, 0.02, carry=0.0)
        average = asian(REFERENCE, contract, steps=16, settings=self.SETTINGS)
        endpoint = transform_price(REFERENCE, contract)
        assert average.value < endpoint
        assert average.value > 0.0

    def test_a_knock_out_and_its_knock_in_add_to_the_vanilla(self) -> None:
        # In-out parity, which holds for any path law and so is a check on the
        # bookkeeping rather than on the model.
        contract = Contract(100.0, 100.0, 1.0, 0.0, carry=0.0)
        out = barrier(
            REFERENCE, contract, OptionType.CALL, Barrier.DOWN_AND_OUT, 80.0,
            steps=16, settings=self.SETTINGS,
        )
        inside = barrier(
            REFERENCE, contract, OptionType.CALL, Barrier.DOWN_AND_IN, 80.0,
            steps=16, settings=self.SETTINGS,
        )
        vanilla = transform_price(REFERENCE, contract)
        total = out.value + inside.value
        error = math.hypot(out.standard_error, inside.standard_error)
        assert abs(total - vanilla) < 4.0 * error

    def test_the_bridge_lowers_a_knock_out(self) -> None:
        # Monitoring only on the grid misses crossings inside a step, so it
        # keeps options alive that should have knocked out.
        contract = Contract(100.0, 100.0, 1.0, 0.0, carry=0.0)
        discrete = barrier(
            REFERENCE, contract, OptionType.CALL, Barrier.DOWN_AND_OUT, 90.0,
            steps=8, settings=self.SETTINGS, bridge=False,
        )
        continuous = barrier(
            REFERENCE, contract, OptionType.CALL, Barrier.DOWN_AND_OUT, 90.0,
            steps=8, settings=self.SETTINGS, bridge=True,
        )
        assert continuous.value < discrete.value

    def test_the_bridge_closes_the_gap_between_a_coarse_and_a_fine_grid(self) -> None:
        # The point of the correction: with it, the answer stops depending so
        # strongly on how often the grid happens to look.
        contract = Contract(100.0, 100.0, 1.0, 0.0, carry=0.0)

        def value(steps: int, bridge: bool) -> float:
            return barrier(
                REFERENCE, contract, OptionType.CALL, Barrier.DOWN_AND_OUT, 90.0,
                steps=steps, settings=self.SETTINGS, bridge=bridge,
            ).value

        without = abs(value(8, False) - value(32, False))
        with_it = abs(value(8, True) - value(32, True))
        assert with_it < without

    def test_a_barrier_already_breached_is_worthless(self) -> None:
        contract = Contract(100.0, 100.0, 1.0, 0.0, carry=0.0)
        knocked = barrier(
            REFERENCE, contract, OptionType.CALL, Barrier.UP_AND_OUT, 100.0,
            steps=4, settings=self.SETTINGS,
        )
        assert knocked.value == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("level", [0.0, -5.0])
    def test_a_non_positive_barrier_is_refused(self, level: float) -> None:
        contract = Contract(100.0, 100.0, 1.0, 0.0, carry=0.0)
        with pytest.raises(ValueError, match="level must be positive"):
            barrier(REFERENCE, contract, OptionType.CALL, Barrier.DOWN_AND_OUT, level)


class TestInterface:
    SETTINGS = Settings(paths=2_000, seed=1, antithetic=True, control=True)

    def test_antithetic_sampling_halves_the_sample_count(self) -> None:
        contract = Contract(100.0, 100.0, 0.5, 0.02, carry=0.0)
        paired = european(
            REFERENCE, contract, steps=4,
            settings=Settings(paths=2_000, seed=1, antithetic=True),
        )
        plain = european(
            REFERENCE, contract, steps=4,
            settings=Settings(paths=2_000, seed=1, antithetic=False),
        )
        assert paired.samples == 1_000
        assert plain.samples == 2_000
        assert paired.paths == plain.paths == 2_000

    def test_the_control_variate_is_named_when_it_is_used(self) -> None:
        contract = Contract(100.0, 100.0, 0.5, 0.02, carry=0.0)
        with_control = european(REFERENCE, contract, steps=4, settings=self.SETTINGS)
        without = european(
            REFERENCE, contract, steps=4,
            settings=Settings(paths=2_000, seed=1, control=False),
        )
        assert with_control.control == "discounted terminal price"
        assert without.control is None
        assert with_control.standard_error < without.standard_error

    def test_the_interval_brackets_the_estimate(self) -> None:
        contract = Contract(100.0, 100.0, 0.5, 0.02, carry=0.0)
        estimate = european(REFERENCE, contract, steps=4, settings=self.SETTINGS)
        low, high = estimate.interval(0.95)
        assert low < estimate.value < high

    def test_the_same_seed_gives_the_same_answer(self) -> None:
        contract = Contract(100.0, 100.0, 0.5, 0.02, carry=0.0)
        first = european(REFERENCE, contract, steps=4, settings=self.SETTINGS)
        second = european(REFERENCE, contract, steps=4, settings=self.SETTINGS)
        assert first.value == second.value

    @pytest.mark.parametrize(
        ("spot", "time", "steps", "message"),
        [
            (100.0, 0.0, 4, "time must be positive"),
            (0.0, 1.0, 4, "spot must be positive"),
            (100.0, 1.0, 0, "steps must be at least 1"),
        ],
    )
    def test_a_degenerate_request_is_refused(
        self, spot: float, time: float, steps: int, message: str
    ) -> None:
        contract = Contract(spot, 100.0, time, 0.02, carry=0.0)
        with pytest.raises(ValueError, match=message):
            european(REFERENCE, contract, steps=steps, settings=self.SETTINGS)

    @pytest.mark.parametrize("model", [REFERENCE, FELLER, SPIKY])
    def test_nothing_returns_a_non_finite_price(self, model: Heston) -> None:
        contract = Contract(100.0, 100.0, 3.0, 0.03, carry=0.01)
        for option in OptionType:
            estimate = european(model, contract, option, steps=8, settings=self.SETTINGS)
            assert math.isfinite(estimate.value)
            assert math.isfinite(estimate.standard_error)
            assert estimate.value >= 0.0
