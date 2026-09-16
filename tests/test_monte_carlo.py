"""Monte Carlo, checked against closed forms and against its own claims.

A simulation is easy to test badly. The estimate is random, so any assertion has
to be about a distribution rather than a number, and a tolerance chosen loosely
enough to stop the suite flickering will also accept an estimator that is wrong.

The approach here is to assert the things that are *not* random. In-out parity
holds path by path and so holds exactly. The geometric Asian has a closed form,
so the simulation of it can be checked against a known answer rather than
against itself. The Reiner-Rubinstein barrier formula, implemented in this
module as an independent oracle, gives the continuously monitored price the
bridge correction is supposed to recover. Where an assertion genuinely has to be
statistical — coverage, the convergence rate — it is made over many seeds, and
the tolerance is derived from how much the statistic itself can move.
"""

from __future__ import annotations

import math

import pytest

from moneyness.bsm import Inputs, OptionType, price
from moneyness.monte_carlo import (
    Barrier,
    Estimate,
    Settings,
    asian,
    barrier,
    european,
    geometric_asian,
)
from moneyness.normal import norm_cdf

AT_THE_MONEY = Inputs(100.0, 100.0, 1.0, 0.05, 0.2)

# A grid spanning moneyness, maturity, volatility and the carry conventions.
CASES = [
    Inputs(100.0, 100.0, 1.0, 0.05, 0.2),
    Inputs(100.0, 80.0, 0.5, 0.03, 0.35),
    Inputs(100.0, 130.0, 2.0, 0.02, 0.25),
    Inputs(50.0, 55.0, 0.25, 0.04, 0.4),
    Inputs.with_dividend(100.0, 100.0, 1.0, 0.05, 0.2, 0.03),
    Inputs.on_future(100.0, 95.0, 1.5, 0.04, 0.3),
]


def reference_down_and_out_call(inputs: Inputs, level: float) -> float:
    """The Reiner-Rubinstein price of a continuously monitored down-and-out call.

    An independent oracle, in the spirit of ``tests/reference.py``: it shares no
    code with the simulation and arrives at the answer by a completely different
    route, so agreement is evidence about the bridge correction rather than
    about either implementation.

    Restricted to the branch where the strike is above the barrier, which is the
    only one used here and the one with the simplest decomposition, ``A - C``.
    """
    spot, strike, time = inputs.spot, inputs.strike, inputs.time
    rate, carry, vol = inputs.rate, inputs.b, inputs.vol
    assert strike > level, "this branch of the formula needs the strike above the barrier"

    drift = (carry - vol * vol / 2.0) / (vol * vol)
    spread = vol * math.sqrt(time)
    x1 = math.log(spot / strike) / spread + (1.0 + drift) * spread
    y1 = math.log(level * level / (spot * strike)) / spread + (1.0 + drift) * spread

    carry_factor = math.exp((carry - rate) * time)
    discount = math.exp(-rate * time)
    reflection = level / spot

    vanilla_part: float = spot * carry_factor * norm_cdf(x1) - strike * discount * norm_cdf(
        x1 - spread
    )
    reflected_part: float = spot * carry_factor * reflection ** (
        2.0 * (drift + 1.0)
    ) * norm_cdf(y1) - strike * discount * reflection ** (2.0 * drift) * norm_cdf(y1 - spread)
    return vanilla_part - reflected_part


class TestSettings:
    def test_antithetic_halves_the_independent_sample_count(self) -> None:
        assert Settings(paths=1000, antithetic=True).samples == 500
        assert Settings(paths=1000, antithetic=False).samples == 1000

    def test_an_odd_path_count_rounds_down_when_paired(self) -> None:
        assert Settings(paths=1001, antithetic=True).samples == 500

    @pytest.mark.parametrize(
        ("paths", "antithetic"), [(1, True), (0, False), (-10, False), (3, True)]
    )
    def test_rejects_too_few_paths(self, paths: int, antithetic: bool) -> None:
        with pytest.raises(ValueError):
            Settings(paths=paths, antithetic=antithetic)


class TestEstimate:
    def test_the_interval_is_the_standard_error_scaled_by_a_quantile(self) -> None:
        estimate = Estimate(10.0, 0.25, 1000, 2000, None)
        lower, upper = estimate.interval(0.95)
        assert (upper - lower) / 2.0 == pytest.approx(1.959963984540054 * 0.25, rel=1e-12)
        assert (lower + upper) / 2.0 == pytest.approx(10.0, rel=1e-15)

    def test_a_higher_level_gives_a_wider_interval(self) -> None:
        estimate = Estimate(10.0, 0.25, 1000, 2000, None)
        narrow = estimate.interval(0.90)
        wide = estimate.interval(0.99)
        assert wide[0] < narrow[0] < narrow[1] < wide[1]

    @pytest.mark.parametrize("level", [0.0, 1.0, -0.5, 1.5])
    def test_rejects_an_impossible_level(self, level: float) -> None:
        with pytest.raises(ValueError, match="strictly between"):
            Estimate(10.0, 0.25, 1000, 2000, None).interval(level)


class TestEuropean:
    @pytest.mark.parametrize("inputs", CASES)
    @pytest.mark.parametrize("option", list(OptionType))
    def test_agrees_with_the_closed_form(self, inputs: Inputs, option: OptionType) -> None:
        """The estimate must sit inside its own interval around the true value.

        This is the check the whole module is calibrated by: the answer is known
        exactly, so a failure is unambiguous.
        """
        exact = price(inputs, option)
        estimate = european(inputs, option, Settings(paths=80_000, seed=17))
        lower, upper = estimate.interval(0.999)
        assert lower <= exact <= upper

    def test_antithetic_sampling_reduces_the_standard_error(self) -> None:
        plain = european(
            AT_THE_MONEY,
            OptionType.CALL,
            Settings(paths=200_000, seed=1, antithetic=False, control=False),
        )
        paired = european(
            AT_THE_MONEY,
            OptionType.CALL,
            Settings(paths=200_000, seed=1, antithetic=True, control=False),
        )
        assert paired.standard_error < plain.standard_error

    def test_the_control_variate_reduces_the_standard_error_further(self) -> None:
        """Measured, not assumed: the control cuts the error by about a factor of four."""
        paired = european(
            AT_THE_MONEY,
            OptionType.CALL,
            Settings(paths=200_000, seed=1, antithetic=True, control=False),
        )
        controlled = european(
            AT_THE_MONEY,
            OptionType.CALL,
            Settings(paths=200_000, seed=1, antithetic=True, control=True),
        )
        assert controlled.standard_error < paired.standard_error / 3.0
        assert controlled.control == "discounted terminal price"

    def test_reports_no_control_when_it_is_switched_off(self) -> None:
        estimate = european(AT_THE_MONEY, OptionType.CALL, Settings(paths=1000, control=False))
        assert estimate.control is None

    def test_the_sample_count_is_independent_samples_not_paths(self) -> None:
        """The distinction the standard error depends on."""
        estimate = european(AT_THE_MONEY, OptionType.CALL, Settings(paths=10_000))
        assert estimate.paths == 10_000
        assert estimate.samples == 5_000

    def test_is_reproducible_for_a_fixed_seed(self) -> None:
        first = european(AT_THE_MONEY, OptionType.CALL, Settings(paths=5000, seed=42))
        second = european(AT_THE_MONEY, OptionType.CALL, Settings(paths=5000, seed=42))
        assert first == second

    def test_a_different_seed_gives_a_different_answer(self) -> None:
        first = european(AT_THE_MONEY, OptionType.CALL, Settings(paths=5000, seed=1))
        second = european(AT_THE_MONEY, OptionType.CALL, Settings(paths=5000, seed=2))
        assert first.value != second.value

    @pytest.mark.parametrize(
        "inputs",
        [
            Inputs(100.0, 100.0, 0.0, 0.05, 0.2),
            Inputs(100.0, 100.0, 1.0, 0.05, 0.0),
            Inputs(0.0, 100.0, 1.0, 0.05, 0.2),
        ],
    )
    def test_rejects_degenerate_inputs(self, inputs: Inputs) -> None:
        """Degenerate cases have closed forms and no randomness; simulating them is a
        category error, so the caller is sent back to the pricer."""
        with pytest.raises(ValueError, match="must be positive"):
            european(inputs, OptionType.CALL)


class TestConvergenceRate:
    def test_the_error_falls_as_one_over_the_square_root_of_the_sample_count(self) -> None:
        """Quadrupling the sample count should halve the error.

        A single run says nothing about a rate, so this measures the
        root-mean-square error over forty independent seeds at each size. Forty
        replications pin the RMS itself to about eleven percent, so the ratio
        between consecutive sizes carries roughly sixteen percent of noise and
        the band below is set accordingly — wide enough not to flicker, narrow
        enough that a rate of one or one quarter would fail it outright.
        """
        exact = price(AT_THE_MONEY, OptionType.CALL)
        previous = None
        ratios = []
        for paths in (1000, 4000, 16_000, 64_000):
            errors = [
                european(
                    AT_THE_MONEY,
                    OptionType.CALL,
                    Settings(paths=paths, seed=1000 + seed, antithetic=False, control=False),
                ).value
                - exact
                for seed in range(40)
            ]
            rms = math.sqrt(math.fsum(e * e for e in errors) / len(errors))
            if previous is not None:
                ratios.append(previous / rms)
            previous = rms

        assert len(ratios) == 3
        for ratio in ratios:
            assert 1.6 < ratio < 2.6, f"observed error ratios {ratios}, expected about 2"

    def test_the_reported_standard_error_matches_the_observed_spread(self) -> None:
        """The estimator's own claim about its noise, checked against reality.

        A standard error is a prediction: the spread of the estimate across
        seeds should match it. An implementation that divided by the path count
        instead of the sample count would pass every agreement test above and
        fail this one.
        """
        values = []
        reported = []
        for seed in range(60):
            estimate = european(AT_THE_MONEY, OptionType.CALL, Settings(paths=20_000, seed=seed))
            values.append(estimate.value)
            reported.append(estimate.standard_error)

        mean = math.fsum(values) / len(values)
        observed = math.sqrt(math.fsum((v - mean) ** 2 for v in values) / (len(values) - 1))
        claimed = math.fsum(reported) / len(reported)
        assert observed == pytest.approx(claimed, rel=0.35)

    def test_intervals_cover_the_truth_at_about_their_nominal_rate(self) -> None:
        """Coverage over many seeds, which is what a confidence interval promises.

        This is also the check that would catch the control variate's
        small-sample bias if it ever grew large enough to matter.
        """
        exact = price(AT_THE_MONEY, OptionType.CALL)
        trials = 120
        covered = sum(
            1
            for seed in range(trials)
            if (
                lambda bounds: bounds[0] <= exact <= bounds[1]
            )(
                european(
                    AT_THE_MONEY, OptionType.CALL, Settings(paths=20_000, seed=5000 + seed)
                ).interval(0.95)
            )
        )
        # The count is binomial with p = 0.95; three standard deviations over
        # 120 trials is about six.
        assert covered >= trials - 12, f"only {covered} of {trials} intervals covered the truth"


class TestGeometricAsianClosedForm:
    @pytest.mark.parametrize("inputs", CASES)
    @pytest.mark.parametrize("option", list(OptionType))
    def test_a_single_monitoring_date_is_the_vanilla(
        self, inputs: Inputs, option: OptionType
    ) -> None:
        """With one date, at expiry, the average is the terminal price.

        Exact rather than approximate, and a sharp check on the moment algebra:
        at ``m = 1`` the mean and variance expressions must collapse to the
        plain lognormal ones.
        """
        assert geometric_asian(inputs, option, 1) == pytest.approx(
            price(inputs, option), rel=1e-12
        )

    @pytest.mark.parametrize("inputs", CASES)
    def test_averaging_reduces_the_value_of_a_call(self, inputs: Inputs) -> None:
        """Averaging damps the variance, so an average-price call is worth less."""
        monitored = [geometric_asian(inputs, OptionType.CALL, m) for m in (1, 2, 4, 12, 52)]
        assert monitored == sorted(monitored, reverse=True)

    @pytest.mark.parametrize("steps", [1, 4, 12, 52])
    @pytest.mark.parametrize("option", list(OptionType))
    def test_the_simulation_agrees_with_the_closed_form(
        self, steps: int, option: OptionType
    ) -> None:
        """Simulating the geometric average, whose answer is known exactly."""
        exact = geometric_asian(AT_THE_MONEY, option, steps)
        estimate = asian(
            AT_THE_MONEY, option, steps, Settings(paths=80_000, seed=3), geometric=True
        )
        lower, upper = estimate.interval(0.999)
        assert lower <= exact <= upper

    def test_rejects_a_nonsensical_schedule(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            geometric_asian(AT_THE_MONEY, OptionType.CALL, 0)


class TestArithmeticAsian:
    @pytest.mark.parametrize("steps", [4, 12, 52])
    def test_is_worth_more_than_its_geometric_counterpart(self, steps: int) -> None:
        """The arithmetic mean dominates the geometric one, so the call is worth more.

        An inequality that holds path by path, so no tolerance is needed beyond
        the simulation's own noise — and the two are computed on the same paths,
        which removes most of even that.
        """
        arithmetic = asian(AT_THE_MONEY, OptionType.CALL, steps, Settings(paths=60_000, seed=9))
        geometric = geometric_asian(AT_THE_MONEY, OptionType.CALL, steps)
        assert arithmetic.value > geometric

    def test_the_geometric_control_is_far_better_than_no_control(self) -> None:
        """Measured: the geometric average tracks the arithmetic one closely.

        A control is only worth its cost if it correlates with the target. This
        one is a function of the same whole path, and the reduction is an order
        of magnitude rather than a few percent.
        """
        plain = asian(
            AT_THE_MONEY, OptionType.CALL, 12, Settings(paths=60_000, seed=9, control=False)
        )
        controlled = asian(
            AT_THE_MONEY, OptionType.CALL, 12, Settings(paths=60_000, seed=9, control=True)
        )
        assert controlled.standard_error < plain.standard_error / 10.0
        assert controlled.control == "geometric-average Asian"

    def test_the_control_is_switched_off_when_simulating_the_geometric_average(self) -> None:
        """Otherwise the control would equal the payoff and the estimator would
        collapse onto the closed form, testing nothing."""
        estimate = asian(AT_THE_MONEY, OptionType.CALL, 12, geometric=True)
        assert estimate.control is None

    def test_rejects_a_nonsensical_schedule(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            asian(AT_THE_MONEY, OptionType.CALL, 0)


class TestBarrier:
    @pytest.mark.parametrize("steps", [12, 50])
    @pytest.mark.parametrize("option", list(OptionType))
    def test_in_out_parity_holds_exactly(self, steps: int, option: OptionType) -> None:
        """Every path knocks out or it does not, so the two prices must sum to the vanilla.

        Exactly, not approximately. The knock-in is computed as the vanilla
        minus the knock-out on the very same paths, and the control variate is
        the vanilla payoff with its known mean, so the two regression
        adjustments sum to the vanilla price by construction. An implementation
        that priced the knock-in independently would only get this to within two
        simulations' noise.
        """
        settings = Settings(paths=60_000, seed=5)
        out = barrier(AT_THE_MONEY, option, 90.0, Barrier.DOWN_AND_OUT, steps, settings)
        into = barrier(AT_THE_MONEY, option, 90.0, Barrier.DOWN_AND_IN, steps, settings)
        assert out.value + into.value == pytest.approx(
            price(AT_THE_MONEY, option), abs=1e-9
        )

    @pytest.mark.parametrize("style", [Barrier.UP_AND_OUT, Barrier.UP_AND_IN])
    def test_in_out_parity_holds_for_up_barriers_too(self, style: Barrier) -> None:
        settings = Settings(paths=40_000, seed=6)
        other = Barrier.UP_AND_IN if style is Barrier.UP_AND_OUT else Barrier.UP_AND_OUT
        first = barrier(AT_THE_MONEY, OptionType.CALL, 130.0, style, 25, settings)
        second = barrier(AT_THE_MONEY, OptionType.CALL, 130.0, other, 25, settings)
        assert first.value + second.value == pytest.approx(
            price(AT_THE_MONEY, OptionType.CALL), abs=1e-9
        )

    def test_a_knock_out_is_worth_less_than_the_vanilla(self) -> None:
        out = barrier(AT_THE_MONEY, OptionType.CALL, 90.0, Barrier.DOWN_AND_OUT, 25)
        assert 0.0 < out.value < price(AT_THE_MONEY, OptionType.CALL)

    @pytest.mark.parametrize("steps", [10, 50, 200])
    def test_the_bridge_recovers_the_continuously_monitored_price(self, steps: int) -> None:
        """Against the Reiner-Rubinstein formula, an entirely separate derivation.

        The bridge is essentially unbiased even at ten monitoring dates, which
        is the whole claim being made for it.
        """
        exact = reference_down_and_out_call(AT_THE_MONEY, 90.0)
        estimate = barrier(
            AT_THE_MONEY,
            OptionType.CALL,
            90.0,
            Barrier.DOWN_AND_OUT,
            steps,
            Settings(paths=80_000, seed=11),
        )
        lower, upper = estimate.interval(0.999)
        assert lower <= exact <= upper

    def test_without_the_bridge_the_price_is_biased_upward(self) -> None:
        """The bias the correction exists to remove, and its direction.

        A naive grid misses paths that cross the barrier and return between
        observations, so options get paid out on paths that should have been
        extinguished, and the knock-out is worth too much.
        """
        exact = reference_down_and_out_call(AT_THE_MONEY, 90.0)
        naive = barrier(
            AT_THE_MONEY,
            OptionType.CALL,
            90.0,
            Barrier.DOWN_AND_OUT,
            10,
            Settings(paths=60_000, seed=7),
            bridge=False,
        )
        assert naive.value > exact + 10.0 * naive.standard_error

    def test_the_bridge_removes_most_of_the_dependence_on_the_grid(self) -> None:
        """The measured claim: about a fifteenfold reduction in step dependence.

        Sweeping the monitoring frequency over a factor of eighty, the naive
        price moves by about 1.0 while the bridged one moves by about 0.03 — and
        the latter is within a couple of standard errors, so what remains is
        simulation noise rather than residual bias. The asserted bounds are
        looser than those figures, since the spread of a handful of noisy
        estimates is itself noisy.
        """
        naive = []
        bridged = []
        for steps in (5, 25, 100, 400):
            settings = Settings(paths=40_000, seed=7)
            naive.append(
                barrier(
                    AT_THE_MONEY,
                    OptionType.CALL,
                    90.0,
                    Barrier.DOWN_AND_OUT,
                    steps,
                    settings,
                    bridge=False,
                ).value
            )
            bridged.append(
                barrier(
                    AT_THE_MONEY,
                    OptionType.CALL,
                    90.0,
                    Barrier.DOWN_AND_OUT,
                    steps,
                    settings,
                    bridge=True,
                ).value
            )

        naive_spread = max(naive) - min(naive)
        bridged_spread = max(bridged) - min(bridged)
        assert naive_spread > 0.8
        assert bridged_spread < 0.15
        assert naive_spread / bridged_spread > 8.0
        # The naive price falls towards the continuous one as the grid refines.
        assert naive[0] > naive[-1]

    @pytest.mark.parametrize(
        ("level", "style", "match"),
        [
            (110.0, Barrier.DOWN_AND_OUT, "already breached"),
            (100.0, Barrier.DOWN_AND_IN, "already breached"),
            (90.0, Barrier.UP_AND_OUT, "already breached"),
            (100.0, Barrier.UP_AND_IN, "already breached"),
            (0.0, Barrier.DOWN_AND_OUT, "must be positive"),
            (-5.0, Barrier.DOWN_AND_OUT, "must be positive"),
        ],
    )
    def test_rejects_a_barrier_on_the_wrong_side_of_spot(
        self, level: float, style: Barrier, match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            barrier(AT_THE_MONEY, OptionType.CALL, level, style, 10)

    def test_rejects_a_nonsensical_schedule(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            barrier(AT_THE_MONEY, OptionType.CALL, 90.0, Barrier.DOWN_AND_OUT, 0)


class TestBarrierStyle:
    @pytest.mark.parametrize(
        ("style", "down", "knock_out"),
        [
            (Barrier.DOWN_AND_OUT, True, True),
            (Barrier.DOWN_AND_IN, True, False),
            (Barrier.UP_AND_OUT, False, True),
            (Barrier.UP_AND_IN, False, False),
        ],
    )
    def test_the_flags_agree_with_the_name(
        self, style: Barrier, down: bool, knock_out: bool
    ) -> None:
        assert style.is_down is down
        assert style.is_knock_out is knock_out
