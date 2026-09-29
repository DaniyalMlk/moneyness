"""Simulating Heston's model, by Andersen's quadratic-exponential scheme.

The transform in :mod:`moneyness.heston` prices European options and nothing
else. This module simulates the process itself, which buys two things: payoffs
the transform cannot reach, and a check on the transform that shares no code
with it.

**Why not Euler.** The variance follows a square-root process, and an Euler
step can take it negative — certainly when the Feller condition fails, which
is the normal case for a model fitted to index quotes. Every repair for that
biases the result, and the bias is large. Full truncation, the mildest of
them, misprices a one-year at-the-money call by 0.718 on a value of 6.809 at
four steps a year, and the error halves with each doubling of the grid: 0.287
at eight steps, 0.106 at sixteen, 0.039 at thirty-two. The scheme below is
inside its own Monte Carlo error at *four* steps, and stays there. Reaching
that by refining an Euler grid costs upwards of a sixty-fourfold increase in
work.

**What the scheme does instead.** Over one step the square-root process has a
known conditional mean and variance, in closed form, whatever the parameters.
Andersen's observation is that the conditional distribution is well
approximated by one of two simple laws chosen by a single statistic — the
ratio of that variance to the square of that mean. Where the ratio is small
the distribution is close to a scaled non-central chi-square with one degree
of freedom, so a squared normal is fitted to the two moments. Where it is
large the distribution has an atom near zero and an exponential tail, so a
point mass plus an exponential is fitted to the same two moments. Both
proposals are non-negative by construction, both are exact on the first two
moments, and neither needs the Feller condition.

**The log price.** Integrating the variance over the step rather than sampling
it at the endpoints is what makes the scheme accurate rather than merely
stable. The update carries the endpoint variances with weights, together with
the correlation decomposition that makes the price's own shock independent of
the variance's; the drift constant absorbs the correlation term exactly.

**The martingale correction.** Nothing above guarantees the simulated forward
equals the real one. The discretised price is a product of exponentials whose
expectation is available in closed form for each of the two proposals, so the
drift constant can be set to make it exactly one per step instead of
approximately one.

It is worth saying how small that turns out to be, because the correction is
often described as though it were the point. The uncorrected drift constant
is chosen so the defect vanishes when the variance sits at its long-run
level, and it is third order in the step: on the reference parameters it is
0.02 basis points per step at twenty steps a year and 2.1 at four. What makes
it visible is distance from that long-run level, not the step alone — at a
variance six times ``theta`` and four steps a year it reaches 12.8 basis
points per step, and it changes sign as the variance crosses ``theta``. So
the correction earns its place on coarse grids and in the wings of a
simulated distribution, and does nothing measurable in the middle. It is on
by default and can be switched off, which is what the tests that measure the
uncorrected bias do.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum

from .bsm import OptionType
from .heston import Contract, Heston
from .monte_carlo import Barrier, Estimate, Settings, _summarise
from .normal import norm_ppf

__all__ = [
    "MartingaleCorrectionError",
    "Scheme",
    "asian",
    "barrier",
    "conditional_mean",
    "conditional_variance",
    "european",
    "paths",
    "variance_path",
]

# Andersen's switching level between the two proposals. The squared normal can
# be fitted to the moments only while the ratio is at most 2; the exponential
# only while it is at least 1. Anything in the overlap works, and 1.5 is the
# midpoint of it.
_PSI_CRITICAL = 1.5

# Weights on the two endpoint variances in the integral over a step. Equal
# weights are the trapezium rule, which is second order where taking the left
# endpoint alone is first.
_GAMMA_1 = 0.5
_GAMMA_2 = 0.5


class MartingaleCorrectionError(ValueError):
    """The martingale correction is not defined at this step.

    Its constant is the logarithm of a moment generating function, which is
    finite only below a parameter-dependent threshold. For a negative
    correlation the argument is negative and the threshold never binds; for a
    strongly positive one with a small volatility of variance it can. Running
    without the correction is the fallback, and the error says so rather than
    quietly dropping it.
    """


class Scheme(str, Enum):
    """Which proposal the variance step used. Reported for diagnostics."""

    QUADRATIC = "quadratic"
    EXPONENTIAL = "exponential"


def conditional_mean(model: Heston, variance: float, step: float) -> float:
    """``E[V_{t+dt} | V_t]`` for the square-root process.

    Args:
        model: The variance process.
        variance: ``V_t``. Non-negative.
        step: Length of the step. Non-negative.

    Returns:
        The conditional mean.
    """
    _check_step(variance, step)
    decay = math.exp(-model.kappa * step)
    return model.theta + (variance - model.theta) * decay


def conditional_variance(model: Heston, variance: float, step: float) -> float:
    """``Var[V_{t+dt} | V_t]`` for the square-root process.

    Exact, and the Feller condition plays no part in it. Together with
    :func:`conditional_mean` this is everything the scheme fits its proposals
    to.

    Args:
        model: The variance process.
        variance: ``V_t``. Non-negative.
        step: Length of the step. Non-negative.

    Returns:
        The conditional variance.
    """
    _check_step(variance, step)
    kappa, theta, sigma = model.kappa, model.theta, model.sigma
    decay = math.exp(-kappa * step)
    gap = -math.expm1(-kappa * step)
    return (
        variance * sigma * sigma * decay * gap / kappa
        + theta * sigma * sigma * gap * gap / (2.0 * kappa)
    )


def _check_step(variance: float, step: float) -> None:
    if variance < 0.0:
        raise ValueError(f"variance must be non-negative, got {variance}")
    if step < 0.0:
        raise ValueError(f"step must be non-negative, got {step}")


@dataclass(frozen=True, slots=True)
class _Proposal:
    """The fitted one-step law for the variance, and its moment generator.

    ``kind`` says which of the two was fitted. The remaining fields are that
    proposal's parameters: ``a`` and ``b_squared`` for the quadratic branch,
    ``p`` and ``beta`` for the exponential one. Holding both in one object
    keeps the martingale correction beside the draw it corrects, which is the
    only place the distinction matters to a caller.
    """

    kind: Scheme
    a: float
    b_squared: float
    p: float
    beta: float

    def draw(self, uniform: float) -> float:
        """One variance from one uniform on [0, 1)."""
        if self.kind is Scheme.QUADRATIC:
            z = norm_ppf(min(max(uniform, 1e-300), 1.0 - 1e-16))
            root = math.sqrt(self.b_squared) + z
            return self.a * root * root
        if uniform <= self.p:
            return 0.0
        return math.log((1.0 - self.p) / (1.0 - uniform)) / self.beta

    def log_mgf(self, argument: float) -> float:
        """``log E[exp(argument * V_{t+dt})]`` under this proposal.

        Raises:
            MartingaleCorrectionError: If the expectation is infinite.
        """
        if self.kind is Scheme.QUADRATIC:
            denominator = 1.0 - 2.0 * argument * self.a
            if denominator <= 0.0:
                raise MartingaleCorrectionError(
                    f"the quadratic proposal's moment generating function diverges: "
                    f"2 * {argument} * {self.a} reached 1. Run with "
                    "martingale=False, or take smaller steps."
                )
            return argument * self.b_squared * self.a / denominator - 0.5 * math.log(
                denominator
            )
        if argument >= self.beta:
            raise MartingaleCorrectionError(
                f"the exponential proposal's moment generating function diverges: "
                f"the argument {argument} reached the rate {self.beta}. Run with "
                "martingale=False, or take smaller steps."
            )
        return math.log(self.p + self.beta * (1.0 - self.p) / (self.beta - argument))


def _fit(model: Heston, variance: float, step: float) -> _Proposal:
    """Fit a one-step proposal to the exact conditional moments."""
    mean = conditional_mean(model, variance, step)
    if mean <= 0.0:
        # Only reachable when theta and the current variance are both zero, in
        # which case the process is pinned there and so is the proposal.
        return _Proposal(Scheme.EXPONENTIAL, 0.0, 0.0, 1.0, math.inf)
    ratio = conditional_variance(model, variance, step) / (mean * mean)
    if ratio <= _PSI_CRITICAL:
        inverse = 2.0 / ratio
        b_squared = inverse - 1.0 + math.sqrt(inverse) * math.sqrt(inverse - 1.0)
        return _Proposal(Scheme.QUADRATIC, mean / (1.0 + b_squared), b_squared, 0.0, 0.0)
    p = (ratio - 1.0) / (ratio + 1.0)
    return _Proposal(Scheme.EXPONENTIAL, 0.0, 0.0, p, (1.0 - p) / mean)


def variance_path(
    model: Heston, time: float, steps: int, uniforms: Sequence[float]
) -> list[float]:
    """The variance at ``steps`` equally spaced dates, from ``steps`` uniforms.

    The path starts at ``model.v0``, which is not returned; the list holds the
    variance at the end of each step.

    Args:
        model: The variance process.
        time: Horizon.
        steps: Number of steps.
        uniforms: One value in [0, 1) per step.

    Returns:
        The simulated variances, in time order.

    Raises:
        ValueError: If ``steps`` is not positive or the draw is the wrong length.
    """
    if steps < 1:
        raise ValueError(f"steps must be at least 1, got {steps}")
    if len(uniforms) != steps:
        raise ValueError(f"expected {steps} uniforms, got {len(uniforms)}")
    step = time / steps
    variance = model.v0
    out: list[float] = []
    for uniform in uniforms:
        variance = _fit(model, variance, step).draw(uniform)
        out.append(variance)
    return out


@dataclass(frozen=True, slots=True)
class _Coefficients:
    """The log-price update's constants, which depend only on the step."""

    k0: float
    k1: float
    k2: float
    k3: float
    k4: float

    @property
    def mgf_argument(self) -> float:
        """The coefficient the martingale correction takes an expectation over."""
        return self.k2 + 0.5 * self.k4


def _coefficients(model: Heston, step: float) -> _Coefficients:
    """Andersen's constants, with the drift left out.

    The prices here are of ``S_t / F_t``, so the risk-neutral drift has already
    been divided out and only the volatility terms remain. ``k0`` carries the
    part of the correlation term that does not multiply a variance.
    """
    rho, sigma, kappa, theta = model.rho, model.sigma, model.kappa, model.theta
    if sigma == 0.0:
        # Deterministic variance: the correlation has nothing to act on, and
        # the ratios below are zero over zero rather than large.
        return _Coefficients(0.0, -0.5 * _GAMMA_1 * step, -0.5 * _GAMMA_2 * step, 0.0, 0.0)
    ratio = rho / sigma
    common = kappa * ratio - 0.5
    return _Coefficients(
        -rho * kappa * theta * step / sigma,
        _GAMMA_1 * step * common - ratio,
        _GAMMA_2 * step * common + ratio,
        _GAMMA_1 * step * (1.0 - rho * rho),
        _GAMMA_2 * step * (1.0 - rho * rho),
    )


def paths(
    model: Heston,
    time: float,
    steps: int,
    uniforms: Sequence[float],
    martingale: bool = True,
) -> tuple[list[float], list[float]]:
    """One path of the log price and the variance, from ``2 * steps`` uniforms.

    The log price is of ``S_t / F_t``, so it starts at zero and is a martingale
    in levels. Multiply by the forward to get a price.

    The uniforms are consumed in pairs, the variance's first. Drawing uniforms
    rather than normals is what makes antithetic sampling work here: the
    exponential proposal reads its uniform directly, and mirroring a path means
    replacing every uniform by its complement, which is the reflection for the
    normal branch and for the price shock as well.

    Args:
        model: The variance process.
        time: Horizon.
        steps: Number of steps.
        uniforms: ``2 * steps`` values in [0, 1).
        martingale: Correct the drift so the simulated forward is exact.

    Returns:
        ``(log_prices, variances)``, each of length ``steps``.

    Raises:
        ValueError: If ``steps`` is not positive or the draw is the wrong length.
        MartingaleCorrectionError: If the correction is not defined at a step.
    """
    if steps < 1:
        raise ValueError(f"steps must be at least 1, got {steps}")
    if len(uniforms) != 2 * steps:
        raise ValueError(f"expected {2 * steps} uniforms, got {len(uniforms)}")

    step = time / steps
    coefficients = _coefficients(model, step)
    argument = coefficients.mgf_argument
    variance = model.v0
    log_price = 0.0
    prices: list[float] = []
    variances: list[float] = []

    for index in range(steps):
        proposal = _fit(model, variance, step)
        next_variance = proposal.draw(uniforms[2 * index])
        if martingale:
            drift = -proposal.log_mgf(argument) - (
                coefficients.k1 + 0.5 * coefficients.k3
            ) * variance
        else:
            drift = coefficients.k0
        spread = coefficients.k3 * variance + coefficients.k4 * next_variance
        shock = norm_ppf(min(max(uniforms[2 * index + 1], 1e-300), 1.0 - 1e-16))
        log_price += (
            drift
            + coefficients.k1 * variance
            + coefficients.k2 * next_variance
            + math.sqrt(max(spread, 0.0)) * shock
        )
        variance = next_variance
        prices.append(log_price)
        variances.append(variance)

    return prices, variances


def _check(contract: Contract, steps: int) -> None:
    if contract.time <= 0.0:
        raise ValueError(f"time must be positive for a simulation, got {contract.time}")
    if contract.spot <= 0.0:
        raise ValueError(f"spot must be positive for a simulation, got {contract.spot}")
    if steps < 1:
        raise ValueError(f"steps must be at least 1, got {steps}")


def _replicate(
    settings: Settings,
    dimension: int,
    payoff: Callable[[Sequence[float]], tuple[float, float]],
) -> tuple[list[float], list[float]]:
    """Draw the uniforms, mirroring each path if asked.

    The mirror of a uniform is its complement. That is the same reflection as
    negating a normal once the quantile function has been applied, and unlike
    negating a normal it also mirrors the exponential proposal's draw, which
    never becomes a normal at all.
    """
    rng = random.Random(settings.seed)
    payoffs: list[float] = []
    controls: list[float] = []
    for _ in range(settings.samples):
        draw = [rng.random() for _ in range(dimension)]
        value, control = payoff(draw)
        if settings.antithetic:
            mirrored_value, mirrored_control = payoff([1.0 - u for u in draw])
            value = (value + mirrored_value) / 2.0
            control = (control + mirrored_control) / 2.0
        payoffs.append(value)
        controls.append(control)
    return payoffs, controls


def european(
    model: Heston,
    contract: Contract,
    option: OptionType = OptionType.CALL,
    steps: int = 64,
    settings: Settings | None = None,
    martingale: bool = True,
) -> Estimate:
    """Price a European option by simulation.

    The transform prices this exactly, which is the point: the two share no
    code, so agreement between them is evidence about both.

    The control variate is the discounted terminal price, whose mean is the
    discounted forward. Under the martingale correction that mean is exact by
    construction, so the control removes the simulation's own drift error along
    with the payoff's variance — and without the correction it removes it too,
    which is why the bias measurements in the tests switch the control off.

    Args:
        model: The variance process.
        contract: The option and its market.
        option: Call or put.
        steps: Time steps per path.
        settings: Paths, seed, antithetic sampling and the control variate.
        martingale: Correct the drift so the simulated forward is exact.

    Returns:
        The estimate, with its standard error.
    """
    _check(contract, steps)
    settings = settings or Settings()
    forward, discount, strike = contract.forward, contract.discount, contract.strike
    time = contract.time

    def payoff(draw: Sequence[float]) -> tuple[float, float]:
        log_prices, _ = paths(model, time, steps, draw, martingale)
        terminal = forward * math.exp(log_prices[-1])
        value = discount * max(option.sign * (terminal - strike), 0.0)
        return value, discount * terminal

    values, controls = _replicate(settings, 2 * steps, payoff)
    return _summarise(
        values,
        controls if settings.control else None,
        discount * forward,
        "discounted terminal price",
        settings.samples * (2 if settings.antithetic else 1),
    )


def asian(
    model: Heston,
    contract: Contract,
    option: OptionType = OptionType.CALL,
    steps: int = 64,
    settings: Settings | None = None,
    martingale: bool = True,
) -> Estimate:
    """Price an arithmetic-average option, averaged over the simulation grid.

    The averaging dates are the step dates, so the number of steps is part of
    the contract here rather than a numerical parameter to be refined away.
    Saying that plainly matters: a caller who doubles the steps to reduce the
    discretisation error has also priced a different option.

    Args:
        model: The variance process.
        contract: The option and its market.
        option: Call or put.
        steps: Averaging dates, equally spaced, ending at expiry.
        settings: Paths, seed and antithetic sampling.
        martingale: Correct the drift so the simulated forward is exact.

    Returns:
        The estimate, with its standard error.
    """
    _check(contract, steps)
    settings = settings or Settings()
    forward, discount, strike = contract.forward, contract.discount, contract.strike
    time = contract.time

    def payoff(draw: Sequence[float]) -> tuple[float, float]:
        log_prices, _ = paths(model, time, steps, draw, martingale)
        average = math.fsum(forward * math.exp(x) for x in log_prices) / steps
        value = discount * max(option.sign * (average - strike), 0.0)
        return value, discount * forward * math.exp(log_prices[-1])

    values, controls = _replicate(settings, 2 * steps, payoff)
    return _summarise(
        values,
        controls if settings.control else None,
        discount * forward,
        "discounted terminal price",
        settings.samples * (2 if settings.antithetic else 1),
    )


def barrier(
    model: Heston,
    contract: Contract,
    option: OptionType,
    kind: Barrier,
    level: float,
    steps: int = 64,
    settings: Settings | None = None,
    martingale: bool = True,
    bridge: bool = True,
) -> Estimate:
    """Price a barrier option, monitored continuously by a Brownian bridge.

    Monitoring only on the grid misses a crossing that happens inside a step
    and returns to the same side of the barrier before the next date, which
    overprices a knock-out. The correction is the usual one: conditional on the
    two endpoints, the probability the path crossed in between has a closed
    form, and the survival indicator becomes the product of one minus that
    probability over the steps.

    Under stochastic volatility that bridge is not exact — the variance is
    moving inside the step too. The step's own variance is used, integrated
    across it with the same weights the price update uses, which is the same
    approximation the price update already makes and no worse. Setting
    ``bridge=False`` gives the discretely monitored contract instead, which is
    a real contract rather than an approximation to this one.

    Args:
        model: The variance process.
        contract: The option and its market.
        option: Call or put.
        kind: Which barrier and on which side the option survives.
        level: The barrier.
        steps: Monitoring dates.
        settings: Paths, seed and antithetic sampling.
        martingale: Correct the drift so the simulated forward is exact.
        bridge: Apply the continuity correction.

    Returns:
        The estimate, with its standard error.

    Raises:
        ValueError: If the barrier is not positive.
    """
    _check(contract, steps)
    if level <= 0.0:
        raise ValueError(f"level must be positive, got {level}")
    settings = settings or Settings()
    forward, discount, strike = contract.forward, contract.discount, contract.strike
    time = contract.time
    step = time / steps
    is_down = kind.is_down
    log_level = math.log(level / forward)

    def payoff(draw: Sequence[float]) -> tuple[float, float]:
        log_prices, variances = paths(model, time, steps, draw, martingale)
        survival = 1.0
        previous_price, previous_variance = 0.0, model.v0
        for price, variance in zip(log_prices, variances, strict=True):
            crossed = price <= log_level if is_down else price >= log_level
            if crossed:
                survival = 0.0
                break
            if bridge:
                local = (_GAMMA_1 * previous_variance + _GAMMA_2 * variance) * step
                if local > 0.0:
                    gap_now = (log_level - price) if is_down else (price - log_level)
                    gap_before = (
                        (log_level - previous_price)
                        if is_down
                        else (previous_price - log_level)
                    )
                    survival *= -math.expm1(-2.0 * gap_now * gap_before / local)
            previous_price, previous_variance = price, variance
        terminal = forward * math.exp(log_prices[-1])
        intrinsic = max(option.sign * (terminal - strike), 0.0)
        alive = survival if kind.is_knock_out else 1.0 - survival
        return discount * alive * intrinsic, discount * terminal

    values, controls = _replicate(settings, 2 * steps, payoff)
    return _summarise(
        values,
        controls if settings.control else None,
        discount * forward,
        "discounted terminal price",
        settings.samples * (2 if settings.antithetic else 1),
    )
