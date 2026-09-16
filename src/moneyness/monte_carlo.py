"""Monte Carlo, for the payoffs that have no closed form.

Three decisions shape this module.

**The simulation is exact, not discretised.** Geometric Brownian motion has
lognormal increments in closed form, so a path can be drawn with exactly the
right joint law at its monitoring dates. Nothing here runs an Euler scheme, and
there is no step-size bias to trade against sample count. That matters most for
the barrier options: a discretely monitored barrier price genuinely differs from
a continuously monitored one, and if the simulation also carried a
discretisation error the two effects would be inseparable. Here the only
discretisation is the monitoring schedule, which is a term of the contract.

**Normals come from inverting the quantile function.** A uniform draw pushed
through :func:`moneyness.normal.norm_ppf`, rather than Box-Muller or the
standard library's ``gauss``. Two reasons. Antithetic sampling becomes exact and
obvious, because ``norm_ppf(1 - u) = -norm_ppf(u)`` identically, where a
transform that caches a second variate makes pairing depend on call order. And
the quantile function is already the most heavily tested function in the
library, accurate into the tails, which is where a rare-event payoff gets its
value.

**The standard error is computed on independent samples, not on paths.** Under
antithetic sampling a path and its mirror are not independent — that is the
entire point of drawing them — so the independent unit is the pair and the
sample is its average. Dividing by the square root of the path count instead
would understate the error by about the square root of two, and would do it
silently, in the direction that flatters the estimate.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum

from .bsm import Inputs, OptionType, price
from .normal import norm_ppf

__all__ = [
    "Barrier",
    "Estimate",
    "Settings",
    "asian",
    "barrier",
    "european",
    "geometric_asian",
]


@dataclass(frozen=True, slots=True)
class Estimate:
    """A Monte Carlo result, and how much of it is noise.

    Attributes:
        value: The estimate.
        standard_error: Standard error of the mean, computed on independent
            samples.
        samples: Number of independent samples. Under antithetic sampling this
            is half the number of paths, because a path and its mirror are one
            sample between them.
        paths: Number of paths simulated.
        control: Name of the control variate used, or ``None``. Reported rather
            than left implicit, because which control was applied changes what
            the standard error means and a caller comparing two runs needs to
            know they used the same one.
    """

    value: float
    standard_error: float
    samples: int
    paths: int
    control: str | None

    def interval(self, level: float = 0.95) -> tuple[float, float]:
        """A confidence interval at the given level.

        Normal rather than Student's t. The sample counts here are in the
        thousands at least, where the two differ in the fourth decimal of the
        multiplier, and far below that the central limit theorem is doing more
        work than the choice of distribution: an option payoff is a truncated
        lognormal, badly skewed, and no quantile choice rescues an interval
        built on a few dozen samples of it.

        Args:
            level: Coverage, strictly between 0 and 1.

        Returns:
            ``(lower, upper)``.

        Raises:
            ValueError: If ``level`` is not strictly between 0 and 1.
        """
        if not 0.0 < level < 1.0:
            raise ValueError(f"level must lie strictly between 0 and 1, got {level}")
        half_width = norm_ppf(1.0 - (1.0 - level) / 2.0) * self.standard_error
        return self.value - half_width, self.value + half_width


@dataclass(frozen=True, slots=True)
class Settings:
    """How to run a simulation.

    Attributes:
        paths: Total paths simulated. Under antithetic sampling this is rounded
            down to an even number, since paths are drawn in mirrored pairs.
        seed: Seed for the generator. Fixed by default, so a result is
            reproducible unless the caller asks otherwise; a silently varying
            answer is a poor default for a library whose other functions are
            deterministic.
        antithetic: Pair each path with its mirror image.
        control: Use a control variate where one is available.

    Raises:
        ValueError: If fewer than two paths are asked for, or fewer than two
            independent samples would remain after pairing — a standard error
            needs at least two.
    """

    paths: int = 100_000
    seed: int = 0
    antithetic: bool = True
    control: bool = True

    def __post_init__(self) -> None:
        if self.paths < 2:
            raise ValueError(f"paths must be at least 2, got {self.paths}")
        if self.samples < 2:
            raise ValueError(
                f"{self.paths} antithetic paths leave {self.samples} independent "
                "samples, which is too few for a standard error"
            )

    @property
    def samples(self) -> int:
        """Independent samples: half the paths when antithetic, all of them otherwise."""
        return self.paths // 2 if self.antithetic else self.paths


class Barrier(str, Enum):
    """Which barrier, and on which side of it the option survives."""

    DOWN_AND_OUT = "down-and-out"
    DOWN_AND_IN = "down-and-in"
    UP_AND_OUT = "up-and-out"
    UP_AND_IN = "up-and-in"

    @property
    def is_down(self) -> bool:
        return self in (Barrier.DOWN_AND_OUT, Barrier.DOWN_AND_IN)

    @property
    def is_knock_out(self) -> bool:
        return self in (Barrier.DOWN_AND_OUT, Barrier.UP_AND_OUT)


def _check(inputs: Inputs) -> None:
    if inputs.time <= 0.0:
        raise ValueError(f"time must be positive for a simulation, got {inputs.time}")
    if inputs.vol <= 0.0:
        raise ValueError(f"vol must be positive for a simulation, got {inputs.vol}")
    if inputs.spot <= 0.0:
        raise ValueError(f"spot must be positive for a simulation, got {inputs.spot}")


def _normal(rng: random.Random) -> float:
    """One standard normal, by inverting the quantile function.

    ``random()`` returns a value in ``[0, 1)``, and the quantile function is
    negatively infinite at the closed end. The redraw costs nothing — the event
    has probability two to the minus fifty-three — and is cheaper to reason
    about than clamping, which would put a spike of finite mass at whatever
    value the clamp chose.
    """
    u = rng.random()
    while u <= 0.0:
        u = rng.random()
    return norm_ppf(u)


def _summarise(
    payoffs: list[float],
    controls: list[float] | None,
    control_mean: float,
    control_name: str | None,
    paths: int,
) -> Estimate:
    """Mean, standard error, and the control-variate adjustment if there is one.

    The adjustment is ``Y = X - beta (C - E[C])`` with ``beta`` the regression
    coefficient of the payoff on the control. Any ``beta`` leaves the estimator
    unbiased, since the correction has zero mean; the optimal one minimises the
    variance of ``Y`` and equals ``Cov(X, C) / Var(C)``.

    Estimating ``beta`` from the same sample it is applied to introduces a bias
    of order one over the sample count, because the fitted coefficient is
    correlated with the sample it was fitted on. It is left in rather than
    removed with a pilot sample. At the sample sizes this module is used at, that
    bias is several orders of magnitude below the standard error being reported,
    while a pilot sample would throw away a fixed fraction of the paths to fix
    something invisible. The trade is stated rather than hidden, and the
    coverage test in the suite is what would catch it if it ever stopped
    holding.
    """
    n = len(payoffs)
    adjusted = payoffs
    used = None
    if controls is not None:
        mean_x = math.fsum(payoffs) / n
        mean_c = math.fsum(controls) / n
        covariance = math.fsum(
            (x - mean_x) * (c - mean_c) for x, c in zip(payoffs, controls, strict=True)
        )
        variance = math.fsum((c - mean_c) ** 2 for c in controls)
        if variance > 0.0:
            beta = covariance / variance
            adjusted = [
                x - beta * (c - control_mean)
                for x, c in zip(payoffs, controls, strict=True)
            ]
            used = control_name

    mean = math.fsum(adjusted) / n
    # The two-pass form. The textbook single-pass shortcut subtracts two large
    # nearly equal sums, and for deep out-of-the-money payoffs -- mostly zeros
    # with rare large values -- it can return a negative variance.
    spread = math.fsum((value - mean) ** 2 for value in adjusted)
    standard_error = math.sqrt(spread / (n - 1) / n) if n > 1 else math.inf
    return Estimate(mean, standard_error, n, paths, used)


def _replicate(
    settings: Settings,
    dimension: int,
    payoff: Callable[[Sequence[float]], tuple[float, float]],
) -> tuple[list[float], list[float]]:
    """Draw the samples, mirroring each one if asked.

    ``payoff`` receives a vector of standard normals and returns the discounted
    payoff together with the value of the control on the same path, so that the
    two are guaranteed to come from one set of random numbers. A control
    computed from a separate draw would be independent of the payoff and would
    reduce no variance at all.
    """
    rng = random.Random(settings.seed)
    payoffs: list[float] = []
    controls: list[float] = []
    for _ in range(settings.samples):
        draw = [_normal(rng) for _ in range(dimension)]
        value, control = payoff(draw)
        if settings.antithetic:
            mirrored_value, mirrored_control = payoff([-z for z in draw])
            value = (value + mirrored_value) / 2.0
            control = (control + mirrored_control) / 2.0
        payoffs.append(value)
        controls.append(control)
    return payoffs, controls


def _terminal(inputs: Inputs, z: float) -> float:
    """``S_T`` from one normal, exactly on the terminal law."""
    drift = (inputs.b - 0.5 * inputs.vol * inputs.vol) * inputs.time
    return inputs.spot * math.exp(drift + inputs.std_dev * z)


def _path(inputs: Inputs, draw: Sequence[float]) -> list[float]:
    """The path at ``len(draw)`` equally spaced dates, exactly on its joint law.

    Each step applies the same exact lognormal increment over ``T / m``, so the
    values are the true process observed on the grid rather than an
    approximation to it. There is no step-size error to trade against sample
    count; a finer grid means a different contract, not a better approximation
    to this one.
    """
    steps = len(draw)
    dt = inputs.time / steps
    drift = (inputs.b - 0.5 * inputs.vol * inputs.vol) * dt
    diffusion = inputs.vol * math.sqrt(dt)
    level = inputs.spot
    out: list[float] = []
    for z in draw:
        level *= math.exp(drift + diffusion * z)
        out.append(level)
    return out


def european(
    inputs: Inputs, option: OptionType, settings: Settings | None = None
) -> Estimate:
    """Price a European option by simulation.

    The closed form is right there in :func:`moneyness.bsm.price`, so this is
    not how one would price a vanilla. It is the calibration of the machinery:
    the answer is known exactly, so the estimate, the standard error and the
    interval can all be checked against something rather than against each
    other.

    The control is the discounted terminal price, whose mean is ``S e^{(b-r)T}``
    exactly. It is strongly correlated with a call payoff and leaves the
    estimator unbiased.

    Args:
        inputs: The option and its market.
        option: Call or put.
        settings: Simulation settings.

    Returns:
        An :class:`Estimate`.

    Raises:
        ValueError: If time, volatility or spot is not positive.
    """
    _check(inputs)
    settings = settings or Settings()
    discount = inputs.discount
    sign = option.sign
    control_mean = inputs.spot * math.exp((inputs.b - inputs.rate) * inputs.time)

    def sample(draw: Sequence[float]) -> tuple[float, float]:
        terminal = _terminal(inputs, draw[0])
        payoff = max(sign * (terminal - inputs.strike), 0.0)
        return discount * payoff, discount * terminal

    payoffs, controls = _replicate(settings, 1, sample)
    return _summarise(
        payoffs,
        controls if settings.control else None,
        control_mean,
        "discounted terminal price",
        settings.samples * (2 if settings.antithetic else 1),
    )


def geometric_asian(inputs: Inputs, option: OptionType, steps: int) -> float:
    """The closed form for a discretely monitored geometric-average Asian option.

    The geometric average of lognormals is itself lognormal, which is the whole
    reason this has a closed form while its arithmetic sibling does not. Writing
    the average over dates ``t_i = iT/m``,

        E[log G] = log S + (b - v^2/2) T (m+1) / (2m)
        Var[log G] = v^2 T (m+1)(2m+1) / (6 m^2)

    the second following from ``sum_i sum_j min(i, j) = m(m+1)(2m+1)/6``.

    Rather than write a second pricing formula, the option is expressed as an
    ordinary Black-Scholes one with an effective volatility and an effective
    cost of carry chosen to reproduce exactly that mean and variance, and handed
    to the pricer the rest of the library already uses. One exercise rule in the
    codebase instead of two that can drift apart — the same reasoning that put
    the American put through the call.

    This is both an oracle for the simulation and, more usefully, the control
    variate for the arithmetic average, which it tracks closely.

    Args:
        inputs: The option and its market.
        option: Call or put.
        steps: Number of equally spaced monitoring dates, at least one, the last
            of them at expiry.

    Returns:
        The price.

    Raises:
        ValueError: If ``steps`` is below one, or time, volatility or spot is
            not positive.
    """
    _check(inputs)
    if steps < 1:
        raise ValueError(f"steps must be at least 1, got {steps}")

    m = float(steps)
    variance = inputs.vol * inputs.vol * inputs.time * (m + 1.0) * (2.0 * m + 1.0) / (6.0 * m * m)
    mean_log = math.log(inputs.spot) + (inputs.b - 0.5 * inputs.vol**2) * inputs.time * (
        m + 1.0
    ) / (2.0 * m)

    # Match the effective forward and variance, then reuse the pricer.
    effective_vol = math.sqrt(variance / inputs.time)
    forward = math.exp(mean_log + variance / 2.0)
    effective_carry = math.log(forward / inputs.spot) / inputs.time
    equivalent = Inputs(
        spot=inputs.spot,
        strike=inputs.strike,
        time=inputs.time,
        rate=inputs.rate,
        vol=effective_vol,
        carry=effective_carry,
    )
    return price(equivalent, option)


def asian(
    inputs: Inputs,
    option: OptionType,
    steps: int,
    settings: Settings | None = None,
    *,
    geometric: bool = False,
) -> Estimate:
    """Price an average-price Asian option by simulation.

    The arithmetic average of lognormals has no tractable distribution, which is
    why this payoff needs simulation at all. Its geometric counterpart does, and
    the two averages are close enough on the same path that the geometric payoff
    makes an unusually effective control variate — far better than the terminal
    price, because it is a function of the whole path in the same way the target
    is.

    Args:
        inputs: The option and its market.
        option: Call or put.
        steps: Monitoring dates, equally spaced, the last at expiry.
        settings: Simulation settings.
        geometric: Simulate the geometric average instead. Only useful as a
            check against :func:`geometric_asian`, since that is exact; the
            control is switched off in this case, because a control perfectly
            correlated with the payoff would collapse the estimator onto the
            closed form and test nothing.

    Returns:
        An :class:`Estimate`.

    Raises:
        ValueError: If ``steps`` is below one, or time, volatility or spot is
            not positive.
    """
    _check(inputs)
    if steps < 1:
        raise ValueError(f"steps must be at least 1, got {steps}")
    settings = settings or Settings()
    discount = inputs.discount
    sign = option.sign
    strike = inputs.strike
    control_mean = geometric_asian(inputs, option, steps)

    def sample(draw: Sequence[float]) -> tuple[float, float]:
        levels = _path(inputs, draw)
        if geometric:
            average = math.exp(math.fsum(math.log(x) for x in levels) / steps)
        else:
            average = math.fsum(levels) / steps
        payoff = discount * max(sign * (average - strike), 0.0)
        geometric_average = math.exp(math.fsum(math.log(x) for x in levels) / steps)
        control = discount * max(sign * (geometric_average - strike), 0.0)
        return payoff, control

    payoffs, controls = _replicate(settings, steps, sample)
    use_control = settings.control and not geometric
    return _summarise(
        payoffs,
        controls if use_control else None,
        control_mean,
        "geometric-average Asian",
        settings.samples * (2 if settings.antithetic else 1),
    )


def barrier(
    inputs: Inputs,
    option: OptionType,
    level: float,
    style: Barrier,
    steps: int,
    settings: Settings | None = None,
    *,
    bridge: bool = True,
) -> Estimate:
    """Price a barrier option by simulation, optionally with the bridge correction.

    A path observed on a grid can dip across the barrier and back between two
    observations, and a simulation that only looks at the grid will miss it.
    The resulting knock-out price is biased *upward*: options are being paid out
    on paths that should have been extinguished. The bias is large and it
    vanishes slowly, like one over the square root of the step count, so
    throwing steps at it is expensive.

    The Brownian bridge fixes most of it analytically. Conditional on its two
    endpoints, a Brownian path between them is a bridge, and the probability
    that a bridge crosses a level has a closed form. For a down barrier at
    ``H``, over a step of length ``dt``,

        P(cross) = exp(-2 log(S_i / H) log(S_{i+1} / H) / (v^2 dt))

    So instead of a path surviving a step or not, it survives with a
    probability, and the payoff is weighted by the product of those
    probabilities over the path. That turns a discrete indicator into a smooth
    function of the path, which is why it removes bias and reduces variance at
    the same time.

    Knock-ins are priced as the vanilla minus the knock-out on the same paths,
    which makes in-out parity hold exactly rather than to within two
    simulations' noise.

    Args:
        inputs: The option and its market.
        option: Call or put.
        level: The barrier. Positive, and on the correct side of spot: a down
            barrier already breached at inception is not a barrier option.
        style: Which barrier, and whether it knocks in or out.
        steps: Monitoring dates, equally spaced, the last at expiry.
        settings: Simulation settings.
        bridge: Apply the Brownian bridge correction, giving the continuously
            monitored price. With it off, the result is the genuinely discretely
            monitored price, which is a different contract and not merely a
            worse estimate of this one.

    Returns:
        An :class:`Estimate`.

    Raises:
        ValueError: If ``steps`` is below one, the barrier is not positive, the
            barrier is on the wrong side of spot, or time, volatility or spot is
            not positive.
    """
    _check(inputs)
    if steps < 1:
        raise ValueError(f"steps must be at least 1, got {steps}")
    if level <= 0.0:
        raise ValueError(f"the barrier must be positive, got {level}")
    if style.is_down and level >= inputs.spot:
        raise ValueError(
            f"a down barrier at {level} is at or above the spot of {inputs.spot}, "
            "so it is already breached"
        )
    if not style.is_down and level <= inputs.spot:
        raise ValueError(
            f"an up barrier at {level} is at or below the spot of {inputs.spot}, "
            "so it is already breached"
        )

    settings = settings or Settings()
    discount = inputs.discount
    sign = option.sign
    strike = inputs.strike
    knock_out = style.is_knock_out
    down = style.is_down
    dt = inputs.time / steps
    bridge_scale = inputs.vol * inputs.vol * dt
    control_mean = price(inputs, option)

    def sample(draw: Sequence[float]) -> tuple[float, float]:
        levels = _path(inputs, draw)
        survival = 1.0
        previous = inputs.spot
        for current in levels:
            if (down and current <= level) or (not down and current >= level):
                survival = 0.0
                break
            if bridge:
                # Both endpoints are on the surviving side here, so both logs
                # share a sign and their product is positive.
                crossing = math.exp(
                    -2.0
                    * math.log(previous / level)
                    * math.log(current / level)
                    / bridge_scale
                )
                survival *= 1.0 - crossing
            previous = current

        terminal = levels[-1]
        vanilla = discount * max(sign * (terminal - strike), 0.0)
        knocked_out = survival * vanilla
        payoff = knocked_out if knock_out else vanilla - knocked_out
        return payoff, vanilla

    payoffs, controls = _replicate(settings, steps, sample)
    return _summarise(
        payoffs,
        controls if settings.control else None,
        control_mean,
        "vanilla payoff on the same paths",
        settings.samples * (2 if settings.antithetic else 1),
    )
