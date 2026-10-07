"""Average-price options, analytically: moments, bounds, and a sharp lower bound.

:func:`~moneyness.monte_carlo.geometric_asian` is exact because the geometric
average of lognormals is lognormal. The arithmetic average is not lognormal, has
no closed form, and until now had only
:func:`~moneyness.monte_carlo.asian` — a simulation — behind it. This module is
the analytic side of the same instrument, and the reason to want it is not speed.

**A simulation cannot bound anything.** It returns an estimate and a standard
error, so it can state that a price is probably near a number and never that it
is certainly above or below one. Two bounds here are inequalities rather than
approximations, each resting on a statement about the payoff that holds path by
path:

* **Lower**, from AM-GM. The geometric average never exceeds the arithmetic one,
  so ``max(G - K, 0) <= max(A - K, 0)`` on every path and the geometric price is
  below the arithmetic one. Cheap, exact as an inequality, and loose.
* **Upper**, from convexity. ``A - K`` is an average of the ``S_i - K``, so
  ``max(A - K, 0) <= (1/m) sum_i max(S_i - K, 0)``: the Asian call is below the
  average of ordinary calls struck at ``K`` and expiring on the monitoring dates.

Between them sits something much better. **Curran's bound conditions on the
geometric average** instead of discarding it. Given ``log G``, every ``log S_i``
is normal with a known conditional mean, so ``E[A | G]`` is available in closed
form and increasing in ``G``; exercising when ``E[A | G] > K`` is *a* strategy,
so its value is a lower bound, and because the two averages track each other
closely it is a very tight one. One root solve and a sum.

**The moments are closed form and they do real work.** Under geometric Brownian
motion ``E[S_i S_j] = S^2 exp(b (t_i + t_j) + v^2 min(t_i, t_j))``, so the first
two moments of ``A`` are finite sums. They give put-call parity for the average
exactly, they give the moment-matched price below, and they give an oracle the
simulation has to reproduce to within its own standard error — which is the only
check on that simulation that does not go through another approximation.

**Moment matching is the approximation, and its error changes sign.** Fitting a
lognormal to those two moments — Turnbull and Wakeman's method — is exact only in
the limit of zero volatility. At the money it reads **high**, by 6.1e-04, 1.4e-03,
3.0e-03, 8.5e-03 and 3.0e-02 relative as the volatility goes 5%, 10%, 20%, 40% and
80% over a year: two to three times worse per doubling, not the order of magnitude
one might guess. Out of the money it reads **low**, and far enough low to break the
lower bound: a call struck at 120 against a spot of 100 comes out **9.2% below
Curran's bound** at 10% volatility, 3.4% below at 20% and 0.79% below at 40% —
and then 2.7% *above* it at 80%, so the error changes sign across the volatility
as well as across the strike.

Those two sign changes are the reason the bounds are not decoration. No single
tolerance describes an error that is positive in one part of the strike range and
negative in another, so there is no honest way to quote moment matching with an
accuracy figure attached — but :func:`price_bounds` will say when it has left the
interval the price is known to lie in. Curran's bound, by contrast, is low by
6.0e-05 at 20% volatility, 3.2e-04 at 40% and 1.5e-03 at 80%, measured against two
million paths, and is never high, because it is a bound. At 10% volatility the
shortfall sits inside that run's own standard error, so it is reported as
unresolved rather than as a number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .bsm import Inputs, OptionType, price
from .monte_carlo import geometric_asian
from .normal import norm_cdf

__all__ = [
    "AsianBounds",
    "AverageMoments",
    "average_moments",
    "curran",
    "monitoring_times",
    "parity_difference",
    "price_bounds",
    "turnbull_wakeman",
]


def _check(inputs: Inputs, steps: int) -> None:
    if inputs.time <= 0.0:
        raise ValueError(f"time must be positive for an average, got {inputs.time}")
    if inputs.spot <= 0.0:
        raise ValueError(f"spot must be positive, got {inputs.spot}")
    if inputs.vol <= 0.0:
        raise ValueError(
            f"vol must be positive, got {inputs.vol}. A zero-volatility average is "
            "deterministic and worth its intrinsic value, which needs no averaging "
            "machinery to compute."
        )
    if steps < 1:
        raise ValueError(f"steps must be at least 1, got {steps}")


def monitoring_times(time: float, steps: int) -> tuple[float, ...]:
    """The ``steps`` equally spaced monitoring dates, the last one at expiry.

    Stated once, here, because every formula in this module and in
    :func:`~moneyness.monte_carlo.asian` has to agree about them. Averaging over
    ``i T / m`` for ``i = 1 .. m`` excludes the spot and includes the expiry,
    which is the convention a term sheet means by "the average of the closing
    prices on the last ``m`` business days" and the convention
    :func:`~moneyness.monte_carlo.geometric_asian` already uses.
    """
    if time <= 0.0:
        raise ValueError(f"time must be positive, got {time}")
    if steps < 1:
        raise ValueError(f"steps must be at least 1, got {steps}")
    step = time / steps
    return tuple(step * (index + 1) for index in range(steps))


@dataclass(frozen=True, slots=True)
class AverageMoments:
    """The first two moments of the arithmetic average, and what they imply.

    Attributes:
        first: ``E[A]``, which is the average's forward price.
        second: ``E[A**2]``.
        log_variance: ``log(second / first**2)``, the variance of the log of a
            lognormal matched to these two moments. Non-negative, and zero only
            where the average is deterministic.
    """

    first: float
    second: float
    log_variance: float

    @property
    def variance(self) -> float:
        """``Var[A]``, in price units squared."""
        return self.second - self.first * self.first

    @property
    def effective_vol(self) -> float:
        """The matched lognormal's volatility, given a time to work over.

        Returned per unit of the *averaging* period rather than annualised,
        because the caller holds the time. :func:`turnbull_wakeman` divides.
        """
        return math.sqrt(self.log_variance)


def average_moments(inputs: Inputs, steps: int) -> AverageMoments:
    """Exact first and second moments of the discretely monitored arithmetic average.

    Under geometric Brownian motion with cost of carry ``b`` and volatility
    ``v``, ``E[S_t] = S exp(b t)`` and
    ``E[S_s S_t] = S**2 exp(b (s + t) + v**2 min(s, t))``, so with
    ``A = (1/m) sum_i S_{t_i}``::

        E[A]    = (S / m)    sum_i exp(b t_i)
        E[A**2] = (S**2/m**2) sum_i sum_j exp(b (t_i + t_j) + v**2 min(t_i, t_j))

    Both are finite sums with no approximation in them. The double sum is
    computed over the upper triangle and doubled, which halves the work and —
    more to the point — makes the symmetry of ``min`` structural rather than
    something the loop has to get right twice.

    Args:
        inputs: The option and its market. The strike is not used.
        steps: Number of equally spaced monitoring dates, at least one.

    Returns:
        The moments, with the matched lognormal's log-variance.

    Raises:
        ValueError: If time, spot or volatility is not positive, or ``steps`` is
            below one.
    """
    _check(inputs, steps)
    times = monitoring_times(inputs.time, steps)
    carry, variance = inputs.b, inputs.vol * inputs.vol
    count = float(steps)

    first = inputs.spot * math.fsum(math.exp(carry * one) for one in times) / count
    diagonal = math.fsum(
        math.exp((2.0 * carry + variance) * one) for one in times
    )
    upper = math.fsum(
        math.exp(carry * (times[i] + times[j]) + variance * times[i])
        for i in range(steps)
        for j in range(i + 1, steps)
    )
    second = inputs.spot**2 * (diagonal + 2.0 * upper) / (count * count)
    # Clamped at zero: the ratio is above one by Jensen, and only floating point
    # can put it below, which it can at a volatility small enough that the
    # average is deterministic to machine precision.
    log_variance = max(0.0, math.log(second / (first * first)))
    return AverageMoments(first=first, second=second, log_variance=log_variance)


def turnbull_wakeman(inputs: Inputs, option: OptionType, steps: int) -> float:
    """Moment-matched lognormal price for an arithmetic-average option.

    Fit a lognormal to the exact first two moments of the average and price it as
    an ordinary option on that lognormal. The fit is handed to the Black-Scholes
    pricer already in the package with an effective volatility and an effective
    cost of carry, rather than written out again — the same choice
    :func:`~moneyness.monte_carlo.geometric_asian` makes, and for the same reason:
    one exercise rule in the codebase instead of two that can drift apart.

    **This is an approximation, and it is not always inside the bounds.** The
    arithmetic average is a sum of lognormals, whose distribution is not lognormal
    and becomes less so as the volatility rises. Measured against Curran's bound,
    which is within 6e-05 relative at the money at 20% volatility:

    * **at the money it reads high** — 6.1e-04, 1.4e-03, 3.0e-03, 8.5e-03 and
      3.0e-02 relative at 5%, 10%, 20%, 40% and 80% volatility over a year, so two
      to three times worse per doubling;
    * **out of the money it reads low**, and breaks the lower bound doing it. A
      call struck at 120 against a spot of 100 is 9.2% below Curran's bound at 10%
      volatility, 3.4% below at 20%, 1.8% below at 30% and 0.79% below at 40% —
      then 0.9% *above* it at 60% and 2.7% above at 80%. The violation shrinks with
      volatility and crosses over rather than easing off, so the trend at moderate
      volatility points the wrong way about what happens next.

    An error that changes sign across the strike and again across the volatility
    cannot be quoted as a tolerance, which is why none is offered. Use :func:`curran` when
    accuracy matters and :func:`price_bounds` when an inequality does; this
    function is here because moment matching is what the literature and most
    trading systems use, and a library that cannot reproduce it cannot be compared
    against them.

    Args:
        inputs: The option and its market.
        option: Call or put.
        steps: Number of equally spaced monitoring dates.

    Returns:
        The price.

    Raises:
        ValueError: If time, spot or volatility is not positive, or ``steps`` is
            below one.
    """
    moments = average_moments(inputs, steps)
    effective_vol = moments.effective_vol / math.sqrt(inputs.time)
    effective_carry = math.log(moments.first / inputs.spot) / inputs.time
    return price(
        Inputs(
            spot=inputs.spot,
            strike=inputs.strike,
            time=inputs.time,
            rate=inputs.rate,
            vol=effective_vol,
            carry=effective_carry,
        ),
        option,
    )


def _log_average_moments(
    inputs: Inputs, steps: int
) -> tuple[tuple[float, ...], float, float, tuple[float, ...]]:
    """Means of ``log S_i``, the mean and standard deviation of ``log G``, and
    the covariances between them. The whole of what Curran's bound needs."""
    times = monitoring_times(inputs.time, steps)
    variance = inputs.vol * inputs.vol
    drift = inputs.b - 0.5 * variance
    log_spot = math.log(inputs.spot)
    means = tuple(log_spot + drift * one for one in times)
    count = float(steps)
    geometric_mean = log_spot + drift * math.fsum(times) / count
    # Var[log G] = (v**2 / m**2) sum_i sum_j min(t_i, t_j), the double sum done
    # over the upper triangle for the same reason as in average_moments.
    diagonal = math.fsum(times)
    upper = math.fsum(times[i] for i in range(steps) for _ in range(i + 1, steps))
    geometric_variance = variance * (diagonal + 2.0 * upper) / (count * count)
    covariances = tuple(
        variance * math.fsum(min(times[i], one) for one in times) / count
        for i in range(steps)
    )
    return means, geometric_mean, math.sqrt(geometric_variance), covariances


def curran(inputs: Inputs, option: OptionType, steps: int) -> float:
    """Curran's lower bound: condition on the geometric average rather than discard it.

    Exercising when ``E[A | G] > K`` is a strategy a holder could follow, so its
    value is below the value of exercising when ``A > K``, which is the option. It
    is therefore a genuine lower bound and not an approximation that happens to
    sit low — and because the two averages move together it is a very tight one.

    The mechanics. ``(log S_1, ..., log S_m, log G)`` is jointly normal, so
    ``E[S_i | log G = y] = exp(mu_i + c_i (y - mu_G) / s_G**2 + (sigma_i**2 -
    c_i**2 / s_G**2) / 2)``, every term of which is known. ``E[A | log G = y]`` is
    increasing in ``y``, so there is one threshold ``y*`` where it crosses the
    strike, found by bisection on a bracket widened until it contains the root.
    The payoff's expectation above that threshold is then the usual pair of
    normal tail integrals::

        E[S_i 1{log G > y*}] = S exp(b t_i) Phi((mu_G + c_i - y*) / s_G)

    The put comes from parity rather than from a second derivation. The bound's
    error is the same for both — parity is exact — so subtracting a known forward
    carries a lower bound for the call to a lower bound for the put.

    Args:
        inputs: The option and its market.
        option: Call or put.
        steps: Number of equally spaced monitoring dates.

    Returns:
        A lower bound on the price, tight enough to be used as one.

    Raises:
        ValueError: If time, spot or volatility is not positive, or ``steps`` is
            below one.
    """
    _check(inputs, steps)
    times = monitoring_times(inputs.time, steps)
    means, geometric_mean, geometric_sd, covariances = _log_average_moments(
        inputs, steps
    )
    variance = inputs.vol * inputs.vol
    count = float(steps)
    discount = math.exp(-inputs.rate * inputs.time)
    moments = average_moments(inputs, steps)

    def conditional_average(level: float) -> float:
        total = 0.0
        for index, one in enumerate(times):
            beta = covariances[index] / (geometric_sd * geometric_sd)
            residual = variance * one - beta * covariances[index]
            total += math.exp(
                means[index] + beta * (level - geometric_mean) + 0.5 * residual
            )
        return total / count

    if inputs.strike <= 0.0:
        # Nothing to solve: a call on a non-negative average struck at zero is the
        # average itself, and a put is worthless.
        forward = discount * moments.first
        return forward if option is OptionType.CALL else 0.0

    # Widen a bracket rather than assume one. The conditional average is
    # increasing and spans (0, inf), so a bracket always exists, but how far out
    # it sits depends on the strike and the volatility and guessing costs
    # accuracy where it is wrong.
    low, high = geometric_mean - geometric_sd, geometric_mean + geometric_sd
    for _ in range(200):
        if conditional_average(low) <= inputs.strike:
            break
        low -= geometric_sd
    else:  # pragma: no cover - 200 standard deviations is unreachable
        raise ValueError("could not bracket the exercise threshold from below")
    for _ in range(200):
        if conditional_average(high) >= inputs.strike:
            break
        high += geometric_sd
    else:  # pragma: no cover - unreachable for the same reason
        raise ValueError("could not bracket the exercise threshold from above")
    for _ in range(200):
        middle = 0.5 * (low + high)
        if conditional_average(middle) < inputs.strike:
            low = middle
        else:
            high = middle
    threshold = 0.5 * (low + high)

    tail = norm_cdf((geometric_mean - threshold) / geometric_sd)
    weighted = math.fsum(
        inputs.spot
        * math.exp(inputs.b * times[index])
        * norm_cdf(
            (geometric_mean + covariances[index] - threshold) / geometric_sd
        )
        for index in range(steps)
    )
    call = discount * (weighted / count - inputs.strike * tail)
    if option is OptionType.CALL:
        return call
    return call - parity_difference(inputs, steps)


def parity_difference(inputs: Inputs, steps: int) -> float:
    """``call - put`` for the average, which is exact and needs no model of ``A``.

    ``(A - K) = max(A - K, 0) - max(K - A, 0)`` path by path, so the difference of
    the two prices is the discounted expectation of ``A - K`` — and that only
    needs the *first* moment, which is a finite sum of forwards. No distributional
    assumption enters, which is why this holds for the simulation, for Curran's
    bound and for the moment-matched price alike, and why it catches a sign or
    convention error in any of them.

    **Not the same kind of quantity as** :func:`~moneyness.bsm.parity_gap`, which
    the names invite confusing. That one returns a *residual* — the amount by
    which a computed call and put fail the identity — and is zero up to rounding
    for any correct pair. This one returns the difference itself, which is a
    price and is not zero. Hence the different name rather than an overload.
    """
    moments = average_moments(inputs, steps)
    return math.exp(-inputs.rate * inputs.time) * (moments.first - inputs.strike)


@dataclass(frozen=True, slots=True)
class AsianBounds:
    """Prices that bracket an arithmetic-average option as inequalities.

    Attributes:
        lower: The best available lower bound — Curran's, which is tight.
        upper: The tightest rigorous upper bound available. Convexity gives one
            for either payoff; for a put the geometric price gives a second, and
            the smaller of the two is reported.
        geometric: The geometric-average price. **Which side of the price this
            falls on depends on the payoff**, and getting it backwards is the
            easiest mistake here. AM-GM says ``G <= A`` path by path, so
            ``max(G - K, 0) <= max(A - K, 0)`` and the geometric *call* is below
            the arithmetic one — while ``max(K - G, 0) >= max(K - A, 0)``, so the
            geometric *put* is above it. Measured at 100 spot, 20% volatility and
            twelve monthly fixings: the geometric call is 5.9402 against a true
            6.1571, below it, and the geometric put is 3.6517 against 3.5355,
            above it. Kept as its own field because for a call it is the bound
            Curran's has to beat, and the margin says whether conditioning bought
            anything.
    """

    lower: float
    upper: float
    geometric: float

    @property
    def width(self) -> float:
        """``upper - lower``, the interval the price is known to lie in."""
        return self.upper - self.lower

    @property
    def midpoint(self) -> float:
        """The middle of the interval, which is not a price and is not claimed to be."""
        return 0.5 * (self.lower + self.upper)

    def brackets(self, candidate: float, tolerance: float = 0.0) -> bool:
        """Whether ``candidate`` lies inside the interval, within ``tolerance``."""
        return self.lower - tolerance <= candidate <= self.upper + tolerance


def price_bounds(inputs: Inputs, option: OptionType, steps: int) -> AsianBounds:
    """Rigorous bounds on an arithmetic-average option.

    The upper bound is convexity. ``A - K`` is the average of ``S_i - K``, and
    the positive part of an average is at most the average of the positive parts,
    so an Asian call is worth at most the average of ordinary calls struck at
    ``K`` and expiring on the monitoring dates — each discounted from its own
    expiry to the Asian's, since the Asian pays later than all but the last of
    them. The same argument gives the put, with no change of sign anywhere, which
    is worth noticing: the inequality is about the positive part and not about the
    direction of the payoff.

    The lower bound is Curran's. :attr:`AsianBounds.geometric` carries the AM-GM
    bound alongside it, and that one changes sides with the payoff: below an
    arithmetic call, above an arithmetic put. So for a put there are two upper
    bounds available and the smaller is taken, which is usually the geometric one
    by a wide margin — convexity is loose because it allows the ``S_i`` to be
    independent, and averaging destroys far more variance than that.

    Args:
        inputs: The option and its market.
        option: Call or put.
        steps: Number of equally spaced monitoring dates.

    Returns:
        The bounds.

    Raises:
        ValueError: If time, spot or volatility is not positive, or ``steps`` is
            below one.
    """
    _check(inputs, steps)
    times = monitoring_times(inputs.time, steps)
    upper = (
        math.fsum(
            math.exp(-inputs.rate * (inputs.time - one))
            * price(
                Inputs(
                    spot=inputs.spot,
                    strike=inputs.strike,
                    time=one,
                    rate=inputs.rate,
                    vol=inputs.vol,
                    carry=inputs.b,
                ),
                option,
            )
            for one in times
        )
        / float(steps)
    )
    geometric = geometric_asian(inputs, option, steps)
    return AsianBounds(
        lower=curran(inputs, option, steps),
        upper=upper if option is OptionType.CALL else min(upper, geometric),
        geometric=geometric,
    )
