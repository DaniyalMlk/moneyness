"""A single maturity of the volatility surface, and the arbitrage it can hide.

Everything here is in *total implied variance* against *log-moneyness on the
forward*:

    k = log(K / F)        w(k) = sigma_BS(k)^2 T

The choice is the same one the implied-volatility solver makes, for the same
reasons, and it pays off twice over at the level of a whole slice. Carry and
discounting are gone, so a slice says nothing about which of the four market
conventions produced it. And the two no-arbitrage conditions a slice has to
satisfy — that it implies a non-negative probability density, and that it does
not cross a slice at another maturity — are awkward statements about prices and
clean statements about ``w``.

The parametrisation is Gatheral's raw SVI:

    w(k) = a + b (rho (k - m) + sqrt((k - m)^2 + s^2))

Five parameters, and each does one recognisable thing. ``a`` sets the level.
``b`` sets the angle between the two asymptotes, which is to say the overall
steepness of the smile. ``rho`` tilts it, and is what a skew looks like in this
coordinate system. ``m`` translates it. ``s`` rounds off the vertex; at ``s = 0``
the slice is two straight lines meeting in a corner.

That the wings are asymptotically linear in ``k`` is not a convenience of the
functional form, it is the right shape: Lee's moment formula puts a hard linear
bound on how fast implied variance may grow in the wings, and a parametrisation
which curved away instead would be claiming something about the tails of the
underlying's distribution that no real market supports.

Two derivatives are available in closed form, which matters because both the
butterfly condition below and the Dupire identity in :mod:`moneyness.surface`
are written in terms of them. Nothing in this module differentiates anything
numerically; the tests do that, and check the analytic forms against it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .optimise import cholesky_solve, nelder_mead, project_onto_cone

__all__ = [
    "SVI",
    "Butterfly",
    "Fit",
    "calibrate",
    "density",
    "durrleman",
]

# The wings are scanned this far out in log-moneyness when looking for a
# butterfly violation. At k = +-5 the strike is about 150 times the forward or
# about a hundred and fiftieth of it; a density defect further out than that is
# not something any quote in the fit had an opinion about.
_WING = 5.0


@dataclass(frozen=True, slots=True)
class SVI:
    """A raw SVI slice: one maturity of the surface.

    Attributes:
        a: Level. Shifts the whole slice vertically.
        b: Wing steepness. Non-negative; the asymptotic slopes are
            ``b(1 - rho)`` on the left and ``b(1 + rho)`` on the right.
        rho: Tilt, strictly between -1 and 1. Negative is the usual equity
            shape, where downside strikes carry the higher variance.
        m: Horizontal translation of the vertex.
        s: Vertex curvature, strictly positive. As it goes to zero the slice
            becomes two straight lines meeting at a point.

    Raises:
        ValueError: If the parameters are outside the domain, or if they place
            any part of the slice at negative total variance. The second check
            is the important one: the first four constraints are individually
            natural, but it is their combination that decides whether the slice
            represents a possible market, and a negative total variance is not
            a mispriced option, it is a meaningless one.
    """

    a: float
    b: float
    rho: float
    m: float
    s: float

    def __post_init__(self) -> None:
        for name in ("a", "b", "rho", "m", "s"):
            value = getattr(self, name)
            if value != value or math.isinf(value):
                raise ValueError(f"{name} must be finite, got {value}")
        if self.b < 0.0:
            raise ValueError(f"b must be non-negative, got {self.b}")
        if not -1.0 < self.rho < 1.0:
            raise ValueError(f"rho must lie strictly inside (-1, 1), got {self.rho}")
        if self.s <= 0.0:
            raise ValueError(f"s must be positive, got {self.s}")
        # The minimum of the slice, which is attained where the derivative
        # vanishes, and is the only place non-negativity can fail.
        floor = self.a + self.b * self.s * math.sqrt(1.0 - self.rho * self.rho)
        if floor < 0.0:
            raise ValueError(
                f"the slice reaches a total variance of {floor}, which is negative; "
                "a + b s sqrt(1 - rho^2) must be non-negative"
            )

    def total_variance(self, k: float) -> float:
        """Total implied variance ``w(k)`` at log-moneyness ``k``."""
        y = k - self.m
        return self.a + self.b * (self.rho * y + math.hypot(y, self.s))

    def d_total_variance(self, k: float) -> float:
        """``dw/dk``.

        Bounded between ``-b(1 - rho)`` and ``b(1 + rho)``, the two asymptotic
        slopes, and monotonically increasing between them.
        """
        y = k - self.m
        return self.b * (self.rho + y / math.hypot(y, self.s))

    def d2_total_variance(self, k: float) -> float:
        """``d2w/dk2``.

        Strictly positive wherever ``b`` is, so a slice is always convex in
        log-moneyness. Convexity is necessary for a slice to be free of
        butterfly arbitrage but nowhere near sufficient, which is the whole
        reason :func:`durrleman` exists.

        Written as ``b (s/r)^2 / r`` rather than the algebraically identical
        ``b s^2 / r^3``. The two differ in floating point at the extremes of the
        domain, and only one of them survives: a very sharp vertex makes ``r``
        small, and cubing a small number underflows to zero long before the
        ratio itself is in any trouble, turning a finite answer into a division
        by zero. Grouping the division as a squared ratio keeps every
        intermediate inside the representable range, since ``s <= r`` bounds the
        ratio by one.
        """
        y = k - self.m
        r = math.hypot(y, self.s)
        ratio = self.s / r
        return self.b * ratio * ratio / r

    def volatility(self, k: float, time: float) -> float:
        """The Black-Scholes volatility the slice implies at maturity ``time``."""
        if time <= 0.0:
            raise ValueError(f"time must be positive, got {time}")
        return math.sqrt(self.total_variance(k) / time)

    @property
    def minimum(self) -> float:
        """The smallest total variance on the slice."""
        return self.a + self.b * self.s * math.sqrt(1.0 - self.rho * self.rho)

    @property
    def wing_slopes(self) -> tuple[float, float]:
        """Asymptotic slopes ``(left, right)`` of ``w`` in ``k``.

        Lee's moment formula caps each at 2. A slope above that implies the
        underlying has no moment of the corresponding order, which is a strong
        claim to have made by accident while fitting five parameters.
        """
        return -self.b * (1.0 - self.rho), self.b * (1.0 + self.rho)


def durrleman(slice_: SVI, k: float) -> float:
    """Durrleman's function ``g(k)``, which is negative exactly where the slice implies
    a negative probability.

    Write ``w`` for the total variance at ``k`` and ``w'``, ``w''`` for its
    derivatives. Then

        g = (1 - k w' / (2w))^2 - (w'^2 / 4)(1/w + 1/4) + w'' / 2

    and the risk-neutral density of log-price implied by the slice is ``g``
    times a strictly positive factor. So ``g >= 0`` everywhere is precisely the
    statement that the slice is free of butterfly arbitrage, and where ``g`` is
    negative the surface is asserting that some range of outcomes has negative
    probability.

    Returning the number rather than a verdict is deliberate. A caller fitting a
    surface wants to know where the density fails and by how much, because that
    says which quotes to distrust; a boolean throws exactly that away. The same
    quantity reappears as the denominator of the Dupire identity in
    :mod:`moneyness.surface`, which is not a coincidence and is asserted as a
    test rather than left as a remark here.

    Args:
        slice_: The slice to evaluate.
        k: Log-moneyness.

    Returns:
        ``g(k)``. Non-negative everywhere iff the slice admits no butterfly
        arbitrage.

    Raises:
        ValueError: If the total variance vanishes at ``k``, where ``g`` has a
            pole and the question has no answer.
    """
    w = slice_.total_variance(k)
    if w <= 0.0:
        raise ValueError(
            f"total variance is {w} at k={k}; Durrleman's function is undefined there"
        )
    dw = slice_.d_total_variance(k)
    d2w = slice_.d2_total_variance(k)
    first = 1.0 - k * dw / (2.0 * w)
    return first * first - (dw * dw / 4.0) * (1.0 / w + 0.25) + d2w / 2.0


def density(slice_: SVI, k: float) -> float:
    """The risk-neutral density of terminal log-moneyness implied by the slice.

    Differentiating the call price twice in strike gives the density of the
    underlying at expiry — the Breeden-Litzenberger identity — and carrying
    that through the Black-Scholes formula with ``w`` a function of ``k``
    rather than a constant produces

        p(k) = g(k) / sqrt(2 pi w(k)) exp(-d_minus(k)^2 / 2)

    with ``d_minus = -k / sqrt(w) - sqrt(w) / 2`` and ``g`` the Durrleman
    function above. Everything except ``g`` is strictly positive, which is what
    makes :func:`durrleman` the butterfly condition rather than merely
    correlated with it.

    Two things follow, and both are checked in the tests rather than asserted
    here. The density integrates to one over the whole line, for any admissible
    slice, which is a demanding check on the analytic derivatives because an
    error in ``w'`` or ``w''`` shows up directly in the mass. And on a flat
    slice it collapses to the lognormal density in log-moneyness, where ``g``
    is identically one.

    Args:
        slice_: The slice.
        k: Log-moneyness.

    Returns:
        The density at ``k``. Negative exactly where the slice admits butterfly
        arbitrage, which is not a possible density and is the point.

    Raises:
        ValueError: If the total variance vanishes at ``k``.
    """
    w = slice_.total_variance(k)
    if w <= 0.0:
        raise ValueError(f"total variance is {w} at k={k}; the density is undefined there")
    root = math.sqrt(w)
    d_minus = -k / root - root / 2.0
    return durrleman(slice_, k) * math.exp(-d_minus * d_minus / 2.0) / math.sqrt(2.0 * math.pi * w)


@dataclass(frozen=True, slots=True)
class Butterfly:
    """The worst butterfly violation found on a slice.

    Attributes:
        worst: The smallest value of Durrleman's function on the scan.
        at: Where it was attained.
        free: True if the slice showed no violation.
    """

    worst: float
    at: float
    free: bool

    @classmethod
    def scan(cls, slice_: SVI, *, wing: float = _WING, points: int = 601) -> Butterfly:
        """Look for a butterfly violation across the wings.

        A scan rather than a solve. Durrleman's function for raw SVI is a
        ratio of polynomials in ``k`` and the square root, and its minimum can
        be found exactly with enough algebra, but the exact root is fragile in
        the cases that matter — a slice only barely admissible has a ``g`` that
        grazes zero, where a closed-form root is dominated by cancellation. A
        dense scan degrades gracefully instead: it can understate a violation
        confined between two grid points, but it never reports one that is not
        there, and the quantity it returns is directly interpretable.

        Args:
            slice_: The slice to check.
            wing: Half-width in log-moneyness.
            points: Grid resolution.

        Returns:
            A :class:`Butterfly`.

        Raises:
            ValueError: If ``wing`` is not positive or ``points`` is below 3.
        """
        if wing <= 0.0:
            raise ValueError(f"wing must be positive, got {wing}")
        if points < 3:
            raise ValueError(f"points must be at least 3, got {points}")
        step = 2.0 * wing / (points - 1)
        worst, at = math.inf, 0.0
        for i in range(points):
            k = -wing + i * step
            g = durrleman(slice_, k)
            if g < worst:
                worst, at = g, k
        return cls(worst, at, worst >= 0.0)


@dataclass(frozen=True, slots=True)
class Fit:
    """A calibrated slice, with the evidence for how well it fits.

    Attributes:
        slice_: The fitted slice.
        rmse: Root mean squared error in total variance across the quotes.
        max_error: Largest absolute error in total variance at any quote.
        iterations: Outer simplex iterations used.
        converged: Whether the outer search converged rather than exhausting
            its iteration budget.
    """

    slice_: SVI
    rmse: float
    max_error: float
    iterations: int
    converged: bool


def _linear_fit(
    log_moneyness: Sequence[float],
    variance: Sequence[float],
    weight: Sequence[float],
    m: float,
    s: float,
) -> tuple[float, float, float] | None:
    """Best ``(a, d, c)`` for a fixed vertex, where ``w = a + d y + c sqrt(y^2 + 1)``.

    This substitution is the reason calibration is tractable. Writing
    ``y = (k - m) / s`` and collecting terms, raw SVI becomes

        w = a + (rho b s) y + (b s) sqrt(y^2 + 1)

    which is *linear* in the three coefficients ``a``, ``d = rho b s`` and
    ``c = b s``. So for any fixed vertex ``(m, s)`` the best remaining three
    parameters come from a least-squares problem with a closed-form solution,
    and only the two vertex parameters are left to search over. A five-parameter
    nonlinear fit becomes a two-parameter one wrapped around a linear solve,
    which is both far faster and far less inclined to find a local minimum.

    The constraints come along cleanly too. Requiring ``c >= |d|`` is exactly
    ``b >= 0`` together with ``|rho| <= 1``, and it also delivers non-negative
    total variance whenever ``a >= 0``: since ``|d y| <= c |y| <= c sqrt(y^2+1)``,
    the two variable terms cannot drag the slice below ``a``. So the feasible
    set is a half-space crossed with a second-order cone, and the projection
    onto it is the one in :mod:`moneyness.optimise`.

    The unconstrained solution is taken first and returned if it is already
    feasible, which it usually is on well-behaved quotes. Otherwise a few
    hundred projected gradient steps run on the same objective, which converges
    because the objective is a convex quadratic and the feasible set is convex.
    """
    rows = [(1.0, y, math.hypot(y, 1.0)) for y in ((k - m) / s for k in log_moneyness)]

    gram = [[0.0] * 3 for _ in range(3)]
    rhs = [0.0] * 3
    for row, target, wt in zip(rows, variance, weight, strict=True):
        for i in range(3):
            rhs[i] += wt * row[i] * target
            for j in range(3):
                gram[i][j] += wt * row[i] * row[j]

    try:
        a, d, c = cholesky_solve(gram, rhs)
    except ValueError:
        return None

    if a >= 0.0 and c >= abs(d):
        return a, d, c

    # Infeasible. Project, then descend on the convex quadratic while staying
    # feasible. The step is scaled by the largest Gram diagonal, which bounds
    # the curvature of the objective and so keeps the step below the point
    # where gradient descent on a quadratic diverges.
    scale = max(gram[i][i] for i in range(3))
    if scale <= 0.0:
        return None
    step = 1.0 / scale

    a = max(a, 0.0)
    d, c = project_onto_cone(d, c)
    for _ in range(500):
        grad = [
            sum(gram[i][j] * coefficient for j, coefficient in enumerate((a, d, c))) - rhs[i]
            for i in range(3)
        ]
        a_next = max(a - step * grad[0], 0.0)
        d_next, c_next = project_onto_cone(d - step * grad[1], c - step * grad[2])
        moved = abs(a_next - a) + abs(d_next - d) + abs(c_next - c)
        a, d, c = a_next, d_next, c_next
        if moved < 1e-14:
            break
    return a, d, c


def calibrate(
    log_moneyness: Sequence[float],
    total_variance: Sequence[float],
    *,
    weight: Sequence[float] | None = None,
    restarts: int = 4,
) -> Fit:
    """Fit a raw SVI slice to quoted total variances.

    The search is two-dimensional, over the vertex ``(m, s)`` only; the other
    three parameters are solved exactly at each trial vertex by
    :func:`_linear_fit`. The outer search is Nelder-Mead, restarted from several
    vertices, because the reduced objective is well behaved but not convex and a
    single start can settle into a shoulder of the smile rather than its centre.

    ``s`` is searched in logarithms. It is a scale parameter, strictly positive
    and spanning orders of magnitude between a sharply peaked short-dated smile
    and a nearly flat long-dated one, so a search that steps in ``s`` itself
    would take steps that are enormous at one end and invisible at the other,
    and would have to be told separately not to go negative.

    Args:
        log_moneyness: Quote abscissae, ``log(K / F)``.
        total_variance: Quoted total variances, one per abscissa. Positive.
        weight: Optional weights, one per quote. Uniform if omitted. Quotes in
            the wings are the least reliable and the most influential on ``b``
            and ``rho``, so downweighting them is common and is left to the
            caller rather than guessed at here.
        restarts: Number of starting vertices.

    Returns:
        A :class:`Fit`.

    Raises:
        ValueError: If the inputs disagree in length, if there are fewer than
            five quotes, if any total variance is not positive, if any weight
            is negative, or if no start produced a usable fit.
    """
    ks = [float(k) for k in log_moneyness]
    ws = [float(w) for w in total_variance]
    if len(ks) != len(ws):
        raise ValueError(f"got {len(ks)} abscissae and {len(ws)} variances")
    if len(ks) < 5:
        raise ValueError(
            f"a five-parameter slice needs at least five quotes, got {len(ks)}"
        )
    if any(w <= 0.0 for w in ws):
        raise ValueError("every total variance must be positive")
    if weight is None:
        weights = [1.0] * len(ks)
    else:
        weights = [float(x) for x in weight]
        if len(weights) != len(ks):
            raise ValueError(f"got {len(weights)} weights for {len(ks)} quotes")
        if any(x < 0.0 for x in weights):
            raise ValueError("weights must be non-negative")
        if sum(weights) <= 0.0:
            raise ValueError("at least one weight must be positive")
    if len({round(k, 12) for k in ks}) < 5:
        raise ValueError("the quotes must sit at at least five distinct log-moneyness values")

    total_weight = sum(weights)

    def residual(m: float, s: float) -> tuple[float, tuple[float, float, float]] | None:
        if not math.isfinite(m) or not math.isfinite(s) or s <= 0.0:
            return None
        solved = _linear_fit(ks, ws, weights, m, s)
        if solved is None:
            return None
        a, d, c = solved
        error = 0.0
        for k, target, wt in zip(ks, ws, weights, strict=True):
            y = (k - m) / s
            gap = a + d * y + c * math.hypot(y, 1.0) - target
            error += wt * gap * gap
        return error / total_weight, solved

    def objective(point: tuple[float, ...]) -> float:
        outcome = residual(point[0], math.exp(point[1]))
        return math.inf if outcome is None else outcome[0]

    spread = max(ks) - min(ks)
    centre = sum(k * wt for k, wt in zip(ks, weights, strict=True)) / total_weight
    span = spread if spread > 0.0 else 1.0
    starts = [
        (centre, math.log(span / 2.0)),
        (centre, math.log(span / 8.0)),
        (min(ks), math.log(span)),
        (max(ks), math.log(span / 4.0)),
    ][:max(1, restarts)]

    best: tuple[float, tuple[float, ...], int, bool] | None = None
    for start in starts:
        if not math.isfinite(objective(start)):
            continue
        found = nelder_mead(objective, start, step=0.25, tolerance=1e-12, max_iterations=3000)
        if best is None or found.value < best[0]:
            best = (found.value, found.point, found.iterations, found.converged)

    if best is None:
        raise ValueError("no starting vertex produced a usable fit")

    _, point, iterations, converged = best
    m, s = point[0], math.exp(point[1])
    outcome = residual(m, s)
    if outcome is None:  # pragma: no cover - the search only returns feasible points
        raise ValueError("the fitted vertex does not admit a linear solution")
    mean_square, (a, d, c) = outcome

    # Back out of the substitution. c = b s and d = rho b s, so b = c / s and
    # rho = d / c, with the degenerate c = 0 meaning a flat slice, where rho is
    # not identified and any admissible value describes the same function.
    b = c / s
    rho = 0.0 if c == 0.0 else max(-0.999999, min(0.999999, d / c))
    fitted = SVI(a=a, b=b, rho=rho, m=m, s=s)

    max_error = max(
        abs(fitted.total_variance(k) - target) for k, target in zip(ks, ws, strict=True)
    )
    return Fit(fitted, math.sqrt(max(mean_square, 0.0)), max_error, iterations, converged)
