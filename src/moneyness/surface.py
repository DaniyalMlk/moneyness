"""A whole volatility surface: several maturities, and what holds them together.

A slice says what the market thinks about one expiry. A surface is several of
those plus an answer to the question the quotes do not address, which is what
happens at the maturities nobody quoted. The answer has to respect two
conditions, and the interesting thing about this module is how much falls out of
taking them seriously.

**Calendar arbitrage.** At fixed log-moneyness, total implied variance must not
decrease as maturity grows. If it did, one could buy the longer-dated option and
sell the shorter one and collect a certain profit, because the longer option
dominates the shorter payoff by payoff. So the slices must not cross.

**Butterfly arbitrage.** Each slice must imply a non-negative density, which is
:func:`moneyness.svi.durrleman` being non-negative.

The interpolation is linear in total variance at fixed log-moneyness. That is
not the only defensible choice, but it is the one that makes the first condition
automatic rather than hoped for: a linear function of maturity between two
ordered endpoints is monotone in maturity, so if the quoted slices do not cross
then nothing interpolated between them can cross either. The proof is one line
and the test asserts it on a grid anyway.

The second condition is not preserved by the same argument, because Durrleman's
function is not linear in ``w`` and so does not inherit anything from being
non-negative at the two endpoints. Whether it survives interpolation anyway is a
separate question, and the honest answer is that it seems to but is not proved
here: a randomised search over tens of thousands of admissible, non-crossing
slice pairs failed to produce an interpolated slice that violates it. That is
evidence and not a proof, so the surface checks the interpolated slices rather
than assuming them, and :meth:`Surface.butterfly` looks between the quoted
maturities by default rather than only at them. The search is in the tests, at a
smaller budget, so that a future change which does break it is caught.

**Local volatility.** Dupire's identity in these coordinates is

    sigma_loc^2(k, T) = (dw/dT) / g(k, T)

where ``g`` is exactly Durrleman's function. This is worth staring at. The
numerator is non-negative precisely when the surface is free of calendar
arbitrage; the denominator is non-negative precisely when it is free of
butterfly arbitrage. A local volatility exists — a real, non-negative number —
if and only if the surface admits neither. The two conditions are not side
constraints that a practitioner checks out of tidiness; they are the exact
conditions under which the question "what volatility would reproduce these
prices" has an answer at all. The identity is asserted against a separate
finite-difference implementation in the tests.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import pairwise

from .svi import SVI

__all__ = [
    "Calendar",
    "LocalVol",
    "Surface",
]


@dataclass(frozen=True, slots=True)
class Calendar:
    """The worst calendar violation found between two maturities.

    Attributes:
        worst: The largest amount by which total variance decreases with
            maturity, at any scanned log-moneyness. Zero or negative means no
            violation; a positive number is the size of the crossing.
        at: The log-moneyness where it was attained.
        earlier: Maturity of the earlier slice.
        later: Maturity of the later slice.
        free: True if no violation was found.
    """

    worst: float
    at: float
    earlier: float
    later: float
    free: bool


@dataclass(frozen=True, slots=True)
class LocalVol:
    """A local volatility, with the two quantities it was built from.

    Keeping the numerator and denominator alongside the answer is not
    bookkeeping. When a local volatility comes out unusable, the caller needs to
    know which condition failed, because the two have different remedies: a
    negative numerator is a calendar crossing between quoted maturities, which
    is a data problem, while a non-positive denominator is a density defect
    inside a single slice, which is a fitting problem.

    Attributes:
        variance: Local variance. ``nan`` if the identity has no admissible
            answer here.
        volatility: Its square root. ``nan`` in the same case.
        dw_dt: The numerator, ``dw/dT``. Non-negative iff no calendar arbitrage.
        durrleman: The denominator, ``g``. Positive iff no butterfly arbitrage.
        admissible: Whether both were in range.
    """

    variance: float
    volatility: float
    dw_dt: float
    durrleman: float
    admissible: bool


class Surface:
    """Quoted slices, and the surface interpolated through them.

    Args:
        slices: Pairs of ``(maturity, slice)``. At least one, maturities
            strictly positive and distinct. Order does not matter; they are
            sorted on construction.

    Raises:
        ValueError: If no slices are given, if any maturity is not positive, or
            if two slices share a maturity.
    """

    __slots__ = ("_maturities", "_slices")

    def __init__(self, slices: Iterable[tuple[float, SVI]]) -> None:
        ordered = sorted(((float(t), s) for t, s in slices), key=lambda pair: pair[0])
        if not ordered:
            raise ValueError("a surface needs at least one slice")
        for time, _ in ordered:
            if not math.isfinite(time) or time <= 0.0:
                raise ValueError(f"every maturity must be finite and positive, got {time}")
        for earlier, later in pairwise(ordered):
            if earlier[0] == later[0]:
                raise ValueError(f"two slices share the maturity {earlier[0]}")
        self._maturities = tuple(t for t, _ in ordered)
        self._slices = tuple(s for _, s in ordered)

    @property
    def maturities(self) -> tuple[float, ...]:
        """The quoted maturities, ascending."""
        return self._maturities

    @property
    def slices(self) -> tuple[SVI, ...]:
        """The quoted slices, in the same order as :attr:`maturities`."""
        return self._slices

    def _bracket(self, time: float) -> tuple[int, int, float]:
        """Indices either side of ``time``, and the weight on the later one.

        Outside the quoted range the weight is clamped, which makes the surface
        flat in total variance beyond the ends. Flat in *total* variance, not in
        volatility: extrapolating total variance flat means implied volatility
        decays as ``1 / sqrt(T)``, which is the conservative direction. The
        alternative — holding volatility flat and letting total variance grow
        linearly — invents variance nobody quoted, and does it fastest exactly
        where there is least information.
        """
        times = self._maturities
        if len(times) == 1 or time <= times[0]:
            return 0, 0, 0.0
        if time >= times[-1]:
            last = len(times) - 1
            return last, last, 0.0
        lo = 0
        while times[lo + 1] < time:
            lo += 1
        hi = lo + 1
        span = times[hi] - times[lo]
        return lo, hi, (time - times[lo]) / span

    def total_variance(self, k: float, time: float) -> float:
        """Total implied variance at log-moneyness ``k`` and maturity ``time``."""
        lo, hi, weight = self._bracket(time)
        if lo == hi:
            return self._slices[lo].total_variance(k)
        early = self._slices[lo].total_variance(k)
        late = self._slices[hi].total_variance(k)
        return early + weight * (late - early)

    def d_total_variance(self, k: float, time: float) -> float:
        """``dw/dk``. Linear in the same weights, since interpolation is linear in ``w``."""
        lo, hi, weight = self._bracket(time)
        if lo == hi:
            return self._slices[lo].d_total_variance(k)
        early = self._slices[lo].d_total_variance(k)
        late = self._slices[hi].d_total_variance(k)
        return early + weight * (late - early)

    def d2_total_variance(self, k: float, time: float) -> float:
        """``d2w/dk2``, likewise."""
        lo, hi, weight = self._bracket(time)
        if lo == hi:
            return self._slices[lo].d2_total_variance(k)
        early = self._slices[lo].d2_total_variance(k)
        late = self._slices[hi].d2_total_variance(k)
        return early + weight * (late - early)

    def dw_dt(self, k: float, time: float) -> float:
        """``dw/dT``, the slope of total variance in maturity.

        Piecewise constant in ``T``, because the interpolation is piecewise
        linear, and zero outside the quoted range where the surface is flat.
        This is the honest derivative of the surface as defined rather than a
        smoothed version of it: a smoother rule would give a prettier local
        volatility, but it would be the derivative of a different surface than
        the one :meth:`total_variance` returns, and the two would disagree.
        """
        lo, hi, _ = self._bracket(time)
        if lo == hi:
            return 0.0
        span = self._maturities[hi] - self._maturities[lo]
        early = self._slices[lo].total_variance(k)
        late = self._slices[hi].total_variance(k)
        return (late - early) / span

    def volatility(self, k: float, time: float) -> float:
        """The Black-Scholes volatility the surface implies.

        Raises:
            ValueError: If ``time`` is not positive.
        """
        if time <= 0.0:
            raise ValueError(f"time must be positive, got {time}")
        return math.sqrt(self.total_variance(k, time) / time)

    def durrleman(self, k: float, time: float) -> float:
        """Durrleman's function on the interpolated slice at ``time``.

        Between quoted maturities this is *not* the interpolation of the
        endpoints' Durrleman functions. ``g`` is a nonlinear function of ``w``
        and its derivatives, so it has to be recomputed from the interpolated
        ``w``, which is exactly why the butterfly condition can fail between two
        slices that each satisfy it.
        """
        w = self.total_variance(k, time)
        if w <= 0.0:
            raise ValueError(
                f"total variance is {w} at k={k}, T={time}; Durrleman's function is "
                "undefined there"
            )
        dw = self.d_total_variance(k, time)
        d2w = self.d2_total_variance(k, time)
        first = 1.0 - k * dw / (2.0 * w)
        return first * first - (dw * dw / 4.0) * (1.0 / w + 0.25) + d2w / 2.0

    def calendar(self, *, wing: float = 5.0, points: int = 401) -> list[Calendar]:
        """Check every adjacent pair of quoted slices for a crossing.

        Only adjacent pairs are checked, and that is sufficient: total variance
        being non-decreasing between each consecutive pair gives it
        non-decreasing across the whole set by transitivity, so a scan over all
        pairs would do quadratic work to learn nothing more.

        Args:
            wing: Half-width in log-moneyness to scan.
            points: Grid resolution.

        Returns:
            One :class:`Calendar` per adjacent pair, in maturity order. Empty
            for a single-slice surface, which cannot have a calendar arbitrage.

        Raises:
            ValueError: If ``wing`` is not positive or ``points`` is below 3.
        """
        if wing <= 0.0:
            raise ValueError(f"wing must be positive, got {wing}")
        if points < 3:
            raise ValueError(f"points must be at least 3, got {points}")
        step = 2.0 * wing / (points - 1)
        report: list[Calendar] = []
        for index in range(len(self._slices) - 1):
            early, late = self._slices[index], self._slices[index + 1]
            worst, at = -math.inf, 0.0
            for i in range(points):
                k = -wing + i * step
                drop = early.total_variance(k) - late.total_variance(k)
                if drop > worst:
                    worst, at = drop, k
            report.append(
                Calendar(
                    worst=worst,
                    at=at,
                    earlier=self._maturities[index],
                    later=self._maturities[index + 1],
                    free=worst <= 0.0,
                )
            )
        return report

    def butterfly(
        self,
        *,
        wing: float = 5.0,
        points: int = 401,
        maturities: Sequence[float] | None = None,
    ) -> list[tuple[float, float, float]]:
        """Find the worst density defect at each of several maturities.

        By default this checks the quoted maturities and the midpoint of every
        gap between them. The midpoints are the point of the exercise: the
        quoted slices are usually fitted to be admissible, and the interpolation
        is where the condition can quietly stop holding.

        Args:
            wing: Half-width in log-moneyness.
            points: Grid resolution.
            maturities: Maturities to check, overriding the default.

        Returns:
            Triples of ``(maturity, worst g, log-moneyness)``, in maturity
            order.

        Raises:
            ValueError: If ``wing`` is not positive, ``points`` is below 3, or
                any requested maturity is not positive.
        """
        if wing <= 0.0:
            raise ValueError(f"wing must be positive, got {wing}")
        if points < 3:
            raise ValueError(f"points must be at least 3, got {points}")
        if maturities is None:
            times = list(self._maturities)
            times.extend((a + b) / 2.0 for a, b in pairwise(self._maturities))
            times.sort()
        else:
            times = [float(t) for t in maturities]
            if any(t <= 0.0 for t in times):
                raise ValueError("every maturity must be positive")

        step = 2.0 * wing / (points - 1)
        report: list[tuple[float, float, float]] = []
        for time in times:
            worst, at = math.inf, 0.0
            for i in range(points):
                k = -wing + i * step
                g = self.durrleman(k, time)
                if g < worst:
                    worst, at = g, k
            report.append((time, worst, at))
        return report

    def local_vol(self, k: float, time: float) -> LocalVol:
        """Local volatility at ``(k, T)`` by the Dupire identity.

        ``sigma_loc^2 = (dw/dT) / g``, with ``g`` Durrleman's function. See the
        module docstring for why the two arbitrage conditions are exactly the
        conditions under which this has an answer.

        Where the identity is inadmissible the variance and volatility come back
        as ``nan`` rather than raising, because the usual caller is sweeping a
        grid and wants to see the shape of the region that failed rather than
        stop at its first point. The diagnostic fields say which condition broke.

        Args:
            k: Log-moneyness.
            time: Maturity. Positive.

        Returns:
            A :class:`LocalVol`.

        Raises:
            ValueError: If ``time`` is not positive, or the total variance
                vanishes at ``(k, T)``.
        """
        if time <= 0.0:
            raise ValueError(f"time must be positive, got {time}")
        g = self.durrleman(k, time)
        numerator = self.dw_dt(k, time)
        if g <= 0.0 or numerator < 0.0:
            return LocalVol(math.nan, math.nan, numerator, g, False)
        variance = numerator / g
        return LocalVol(variance, math.sqrt(variance), numerator, g, True)

    def __repr__(self) -> str:
        return f"Surface({len(self._slices)} slices, maturities {self._maturities})"
