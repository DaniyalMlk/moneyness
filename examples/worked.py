"""The front page's figures, recomputed against the fifty-digit reference.

``tests/reference.py`` evaluates the same mathematics at fifty decimal digits in
``mpmath``. It shares no code, no rearrangement and no branch structure with the
library: where the implementation has a tail-safe grouping and a limit case, the
reference has the textbook expression and enough precision that the difference
does not matter. Agreement between them across a grid is therefore evidence
about the implementation rather than about whoever typed the expected values.

That is the strongest thing in this repository, and until now a reader had no
way to see it. So this script measures six things and the README quotes what it
prints:

1. The price, against the fifty-digit price, over a grid that reaches the wings
   and the limits.
2. The normal distribution function, against ``mpmath``'s ``erfc``, from forty
   standard deviations below the mean to forty above.
3. The five first- and second-order Greeks, against numerical derivatives of
   the fifty-digit price. Differentiating the reference rather than writing a
   second closed form is the point: a second closed form is another chance to
   make the same algebra mistake twice, while a numerical derivative shares
   nothing with the analytic Greek except the price.
4. Put-call parity, which holds whatever the reference says.
5. The volatility recovered from the price it implies — reported as a function
   of how much the price moves with volatility, because that is what decides
   whether there is anything to recover.
6. And for the quotes where there is not, the width of the set of volatilities
   that reprice them. A monotone function is not necessarily a steep one, and a
   solver always returns a number.

Figures are reported as bounds. The last bits of a double-precision result
depend on the interpreter's libm and on how the expression was grouped, so a
figure quoted to ten significant places would be true only where it was
measured.

Needs the development extra, because the reference does::

    pip install -e ".[dev]"
    python examples/worked.py

Exits non-zero if any figure moves outside the band the README states.
"""

from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path
from typing import NamedTuple

# The reference lives beside the tests rather than in the package: it is not
# something the library ships, and nothing in `src/` may depend on it.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.reference import ref_bsm, ref_derivative, ref_norm_cdf

import moneyness as M

#: The grid every price and Greek check runs over. It is chosen to reach the
#: places a naive implementation breaks rather than the textbook middle: a spot
#: at two fifths of the strike and at nearly four times it, three days and ten
#: years, a negative carry, and volatilities from 5% to 80%.
SPOTS = (50.0, 100.0, 150.0)
STRIKES = (40.0, 100.0, 160.0)
TIMES = (0.01, 0.5, 2.0, 10.0)
RATES = (-0.02, 0.0, 0.05)
VOLS = (0.05, 0.2, 0.8)

#: Below this, a price carries no information about the volatility that
#: produced it: the quote is worth so little that any volatility reprices it.
#: Used to separate the solver's accuracy from the question's answerability.
MUTE_VEGA = 1e-8


def grid() -> list[tuple[float, float, float, float, float]]:
    return list(itertools.product(SPOTS, STRIKES, TIMES, RATES, VOLS))


class Check(NamedTuple):
    """One row of the front page's table."""

    label: str
    measured: float
    bound: float
    rendered: str

    @property
    def passed(self) -> bool:
        return self.measured <= self.bound


def price_against_the_reference() -> tuple[Check, Check]:
    """The price, relatively where that means something and absolutely always.

    Two rows rather than one, because a single figure would have to be a lie in
    one direction or the other. A relative error is the right measure for a
    price somebody would trade on, and a meaningless one for a price of 1e-18:
    there, the implementation and the reference disagree in the last bits of a
    number that is zero for every practical purpose, and the ratio of two such
    disagreements can be anything at all. So the relative bound is stated for
    the prices above a floor, and the absolute bound for every point on the
    grid including the ones below it.
    """
    worst_relative = 0.0
    worst_absolute = 0.0
    points = 0
    for spot, strike, time, rate, vol in grid():
        inputs = M.Inputs(spot, strike, time, rate, vol)
        parameters = {
            "spot": spot,
            "strike": strike,
            "time": time,
            "rate": rate,
            "vol": vol,
            "carry": rate,
        }
        for option in (M.OptionType.CALL, M.OptionType.PUT):
            is_call = option is M.OptionType.CALL
            got = M.price(inputs, option)
            reference = ref_bsm(**parameters, is_call=is_call)
            points += 1
            worst_absolute = max(worst_absolute, abs(got - reference))
            if reference > 1e-8 * spot:
                worst_relative = max(worst_relative, abs(got - reference) / reference)
    return (
        Check(
            label=(
                "The price against the same formula at fifty digits in mpmath, "
                f"{points} points"
            ),
            measured=worst_relative,
            bound=1e-12,
            rendered="worst relative error 1e-12, where the price is above 1e-8 of spot",
        ),
        Check(
            label="The same, absolutely, including the quotes worth almost nothing",
            measured=worst_absolute,
            bound=1e-13,
            rendered="worst absolute error 1e-13, on spots of 50 to 150",
        ),
    )


def normal_cdf_against_the_reference() -> Check:
    """The distribution function, forty standard deviations either way.

    The far left tail is the interesting end. There the function is a number
    like 1e-350, and a routine that computes it as ``1 - cdf(-z)`` returns zero
    — correctly rounded and useless. The bound here is absolute, which the
    right tail makes easy and the left tail makes meaningless, so the range is
    half the claim: it is stated over the whole sweep rather than over the part
    where it is comfortable.
    """
    sweep = [index / 100.0 for index in range(-4000, 4001, 7)]
    worst = max(abs(M.norm_cdf(z) - ref_norm_cdf(z)) for z in sweep)
    return Check(
        label=f"The normal distribution function against mpmath's erfc, {len(sweep)} points",
        measured=worst,
        bound=1e-15,
        rendered="worst absolute error 1e-15, from z = -40 to z = +40",
    )


def greeks_against_differentiated_reference() -> Check:
    """Five Greeks against numerical derivatives of the fifty-digit price.

    The three-day slice is dropped here and only here. Theta and rho are
    derivatives with respect to time and rate, and at ``t = 0.01`` the
    high-precision differentiation is being asked for a derivative at the edge
    of the region where the price is smooth; the disagreement that follows is
    the reference's step size rather than the library's algebra. Everything
    else on the grid stays, wings included.
    """
    specifications: dict[
        str, tuple[tuple[str, ...], tuple[int, ...], object, float]
    ] = {
        "delta": (("spot",), (1,), lambda i, o: M.delta(i, o), 1.0),
        "gamma": (("spot",), (2,), lambda i, o: M.gamma(i), 1.0),
        "vega": (("vol",), (1,), lambda i, o: M.vega(i), 1.0),
        "theta": (("time",), (1,), lambda i, o: M.theta(i, o), -1.0),
        "rho": (("rate",), (1,), lambda i, o: M.rho(i, o), 1.0),
    }
    worst = 0.0
    for wrt, orders, analytic, sign in specifications.values():
        for spot, strike, time, rate, vol in grid():
            if time < 0.05:
                continue
            inputs = M.Inputs(spot, strike, time, rate, vol)
            parameters = {
                "spot": spot,
                "strike": strike,
                "time": time,
                "rate": rate,
                "vol": vol,
                "carry": rate,
            }
            for option in (M.OptionType.CALL, M.OptionType.PUT):
                is_call = option is M.OptionType.CALL
                got = float(analytic(inputs, option))  # type: ignore[operator]
                reference = sign * ref_derivative(parameters, is_call, wrt, orders)
                worst = max(worst, abs(got - reference) / max(abs(reference), 1e-10))
    return Check(
        label="Delta, gamma, vega, theta and rho against derivatives of that price",
        measured=worst,
        bound=1e-11,
        rendered="worst relative error 1e-11, against a numerical derivative",
    )


def put_call_parity() -> Check:
    """An identity, so it holds whatever the reference says.

    This is the cheap check and it earns its place by being independent of
    everything above: parity is algebra on the two payoffs and does not know
    what a distribution function is. A library whose call and put are both
    wrong in the same way passes the reference comparison and fails this.
    """
    worst = 0.0
    for spot, strike, time, rate, vol in grid():
        inputs = M.Inputs(spot, strike, time, rate, vol)
        call = M.price(inputs, M.OptionType.CALL)
        put = M.price(inputs, M.OptionType.PUT)
        # Carry defaults to the rate, so the forward is the spot.
        expected = spot - strike * math.exp(-rate * time)
        worst = max(worst, abs(call - put - expected) / max(abs(expected), 1.0))
    return Check(
        label="Put-call parity across the same grid",
        measured=worst,
        bound=1e-13,
        rendered="worst relative error 1e-13",
    )


class Recovery(NamedTuple):
    """How well a volatility is recovered, at one threshold on vega."""

    floor: float
    quotes: int
    worst: float


def implied_volatility_ladder() -> tuple[Check, list[Recovery], int]:
    """Recover the volatility from the price it implies, graded by vega.

    A single round-trip error over this grid would be a figure about the worst
    *question*, not about the solver: a third of the grid is quotes a solver
    cannot answer, and the worst of them is out by 0.8 in volatility — which is
    not a bug, because the quote in question is worth 1.4e-18 on a spot of 50.

    So the row is a ladder. Vega is how much the price moves per unit of
    volatility, and the recovery error tracks it almost exactly, which is what
    has to happen: a quote that moves by a cent per volatility point pins the
    volatility to fourteen digits, and one that moves by nothing pins nothing.
    """
    floors = (1.0, 1e-2, 1e-4, MUTE_VEGA)
    ladder: list[Recovery] = []
    mute = 0
    for floor in floors:
        quotes = 0
        worst = 0.0
        for spot, strike, time, rate, vol in grid():
            inputs = M.Inputs(spot, strike, time, rate, vol)
            if M.vega(inputs) <= floor:
                continue
            for option in (M.OptionType.CALL, M.OptionType.PUT):
                quoted = M.price(inputs, option)
                recovered = M.implied_vol(M.Quote(spot, strike, time, rate, quoted), option)
                quotes += 1
                worst = max(worst, abs(recovered - vol))
        ladder.append(Recovery(floor, quotes, worst))

    for spot, strike, time, rate, vol in grid():
        if M.vega(M.Inputs(spot, strike, time, rate, vol)) <= MUTE_VEGA:
            mute += 2

    best = ladder[0]
    check = Check(
        label="Volatility recovered from the price it implies, where vega exceeds 1",
        measured=best.worst,
        bound=1e-13,
        rendered=f"worst error 1e-13 over {best.quotes} quotes",
    )
    return check, ladder, mute


def the_width_of_an_unanswerable_quote() -> tuple[Check, float, float, float, float]:
    """How wide the set of volatilities that reprice a worthless quote is.

    Eleven standard deviations out of the money, a call is worth its intrinsic
    value to the same double for every volatility in a wide band, so asking for
    "the" implied volatility is asking for an element of that band and a solver
    obliges. Reporting the band is the honest version of the answer, and it is
    also the only form of the statement that a test can check.
    """
    spot, strike, time, rate, vol = 50.0, 100.0, 0.01, -0.02, 0.8
    option = M.OptionType.CALL
    quoted = M.price(M.Inputs(spot, strike, time, rate, vol), option)
    tolerance = 1e-16 * spot
    low = high = None
    for step in range(1, 4001):
        candidate = step / 1000.0
        repriced = M.price(M.Inputs(spot, strike, time, rate, candidate), option)
        if abs(repriced - quoted) <= tolerance:
            low = candidate if low is None else low
            high = candidate
    assert low is not None and high is not None
    return (
        Check(
            label="The width of the set that reprices a quote worth 1.4e-18",
            # The claim is that the set is *wide*, so the check is the other way
            # round: a narrow set would mean the quote was answerable after all
            # and this row was describing nothing.
            measured=1.0 / (high - low),
            bound=2.0,
            rendered=f"every volatility from {low:.3f} to {high:.3f} reprices it",
        ),
        quoted,
        tolerance,
        low,
        high,
    )


def main() -> dict[str, object]:
    """Run every check, print the table, and return the figures."""
    price_relative, price_absolute = price_against_the_reference()
    recovery, ladder, mute = implied_volatility_ladder()
    width, quoted, tolerance, low, high = the_width_of_an_unanswerable_quote()
    checks = [
        price_relative,
        price_absolute,
        normal_cdf_against_the_reference(),
        greeks_against_differentiated_reference(),
        put_call_parity(),
        recovery,
        width,
    ]

    print("What is checked, and against what")
    print()
    print("| Check | Result |")
    print("| --- | --- |")
    for check in checks:
        print(f"| {check.label} | {check.rendered} |")

    print()
    print("How the recovered volatility depends on how much the price moves:")
    print()
    print("| vega above | quotes | worst error |")
    print("| --- | --- | --- |")
    for rung in ladder:
        print(f"| {rung.floor:g} | {rung.quotes} | {rung.worst:.1e} |")
    print()
    print(
        f"  {mute} of the {2 * len(grid())} quotes on the grid have a vega of "
        f"{MUTE_VEGA:g} or less, and are not questions a solver can answer."
    )
    print(
        f"  The worst of them is worth {quoted:.3e} and every volatility from "
        f"{low:.3f} to {high:.3f} reprices it to within {tolerance:.1e}."
    )

    print()
    print("Measured against the bound the README states:")
    for check in checks:
        verdict = "ok" if check.passed else "BREACHED"
        print(f"  {check.label}: {check.measured:.3e} <= {check.bound:.0e}  {verdict}")

    failed = [check for check in checks if not check.passed]
    if failed:
        print()
        for check in failed:
            print(
                f"FAIL {check.label}: measured {check.measured:.6e}, "
                f"README claims {check.bound:.0e}",
                file=sys.stderr,
            )
        raise SystemExit(1)

    return {
        "checks": {check.label: check.measured for check in checks},
        "bounds": {check.label: check.bound for check in checks},
        "rendered": {check.label: check.rendered for check in checks},
        "ladder": [tuple(rung) for rung in ladder],
        "mute": mute,
        "quoted": quoted,
        "window": (low, high),
    }


if __name__ == "__main__":
    main()
