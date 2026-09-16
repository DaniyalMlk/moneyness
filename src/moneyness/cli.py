"""Command-line access to the library.

Five subcommands: ``price`` for a single option, ``greeks`` for its
sensitivities, ``iv`` to invert a quoted price, ``ladder`` for a table of
strikes at one maturity, and ``american`` for a lattice valuation with the
right to exercise early.

The carry is specified in whichever way suits the instrument — ``--carry``
directly, ``--dividend`` for a share, ``--future`` for Black's model — rather
than making the caller do the arithmetic.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

from .american import bjerksund_stensland, bjerksund_stensland_2002, trigger_price
from .bsm import Inputs, OptionType, forward, parity_gap, price
from .greeks import (
    charm,
    colour,
    delta,
    dual_delta,
    dual_gamma,
    gamma,
    rho,
    rho_carry,
    speed,
    theta,
    vanna,
    vega,
    veta,
    volga,
    zomma,
)
from .implied import Method, Quote, bounds, solve
from .lattice import Exercise, Lattice, boundary, min_steps, price_lattice, richardson
from .surface import Surface
from .svi import SVI, Butterfly, calibrate

__all__ = ["main"]


def _carry_from(args: argparse.Namespace) -> float | None:
    """Resolve the cost of carry from whichever flag was given."""
    given = [args.carry is not None, args.dividend is not None, args.future]
    if sum(given) > 1:
        raise SystemExit("give at most one of --carry, --dividend and --future")
    if args.future:
        return 0.0
    if args.dividend is not None:
        return float(args.rate) - float(args.dividend)
    return None if args.carry is None else float(args.carry)


def _add_market(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--spot", type=float, required=True, help="price of the underlying")
    parser.add_argument("--strike", type=float, required=True, help="exercise price")
    parser.add_argument("--time", type=float, required=True, help="year fraction to expiry")
    parser.add_argument("--rate", type=float, default=0.0, help="continuous discount rate")
    carry = parser.add_argument_group("cost of carry (at most one)")
    carry.add_argument("--carry", type=float, default=None, help="cost of carry directly")
    carry.add_argument("--dividend", type=float, default=None, help="continuous dividend yield")
    carry.add_argument(
        "--future", action="store_true", help="option on a future, so the carry is zero"
    )
    parser.add_argument(
        "--put", action="store_true", help="price a put; the default is a call"
    )


def _option_of(args: argparse.Namespace) -> OptionType:
    return OptionType.PUT if args.put else OptionType.CALL


def _inputs_of(args: argparse.Namespace) -> Inputs:
    return Inputs(args.spot, args.strike, args.time, args.rate, args.vol, carry=_carry_from(args))


def _run_price(args: argparse.Namespace) -> int:
    inputs = _inputs_of(args)
    option = _option_of(args)
    print(f"{option.value:>18}  {price(inputs, option):.10f}")
    print(f"{'forward':>18}  {forward(inputs):.10f}")
    print(f"{'discount factor':>18}  {inputs.discount:.10f}")
    print(f"{'parity residual':>18}  {parity_gap(inputs):.3e}")
    return 0


def _run_greeks(args: argparse.Namespace) -> int:
    inputs = _inputs_of(args)
    option = _option_of(args)
    rows: list[tuple[str, float]] = [
        ("price", price(inputs, option)),
        ("delta", delta(inputs, option)),
        ("gamma", gamma(inputs)),
        ("vega", vega(inputs)),
        ("theta", theta(inputs, option)),
        ("rho (rate)", rho(inputs, option)),
        ("rho (carry)", rho_carry(inputs, option)),
        ("vanna", vanna(inputs)),
        ("volga", volga(inputs)),
        ("charm", charm(inputs, option)),
        ("veta", veta(inputs)),
        ("colour", colour(inputs)),
        ("speed", speed(inputs)),
        ("zomma", zomma(inputs)),
        ("dual delta", dual_delta(inputs, option)),
        ("dual gamma", dual_gamma(inputs)),
    ]
    width = max(len(name) for name, _ in rows)
    for name, value in rows:
        print(f"{name:>{width}}  {value:>18.10f}")
    print()
    print("vega is per unit of volatility and theta is per year.")
    return 0


def _run_iv(args: argparse.Namespace) -> int:
    quote = Quote(
        args.spot, args.strike, args.time, args.rate, args.price, carry=_carry_from(args)
    )
    option = _option_of(args)
    limits = bounds(quote, option)
    method = Method.BRENT if args.brent else Method.NEWTON
    try:
        solution = solve(quote, option, method)
    except ValueError as error:
        print(f"could not invert the quote: {error}", file=sys.stderr)
        print(
            f"the price must lie in [{limits.lower:.10f}, {limits.upper:.10f}]",
            file=sys.stderr,
        )
        return 1
    print(f"{'implied volatility':>20}  {solution.vol:.10f}")
    print(f"{'total volatility':>20}  {solution.total_vol:.10f}")
    print(f"{'method':>20}  {solution.method.value}")
    print(f"{'iterations':>20}  {solution.iterations}")
    print(f"{'price residual':>20}  {solution.residual:.3e}")
    print(f"{'arbitrage bounds':>20}  [{limits.lower:.10f}, {limits.upper:.10f}]")
    return 0


def _run_ladder(args: argparse.Namespace) -> int:
    carry = _carry_from(args)
    step = (args.high - args.low) / max(args.steps - 1, 1)
    header = f"{'strike':>10} {'call':>12} {'put':>12} {'delta':>10} {'gamma':>10} {'vega':>12}"
    print(header)
    print("-" * len(header))
    for index in range(args.steps):
        strike = args.low + index * step
        inputs = Inputs(args.spot, strike, args.time, args.rate, args.vol, carry=carry)
        print(
            f"{strike:>10.4f}"
            f" {price(inputs, OptionType.CALL):>12.6f}"
            f" {price(inputs, OptionType.PUT):>12.6f}"
            f" {delta(inputs, OptionType.CALL):>10.6f}"
            f" {gamma(inputs):>10.6f}"
            f" {vega(inputs):>12.6f}"
        )
    return 0


def _run_term(args: argparse.Namespace) -> int:
    """One strike across a range of maturities: the transpose of the ladder.

    The maturities are spaced geometrically rather than evenly. A term
    structure is read in ratios -- a week against a month against a year --
    and an even grid from one week to two years spends almost all of its rows
    on maturities that differ from each other by a rounding error while
    skipping the short end entirely, which is the part that moves.
    """
    carry = _carry_from(args)
    if args.near <= 0.0 or args.far <= 0.0:
        raise SystemExit("--near and --far must both be positive")
    if args.far < args.near:
        raise SystemExit("--far must not be before --near")
    if args.steps < 1:
        raise SystemExit("--steps must be at least 1")

    header = (
        f"{'maturity':>10} {'call':>12} {'put':>12} {'delta':>10}"
        f" {'gamma':>10} {'vega':>12} {'theta':>12}"
    )
    print(header)
    print("-" * len(header))

    ratio = 1.0 if args.steps == 1 else (args.far / args.near) ** (1.0 / (args.steps - 1))
    for index in range(args.steps):
        time = args.near * ratio**index
        inputs = Inputs(args.spot, args.strike, time, args.rate, args.vol, carry=carry)
        print(
            f"{time:>10.4f}"
            f" {price(inputs, OptionType.CALL):>12.6f}"
            f" {price(inputs, OptionType.PUT):>12.6f}"
            f" {delta(inputs, OptionType.CALL):>10.6f}"
            f" {gamma(inputs):>10.6f}"
            f" {vega(inputs):>12.6f}"
            f" {theta(inputs, OptionType.CALL):>12.6f}"
        )
    return 0


def _read_quotes(source: str) -> dict[float, list[tuple[float, float]]]:
    """Read ``maturity,strike,vol`` rows, grouped by maturity.

    A header row is optional and detected by trying to parse it as numbers,
    which is more forgiving than demanding a particular spelling and less
    fragile than guessing from the text. It is allowed to follow comment lines
    rather than having to be the very first line, since a file that explains
    where its quotes came from is more useful than one that does not.

    Args:
        source: Path to a CSV, or ``-`` for standard input.

    Returns:
        Quotes grouped by maturity, in file order within each group.

    Raises:
        SystemExit: If the file cannot be read, a row is malformed, or a value
            is outside its domain. The row number is always reported, because a
            surface is typically a few hundred rows and "one of them is wrong"
            is not a usable diagnostic.
    """
    if source == "-":
        lines: Iterable[str] = sys.stdin.read().splitlines()
    else:
        path = Path(source)
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as error:
            raise SystemExit(f"cannot read {source}: {error}") from error

    grouped: dict[float, list[tuple[float, float]]] = {}
    for number, row in enumerate(csv.reader(lines), start=1):
        cleaned = [field.strip() for field in row if field.strip()]
        if not cleaned or cleaned[0].startswith("#"):
            continue
        if len(cleaned) < 3:
            raise SystemExit(
                f"row {number}: expected maturity, strike and vol, got {len(cleaned)} fields"
            )
        try:
            maturity, strike, vol = (float(field) for field in cleaned[:3])
        except ValueError:
            if not grouped:
                continue  # a header, wherever it happens to sit
            raise SystemExit(f"row {number}: {cleaned[:3]} are not three numbers") from None
        if maturity <= 0.0:
            raise SystemExit(f"row {number}: maturity {maturity} is not positive")
        if strike <= 0.0:
            raise SystemExit(f"row {number}: strike {strike} is not positive")
        if vol <= 0.0:
            raise SystemExit(f"row {number}: volatility {vol} is not positive")
        grouped.setdefault(maturity, []).append((strike, vol))

    if not grouped:
        raise SystemExit(f"no quotes found in {source}")
    return grouped


def _run_surface(args: argparse.Namespace) -> int:
    """Fit a surface to quoted volatilities and report what it implies.

    Returns 1 when the fitted surface admits arbitrage, so that the command is
    usable as a check in a pipeline. That is a finding about the data rather
    than a failure of the request, which is why it is an exit status and not an
    exception.
    """
    grouped = _read_quotes(args.quotes)
    carry = args.rate if args.carry is None else args.carry

    fitted: list[tuple[float, SVI]] = []
    print(f"{'maturity':>9} {'quotes':>7} {'rmse':>11} {'max err':>11}"
          f" {'a':>9} {'b':>8} {'rho':>8} {'m':>8} {'s':>8}"
          f" {'left':>7} {'right':>7} {'worst g':>10}")
    print("-" * 120)

    arbitrage = False
    for maturity in sorted(grouped):
        quotes = grouped[maturity]
        if len(quotes) < 5:
            raise SystemExit(
                f"maturity {maturity} has {len(quotes)} quotes; "
                "a five-parameter slice needs at least five"
            )
        forward_price = args.spot * math.exp(carry * maturity)
        log_moneyness = [math.log(strike / forward_price) for strike, _ in quotes]
        variance = [vol * vol * maturity for _, vol in quotes]

        try:
            fit = calibrate(log_moneyness, variance)
        except ValueError as error:
            raise SystemExit(f"maturity {maturity}: {error}") from error

        slice_ = fit.slice_
        left, right = slice_.wing_slopes
        report = Butterfly.scan(slice_)
        if not report.free:
            arbitrage = True
        print(
            f"{maturity:>9.4f} {len(quotes):>7d} {fit.rmse:>11.3e} {fit.max_error:>11.3e}"
            f" {slice_.a:>9.5f} {slice_.b:>8.5f} {slice_.rho:>8.5f}"
            f" {slice_.m:>8.5f} {slice_.s:>8.5f}"
            f" {left:>7.3f} {right:>7.3f} {report.worst:>10.5f}"
        )
        fitted.append((maturity, slice_))

    surface = Surface(fitted)

    print()
    print("butterfly (worst value of Durrleman's function; negative implies a negative density)")
    for maturity, worst, at in surface.butterfly():
        quoted = "quoted" if maturity in grouped else "interpolated"
        flag = "  ARBITRAGE" if worst < 0.0 else ""
        if worst < 0.0:
            arbitrage = True
        print(f"  T={maturity:<9.4f} {quoted:<13} g={worst:>10.5f} at k={at:>7.3f}{flag}")

    print()
    if len(fitted) < 2:
        print("calendar: a single maturity cannot cross another")
    else:
        print("calendar (total variance must not fall as maturity grows)")
        for entry in surface.calendar():
            if entry.free:
                margin = -entry.worst
                print(
                    f"  {entry.earlier:.4f} -> {entry.later:.4f}  clear,"
                    f" closest approach {margin:.5f}"
                )
            else:
                arbitrage = True
                print(
                    f"  {entry.earlier:.4f} -> {entry.later:.4f}  CROSSES by"
                    f" {entry.worst:.5f} at k={entry.at:.3f}"
                )

    if args.local_vol:
        print()
        print("local volatility by the Dupire identity")
        maturities = surface.maturities
        if len(maturities) < 2:
            print("  a single maturity gives no slope in maturity, so dw/dT is zero")
            print("  everywhere and the identity has nothing to report")
            print()
            if arbitrage:
                print("verdict: the fitted surface admits arbitrage")
                return 1
            print("verdict: no arbitrage found")
            return 0

        # Strictly inside the quoted range. At a quoted endpoint the surface is
        # flat in maturity by construction -- that is what the extrapolation
        # rule says -- so dw/dT is zero there and the identity returns a local
        # volatility of zero. That number is a property of the extrapolation,
        # not of the market, and printing it in a column headed by a real
        # maturity would invite it to be read as one.
        near, far = maturities[0], maturities[-1]
        columns = 5
        times = [near + (far - near) * (i + 1) / (columns + 1) for i in range(columns)]
        header = f"{'k':>8}" + "".join(f"{t:>12.4f}" for t in times)
        print(header)
        print("-" * len(header))
        for step in range(-4, 5):
            k = step * 0.1
            cells = []
            for time in times:
                result = surface.local_vol(k, time)
                cells.append(
                    f"{result.volatility:>12.6f}" if result.admissible else f"{'--':>12}"
                )
            print(f"{k:>8.2f}" + "".join(cells))

    print()
    if arbitrage:
        print("verdict: the fitted surface admits arbitrage")
        return 1
    print("verdict: no arbitrage found")
    return 0


def _format_trigger(level: float) -> str:
    """An unreachable boundary is reported as such rather than as a number."""
    return "never reached" if math.isinf(level) else f"{level:.10f}"


def _run_american(args: argparse.Namespace) -> int:
    inputs = _inputs_of(args)
    option = _option_of(args)
    lattice = Lattice(args.lattice)

    floor = min_steps(inputs, lattice)
    if args.steps < floor:
        print(
            f"error: {args.steps} layers is below the {floor} this lattice needs "
            f"for the drift to stay inside the spread",
            file=sys.stderr,
        )
        return 1

    american = price_lattice(
        inputs, option, steps=args.steps, lattice=lattice, smooth=not args.raw
    )
    european = price_lattice(
        inputs,
        option,
        steps=args.steps,
        lattice=lattice,
        exercise=Exercise.EUROPEAN,
        smooth=not args.raw,
    )
    extrapolated = richardson(
        inputs, option, steps=args.steps, lattice=lattice, smooth=not args.raw
    )

    rows: list[tuple[str, str]] = [
        ("lattice", lattice.value),
        ("layers", str(args.steps)),
        ("smoothing", "off" if args.raw else "on"),
        ("american", f"{american.value:.10f}"),
        ("european (lattice)", f"{european.value:.10f}"),
        ("european (closed form)", f"{price(inputs, option):.10f}"),
        ("early exercise premium", f"{american.early_exercise_premium:.10f}"),
        ("american (extrapolated)", f"{extrapolated:.10f}"),
        ("bjerksund-stensland 1993", f"{bjerksund_stensland(inputs, option):.10f}"),
        ("bjerksund-stensland 2002", f"{bjerksund_stensland_2002(inputs, option):.10f}"),
        ("bs trigger price", _format_trigger(trigger_price(inputs, option))),
    ]
    width = max(len(name) for name, _ in rows)
    for name, value in rows:
        print(f"{name:>{width}}  {value}")

    if args.boundary:
        curve = boundary(inputs, option, steps=args.steps, lattice=lattice)
        print()
        if len(curve) <= 1:
            print("the exercise region is empty: early exercise is never optimal here.")
            return 0
        print(f"{'time':>10} {'critical spot':>16}")
        print("-" * 27)
        shown = curve[:: max(len(curve) // args.boundary_rows, 1)]
        if shown[-1] != curve[-1]:
            shown.append(curve[-1])
        for when, critical in shown:
            print(f"{when:>10.6f} {critical:>16.6f}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="moneyness", description="Option pricing, Greeks and implied volatility."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    price_parser = sub.add_parser("price", help="price one option")
    _add_market(price_parser)
    price_parser.add_argument("--vol", type=float, required=True, help="annualised volatility")
    price_parser.set_defaults(handler=_run_price)

    greeks_parser = sub.add_parser("greeks", help="price and all sensitivities")
    _add_market(greeks_parser)
    greeks_parser.add_argument("--vol", type=float, required=True, help="annualised volatility")
    greeks_parser.set_defaults(handler=_run_greeks)

    iv_parser = sub.add_parser("iv", help="imply volatility from a quoted price")
    _add_market(iv_parser)
    iv_parser.add_argument("--price", type=float, required=True, help="quoted option price")
    iv_parser.add_argument(
        "--brent", action="store_true", help="use Brent rather than safeguarded Newton"
    )
    iv_parser.set_defaults(handler=_run_iv)

    ladder_parser = sub.add_parser("ladder", help="a table of strikes at one maturity")
    ladder_parser.add_argument("--spot", type=float, required=True)
    ladder_parser.add_argument("--time", type=float, required=True)
    ladder_parser.add_argument("--vol", type=float, required=True)
    ladder_parser.add_argument("--rate", type=float, default=0.0)
    ladder_parser.add_argument("--carry", type=float, default=None)
    ladder_parser.add_argument("--dividend", type=float, default=None)
    ladder_parser.add_argument("--future", action="store_true")
    ladder_parser.add_argument("--low", type=float, required=True, help="lowest strike")
    ladder_parser.add_argument("--high", type=float, required=True, help="highest strike")
    ladder_parser.add_argument("--steps", type=int, default=9, help="number of strikes")
    ladder_parser.set_defaults(handler=_run_ladder)

    term_parser = sub.add_parser(
        "term", help="a table of maturities at one strike"
    )
    term_parser.add_argument("--spot", type=float, required=True)
    term_parser.add_argument("--strike", type=float, required=True)
    term_parser.add_argument("--vol", type=float, required=True)
    term_parser.add_argument("--rate", type=float, default=0.0)
    term_parser.add_argument("--carry", type=float, default=None)
    term_parser.add_argument("--dividend", type=float, default=None)
    term_parser.add_argument("--future", action="store_true")
    term_parser.add_argument("--near", type=float, required=True, help="shortest maturity")
    term_parser.add_argument("--far", type=float, required=True, help="longest maturity")
    term_parser.add_argument(
        "--steps", type=int, default=9, help="number of maturities, spaced geometrically"
    )
    term_parser.set_defaults(handler=_run_term)

    surface_parser = sub.add_parser(
        "surface",
        help="fit a volatility surface to quotes and report the arbitrage it admits",
        description=(
            "Reads maturity,strike,vol rows from a CSV file or standard input, fits an "
            "SVI slice per maturity, and reports the fit quality, the butterfly and "
            "calendar conditions, and optionally a grid of local volatilities. Exits 1 "
            "if the fitted surface admits arbitrage."
        ),
    )
    surface_parser.add_argument(
        "--quotes", required=True, help="CSV of maturity,strike,vol, or - for standard input"
    )
    surface_parser.add_argument(
        "--spot", type=float, required=True, help="price of the underlying"
    )
    surface_parser.add_argument("--rate", type=float, default=0.0)
    surface_parser.add_argument(
        "--carry", type=float, default=None, help="cost of carry; defaults to the rate"
    )
    surface_parser.add_argument(
        "--local-vol",
        action="store_true",
        dest="local_vol",
        help="also print a grid of local volatilities",
    )
    surface_parser.set_defaults(handler=_run_surface)

    american_parser = sub.add_parser(
        "american", help="lattice valuation with the right to exercise early"
    )
    _add_market(american_parser)
    american_parser.add_argument(
        "--vol", type=float, required=True, help="annualised volatility"
    )
    american_parser.add_argument(
        "--steps", type=int, default=512, help="number of lattice layers"
    )
    american_parser.add_argument(
        "--lattice",
        choices=[item.value for item in Lattice],
        default=Lattice.CRR.value,
        help="which lattice construction to build",
    )
    american_parser.add_argument(
        "--raw",
        action="store_true",
        help="disable Broadie-Detemple smoothing, to see the unsmoothed lattice",
    )
    american_parser.add_argument(
        "--boundary", action="store_true", help="also print the early-exercise boundary"
    )
    american_parser.add_argument(
        "--boundary-rows", type=int, default=12, help="how many boundary rows to show"
    )
    american_parser.set_defaults(handler=_run_american)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit status."""
    args = _parser().parse_args(argv)
    try:
        result: int = args.handler(args)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return result


if __name__ == "__main__":
    raise SystemExit(main())
