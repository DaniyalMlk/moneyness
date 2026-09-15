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
import math
import sys
from collections.abc import Sequence

from .american import bjerksund_stensland, trigger_price
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
        ("bjerksund-stensland", f"{bjerksund_stensland(inputs, option):.10f}"),
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
