"""Tests for the command-line interface.

These exercise the library end to end through the same path a user takes, which
is the only way to catch the wiring mistakes unit tests cannot see: a flag that
resolves to the wrong carry, a subcommand that never reaches its handler, an
error that escapes as a traceback instead of an exit status.
"""

from __future__ import annotations

import io
import math
import pathlib
import re

import pytest

from moneyness.bsm import Inputs, OptionType, price
from moneyness.cli import main


def test_price_reports_the_model_price(capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        main(
            [
                "price",
                "--spot",
                "100",
                "--strike",
                "95",
                "--time",
                "0.5",
                "--rate",
                "0.04",
                "--vol",
                "0.22",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    expected = price(Inputs(100.0, 95.0, 0.5, 0.04, 0.22), OptionType.CALL)
    assert f"{expected:.10f}" in out
    assert "forward" in out


def test_put_flag_selects_the_put(capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        main(
            [
                "price",
                "--spot",
                "100",
                "--strike",
                "95",
                "--time",
                "0.5",
                "--rate",
                "0.04",
                "--vol",
                "0.22",
                "--put",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    expected = price(Inputs(100.0, 95.0, 0.5, 0.04, 0.22), OptionType.PUT)
    assert f"{expected:.10f}" in out
    assert "put" in out


def test_future_flag_zeroes_the_carry(capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        main(
            [
                "price",
                "--spot",
                "100",
                "--strike",
                "95",
                "--time",
                "0.5",
                "--rate",
                "0.04",
                "--vol",
                "0.22",
                "--future",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    expected = price(Inputs.on_future(100.0, 95.0, 0.5, 0.04, 0.22), OptionType.CALL)
    assert f"{expected:.10f}" in out
    # With zero carry the forward is the quoted price of the future itself.
    assert "100.0000000000" in out


def test_dividend_flag_subtracts_from_the_rate(capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        main(
            [
                "price",
                "--spot",
                "100",
                "--strike",
                "95",
                "--time",
                "0.5",
                "--rate",
                "0.04",
                "--vol",
                "0.22",
                "--dividend",
                "0.015",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    expected = price(Inputs.with_dividend(100.0, 95.0, 0.5, 0.04, 0.22, 0.015), OptionType.CALL)
    assert f"{expected:.10f}" in out


def test_conflicting_carry_flags_are_refused() -> None:
    with pytest.raises(SystemExit):
        main(
            [
                "price",
                "--spot",
                "100",
                "--strike",
                "95",
                "--time",
                "0.5",
                "--rate",
                "0.04",
                "--vol",
                "0.22",
                "--future",
                "--dividend",
                "0.01",
            ]
        )


def test_greeks_prints_every_sensitivity(capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        main(
            [
                "greeks",
                "--spot",
                "100",
                "--strike",
                "95",
                "--time",
                "0.5",
                "--rate",
                "0.04",
                "--vol",
                "0.22",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    for name in (
        "delta",
        "gamma",
        "vega",
        "theta",
        "vanna",
        "volga",
        "charm",
        "veta",
        "colour",
        "speed",
        "zomma",
        "dual delta",
        "dual gamma",
    ):
        assert name in out


def test_iv_round_trips_through_the_command_line(capsys: pytest.CaptureFixture[str]) -> None:
    quoted = price(Inputs(100.0, 95.0, 0.5, 0.04, 0.22), OptionType.CALL)
    assert (
        main(
            [
                "iv",
                "--spot",
                "100",
                "--strike",
                "95",
                "--time",
                "0.5",
                "--rate",
                "0.04",
                "--price",
                repr(quoted),
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    recovered = float(out.splitlines()[0].split()[-1])
    assert recovered == pytest.approx(0.22, rel=1e-9)
    assert "newton" in out


def test_iv_can_be_asked_for_brent(capsys: pytest.CaptureFixture[str]) -> None:
    quoted = price(Inputs(100.0, 95.0, 0.5, 0.04, 0.22), OptionType.CALL)
    assert (
        main(
            [
                "iv",
                "--spot",
                "100",
                "--strike",
                "95",
                "--time",
                "0.5",
                "--rate",
                "0.04",
                "--price",
                repr(quoted),
                "--brent",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "brent" in out
    assert float(out.splitlines()[0].split()[-1]) == pytest.approx(0.22, rel=1e-9)


def test_iv_reports_an_unreachable_quote_without_a_traceback(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        main(
            [
                "iv",
                "--spot",
                "100",
                "--strike",
                "50",
                "--time",
                "1",
                "--rate",
                "0.05",
                "--price",
                "1.0",
            ]
        )
        == 1
    )
    captured = capsys.readouterr()
    assert "no-arbitrage" in captured.err
    assert "must lie in" in captured.err


def test_ladder_prints_one_row_per_strike(capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        main(
            [
                "ladder",
                "--spot",
                "100",
                "--time",
                "0.25",
                "--vol",
                "0.3",
                "--rate",
                "0.03",
                "--low",
                "80",
                "--high",
                "120",
                "--steps",
                "5",
            ]
        )
        == 0
    )
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 7  # header, rule, five strikes
    strikes = [float(line.split()[0]) for line in lines[2:]]
    assert strikes == [80.0, 90.0, 100.0, 110.0, 120.0]


def test_ladder_rows_agree_with_the_library(capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        main(
            [
                "ladder",
                "--spot",
                "100",
                "--time",
                "0.25",
                "--vol",
                "0.3",
                "--rate",
                "0.03",
                "--low",
                "80",
                "--high",
                "120",
                "--steps",
                "5",
            ]
        )
        == 0
    )
    lines = capsys.readouterr().out.strip().splitlines()[2:]
    for line in lines:
        strike, call, put = (float(field) for field in line.split()[:3])
        inputs = Inputs(100.0, strike, 0.25, 0.03, 0.3)
        assert call == pytest.approx(price(inputs, OptionType.CALL), abs=5e-7)
        assert put == pytest.approx(price(inputs, OptionType.PUT), abs=5e-7)


def test_invalid_market_data_exits_cleanly(capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        main(
            [
                "price",
                "--spot",
                "-100",
                "--strike",
                "95",
                "--time",
                "0.5",
                "--rate",
                "0.04",
                "--vol",
                "0.22",
            ]
        )
        == 1
    )
    assert "spot" in capsys.readouterr().err


def test_a_single_strike_ladder_does_not_divide_by_zero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        main(
            [
                "ladder",
                "--spot",
                "100",
                "--time",
                "0.25",
                "--vol",
                "0.3",
                "--low",
                "100",
                "--high",
                "100",
                "--steps",
                "1",
            ]
        )
        == 0
    )
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 3
    assert not any(math.isnan(float(f)) for f in lines[2].split())


def _american_args(*extra: str) -> list[str]:
    return [
        "american",
        "--spot",
        "100",
        "--strike",
        "100",
        "--time",
        "1",
        "--rate",
        "0.06",
        "--vol",
        "0.25",
        "--dividend",
        "0.06",
        *extra,
    ]


def test_american_reports_a_premium_over_the_european_value(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(_american_args("--put", "--steps", "200")) == 0
    out = capsys.readouterr().out
    values = {
        line.rsplit("  ", 1)[0].strip(): line.rsplit("  ", 1)[1].strip()
        for line in out.strip().splitlines()
    }
    american = float(values["american"])
    european = float(values["european (lattice)"])
    premium = float(values["early exercise premium"])
    assert american > european
    assert premium == pytest.approx(american - european, abs=1e-12)


def test_american_call_without_dividends_shows_no_premium(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The wiring must carry the carry through, or this silently shows a premium."""
    assert (
        main(
            [
                "american",
                "--spot",
                "100",
                "--strike",
                "95",
                "--time",
                "0.5",
                "--rate",
                "0.04",
                "--vol",
                "0.22",
                "--steps",
                "150",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    line = next(ln for ln in out.splitlines() if "early exercise premium" in ln)
    assert float(line.split()[-1]) == 0.0


def test_american_accepts_every_lattice(capsys: pytest.CaptureFixture[str]) -> None:
    prices = []
    for name in ("crr", "jarrow-rudd", "trinomial"):
        assert main(_american_args("--put", "--steps", "200", "--lattice", name)) == 0
        out = capsys.readouterr().out
        assert name in out
        line = next(ln for ln in out.splitlines() if ln.strip().startswith("american  "))
        prices.append(float(line.split()[-1]))
    assert max(prices) - min(prices) < 5e-3


def test_american_boundary_table_rises_towards_the_strike(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(_american_args("--put", "--steps", "200", "--boundary")) == 0
    out = capsys.readouterr().out
    body = out.split("critical spot")[1].strip().splitlines()[1:]
    levels = [float(line.split()[1]) for line in body if line.strip()]
    assert len(levels) > 3
    assert levels[-1] == pytest.approx(100.0)
    assert levels[0] < levels[-1]


def test_american_boundary_says_so_when_the_region_is_empty(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        main(
            [
                "american",
                "--spot",
                "100",
                "--strike",
                "95",
                "--time",
                "0.5",
                "--rate",
                "0.04",
                "--vol",
                "0.22",
                "--steps",
                "150",
                "--boundary",
            ]
        )
        == 0
    )
    assert "exercise region is empty" in capsys.readouterr().out


def test_american_refuses_a_layer_count_below_the_stability_floor(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        main(
            [
                "american",
                "--spot",
                "100",
                "--strike",
                "100",
                "--time",
                "5",
                "--rate",
                "0.02",
                "--vol",
                "0.05",
                "--carry",
                "0.60",
                "--steps",
                "2",
            ]
        )
        == 1
    )
    err = capsys.readouterr().err
    assert "below the" in err and "layers" in err


def test_american_reports_the_closed_form_alongside_the_lattice(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(_american_args("--put", "--steps", "200")) == 0
    out = capsys.readouterr().out
    values = {
        line.rsplit("  ", 1)[0].strip(): line.rsplit("  ", 1)[1].strip()
        for line in out.strip().splitlines()
    }
    single = float(values["bjerksund-stensland 1993"])
    double = float(values["bjerksund-stensland 2002"])
    american = float(values["american"])
    european = float(values["european (closed form)"])

    # Both approximations sit inside the bracket, and the two-step one is at
    # least as sharp -- the same inequalities the library guarantees, checked
    # through the printed output so that a formatting change cannot quietly
    # start reporting one number under the other's label.
    assert european - 1e-9 <= single <= american + 1e-3
    assert european - 1e-9 <= double <= american + 1e-3
    assert double >= single - 1e-9
    assert float(values["bs trigger price"]) < 100.0


def test_american_says_when_the_trigger_is_never_reached(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An infinite boundary must read as words, not as `inf`."""
    assert (
        main(
            [
                "american",
                "--spot",
                "100",
                "--strike",
                "95",
                "--time",
                "0.5",
                "--rate",
                "0.04",
                "--vol",
                "0.22",
                "--steps",
                "150",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "never reached" in out
    assert "inf" not in out


# ---------------------------------------------------------------------------
# term: one strike across maturities
# ---------------------------------------------------------------------------


def test_term_prints_one_row_per_maturity(capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        main(
            [
                "term",
                "--spot",
                "100",
                "--strike",
                "100",
                "--vol",
                "0.2",
                "--rate",
                "0.05",
                "--near",
                "0.1",
                "--far",
                "2.0",
                "--steps",
                "7",
            ]
        )
        == 0
    )
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 9  # header, rule, seven rows
    assert "maturity" in lines[0]


def test_term_rows_agree_with_the_library(capsys: pytest.CaptureFixture[str]) -> None:
    """The point of an end-to-end test: the numbers printed are the library's."""
    assert (
        main(
            [
                "term",
                "--spot",
                "100",
                "--strike",
                "95",
                "--vol",
                "0.25",
                "--rate",
                "0.03",
                "--near",
                "0.25",
                "--far",
                "4.0",
                "--steps",
                "5",
            ]
        )
        == 0
    )
    rows = capsys.readouterr().out.strip().splitlines()[2:]
    assert len(rows) == 5
    for row in rows:
        fields = row.split()
        time = float(fields[0])
        inputs = Inputs(100.0, 95.0, time, 0.03, 0.25)
        assert float(fields[1]) == pytest.approx(price(inputs, OptionType.CALL), abs=1e-6)
        assert float(fields[2]) == pytest.approx(price(inputs, OptionType.PUT), abs=1e-6)


def test_term_spaces_maturities_geometrically(capsys: pytest.CaptureFixture[str]) -> None:
    """Ratios, not differences.

    An even grid from a month to two years puts almost every row at the long
    end, where the term structure barely moves, and none at the short end,
    where it moves most.
    """
    assert (
        main(
            [
                "term",
                "--spot",
                "100",
                "--strike",
                "100",
                "--vol",
                "0.2",
                "--near",
                "0.1",
                "--far",
                "10.0",
                "--steps",
                "5",
            ]
        )
        == 0
    )
    rows = capsys.readouterr().out.strip().splitlines()[2:]
    times = [float(row.split()[0]) for row in rows]
    near, far, steps = 0.1, 10.0, 5
    ratio = (far / near) ** (1.0 / (steps - 1))
    expected = [near * ratio**i for i in range(steps)]
    # The column prints four decimals, so the comparison is to that precision
    # rather than to the float the program actually computed.
    for shown, want in zip(times, expected, strict=True):
        assert shown == pytest.approx(want, abs=1e-4)

    # The distinction that matters: the middle row is the geometric mean of the
    # endpoints, not the arithmetic one. An even grid would put it at 5.05.
    assert times[2] == pytest.approx(math.sqrt(near * far), abs=1e-4)


def test_term_accepts_a_single_maturity(capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        main(
            [
                "term",
                "--spot",
                "100",
                "--strike",
                "100",
                "--vol",
                "0.2",
                "--near",
                "1.0",
                "--far",
                "1.0",
                "--steps",
                "1",
            ]
        )
        == 0
    )
    rows = capsys.readouterr().out.strip().splitlines()[2:]
    assert len(rows) == 1
    assert float(rows[0].split()[0]) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "extra",
    [
        ["--near", "0", "--far", "1"],
        ["--near", "-1", "--far", "1"],
        ["--near", "2", "--far", "1"],
        ["--near", "0.1", "--far", "1", "--steps", "0"],
    ],
)
def test_term_rejects_a_nonsensical_range(extra: list[str]) -> None:
    with pytest.raises(SystemExit):
        main(["term", "--spot", "100", "--strike", "100", "--vol", "0.2", *extra])


# ---------------------------------------------------------------------------
# surface: fit a whole surface and report the arbitrage it admits
# ---------------------------------------------------------------------------


def _quote_file(
    path: pathlib.Path,
    slices: dict[float, tuple[float, float, float, float, float]],
    *,
    spot: float = 100.0,
    rate: float = 0.05,
    header: bool = True,
) -> pathlib.Path:
    """Write quotes generated from known SVI slices.

    Generating the quotes from slices whose parameters are known turns the
    whole command into a round trip: strike and volatility go in, the CLI
    converts to log-moneyness and total variance, fits, and the parameters that
    come back out must be the ones that went in. That checks the coordinate
    conversion inside the command, which no unit test of the fit can reach.
    """
    lines = ["maturity,strike,vol"] if header else []
    for time, (a, b, rho, m, s) in slices.items():
        forward_price = spot * math.exp(rate * time)
        for strike in (70.0, 80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0, 140.0):
            y = math.log(strike / forward_price) - m
            total = a + b * (rho * y + math.hypot(y, s))
            lines.append(f"{time},{strike},{math.sqrt(total / time):.12f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _local_vol_grid(out: str) -> tuple[list[float], list[list[str]]]:
    """Pull the column maturities and the data rows out of the local-vol table."""
    body = out.split("local volatility by the Dupire identity")[1].strip().splitlines()
    columns = [float(field) for field in body[0].split()[1:]]
    rows = []
    for line in body[2:]:
        fields = line.split()
        if not fields or not fields[0].lstrip("-").replace(".", "", 1).isdigit():
            break  # the table has ended and the verdict has begun
        rows.append(fields)
    return columns, rows


ORDERED_SLICES = {
    0.25: (0.020, 0.15, -0.35, 0.0, 0.12),
    1.00: (0.050, 0.22, -0.30, 0.0, 0.18),
    2.00: (0.100, 0.30, -0.25, 0.0, 0.25),
}

# The same slices with their maturities swapped, so total variance falls as
# maturity grows: a calendar arbitrage, and a certain profit.
CROSSED_SLICES = {
    0.50: (0.050, 0.22, -0.30, 0.0, 0.18),
    1.50: (0.020, 0.15, -0.35, 0.0, 0.12),
}


def test_surface_recovers_the_parameters_the_quotes_were_built_from(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The end-to-end round trip, and the sharpest check on the command.

    The quotes are volatilities against strikes; the command converts them to
    total variance against log-moneyness on the forward and fits. If the
    conversion were wrong in any way -- the wrong forward, a missing maturity
    scaling, a sign -- the fit would still succeed and the recovered parameters
    would be wrong. They come back to eight decimals.
    """
    path = _quote_file(tmp_path / "quotes.csv", ORDERED_SLICES)
    assert main(["surface", "--quotes", str(path), "--spot", "100", "--rate", "0.05"]) == 0

    out = capsys.readouterr().out
    rows = [
        line.split() for line in out.splitlines() if line.startswith(("   0.2", "   1.0", "   2.0"))
    ]
    assert len(rows) == 3
    for row, (time, (a, b, rho, _m, s)) in zip(rows, sorted(ORDERED_SLICES.items()), strict=True):
        assert float(row[0]) == pytest.approx(time)
        assert float(row[4]) == pytest.approx(a, abs=1e-5)
        assert float(row[5]) == pytest.approx(b, abs=1e-5)
        assert float(row[6]) == pytest.approx(rho, abs=1e-5)
        assert float(row[8]) == pytest.approx(s, abs=1e-5)


def test_surface_reports_no_arbitrage_on_an_ordered_surface(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _quote_file(tmp_path / "quotes.csv", ORDERED_SLICES)
    assert main(["surface", "--quotes", str(path), "--spot", "100", "--rate", "0.05"]) == 0
    out = capsys.readouterr().out
    assert "verdict: no arbitrage found" in out
    assert "CROSSES" not in out
    assert "ARBITRAGE" not in out


def test_surface_exits_non_zero_on_a_calendar_crossing(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A finding about the data, reported in the exit status so a pipeline can act."""
    path = _quote_file(tmp_path / "crossed.csv", CROSSED_SLICES)
    assert main(["surface", "--quotes", str(path), "--spot", "100", "--rate", "0.05"]) == 1
    out = capsys.readouterr().out
    assert "CROSSES" in out
    assert "verdict: the fitted surface admits arbitrage" in out


def test_surface_checks_interpolated_maturities_as_well_as_quoted_ones(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _quote_file(tmp_path / "quotes.csv", ORDERED_SLICES)
    assert main(["surface", "--quotes", str(path), "--spot", "100", "--rate", "0.05"]) == 0
    out = capsys.readouterr().out
    assert "interpolated" in out
    assert out.count("quoted") == 3


def test_surface_reads_standard_input(
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _quote_file(tmp_path / "quotes.csv", ORDERED_SLICES)
    monkeypatch.setattr("sys.stdin", io.StringIO(path.read_text(encoding="utf-8")))
    assert main(["surface", "--quotes", "-", "--spot", "100", "--rate", "0.05"]) == 0
    assert "verdict: no arbitrage found" in capsys.readouterr().out


def test_surface_accepts_a_file_without_a_header(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The header is optional and detected by trying to parse it, not by its spelling."""
    path = _quote_file(tmp_path / "bare.csv", ORDERED_SLICES, header=False)
    assert main(["surface", "--quotes", str(path), "--spot", "100", "--rate", "0.05"]) == 0
    out = capsys.readouterr().out
    assert out.count("quoted") == 3


def test_surface_prints_a_local_volatility_grid(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _quote_file(tmp_path / "quotes.csv", ORDERED_SLICES)
    assert (
        main(["surface", "--quotes", str(path), "--spot", "100", "--rate", "0.05", "--local-vol"])
        == 0
    )
    out = capsys.readouterr().out
    assert "local volatility by the Dupire identity" in out
    _, rows = _local_vol_grid(out)
    for row in rows:
        for cell in row[1:]:
            # An inadmissible cell prints as "--"; every cell here has an answer.
            assert float(cell) > 0.0


def test_the_local_volatility_grid_stays_inside_the_quoted_maturities(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Regression: the grid used to include the endpoints and print zeros there.

    At a quoted endpoint the surface is flat in maturity by construction, so
    ``dw/dT`` is zero and the identity returns a local volatility of zero. That
    is a property of the extrapolation rule and not of the market, and printing
    it under a column headed by a real maturity invited it to be read as one.
    """
    path = _quote_file(tmp_path / "quotes.csv", ORDERED_SLICES)
    assert (
        main(["surface", "--quotes", str(path), "--spot", "100", "--rate", "0.05", "--local-vol"])
        == 0
    )
    columns, rows = _local_vol_grid(capsys.readouterr().out)
    assert all(0.25 < time < 2.0 for time in columns), columns
    for row in rows:
        for cell in row[1:]:
            assert float(cell) > 0.01


@pytest.mark.parametrize(
    ("contents", "match"),
    [
        ("0.5,100\n0.5,110\n", "expected maturity, strike and vol"),
        ("0.5,100,0.2\nnot,a,number\n", "are not three numbers"),
        ("0.5,100,0.2\n-1,100,0.2\n", "maturity -1.0 is not positive"),
        ("0.5,100,0.2\n0.5,0,0.2\n", "strike 0.0 is not positive"),
        ("0.5,100,0.2\n0.5,100,-0.2\n", "volatility -0.2 is not positive"),
        ("\n\n", "no quotes found"),
    ],
)
def test_surface_rejects_a_malformed_file(
    tmp_path: pathlib.Path, contents: str, match: str
) -> None:
    """Every diagnostic names the row, because "one of them is wrong" is not usable."""
    path = tmp_path / "bad.csv"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(SystemExit, match=match):
        main(["surface", "--quotes", str(path), "--spot", "100"])


def test_surface_reports_a_missing_file_rather_than_raising(tmp_path: pathlib.Path) -> None:
    with pytest.raises(SystemExit, match="cannot read"):
        main(["surface", "--quotes", str(tmp_path / "absent.csv"), "--spot", "100"])


def test_surface_needs_five_quotes_at_each_maturity(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "thin.csv"
    path.write_text(
        "maturity,strike,vol\n" + "".join(f"1.0,{k},0.2\n" for k in (90, 95, 100, 105)),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="needs at least five"):
        main(["surface", "--quotes", str(path), "--spot", "100"])


def test_surface_handles_a_single_maturity(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """One slice cannot cross another, and gives no slope in maturity to differentiate."""
    path = _quote_file(tmp_path / "one.csv", {1.0: ORDERED_SLICES[1.00]})
    assert (
        main(["surface", "--quotes", str(path), "--spot", "100", "--rate", "0.05", "--local-vol"])
        == 0
    )
    out = capsys.readouterr().out
    assert "a single maturity cannot cross another" in out
    assert "dw/dT is zero" in out
    assert "verdict: no arbitrage found" in out


def test_surface_ignores_comment_lines(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _quote_file(tmp_path / "quotes.csv", ORDERED_SLICES)
    body = path.read_text(encoding="utf-8")
    path.write_text("# a note about where these came from\n" + body, encoding="utf-8")
    assert main(["surface", "--quotes", str(path), "--spot", "100", "--rate", "0.05"]) == 0
    assert "verdict: no arbitrage found" in capsys.readouterr().out


LADDER_ROWS = ("80.", "90.", "100.", "110.", "120.")

HESTON_ARGS = [
    "heston",
    "--spot",
    "100",
    "--time",
    "1.0",
    "--future",
    "--v0",
    "0.04",
    "--kappa",
    "1.5768",
    "--theta",
    "0.04",
    "--sigma",
    "0.5751",
    "--rho",
    "-0.5711",
    "--low",
    "80",
    "--high",
    "120",
    "--steps",
    "5",
]


def test_heston_prints_a_smile(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(HESTON_ARGS) == 0
    out = capsys.readouterr().out
    assert "implied vol" in out
    assert "Feller condition  fails" in out
    # One row per strike, plus the header, the rule and the four summary lines.
    rows = [line for line in out.splitlines() if line.strip().startswith(LADDER_ROWS)]
    assert len(rows) == 5


def test_heston_prices_each_strike_out_of_the_money(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(HESTON_ARGS) == 0
    rows = [
        line.split()
        for line in capsys.readouterr().out.splitlines()
        if line.strip().startswith(LADDER_ROWS)
    ]
    sides = [row[1] for row in rows]
    assert sides == ["put", "put", "call", "call", "call"]


def test_heston_reproduces_the_library_prices(capsys: pytest.CaptureFixture[str]) -> None:
    from moneyness.heston import Contract, Heston
    from moneyness.heston import price as heston_price

    assert main(HESTON_ARGS) == 0
    rows = [
        line.split()
        for line in capsys.readouterr().out.splitlines()
        if line.strip().startswith(LADDER_ROWS)
    ]
    model = Heston(0.04, 1.5768, 0.04, 0.5751, -0.5711)
    for row in rows:
        strike, side, value = float(row[0]), row[1], float(row[2])
        contract = Contract(100.0, strike, 1.0, 0.0, carry=0.0)
        want = heston_price(model, contract, OptionType(side))
        assert value == pytest.approx(want, abs=1e-8)


def test_heston_reports_the_agreement_between_the_two_routes(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main([*HESTON_ARGS, "--check"]) == 0
    out = capsys.readouterr().out
    assert "largest gap between the Lewis and Gil-Pelaez routes" in out
    gap = float(out.rsplit(":", 1)[1])
    assert gap < 1e-9


def test_heston_smile_is_downward_sloping_at_negative_correlation(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(HESTON_ARGS) == 0
    vols = [
        float(line.split()[3].rstrip("%"))
        for line in capsys.readouterr().out.splitlines()
        if line.strip().startswith(LADDER_ROWS)
    ]
    assert vols == sorted(vols, reverse=True)


def test_heston_refuses_a_bad_ladder(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main([*HESTON_ARGS[:-4], "--low", "0", "--high", "120"])


def test_heston_refuses_a_bad_parameter(capsys: pytest.CaptureFixture[str]) -> None:
    args = list(HESTON_ARGS)
    args[args.index("--kappa") + 1] = "0"
    assert main(args) == 1
    assert "kappa must be positive" in capsys.readouterr().err


PATH_ARGS = [
    "heston-path",
    "--spot",
    "100",
    "--strike",
    "100",
    "--time",
    "1.0",
    "--future",
    "--v0",
    "0.04",
    "--kappa",
    "1.5768",
    "--theta",
    "0.04",
    "--sigma",
    "0.5751",
    "--rho",
    "-0.5711",
    "--steps",
    "8",
    "--paths",
    "4000",
]


def test_heston_path_reports_the_gap_to_the_transform(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(PATH_ARGS) == 0
    out = capsys.readouterr().out
    assert "transform price" in out
    assert "standard error" in out
    deviations = float(out.rsplit("(", 1)[1].split()[0])
    assert abs(deviations) < 4.0


def test_heston_path_prices_an_asian(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([*PATH_ARGS, "--payoff", "asian"]) == 0
    out = capsys.readouterr().out
    assert "payoff            asian call" in out
    # No closed form to compare against, so no gap line is printed.
    assert "transform price" not in out
    value = float(out.split("value             ")[1].split()[0])
    assert 0.0 < value < 100.0


def test_heston_path_prices_a_barrier_below_the_vanilla(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main([*PATH_ARGS, "--payoff", "barrier", "--level", "85"]) == 0
    knock_out = float(capsys.readouterr().out.split("value             ")[1].split()[0])
    assert main(PATH_ARGS) == 0
    vanilla = float(capsys.readouterr().out.split("value             ")[1].split()[0])
    assert 0.0 < knock_out < vanilla


def test_heston_path_needs_a_level_for_a_barrier() -> None:
    with pytest.raises(SystemExit, match="--level is required"):
        main([*PATH_ARGS, "--payoff", "barrier"])


def test_heston_path_refuses_a_bad_step_count() -> None:
    args = list(PATH_ARGS)
    args[args.index("--steps") + 1] = "0"
    with pytest.raises(SystemExit, match="--steps must be at least 1"):
        main(args)


def test_heston_path_reports_a_bad_parameter_as_an_exit_status(
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = list(PATH_ARGS)
    args[args.index("--rho") + 1] = "-2"
    assert main(args) == 1
    assert "rho must be in" in capsys.readouterr().err


VARIANCE_ARGS = ["variance", "--forward", "102.37", "--time", "1.0", "--rate", "0.03"]


def test_variance_recovers_a_flat_volatility(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([*VARIANCE_ARGS, "--vol", "0.2", "--width", "2.0"]) == 0
    out = capsys.readouterr().out
    assert "continuous replication" in out
    fair = float(out.split("fair variance  ")[1].split()[0])
    assert fair == pytest.approx(0.04, rel=1e-12)
    # The exact answer and the error against it are both printed, which is the
    # point of offering a flat volatility as a source at all.
    assert "exact answer   0.0400000000" in out
    error = float(out.split("relative error ")[1].split()[0])
    assert abs(error) < 1e-13


def test_variance_prints_the_truncation_prediction(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main([*VARIANCE_ARGS, "--vol", "0.2", "--width", "0.3"]) == 0
    out = capsys.readouterr().out
    predicted = float(out.split("predicted truncation ")[1].split()[0])
    measured = float(out.split("relative error ")[1].split()[0]) * 0.04
    assert predicted < 0.0
    # Both are printed to four significant figures, so that is as closely as
    # they can be compared through the output; the tight comparison lives in
    # the unit tests.
    assert measured == pytest.approx(predicted, rel=1e-3)


def test_variance_tabulates_the_three_centrings(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main([*VARIANCE_ARGS, "--vol", "0.2", "--width", "2.0", "--step", "5.0"]) == 0
    out = capsys.readouterr().out
    assert "listed strip of" in out
    values = {}
    for line in out.split("listed strip")[1].splitlines():
        fields = line.split()
        if len(fields) >= 3 and fields[0] in {"none", "quadratic", "exact"}:
            values[fields[0]] = float(fields[1])
    assert set(values) == {"none", "quadratic", "exact"}
    # The split strike is 100 against a forward of 102.37, so the uncorrected
    # sum is the highest of the three and both corrections pull it down.
    assert values["none"] > values["quadratic"]
    assert values["none"] > values["exact"]
    assert "split strike   100.000000" in out


def test_variance_under_heston_matches_the_closed_form(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        main(
            [
                "variance",
                "--forward",
                "100",
                "--time",
                "1.0",
                "--rate",
                "0.02",
                "--heston",
                "--v0",
                "0.09",
                "--kappa",
                "2.0",
                "--theta",
                "0.04",
                "--sigma",
                "0.8",
                "--rho",
                "-0.5",
                "--width",
                "6.0",
                "--tol",
                "1e-12",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    error = float(out.split("relative error ")[1].split()[0])
    assert abs(error) < 1e-7
    # The volatility swap block reports a strike strictly below the square root
    # of the fair variance, and says that the correction is too large.
    assert "volatility swap" in out
    root = float(out.split("root of variance ")[1].split()[0].rstrip("%"))
    second = float(out.split("second-order     ")[1].split()[0].rstrip("%"))
    assert 0.0 < second < root
    assert "too large" in out


def test_variance_reads_a_quote_file(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    forward, time, rate, vol = 99.5, 0.5, 0.01, 0.25
    strikes = [60.0 + 5.0 * index for index in range(21)]
    reference = max(k for k in strikes if k <= forward)
    rows = ["# out-of-the-money prices", "strike,price"]
    for strike in strikes:
        put = price(Inputs.on_future(forward, strike, time, rate, vol), OptionType.PUT)
        call = price(Inputs.on_future(forward, strike, time, rate, vol), OptionType.CALL)
        if strike < reference:
            quote = put
        elif strike > reference:
            quote = call
        else:
            quote = 0.5 * (put + call)
        rows.append(f"{strike},{quote:.12f}")
    source = tmp_path / "quotes.csv"
    source.write_text("\n".join(rows) + "\n", encoding="utf-8")

    assert (
        main(
            [
                "variance",
                "--forward",
                "99.5",
                "--time",
                "0.5",
                "--rate",
                "0.01",
                "--quotes",
                str(source),
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "listed strip of 21 strikes" in out
    # No exact answer is known for a quote file, so no error column is printed.
    assert "exact answer" not in out
    assert "error" not in out.split("listed strip")[1]


def test_variance_reads_quotes_from_standard_input(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = "strike,price\n80,0.4\n90,1.2\n100,3.0\n110,1.1\n120,0.3\n"
    monkeypatch.setattr("sys.stdin", io.StringIO(rows))
    assert main(["variance", "--forward", "102", "--time", "1.0", "--quotes", "-"]) == 0
    assert "listed strip of 5 strikes" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ([], "exactly one of"),
        (["--vol", "0.2", "--heston"], "exactly one of"),
        (["--vol", "-0.2"], "--vol must be positive"),
        (["--vol", "0.2", "--width", "0"], "--width must be positive"),
        (["--vol", "0.2", "--step", "0"], "--step must be positive"),
        (["--heston", "--v0", "0.04"], "--heston needs --kappa"),
    ],
)
def test_variance_refuses_bad_arguments(extra: list[str], message: str) -> None:
    with pytest.raises(SystemExit, match=message):
        main([*VARIANCE_ARGS, *extra])


@pytest.mark.parametrize(
    ("time", "forward", "message"),
    [("0", "100", "--time must be positive"), ("1", "0", "--forward must be positive")],
)
def test_variance_refuses_a_bad_market(time: str, forward: str, message: str) -> None:
    with pytest.raises(SystemExit, match=message):
        main(["variance", "--forward", forward, "--time", time, "--vol", "0.2"])


def test_variance_refuses_a_strip_that_misses_the_forward(
    tmp_path: pathlib.Path,
) -> None:
    source = tmp_path / "high.csv"
    source.write_text("strike,price\n110,1.0\n120,0.5\n130,0.2\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="bracket the forward"):
        main(["variance", "--forward", "100", "--time", "1", "--quotes", str(source)])


def test_variance_reports_a_missing_quote_file() -> None:
    with pytest.raises(SystemExit, match="cannot read"):
        main(["variance", "--forward", "100", "--time", "1", "--quotes", "nowhere.csv"])


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ("strike\n100\n110\n", "expected strike and price"),
        ("strike,price\n-5,1.0\n100,2.0\n110,1.0\n", "is not positive"),
        ("strike,price\n90,1.0\n100,-2.0\n110,1.0\n", "is negative"),
        ("strike,price\n90,1.0\nbad,rows\n", "are not two numbers"),
        ("# only a comment\n", "no quotes found"),
    ],
)
def test_variance_reports_a_malformed_quote_row(
    tmp_path: pathlib.Path, rows: str, message: str
) -> None:
    source = tmp_path / "rows.csv"
    source.write_text(rows, encoding="utf-8")
    with pytest.raises(SystemExit, match=message):
        main(["variance", "--forward", "100", "--time", "1", "--quotes", str(source)])


def test_surface_reprices_its_own_quotes_through_the_local_volatility(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The end-to-end check the two arbitrage conditions do not make.

    Those ask whether a local volatility exists. This asks whether the one
    that does reproduces the quotes it came from, which is a different
    question and the one a user of a fitted surface actually has.
    """
    path = _quote_file(tmp_path / "quotes.csv", ORDERED_SLICES)
    assert (
        main(["surface", "--quotes", str(path), "--spot", "100", "--rate", "0.05", "--reprice"])
        == 0
    )
    out = capsys.readouterr().out
    assert "repriced through the local volatility" in out
    worst = [line for line in out.splitlines() if line.startswith("worst gap")]
    assert len(worst) == 1
    assert float(worst[0].split(":")[1]) < 5e-3
    # Every quote got an answer rather than a dash.
    assert " --  " not in out


def _barrier_argv(*extra: str) -> list[str]:
    """The base command, with later flags overriding the defaults here.

    argparse takes the last occurrence, so an extra ``--level`` wins over the
    one below without the helper needing to know which flags were overridden.
    """
    return [
        "barrier",
        "--spot",
        "100",
        "--strike",
        "100",
        "--time",
        "1",
        "--rate",
        "0.05",
        "--dividend",
        "0.03",
        "--vol",
        "0.20",
        "--level",
        "120",
        "--barrier",
        "up-and-out",
        *extra,
    ]


def test_barrier_reports_both_halves_and_the_parity_residual(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(_barrier_argv()) == 0
    out = capsys.readouterr().out
    assert "up-and-out call at 120" in out
    assert "up-and-in" in out
    assert "parity residual" in out
    # The residual is printed in exponent form and has to be at rounding.
    line = next(one for one in out.splitlines() if "parity residual" in one)
    assert abs(float(line.split()[-1])) < 1e-12


def test_barrier_shows_the_price_turning_over_in_volatility(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The ladder is the report's reason for existing, so its shape is asserted.

    The knock-out falls across this range while the vanilla rises, and the
    command says so in words rather than leaving the reader to notice.
    """
    assert main(_barrier_argv("--vols", "0.10", "0.20", "0.30")) == 0
    out = capsys.readouterr().out
    rows = [one.split() for one in out.splitlines() if one.startswith("      0.")]
    assert len(rows) == 3
    knocked = [float(row[1]) for row in rows]
    vanillas = [float(row[2]) for row in rows]
    assert knocked == sorted(knocked, reverse=True)
    assert vanillas == sorted(vanillas)
    assert "turns over" in out


def test_barrier_prices_the_monitoring_frequencies(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(_barrier_argv("--monitorings", "252", "12")) == 0
    out = capsys.readouterr().out
    assert "over continuous" in out
    assert "+12.9" in out
    assert "+65.2" in out
    # And it says how good the correction is rather than implying it is exact.
    assert "+6.82% monthly" in out


def test_barrier_can_omit_the_monitoring_table(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(_barrier_argv("--monitorings")) == 0
    assert "over continuous" not in capsys.readouterr().out


def test_barrier_names_a_structurally_worthless_contract(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Zero with a reason beats zero on its own."""
    assert (
        main(
            [
                "barrier",
                "--spot",
                "100",
                "--strike",
                "130",
                "--time",
                "1",
                "--rate",
                "0.05",
                "--vol",
                "0.20",
                "--level",
                "120",
                "--barrier",
                "up-and-out",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "worth exactly nothing" in out
    assert "requires breaching the barrier" in out


def test_barrier_refuses_a_level_that_is_not_one() -> None:
    with pytest.raises(SystemExit):
        main(_barrier_argv("--level", "0"))
    with pytest.raises(SystemExit):
        main(_barrier_argv("--monitorings", "0"))


def test_asian_reports_the_interval_before_the_price(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The bounds lead, because an interval is what can be stated about an average."""
    assert (
        main(
            [
                "asian",
                "--spot",
                "100",
                "--strike",
                "100",
                "--time",
                "1",
                "--rate",
                "0.05",
                "--vol",
                "0.2",
                "--fixings",
                "12",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "lower (Curran)" in out
    assert "interval width" in out
    assert "below by AM-GM" in out
    assert "moment matched" in out
    assert "(exact)" in out
    # the average is less volatile than the terminal price, so the Asian is cheaper
    labelled: dict[str, str] = {}
    for line in out.splitlines():
        match = re.match(r"\s*(\S.*?)\s\s+(\S+)", line)
        if match is not None:
            labelled[match.group(1)] = match.group(2)
    assert float(labelled["lower (Curran)"]) < float(labelled["vanilla"])
    assert float(labelled["effective vol"]) < 0.2


def test_asian_flags_a_moment_matched_price_outside_the_bounds(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The violation is the point of computing the bounds, so it has to be visible."""
    assert (
        main(
            [
                "asian",
                "--spot",
                "100",
                "--strike",
                "120",
                "--time",
                "1",
                "--rate",
                "0.05",
                "--vol",
                "0.2",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "outside the bounds, below the lower bound by" in out


def test_asian_puts_report_the_geometric_price_as_an_upper_bound(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        main(
            [
                "asian",
                "--put",
                "--spot",
                "100",
                "--strike",
                "100",
                "--time",
                "1",
                "--rate",
                "0.05",
                "--vol",
                "0.2",
            ]
        )
        == 0
    )
    assert "above by AM-GM" in capsys.readouterr().out


def test_asian_fixing_table_shows_the_bounds_tightening(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        main(
            [
                "asian",
                "--fixing-table",
                "--spot",
                "100",
                "--strike",
                "100",
                "--time",
                "1",
                "--rate",
                "0.05",
                "--vol",
                "0.2",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "fixings" in out
    rows = [line.split() for line in out.splitlines() if line.strip() and line.split()[0].isdigit()]
    assert [row[0] for row in rows] == ["1", "2", "4", "12", "52", "252"]
    # a call on a denser average is cheaper, and the single fixing is the vanilla
    lowers = [float(row[1]) for row in rows]
    assert lowers == sorted(lowers, reverse=True)


def test_spread_reports_the_interval_and_flags_kirk_outside_it(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The flag is the point of the command, so the case shown is one that trips it.

    At a correlation of 0.9 and a strike of 1 Kirk reads below the
    sub-replicating lower bound, which no price can do.
    """
    assert (
        main(
            [
                "spread",
                "--forward1",
                "100",
                "--forward2",
                "95",
                "--strike",
                "1",
                "--time",
                "1",
                "--rate",
                "0.03",
                "--vol1",
                "0.30",
                "--vol2",
                "0.25",
                "--rho",
                "0.9",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "lower (half-space)" in out
    assert "upper (vanillas)" in out
    assert "outside the bounds, below the lower bound" in out
    assert "exact, Margrabe" in out


def test_spread_puts_price_and_stay_inside_the_interval(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        main(
            [
                "spread",
                "--forward1",
                "100",
                "--forward2",
                "95",
                "--strike",
                "5",
                "--time",
                "1",
                "--rate",
                "0.03",
                "--vol1",
                "0.30",
                "--vol2",
                "0.25",
                "--rho",
                "-0.4",
                "--option",
                "put",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "put on S1 - S2" in out
    assert "outside the bounds" not in out


def test_spread_reports_a_pair_it_cannot_price(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A total volatility past the quadrature's ceiling is a failure, not a number."""
    assert (
        main(
            [
                "spread",
                "--forward1",
                "100",
                "--forward2",
                "95",
                "--strike",
                "5",
                "--time",
                "100",
                "--rate",
                "0",
                "--vol1",
                "5.0",
                "--vol2",
                "0.3",
                "--rho",
                "0.9",
            ]
        )
        == 1
    )
    captured = capsys.readouterr()
    assert "cannot price" in captured.err


def test_spread_at_a_zero_strike_closes_its_interval(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Both sides are then the exchange option, so the width prints as zero."""
    assert (
        main(
            [
                "spread",
                "--forward1",
                "100",
                "--forward2",
                "95",
                "--strike",
                "0",
                "--time",
                "1",
                "--rate",
                "0.03",
                "--vol1",
                "0.30",
                "--vol2",
                "0.25",
                "--rho",
                "0.3",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    width = next(float(line.split()[-1]) for line in out.splitlines() if "interval width" in line)
    assert abs(width) < 1e-9
