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

import pytest

from moneyness.bsm import Inputs, OptionType, price
from moneyness.cli import main


def test_price_reports_the_model_price(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["price", "--spot", "100", "--strike", "95", "--time", "0.5",
                 "--rate", "0.04", "--vol", "0.22"]) == 0
    out = capsys.readouterr().out
    expected = price(Inputs(100.0, 95.0, 0.5, 0.04, 0.22), OptionType.CALL)
    assert f"{expected:.10f}" in out
    assert "forward" in out


def test_put_flag_selects_the_put(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["price", "--spot", "100", "--strike", "95", "--time", "0.5",
                 "--rate", "0.04", "--vol", "0.22", "--put"]) == 0
    out = capsys.readouterr().out
    expected = price(Inputs(100.0, 95.0, 0.5, 0.04, 0.22), OptionType.PUT)
    assert f"{expected:.10f}" in out
    assert "put" in out


def test_future_flag_zeroes_the_carry(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["price", "--spot", "100", "--strike", "95", "--time", "0.5",
                 "--rate", "0.04", "--vol", "0.22", "--future"]) == 0
    out = capsys.readouterr().out
    expected = price(Inputs.on_future(100.0, 95.0, 0.5, 0.04, 0.22), OptionType.CALL)
    assert f"{expected:.10f}" in out
    # With zero carry the forward is the quoted price of the future itself.
    assert "100.0000000000" in out


def test_dividend_flag_subtracts_from_the_rate(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["price", "--spot", "100", "--strike", "95", "--time", "0.5",
                 "--rate", "0.04", "--vol", "0.22", "--dividend", "0.015"]) == 0
    out = capsys.readouterr().out
    expected = price(
        Inputs.with_dividend(100.0, 95.0, 0.5, 0.04, 0.22, 0.015), OptionType.CALL
    )
    assert f"{expected:.10f}" in out


def test_conflicting_carry_flags_are_refused() -> None:
    with pytest.raises(SystemExit):
        main(["price", "--spot", "100", "--strike", "95", "--time", "0.5",
              "--rate", "0.04", "--vol", "0.22", "--future", "--dividend", "0.01"])


def test_greeks_prints_every_sensitivity(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["greeks", "--spot", "100", "--strike", "95", "--time", "0.5",
                 "--rate", "0.04", "--vol", "0.22"]) == 0
    out = capsys.readouterr().out
    for name in (
        "delta", "gamma", "vega", "theta", "vanna", "volga", "charm",
        "veta", "colour", "speed", "zomma", "dual delta", "dual gamma",
    ):
        assert name in out


def test_iv_round_trips_through_the_command_line(capsys: pytest.CaptureFixture[str]) -> None:
    quoted = price(Inputs(100.0, 95.0, 0.5, 0.04, 0.22), OptionType.CALL)
    assert main(["iv", "--spot", "100", "--strike", "95", "--time", "0.5",
                 "--rate", "0.04", "--price", repr(quoted)]) == 0
    out = capsys.readouterr().out
    recovered = float(out.splitlines()[0].split()[-1])
    assert recovered == pytest.approx(0.22, rel=1e-9)
    assert "newton" in out


def test_iv_can_be_asked_for_brent(capsys: pytest.CaptureFixture[str]) -> None:
    quoted = price(Inputs(100.0, 95.0, 0.5, 0.04, 0.22), OptionType.CALL)
    assert main(["iv", "--spot", "100", "--strike", "95", "--time", "0.5",
                 "--rate", "0.04", "--price", repr(quoted), "--brent"]) == 0
    out = capsys.readouterr().out
    assert "brent" in out
    assert float(out.splitlines()[0].split()[-1]) == pytest.approx(0.22, rel=1e-9)


def test_iv_reports_an_unreachable_quote_without_a_traceback(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["iv", "--spot", "100", "--strike", "50", "--time", "1",
                 "--rate", "0.05", "--price", "1.0"]) == 1
    captured = capsys.readouterr()
    assert "no-arbitrage" in captured.err
    assert "must lie in" in captured.err


def test_ladder_prints_one_row_per_strike(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["ladder", "--spot", "100", "--time", "0.25", "--vol", "0.3",
                 "--rate", "0.03", "--low", "80", "--high", "120", "--steps", "5"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 7  # header, rule, five strikes
    strikes = [float(line.split()[0]) for line in lines[2:]]
    assert strikes == [80.0, 90.0, 100.0, 110.0, 120.0]


def test_ladder_rows_agree_with_the_library(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["ladder", "--spot", "100", "--time", "0.25", "--vol", "0.3",
                 "--rate", "0.03", "--low", "80", "--high", "120", "--steps", "5"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()[2:]
    for line in lines:
        strike, call, put = (float(field) for field in line.split()[:3])
        inputs = Inputs(100.0, strike, 0.25, 0.03, 0.3)
        assert call == pytest.approx(price(inputs, OptionType.CALL), abs=5e-7)
        assert put == pytest.approx(price(inputs, OptionType.PUT), abs=5e-7)


def test_invalid_market_data_exits_cleanly(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["price", "--spot", "-100", "--strike", "95", "--time", "0.5",
                 "--rate", "0.04", "--vol", "0.22"]) == 1
    assert "spot" in capsys.readouterr().err


def test_a_single_strike_ladder_does_not_divide_by_zero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["ladder", "--spot", "100", "--time", "0.25", "--vol", "0.3",
                 "--low", "100", "--high", "100", "--steps", "1"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 3
    assert not any(math.isnan(float(f)) for f in lines[2].split())


def _american_args(*extra: str) -> list[str]:
    return [
        "american", "--spot", "100", "--strike", "100", "--time", "1",
        "--rate", "0.06", "--vol", "0.25", "--dividend", "0.06", *extra,
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
    assert main(["american", "--spot", "100", "--strike", "95", "--time", "0.5",
                 "--rate", "0.04", "--vol", "0.22", "--steps", "150"]) == 0
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
    assert main(["american", "--spot", "100", "--strike", "95", "--time", "0.5",
                 "--rate", "0.04", "--vol", "0.22", "--steps", "150", "--boundary"]) == 0
    assert "exercise region is empty" in capsys.readouterr().out


def test_american_refuses_a_layer_count_below_the_stability_floor(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["american", "--spot", "100", "--strike", "100", "--time", "5",
                 "--rate", "0.02", "--vol", "0.05", "--carry", "0.60", "--steps", "2"]) == 1
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
    approximation = float(values["bjerksund-stensland"])
    american = float(values["american"])
    european = float(values["european (closed form)"])
    assert european - 1e-9 <= approximation <= american + 1e-3
    assert float(values["bs trigger price"]) < 100.0


def test_american_says_when_the_trigger_is_never_reached(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An infinite boundary must read as words, not as `inf`."""
    assert main(["american", "--spot", "100", "--strike", "95", "--time", "0.5",
                 "--rate", "0.04", "--vol", "0.22", "--steps", "150"]) == 0
    out = capsys.readouterr().out
    assert "never reached" in out
    assert "inf" not in out


# ---------------------------------------------------------------------------
# term: one strike across maturities
# ---------------------------------------------------------------------------


def test_term_prints_one_row_per_maturity(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["term", "--spot", "100", "--strike", "100", "--vol", "0.2",
                 "--rate", "0.05", "--near", "0.1", "--far", "2.0", "--steps", "7"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 9  # header, rule, seven rows
    assert "maturity" in lines[0]


def test_term_rows_agree_with_the_library(capsys: pytest.CaptureFixture[str]) -> None:
    """The point of an end-to-end test: the numbers printed are the library's."""
    assert main(["term", "--spot", "100", "--strike", "95", "--vol", "0.25",
                 "--rate", "0.03", "--near", "0.25", "--far", "4.0", "--steps", "5"]) == 0
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
    assert main(["term", "--spot", "100", "--strike", "100", "--vol", "0.2",
                 "--near", "0.1", "--far", "10.0", "--steps", "5"]) == 0
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
    assert main(["term", "--spot", "100", "--strike", "100", "--vol", "0.2",
                 "--near", "1.0", "--far", "1.0", "--steps", "1"]) == 0
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
        line.split()
        for line in out.splitlines()
        if line.startswith(("   0.2", "   1.0", "   2.0"))
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
    assert main(
        ["surface", "--quotes", str(path), "--spot", "100", "--rate", "0.05", "--local-vol"]
    ) == 0
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
    assert main(
        ["surface", "--quotes", str(path), "--spot", "100", "--rate", "0.05", "--local-vol"]
    ) == 0
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
    assert main(
        ["surface", "--quotes", str(path), "--spot", "100", "--rate", "0.05", "--local-vol"]
    ) == 0
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
