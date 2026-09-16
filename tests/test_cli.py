"""Tests for the command-line interface.

These exercise the library end to end through the same path a user takes, which
is the only way to catch the wiring mistakes unit tests cannot see: a flag that
resolves to the wrong carry, a subcommand that never reaches its handler, an
error that escapes as a traceback instead of an exit status.
"""

from __future__ import annotations

import math

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
