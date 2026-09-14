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
