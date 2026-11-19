"""The README's known-gaps list defends itself.

A list of shortcomings goes stale first, because closing a gap is satisfying and
deleting the line that described it is not. So each line carries a predicate
that holds only while the gap is real: build the feature and its line fails,
forcing the line off the page.

Two rules decide what is allowed on the list.

**Absences, not refusals.** Much of what this package declines it declines on
purpose — a quote below intrinsic has no implied volatility, an SVI fit with a
butterfly arbitrage is reported rather than repaired — and those are documented
as behaviour where they belong. Relabelling them as gaps would read as candour
and be worth less than no list.

**Call something.** A predicate that searches for a name cannot see the feature
arriving under another one. Each predicate below calls an entry point and
observes the result, or asserts the *exact* shape of a public signature or
result type, so that a new parameter or field under any name breaks it.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import pkgutil
import re
import typing
from collections.abc import Callable
from pathlib import Path

import pytest

import moneyness
from moneyness import (
    Inputs,
    OptionType,
    heston,
    heston_mc,
    implied,
    lattice,
    monte_carlo,
    pde,
    sabr,
    svi,
)

README = Path(__file__).resolve().parent.parent / "README.md"

#: The heading the list lives under, and the heading that ends it.
SECTION = "## Known gaps"
NEXT_SECTION = "## Running the tests"


def parameters(function: Callable[..., object]) -> tuple[str, ...]:
    return tuple(inspect.signature(function).parameters)


def fields(cls: type) -> tuple[str, ...]:
    return tuple(field.name for field in dataclasses.fields(cls))


def functions_producing(target: type) -> list[str]:
    """Public functions whose result is ``target`` or a dataclass holding one.

    Every module's ``__all__`` is walked, not only the root's: the root is one
    flat namespace and ``calibrate`` there is SVI's, so SABR's — and a Heston
    calibration that collided the same way — would be invisible from it.

    A calibration returns its fitted model directly or inside a result record —
    SABR's returns a ``Calibration`` whose ``parameters`` field is the fitted
    ``SabrParameters`` — so both shapes are looked for, by type, not by name.
    Names come back as ``module.function``.
    """
    found: list[str] = []
    for info in pkgutil.iter_modules(moneyness.__path__):
        module = importlib.import_module(f"moneyness.{info.name}")
        for name in getattr(module, "__all__", ()):
            obj = getattr(module, name)
            if not inspect.isfunction(obj) or obj.__module__ != module.__name__:
                continue
            result = typing.get_type_hints(obj).get("return")
            if result is target or (
                isinstance(result, type)
                and dataclasses.is_dataclass(result)
                and target in typing.get_type_hints(result).values()
            ):
                found.append(f"{info.name}.{name}")
    return found


# --- one predicate per line of the list ----------------------------------- #


def dividends_are_a_continuous_yield() -> None:
    """No input anywhere takes a dated cash amount.

    The inputs record, the dividend constructor and both early-exercise pricers
    have exactly these parameters, so a schedule under any name breaks one of
    them; and handing the constructor a schedule where the yield goes fails in
    the arithmetic rather than being read.
    """
    assert fields(Inputs) == ("spot", "strike", "time", "rate", "vol", "carry")
    assert parameters(Inputs.with_dividend) == (
        "spot",
        "strike",
        "time",
        "rate",
        "vol",
        "dividend",
    )
    assert parameters(lattice.price_lattice) == (
        "inputs",
        "option",
        "steps",
        "lattice",
        "exercise",
        "smooth",
    )
    assert parameters(pde.price_pde) == (
        "spot",
        "strike",
        "time",
        "rate",
        "vol",
        "option",
        "carry",
        "exercise",
        "mesh",
        "breakpoints",
        "barrier_level",
        "barrier_style",
    )
    schedule: typing.Any = [(0.5, 2.0)]
    with pytest.raises(TypeError):
        Inputs.with_dividend(100.0, 100.0, 1.0, 0.05, 0.2, schedule)
    # The yield form the line says does exist works on the same numbers.
    Inputs.with_dividend(100.0, 100.0, 1.0, 0.05, 0.2, 0.02)


def rates_and_carry_are_constants() -> None:
    """A rate as a function of time is refused by both families of pricer.

    The PDE solver already takes a volatility that varies with spot and time,
    which is exactly where a curve would be accepted if one were; it is called
    with such a volatility here so the refusal is about the rate.
    """
    curve: typing.Any = lambda t: 0.05  # noqa: E731
    with pytest.raises(TypeError):
        Inputs(100.0, 100.0, 1.0, curve, 0.2)
    with pytest.raises(TypeError):
        Inputs(100.0, 100.0, 1.0, 0.05, 0.2, carry=curve)
    with pytest.raises(TypeError):
        pde.price_pde(
            spot=100.0,
            strike=100.0,
            time=1.0,
            rate=curve,
            vol=lambda s, t: 0.2,
            option=OptionType.PUT,
        )
    pde.price_pde(
        spot=100.0,
        strike=100.0,
        time=1.0,
        rate=0.05,
        vol=lambda s, t: 0.2,
        option=OptionType.PUT,
    )
    assert fields(heston.Contract) == ("spot", "strike", "time", "rate", "carry")


def heston_is_priced_and_not_calibrated() -> None:
    """Nothing in the package produces a ``Heston`` from quotes.

    The scan is checked against SABR, which does calibrate, so a scan that has
    stopped seeing anything cannot pass this on its own.
    """
    assert functions_producing(heston.Heston) == []
    assert "sabr.calibrate" in functions_producing(sabr.SabrParameters)
    assert set(heston.__all__) == {
        "Branch",
        "Contract",
        "Heston",
        "SmilePoint",
        "branch_discrepancy",
        "char_func",
        "gil_pelaez_price",
        "lewis_price",
        "price",
        "smile",
    }


def heston_has_no_early_exercise() -> None:
    """Every Heston pricer prices a European, and the PDE solver cannot take one.

    The solver is the package's only route to early exercise off a lattice, and
    its volatility is a function of spot and time; a stochastic-volatility
    model is not one.
    """
    contract_pricer = ("model", "contract", "option", "tol")
    for pricer in (heston.price, heston.lewis_price, heston.gil_pelaez_price):
        assert parameters(pricer) == contract_pricer, pricer.__name__
    simulated = ("model", "contract", "option", "steps", "settings", "martingale")
    assert parameters(heston_mc.european) == simulated
    assert parameters(heston_mc.asian) == simulated
    model = heston.Heston(0.04, 1.5, 0.04, 0.5, -0.7)
    as_volatility: typing.Any = model
    with pytest.raises(TypeError, match="not callable"):
        pde.price_pde(
            spot=100.0,
            strike=100.0,
            time=1.0,
            rate=0.05,
            vol=as_volatility,
            option=OptionType.PUT,
            exercise=lattice.Exercise.AMERICAN,
        )


#: The at-the-money American put the README quotes: spot and strike 100, one
#: year, an 8% rate and 20% volatility, priced on 2,000 CRR steps.
AMERICAN_PUT = Inputs(100.0, 100.0, 1.0, 0.08, 0.20)
AMERICAN_STEPS = 2000


def implied_volatility_of_an_american_price() -> float:
    american = lattice.price_lattice(AMERICAN_PUT, OptionType.PUT, steps=AMERICAN_STEPS)
    quote = implied.Quote(100.0, 100.0, 1.0, 0.08, american.value)
    return implied.implied_vol(quote, OptionType.PUT)


def implied_volatility_inverts_the_european_only() -> None:
    """An American price goes in, and the early-exercise premium comes out as vol.

    The European price at the same volatility is inverted alongside as the
    control, so the bias is the exercise premium and not the solver.
    """
    assert fields(implied.Quote) == ("spot", "strike", "time", "rate", "price", "carry")
    assert parameters(implied.solve) == ("quote", "option", "method")
    assert {method.name for method in implied.Method} == {"BOUND", "NEWTON", "BRENT"}
    european = moneyness.price(AMERICAN_PUT, OptionType.PUT)
    control = implied.implied_vol(
        implied.Quote(100.0, 100.0, 1.0, 0.08, european), OptionType.PUT
    )
    assert control == pytest.approx(0.20, rel=1e-10, abs=0.0)
    assert implied_volatility_of_an_american_price() > 0.22


def lattices_and_simulations_return_no_greeks() -> None:
    """A lattice or a simulation hands back a price and nothing to hedge with.

    The finite-difference solver is the exception and is asserted as one, so
    the line on the page can say precisely which routes lack them.
    """
    assert fields(lattice.LatticePrice) == (
        "value",
        "steps",
        "lattice",
        "exercise",
        "early_exercise_premium",
    )
    assert fields(monte_carlo.Estimate) == (
        "value",
        "standard_error",
        "samples",
        "paths",
        "control",
    )
    assert {"delta", "gamma"} <= set(fields(pde.MeshPrice))


GAPS: tuple[tuple[str, Callable[[], None]], ...] = (
    (
        "Dividends are a continuous yield.",
        dividends_are_a_continuous_yield,
    ),
    (
        "Rates and carry are constants.",
        rates_and_carry_are_constants,
    ),
    (
        "Heston can be priced and not calibrated.",
        heston_is_priced_and_not_calibrated,
    ),
    (
        "There is no early exercise under Heston.",
        heston_has_no_early_exercise,
    ),
    (
        "Implied volatility inverts the European price only.",
        implied_volatility_inverts_the_european_only,
    ),
    (
        "A lattice or a simulation returns no Greeks.",
        lattices_and_simulations_return_no_greeks,
    ),
)


def gaps_section() -> str:
    text = README.read_text(encoding="utf-8")
    start = text.index(f"\n{SECTION}\n")
    end = text.index(f"\n{NEXT_SECTION}\n", start + len(SECTION))
    return text[start:end]


def rendered_bullets() -> list[str]:
    """The bold lead of each bullet under the heading, in page order."""
    return re.findall(r"^- \*\*(.+?)\*\*", gaps_section(), re.MULTILINE)


@pytest.mark.parametrize(("line", "predicate"), GAPS, ids=[line for line, _ in GAPS])
def test_the_gap_is_still_real(line: str, predicate: Callable[[], None]) -> None:
    predicate()


def test_the_page_lists_exactly_the_gaps_with_predicates() -> None:
    """Same lines, same order: a bullet added to the page needs a predicate too."""
    assert rendered_bullets() == [line for line, _ in GAPS]


def test_every_predicate_is_distinct() -> None:
    predicates = [predicate for _, predicate in GAPS]
    assert len(set(predicates)) == len(predicates)


def test_the_section_is_bounded_by_the_heading_that_follows_it() -> None:
    """A section inserted between the two would have its bullets read as gaps."""
    headings = re.findall(r"^## .+$", README.read_text(encoding="utf-8"), re.MULTILINE)
    assert headings.count(SECTION) == 1
    assert headings[headings.index(SECTION) + 1] == NEXT_SECTION


def test_the_quoted_bias_is_the_measured_one() -> None:
    """The figure on the page is recomputed, to the digits it is quoted to."""
    measured = implied_volatility_of_an_american_price()
    quoted = re.search(r"reads back as (\d+\.\d)%", gaps_section())
    assert quoted is not None, "the quoted implied volatility moved; fix the pattern"
    assert f"{measured * 100:.1f}" == quoted.group(1)


def test_the_producer_scan_reads_both_shapes() -> None:
    """A direct return and a return through a result record are both seen.

    SABR's ``calibrate`` returns a ``Calibration`` directly and a
    ``SabrParameters`` only inside it, so it must turn up under both. SVI's,
    which is the one the root binds, returns its slice inside a ``Fit``.
    """
    assert "sabr.calibrate" in functions_producing(sabr.Calibration)
    assert "sabr.calibrate" in functions_producing(sabr.SabrParameters)
    assert "svi.calibrate" in functions_producing(svi.SVI)
