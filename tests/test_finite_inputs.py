"""No public input record accepts a non-finite number in any float field.

``Inputs``, ``heston.Contract`` and ``implied.Quote`` each validated finiteness
with a loop over a hand-written tuple of field names, the same loop copied
three times, and each copy left out ``carry`` — the one field that is optional.
A carry of minus infinity priced a put at a plain-looking 95.12. Fixing three
call sites leaves the fourth copy of the loop free to make the same mistake, so
the check here is on the records rather than on the loops: every float field of
every public record that validates, set to NaN and to both infinities on an
otherwise valid instance, must be refused.

The table of valid instances is asserted to cover every public dataclass in the
package that defines ``__post_init__``, so a new record with a validator joins
the sweep or fails it.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import math
import pkgutil
import typing

import pytest

import moneyness
from moneyness import (
    AssetPair,
    Compound,
    Contract,
    CrossCurrency,
    Heston,
    Inputs,
    Mesh,
    OptionType,
    Quote,
    SabrParameters,
    Settings,
    Strip,
)
from moneyness.svi import SVI

#: One valid instance per record, as keyword arguments. Every float field is
#: given a value, including the optional ones, so that the baseline exercises
#: the branch where the optional field is present.
VALID: dict[type, dict[str, typing.Any]] = {
    Inputs: {"spot": 100.0, "strike": 100.0, "time": 1.0, "rate": 0.05, "vol": 0.2,
             "carry": 0.03},
    Contract: {"spot": 100.0, "strike": 100.0, "time": 1.0, "rate": 0.05, "carry": 0.03},
    Quote: {"spot": 100.0, "strike": 100.0, "time": 1.0, "rate": 0.05, "price": 10.0,
            "carry": 0.03},
    Heston: {"v0": 0.04, "kappa": 1.5, "theta": 0.04, "sigma": 0.5, "rho": -0.7},
    Compound: {"outer": OptionType.CALL, "inner": OptionType.CALL, "premium": 2.0,
               "decision": 0.5},
    CrossCurrency: {"spot": 100.0, "exchange": 1.2, "strike": 100.0, "time": 1.0,
                    "domestic_rate": 0.04, "foreign_rate": 0.02, "dividend": 0.01,
                    "asset_vol": 0.2, "exchange_vol": 0.1, "correlation": 0.3},
    AssetPair: {"forward1": 100.0, "forward2": 95.0, "time": 1.0, "rate": 0.05,
                "vol1": 0.2, "vol2": 0.25, "correlation": 0.5},
    SabrParameters: {"alpha": 0.2, "beta": 0.5, "rho": -0.3, "nu": 0.4},
    SVI: {"a": 0.04, "b": 0.1, "rho": -0.3, "m": 0.0, "s": 0.2},
    Strip: {"strikes": (90.0, 100.0, 110.0), "prices": (12.0, 6.0, 2.0),
            "forward": 100.0, "time": 1.0, "discount": 0.95},
    Mesh: {"reference_vol": 0.2},
    Settings: {},
}

NON_FINITE = (math.nan, math.inf, -math.inf)


def float_fields(cls: type) -> list[str]:
    """Fields annotated ``float`` or ``float | None``, read from the hints.

    Read from the resolved type hints rather than the annotation strings, so
    ``Optional[float]`` and ``float | None`` are the same thing here.
    """
    hints = typing.get_type_hints(cls)
    floats = {float, float | None}
    return [f.name for f in dataclasses.fields(cls) if hints[f.name] in floats]


def validating_records() -> set[type]:
    """Every public dataclass in the package that defines its own validator."""
    found: set[type] = set()
    for info in pkgutil.iter_modules(moneyness.__path__):
        module = importlib.import_module(f"moneyness.{info.name}")
        for name in getattr(module, "__all__", ()):
            obj = getattr(module, name)
            if (
                inspect.isclass(obj)
                and dataclasses.is_dataclass(obj)
                and obj.__module__ == module.__name__
                and "__post_init__" in vars(obj)
            ):
                found.add(obj)
    return found


def accepted(cls: type, kwargs: dict[str, typing.Any]) -> list[tuple[str, float]]:
    """The (field, value) pairs a record does not refuse by name.

    A refusal is a ``ValueError`` whose message names the field. Any other
    ``ValueError`` is arithmetic tripping over the value somewhere downstream
    — ``cannot convert float NaN to integer`` is what the lattice said about a
    NaN carry — and tells a caller nothing about which input was wrong.
    """
    through: list[tuple[str, float]] = []
    for field in float_fields(cls):
        for bad in NON_FINITE:
            try:
                cls(**{**kwargs, field: bad})
            except ValueError as error:
                if field in str(error):
                    continue
            through.append((field, bad))
    return through


def test_the_table_covers_every_record_that_validates() -> None:
    assert validating_records() == set(VALID)


@pytest.mark.parametrize("cls", list(VALID), ids=lambda cls: cls.__name__)
def test_the_baseline_is_valid(cls: type) -> None:
    cls(**VALID[cls])


@pytest.mark.parametrize("cls", list(VALID), ids=lambda cls: cls.__name__)
def test_no_float_field_takes_a_non_finite_value(cls: type) -> None:
    assert accepted(cls, VALID[cls]) == []


def test_the_sweep_sees_the_optional_fields() -> None:
    """``carry`` and ``reference_vol`` are ``float | None`` and must be swept.

    A filter on the annotation ``float`` alone would skip exactly the fields
    the original defect was in.
    """
    assert "carry" in float_fields(Inputs)
    assert "carry" in float_fields(Contract)
    assert "carry" in float_fields(Quote)
    assert "reference_vol" in float_fields(Mesh)


def test_the_sweep_reports_a_record_that_lets_one_through() -> None:
    """Built for the purpose: a record whose loop forgets a field, as all three did."""

    @dataclasses.dataclass(frozen=True)
    class Forgetful:
        rate: float
        carry: float | None = None

        def __post_init__(self) -> None:
            if not math.isfinite(self.rate):
                raise ValueError("rate must be finite")

    assert accepted(Forgetful, {"rate": 0.05, "carry": 0.01}) == [
        ("carry", value) for value in NON_FINITE
    ]


def test_a_value_error_that_does_not_name_the_field_is_not_a_refusal() -> None:
    """Arithmetic failing on the value is not the record refusing it."""

    @dataclasses.dataclass(frozen=True)
    class Crashes:
        rate: float

        def __post_init__(self) -> None:
            if self.rate != self.rate:
                int(self.rate)

    assert accepted(Crashes, {"rate": 0.05}) == [("rate", value) for value in NON_FINITE]
