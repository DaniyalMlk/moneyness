"""The figures on the front page are the figures the code produces.

The README opens with a table of results checked against a fifty-digit
reference. A table like that is worth exactly as much as the guarantee that it
still matches, and a number copied into a document once stops being true
quietly: the library changes, the suite stays green because nothing in it reads
the document, and the front page goes on claiming an accuracy the code no
longer has.

So the claim is pinned here, in both directions. Every row the README's table
holds has to be a row ``examples/worked.py`` renders, and every row the script
renders has to be on the front page — the second direction catching a check
added to the script that nobody quoted, the first catching a figure that moved.

This test is also what makes "runs in CI" true. CI runs the suite, so the
worked example runs here rather than in a workflow step of its own.
"""

from __future__ import annotations

import re
import runpy
import subprocess
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
WORKED = ROOT / "examples" / "worked.py"

#: Row labels of the README's evidence table, as the script renders them.
#: Written out rather than parsed from the README so that deleting a row from
#: the table does not silently delete the check that defends it.
ROWS = (
    "The price against the same formula at fifty digits in mpmath, 648 points",
    "The same, absolutely, including the quotes worth almost nothing",
    "The normal distribution function against mpmath's erfc, 1143 points",
    "Delta, gamma, vega, theta and rho against derivatives of that price",
    "Put-call parity across the same grid",
    "Volatility recovered from the price it implies, where vega exceeds 1",
    "The width of the set that reprices a quote worth 1.4e-18",
)

#: The bound each check promises, read back off the measurement. The script
#: already enforces these; asserting them again catches an edit that turns the
#: gate into a printout by dropping the raise.
BOUNDS = {
    "The price against the same formula at fifty digits in mpmath, 648 points": 1e-12,
    "The same, absolutely, including the quotes worth almost nothing": 1e-13,
    "The normal distribution function against mpmath's erfc, 1143 points": 1e-15,
    "Delta, gamma, vega, theta and rho against derivatives of that price": 1e-11,
    "Put-call parity across the same grid": 1e-13,
    "Volatility recovered from the price it implies, where vega exceeds 1": 1e-13,
    "The width of the set that reprices a quote worth 1.4e-18": 2.0,
}


@pytest.fixture(scope="module")
def worked() -> dict[str, Any]:
    """Run the worked example once and share its figures across the tests."""
    with redirect_stdout(StringIO()):
        result = runpy.run_path(str(WORKED))["main"]()
    assert isinstance(result, dict)
    return result


@pytest.fixture(scope="module")
def readme() -> str:
    return README.read_text(encoding="utf-8")


@pytest.mark.parametrize("row", ROWS)
def test_the_evidence_table_still_has_the_row(row: str, readme: str) -> None:
    assert row in readme, f"the evidence table no longer has: {row}"


def test_every_row_the_script_renders_is_on_the_front_page(
    worked: dict[str, Any], readme: str
) -> None:
    """Both halves of the table, label and result, as one line.

    Checking the pair together rather than separately is what makes this bite:
    a figure that moves changes the result half while the label stays put, and
    a test that only looked for labels would not notice.
    """
    rendered = worked["rendered"]
    assert isinstance(rendered, dict)
    assert set(rendered) == set(ROWS), "the script's checks and the pinned rows disagree"
    for label, result in rendered.items():
        assert f"| {label} | {result} |" in readme, f"not on the front page: {label}"


@pytest.mark.parametrize(("label", "bound"), sorted(BOUNDS.items()))
def test_the_measurement_is_inside_the_bound(
    label: str, bound: float, worked: dict[str, Any]
) -> None:
    checks = worked["checks"]
    assert isinstance(checks, dict)
    assert checks[label] <= bound, f"{label} measured {checks[label]}, bound {bound}"


#: The one row whose rendered result does not state its own bound, because the
#: bound is not a tolerance: the width row asserts the repricing set is *wide*,
#: so what it enforces is a reciprocal and what it reports is the interval.
WIDTH_ROW = "The width of the set that reprices a quote worth 1.4e-18"


@pytest.mark.parametrize("label", sorted(set(ROWS) - {WIDTH_ROW}))
def test_the_rendered_result_states_the_bound_the_check_enforces(
    label: str, worked: dict[str, Any]
) -> None:
    """The number a row claims and the number it enforces are the same number.

    Without this, the two can drift apart in the one direction nothing else
    notices. Every other test here compares the measurement against the bound
    or the rendering against the README; none of them looks at the bound and
    the rendering together, so loosening ``bound=1e-13`` to ``bound=1e-11``
    while the text still says 1e-13 leaves the whole suite green and the front
    page overstating what is checked. Found exactly that way, by a careless
    edit during this phase that nothing caught.
    """
    rendered = worked["rendered"]
    enforced = worked["bounds"]
    assert isinstance(rendered, dict)
    assert isinstance(enforced, dict)
    stated = re.findall(r"1e-\d+", rendered[label])
    assert stated, f"{label} renders no bound at all: {rendered[label]!r}"
    # Against the bound the script *enforces*, read off the check itself --
    # not against this module's copy of it, which would only compare two
    # transcriptions and miss the drift this test exists for. The copy is
    # asserted too, so all three have to agree.
    assert f"{enforced[label]:.0e}" in stated, (
        f"{label} enforces {enforced[label]:.0e} and claims {stated}"
    )
    assert enforced[label] == BOUNDS[label], (
        f"{label} enforces {enforced[label]:.0e}, pinned here as {BOUNDS[label]:.0e}"
    )


def test_the_implied_volatility_ladder_is_on_the_front_page(
    worked: dict[str, Any], readme: str
) -> None:
    """Every rung of the ladder, with its count and its error.

    The ladder is the row that most needs pinning, because it is the one whose
    shape is the argument: the error has to track vega, and a reader checking
    that has to be able to trust the three figures on each line.
    """
    ladder = worked["ladder"]
    assert isinstance(ladder, list)
    assert len(ladder) == 4
    for floor, quotes, worst in ladder:
        assert f"| {floor:g} | {quotes} | {worst:.1e} |" in readme, (
            f"the ladder's {floor:g} rung is not on the front page as rendered"
        )


def test_the_ladder_is_monotone_in_vega(worked: dict[str, Any]) -> None:
    """The error gets worse as the price stops depending on the volatility.

    This is the claim the ladder makes, and it is a claim about the *shape* of
    the measurement rather than about any one figure, so pinning the figures
    does not cover it. A change that made the solver uniformly accurate across
    the grid would be a change that had stopped solving the mute quotes
    honestly, and it would fail here.
    """
    ladder = worked["ladder"]
    errors = [worst for _, _, worst in ladder]
    counts = [quotes for _, quotes, _ in ladder]
    assert errors == sorted(errors), f"the error is not ordered in vega: {errors}"
    assert counts == sorted(counts), f"the quote counts are not ordered: {counts}"
    # And the span is a real span rather than a rounding artefact: the worst
    # rung is at least a million times the best, which is what makes the row
    # worth printing instead of a single figure.
    assert errors[-1] > 1e6 * errors[0]


def test_the_mute_quotes_are_reported_rather_than_hidden(
    worked: dict[str, Any], readme: str
) -> None:
    """The count of unanswerable quotes, and the window that reprices one.

    A third of this grid is quotes a solver cannot answer. The README says so
    with a number, and the number is this one.
    """
    mute = worked["mute"]
    assert isinstance(mute, int)
    assert mute > 0
    assert f"{mute} of the 648 quotes" in readme
    low, high = worked["window"]
    assert high - low > 0.5, "the window is no longer wide enough to be the point"
    assert f"from {low:.3f} to {high:.3f}" in readme


def test_the_suite_size_on_the_front_page_is_not_wildly_stale() -> None:
    """A loose bound, deliberately.

    Pinning the exact count would fail on every commit that adds a test, which
    trains people to update the number without reading it. A quarter either way
    still catches the README claiming hundreds when there are thousands, which
    is the error that matters.
    """
    claimed = re.search(r"([\d,]+) tests", README.read_text(encoding="utf-8"))
    assert claimed is not None, "the README no longer states a suite size"
    stated = int(claimed.group(1).replace(",", ""))
    collected = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "--collect-only", "-q", "-o", "addopts="],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
    )
    found = re.search(r"(\d+)[/ ]tests collected", collected.stdout) or re.search(
        r"(\d+) tests collected", collected.stdout
    )
    assert found is not None, collected.stdout[-500:]
    actual = int(found.group(1))
    assert 0.75 * actual <= stated <= 1.25 * actual, (
        f"the README says {stated} tests and there are {actual}"
    )
