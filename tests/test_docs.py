"""The two documents agree with each other, and each agrees with itself.

The front page's status section narrated phases 1 to 17 for seven phases after
the roadmap passed them, opened by saying "most of" the command line was there
two paragraphs before saying all of it was, and closed by saying the first
release was still waiting on the package index weeks after 0.1.0 was published.
The roadmap carried the same stale release item twice, the second copy filed
under a phase about chooser options. The suite was green throughout, because
nothing in it read either document.

A claim written into prose is checked only if something recomputes it, so the
claims are recomputed here from the documents:

* the phase count the status section states is the number of sections the
  roadmap has;
* the status section names every one of those phases, so the narrative cannot
  stop part way again while the closing count is kept current;
* a document that says every phase is done has no unchecked box;
* the roadmap's numbering has no duplicate, no gap and no descent, and every
  cross-reference names an earlier phase that exists;
* the README's numbered decisions run first, second, third with nothing
  skipped or repeated;
* the release example names no literal version, since 0.1.0 is published and
  an index never takes the same version twice.

Each rule is a named function, tested against text built for the purpose as
well as against the documents, so a rule the current documents happen not to
exercise is still shown to fire.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
ROADMAP = ROOT / "ROADMAP.md"

#: ``## Phase 12 — Variance and volatility swaps``. The separator is an em dash
#: in every heading; matching it rather than any dash keeps a hyphen inside a
#: title from being read as the separator.
HEADING = re.compile(r"^## Phase (\d+) — (.+)$", re.MULTILINE)

#: A phase named in prose: ``phase 4``, ``Phase 13``, ``Phases 14 and 15`` and
#: ``Phases 20 to 24``. A range names every phase in it; ``and`` names two.
REFERENCE = re.compile(r"\b[Pp]hases? (\d+)(?: (and|to) (\d+))?")

#: The count is written as a word on the front page.
WORDS = {
    "twenty": 20,
    "twenty-one": 21,
    "twenty-two": 22,
    "twenty-three": 23,
    "twenty-four": 24,
    "twenty-five": 25,
    "twenty-six": 26,
    "twenty-seven": 27,
    "twenty-eight": 28,
    "twenty-nine": 29,
    "thirty": 30,
}

STATED_COUNT = re.compile(r"All ([a-z]+(?:-[a-z]+)?) phases in \[the roadmap\]")

#: The README's design decisions are headed ``## The decision that mattered``
#: and then ``## A second decision: ...``, ``## An eighth decision: ...``.
ORDINALS = [
    "second",
    "third",
    "fourth",
    "fifth",
    "sixth",
    "seventh",
    "eighth",
    "ninth",
    "tenth",
    "eleventh",
    "twelfth",
    "thirteenth",
    "fourteenth",
    "fifteenth",
    "sixteenth",
    "seventeenth",
    "eighteenth",
    "nineteenth",
    "twentieth",
    "twenty-first",
    "twenty-second",
    "twenty-third",
    "twenty-fourth",
    "twenty-fifth",
]
DECISION = re.compile(r"^## An? ([a-z-]+) decision\b", re.MULTILINE)
FIRST_DECISION = "## The decision that mattered"

#: A literal version tag such as ``v0.1.0``.
LITERAL_TAG = re.compile(r"\bv\d+\.\d+\.\d+\b")

UNCHECKED = re.compile(r"^\s*- \[ \]", re.MULTILINE)


# --------------------------------------------------------------------------
# The rules, as functions of text
# --------------------------------------------------------------------------


def section(markdown: str, heading: str) -> str:
    """The body under a ``## `` heading, up to the next ``## `` heading.

    Stopping at the next level-two heading matters: without it the status
    section would run to the end of the file and a phase named anywhere after
    it would count as covered.
    """
    start = markdown.find(f"\n{heading}\n")
    assert start != -1, f"no {heading!r} section"
    body_start = start + len(heading) + 2
    following = re.search(r"^## ", markdown[body_start:], re.MULTILINE)
    end = body_start + following.start() if following else len(markdown)
    return markdown[body_start:end]


def phase_headings(roadmap: str) -> list[int]:
    """Every phase number in the roadmap, in the order it appears."""
    return [int(number) for number, _ in HEADING.findall(roadmap)]


def named_phases(prose: str) -> list[int]:
    """Every phase a passage names, a range expanded to all of its members."""
    found: list[int] = []
    for match in REFERENCE.finditer(prose):
        first = int(match.group(1))
        joiner, last = match.group(2), match.group(3)
        if joiner == "to":
            found.extend(range(first, int(last) + 1))
        elif joiner == "and":
            found.extend((first, int(last)))
        else:
            found.append(first)
    return found


def stated_count(prose: str) -> int:
    found = STATED_COUNT.search(prose)
    assert found is not None, (
        "the status section no longer states a phase count in the form this "
        "gate reads; fix the pattern if the sentence moved, or delete the check "
        "with the claim rather than leaving a pattern that matches nothing"
    )
    word = found.group(1)
    assert word in WORDS, f"unmapped number word {word!r}; add it to WORDS"
    return WORDS[word]


def numbering_faults(numbers: list[int]) -> list[str]:
    """Everything wrong with a list of phase numbers, as readable faults."""
    faults: list[str] = []
    repeated = sorted(n for n, count in Counter(numbers).items() if count > 1)
    if repeated:
        faults.append(f"repeated {repeated}")
    expected = set(range(1, len(numbers) + 1))
    if set(numbers) != expected:
        faults.append(
            f"missing {sorted(expected - set(numbers))}, "
            f"unexpected {sorted(set(numbers) - expected)}"
        )
    descents = [
        (numbers[i], numbers[i + 1])
        for i in range(len(numbers) - 1)
        if numbers[i + 1] <= numbers[i]
    ]
    if descents:
        faults.append(f"out of order at {descents}")
    return faults


def cross_references(roadmap: str) -> list[tuple[int, int]]:
    """Every prose reference as (phase it is written in, phase it names).

    A heading's own number is excluded, being the phase rather than a reference
    to it, but a phase its title names is not: excluding the whole heading
    line would make ``## Phase 2 — Building on phase 3`` invisible. Text before
    the first heading is attributed to phase 0, which the backwards rule then
    treats as naming something later than itself.
    """
    bounds = [(match.start(), int(match.group(1))) for match in HEADING.finditer(roadmap)]
    own_numbers = {start + len("## ") for start, _ in bounds}
    found: list[tuple[int, int]] = []
    for match in REFERENCE.finditer(roadmap):
        if match.start() in own_numbers:
            continue
        written_in = 0
        for start, number in bounds:
            if start > match.start():
                break
            written_in = number
        found.extend((written_in, named) for named in named_phases(match.group(0)))
    return found


def forward_references(roadmap: str) -> list[tuple[int, int]]:
    return [(at, named) for at, named in cross_references(roadmap) if named >= at]


def decision_faults(readme: str) -> list[str]:
    """The numbered decisions must run first, second, third, with no skip."""
    faults: list[str] = []
    if readme.count(f"\n{FIRST_DECISION}\n") != 1:
        faults.append(f"{FIRST_DECISION!r} should appear exactly once")
    found = DECISION.findall(readme)
    unknown = [word for word in found if word not in ORDINALS]
    if unknown:
        faults.append(f"unmapped ordinals {unknown}; add them to ORDINALS")
        return faults
    expected = ORDINALS[: len(found)]
    if found != expected:
        faults.append(f"decisions read {found}, expected {expected}")
    return faults


# --------------------------------------------------------------------------
# The documents
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def readme() -> str:
    return README.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def roadmap() -> str:
    return ROADMAP.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def status(readme: str) -> str:
    return section(readme, "## Status")


def test_the_stated_count_is_the_number_of_phases(status: str, roadmap: str) -> None:
    assert stated_count(status) == len(phase_headings(roadmap))


def test_the_status_names_every_phase(status: str, roadmap: str) -> None:
    """The narrative reaches the last phase, not only the closing count.

    This is what the original drift needed: the count can be kept current in
    one sentence while the paragraphs above it stop seven phases early.
    """
    phases = set(phase_headings(roadmap))
    named = set(named_phases(status))
    assert phases - named == set(), f"phases the status never names: {sorted(phases - named)}"
    assert named - phases == set(), f"phases named that do not exist: {sorted(named - phases)}"


def test_a_document_that_says_everything_is_done_has_no_open_box(
    status: str, roadmap: str
) -> None:
    stated_count(status)  # the claim is present, so it has to be true
    open_items = UNCHECKED.findall(roadmap)
    assert not open_items, (
        f"the README says every phase is done and the roadmap has "
        f"{len(open_items)} unchecked item(s)"
    )


def test_the_roadmap_numbering_is_sound(roadmap: str) -> None:
    assert numbering_faults(phase_headings(roadmap)) == []


def test_every_reference_names_an_earlier_phase_that_exists(roadmap: str) -> None:
    numbers = set(phase_headings(roadmap))
    dangling = sorted({named for _, named in cross_references(roadmap)} - numbers)
    assert dangling == [], f"references to phases that do not exist: {dangling}"
    assert forward_references(roadmap) == []


def test_the_reference_probe_reads_what_is_there(roadmap: str) -> None:
    """Both reference checks pass vacuously if the pattern stops matching.

    Phase 16 names phase 15, and phase 17 names phase 14 twice, once
    capitalised and once not. Phase 25 names the range 1 to 17, phase 8 twice
    (once inside that range) and phase 18, so the range form is read as well
    as the single one.
    """
    expected = [(16, 15), (17, 14), (17, 14)]
    expected += [(25, n) for n in range(1, 18)] + [(25, 8), (25, 18)]
    assert sorted(cross_references(roadmap)) == sorted(expected)


def test_the_decisions_are_numbered_in_order(readme: str) -> None:
    assert decision_faults(readme) == []
    assert len(DECISION.findall(readme)) == 18, (
        "the README has the first decision plus eighteen numbered ones; if a "
        "decision was added, change this count in the same change"
    )


def test_the_release_example_names_no_literal_version(readme: str) -> None:
    releasing = section(readme, "## Releasing")
    assert "git tag" in releasing, "the release example moved; fix this check"
    assert LITERAL_TAG.findall(releasing) == []


# --------------------------------------------------------------------------
# The rules against text built for them
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("prose", "expected"),
    [
        ("Phase 4 adds", [4]),
        ("and phase 5 the", [5]),
        ("Phases 14 and 15 are", [14, 15]),
        ("Phases 20 to 24 are", [20, 21, 22, 23, 24]),
        ("Phases 1 to 3 of", [1, 2, 3]),
        ("multiphase 9 is not a reference", []),
        ("no reference here", []),
    ],
)
def test_named_phases_reads_each_form(prose: str, expected: list[int]) -> None:
    assert named_phases(prose) == expected


@pytest.mark.parametrize(
    ("numbers", "fragment"),
    [
        ([1, 2, 2, 4], "repeated [2]"),
        ([1, 2, 4], "missing [3]"),
        ([1, 3, 2], "out of order"),
        ([2, 3], "missing [1]"),
    ],
)
def test_numbering_faults_fires_on_each_fault(numbers: list[int], fragment: str) -> None:
    assert any(fragment in fault for fault in numbering_faults(numbers))


def test_numbering_faults_is_silent_on_a_clean_list() -> None:
    assert numbering_faults([1, 2, 3, 4]) == []


def test_a_forward_reference_is_reported() -> None:
    text = "## Phase 1 — A\n\nSee phase 2.\n\n## Phase 2 — B\n\nAfter phase 1.\n"
    assert forward_references(text) == [(1, 2)]


def test_a_reference_to_its_own_phase_is_reported() -> None:
    text = "## Phase 1 — A\n\n## Phase 2 — B\n\nAs phase 2 says.\n"
    assert forward_references(text) == [(2, 2)]


def test_a_reference_before_any_phase_is_reported() -> None:
    text = "Read phase 1 first.\n\n## Phase 1 — A\n"
    assert forward_references(text) == [(0, 1)]


def test_a_heading_is_not_a_reference_to_itself() -> None:
    text = "## Phase 1 — A\n\n## Phase 2 — Building on phase 1\n"
    assert cross_references(text) == [(2, 1)]


def test_the_section_stops_at_the_next_heading() -> None:
    text = "# T\n\n## Status\n\nPhase 1.\n\n## Other\n\nPhase 2.\n"
    assert named_phases(section(text, "## Status")) == [1]


@pytest.mark.parametrize(
    ("readme", "fragment"),
    [
        (f"\n{FIRST_DECISION}\n\n## A third decision: x\n", "expected ['second']"),
        (
            f"\n{FIRST_DECISION}\n\n## A second decision: x\n## A second decision: y\n",
            "expected ['second', 'third']",
        ),
        ("\n## A second decision: x\n", "exactly once"),
        (f"\n{FIRST_DECISION}\n\n## A zeroth decision: x\n", "unmapped"),
    ],
)
def test_decision_faults_fires_on_each_fault(readme: str, fragment: str) -> None:
    assert any(fragment in fault for fault in decision_faults(readme))


def test_an_eighth_decision_reads_with_an() -> None:
    text = f"\n{FIRST_DECISION}\n\n## A second decision: x\n" + "\n".join(
        f"## A {word} decision: x" for word in ORDINALS[1:6]
    ) + "\n## An eighth decision: x\n"
    assert decision_faults(text) == []


def test_a_literal_tag_is_seen() -> None:
    assert LITERAL_TAG.findall("git tag v0.1.0\ngit push origin v0.1.0") == ["v0.1.0"] * 2
    assert LITERAL_TAG.findall('git tag "v$version"') == []
