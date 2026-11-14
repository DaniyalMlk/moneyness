"""Names declared by more than one module, and which one the root carries.

Three models here offer the same operation. Black-Scholes-Merton, Heston and the
quanto contract all have a ``price``; Heston and SABR both have a ``smile``; SVI
and SABR both have a ``calibrate`` and a ``density``; and both simulators have a
``european``, an ``asian`` and a ``barrier``. A flat namespace holds one of each.
The package root is that flat namespace, so eight names arrive from more than one
place and nine declarations lose.

That is not a bug to be fixed by renaming. ``bsm.price`` and ``heston.price``
really are both the price, and a caller who has a :class:`~moneyness.Heston` in
hand wants the second one under a name that says so -- which is what the module
is for. What *was* wrong is that nothing said so. The README reached into
``moneyness.heston`` for a price in the middle of a page whose other examples
import from the root, and a submodule import under a paragraph of prose reads as
a style choice rather than as the only path there is.

So the table below is the statement, and this module is its proof. For every
name in it:

* every module the table names really does declare it, so a renamed or deleted
  function cannot leave a stale row behind;
* the root really does bind it from the module the table calls the winner, so
  reordering the imports in ``__init__.py`` -- which is all it would take to
  change which ``price`` a caller gets -- fails here;
* every shadowed module still carries it in its own ``__all__``, so the access
  path the README documents keeps working.

And the table is asserted to be *exactly* the set of multiply-declared names, in
both directions. A new collision is the interesting case: it means a name has
quietly stopped meaning one thing, and whoever added it has a choice to make
rather than an import to write.

``ruff`` cannot do this job. ``F811`` fires on a redefinition, and a redefinition
needs both sides in one namespace; the root imports one side of each pair, so
there has never been anything for it to see.

Each rule here was broken on purpose before being relied on: a module dropping
a shadowed name, a row deleted, a wrong winner, the imports in ``__init__.py``
reordered so the root hands out the other ``price``, a module losing its
``__all__``, ``cli``'s ``main`` appearing at the root, a wrong README cell, a
stale README count, an example reaching in for a name the root already has, a
new collision, and a new module the root never imports. Eleven breaks, eleven
caught by the rule aimed at them.

The equality below was broken four more ways, in both directions: a name going
public in its module with nobody adding it to the root, the root keeping a name
its own module stopped declaring, a name dropped from the root while its module
still declares it, and a name imported by the root and never declared. Four
breaks, four caught.

**The probe reads source, not the imported module.** Asking the package which
attributes it has would be shorter and would miss things: a name bound to a bare
``int`` or ``float`` carries no ``__module__`` to attribute it with, and three of
``deferred``'s exports are plain numbers. Reading ``__all__`` out of the syntax
tree sees exactly what the author wrote down.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "moneyness"
README = ROOT / "README.md"

#: Modules whose names are deliberately absent from the root.
#:
#: ``cli`` exists for the console entry point declared in ``pyproject.toml``;
#: its ``main`` takes ``argv`` and returns an exit status, which is not
#: something a library caller wants.
#:
#: ``optimise`` is the arithmetic the calibration rests on -- a Cholesky solve
#: and a Nelder-Mead search -- and its own docstring declines to offer itself as
#: a general-purpose numerical library. Putting ``cholesky_solve`` and
#: ``nelder_mead`` into a root namespace of options functions would read as
#: exactly that offer. This one is a judgement rather than a fact: the
#: alternative is to export all four names and let callers use what is already
#: written and tested. It is recorded here so that it is a decision and not an
#: oversight, which is the distinction this module exists to make.
MODULE_ONLY = ("cli", "optimise")


@dataclass(frozen=True)
class Collision:
    """One name declared by several modules, and the one the root carries.

    Attributes:
        name: The declared name.
        winner: The module ``moneyness/__init__.py`` imports it from.
        shadowed: Every other module declaring the same name, which a caller
            must reach through the module itself.
        because: Why the two are different, in one line, so a reader of the
            table does not have to open both modules to find out.
    """

    name: str
    winner: str
    shadowed: tuple[str, ...]
    because: str

    @property
    def modules(self) -> tuple[str, ...]:
        """Every module declaring this name, the winner first."""
        return (self.winner, *self.shadowed)


COLLISIONS: tuple[Collision, ...] = (
    Collision(
        "asian",
        "monte_carlo",
        ("heston_mc",),
        "lognormal paths against a stochastic-variance path",
    ),
    Collision(
        "barrier",
        "monte_carlo",
        ("heston_mc",),
        "lognormal paths against a stochastic-variance path",
    ),
    Collision(
        "european",
        "monte_carlo",
        ("heston_mc",),
        "lognormal paths against a stochastic-variance path",
    ),
    Collision(
        "calibrate",
        "svi",
        ("sabr",),
        "a slice fitted in total variance against two SABR parameters",
    ),
    Collision(
        "density",
        "svi",
        ("sabr",),
        "the density of an SVI slice against the one Hagan's smile implies",
    ),
    Collision(
        "parity_gap",
        "bsm",
        ("quanto",),
        "one currency against the forward a cross-currency contract implies",
    ),
    Collision(
        "price",
        "bsm",
        ("heston", "quanto"),
        "a closed form against a transform and against a cross-currency payoff",
    ),
    Collision(
        "smile",
        "heston",
        ("sabr",),
        "strikes repriced through the transform against Hagan's expansion",
    ),
)


def declared_names(source: str) -> list[str] | None:
    """The string entries of a module's ``__all__``, read out of its source.

    Returns ``None`` when the module declares no ``__all__`` at all, which is a
    different thing from declaring an empty one and is reported separately.

    Both assignment forms are read. Nothing in this package currently writes
    ``__all__: list[str] = [...]``, so that branch is exercised by
    :func:`test_probe_reads_an_annotated_all` rather than by the source.
    """
    found: list[str] | None = None
    for node in ast.parse(source).body:
        target: ast.expr | None = None
        value: ast.expr | None = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign):
            target, value = node.target, node.value
        if not isinstance(target, ast.Name) or target.id != "__all__":
            continue
        if not isinstance(value, ast.List | ast.Tuple):
            raise AssertionError("__all__ is not a literal list or tuple")
        names: list[str] = []
        for element in value.elts:
            # A name is a string literal. A starred expression or a computed
            # entry is neither read nor guessed at.
            if isinstance(element, ast.Constant) and isinstance(element.value, str):
                names.append(element.value)
        found = names
    return found


def root_bindings(source: str) -> dict[str, str]:
    """Map each name the root imports from a sibling to the module it came from.

    Later imports win, exactly as they do at runtime, so the result says which
    module supplies each name after ``__init__.py`` has finished executing.
    """
    bindings: dict[str, str] = {}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
            for alias in node.names:
                bindings[alias.asname or alias.name] = node.module
    return bindings


def _modules() -> dict[str, list[str]]:
    """Every non-root module, mapped to the names it declares."""
    out: dict[str, list[str]] = {}
    for path in sorted(PACKAGE.glob("*.py")):
        if path.name == "__init__.py":
            continue
        names = declared_names(path.read_text())
        assert names is not None, f"{path.name} declares no __all__"
        out[path.stem] = names
    return out


def _owners() -> dict[str, list[str]]:
    """Every declared name, mapped to the modules declaring it."""
    owners: dict[str, list[str]] = defaultdict(list)
    for module, names in _modules().items():
        for name in names:
            owners[name].append(module)
    return dict(owners)


# --------------------------------------------------------------------------
# The table against the source.
# --------------------------------------------------------------------------


def test_every_module_declares_an_all() -> None:
    """A module with no ``__all__`` has no stated surface to check."""
    for path in sorted(PACKAGE.glob("*.py")):
        assert declared_names(path.read_text()) is not None, (
            f"{path.name} declares no __all__, so nothing here can tell "
            "which of its names are meant to be public"
        )


def test_collisions_are_exactly_the_multiply_declared_names() -> None:
    """The table is the whole truth, in both directions.

    A missing row means a name has quietly stopped meaning one thing. A row
    with nothing behind it means a rename or a deletion left the table stale.
    """
    actual = {name for name, modules in _owners().items() if len(modules) > 1}
    tabled = {c.name for c in COLLISIONS}
    assert actual == tabled, (
        f"not in the table: {sorted(actual - tabled)}; "
        f"in the table but declared once or not at all: {sorted(tabled - actual)}"
    )


def test_each_collision_names_every_module_that_declares_it() -> None:
    """``winner`` and ``shadowed`` together are the full set of declarers."""
    owners = _owners()
    for collision in COLLISIONS:
        assert sorted(owners[collision.name]) == sorted(collision.modules), (
            f"{collision.name} is declared by {sorted(owners[collision.name])}, "
            f"but the table says {sorted(collision.modules)}"
        )


def test_the_root_binds_the_winner() -> None:
    """Which ``price`` a caller gets is a decision, recorded here.

    Reordering the imports in ``__init__.py`` would change it, and would change
    nothing else that any other test looks at.
    """
    bindings = root_bindings((PACKAGE / "__init__.py").read_text())
    for collision in COLLISIONS:
        assert bindings.get(collision.name) == collision.winner, (
            f"the root takes {collision.name} from "
            f"{bindings.get(collision.name)!r}, not from {collision.winner!r}"
        )


def test_shadowed_declarations_stay_reachable_through_their_module() -> None:
    """The documented path is the module, so the module must still export it.

    This cannot fail on its own: the test above compares the same declarers
    against the same table and fires first. It is kept for the message, which
    names the README's dependency on the module path rather than reporting a
    set difference, and it is recorded here as a restatement so that nobody
    reads it as a second, independent check.
    """
    modules = _modules()
    for collision in COLLISIONS:
        for module in collision.shadowed:
            assert collision.name in modules[module], (
                f"{module}.{collision.name} is the access path the README "
                "documents, and the module no longer declares it"
            )


def test_the_root_is_exactly_the_union_of_the_public_modules() -> None:
    """An equality, not a containment, and in both directions.

    A module putting a name in its ``__all__`` is the module saying that name is
    public. The root is where a caller looks, so the two have to agree, and the
    only reasons a declared name may be absent are the two recorded above: it
    belongs to a module-only module, or it is a collision loser reached through
    its own module. Both of those are tables rather than judgements made at read
    time.

    Written as a containment this would have passed for as long as it existed.
    Nine names were public in their own module and absent from the root --
    ``black`` and ``black_vega``, ``paths``, three of ``spread``'s, and
    ``deferred``'s three ``CRITICAL_*`` solver bounds -- none of them colliding
    with anything. The precedent for exporting a module's tuning constants was
    already set in the other direction by ``BARRIER_DIGITS`` and
    ``CARRY_CROSSOVER``, which are at the root.

    The reverse direction matters as much. ``SmilePoint`` was in the root's
    ``__all__`` and not in ``heston``'s, so ``from moneyness.heston import *``
    handed back ``smile`` without the type of its own return value.
    """
    modules = _modules()
    union = {
        name
        for module, names in modules.items()
        if module not in MODULE_ONLY
        for name in names
    }
    root_all = declared_names((PACKAGE / "__init__.py").read_text())
    assert root_all is not None

    assert set(root_all) == union | {"__version__"}, (
        "public in a module and missing from the root: "
        f"{sorted(union - set(root_all))}; "
        "at the root and declared by no public module: "
        f"{sorted(set(root_all) - union - {'__version__'})}"
    )


def test_the_root_declares_every_name_it_imports() -> None:
    """Nothing reaches the root without being named in its ``__all__``."""
    init = (PACKAGE / "__init__.py").read_text()
    root_all = declared_names(init)
    assert root_all is not None
    undeclared = sorted(set(root_bindings(init)) - set(root_all))
    assert not undeclared, f"imported by the root but not in its __all__: {undeclared}"


def test_no_shadowed_name_is_at_the_root_under_its_own_module() -> None:
    """A shadowed name must not also arrive from its own module.

    Both sides in one namespace is the case ``F811`` already catches; asserting
    it here means the table cannot describe a collision that is not one.
    """
    bindings = root_bindings((PACKAGE / "__init__.py").read_text())
    for collision in COLLISIONS:
        assert bindings.get(collision.name) not in collision.shadowed


def test_every_public_module_is_imported_by_the_root() -> None:
    """Only the declared module-only modules may be absent from the root."""
    imported = set(root_bindings((PACKAGE / "__init__.py").read_text()).values())
    absent = set(_modules()) - imported
    assert absent == set(MODULE_ONLY), (
        f"modules the root never imports: {sorted(absent)}; "
        f"declared module-only: {sorted(MODULE_ONLY)}"
    )


def test_module_only_modules_contribute_nothing_to_the_root() -> None:
    """``cli`` is for the console entry point, not for callers."""
    root_all = declared_names((PACKAGE / "__init__.py").read_text())
    assert root_all is not None
    modules = _modules()
    for module in MODULE_ONLY:
        for name in modules[module]:
            assert name not in root_all, (
                f"{module}.{name} is at the root, so {module} is not module-only"
            )


# --------------------------------------------------------------------------
# The table against the README.
# --------------------------------------------------------------------------


def test_the_readme_table_matches_the_collisions() -> None:
    """The prose and the gate cannot drift apart.

    The README's table is parsed rather than searched for: a row per collision,
    naming the winner and every shadowed module, so a name added here without
    being documented fails, and a documented row with no collision behind it
    fails too.
    """
    rows: dict[str, tuple[str, tuple[str, ...]]] = {}
    for line in README.read_text().splitlines():
        if not line.startswith("| `"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 4:
            continue
        name = cells[0].strip("`")
        winner = cells[1].strip("`")
        shadowed = tuple(sorted(part.strip().strip("`") for part in cells[2].split(",")))
        rows[name] = (winner, shadowed)

    expected = {c.name: (c.winner, tuple(sorted(c.shadowed))) for c in COLLISIONS}
    assert rows == expected, (
        "the README's collision table and COLLISIONS disagree: "
        f"documented {sorted(rows)}, tabled {sorted(expected)}"
    )


def test_a_readme_submodule_import_is_one_the_root_cannot_serve() -> None:
    """Reaching into a module must mean something, or it reads as habit.

    Before this, one example imported ``Quote`` and ``solve`` from
    ``moneyness.implied`` while both sit at the root, and another imported
    ``price`` from ``moneyness.heston`` because the root's ``price`` is
    Black-Scholes-Merton's. A reader had no way to tell the two apart, which is
    what let the second one read as a style choice for as long as it did.

    So every name a README example imports from a submodule must be one the
    root genuinely cannot supply: a shadowed declaration, or a name that is not
    at the root at all.
    """
    root_all = declared_names((PACKAGE / "__init__.py").read_text())
    assert root_all is not None
    losers = {(module, c.name) for c in COLLISIONS for module in c.shadowed}

    offenders: list[str] = []
    for line in README.read_text().splitlines():
        stripped = line.strip()
        if not stripped.startswith("from moneyness."):
            continue
        module = stripped.removeprefix("from moneyness.").split(" import ")[0].strip()
        names = [n.strip() for n in stripped.split(" import ", 1)[1].split(",")]
        for name in names:
            if (module, name) in losers or name not in root_all:
                continue
            offenders.append(f"from moneyness.{module} import {name}")

    assert not offenders, (
        "these README imports reach into a module for a name the root already "
        f"supplies: {offenders}"
    )


def test_the_readme_counts_the_collisions_it_documents() -> None:
    """A count in prose that nothing recomputes goes stale on the next model."""
    text = README.read_text()
    names = len(COLLISIONS)
    declarations = sum(len(c.shadowed) for c in COLLISIONS)
    assert f"{_words(names)} names" in text, f"the README should say {_words(names)} names"
    assert f"{_words(declarations)} declarations" in text, (
        f"the README should say {_words(declarations)} declarations"
    )


_NUMBERS = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
}


def _words(value: int) -> str:
    """``8`` as ``"eight"``, because the README is written in words."""
    return _NUMBERS[value]


# --------------------------------------------------------------------------
# The probes against source written for the purpose.
#
# Several branches below are not exercised by this package's own modules, and a
# gate rule the source never reaches is a rule nobody has tested.
# --------------------------------------------------------------------------


def test_probe_reads_a_plain_list() -> None:
    assert declared_names("__all__ = ['a', 'b']\n") == ["a", "b"]


def test_probe_reads_a_tuple() -> None:
    assert declared_names("__all__ = ('a', 'b')\n") == ["a", "b"]


def test_probe_reads_an_annotated_all() -> None:
    """``__all__: list[str] = [...]`` is an ``AnnAssign``, not an ``Assign``."""
    assert declared_names("__all__: list[str] = ['a']\n") == ["a"]


def test_probe_reports_a_missing_all_as_none() -> None:
    """Distinct from an empty one, which is a deliberate statement."""
    assert declared_names("def f() -> None:\n    pass\n") is None
    assert declared_names("__all__ = []\n") == []


def test_probe_takes_the_last_assignment() -> None:
    """Later wins, as it does at import time."""
    assert declared_names("__all__ = ['a']\n__all__ = ['b']\n") == ["b"]


def test_probe_ignores_a_name_that_merely_contains_all() -> None:
    assert declared_names("__all_names__ = ['a']\n") is None


def test_probe_ignores_an_all_that_is_not_at_module_level() -> None:
    """A local named ``__all__`` inside a function is not a declaration."""
    assert declared_names("def f() -> None:\n    __all__ = ['a']\n    print(__all__)\n") is None


def test_probe_refuses_a_computed_all() -> None:
    """``__all__ = something + other`` cannot be read, and is not guessed at."""
    try:
        declared_names("__all__ = _a + _b\n")
    except AssertionError:
        return
    raise AssertionError("a computed __all__ should not be read as a literal")


def test_probe_skips_non_string_entries() -> None:
    """A name is a string; anything else is not a name and is not reported."""
    assert declared_names("__all__ = ['a', *_rest, 'b']\n") == ["a", "b"]
    assert declared_names("__all__ = ['a', 1, None, 'b']\n") == ["a", "b"]


def test_bindings_take_the_later_import() -> None:
    source = "from .one import x\nfrom .two import x\n"
    assert root_bindings(source) == {"x": "two"}


def test_bindings_follow_an_alias() -> None:
    """``as`` rebinds, so the name at the root is the alias."""
    assert root_bindings("from .one import x as y\n") == {"y": "one"}


def test_bindings_ignore_absolute_and_deeper_relative_imports() -> None:
    """Only a sibling module can be the source of a root name here."""
    assert root_bindings("import math\nfrom math import sqrt\n") == {}
    assert root_bindings("from ..other import x\n") == {}
