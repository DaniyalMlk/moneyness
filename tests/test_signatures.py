"""A caller who can call something can name what it takes and returns.

The surface checks elsewhere ask whether every name a module declares reaches
the root. That cannot see a *private* type: nothing declares it, so nothing
expects it at the root. ``adaptive_quad`` was at the root with a last parameter
of ``budget: _Budget | None``, and the only way to fill it was to import
``moneyness.quadrature._Budget`` -- which every test of that parameter did, and
which no caller outside the package could reasonably be expected to.

So this asks the question from the other side. For everything reachable from
the root -- each root function, each public method and property of each root
class including those it inherits from a package class, and each field of a
root class -- every package class its annotations mention must be at the root,
and every package *type alias* its annotations are written in must be too.

Two things make the obvious version of this wrong here:

* **Identity, not name.** Eight names are declared by more than one module and
  the root carries one of each. A signature mentioning ``heston.Heston`` would
  pass a name check if *any* ``Heston`` were at the root. The check is that the
  root binds the very same object.
* **An alias is invisible once resolved.** ``typing.get_type_hints`` expands
  ``Matrix`` into ``list[list[float]]`` before anything can look at it, so a
  check built only on resolved hints passes every alias whatever the root
  does. Aliases are therefore read from source: which module-level names are
  type expressions, and which of those a signature's annotation spells out.

A parameter named with an underscore is private by name and exempt, as is a
member named with one, except ``__init__``.
"""

from __future__ import annotations

import ast
import inspect
import pkgutil
import textwrap
import typing
from collections.abc import Callable, Iterator
from pathlib import Path

import moneyness

PACKAGE = Path(moneyness.__file__).resolve().parent
PREFIX = "moneyness."

#: Subscripted names that make a module-level assignment a type expression.
TYPE_CONSTRUCTORS = frozenset(
    {
        "Annotated",
        "Callable",
        "Literal",
        "Mapping",
        "NDArray",
        "Optional",
        "Sequence",
        "Union",
        "dict",
        "frozenset",
        "list",
        "set",
        "tuple",
        "type",
    }
)


# ---------------------------------------------------------------------------
# The probe.


def package_classes_in(hint: object, prefix: str = PREFIX) -> set[type]:
    """Every class from the package mentioned anywhere inside a resolved hint.

    ``typing.get_args(Callable[[A], B])`` is ``([A], B)``: the parameters of a
    callable arrive as a plain list, which is not a hint and has no arguments
    of its own, so a walk that only recurses through ``get_args`` never looks
    inside it. A class that appears only as a callback's parameter was
    invisible to the first version of this probe, and to the copies of it the
    sibling packages carry.
    """
    if isinstance(hint, list | tuple):
        return set().union(*(package_classes_in(item, prefix) for item in hint))
    found: set[type] = set()
    if isinstance(hint, type) and hint.__module__.startswith(prefix):
        found.add(hint)
    origin = typing.get_origin(hint)
    if isinstance(origin, type) and origin.__module__.startswith(prefix):
        found.add(origin)
    for argument in typing.get_args(hint):
        found |= package_classes_in(argument, prefix)
    return found


def _subscripted_name(node: ast.expr) -> str | None:
    """``list`` for ``list[...]``, ``NDArray`` for ``npt.NDArray[...]``."""
    if not isinstance(node, ast.Subscript):
        return None
    value = node.value
    if isinstance(value, ast.Name):
        return value.id
    if isinstance(value, ast.Attribute):
        return value.attr
    return None


def _is_type_expression(node: ast.expr) -> bool:
    """Whether a right-hand side reads as a type rather than a value."""
    if _subscripted_name(node) in TYPE_CONSTRUCTORS:
        return True
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        # ``int | str`` and ``float | None`` are types; ``1 | 2`` is a number.
        # Two named integer flags or'd together would read as a type too, and
        # that costs nothing: a name only matters here if an annotation spells it.
        return all(_is_type_operand(side) for side in (node.left, node.right))
    return False


def _is_type_operand(node: ast.expr) -> bool:
    if isinstance(node, ast.Constant):
        return node.value is None
    return isinstance(node, ast.Name | ast.Attribute) or _is_type_expression(node)


def aliases_in(source: str) -> frozenset[str]:
    """The module-level names a source file binds to a type expression.

    ``X: TypeAlias = ...`` and a ``type X = ...`` statement always count. A
    plain assignment counts when its value is a subscripted type constructor
    (``list[float]``, ``Literal["a"]``, ``npt.NDArray[np.float64]``) or a ``|``
    union. A number, a string, a call, or a bare name does not.
    """
    found: set[str] = set()
    for node in ast.parse(source).body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if "TypeAlias" in ast.unparse(node.annotation):
                found.add(node.target.id)
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and _is_type_expression(node.value):
                found.add(target.id)
        elif type(node).__name__ == "TypeAlias":  # ``type X = ...``, 3.12 and later
            found.add(node.name.id)  # type: ignore[attr-defined]
    return frozenset(found)


def names_in_annotation(node: ast.expr) -> frozenset[str]:
    """Every bare name an annotation spells, including inside a string."""
    found: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            found.add(child.id)
        elif isinstance(child, ast.Constant) and isinstance(child.value, str):
            try:
                inner = ast.parse(child.value, mode="eval")
            except SyntaxError:
                continue
            found |= names_in_annotation(inner.body)
    return frozenset(found)


def public_annotations(function: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.expr]:
    """The annotations of a function's public parameters and its return."""
    arguments = function.args
    parameters = [
        *arguments.posonlyargs,
        *arguments.args,
        *arguments.kwonlyargs,
        *(a for a in (arguments.vararg, arguments.kwarg) if a is not None),
    ]
    found = [
        parameter.annotation
        for parameter in parameters
        if parameter.annotation is not None and not parameter.arg.startswith("_")
    ]
    if function.returns is not None:
        found.append(function.returns)
    return found


def field_annotations(cls: ast.ClassDef) -> list[ast.expr]:
    """The annotations of a class body's public fields."""
    return [
        node.annotation
        for node in cls.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and not node.target.id.startswith("_")
    ]


# ---------------------------------------------------------------------------
# What the probe is pointed at.


def package_aliases() -> dict[str, list[tuple[str, object]]]:
    """Each alias name, with every (module, object) that binds it."""
    import importlib

    found: dict[str, list[tuple[str, object]]] = {}
    for info in pkgutil.iter_modules([str(PACKAGE)]):
        module_name = f"{PREFIX}{info.name}"
        source = (PACKAGE / f"{info.name}.py").read_text(encoding="utf-8")
        module = importlib.import_module(module_name)
        for name in aliases_in(source):
            if not name.startswith("_"):
                found.setdefault(name, []).append((module_name, getattr(module, name)))
    return found


def root_objects() -> dict[str, object]:
    return {name: getattr(moneyness, name) for name in moneyness.__all__}


def reachable_callables() -> Iterator[tuple[str, Callable[..., object]]]:
    """Each root function, and each public method or property of each root class."""
    for name, obj in sorted(root_objects().items()):
        if inspect.isfunction(obj):
            yield name, obj
        elif inspect.isclass(obj) and obj.__module__.startswith(PREFIX):
            seen: set[str] = set()
            for klass in obj.__mro__:
                if not klass.__module__.startswith(PREFIX):
                    continue
                for attribute, value in vars(klass).items():
                    if attribute in seen:
                        continue
                    seen.add(attribute)
                    if attribute.startswith("_") and attribute != "__init__":
                        continue
                    if isinstance(value, classmethod | staticmethod):
                        value = value.__func__
                    if isinstance(value, property):
                        value = value.fget
                    if inspect.isfunction(value):
                        yield f"{name}.{attribute}", value


def reachable_classes() -> Iterator[tuple[str, type]]:
    for name, obj in sorted(root_objects().items()):
        if inspect.isclass(obj) and obj.__module__.startswith(PREFIX):
            yield name, obj


def _resolved_public_hints(function: Callable[..., object]) -> list[object]:
    hints = typing.get_type_hints(function)
    return [hint for name, hint in hints.items() if not name.startswith("_")]


def _function_node(
    function: Callable[..., object],
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    # A dataclass's generated ``__init__`` has no source to read; its fields
    # are covered through the class body instead.
    try:
        source = textwrap.dedent(inspect.getsource(function))
    except (OSError, TypeError):
        return None
    node = ast.parse(source).body[0]
    return node if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) else None


def unreachable_classes() -> dict[str, list[str]]:
    """Reachable signatures and fields written in a class the root lacks."""
    root = root_objects()
    bound = {id(obj) for obj in root.values()}
    missing: dict[str, set[str]] = {}

    def note(label: str, hints: list[object]) -> None:
        for kind in set().union(*(package_classes_in(hint) for hint in hints)):
            if id(kind) not in bound:
                missing.setdefault(f"{kind.__module__}.{kind.__qualname__}", set()).add(label)

    for label, function in reachable_callables():
        note(label, _resolved_public_hints(function))
    for name, cls in reachable_classes():
        hints = typing.get_type_hints(cls)
        public = [hint for field, hint in hints.items() if not field.startswith("_")]
        note(f"{name} (fields)", public)
    return {kind: sorted(labels) for kind, labels in sorted(missing.items())}


def unreachable_aliases() -> dict[str, list[str]]:
    """Reachable signatures and fields spelled with an alias the root lacks."""
    aliases = package_aliases()
    root = root_objects()
    missing: dict[str, set[str]] = {}

    def note(label: str, annotations: list[ast.expr]) -> None:
        for annotation in annotations:
            for name in names_in_annotation(annotation) & aliases.keys():
                if not any(root.get(name) is value for _, value in aliases[name]):
                    owners = ", ".join(module for module, _ in aliases[name])
                    missing.setdefault(f"{owners}.{name}", set()).add(label)

    for label, function in reachable_callables():
        node = _function_node(function)
        if node is not None:
            note(label, public_annotations(node))
    for name, cls in reachable_classes():
        try:
            source = textwrap.dedent(inspect.getsource(cls))
        except (OSError, TypeError):
            continue
        statement = ast.parse(source).body[0]
        if isinstance(statement, ast.ClassDef):
            note(f"{name} (fields)", field_annotations(statement))
    return {alias: sorted(labels) for alias, labels in sorted(missing.items())}


# ---------------------------------------------------------------------------
# The checks.


def test_the_survey_is_worth_checking() -> None:
    # Every check below passes trivially over an empty walk, so the walk is
    # asserted first. The floors sit under what is there -- 181 root names,
    # 231 callables, 55 classes and one public alias, ``LocalVolatility``,
    # when this was written -- so that adding a module does not move them.
    assert len(moneyness.__all__) > 150
    assert sum(1 for _ in reachable_callables()) > 200
    assert sum(1 for _ in reachable_classes()) > 45
    assert "LocalVolatility" in package_aliases()


def test_a_class_in_a_reachable_signature_is_the_one_at_the_root() -> None:
    """The check that ``adaptive_quad(budget: _Budget | None)`` failed."""
    unreachable = unreachable_classes()
    assert not unreachable, (
        "reachable from moneyness, written in classes the root does not bind: "
        f"{unreachable}"
    )


def test_an_alias_a_reachable_signature_is_written_in_is_at_the_root() -> None:
    unreachable = unreachable_aliases()
    assert not unreachable, (
        "reachable from moneyness, written in type aliases the root does not bind: "
        f"{unreachable}"
    )


def test_the_quadrature_entry_point_takes_only_what_a_caller_can_name() -> None:
    """The defect, named, so a regression says what it is rather than listing."""
    hints = typing.get_type_hints(moneyness.adaptive_quad)
    assert "budget" not in hints, (
        "adaptive_quad takes a budget again; the allowance is a private _Budget, "
        "so pass it through _adaptive_quad instead"
    )


# ---------------------------------------------------------------------------
# The probe's own rules, against source written for each, since the package
# does not exercise every branch.


class _Local:
    pass


def test_classes_are_found_inside_every_kind_of_hint() -> None:
    prefix = _Local.__module__
    hints: list[object] = [
        _Local,
        list[_Local],
        _Local | None,
        dict[str, tuple[int, _Local]],
        Callable[[_Local], float],
        type[_Local],
    ]
    for hint in hints:
        assert package_classes_in(hint, prefix) == {_Local}, hint


def test_a_class_only_in_a_callbacks_parameters_is_found() -> None:
    # The branch the package's own signatures did not need and the test above
    # found missing: the parameter list of a ``Callable`` is a plain list.
    prefix = _Local.__module__
    assert package_classes_in(Callable[[int, _Local], None], prefix) == {_Local}
    assert package_classes_in(Callable[[Callable[[_Local], float]], float], prefix) == {_Local}


def test_classes_outside_the_package_are_not_found() -> None:
    assert package_classes_in(list[int] | dict[str, float] | None) == set()
    assert package_classes_in(Path) == set()


def test_aliases_are_the_assignments_that_read_as_types() -> None:
    source = textwrap.dedent(
        """
        from typing import Literal, TypeAlias
        import numpy.typing as npt
        Vector = list[float]
        Pair = tuple[float, float]
        Side = Literal["call", "put"]
        Array = npt.NDArray[np.float64]
        Either = int | str
        Maybe = float | None
        Explicit: TypeAlias = "dict[str, float]"
        """
    )
    assert aliases_in(source) == {"Vector", "Pair", "Side", "Array", "Either", "Maybe", "Explicit"}


def test_values_are_not_aliases() -> None:
    source = textwrap.dedent(
        """
        TOLERANCE = 1e-12
        NAME = "moneyness"
        DEFAULT = make_default()
        Same = Vector
        FLAGS = 1 | 2
        TABLE = {"a": 1}
        CHOICES = ("a", "b")
        LOOKUP = mapping["key"]
        count: int = 3
        """
    )
    assert aliases_in(source) == set()


def test_an_alias_inside_a_function_is_not_module_level() -> None:
    assert aliases_in("def f():\n    Local = list[float]\n") == set()


def test_annotation_names_include_those_inside_a_string() -> None:
    node = ast.parse("x: 'Vector | None'").body[0]
    assert isinstance(node, ast.AnnAssign)
    # ``None`` parses as a constant, not a name, so it is not among them.
    assert names_in_annotation(node.annotation) == {"Vector"}


def test_annotation_names_walk_into_subscripts() -> None:
    node = ast.parse("x: dict[str, Sequence[Vector]]").body[0]
    assert isinstance(node, ast.AnnAssign)
    assert {"dict", "str", "Sequence", "Vector"} <= names_in_annotation(node.annotation)


def test_an_unparseable_string_annotation_contributes_nothing() -> None:
    node = ast.parse("x: 'not a [type'").body[0]
    assert isinstance(node, ast.AnnAssign)
    assert names_in_annotation(node.annotation) == set()


def test_public_annotations_skip_private_parameters_and_keep_the_return() -> None:
    node = ast.parse(
        "def f(a: A, /, b: B, *args: C, _c: Hidden, d: D = 1, **kw: E) -> R: ..."
    ).body[0]
    assert isinstance(node, ast.FunctionDef)
    spelled = {ast.unparse(annotation) for annotation in public_annotations(node)}
    assert spelled == {"A", "B", "C", "D", "E", "R"}


def test_an_unannotated_parameter_contributes_nothing() -> None:
    node = ast.parse("def f(a, b: B): ...").body[0]
    assert isinstance(node, ast.FunctionDef)
    assert [ast.unparse(a) for a in public_annotations(node)] == ["B"]


def test_field_annotations_skip_private_fields_and_methods() -> None:
    node = ast.parse(
        "class C:\n    a: A\n    _b: Hidden\n    c = 1\n    def m(self) -> R: ...\n"
    ).body[0]
    assert isinstance(node, ast.ClassDef)
    assert [ast.unparse(a) for a in field_annotations(node)] == ["A"]
