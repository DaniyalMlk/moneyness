"""The linear solve, the cone projection and the simplex search.

These are the pieces calibration stands on, so they are checked against
properties that hold by definition rather than against the calibration that uses
them. A bug here would otherwise appear as a slightly worse fit somewhere else,
which is exactly the kind of failure that gets explained away.
"""

from __future__ import annotations

import math
import random

import pytest

from moneyness.optimise import cholesky_solve, nelder_mead, project_onto_cone


def _gram(rows: list[list[float]]) -> list[list[float]]:
    """``X^T X`` for a design matrix, the shape the solver is built for."""
    columns = len(rows[0])
    return [
        [sum(row[i] * row[j] for row in rows) for j in range(columns)] for i in range(columns)
    ]


def _matvec(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [sum(a * b for a, b in zip(row, vector, strict=True)) for row in matrix]


class TestCholesky:
    def test_solves_a_known_system(self) -> None:
        matrix = [[4.0, 1.0, 1.0], [1.0, 3.0, 0.0], [1.0, 0.0, 2.0]]
        solution = cholesky_solve(matrix, [6.0, 4.0, 3.0])
        assert solution == pytest.approx([1.0, 1.0, 1.0], abs=1e-12)

    @pytest.mark.parametrize("seed", range(25))
    def test_residual_vanishes_on_random_gram_matrices(self, seed: int) -> None:
        """The defining property: ``A x - b`` is zero.

        Random designs rather than hand-built matrices, because a
        hand-built one tends to be well conditioned by accident and never
        exercises the substitutions.
        """
        rng = random.Random(seed)
        size = rng.randint(2, 6)
        rows = [[rng.uniform(-3.0, 3.0) for _ in range(size)] for _ in range(size + 4)]
        matrix = _gram(rows)
        expected = [rng.uniform(-2.0, 2.0) for _ in range(size)]
        rhs = _matvec(matrix, expected)

        solution = cholesky_solve(matrix, rhs)
        residual = [a - b for a, b in zip(_matvec(matrix, solution), rhs, strict=True)]
        scale = max(abs(x) for x in rhs) or 1.0
        assert max(abs(r) for r in residual) < 1e-8 * scale

    def test_rejects_a_singular_matrix(self) -> None:
        """A dependent column is a statement about the data, not a numerical hiccup.

        Two identical columns mean the quotes cannot distinguish two
        parameters, and the solver says so rather than returning one of the
        infinitely many answers.
        """
        rows = [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
        with pytest.raises(ValueError, match="positive definite"):
            cholesky_solve(_gram(rows), [1.0, 1.0])

    def test_rejects_an_indefinite_matrix(self) -> None:
        with pytest.raises(ValueError, match="positive definite"):
            cholesky_solve([[-1.0, 0.0], [0.0, 1.0]], [1.0, 1.0])

    @pytest.mark.parametrize(
        ("matrix", "rhs"),
        [
            ([], [1.0]),
            ([[1.0, 0.0]], [1.0]),
            ([[1.0, 0.0], [0.0, 1.0]], [1.0]),
        ],
    )
    def test_rejects_malformed_input(self, matrix: list[list[float]], rhs: list[float]) -> None:
        with pytest.raises(ValueError):
            cholesky_solve(matrix, rhs)


class TestConeProjection:
    @pytest.mark.parametrize(
        ("point", "expected"),
        [
            ((1.0, 2.0), (1.0, 2.0)),  # already inside
            ((0.0, 0.0), (0.0, 0.0)),  # the apex
            ((3.0, 1.0), (2.0, 2.0)),  # right face
            ((-3.0, 1.0), (-2.0, 2.0)),  # left face
            ((1.0, -3.0), (0.0, 0.0)),  # polar cone collapses to the apex
            ((-1.0, -1.0), (0.0, 0.0)),  # on the polar boundary
        ],
    )
    def test_known_projections(
        self, point: tuple[float, float], expected: tuple[float, float]
    ) -> None:
        assert project_onto_cone(*point) == pytest.approx(expected, abs=1e-12)

    @pytest.mark.parametrize("seed", range(40))
    def test_is_the_nearest_feasible_point(self, seed: int) -> None:
        """Checked against a brute-force search over the feasible set.

        The projection has a closed form, so the test needs an independent
        witness. A dense sweep of the cone is slow and crude but shares no
        algebra with the implementation, which is the point.
        """
        rng = random.Random(seed)
        d0, c0 = rng.uniform(-4.0, 4.0), rng.uniform(-4.0, 4.0)
        d, c = project_onto_cone(d0, c0)

        assert c >= abs(d) - 1e-12, "the projection must land in the cone"

        best = math.hypot(d - d0, c - c0)
        for i in range(2001):
            trial_d = -6.0 + i * 12.0 / 2000
            for trial_c in (abs(trial_d), abs(trial_d) + rng.uniform(0.0, 6.0)):
                distance = math.hypot(trial_d - d0, trial_c - c0)
                assert distance >= best - 1e-9, (
                    f"({trial_d}, {trial_c}) is feasible and closer to ({d0}, {c0})"
                )

    @pytest.mark.parametrize("seed", range(20))
    def test_is_idempotent(self, seed: int) -> None:
        rng = random.Random(seed)
        first = project_onto_cone(rng.uniform(-5.0, 5.0), rng.uniform(-5.0, 5.0))
        assert project_onto_cone(*first) == pytest.approx(first, abs=1e-15)


class TestNelderMead:
    def test_finds_the_minimum_of_a_quadratic(self) -> None:
        result = nelder_mead(
            lambda p: (p[0] - 3.0) ** 2 + (p[1] + 1.0) ** 2, (0.0, 0.0), tolerance=1e-12
        )
        assert result.converged
        assert result.point == pytest.approx((3.0, -1.0), abs=1e-6)

    def test_crosses_the_rosenbrock_valley(self) -> None:
        """The standard hard case for a derivative-free method.

        The minimum sits at the end of a curved valley whose floor is nearly
        flat, so a search that judges convergence on function values alone stops
        early and one that judges on the simplex alone never stops. Getting here
        is what the two-part convergence test in the implementation buys.
        """
        result = nelder_mead(
            lambda p: (1.0 - p[0]) ** 2 + 100.0 * (p[1] - p[0] ** 2) ** 2,
            (-1.2, 1.0),
            step=0.2,
            tolerance=1e-12,
            max_iterations=5000,
        )
        assert result.converged
        assert result.point == pytest.approx((1.0, 1.0), abs=1e-5)
        assert result.value < 1e-12

    def test_handles_one_dimension(self) -> None:
        result = nelder_mead(lambda p: (p[0] - 2.5) ** 4, (0.0,), tolerance=1e-14)
        assert result.point[0] == pytest.approx(2.5, abs=1e-3)

    def test_an_infinite_objective_marks_a_point_infeasible(self) -> None:
        """Constraints reach the search as ``inf``, and it must not chase them.

        The calibration expresses "this vertex has no linear solution" exactly
        this way, so a search that wandered into the rejected region and stayed
        there would silently return a point that means nothing.
        """

        def constrained(point: tuple[float, ...]) -> float:
            if point[0] < 0.0:
                return math.inf
            return (point[0] - 1.0) ** 2 + (point[1] - 1.0) ** 2

        result = nelder_mead(constrained, (2.0, 2.0), tolerance=1e-12)
        assert result.point[0] >= 0.0
        assert result.point == pytest.approx((1.0, 1.0), abs=1e-5)

    def test_reports_failure_rather_than_pretending(self) -> None:
        """An exhausted budget comes back as ``converged=False``, with the best point.

        The distinction matters to the calibration, which compares results from
        several starts: a point that merely ran out of iterations is still worth
        comparing, but a caller that treated it as converged would be reporting
        a fit quality it has no evidence for.
        """
        result = nelder_mead(
            lambda p: (1.0 - p[0]) ** 2 + 100.0 * (p[1] - p[0] ** 2) ** 2,
            (-1.2, 1.0),
            max_iterations=5,
        )
        assert not result.converged
        assert result.iterations == 5

    def test_rejects_an_empty_start(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            nelder_mead(lambda p: 0.0, ())

    def test_rejects_an_infeasible_start(self) -> None:
        with pytest.raises(ValueError, match="not finite"):
            nelder_mead(lambda p: math.inf, (1.0, 1.0))

    @pytest.mark.parametrize("scale", [1e-6, 1.0, 1e6])
    def test_copes_with_parameters_of_very_different_scale(self, scale: float) -> None:
        """The initial simplex steps relatively, which is why this works.

        A fixed absolute step would be far too large for the small coordinate
        and far too small for the large one, and the search would spend its
        budget on one axis.
        """
        target = (scale, 1.0 / scale)

        def objective(p: tuple[float, ...]) -> float:
            return ((p[0] - target[0]) / scale) ** 2 + ((p[1] - target[1]) * scale) ** 2

        result = nelder_mead(objective, (scale * 0.5, 2.0 / scale), tolerance=1e-14)
        assert result.value < 1e-12
