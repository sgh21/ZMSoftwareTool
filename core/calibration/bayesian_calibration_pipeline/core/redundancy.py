"""Correlation analysis for multisource calibration parameters."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb

import numpy as np
from scipy.linalg import qr

from core.calibration.bayesian_calibration_pipeline.core.analytic_jacobian import (
    geometry33_output_jacobian,
    supports_geometry33_jacobian,
)
from core.calibration.bayesian_calibration_pipeline.core.parameters import ErrorParameter, parameter_scales
from core.calibration.bayesian_calibration_pipeline.core.robot_model import MultiSourceRobotModel


@dataclass
class RedundancyResult:
    """Result of the Jacobian rank/correlation analysis."""

    jacobian: np.ndarray
    normal_matrix: np.ndarray
    nullspace: np.ndarray
    correlated_sets: list[list[int]]
    independent_indices: list[int]
    redundant_indices: list[int]
    rank: int
    nullity: int
    condition_number: float
    singular_values: np.ndarray
    normal_singular_values: np.ndarray
    used_exhaustive_search: bool


def output_jacobian(
    model: MultiSourceRobotModel,
    joint_configs: np.ndarray,
    error_vector: np.ndarray,
    parameters: list[ErrorParameter],
    payloads: np.ndarray | float | None = None,
    directions: np.ndarray | None = None,
    step_ratio: float = 1.0e-4,
    method: str = "auto",
) -> np.ndarray:
    """Jacobian of stacked xyz outputs wrt parameters.

    ``method='auto'`` uses the analytic geometry33 Jacobian when all
    parameters are supported, and otherwise falls back to finite differences.
    """
    method_key = str(method).lower()
    if method_key not in ("auto", "analytic", "finite_difference"):
        raise ValueError(f"Unknown Jacobian method: {method!r}")
    if method_key in ("auto", "analytic") and supports_geometry33_jacobian(parameters):
        return geometry33_output_jacobian(
            model,
            joint_configs,
            error_vector,
            parameters,
            payloads,
            directions,
        )
    if method_key == "analytic":
        raise ValueError("Analytic Jacobian is currently implemented only for geometry33 parameters.")
    return finite_difference_output_jacobian(
        model,
        joint_configs,
        error_vector,
        parameters,
        payloads=payloads,
        directions=directions,
        step_ratio=step_ratio,
    )


def finite_difference_output_jacobian(
    model: MultiSourceRobotModel,
    joint_configs: np.ndarray,
    error_vector: np.ndarray,
    parameters: list[ErrorParameter],
    payloads: np.ndarray | float | None = None,
    directions: np.ndarray | None = None,
    step_ratio: float = 1.0e-4,
) -> np.ndarray:
    """Finite-difference Jacobian of stacked xyz outputs wrt parameters."""
    x0 = np.asarray(error_vector, dtype=float).reshape(len(parameters))
    y0 = model.batch_positions(joint_configs, x0, parameters, payloads, directions).reshape(-1)
    scales = parameter_scales(parameters)
    jacobian = np.zeros((y0.size, x0.size), dtype=float)
    for index in range(x0.size):
        step = max(scales[index] * step_ratio, 1.0e-8)
        x_step = x0.copy()
        x_step[index] += step
        y_step = model.batch_positions(
            joint_configs, x_step, parameters, payloads, directions
        ).reshape(-1)
        jacobian[:, index] = (y_step - y0) / step
    return jacobian


def analyze_redundancy(
    model: MultiSourceRobotModel,
    joint_configs: np.ndarray,
    error_vector: np.ndarray,
    parameters: list[ErrorParameter],
    payloads: np.ndarray | float | None = None,
    directions: np.ndarray | None = None,
    tolerance: float = 1.0e-7,
    max_combinations: int = 200_000,
    preferred_indices: list[int] | None = None,
    jacobian_method: str = "auto",
) -> RedundancyResult:
    """Identify correlated parameters following the paper's SVD criterion.

    The paper forms ``T = J.T @ J`` and applies SVD. If ``rank(T)=r<n``,
    the last ``n-r`` right-singular vectors span the parameter nullspace. A
    candidate set of ``n-r`` parameters is removable when:

    1. the candidate rows of the nullspace have full rank ``n-r``; and
    2. the remaining columns of ``T`` keep rank ``r``.

    Exact enumeration is used when the number of combinations is practical.
    For the full 54-parameter baseline the exact search can be combinatorial.
    If the strict nullspace fallback cannot find a removable set, a
    rank-preserving independent set is selected from high-priority parameters
    first, then QR pivots.  The remaining parameters are fixed as redundant.
    """
    jacobian = output_jacobian(
        model,
        joint_configs,
        error_vector,
        parameters,
        payloads,
        directions,
        method=jacobian_method,
    )
    normal_matrix = jacobian.T @ jacobian
    _, normal_singular_values, vh = np.linalg.svd(normal_matrix, full_matrices=True)
    if normal_singular_values.size == 0 or normal_singular_values[0] <= 1.0e-30:
        rank = 0
    else:
        rank = int(np.sum(normal_singular_values > tolerance * normal_singular_values[0]))

    n_params = len(parameters)
    nullity = n_params - rank
    nullspace = vh.T[:, rank:] if nullity > 0 else np.zeros((n_params, 0), dtype=float)
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    positive_singular_values = singular_values[singular_values > 1.0e-15]
    if positive_singular_values.size == 0:
        condition = float("inf")
    else:
        condition = float(positive_singular_values[0] / positive_singular_values[-1])

    if nullity <= 0:
        correlated_sets: list[list[int]] = []
        redundant = []
        independent = list(range(n_params))
        used_exhaustive = True
    else:
        correlated_sets, used_exhaustive = _paper_correlated_sets(
            normal_matrix=normal_matrix,
            nullspace=nullspace,
            rank=rank,
            nullity=nullity,
            tolerance=tolerance,
            max_combinations=max_combinations,
        )
        if correlated_sets:
            redundant = correlated_sets[0]
            independent = [index for index in range(n_params) if index not in set(redundant)]
        else:
            independent = _rank_preserving_independent_columns(
                jacobian,
                rank,
                tolerance,
                preferred_indices,
            )
            independent_set = set(independent)
            redundant = [index for index in range(n_params) if index not in independent_set]
            correlated_sets = [redundant] if redundant else []

    return RedundancyResult(
        jacobian=jacobian,
        normal_matrix=normal_matrix,
        nullspace=nullspace,
        correlated_sets=correlated_sets,
        independent_indices=independent,
        redundant_indices=redundant,
        rank=rank,
        nullity=nullity,
        condition_number=condition,
        singular_values=singular_values,
        normal_singular_values=normal_singular_values,
        used_exhaustive_search=used_exhaustive,
    )


def _paper_correlated_sets(
    normal_matrix: np.ndarray,
    nullspace: np.ndarray,
    rank: int,
    nullity: int,
    tolerance: float,
    max_combinations: int,
) -> tuple[list[list[int]], bool]:
    """Return removable parameter sets using the paper's two rank tests."""
    n_params = normal_matrix.shape[1]
    combination_count = comb(n_params, nullity)
    if combination_count <= max_combinations:
        valid_sets = []
        for candidate in combinations(range(n_params), nullity):
            if _is_paper_correlated_set(
                normal_matrix, nullspace, candidate, rank, nullity, tolerance
            ):
                valid_sets.append([int(index) for index in candidate])
        return valid_sets, True

    candidate = _pivot_nullspace_rows(nullspace, nullity)
    if _is_paper_correlated_set(
        normal_matrix, nullspace, candidate, rank, nullity, tolerance
    ):
        return [[int(index) for index in candidate]], False
    return [], False


def _is_paper_correlated_set(
    normal_matrix: np.ndarray,
    nullspace: np.ndarray,
    candidate: tuple[int, ...] | list[int],
    rank: int,
    nullity: int,
    tolerance: float,
) -> bool:
    """Check the paper's ``rank(V_tilde)`` and ``rank(T_tilde)`` criteria."""
    candidate_indices = list(candidate)
    v_tilde = nullspace[candidate_indices, :]
    if _matrix_rank(v_tilde, tolerance) != nullity:
        return False
    remaining = [index for index in range(normal_matrix.shape[1]) if index not in candidate_indices]
    t_tilde = normal_matrix[:, remaining]
    return _matrix_rank(t_tilde, tolerance) == rank


def _pivot_nullspace_rows(nullspace: np.ndarray, nullity: int) -> list[int]:
    """Select full-rank nullspace rows when exhaustive enumeration is too large."""
    _, _, pivots = qr(nullspace.T, mode="economic", pivoting=True)
    return sorted(int(index) for index in pivots[:nullity])


def _pivot_independent_columns(jacobian: np.ndarray, rank: int) -> list[int]:
    """Select one identifiable parameter set when paper enumeration is impractical.

    The paper removes ``n-r`` correlated parameters and only identifies the
    remaining ``r`` independent parameters.  Exhaustively enumerating all
    removable sets is combinatorial for the 54-parameter model, so column
    pivoting on the output Jacobian gives one numerically independent set with
    the same rank.
    """
    if rank <= 0:
        return []
    _, _, pivots = qr(jacobian, mode="economic", pivoting=True)
    return sorted(int(index) for index in pivots[:rank])


def _rank_preserving_independent_columns(
    jacobian: np.ndarray,
    rank: int,
    tolerance: float,
    preferred_indices: list[int] | None,
) -> list[int]:
    """Pick high-priority independent columns, then fill gaps with QR pivots."""
    if rank <= 0:
        return []
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    if singular_values.size == 0 or singular_values[0] <= 1.0e-30:
        return []
    absolute_tolerance = float(tolerance * singular_values[0])

    selected: list[int] = []
    seen: set[int] = set()
    preferred = [] if preferred_indices is None else list(preferred_indices)
    for index in preferred + _pivot_column_order(jacobian):
        if index in seen or index < 0 or index >= jacobian.shape[1]:
            continue
        seen.add(index)
        candidate_rank = _absolute_matrix_rank(
            jacobian[:, selected + [index]],
            absolute_tolerance,
        )
        if candidate_rank > len(selected):
            selected.append(int(index))
            if len(selected) >= rank:
                break
    return sorted(selected)


def _pivot_column_order(jacobian: np.ndarray) -> list[int]:
    _, _, pivots = qr(jacobian, mode="economic", pivoting=True)
    return [int(index) for index in pivots]


def _matrix_rank(matrix: np.ndarray, tolerance: float) -> int:
    """Rank with the same relative tolerance style used for SVD of ``T``."""
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    if singular_values.size == 0 or singular_values[0] <= 1.0e-30:
        return 0
    return int(np.sum(singular_values > tolerance * singular_values[0]))


def _absolute_matrix_rank(matrix: np.ndarray, tolerance: float) -> int:
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    if singular_values.size == 0:
        return 0
    return int(np.sum(singular_values > tolerance))


