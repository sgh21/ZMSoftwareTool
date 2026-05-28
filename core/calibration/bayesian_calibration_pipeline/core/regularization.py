"""Reusable regularized LM utilities for calibration experiments.

Supports three regularization norms over scaled parameters z_i = theta_i / scale_i:

* ``l2``        鈥?``lambda * sum z_i^2`` (Tikhonov). Single LM call.
* ``smooth_l1`` 鈥?``lambda * sum H_beta(z_i)`` with Huber kernel
  ``H_beta(z) = z^2/(2*beta)`` for ``|z| <= beta`` else ``|z| - beta/2``.
  H_beta is C^1, so an exact ``r = sign(z)*sqrt(2*lambda*H_beta(|z|))``
  square-root residual can be plugged into a single LM call (exact solver).
* ``l1``        鈥?``lambda * sum |z_i|`` (non-smooth). Solved by ADMM:
  ``theta`` step is a Tikhonov-augmented LM; ``z`` step is a closed-form
  soft-threshold; ``u`` is the scaled dual variable. Because the robot model is
  nonlinear, this is a practical nonconvex ADMM solver rather than a global
  optimality guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from scipy.optimize import least_squares

from core.calibration.bayesian_calibration_pipeline.core.parameters import (
    ErrorParameter,
    parameter_scales,
    zero_error_vector,
)
from core.calibration.bayesian_calibration_pipeline.core.redundancy import RedundancyResult, analyze_redundancy
from core.calibration.bayesian_calibration_pipeline.core.robot_model import MultiSourceRobotModel


VALID_NORMS = ("l2", "l1", "smooth_l1")


@dataclass
class RegularizedLMResult:
    """Result of fitting a fixed active parameter set."""

    vector: np.ndarray
    active_indices: list[int]
    lambda_value: float
    cost: float
    component_rmse: float
    position_residual_norm: float
    normalized_parameter_norm: float
    weighted_normalized_parameter_l2: float
    regularized_residual_norm: float
    nfev: int
    success: bool
    message: str
    norm: str = "l2"
    penalty_value: float = 0.0
    regularization_weight_min: float = 1.0
    regularization_weight_max: float = 1.0
    admm_iterations: int = 0
    admm_primal_residual: float = 0.0
    admm_dual_residual: float = 0.0
    final_z: np.ndarray | None = field(default=None, repr=False)
    final_u: np.ndarray | None = field(default=None, repr=False)


@dataclass
class LambdaScore:
    """Cross-validation score for one lambda value."""

    lambda_value: float
    fold_rmse: list[float]
    mean_rmse: float
    max_rmse: float
    std_rmse: float


@dataclass
class LCurvePoint:
    """One point on the L curve."""

    lambda_value: float
    residual_norm: float
    normalized_parameter_norm: float
    train_rmse: float
    nfev: int


def select_independent_parameters(
    model: MultiSourceRobotModel,
    joint_configs: np.ndarray,
    parameters: list[ErrorParameter],
    payloads: np.ndarray | float | None = None,
    directions: np.ndarray | None = None,
    tolerance: float = 1.0e-7,
    max_combinations: int = 200_000,
    jacobian_method: str = "auto",
) -> RedundancyResult:
    """Run the project SVD redundancy analysis and return independent indices."""
    return analyze_redundancy(
        model,
        np.asarray(joint_configs, dtype=float).reshape(-1, 6),
        zero_error_vector(parameters),
        parameters,
        payloads=payloads,
        directions=directions,
        tolerance=tolerance,
        max_combinations=max_combinations,
        preferred_indices=None,
        jacobian_method=jacobian_method,
    )


def fit_l2_lm(
    model: MultiSourceRobotModel,
    joint_configs: np.ndarray,
    measured_positions: np.ndarray,
    parameters: list[ErrorParameter],
    active_indices: Iterable[int],
    lambda_value: float = 0.0,
    payloads: np.ndarray | float | None = None,
    directions: np.ndarray | None = None,
    initial_vector: np.ndarray | None = None,
    max_nfev: int = 120,
    norm: str = "l2",
    smooth_l1_beta: float = 1.0e-3,
    admm_rho: float = 0.0,
    admm_max_iter: int = 80,
    admm_tol_abs: float = 1.0e-6,
    admm_tol_rel: float = 1.0e-4,
    admm_warm_start: tuple[np.ndarray, np.ndarray] | None = None,
    regularization_weights: np.ndarray | None = None,
    position_noise_std_m: float | None = None,
) -> RegularizedLMResult:
    """Fit active parameters with the requested regularization norm.

    ``norm='l2'`` (default) uses a Tikhonov objective. With
    ``position_noise_std_m=None`` it keeps the historical form
    ``||p(theta)-y||^2 + lambda * sum((theta_i/scale_i)^2)``; with a positive
    noise value it uses the MAP-scaled data term
    ``||(p(theta)-y)/sigma||^2 + lambda * sum((theta_i/scale_i)^2)``.
    ``norm='smooth_l1'``
    is represented by exact square-root residuals. ``norm='l1'`` uses ADMM with
    exact soft-thresholding for the nonsmooth split variable. ``scale_i`` comes
    from :func:`parameter_scales`, so the penalty is applied in prior-scale
    units instead of raw mixed units.

    The function name keeps the historical ``fit_l2_lm`` for backward
    compatibility, but the same call site now dispatches over all three norms.
    """
    norm_key = (norm or "l2").lower()
    if norm_key not in VALID_NORMS:
        raise ValueError(f"Unknown norm {norm!r}; expected one of {VALID_NORMS}")

    joints = np.asarray(joint_configs, dtype=float).reshape(-1, 6)
    measured = np.asarray(measured_positions, dtype=float).reshape(-1, 3)
    active = _unique([int(index) for index in active_indices])
    full = _initial_vector(initial_vector, len(parameters))
    scales = parameter_scales(parameters)
    weights = _regularization_weights(regularization_weights, len(parameters))
    lam = max(float(lambda_value), 0.0)
    noise_std = _position_noise_std(position_noise_std_m)

    if regularization_weights is not None and norm_key != "l2":
        raise ValueError("regularization_weights is currently supported only for norm='l2'")

    if not active:
        position_residual = _position_residuals(
            model, joints, measured, full, parameters, payloads, directions
        )
        return _empty_result(full, active, lam, position_residual, norm_key)

    if lam <= 0.0:
        norm_key = "l2"

    if norm_key == "smooth_l1":
        return _fit_smooth_l1(
            full, active, scales, lam, smooth_l1_beta,
            model, joints, measured, parameters, payloads, directions, max_nfev,
        )
    if norm_key == "l1":
        return _fit_l1_admm(
            full, active, scales, lam,
            model, joints, measured, parameters, payloads, directions, max_nfev,
            admm_rho, admm_max_iter, admm_tol_abs, admm_tol_rel, admm_warm_start,
        )
    return _fit_l2(
        full, active, scales, weights, lam,
        model, joints, measured, parameters, payloads, directions, max_nfev,
        position_noise_std_m=noise_std,
    )


def regularization_penalty(
    vector: np.ndarray,
    parameters: list[ErrorParameter],
    active_indices: Iterable[int],
    norm: str,
    smooth_l1_beta: float = 1.0e-3,
) -> float:
    """Return the scale-normalized penalty value sum_i f_norm(theta_i/scale_i).

    The lambda multiplier is *not* applied 鈥?callers usually want the raw
    norm value for diagnostics and L-curve plots.
    """
    norm_key = (norm or "l2").lower()
    if norm_key not in VALID_NORMS:
        raise ValueError(f"Unknown norm {norm!r}; expected one of {VALID_NORMS}")
    scales = parameter_scales(parameters)
    active = np.asarray(list(active_indices), dtype=int)
    if active.size == 0:
        return 0.0
    z = np.asarray(vector, dtype=float)[active] / np.maximum(scales[active], 1.0e-20)
    if norm_key == "l2":
        return float(np.sum(z * z))
    if norm_key == "l1":
        return float(np.sum(np.abs(z)))
    beta = max(float(smooth_l1_beta), 1.0e-20)
    abs_z = np.abs(z)
    quadratic = abs_z <= beta
    return float(
        np.sum(np.where(quadratic, z * z / (2.0 * beta), abs_z - beta / 2.0))
    )


def _fit_l2(
    full: np.ndarray,
    active: list[int],
    scales: np.ndarray,
    weights: np.ndarray,
    lam: float,
    model: MultiSourceRobotModel,
    joints: np.ndarray,
    measured: np.ndarray,
    parameters: list[ErrorParameter],
    payloads: np.ndarray | float | None,
    directions: np.ndarray | None,
    max_nfev: int,
    *,
    position_noise_std_m: float,
) -> RegularizedLMResult:
    x0 = full[active].copy()
    residual_count = measured.size + (len(active) if lam > 0.0 else 0)
    method = "lm" if residual_count >= len(active) else "trf"
    result = least_squares(
        _l2_residuals,
        x0=x0,
        args=(
            full,
            active,
            model,
            joints,
            measured,
            parameters,
            payloads,
            directions,
            scales,
            weights,
            lam,
            position_noise_std_m,
        ),
        method=method,
        x_scale=np.maximum(scales[active], 1.0e-12),
        max_nfev=int(max_nfev),
        ftol=1.0e-10,
        xtol=1.0e-10,
        gtol=1.0e-10,
    )
    full[active] = result.x
    return _finalize_result(
        full, active, scales, weights, lam, "l2",
        model, joints, measured, parameters, payloads, directions,
        nfev=int(result.nfev),
        success=bool(result.success),
        message=str(result.message),
        cost=float(result.cost),
        position_noise_std_m=position_noise_std_m,
    )


def _fit_smooth_l1(
    full: np.ndarray,
    active: list[int],
    scales: np.ndarray,
    lam: float,
    beta: float,
    model: MultiSourceRobotModel,
    joints: np.ndarray,
    measured: np.ndarray,
    parameters: list[ErrorParameter],
    payloads: np.ndarray | float | None,
    directions: np.ndarray | None,
    max_nfev: int,
) -> RegularizedLMResult:
    x0 = full[active].copy()
    residual_count = measured.size + (len(active) if lam > 0.0 else 0)
    method = "lm" if residual_count >= len(active) else "trf"
    result = least_squares(
        _smooth_l1_residuals,
        x0=x0,
        args=(
            full,
            active,
            model,
            joints,
            measured,
            parameters,
            payloads,
            directions,
            scales,
            lam,
            max(float(beta), 1.0e-20),
        ),
        method=method,
        x_scale=np.maximum(scales[active], 1.0e-12),
        max_nfev=int(max_nfev),
        ftol=1.0e-10,
        xtol=1.0e-10,
        gtol=1.0e-10,
    )
    full[active] = result.x
    return _finalize_result(
        full, active, scales, np.ones_like(scales), lam, "smooth_l1",
        model, joints, measured, parameters, payloads, directions,
        nfev=int(result.nfev),
        success=bool(result.success),
        message=str(result.message),
        cost=float(result.cost),
        smooth_l1_beta=beta,
    )


def _fit_l1_admm(
    full: np.ndarray,
    active: list[int],
    scales: np.ndarray,
    lam: float,
    model: MultiSourceRobotModel,
    joints: np.ndarray,
    measured: np.ndarray,
    parameters: list[ErrorParameter],
    payloads: np.ndarray | float | None,
    directions: np.ndarray | None,
    max_nfev: int,
    rho: float,
    max_iter: int,
    tol_abs: float,
    tol_rel: float,
    warm_start: tuple[np.ndarray, np.ndarray] | None,
) -> RegularizedLMResult:
    active_arr = np.asarray(active, dtype=int)
    n = len(active)
    active_scales = np.maximum(scales[active_arr], 1.0e-20)
    # rho <= 0 triggers automatic balancing so that the theta-step proximal
    # term (rho/2)*||w-target||^2 has magnitude comparable to ||p-y||^2 at the
    # initial point. Without this, position residuals (~1e-3 m, squared ~1e-6)
    # are dwarfed by the proximal term (~1 with rho=1) and ADMM collapses to
    # the all-zero soft-threshold fixed point regardless of lambda.
    if rho > 0.0:
        rho_eff = float(rho)
    else:
        p0 = model.batch_positions(joints, full, parameters, payloads, directions)
        data_norm_sq = float(np.sum((p0 - measured).reshape(-1) ** 2))
        w_typical_sq = float(max(n, 1))
        rho_eff = max(2.0 * data_norm_sq / w_typical_sq, 1.0e-30)
    theta = full[active_arr].astype(float).copy()
    nfev_total = 0
    if warm_start is None:
        # Nonconvex ADMM is sensitive to the initial split point. Starting at
        # zero makes the first theta-step behave like a strong ridge-to-zero
        # solve and can over-shrink even for tiny lambda. Use the no-reg LM
        # solution as the default split initialization so lambda -> 0 recovers
        # the unregularized independent-parameter fit.
        init_result = least_squares(
            _l2_residuals,
            x0=theta,
            args=(
                full,
                active,
                model,
                joints,
                measured,
                parameters,
                payloads,
                directions,
                scales,
                np.ones_like(scales),
                0.0,
                1.0,
            ),
            method="lm" if measured.size >= n else "trf",
            x_scale=np.maximum(scales[active_arr], 1.0e-12),
            max_nfev=int(max_nfev),
            ftol=1.0e-10,
            xtol=1.0e-10,
            gtol=1.0e-10,
        )
        theta = init_result.x
        full[active_arr] = theta
        nfev_total += int(init_result.nfev)
    z = theta / active_scales
    u = np.zeros(n, dtype=float)
    if warm_start is not None:
        z_init, u_init = warm_start
        if z_init is not None and len(np.asarray(z_init).reshape(-1)) == n:
            z = np.asarray(z_init, dtype=float).reshape(n).copy()
        if u_init is not None and len(np.asarray(u_init).reshape(-1)) == n:
            u = np.asarray(u_init, dtype=float).reshape(n).copy()

    last_message = "admm did not iterate"
    last_success = False
    primal_res = 0.0
    dual_res = 0.0
    iters = 0
    residual_count = measured.size + n
    method = "lm" if residual_count >= n else "trf"

    for it in range(max(int(max_iter), 1)):
        target_w = z - u
        result = least_squares(
            _admm_theta_residuals,
            x0=theta,
            args=(
                full,
                active,
                model,
                joints,
                measured,
                parameters,
                payloads,
                directions,
                active_scales,
                target_w,
                rho_eff,
            ),
            method=method,
            x_scale=np.maximum(scales[active_arr], 1.0e-12),
            max_nfev=int(max_nfev),
            ftol=1.0e-10,
            xtol=1.0e-10,
            gtol=1.0e-10,
        )
        theta = result.x
        nfev_total += int(result.nfev)
        last_message = str(result.message)
        last_success = bool(result.success)

        w = theta / active_scales
        z_prev = z
        v = w + u
        threshold = lam / rho_eff
        z = np.sign(v) * np.maximum(np.abs(v) - threshold, 0.0)
        u = u + w - z

        primal_res = float(np.linalg.norm(w - z))
        dual_res = float(rho_eff * np.linalg.norm(z - z_prev))
        eps_pri = np.sqrt(n) * tol_abs + tol_rel * max(
            float(np.linalg.norm(w)),
            float(np.linalg.norm(z)),
        )
        eps_dual = np.sqrt(n) * tol_abs + tol_rel * float(np.linalg.norm(rho_eff * u))
        iters = it + 1
        if primal_res <= eps_pri and dual_res <= eps_dual:
            break

    full[active_arr] = z * active_scales
    return _finalize_result(
        full, active, scales, np.ones_like(scales), lam, "l1",
        model, joints, measured, parameters, payloads, directions,
        nfev=int(nfev_total),
        success=bool(last_success),
        message=(
            f"admm_iters={iters}, primal={primal_res:.2e}, dual={dual_res:.2e}; "
            f"last_lm={last_message}"
        ),
        cost=None,
        admm_iterations=int(iters),
        admm_primal=float(primal_res),
        admm_dual=float(dual_res),
        z_final=z,
        u_final=u,
    )


def _finalize_result(
    full: np.ndarray,
    active: list[int],
    scales: np.ndarray,
    weights: np.ndarray,
    lam: float,
    norm_key: str,
    model: MultiSourceRobotModel,
    joints: np.ndarray,
    measured: np.ndarray,
    parameters: list[ErrorParameter],
    payloads: np.ndarray | float | None,
    directions: np.ndarray | None,
    *,
    nfev: int,
    success: bool,
    message: str,
    cost: float | None,
    smooth_l1_beta: float = 1.0e-3,
    admm_iterations: int = 0,
    admm_primal: float = 0.0,
    admm_dual: float = 0.0,
    z_final: np.ndarray | None = None,
    u_final: np.ndarray | None = None,
    position_noise_std_m: float = 1.0,
) -> RegularizedLMResult:
    active_arr = np.asarray(active, dtype=int)
    position_residual = _position_residuals(
        model, joints, measured, full, parameters, payloads, directions
    )
    normalized = full[active_arr] / np.maximum(scales[active_arr], 1.0e-20)
    active_weights = weights[active_arr] if active_arr.size else np.ones(0, dtype=float)
    residual_norm = float(np.linalg.norm(position_residual))
    scaled_residual_norm = float(np.linalg.norm(position_residual / _position_noise_std(position_noise_std_m)))
    parameter_norm = float(np.linalg.norm(normalized))
    weighted_parameter_norm = float(np.sqrt(np.sum(active_weights * normalized * normalized)))
    if norm_key == "l2":
        penalty_raw = float(np.sum(active_weights * normalized * normalized))
    else:
        penalty_raw = regularization_penalty(
            full, parameters, active, norm_key, smooth_l1_beta=smooth_l1_beta
        )
    penalty_value = float(lam * penalty_raw)
    regularized_norm = float(np.sqrt(scaled_residual_norm * scaled_residual_norm + penalty_value))
    final_cost = float(cost) if cost is not None else 0.5 * scaled_residual_norm * scaled_residual_norm + penalty_value
    return RegularizedLMResult(
        vector=full,
        active_indices=active,
        lambda_value=float(lam),
        cost=final_cost,
        component_rmse=_rmse(position_residual),
        position_residual_norm=residual_norm,
        normalized_parameter_norm=parameter_norm,
        weighted_normalized_parameter_l2=weighted_parameter_norm,
        regularized_residual_norm=regularized_norm,
        nfev=int(nfev),
        success=bool(success),
        message=str(message),
        norm=norm_key,
        penalty_value=penalty_value,
        regularization_weight_min=float(np.min(active_weights)) if active_weights.size else 1.0,
        regularization_weight_max=float(np.max(active_weights)) if active_weights.size else 1.0,
        admm_iterations=int(admm_iterations),
        admm_primal_residual=float(admm_primal),
        admm_dual_residual=float(admm_dual),
        final_z=None if z_final is None else np.asarray(z_final, dtype=float).copy(),
        final_u=None if u_final is None else np.asarray(u_final, dtype=float).copy(),
    )


def evaluate_lambda_cv(
    model: MultiSourceRobotModel,
    joint_configs: np.ndarray,
    measured_positions: np.ndarray,
    parameters: list[ErrorParameter],
    active_indices: Iterable[int],
    folds: list[np.ndarray],
    lambdas: Iterable[float],
    payloads: np.ndarray | float | None = None,
    directions: np.ndarray | None = None,
    max_nfev: int = 120,
    norm: str = "l2",
    smooth_l1_beta: float = 1.0e-3,
    admm_rho: float = 0.0,
    admm_max_iter: int = 50,
    admm_tol_abs: float = 1.0e-6,
    admm_tol_rel: float = 1.0e-4,
    regularization_weights: np.ndarray | None = None,
) -> list[LambdaScore]:
    """Evaluate lambda values using provided validation folds."""
    joints = np.asarray(joint_configs, dtype=float).reshape(-1, 6)
    measured = np.asarray(measured_positions, dtype=float).reshape(-1, 3)
    count = len(joints)
    scores: list[LambdaScore] = []
    for lam in lambdas:
        fold_rmse: list[float] = []
        for fold in folds:
            validation_indices = np.asarray(fold, dtype=int).reshape(-1)
            if validation_indices.size == 0 or validation_indices.size >= count:
                continue
            mask = np.ones(count, dtype=bool)
            mask[validation_indices] = False
            train_indices = np.flatnonzero(mask)
            result = fit_l2_lm(
                model,
                joints[train_indices],
                measured[train_indices],
                parameters,
                active_indices,
                lambda_value=float(lam),
                payloads=subset_optional(payloads, train_indices, count),
                directions=subset_optional(directions, train_indices, count),
                max_nfev=max_nfev,
                norm=norm,
                smooth_l1_beta=smooth_l1_beta,
                admm_rho=admm_rho,
                admm_max_iter=admm_max_iter,
                admm_tol_abs=admm_tol_abs,
                admm_tol_rel=admm_tol_rel,
                regularization_weights=regularization_weights,
            )
            predicted = model.batch_positions(
                joints[validation_indices],
                result.vector,
                parameters,
                subset_optional(payloads, validation_indices, count),
                subset_optional(directions, validation_indices, count),
            )
            fold_rmse.append(euclidean_rmse(measured[validation_indices], predicted))
        scores.append(_lambda_score(float(lam), fold_rmse))
    return scores


def make_lambda_grid(
    min_power: float = -6.0,
    max_power: float = 2.0,
    count: int = 17,
) -> np.ndarray:
    """Return a log-spaced lambda grid."""
    if int(count) <= 1:
        return np.asarray([10.0 ** float(min_power)], dtype=float)
    return np.power(10.0, np.linspace(float(min_power), float(max_power), int(count)))


def refined_lambda_grid(
    center_lambda: float,
    radius_decades: float = 1.0,
    count: int = 11,
) -> np.ndarray:
    """Return a denser grid around a selected lambda."""
    center = max(float(center_lambda), 1.0e-300)
    center_power = float(np.log10(center))
    return make_lambda_grid(center_power - radius_decades, center_power + radius_decades, count)


def merge_lambdas(*grids: Iterable[float]) -> np.ndarray:
    """Merge lambda grids while keeping a stable ascending order."""
    values: list[float] = []
    for grid in grids:
        for value in grid:
            finite = float(value)
            if np.isfinite(finite) and finite >= 0.0:
                values.append(finite)
    if not values:
        return np.zeros(0, dtype=float)
    rounded = {round(float(value), 15): float(value) for value in values}
    return np.asarray(sorted(rounded.values()), dtype=float)


def select_lambda_score(scores: list[LambdaScore], criterion: str = "max") -> LambdaScore:
    """Select a lambda from CV scores by mean or worst-fold RMSE."""
    if not scores:
        raise ValueError("At least one lambda score is required.")
    if criterion == "mean":
        return min(scores, key=lambda item: (item.mean_rmse, item.max_rmse, item.lambda_value))
    if criterion == "max":
        return min(scores, key=lambda item: (item.max_rmse, item.mean_rmse, item.lambda_value))
    raise ValueError(f"Unknown selection criterion: {criterion}")


def random_folds(sample_count: int, fold_count: int = 4, seed: int = 123) -> list[np.ndarray]:
    """Build random validation folds."""
    count = int(sample_count)
    if count <= 1:
        return [np.arange(count, dtype=int)]
    folds = max(2, min(int(fold_count), count))
    rng = np.random.default_rng(seed)
    return [fold.astype(int) for fold in np.array_split(rng.permutation(count), folds)]


def spatial_folds(positions: np.ndarray, fold_count: int = 4) -> list[np.ndarray]:
    """Partition Cartesian positions along their longest axis."""
    points = np.asarray(positions, dtype=float).reshape(-1, 3)
    count = len(points)
    if count <= 1:
        return [np.arange(count, dtype=int)]
    folds = max(2, min(int(fold_count), count))
    axis = int(np.argmax(np.ptp(points, axis=0)))
    order = np.argsort(points[:, axis], kind="mergesort")
    return [np.asarray(chunk, dtype=int) for chunk in np.array_split(order, folds)]


def select_l_curve_corner(points: list[LCurvePoint]) -> LCurvePoint:
    """Select the L-curve corner by maximum distance from the endpoint line."""
    if not points:
        raise ValueError("At least one L-curve point is required.")
    ordered = sorted(points, key=lambda item: item.lambda_value)
    if len(ordered) <= 2:
        return ordered[len(ordered) // 2]

    x = np.log10(np.maximum([p.normalized_parameter_norm for p in ordered], 1.0e-300))
    y = np.log10(np.maximum([p.residual_norm for p in ordered], 1.0e-300))
    xy = np.column_stack([x, y]).astype(float)
    lower = np.min(xy, axis=0)
    span = np.maximum(np.ptp(xy, axis=0), 1.0e-12)
    scaled = (xy - lower) / span
    start = scaled[0]
    end = scaled[-1]
    line = end - start
    line_norm = float(np.linalg.norm(line))
    if line_norm <= 1.0e-12:
        return min(ordered, key=lambda item: item.train_rmse)
    distances = np.abs(np.cross(line, scaled - start)) / line_norm
    distances[0] = -1.0
    distances[-1] = -1.0
    return ordered[int(np.argmax(distances))]


def l_curve_points_from_results(results: Iterable[RegularizedLMResult]) -> list[LCurvePoint]:
    """Convert fit results into L-curve points."""
    points = []
    for result in results:
        points.append(
            LCurvePoint(
                lambda_value=float(result.lambda_value),
                residual_norm=float(result.position_residual_norm),
                normalized_parameter_norm=float(result.normalized_parameter_norm),
                train_rmse=float(result.component_rmse),
                nfev=int(result.nfev),
            )
        )
    return points


def euclidean_rmse(measured_positions: np.ndarray, predicted_positions: np.ndarray) -> float:
    """Return Euclidean TCP-position RMSE in meters."""
    measured = np.asarray(measured_positions, dtype=float).reshape(-1, 3)
    predicted = np.asarray(predicted_positions, dtype=float).reshape(-1, 3)
    errors = np.linalg.norm(predicted - measured, axis=1)
    return float(np.sqrt(np.mean(np.square(errors))))


def euclidean_max(measured_positions: np.ndarray, predicted_positions: np.ndarray) -> float:
    """Return maximum Euclidean TCP-position error in meters."""
    measured = np.asarray(measured_positions, dtype=float).reshape(-1, 3)
    predicted = np.asarray(predicted_positions, dtype=float).reshape(-1, 3)
    errors = np.linalg.norm(predicted - measured, axis=1)
    return float(np.max(errors))


def subset_optional(
    values: np.ndarray | float | None,
    indices: np.ndarray,
    sample_count: int,
) -> np.ndarray | float | None:
    """Subset per-sample payload or direction arrays while preserving scalars."""
    if values is None:
        return None
    array = np.asarray(values, dtype=float)
    if array.ndim == 0 or array.size == 1:
        return float(array.reshape(-1)[0])
    if array.shape[0] != int(sample_count):
        return values
    return array[np.asarray(indices, dtype=int)]


def _l2_residuals(
    active_values: np.ndarray,
    full_vector: np.ndarray,
    active_indices: list[int],
    model: MultiSourceRobotModel,
    joint_configs: np.ndarray,
    measured_positions: np.ndarray,
    parameters: list[ErrorParameter],
    payloads: np.ndarray | float | None,
    directions: np.ndarray | None,
    scales: np.ndarray,
    regularization_weights: np.ndarray,
    lambda_value: float,
    position_noise_std_m: float,
) -> np.ndarray:
    candidate = full_vector.copy()
    candidate[active_indices] = active_values
    position = _position_residuals(
        model, joint_configs, measured_positions, candidate, parameters, payloads, directions
    ) / _position_noise_std(position_noise_std_m)
    if lambda_value <= 0.0:
        return position
    active = np.asarray(active_indices, dtype=int)
    regularization = (
        np.sqrt(float(lambda_value))
        * np.sqrt(np.maximum(regularization_weights[active], 0.0))
        * candidate[active]
        / np.maximum(scales[active], 1.0e-20)
    )
    return np.concatenate([position, regularization])


def _smooth_l1_residuals(
    active_values: np.ndarray,
    full_vector: np.ndarray,
    active_indices: list[int],
    model: MultiSourceRobotModel,
    joint_configs: np.ndarray,
    measured_positions: np.ndarray,
    parameters: list[ErrorParameter],
    payloads: np.ndarray | float | None,
    directions: np.ndarray | None,
    scales: np.ndarray,
    lambda_value: float,
    beta: float,
) -> np.ndarray:
    """Exact Huber square-root residual.

    For ``|z| <= beta`` the residual is ``sqrt(lambda/beta) * z`` (so the
    squared cost equals ``lambda * z^2 / (2*beta)``). For ``|z| > beta`` it
    is ``sign(z) * sqrt(2*lambda*(|z|-beta/2))`` (squared cost equals
    ``lambda*(|z|-beta/2)``). The two pieces have matching value and
    derivative at ``|z| = beta``, so the residual is C^1 and ``least_squares``
    sees an exact representation of the Huber-penalized objective.
    """
    candidate = full_vector.copy()
    candidate[active_indices] = active_values
    position = _position_residuals(
        model, joint_configs, measured_positions, candidate, parameters, payloads, directions
    )
    if lambda_value <= 0.0:
        return position
    active = np.asarray(active_indices, dtype=int)
    z = candidate[active] / np.maximum(scales[active], 1.0e-20)
    abs_z = np.abs(z)
    quadratic = abs_z <= beta
    quad_part = np.sqrt(float(lambda_value) / max(beta, 1.0e-20)) * z
    linear_excess = np.maximum(abs_z - beta / 2.0, 0.0)
    linear_part = np.sign(z) * np.sqrt(2.0 * float(lambda_value) * linear_excess)
    regularization = np.where(quadratic, quad_part, linear_part)
    return np.concatenate([position, regularization])


def _admm_theta_residuals(
    active_values: np.ndarray,
    full_vector: np.ndarray,
    active_indices: list[int],
    model: MultiSourceRobotModel,
    joint_configs: np.ndarray,
    measured_positions: np.ndarray,
    parameters: list[ErrorParameter],
    payloads: np.ndarray | float | None,
    directions: np.ndarray | None,
    active_scales: np.ndarray,
    target_w: np.ndarray,
    rho: float,
) -> np.ndarray:
    """ADMM theta-step residual: position + sqrt(rho)*(theta/scale - target).

    scipy's ``least_squares`` minimizes ``0.5 * ||residual||^2``. This residual
    therefore adds ``(rho/2) * ||theta/scale - target||^2``, matching the
    standard scaled-ADMM theta subproblem and the ``lambda / rho`` soft
    threshold in the z-step.
    """
    candidate = full_vector.copy()
    candidate[active_indices] = active_values
    position = _position_residuals(
        model, joint_configs, measured_positions, candidate, parameters, payloads, directions
    )
    if rho <= 0.0:
        return position
    w = active_values / np.maximum(active_scales, 1.0e-20)
    penalty = np.sqrt(float(rho)) * (w - target_w)
    return np.concatenate([position, penalty])


def _position_residuals(
    model: MultiSourceRobotModel,
    joint_configs: np.ndarray,
    measured_positions: np.ndarray,
    vector: np.ndarray,
    parameters: list[ErrorParameter],
    payloads: np.ndarray | float | None,
    directions: np.ndarray | None,
) -> np.ndarray:
    predicted = model.batch_positions(joint_configs, vector, parameters, payloads, directions)
    return (predicted - measured_positions).reshape(-1)


def _initial_vector(initial_vector: np.ndarray | None, parameter_count: int) -> np.ndarray:
    if initial_vector is None:
        return np.zeros(parameter_count, dtype=float)
    return np.asarray(initial_vector, dtype=float).reshape(parameter_count).copy()


def _position_noise_std(position_noise_std_m: float | None) -> float:
    if position_noise_std_m is None:
        return 1.0
    value = float(position_noise_std_m)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("position_noise_std_m must be positive and finite")
    return value


def _empty_result(
    vector: np.ndarray,
    active_indices: list[int],
    lambda_value: float,
    position_residual: np.ndarray,
    norm: str = "l2",
) -> RegularizedLMResult:
    residual_norm = float(np.linalg.norm(position_residual))
    return RegularizedLMResult(
        vector=vector,
        active_indices=active_indices,
        lambda_value=float(lambda_value),
        cost=0.5 * residual_norm * residual_norm,
        component_rmse=_rmse(position_residual),
        position_residual_norm=residual_norm,
        normalized_parameter_norm=0.0,
        weighted_normalized_parameter_l2=0.0,
        regularized_residual_norm=residual_norm,
        nfev=0,
        success=True,
        message="no active parameters",
        norm=norm,
        penalty_value=0.0,
        regularization_weight_min=1.0,
        regularization_weight_max=1.0,
    )


def _lambda_score(lambda_value: float, fold_rmse: list[float]) -> LambdaScore:
    values = np.asarray(fold_rmse, dtype=float)
    if values.size == 0:
        return LambdaScore(lambda_value, [], float("inf"), float("inf"), float("inf"))
    return LambdaScore(
        lambda_value=float(lambda_value),
        fold_rmse=[float(value) for value in values],
        mean_rmse=float(np.mean(values)),
        max_rmse=float(np.max(values)),
        std_rmse=float(np.std(values)),
    )


def _rmse(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=float).reshape(-1)
    return float(np.sqrt(np.mean(np.square(array)))) if array.size else 0.0


def _unique(values: list[int]) -> list[int]:
    seen: set[int] = set()
    output: list[int] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def _regularization_weights(
    regularization_weights: np.ndarray | None,
    parameter_count: int,
) -> np.ndarray:
    if regularization_weights is None:
        return np.ones(int(parameter_count), dtype=float)
    weights = np.asarray(regularization_weights, dtype=float).reshape(-1)
    if weights.size != int(parameter_count):
        raise ValueError(
            f"regularization_weights has length {weights.size}, expected {parameter_count}"
        )
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("regularization_weights must be finite and non-negative")
    return weights.copy()


