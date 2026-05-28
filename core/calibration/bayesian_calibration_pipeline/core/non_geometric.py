"""Geometry anchor and Bayesian residual helpers for the main calibration line.

The retained code supports the current mainline only: Stage 1 geometry
baselines, Bayesian-basis residual fitting support, full-parameter MAP
fine-tuning, split metrics, and residual diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats
from scipy.optimize import least_squares

from core.calibration.bayesian_calibration_pipeline.configs.pipeline_config import (
    METHOD_LABELS,
    Geometry33PipelineConfig,
)
from core.calibration.bayesian_calibration_pipeline.core.data_split import (
    canonical_dataset,
    concat_c_dataset,
    head_dataset,
    split_dataset_for_c,
)
from core.calibration.bayesian_calibration_pipeline.core.dynamic_identifiability import (
    build_identifiability_subspace_partition,
    compute_pose_identifiability_metrics,
    fit_dynamic_identifiability_l2,
    fit_subspace_sequential_l2,
)
from core.calibration.bayesian_calibration_pipeline.core.geometric import select_geometric_parameters
from core.calibration.bayesian_calibration_pipeline.core.identifiability import (
    compute_identifiability_metrics,
    strategy_weights,
)
from core.calibration.bayesian_calibration_pipeline.core.parameters import (
    ErrorParameter,
    build_error_parameters,
    parameter_scales,
    vector_to_components,
    zero_error_vector,
)
from core.calibration.bayesian_calibration_pipeline.core.pipeline import (
    dataset_errors,
    method_metrics,
    select_ab_balance_result,
)
from core.calibration.bayesian_calibration_pipeline.core.redundancy import output_jacobian
from core.calibration.bayesian_calibration_pipeline.core.regularization import (
    RegularizedLMResult,
    euclidean_rmse,
    fit_l2_lm,
    make_lambda_grid,
    select_independent_parameters,
)
from core.calibration.bayesian_calibration_pipeline.core.robot_model import MultiSourceRobotModel
from core.calibration.bayesian_calibration_pipeline.core.statistical_residual import (
    DatasetView,
    StatisticalModelResult,
    StatisticalResidualConfig,
)


GEOMETRY_METHODS = ("M0", "M6", "W3", "S1", "D1")
REQUIRED_NONGEOMETRIC_FIELDS = (
    "directions",
    "joint_torques",
    "self_weight_joint_torques",
    "payload_joint_torques",
)


@dataclass(frozen=True)
class NonGeometricConfig:
    """Settings for Stage 2/3 non-geometric identification."""

    alpha: float = 0.01
    eta_direct: float = 0.30
    eta_project: float = 0.10
    harmonics: tuple[int, ...] = (1, 2, 3, 4)
    joint_fd_step: float = 1.0e-6
    fwl_rcond: float = 1.0e-10
    max_selection_steps: int = 8
    max_a_degradation_fraction: float = 0.25
    nonlinear_max_nfev: int = 80
    whiteness_lags: tuple[int, ...] = (1, 2, 3, 5, 10)
    spectrum_permutations: int = 100
    seed: int = 20260524
    selection_objective: str = "train"

    def quickened(self) -> "NonGeometricConfig":
        """Return a smoke-test sized version."""
        return NonGeometricConfig(
            alpha=self.alpha,
            eta_direct=self.eta_direct,
            eta_project=self.eta_project,
            harmonics=tuple(h for h in self.harmonics if h <= 2),
            joint_fd_step=self.joint_fd_step,
            fwl_rcond=self.fwl_rcond,
            max_selection_steps=min(self.max_selection_steps, 4),
            max_a_degradation_fraction=self.max_a_degradation_fraction,
            nonlinear_max_nfev=min(self.nonlinear_max_nfev, 30),
            whiteness_lags=self.whiteness_lags,
            spectrum_permutations=min(self.spectrum_permutations, 20),
            seed=self.seed,
            selection_objective=self.selection_objective,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "alpha": float(self.alpha),
            "eta_direct": float(self.eta_direct),
            "eta_project": float(self.eta_project),
            "harmonics": list(self.harmonics),
            "joint_fd_step": float(self.joint_fd_step),
            "fwl_rcond": float(self.fwl_rcond),
            "max_selection_steps": int(self.max_selection_steps),
            "max_a_degradation_fraction": float(self.max_a_degradation_fraction),
            "nonlinear_max_nfev": int(self.nonlinear_max_nfev),
            "whiteness_lags": list(self.whiteness_lags),
            "spectrum_permutations": int(self.spectrum_permutations),
            "seed": int(self.seed),
            "selection_objective": str(self.selection_objective),
        }


@dataclass
class Stage1Result:
    """Stage 1 geometry baseline artifacts."""

    model: MultiSourceRobotModel
    parameters: list[ErrorParameter]
    zero_vector: np.ndarray
    redundancy: Any
    results: dict[str, RegularizedLMResult]
    curves: dict[str, list[dict[str, float]]]
    rows: list[dict[str, Any]]
    selected_method: str
    selected_vector: np.ndarray
    selected_active: list[int]


@dataclass
class StatisticalFineTuneResult:
    """MAP fine-tune result for fixed statistical residual features."""

    method: str
    accepted: bool
    reason: str
    vector: np.ndarray
    coefficients: np.ndarray
    metrics: dict[str, float]
    objective_initial: float
    objective_final: float
    nfev: int
    success: bool
    message: str
    theta_update_scaled_l2: float
    beta_update_l2_mm: float
    beta_l2_mm: float
    initial_beta_l2_mm: float
    beta_prior_residual_scale: float
    data_residual_scale_m: float
    objective_terms_initial: dict[str, float]
    objective_terms_final: dict[str, float]


def verify_nongeometric_dataset(dataset: dict[str, Any], name: str) -> dict[str, Any]:
    """Validate required non-geometric fields and return a JSON-friendly row."""
    canonical = canonical_dataset(dataset)
    count = len(canonical["joints"])
    row: dict[str, Any] = {
        "name": name,
        "samples": int(count),
        "joints_shape": list(canonical["joints"].shape),
        "measured_positions_shape": list(canonical["measured_positions"].shape),
        "payloads_shape": list(np.asarray(canonical["payloads"]).shape),
        "field_checks": {},
    }
    for key in REQUIRED_NONGEOMETRIC_FIELDS:
        if key not in canonical:
            raise KeyError(f"{name} missing required non-geometric field: {key}")
        values = np.asarray(canonical[key], dtype=float)
        expected = (count,) if key == "payloads" else (count, 6)
        if key != "payloads" and values.shape != expected:
            raise ValueError(f"{name}.{key} expected {expected}, got {values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name}.{key} contains NaN or Inf")
        row["field_checks"][key] = {
            "shape": list(values.shape),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "rms": float(np.sqrt(np.mean(values * values))),
        }
    return row


def fit_stage1_geometry_baselines(
    train: dict[str, np.ndarray],
    selection: dict[str, np.ndarray],
    config: Geometry33PipelineConfig,
    *,
    position_noise_std_m: float | None = None,
    anchor_method: str | None = "S1",
) -> Stage1Result:
    """Run M0/M6/W3/S1/D1 and select the geometry anchor.

    The Bayesian mainline treats S1 as the first-stage anchor by design.  The
    B-space best row is still reported for diagnostics, but changing residual
    scaling or lambda grids should not silently switch the anchor semantics.
    """
    cfg = config.quickened()
    model = MultiSourceRobotModel()
    full_parameters = build_error_parameters()
    parameters = select_geometric_parameters(full_parameters)
    zero = zero_error_vector(parameters)
    redundancy = select_independent_parameters(
        model,
        train["joints"],
        parameters,
        payloads=train.get("payloads"),
        directions=train.get("directions"),
        tolerance=cfg.redundancy_tolerance,
        max_combinations=cfg.redundancy_max_combinations,
        jacobian_method=cfg.jacobian_method,
    )
    active = list(redundancy.independent_indices)
    all33 = list(range(len(parameters)))

    global_metrics = compute_identifiability_metrics(
        redundancy.jacobian,
        parameters,
        tolerance=cfg.redundancy_tolerance,
        rank=int(redundancy.rank),
        rho_threshold=cfg.rho_threshold,
        kappa_threshold=cfg.kappa_threshold,
        risk_beta=cfg.risk_beta,
        risk_power=cfg.risk_power,
        min_weight=cfg.min_weight,
        max_weight=cfg.max_weight,
        scaled_jacobian=True,
    )
    w3_weights = strategy_weights(
        global_metrics,
        active_indices=all33,
        strong=True,
        strong_weight=cfg.strong_weight,
    )
    pose_metrics = compute_pose_identifiability_metrics(
        model,
        train["joints"],
        parameters,
        zero,
        payloads=train.get("payloads"),
        directions=train.get("directions"),
        tolerance=cfg.redundancy_tolerance,
        rho_threshold=cfg.rho_threshold,
        kappa_threshold=cfg.kappa_threshold,
        risk_beta=cfg.risk_beta,
        risk_power=cfg.risk_power,
        min_weight=cfg.min_weight,
        max_weight=cfg.max_weight,
        risk_quantile=cfg.dynamic_risk_quantile,
        jacobian_method=cfg.jacobian_method,
    )
    train_jacobian = output_jacobian(
        model,
        train["joints"],
        zero,
        parameters,
        payloads=train.get("payloads"),
        directions=train.get("directions"),
        method=cfg.jacobian_method,
    )
    subspace_partition = build_identifiability_subspace_partition(
        pose_metrics,
        train_jacobian,
        parameters,
        k_candidates=cfg.subspace_k_candidates,
        min_cluster_size=cfg.subspace_min_cluster_size,
        seed=cfg.seed + 509,
        tolerance=cfg.redundancy_tolerance,
        rho_threshold=cfg.rho_threshold,
        kappa_threshold=cfg.kappa_threshold,
        risk_beta=cfg.risk_beta,
        risk_power=cfg.risk_power,
        min_weight=cfg.min_weight,
        max_weight=cfg.max_weight,
        strong_weight=cfg.strong_weight,
    )
    lambda_grid = make_lambda_grid(
        cfg.lambda_min_power,
        cfg.lambda_max_power,
        cfg.lambda_count,
    )

    results: dict[str, RegularizedLMResult] = {}
    curves: dict[str, list[dict[str, float]]] = {}

    results["M0"] = fit_l2_lm(
        model,
        train["joints"],
        train["measured_positions"],
        parameters,
        active,
        lambda_value=0.0,
        payloads=train.get("payloads"),
        directions=train.get("directions"),
        max_nfev=cfg.max_nfev,
        position_noise_std_m=position_noise_std_m,
    )
    results["M6"], curves["M6"] = select_ab_balance_result(
        model,
        train,
        selection,
        selection,
        parameters,
        lambda_grid,
        cfg,
        lambda lam: fit_l2_lm(
            model,
            train["joints"],
            train["measured_positions"],
            parameters,
            active,
            lambda_value=float(lam),
            payloads=train.get("payloads"),
            directions=train.get("directions"),
            max_nfev=cfg.max_nfev,
            position_noise_std_m=position_noise_std_m,
        ),
    )
    results["W3"], curves["W3"] = select_ab_balance_result(
        model,
        train,
        selection,
        selection,
        parameters,
        lambda_grid,
        cfg,
        lambda lam: fit_l2_lm(
            model,
            train["joints"],
            train["measured_positions"],
            parameters,
            all33,
            lambda_value=float(lam),
            payloads=train.get("payloads"),
            directions=train.get("directions"),
            max_nfev=cfg.max_nfev,
            regularization_weights=w3_weights,
            position_noise_std_m=position_noise_std_m,
        ),
    )
    results["S1"], curves["S1"] = select_ab_balance_result(
        model,
        train,
        selection,
        selection,
        parameters,
        lambda_grid,
        cfg,
        lambda lam: fit_subspace_sequential_l2(
            model,
            train["joints"],
            train["measured_positions"],
            parameters,
            subspace_partition,
            lambda_value=float(lam),
            active_indices=all33,
            payloads=train.get("payloads"),
            directions=train.get("directions"),
            max_nfev=cfg.max_nfev,
            position_noise_std_m=position_noise_std_m,
        ).result,
    )
    results["D1"], curves["D1"] = select_ab_balance_result(
        model,
        train,
        selection,
        selection,
        parameters,
        lambda_grid,
        cfg,
        lambda lam: fit_dynamic_identifiability_l2(
            model,
            train["joints"],
            train["measured_positions"],
            parameters,
            active,
            lambda_value=float(lam),
            payloads=train.get("payloads"),
            directions=train.get("directions"),
            max_nfev=cfg.max_nfev,
            outer_iterations=cfg.dynamic_outer_iterations,
            convergence_tol=cfg.dynamic_convergence_tol,
            tolerance=cfg.redundancy_tolerance,
            rho_threshold=cfg.rho_threshold,
            kappa_threshold=cfg.kappa_threshold,
            risk_beta=cfg.risk_beta,
            risk_power=cfg.risk_power,
            min_weight=cfg.min_weight,
            max_weight=cfg.max_weight,
            risk_quantile=cfg.dynamic_risk_quantile,
            jacobian_method=cfg.jacobian_method,
            position_noise_std_m=position_noise_std_m,
        ).result,
    )

    rows = [
        method_metrics(
            method,
            results[method],
            model,
            parameters,
            train,
            selection,
            selection,
        )
        for method in GEOMETRY_METHODS
    ]
    if anchor_method is not None and str(anchor_method) in results:
        selected_method = str(anchor_method)
    else:
        selected_row = min(rows, key=lambda row: (row["selection_B_rmse_mm"], row["method"]))
        selected_method = str(selected_row["method"])
    selected = results[selected_method]
    return Stage1Result(
        model=model,
        parameters=parameters,
        zero_vector=zero,
        redundancy=redundancy,
        results=results,
        curves=curves,
        rows=rows,
        selected_method=selected_method,
        selected_vector=selected.vector.copy(),
        selected_active=list(selected.active_indices),
    )




















def fit_statistical_global_fine_tune(
    model: MultiSourceRobotModel,
    parameters: list[ErrorParameter],
    train_view: DatasetView,
    eval_views: dict[str, DatasetView],
    anchor_vector: np.ndarray,
    active_indices: list[int],
    statistical_model: StatisticalModelResult,
    config: StatisticalResidualConfig,
    *,
    max_nfev: int,
    geometry_prior_lambda: float = 1.0,
    scale_position_residual_by_noise: bool = True,
) -> StatisticalFineTuneResult:
    """Fine-tune geometry and fixed statistical coefficients with a MAP objective."""
    active = [int(index) for index in active_indices]
    design_train = np.asarray(statistical_model.design_train, dtype=float)
    feature_count = int(design_train.shape[1])
    coef0 = np.asarray(statistical_model.coefficients, dtype=float).reshape(feature_count, 3)
    theta0 = np.asarray(anchor_vector, dtype=float).copy()
    x0 = np.concatenate([theta0[active], coef0.reshape(-1)])
    scales = parameter_scales(parameters)[active]
    beta_scales = np.repeat(
        np.maximum(np.asarray(statistical_model.prior_std_by_column, dtype=float), 1.0e-12),
        3,
    )
    x_scale = np.maximum(np.concatenate([scales, beta_scales]), 1.0e-12)
    tracker_noise_std = max(float(config.noise_std_m), 1.0e-12)
    noise_std = tracker_noise_std if scale_position_residual_by_noise else 1.0
    beta_prior_scale = 1.0 if scale_position_residual_by_noise else tracker_noise_std
    prior_std = np.maximum(
        np.asarray(statistical_model.prior_std_by_column, dtype=float).reshape(-1),
        1.0e-12,
    )

    def unpack(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        vector = np.asarray(anchor_vector, dtype=float).copy()
        vector[active] = np.asarray(x[: len(active)], dtype=float)
        beta = np.asarray(x[len(active) :], dtype=float).reshape(feature_count, 3)
        return vector, beta

    def residuals(x: np.ndarray) -> np.ndarray:
        vector, beta = unpack(x)
        geom = model.batch_positions(
            train_view.dataset["joints"],
            vector,
            parameters,
            train_view.dataset.get("payloads"),
            train_view.dataset.get("directions"),
        )
        correction = design_train @ beta if feature_count else np.zeros_like(geom)
        data = _flat((geom + correction - train_view.measured_positions) / noise_std)
        geom_prior = (
            np.sqrt(float(geometry_prior_lambda))
            * (vector[active] - np.asarray(anchor_vector, dtype=float)[active])
            / np.maximum(scales, 1.0e-12)
        )
        beta_prior = beta_prior_scale * (beta / prior_std.reshape(-1, 1)).reshape(-1)
        return np.concatenate([data, geom_prior, beta_prior])

    def objective_terms(x: np.ndarray) -> dict[str, float]:
        values = residuals(x)
        data_size = int(np.asarray(train_view.measured_positions).size)
        geom_size = int(len(active))
        data = values[:data_size]
        geom = values[data_size : data_size + geom_size]
        beta_prior = values[data_size + geom_size :]
        return {
            "data_term": float(np.dot(data, data)),
            "geometry_prior_term": float(np.dot(geom, geom)),
            "beta_prior_term": float(np.dot(beta_prior, beta_prior)),
            "total": float(np.dot(values, values)),
        }

    initial_residual = residuals(x0)
    method = "lm" if initial_residual.size >= x0.size else "trf"
    result = least_squares(
        residuals,
        x0=x0,
        method=method,
        x_scale=x_scale,
        max_nfev=int(max_nfev),
        ftol=1.0e-10,
        xtol=1.0e-10,
        gtol=1.0e-10,
    )
    vector, beta = unpack(result.x)
    final_residual = residuals(result.x)
    metrics = statistical_position_metrics(
        model,
        parameters,
        train_view,
        eval_views,
        vector,
        beta,
        design_train,
        statistical_model.design_eval,
    )
    pre_c = float(statistical_model.metrics.get("C_all_rmse_mm", float("inf")))
    post_c = float(metrics.get("C_all_rmse_mm", float("inf")))
    accepted = bool(post_c <= pre_c + 1.0e-9)
    reason = "accepted_C_all_not_worse" if accepted else "rejected_C_all_worse"
    beta_delta = beta - coef0
    return StatisticalFineTuneResult(
        method=statistical_model.method,
        accepted=accepted,
        reason=reason,
        vector=vector,
        coefficients=beta,
        metrics=metrics,
        objective_initial=float(np.dot(initial_residual, initial_residual)),
        objective_final=float(np.dot(final_residual, final_residual)),
        nfev=int(result.nfev),
        success=bool(result.success),
        message=str(result.message),
        theta_update_scaled_l2=float(
            np.linalg.norm(
                (vector[active] - np.asarray(anchor_vector, dtype=float)[active])
                / np.maximum(scales, 1.0e-12)
            )
        ),
        beta_update_l2_mm=float(np.linalg.norm(beta_delta) * 1000.0),
        beta_l2_mm=float(np.linalg.norm(beta) * 1000.0),
        initial_beta_l2_mm=float(np.linalg.norm(coef0) * 1000.0),
        beta_prior_residual_scale=float(beta_prior_scale),
        data_residual_scale_m=float(noise_std),
        objective_terms_initial=objective_terms(x0),
        objective_terms_final=objective_terms(result.x),
    )


def statistical_position_metrics(
    model: MultiSourceRobotModel,
    parameters: list[ErrorParameter],
    train_view: DatasetView,
    eval_views: dict[str, DatasetView],
    vector: np.ndarray,
    coefficients: np.ndarray,
    design_train: np.ndarray,
    design_eval: dict[str, np.ndarray],
) -> dict[str, float]:
    """Return position RMSE metrics for p(theta)+X beta on train/eval views."""
    train_pred = statistical_position_prediction(
        model, parameters, train_view, vector, coefficients, design_train
    )
    metrics = position_metric_row(train_view.measured_positions, train_pred, "train")
    c_errors = []
    for name, view in eval_views.items():
        design = np.asarray(design_eval.get(name, np.zeros((len(view.residuals), 0))), dtype=float)
        pred = statistical_position_prediction(model, parameters, view, vector, coefficients, design)
        metrics.update(position_metric_row(view.measured_positions, pred, name))
        if name.endswith("_C"):
            c_errors.append(pred - view.measured_positions)
    if c_errors:
        err = np.vstack(c_errors)
        metrics["C_all_rmse_mm"] = float(
            np.sqrt(np.mean(np.sum(err * err, axis=1))) * 1000.0
        )
    return metrics


def statistical_position_prediction(
    model: MultiSourceRobotModel,
    parameters: list[ErrorParameter],
    view: DatasetView,
    vector: np.ndarray,
    coefficients: np.ndarray,
    design: np.ndarray,
) -> np.ndarray:
    geom = model.batch_positions(
        view.dataset["joints"],
        vector,
        parameters,
        view.dataset.get("payloads"),
        view.dataset.get("directions"),
    )
    beta = np.asarray(coefficients, dtype=float).reshape(design.shape[1], 3)
    correction = design @ beta if design.shape[1] else np.zeros_like(geom)
    return geom + correction


def position_metric_row(target: np.ndarray, predicted: np.ndarray, prefix: str) -> dict[str, float]:
    target_arr = np.asarray(target, dtype=float).reshape(-1, 3)
    pred_arr = np.asarray(predicted, dtype=float).reshape(-1, 3)
    err = pred_arr - target_arr
    return {
        f"{prefix}_rmse_mm": float(np.sqrt(np.mean(np.sum(err * err, axis=1))) * 1000.0),
        f"{prefix}_component_rmse_mm": float(np.sqrt(np.mean(err * err)) * 1000.0),
        f"{prefix}_max_mm": float(np.max(np.linalg.norm(err, axis=1)) * 1000.0),
    }


def fine_tune_to_report(result: StatisticalFineTuneResult, pre_metrics: dict[str, float]) -> dict[str, Any]:
    return {
        "method": result.method,
        "accepted": bool(result.accepted),
        "reason": result.reason,
        "pre_train_rmse_mm": float(pre_metrics.get("train_rmse_mm", float("nan"))),
        "post_train_rmse_mm": float(result.metrics.get("train_rmse_mm", float("nan"))),
        "pre_A_train_rmse_mm": float(pre_metrics.get("A_train_rmse_mm", float("nan"))),
        "post_A_train_rmse_mm": float(result.metrics.get("A_train_rmse_mm", float("nan"))),
        "pre_B_train_rmse_mm": float(pre_metrics.get("B_train_rmse_mm", float("nan"))),
        "post_B_train_rmse_mm": float(result.metrics.get("B_train_rmse_mm", float("nan"))),
        "pre_A_C_rmse_mm": float(pre_metrics.get("A_C_rmse_mm", float("nan"))),
        "post_A_C_rmse_mm": float(result.metrics.get("A_C_rmse_mm", float("nan"))),
        "pre_B_C_rmse_mm": float(pre_metrics.get("B_C_rmse_mm", float("nan"))),
        "post_B_C_rmse_mm": float(result.metrics.get("B_C_rmse_mm", float("nan"))),
        "pre_C_all_rmse_mm": float(pre_metrics.get("C_all_rmse_mm", float("nan"))),
        "post_C_all_rmse_mm": float(result.metrics.get("C_all_rmse_mm", float("nan"))),
        "objective_initial": float(result.objective_initial),
        "objective_final": float(result.objective_final),
        "nfev": int(result.nfev),
        "success": bool(result.success),
        "message": result.message,
        "theta_update_scaled_l2": float(result.theta_update_scaled_l2),
        "beta_update_l2_mm": float(result.beta_update_l2_mm),
        "initial_beta_l2_mm": float(result.initial_beta_l2_mm),
        "beta_l2_mm": float(result.beta_l2_mm),
        "beta_prior_residual_scale": float(result.beta_prior_residual_scale),
        "data_residual_scale_m": float(result.data_residual_scale_m),
        "objective_data_initial": float(result.objective_terms_initial.get("data_term", float("nan"))),
        "objective_geometry_prior_initial": float(result.objective_terms_initial.get("geometry_prior_term", float("nan"))),
        "objective_beta_prior_initial": float(result.objective_terms_initial.get("beta_prior_term", float("nan"))),
        "objective_data_final": float(result.objective_terms_final.get("data_term", float("nan"))),
        "objective_geometry_prior_final": float(result.objective_terms_final.get("geometry_prior_term", float("nan"))),
        "objective_beta_prior_final": float(result.objective_terms_final.get("beta_prior_term", float("nan"))),
    }


def stage1_split_rows(
    stage1: Stage1Result,
    datasets: dict[str, dict[str, np.ndarray]],
) -> list[dict[str, Any]]:
    """Evaluate every Stage 1 geometry method on A/B train and held-out C splits."""
    base_rows = {str(row["method"]): row for row in stage1.rows}
    rows: list[dict[str, Any]] = []
    c_all = concat_c_dataset([datasets["A_C"], datasets["B_C"]], label="C_all")
    for method, result in stage1.results.items():
        row: dict[str, Any] = {
            "method": method,
            "label": METHOD_LABELS.get(method, method),
            "lambda": float(result.lambda_value),
            "active_count": int(len(result.active_indices)),
        }
        if method in base_rows:
            row["selection_rule"] = base_rows[method].get("selection_rule", "")
        for prefix, dataset in (
            ("A_train", datasets["A_train"]),
            ("B_train", datasets["B_train"]),
            ("A_C", datasets["A_C"]),
            ("B_C", datasets["B_C"]),
            ("C_all", c_all),
        ):
            rmse, max_err = dataset_errors(
                stage1.model, stage1.parameters, dataset, result.vector
            )
            row[f"{prefix}_rmse_mm"] = float(rmse * 1000.0)
            row[f"{prefix}_max_mm"] = float(max_err * 1000.0)
        rows.append(row)
    return rows


def statistical_whiteness_report(
    model: MultiSourceRobotModel,
    parameters: list[ErrorParameter],
    vector: np.ndarray,
    coefficients: np.ndarray,
    design_train: np.ndarray,
    design_eval: dict[str, np.ndarray],
    train_view: DatasetView,
    eval_views: dict[str, DatasetView],
    config: NonGeometricConfig,
) -> dict[str, Any]:
    """Run whiteness diagnostics for the selected statistical path."""
    all_views: dict[str, DatasetView] = {
        "A_B_train": train_view,
        **eval_views,
    }
    outputs: dict[str, Any] = {}
    residuals = []
    datasets = []
    for name, view in all_views.items():
        design = design_train if name == "A_B_train" else design_eval.get(name)
        if design is None:
            continue
        pred = statistical_position_prediction(model, parameters, view, vector, coefficients, design)
        residual = pred - view.measured_positions
        outputs[name] = run_whiteness_suite(residual, view.dataset, config, label=name)
        if name in ("A_train", "B_train", "A_C", "B_C"):
            residuals.append(residual)
            datasets.append(view.dataset)
    if residuals:
        merged = concat_c_dataset(datasets, label="A_B_C_all")
        outputs["A_B_C_all"] = run_whiteness_suite(
            np.concatenate(residuals, axis=0),
            merged,
            config,
            label="A+B+C",
        )
    return outputs






def run_whiteness_suite(
    residuals: np.ndarray,
    dataset: dict[str, np.ndarray],
    config: NonGeometricConfig,
    *,
    label: str,
) -> dict[str, Any]:
    """Run residual whiteness diagnostics used by the report."""
    residual = np.asarray(residuals, dtype=float).reshape(-1, 3)
    norm = np.linalg.norm(residual, axis=1)
    series = {
        "x": residual[:, 0],
        "y": residual[:, 1],
        "z": residual[:, 2],
        "norm": norm,
    }
    ljung_rows = []
    ljung_p = []
    for name, values in series.items():
        for lag in config.whiteness_lags:
            if int(lag) >= len(values):
                continue
            row = ljung_box(values, int(lag))
            row["series"] = name
            ljung_rows.append(row)
            ljung_p.append(row["p_value"])
    ljung_adjusted = bh_adjust(ljung_p)
    for row, adj in zip(ljung_rows, ljung_adjusted):
        row["p_adjusted"] = float(adj)

    spectral_rows, spectral_p = spectral_peak_tests(
        series,
        dataset["joints"],
        permutations=int(config.spectrum_permutations),
        seed=int(config.seed) + 17,
    )
    spectral_adjusted = bh_adjust(spectral_p)
    for row, adj in zip(spectral_rows, spectral_adjusted):
        row["p_adjusted"] = float(adj)

    corr_rows, corr_p = partial_correlation_tests(series, dataset)
    corr_adjusted = bh_adjust(corr_p)
    for row, adj in zip(corr_rows, corr_adjusted):
        row["p_adjusted"] = float(adj)

    ljung_pass = bool(not ljung_rows or min(row["p_adjusted"] for row in ljung_rows) > 0.05)
    spectral_pass = bool(
        not spectral_rows or min(row["p_adjusted"] for row in spectral_rows) > 0.05
    )
    corr_pass = bool(not corr_rows or min(row["p_adjusted"] for row in corr_rows) > 0.05)
    return {
        "label": label,
        "passed": bool(ljung_pass and spectral_pass and corr_pass),
        "ljung_box_passed": ljung_pass,
        "spectral_passed": spectral_pass,
        "partial_correlation_passed": corr_pass,
        "ljung_box_min_adjusted_p": _min_or_none([row["p_adjusted"] for row in ljung_rows]),
        "spectral_min_adjusted_p": _min_or_none([row["p_adjusted"] for row in spectral_rows]),
        "partial_correlation_min_adjusted_p": _min_or_none(
            [row["p_adjusted"] for row in corr_rows]
        ),
        "ljung_box": ljung_rows,
        "spectral": spectral_rows,
        "partial_correlation": corr_rows,
    }


def ljung_box(values: np.ndarray, lag: int) -> dict[str, float | int]:
    """Small Ljung-Box implementation to avoid a statsmodels dependency."""
    x = np.asarray(values, dtype=float).reshape(-1)
    x = x - float(np.mean(x))
    n = len(x)
    denom = float(np.dot(x, x))
    if denom <= 1.0e-30:
        return {"lag": int(lag), "statistic": 0.0, "p_value": 1.0}
    autocorr = []
    for k in range(1, int(lag) + 1):
        autocorr.append(float(np.dot(x[k:], x[:-k]) / denom))
    q = n * (n + 2.0) * sum((rho * rho) / max(n - k, 1) for k, rho in enumerate(autocorr, 1))
    return {"lag": int(lag), "statistic": float(q), "p_value": float(stats.chi2.sf(q, lag))}


def spectral_peak_tests(
    series: dict[str, np.ndarray],
    joints: np.ndarray,
    *,
    permutations: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[float]]:
    """Permutation test for the largest joint-sorted residual spectral peak."""
    rng = np.random.default_rng(int(seed))
    rows: list[dict[str, Any]] = []
    p_values: list[float] = []
    q = np.asarray(joints, dtype=float).reshape(-1, 6)
    for joint in range(6):
        order = np.argsort(q[:, joint])
        for name, values in series.items():
            sorted_values = np.asarray(values, dtype=float).reshape(-1)[order]
            observed = spectral_peak_ratio(sorted_values)
            greater = 1
            for _ in range(max(int(permutations), 1)):
                permuted = rng.permutation(sorted_values)
                if spectral_peak_ratio(permuted) >= observed:
                    greater += 1
            p_value = greater / float(max(int(permutations), 1) + 1)
            rows.append(
                {
                    "joint": int(joint + 1),
                    "series": name,
                    "peak_ratio": float(observed),
                    "p_value": float(p_value),
                }
            )
            p_values.append(float(p_value))
    return rows, p_values


def spectral_peak_ratio(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=float).reshape(-1)
    x = x - float(np.mean(x))
    if len(x) < 4:
        return 0.0
    power = np.abs(np.fft.rfft(x)) ** 2
    if len(power) <= 1:
        return 0.0
    power = power[1:]
    total = float(np.sum(power))
    if total <= 1.0e-30:
        return 0.0
    return float(np.max(power) / total)


def partial_correlation_tests(
    series: dict[str, np.ndarray],
    dataset: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], list[float]]:
    """Partial-correlation tests against q, directions and joint torques."""
    features, names = residual_feature_matrix(dataset)
    rows: list[dict[str, Any]] = []
    p_values: list[float] = []
    for series_name, values in series.items():
        y = np.asarray(values, dtype=float).reshape(-1)
        for feature_index, feature_name in enumerate(names):
            x = features[:, feature_index]
            controls = np.delete(features, feature_index, axis=1)
            x_res = residualize(x, controls)
            y_res = residualize(y, controls)
            if np.std(x_res) <= 1.0e-15 or np.std(y_res) <= 1.0e-15:
                corr, p_value = 0.0, 1.0
            else:
                corr, p_value = stats.pearsonr(x_res, y_res)
            rows.append(
                {
                    "series": series_name,
                    "feature": feature_name,
                    "r": float(corr),
                    "p_value": float(p_value),
                }
            )
            p_values.append(float(p_value))
    return rows, p_values


def residual_feature_matrix(dataset: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    q = np.asarray(dataset["joints"], dtype=float).reshape(-1, 6)
    directions = _directions(dataset)
    if "joint_torques" in dataset:
        torques = np.asarray(dataset["joint_torques"], dtype=float).reshape(-1, 6)
    else:
        torques = np.zeros_like(q)
    matrices = [q, directions, torques]
    prefixes = ("q", "direction", "tau")
    names = [
        f"{prefix}_j{joint + 1}"
        for prefix in prefixes
        for joint in range(6)
    ]
    features = np.column_stack(matrices)
    return features, names


def residualize(values: np.ndarray, controls: np.ndarray) -> np.ndarray:
    y = np.asarray(values, dtype=float).reshape(-1)
    x = np.asarray(controls, dtype=float)
    design = np.column_stack([np.ones(len(y)), x])
    coef = _lstsq(design, y, 1.0e-10)
    return y - design @ coef


def bh_adjust(p_values: list[float] | np.ndarray) -> list[float]:
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p_values, dtype=float).reshape(-1)
    if p.size == 0:
        return []
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.empty_like(ranked)
    running = 1.0
    m = float(len(p))
    for i in range(len(ranked) - 1, -1, -1):
        running = min(running, ranked[i] * m / float(i + 1))
        adjusted[i] = running
    out = np.empty_like(adjusted)
    out[order] = np.clip(adjusted, 0.0, 1.0)
    return [float(value) for value in out]








































def _flat(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)


def _lstsq(a: np.ndarray, b: np.ndarray, rcond: float) -> np.ndarray:
    return np.linalg.lstsq(np.asarray(a, dtype=float), np.asarray(b, dtype=float), rcond=rcond)[0]


def _directions(dataset: dict[str, np.ndarray]) -> np.ndarray:
    if "directions" not in dataset:
        joints = np.asarray(dataset["joints"], dtype=float).reshape(-1, 6)
        values = np.sign(joints)
        values[values == 0.0] = 1.0
        return values
    values = np.sign(np.asarray(dataset["directions"], dtype=float).reshape(-1, 6))
    values[values == 0.0] = 1.0
    return values




def _min_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(min(values))


def json_ready(value: Any) -> Any:
    """Convert numpy-heavy objects to JSON-safe values."""
    if isinstance(value, dict):
        return {str(key): json_ready(val) for key, val in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


