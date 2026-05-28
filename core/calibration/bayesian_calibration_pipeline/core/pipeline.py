"""Mature geometry33 identifiability-regularized real-data ablation pipeline."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core.calibration.bayesian_calibration_pipeline.configs.pipeline_config import (
    METHOD_LABELS,
    METHOD_ORDER,
    Geometry33PipelineConfig,
)
from core.calibration.bayesian_calibration_pipeline.core.data_io import save_dataset
from core.calibration.bayesian_calibration_pipeline.core.data_split import (
    concat_c_dataset,
    head_dataset,
    load_real_pair,
    split_dataset_for_c,
)
from core.calibration.bayesian_calibration_pipeline.core.dynamic_identifiability import (
    DynamicIdentifiabilityFit,
    SubspaceIdentifiabilityPartition,
    SubspaceSequentialFit,
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
    build_error_parameters,
    parameter_scales,
    zero_error_vector,
)
from core.calibration.bayesian_calibration_pipeline.core.redundancy import output_jacobian
from core.calibration.bayesian_calibration_pipeline.core.regularization import (
    LambdaScore,
    RegularizedLMResult,
    euclidean_max,
    euclidean_rmse,
    evaluate_lambda_cv,
    fit_l2_lm,
    make_lambda_grid,
    merge_lambdas,
    random_folds,
    refined_lambda_grid,
    select_independent_parameters,
    select_lambda_score,
)
from core.calibration.bayesian_calibration_pipeline.core.robot_model import MultiSourceRobotModel


@dataclass
class ScenarioArtifacts:
    """Detailed artifacts used by report generation."""

    global_metrics: Any
    global_weights: np.ndarray
    dynamic_fit: DynamicIdentifiabilityFit
    subspace_partition: SubspaceIdentifiabilityPartition
    subspace_fit: SubspaceSequentialFit


def run_real_ablation(config: Geometry33PipelineConfig) -> dict[str, Any]:
    """Run the real world200/normal50 geometry33 ablation."""
    cfg = config.quickened()
    output_dir = Path(cfg.output_dir)
    data_dir = output_dir / "data"
    figures_dir = output_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    real_a_full, real_b_full = load_real_pair(str(cfg.real_a), str(cfg.real_b))
    real_a, real_a_c = split_dataset_for_c(
        real_a_full, cfg.real_c_fraction, cfg.seed + 301, "real_world200"
    )
    real_b, real_b_c = split_dataset_for_c(
        real_b_full, cfg.real_c_fraction, cfg.seed + 302, "real_normal50"
    )
    real_c = concat_c_dataset([real_a_c, real_b_c])
    if cfg.quick:
        real_a = head_dataset(real_a, 32)
        real_b = head_dataset(real_b, 24)
        real_c = head_dataset(real_c, 16)

    save_dataset(data_dir / "real_world200_train_for_ab.pkl", real_a)
    save_dataset(data_dir / "real_normal50_train_for_ab.pkl", real_b)
    save_dataset(data_dir / "real_c_validation.pkl", real_c)

    scenarios = [
        ("real_world200_to_normal50", "Real world200 -> normal50", real_a, real_b, real_c),
        ("real_normal50_to_world200", "Real normal50 -> world200", real_b, real_a, real_c),
    ]
    scenario_reports = []
    for index, (scenario_id, label, train, holdout, validation) in enumerate(scenarios):
        print(f"[{index + 1}/{len(scenarios)}] Running {label}...")
        scenario_reports.append(
            run_scenario(scenario_id, label, train, holdout, validation, cfg, figures_dir)
        )

    return {
        "settings": cfg.as_dict(),
        "method_order": list(METHOD_ORDER),
        "method_labels": dict(METHOD_LABELS),
        "scenarios": scenario_reports,
        "artifacts": {
            "output_dir": str(output_dir.resolve()),
            "data_dir": str(data_dir.resolve()),
            "figures_dir": str(figures_dir.resolve()),
            "json": str((output_dir / "report.json").resolve()),
            "html": str((output_dir / "report.html").resolve()),
            "notes": str((output_dir / "experiment_notes.html").resolve()),
        },
    }


def run_scenario(
    scenario_id: str,
    label: str,
    train: dict[str, np.ndarray],
    holdout: dict[str, np.ndarray],
    validation: dict[str, np.ndarray],
    config: Geometry33PipelineConfig,
    figures_dir: Path,
) -> dict[str, Any]:
    """Run all mature ablation methods for one A -> B direction."""
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
        tolerance=config.redundancy_tolerance,
        max_combinations=config.redundancy_max_combinations,
        jacobian_method=config.jacobian_method,
    )
    active = list(redundancy.independent_indices)
    all33 = list(range(len(parameters)))

    global_metrics = compute_identifiability_metrics(
        redundancy.jacobian,
        parameters,
        tolerance=config.redundancy_tolerance,
        rank=int(redundancy.rank),
        rho_threshold=config.rho_threshold,
        kappa_threshold=config.kappa_threshold,
        risk_beta=config.risk_beta,
        risk_power=config.risk_power,
        min_weight=config.min_weight,
        max_weight=config.max_weight,
        scaled_jacobian=True,
    )
    w3_weights = strategy_weights(
        global_metrics,
        active_indices=all33,
        strong=True,
        strong_weight=config.strong_weight,
    )
    pose_metrics = compute_pose_identifiability_metrics(
        model,
        train["joints"],
        parameters,
        zero,
        payloads=train.get("payloads"),
        directions=train.get("directions"),
        tolerance=config.redundancy_tolerance,
        rho_threshold=config.rho_threshold,
        kappa_threshold=config.kappa_threshold,
        risk_beta=config.risk_beta,
        risk_power=config.risk_power,
        min_weight=config.min_weight,
        max_weight=config.max_weight,
        risk_quantile=config.dynamic_risk_quantile,
        jacobian_method=config.jacobian_method,
    )
    train_jacobian = output_jacobian(
        model,
        train["joints"],
        zero,
        parameters,
        payloads=train.get("payloads"),
        directions=train.get("directions"),
        method=config.jacobian_method,
    )
    subspace_partition = build_identifiability_subspace_partition(
        pose_metrics,
        train_jacobian,
        parameters,
        k_candidates=config.subspace_k_candidates,
        min_cluster_size=config.subspace_min_cluster_size,
        seed=config.seed + 509,
        tolerance=config.redundancy_tolerance,
        rho_threshold=config.rho_threshold,
        kappa_threshold=config.kappa_threshold,
        risk_beta=config.risk_beta,
        risk_power=config.risk_power,
        min_weight=config.min_weight,
        max_weight=config.max_weight,
        strong_weight=config.strong_weight,
    )

    lambda_grid = make_lambda_grid(
        config.lambda_min_power,
        config.lambda_max_power,
        config.lambda_count,
    )

    print("  M0...")
    m0 = fit_l2_lm(
        model,
        train["joints"],
        train["measured_positions"],
        parameters,
        active,
        lambda_value=0.0,
        payloads=train.get("payloads"),
        directions=train.get("directions"),
        max_nfev=config.max_nfev,
    )
    print("  M1...")
    m1, m1_scores = select_random_cv_result(model, train, parameters, active, lambda_grid, config)
    print("  M6...")
    m6, m6_curve = select_ab_balance_result(
        model,
        train,
        holdout,
        validation,
        parameters,
        lambda_grid,
        config,
        lambda lam: fit_l2_lm(
            model,
            train["joints"],
            train["measured_positions"],
            parameters,
            active,
            lambda_value=float(lam),
            payloads=train.get("payloads"),
            directions=train.get("directions"),
            max_nfev=config.max_nfev,
        ),
    )
    print("  W3...")
    w3, w3_curve = select_ab_balance_result(
        model,
        train,
        holdout,
        validation,
        parameters,
        lambda_grid,
        config,
        lambda lam: fit_l2_lm(
            model,
            train["joints"],
            train["measured_positions"],
            parameters,
            all33,
            lambda_value=float(lam),
            payloads=train.get("payloads"),
            directions=train.get("directions"),
            max_nfev=config.max_nfev,
            regularization_weights=w3_weights,
        ),
    )
    print("  S1...")
    s1_cache: dict[float, SubspaceSequentialFit] = {}

    def fit_s1(lambda_value: float) -> RegularizedLMResult:
        key = lambda_key(lambda_value)
        if key not in s1_cache:
            s1_cache[key] = fit_subspace_sequential_l2(
                model,
                train["joints"],
                train["measured_positions"],
                parameters,
                subspace_partition,
                lambda_value=float(lambda_value),
                active_indices=all33,
                payloads=train.get("payloads"),
                directions=train.get("directions"),
                max_nfev=config.max_nfev,
            )
        return s1_cache[key].result

    s1, s1_curve = select_ab_balance_result(
        model, train, holdout, validation, parameters, lambda_grid, config, fit_s1
    )
    s1_detail = s1_cache[lambda_key(s1.lambda_value)]

    print("  D1...")
    d1_cache: dict[float, DynamicIdentifiabilityFit] = {}

    def fit_d1(lambda_value: float) -> RegularizedLMResult:
        key = lambda_key(lambda_value)
        if key not in d1_cache:
            d1_cache[key] = fit_dynamic_identifiability_l2(
                model,
                train["joints"],
                train["measured_positions"],
                parameters,
                active,
                lambda_value=float(lambda_value),
                payloads=train.get("payloads"),
                directions=train.get("directions"),
                max_nfev=config.max_nfev,
                outer_iterations=config.dynamic_outer_iterations,
                convergence_tol=config.dynamic_convergence_tol,
                tolerance=config.redundancy_tolerance,
                rho_threshold=config.rho_threshold,
                kappa_threshold=config.kappa_threshold,
                risk_beta=config.risk_beta,
                risk_power=config.risk_power,
                min_weight=config.min_weight,
                max_weight=config.max_weight,
                risk_quantile=config.dynamic_risk_quantile,
                jacobian_method=config.jacobian_method,
            )
        return d1_cache[key].result

    d1, d1_curve = select_ab_balance_result(
        model, train, holdout, validation, parameters, lambda_grid, config, fit_d1
    )
    d1_detail = d1_cache[lambda_key(d1.lambda_value)]

    results = {
        "M0": m0,
        "M1": m1,
        "M6": m6,
        "W3": w3,
        "S1": s1,
        "D1": d1,
    }
    curves = {
        "M1_random_cv": [lambda_score_dict(score) for score in m1_scores],
        "M6": m6_curve,
        "W3": w3_curve,
        "S1": s1_curve,
        "D1": d1_curve,
    }
    method_rows = {
        method: method_metrics(method, result, model, parameters, train, holdout, validation)
        for method, result in results.items()
    }
    add_gains(method_rows)
    parameter_rows = parameter_rows_for_report(
        parameters,
        active,
        global_metrics,
        w3_weights,
        d1_detail.final_weights,
        subspace_partition,
    )
    artifacts = ScenarioArtifacts(
        global_metrics=global_metrics,
        global_weights=w3_weights,
        dynamic_fit=d1_detail,
        subspace_partition=subspace_partition,
        subspace_fit=s1_detail,
    )
    return {
        "id": scenario_id,
        "label": label,
        "train_count": int(len(train["joints"])),
        "holdout_count": int(len(holdout["joints"])),
        "c_count": int(len(validation["joints"])),
        "structure": {
            "candidate_scope": "geometry33",
            "candidate_count": int(len(parameters)),
            "rank": int(redundancy.rank),
            "nullity": int(redundancy.nullity),
            "active_count": int(len(active)),
            "condition_number": finite_or_none(redundancy.condition_number),
            "subspace_K": int(subspace_partition.K),
            "subspace_order": list(subspace_partition.order),
            "subspace_cluster_sizes": list(subspace_partition.cluster_sizes),
        },
        "lambda_selections": {
            "M0": {"lambda": 0.0, "source": "none"},
            "M1": {"lambda": float(m1.lambda_value), "source": "random_cv_max"},
            "M6": selection_from_curve(m6_curve),
            "W3": selection_from_curve(w3_curve),
            "S1": selection_from_curve(s1_curve),
            "D1": selection_from_curve(d1_curve),
        },
        "methods": method_rows,
        "curves": curves,
        "identifiability": {
            "risk_beta": float(config.risk_beta),
            "risk_power": float(config.risk_power),
            "rho_threshold": float(config.rho_threshold),
            "kappa_threshold": float(config.kappa_threshold),
            "global_unidentifiable_count": int(np.sum(global_metrics.unidentifiable_mask)),
            "dynamic_risk_quantile": float(config.dynamic_risk_quantile),
            "parameter_table": parameter_rows,
        },
        "dynamic": {
            "iterations": d1_detail.iterations,
            "final_weight_min": float(np.min(d1_detail.final_weights)),
            "final_weight_max": float(np.max(d1_detail.final_weights)),
            "final_weight_mean": float(np.mean(d1_detail.final_weights)),
        },
        "subspace": {
            "candidate_scores": subspace_partition.candidate_scores,
            "summaries": subspace_partition.subspace_summaries,
            "stage_records": s1_detail.stage_records,
        },
        "_plot_artifacts": artifacts,
        "_datasets": {
            "train": train,
            "holdout": holdout,
            "validation": validation,
        },
    }


def select_random_cv_result(
    model: MultiSourceRobotModel,
    train: dict[str, np.ndarray],
    parameters: list[Any],
    active: list[int],
    lambda_grid: np.ndarray,
    config: Geometry33PipelineConfig,
) -> tuple[RegularizedLMResult, list[LambdaScore]]:
    """Select M1 lambda with random CV worst-fold RMSE."""
    folds = random_folds(len(train["joints"]), config.cv_folds, seed=config.seed + 11)
    coarse_scores = evaluate_lambda_cv(
        model,
        train["joints"],
        train["measured_positions"],
        parameters,
        active,
        folds,
        lambda_grid,
        payloads=train.get("payloads"),
        directions=train.get("directions"),
        max_nfev=config.max_nfev,
        norm="l2",
    )
    coarse_best = select_lambda_score(coarse_scores, criterion="max")
    fine_grid = refined_lambda_grid(coarse_best.lambda_value, count=config.fine_count)
    fine_scores = evaluate_lambda_cv(
        model,
        train["joints"],
        train["measured_positions"],
        parameters,
        active,
        folds,
        fine_grid,
        payloads=train.get("payloads"),
        directions=train.get("directions"),
        max_nfev=config.max_nfev,
        norm="l2",
    )
    scores = merge_scores(coarse_scores + fine_scores)
    best = select_lambda_score(scores, criterion="max")
    return (
        fit_l2_lm(
            model,
            train["joints"],
            train["measured_positions"],
            parameters,
            active,
            lambda_value=float(best.lambda_value),
            payloads=train.get("payloads"),
            directions=train.get("directions"),
            max_nfev=config.max_nfev,
        ),
        scores,
    )


def select_ab_balance_result(
    model: MultiSourceRobotModel,
    train: dict[str, np.ndarray],
    holdout: dict[str, np.ndarray],
    validation: dict[str, np.ndarray],
    parameters: list[Any],
    lambda_grid: np.ndarray,
    config: Geometry33PipelineConfig,
    fit_factory: Callable[[float], RegularizedLMResult],
) -> tuple[RegularizedLMResult, list[dict[str, float]]]:
    """Select lambda by RMSE_A + RMSE_B + alpha * |RMSE_A - RMSE_B|."""
    cache: dict[float, RegularizedLMResult] = {}

    def cached(lambda_value: float) -> RegularizedLMResult:
        key = lambda_key(lambda_value)
        if key not in cache:
            cache[key] = fit_factory(float(lambda_value))
        return cache[key]

    coarse_curve = ab_balance_curve(
        model, train, holdout, validation, parameters, lambda_grid, cached, config.ab_balance_alpha
    )
    coarse_best = min(coarse_curve, key=lambda row: (row["ab_balance_score_m"], row["lambda"]))
    final_grid = merge_lambdas(
        lambda_grid,
        refined_lambda_grid(coarse_best["lambda"], count=config.fine_count),
    )
    final_curve = ab_balance_curve(
        model, train, holdout, validation, parameters, final_grid, cached, config.ab_balance_alpha
    )
    best = min(final_curve, key=lambda row: (row["ab_balance_score_m"], row["lambda"]))
    return cached(best["lambda"]), final_curve


def ab_balance_curve(
    model: MultiSourceRobotModel,
    train: dict[str, np.ndarray],
    holdout: dict[str, np.ndarray],
    validation: dict[str, np.ndarray],
    parameters: list[Any],
    lambdas: np.ndarray,
    fit_for_lambda: Callable[[float], RegularizedLMResult],
    alpha: float,
) -> list[dict[str, float]]:
    """Return A/B-balance curve; validation C is recorded but never selected."""
    rows: list[dict[str, float]] = []
    for value in lambdas:
        result = fit_for_lambda(float(value))
        train_rmse, train_max = dataset_errors(model, parameters, train, result.vector)
        holdout_rmse, holdout_max = dataset_errors(model, parameters, holdout, result.vector)
        c_rmse, c_max = dataset_errors(model, parameters, validation, result.vector)
        gap = abs(train_rmse - holdout_rmse)
        total = train_rmse + holdout_rmse
        rows.append(
            {
                "lambda": float(value),
                "train_rmse_m": float(train_rmse),
                "selection_rmse_m": float(holdout_rmse),
                "holdout_rmse_m": float(holdout_rmse),
                "c_rmse_m": float(c_rmse),
                "train_max_m": float(train_max),
                "holdout_max_m": float(holdout_max),
                "c_max_m": float(c_max),
                "ab_total_rmse_m": float(total),
                "ab_abs_gap_m": float(gap),
                "ab_balance_score_m": float(total + float(alpha) * gap),
            }
        )
    return sorted(rows, key=lambda row: row["lambda"])


def dataset_errors(
    model: MultiSourceRobotModel,
    parameters: list[Any],
    dataset: dict[str, np.ndarray],
    vector: np.ndarray,
) -> tuple[float, float]:
    """Return Euclidean RMSE and max TCP error in meters."""
    predicted = model.batch_positions(
        dataset["joints"],
        vector,
        parameters,
        dataset.get("payloads"),
        dataset.get("directions"),
    )
    return (
        euclidean_rmse(dataset["measured_positions"], predicted),
        euclidean_max(dataset["measured_positions"], predicted),
    )


def method_metrics(
    method: str,
    result: RegularizedLMResult,
    model: MultiSourceRobotModel,
    parameters: list[Any],
    train: dict[str, np.ndarray],
    holdout: dict[str, np.ndarray],
    validation: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Build the standard A/B/C metric row for one method."""
    train_rmse, train_max = dataset_errors(model, parameters, train, result.vector)
    holdout_rmse, holdout_max = dataset_errors(model, parameters, holdout, result.vector)
    c_rmse, c_max = dataset_errors(model, parameters, validation, result.vector)
    return {
        "method": method,
        "label": METHOD_LABELS[method],
        "lambda": float(result.lambda_value),
        "active_count": int(len(result.active_indices)),
        "train_A_rmse_mm": float(train_rmse * 1000.0),
        "selection_B_rmse_mm": float(holdout_rmse * 1000.0),
        "validation_C_rmse_mm": float(c_rmse * 1000.0),
        "A_B_gap_mm": float((holdout_rmse - train_rmse) * 1000.0),
        "C_A_gap_mm": float((c_rmse - train_rmse) * 1000.0),
        "train_A_max_mm": float(train_max * 1000.0),
        "selection_B_max_mm": float(holdout_max * 1000.0),
        "validation_C_max_mm": float(c_max * 1000.0),
        "normalized_parameter_l2": float(result.normalized_parameter_norm),
        "weighted_parameter_l2": float(result.weighted_normalized_parameter_l2),
        "weight_min": float(result.regularization_weight_min),
        "weight_max": float(result.regularization_weight_max),
        "nfev": int(result.nfev),
        "success": bool(result.success),
    }


def add_gains(method_rows: dict[str, dict[str, Any]]) -> None:
    """Add B/C gain columns relative to M0, M6 and W3 baselines."""
    baselines = {"M0": method_rows["M0"], "M6": method_rows["M6"], "W3": method_rows["W3"]}
    for row in method_rows.values():
        for name, baseline in baselines.items():
            row[f"B_gain_vs_{name}_mm"] = float(
                baseline["selection_B_rmse_mm"] - row["selection_B_rmse_mm"]
            )
            row[f"C_gain_vs_{name}_mm"] = float(
                baseline["validation_C_rmse_mm"] - row["validation_C_rmse_mm"]
            )


def parameter_rows_for_report(
    parameters: list[Any],
    active: list[int],
    global_metrics: Any,
    w3_weights: np.ndarray,
    d1_weights: np.ndarray,
    partition: SubspaceIdentifiabilityPartition,
) -> list[dict[str, Any]]:
    """Return per-parameter global/D1/S1 weights."""
    active_set = set(int(index) for index in active)
    rows = []
    for index, parameter in enumerate(parameters):
        row: dict[str, Any] = {
            "index": int(index),
            "parameter": parameter.name,
            "scale": float(global_metrics.scales[index]),
            "svd_active": bool(index in active_set),
            "global_rho": float(global_metrics.rho[index]),
            "global_kappa": float(global_metrics.kappa[index]),
            "global_risk": float(global_metrics.risk[index]),
            "global_base_weight": float(global_metrics.base_weights[index]),
            "W3_weight": float(w3_weights[index]),
            "D1_final_weight": float(d1_weights[index]),
        }
        for subspace in range(partition.K):
            metrics = partition.subspace_metrics[subspace]
            row[f"S1_s{subspace}_rho"] = float(metrics.rho[index])
            row[f"S1_s{subspace}_kappa"] = float(metrics.kappa[index])
            row[f"S1_s{subspace}_risk"] = float(metrics.risk[index])
            row[f"S1_s{subspace}_weight"] = float(partition.subspace_weights[subspace][index])
            row[f"S1_s{subspace}_weak"] = bool(metrics.unidentifiable_mask[index])
        rows.append(row)
    return rows


def selection_from_curve(curve: list[dict[str, float]]) -> dict[str, float | str]:
    best = min(curve, key=lambda row: (row["ab_balance_score_m"], row["lambda"]))
    return {
        "lambda": float(best["lambda"]),
        "source": "ab_balance",
        "score_m": float(best["ab_balance_score_m"]),
        "ab_total_rmse_m": float(best["ab_total_rmse_m"]),
        "ab_abs_gap_m": float(best["ab_abs_gap_m"]),
    }


def lambda_score_dict(score: LambdaScore) -> dict[str, float]:
    return {
        "lambda": float(score.lambda_value),
        "mean_rmse_mm": float(score.mean_rmse * 1000.0),
        "max_rmse_mm": float(score.max_rmse * 1000.0),
        "std_rmse_mm": float(score.std_rmse * 1000.0),
    }


def merge_scores(scores: list[LambdaScore]) -> list[LambdaScore]:
    best_by_lambda: dict[float, LambdaScore] = {}
    for score in scores:
        best_by_lambda[lambda_key(score.lambda_value)] = score
    return [best_by_lambda[key] for key in sorted(best_by_lambda)]


def lambda_key(value: float) -> float:
    return round(float(value), 15)


def finite_or_none(value: float) -> float | None:
    numeric = float(value)
    return numeric if np.isfinite(numeric) else None


