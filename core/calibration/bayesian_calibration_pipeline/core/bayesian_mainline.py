"""Bayesian non-geometric calibration mainline.

This module keeps the accepted strategy isolated from the earlier exploratory
non-geometric experiment: geometry regularization selects an anchor, an
interpretable Bayesian basis model explains residual structure, and a final
MAP step releases all 33 geometric candidates under a geometry prior.
"""

from __future__ import annotations

from dataclasses import replace
from html import escape
from typing import Any

import numpy as np

from core.calibration.bayesian_calibration_pipeline.configs.pipeline_config import (
    METHOD_LABELS,
    Geometry33PipelineConfig,
)
from core.calibration.bayesian_calibration_pipeline.core.data_split import (
    canonical_dataset,
    concat_c_dataset,
    head_dataset,
    split_dataset_for_c,
    subset_dataset,
)
from core.calibration.bayesian_calibration_pipeline.core.non_geometric import (
    NonGeometricConfig,
    fine_tune_to_report,
    fit_stage1_geometry_baselines,
    fit_statistical_global_fine_tune,
    json_ready,
    stage1_split_rows,
    statistical_position_metrics,
    statistical_whiteness_report,
    verify_nongeometric_dataset,
)
from core.calibration.bayesian_calibration_pipeline.core.statistical_residual import (
    StatisticalResidualConfig,
    combine_views,
    make_dataset_view,
    run_bayesian_basis_residual_model,
)


def normalize_train_scope(value: str, name: str) -> str:
    scope = str(value).strip().upper()
    if scope not in {"A", "AB"}:
        raise ValueError(f"{name} must be 'A' or 'AB', got {value!r}")
    return scope


def normalize_selection_scope(value: str) -> str:
    scope = str(value).strip()
    aliases = {
        "internal": "internal",
        "blocked_cv": "internal",
        "B_validation": "B_validation",
        "b_validation": "B_validation",
    }
    if scope not in aliases:
        raise ValueError(
            "bayesian_selection_scope must be 'internal' or 'B_validation', "
            f"got {value!r}"
        )
    return aliases[scope]


def train_view_for_scope(scope: str, a_train_view: Any, ab_train_view: Any) -> Any:
    return a_train_view if scope == "A" else ab_train_view


def statistical_design_for_view(model: Any, view_name: str, row_count: int) -> np.ndarray:
    """Return the statistical design matrix for a named view.

    The raw model's design_train follows the coefficient-fit split, while some
    diagnostics and peeled comparisons always evaluate on A+B.  This helper
    prevents silent row-count mismatches when the model was fit only on A.
    """
    feature_count = int(np.asarray(model.coefficients).shape[0])
    if feature_count == 0:
        return np.zeros((int(row_count), 0), dtype=float)
    if view_name in model.design_eval:
        design = np.asarray(model.design_eval[view_name], dtype=float)
        if design.shape[0] == int(row_count):
            return design
    if view_name == "A_B_train" and "A_train" in model.design_eval and "B_train" in model.design_eval:
        design = np.vstack([
            np.asarray(model.design_eval["A_train"], dtype=float),
            np.asarray(model.design_eval["B_train"], dtype=float),
        ])
        if design.shape[0] == int(row_count):
            return design
    design = np.asarray(model.design_train, dtype=float)
    if design.shape[0] == int(row_count):
        return design
    raise ValueError(
        f"Statistical design for {view_name} has incompatible row count: "
        f"expected {row_count}, got train={model.design_train.shape[0]}"
    )


def statistical_model_for_train_view(model: Any, train_view: Any) -> Any:
    design = statistical_design_for_view(model, train_view.name, len(train_view.residuals))
    if design.shape[1]:
        prediction = design @ np.asarray(model.coefficients, dtype=float)
    else:
        prediction = np.zeros_like(train_view.residuals)
    return replace(model, design_train=design, train_prediction=prediction)


def run_bayesian_calibration_analysis(
    train: dict[str, np.ndarray],
    selection: dict[str, np.ndarray],
    geometry_config: Geometry33PipelineConfig,
    non_geo_config: NonGeometricConfig,
    statistical_config: StatisticalResidualConfig | None = None,
    *,
    fine_tune_lambda_ratios: tuple[float, ...] | None = None,
    external_c: dict[str, np.ndarray] | None = None,
    bayesian_train_scope: str = "AB",
    fine_tune_train_scope: str = "AB",
    bayesian_selection_scope: str = "internal",
) -> dict[str, Any]:
    """Run the real-data Bayesian calibration mainline."""
    cfg = geometry_config.quickened()
    ng_cfg = non_geo_config.quickened() if cfg.quick else non_geo_config
    stat_cfg = statistical_config if statistical_config is not None else StatisticalResidualConfig(seed=ng_cfg.seed)
    stat_cfg_run = stat_cfg.quickened() if cfg.quick else stat_cfg
    bayesian_train_scope = normalize_train_scope(bayesian_train_scope, "bayesian_train_scope")
    fine_tune_train_scope = normalize_train_scope(fine_tune_train_scope, "fine_tune_train_scope")
    bayesian_selection_scope = normalize_selection_scope(bayesian_selection_scope)

    train_full = canonical_dataset(train)
    selection_full = canonical_dataset(selection)
    if external_c is None:
        train_c, train_holdout_c = split_dataset_for_c(
            train_full, cfg.real_c_fraction, cfg.seed + 701, "A_world200"
        )
        selection_c, selection_holdout_c = split_dataset_for_c(
            selection_full, cfg.real_c_fraction, cfg.seed + 702, "B_normal50"
        )
        c_source = "heldout_from_A_and_B"
    else:
        train_c = train_full
        selection_c = selection_full
        c_full = canonical_dataset(external_c)
        c_order = np.random.default_rng(cfg.seed + 703).permutation(len(c_full["joints"]))
        split_at = max(1, min(len(c_order) - 1, len(c_order) // 2))
        train_holdout_c = subset_dataset(c_full, np.sort(c_order[:split_at]), "C_external_part_1")
        selection_holdout_c = subset_dataset(c_full, np.sort(c_order[split_at:]), "C_external_part_2")
        c_source = "external_C_space"
    if cfg.quick:
        train_c = head_dataset(train_c, 32)
        selection_c = head_dataset(selection_c, 24)
        train_holdout_c = head_dataset(train_holdout_c, 12)
        selection_holdout_c = head_dataset(selection_holdout_c, 10)

    data_checks = [
        verify_nongeometric_dataset(train_c, "A_train_world200_nongeometric"),
        verify_nongeometric_dataset(selection_c, "B_train_normal50_nongeometric"),
        verify_nongeometric_dataset(train_holdout_c, "A_C_world200_nongeometric"),
        verify_nongeometric_dataset(selection_holdout_c, "B_C_normal50_nongeometric"),
    ]

    stage1 = fit_stage1_geometry_baselines(
        train_c,
        selection_c,
        cfg,
        position_noise_std_m=None,
        anchor_method="S1",
    )
    anchor_vector = stage1.selected_vector
    model = stage1.model
    parameters = stage1.parameters
    all33_active = list(range(len(parameters)))

    anchor_pred_train = model.batch_positions(
        train_c["joints"],
        anchor_vector,
        parameters,
        train_c.get("payloads"),
        train_c.get("directions"),
    )
    anchor_pred_selection = model.batch_positions(
        selection_c["joints"],
        anchor_vector,
        parameters,
        selection_c.get("payloads"),
        selection_c.get("directions"),
    )
    anchor_pred_train_c = model.batch_positions(
        train_holdout_c["joints"],
        anchor_vector,
        parameters,
        train_holdout_c.get("payloads"),
        train_holdout_c.get("directions"),
    )
    anchor_pred_selection_c = model.batch_positions(
        selection_holdout_c["joints"],
        anchor_vector,
        parameters,
        selection_holdout_c.get("payloads"),
        selection_holdout_c.get("directions"),
    )

    a_train_view = make_dataset_view("A_train", train_c, anchor_pred_train, "A")
    b_train_view = make_dataset_view("B_train", selection_c, anchor_pred_selection, "B")
    a_c_view = make_dataset_view("A_C", train_holdout_c, anchor_pred_train_c, "A")
    b_c_view = make_dataset_view("B_C", selection_holdout_c, anchor_pred_selection_c, "B")
    stat_train_view = combine_views("A_B_train", [a_train_view, b_train_view])
    c_all_view = combine_views("C_all", [a_c_view, b_c_view])
    stat_eval_views = {
        "A_train": a_train_view,
        "B_train": b_train_view,
        "A_C": a_c_view,
        "B_C": b_c_view,
        "C_all": c_all_view,
    }

    bayesian_train_view = train_view_for_scope(
        bayesian_train_scope,
        a_train_view,
        stat_train_view,
    )
    fine_tune_train_view = train_view_for_scope(
        fine_tune_train_scope,
        a_train_view,
        stat_train_view,
    )
    selection_view_name = "B_train" if bayesian_selection_scope == "B_validation" else None
    bayesian_report = run_bayesian_basis_residual_model(
        bayesian_train_view,
        stat_eval_views,
        stat_cfg_run,
        quick=False,
        selection_view_name=selection_view_name,
    )
    bayesian_model = bayesian_report.pop("_raw_model")
    fine_tune_model = statistical_model_for_train_view(bayesian_model, fine_tune_train_view)

    zero_design_train = np.zeros((len(stat_train_view.residuals), 0), dtype=float)
    zero_design_eval = {
        name: np.zeros((len(view.residuals), 0), dtype=float)
        for name, view in stat_eval_views.items()
    }
    geometry_anchor_metrics = statistical_position_metrics(
        model,
        parameters,
        stat_train_view,
        stat_eval_views,
        anchor_vector,
        np.zeros((0, 3), dtype=float),
        zero_design_train,
        zero_design_eval,
    )
    stage1_anchor_lambda = max(float(stage1.results[stage1.selected_method].lambda_value), 1.0e-12)
    ratio_grid = tuple(
        float(value)
        for value in (
            fine_tune_lambda_ratios
            if fine_tune_lambda_ratios is not None
            else (1000.0,)
        )
        if float(value) > 0.0
    )
    if not ratio_grid:
        raise ValueError("fine_tune_lambda_ratios must contain at least one positive value.")
    fine_tune_search: list[dict[str, Any]] = []
    fine_tune_payloads = []
    for ratio in ratio_grid:
        lambda_theta = stage1_anchor_lambda * float(ratio)
        candidate = fit_statistical_global_fine_tune(
            model,
            parameters,
            fine_tune_train_view,
            stat_eval_views,
            anchor_vector,
            all33_active,
            fine_tune_model,
            stat_cfg_run,
            max_nfev=ng_cfg.nonlinear_max_nfev,
            geometry_prior_lambda=lambda_theta,
            scale_position_residual_by_noise=False,
        )
        peeled_metrics = statistical_position_metrics(
            model,
            parameters,
            stat_train_view,
            stat_eval_views,
            candidate.vector,
            np.zeros((0, 3), dtype=float),
            zero_design_train,
            zero_design_eval,
        )
        row = fine_tune_search_row(
            ratio,
            lambda_theta,
            candidate,
            bayesian_model.metrics,
            geometry_anchor_metrics,
            peeled_metrics,
            stat_cfg_run.noise_std_m,
        )
        fine_tune_search.append(row)
        fine_tune_payloads.append((row, candidate, peeled_metrics))
    controlled_payloads = [payload for payload in fine_tune_payloads if bool(payload[0]["controlled_geometry"])]
    accepted_controlled_payloads = [payload for payload in controlled_payloads if bool(payload[1].accepted)]
    accepted_payloads = [payload for payload in fine_tune_payloads if bool(payload[1].accepted)]
    if accepted_controlled_payloads:
        selected_row, fine_tune, peeled_geometry_metrics = min(
            accepted_controlled_payloads,
            key=lambda item: (
                float(item[0]["post_C_all_rmse_mm"]),
                float(item[0]["theta_update_scaled_l2"]),
                float(item[0]["peeled_C_all_rmse_mm"]),
            ),
        )
        selection_rule = (
            "accepted and controlled first: peeled C_all degradation <= sqrt(3)*sigma, "
            "theta_update_scaled_l2 <= 5, and C_all not worse; then lowest C_all RMSE"
        )
    elif accepted_payloads:
        selected_row, fine_tune, peeled_geometry_metrics = min(
            accepted_payloads,
            key=lambda item: (
                float(item[0]["post_C_all_rmse_mm"]),
                float(item[0]["theta_update_scaled_l2"]),
                float(item[0]["peeled_C_all_rmse_mm"]),
            ),
        )
        selection_rule = (
            "no accepted ratio passed controlled-geometry criteria; selected accepted lowest C_all RMSE"
        )
    else:
        selected_row, fine_tune, peeled_geometry_metrics = min(
            controlled_payloads if controlled_payloads else fine_tune_payloads,
            key=lambda item: (
                float(item[0]["peeled_C_all_delta_vs_anchor_mm"]),
                float(item[0]["theta_update_scaled_l2"]),
                float(item[0]["post_C_all_rmse_mm"]),
            ),
        )
        selection_rule = (
            "no fine-tune ratio improved C_all; selected diagnostic candidate only and final accepted model rolls back to Bayesian basis"
        )
    final_accepted_metrics = fine_tune.metrics if fine_tune.accepted else bayesian_model.metrics
    final_accepted_method = "bayesian_basis_fine_tuned" if fine_tune.accepted else "bayesian_basis"
    bayesian_report["geometry_anchor_metrics"] = geometry_anchor_metrics
    bayesian_report["fine_tune"] = {
        "objective": (
            "MAP objective = ||y - p(q;theta) - Phi beta||^2 + "
            "lambda_theta ||D^-1(theta-theta_anchor)||^2 + beta^T Lambda beta"
        ),
        "stage1_anchor_lambda": float(stage1_anchor_lambda),
        "selected_ratio": float(selected_row["ratio_to_stage1_lambda"]),
        "geometry_prior_lambda": float(selected_row["lambda_theta"]),
        "released_geometry_parameter_count": int(len(all33_active)),
        "released_geometry_parameter_rule": "all 33 geometric candidates",
        "lambda_search_rule": selection_rule,
        "bayesian_train_scope": bayesian_train_scope,
        "bayesian_selection_scope": bayesian_selection_scope,
        "fine_tune_train_scope": fine_tune_train_scope,
        "search_rows": fine_tune_search,
        "rows": [fine_tune_to_report(fine_tune, bayesian_model.metrics)],
    }

    stage_metrics = bayesian_stage_metric_rows(
        geometry_anchor_metrics,
        bayesian_model.metrics,
        fine_tune.metrics,
        final_accepted_metrics,
        final_accepted_method,
        peeled_geometry_metrics,
    )
    peeled_comparison = peeled_geometry_comparison_rows(
        geometry_anchor_metrics,
        peeled_geometry_metrics,
    )
    bayesian_report["final_statistical_model"] = {
        "method": final_accepted_method,
        "fine_tuned": bool(fine_tune.accepted),
        "diagnostic_fine_tune_method": "bayesian_basis_fine_tuned",
        "accepted_by_C_all": bool(fine_tune.accepted),
        "acceptance_reason": fine_tune.reason,
        "objective_decreased": bool(fine_tune.objective_final <= fine_tune.objective_initial + 1.0e-9),
        "stage1_anchor_lambda": float(stage1_anchor_lambda),
        "selected_ratio_to_stage1_lambda": float(selected_row["ratio_to_stage1_lambda"]),
        "geometry_prior_lambda": float(selected_row["lambda_theta"]),
        "C_all_rmse_mm": float(final_accepted_metrics.get("C_all_rmse_mm", float("nan"))),
        "train_rmse_mm": float(final_accepted_metrics.get("train_rmse_mm", float("nan"))),
        "A_train_rmse_mm": float(final_accepted_metrics.get("A_train_rmse_mm", float("nan"))),
        "B_train_rmse_mm": float(final_accepted_metrics.get("B_train_rmse_mm", float("nan"))),
        "A_C_rmse_mm": float(final_accepted_metrics.get("A_C_rmse_mm", float("nan"))),
        "B_C_rmse_mm": float(final_accepted_metrics.get("B_C_rmse_mm", float("nan"))),
        "selection_rule": "Bayesian basis + all33 MAP fine-tune is the fixed mainline; C_all is reported as held-out validation.",
    }
    bayesian_report["peeled_geometry_only"] = {
        "interpretation": "Evaluate p(q; theta_finetuned) without Phi beta to test whether the geometric body improved.",
        "metrics": peeled_geometry_metrics,
        "comparison_rows": peeled_comparison,
    }
    whiteness_design_train = statistical_design_for_view(
        bayesian_model,
        "A_B_train",
        len(stat_train_view.residuals),
    )
    bayesian_report["whiteness"] = statistical_whiteness_report(
        model,
        parameters,
        fine_tune.vector,
        fine_tune.coefficients,
        whiteness_design_train,
        bayesian_model.design_eval,
        stat_train_view,
        stat_eval_views,
        ng_cfg,
    )

    stage1_abc_rows = stage1_split_rows(
        stage1,
        {
            "A_train": train_c,
            "B_train": selection_c,
            "A_C": train_holdout_c,
            "B_C": selection_holdout_c,
        },
    )

    selected_basis_groups = [
        str(row["name"])
        for row in bayesian_report["models"]["bayesian_basis"].get("selected_groups", [])
    ]
    summary = {
        "stage1_anchor_method": stage1.selected_method,
        "stage1_anchor_label": METHOD_LABELS.get(stage1.selected_method, stage1.selected_method),
        "stage1_anchor_C_all_rmse_mm": float(geometry_anchor_metrics.get("C_all_rmse_mm", float("nan"))),
        "bayesian_before_finetune_C_all_rmse_mm": float(bayesian_model.metrics.get("C_all_rmse_mm", float("nan"))),
        "bayesian_finetuned_C_all_rmse_mm": float(fine_tune.metrics.get("C_all_rmse_mm", float("nan"))),
        "final_accepted_method": final_accepted_method,
        "final_accepted_C_all_rmse_mm": float(final_accepted_metrics.get("C_all_rmse_mm", float("nan"))),
        "peeled_geometry_C_all_rmse_mm": float(peeled_geometry_metrics.get("C_all_rmse_mm", float("nan"))),
        "bayesian_gain_vs_anchor_C_all_mm": float(
            geometry_anchor_metrics.get("C_all_rmse_mm", float("nan"))
            - bayesian_model.metrics.get("C_all_rmse_mm", float("nan"))
        ),
        "finetune_gain_vs_bayesian_C_all_mm": float(
            bayesian_model.metrics.get("C_all_rmse_mm", float("nan"))
            - fine_tune.metrics.get("C_all_rmse_mm", float("nan"))
        ),
        "peeled_geometry_delta_vs_anchor_C_all_mm": float(
            peeled_geometry_metrics.get("C_all_rmse_mm", float("nan"))
            - geometry_anchor_metrics.get("C_all_rmse_mm", float("nan"))
        ),
        "peeled_geometry_improved": bool(
            peeled_geometry_metrics.get("C_all_rmse_mm", float("inf"))
            <= geometry_anchor_metrics.get("C_all_rmse_mm", float("-inf"))
        ),
        "fine_tune_accepted_by_C_all": bool(fine_tune.accepted),
        "fine_tune_objective_decreased": bool(fine_tune.objective_final <= fine_tune.objective_initial + 1.0e-9),
        "released_geometry_parameter_count": int(len(all33_active)),
        "stage1_anchor_lambda": float(stage1_anchor_lambda),
        "fine_tune_lambda_ratio": float(selected_row["ratio_to_stage1_lambda"]),
        "fine_tune_lambda_theta": float(selected_row["lambda_theta"]),
        "selected_basis_groups": selected_basis_groups,
    }

    return {
        "settings": {
            "geometry": cfg.as_dict(),
            "non_geometric": ng_cfg.as_dict(),
            "statistical_residual": stat_cfg_run.as_dict(),
            "geometry_objective": {
                "position_noise_std_m": float(stat_cfg_run.noise_std_m),
                "position_noise_std_mm": float(stat_cfg_run.noise_std_m * 1000.0),
                "objective": "Stage 1 geometry fits use ||p(theta)-y||^2 + lambda*prior; sigma is kept for residual statistics, not Stage 1 scaling.",
                "stage1_scaled_by_sigma": False,
                "stage4_scaled_by_sigma": False,
            },
            "stage4_lambda_search": {
                "ratios": [float(value) for value in ratio_grid],
                "stage1_anchor_lambda": float(stage1_anchor_lambda),
                "selected_ratio": float(selected_row["ratio_to_stage1_lambda"]),
                "selected_lambda_theta": float(selected_row["lambda_theta"]),
                "selection_rule": selection_rule,
                "objective": "Stage 4 uses an unscaled position residual and lambda_theta = lambda_stage1 * ratio.",
            },
            "training_scopes": {
                "bayesian_train_scope": bayesian_train_scope,
                "bayesian_train_samples": int(len(bayesian_train_view.residuals)),
                "bayesian_selection_scope": bayesian_selection_scope,
                "bayesian_selection_view": selection_view_name or "",
                "fine_tune_train_scope": fine_tune_train_scope,
                "fine_tune_train_samples": int(len(fine_tune_train_view.residuals)),
                "B_train_role": (
                    "coefficient_fit_and_fine_tune"
                    if bayesian_train_scope == "AB" and fine_tune_train_scope == "AB"
                    else "selection_only"
                    if bayesian_selection_scope == "B_validation"
                    else "evaluation_only_for_fine_tune"
                    if fine_tune_train_scope == "A"
                    else "see_scopes"
                ),
            },
            "abc_split": {
                "A_full_samples": int(len(train_full["joints"])),
                "B_full_samples": int(len(selection_full["joints"])),
                "A_train_samples": int(len(train_c["joints"])),
                "B_train_samples": int(len(selection_c["joints"])),
                "A_C_samples": int(len(train_holdout_c["joints"])),
                "B_C_samples": int(len(selection_holdout_c["joints"])),
                "C_fraction": float(cfg.real_c_fraction),
                "C_source": c_source,
            },
            "mainline_note": (
                "C is held-out validation used for model checking and fine-tune acceptance; "
                "it is not claimed as a final untouched test set."
            ),
        },
        "validation": {
            "data_checks": data_checks,
            "synthetic_statistical_check": bayesian_report.get("synthetic_validation", {}),
            "map_objective_decreased": bool(summary["fine_tune_objective_decreased"]),
            "peeled_compensation_check": "fine_tuned geometry-only metrics are computed with Phi beta removed",
        },
        "stage1_geometry": {
            "methods": stage1.rows,
            "split_metrics": stage1_abc_rows,
            "selected_method": stage1.selected_method,
            "selected_label": METHOD_LABELS.get(stage1.selected_method, stage1.selected_method),
            "redundancy": {
                "rank": int(stage1.redundancy.rank),
                "nullity": int(stage1.redundancy.nullity),
                "independent_count": int(len(stage1.redundancy.independent_indices)),
                "independent_indices": [int(i) for i in stage1.redundancy.independent_indices],
            },
        },
        "bayesian_residual": bayesian_report,
        "bayesian_stage_metrics": stage_metrics,
        "fine_tuned_geometry_only": bayesian_report["peeled_geometry_only"],
        "final_summary": summary,
    }


def bayesian_stage_metric_rows(
    anchor_metrics: dict[str, float],
    bayesian_metrics: dict[str, float],
    fine_tuned_metrics: dict[str, float],
    final_accepted_metrics: dict[str, float],
    final_accepted_method: str,
    peeled_metrics: dict[str, float],
) -> list[dict[str, Any]]:
    stages = [
        ("geometry_anchor", "Stage 1 geometry-only anchor", anchor_metrics),
        ("bayesian_basis", "Bayesian residual compensation before fine-tune", bayesian_metrics),
        ("bayesian_basis_fine_tuned_candidate", "Diagnostic Bayesian residual + all33 geometry MAP fine-tune candidate", fine_tuned_metrics),
        ("final_accepted_model", f"Final accepted model ({final_accepted_method})", final_accepted_metrics),
        ("fine_tuned_geometry_only", "Peeled p(theta_finetuned) without Bayesian correction", peeled_metrics),
    ]
    keys = ["train", "A_train", "B_train", "A_C", "B_C", "C_all"]
    rows = []
    for stage, label, metrics in stages:
        row: dict[str, Any] = {"stage": stage, "label": label}
        for key in keys:
            row[f"{key}_rmse_mm"] = float(metrics.get(f"{key}_rmse_mm", float("nan")))
        rows.append(row)
    return rows


def fine_tune_search_row(
    ratio: float,
    lambda_theta: float,
    result: Any,
    pre_metrics: dict[str, float],
    anchor_metrics: dict[str, float],
    peeled_metrics: dict[str, float],
    noise_std_m: float,
) -> dict[str, Any]:
    peeled_delta = float(
        peeled_metrics.get("C_all_rmse_mm", float("nan"))
        - anchor_metrics.get("C_all_rmse_mm", float("nan"))
    )
    allowed_delta = float(np.sqrt(3.0) * float(noise_std_m) * 1000.0)
    theta_update = float(result.theta_update_scaled_l2)
    return {
        "ratio_to_stage1_lambda": float(ratio),
        "lambda_theta": float(lambda_theta),
        "accepted": bool(result.accepted),
        "objective_initial": float(result.objective_initial),
        "objective_final": float(result.objective_final),
        "objective_decreased": bool(result.objective_final <= result.objective_initial + 1.0e-9),
        "pre_train_rmse_mm": float(pre_metrics.get("train_rmse_mm", float("nan"))),
        "post_train_rmse_mm": float(result.metrics.get("train_rmse_mm", float("nan"))),
        "post_A_train_rmse_mm": float(result.metrics.get("A_train_rmse_mm", float("nan"))),
        "post_B_train_rmse_mm": float(result.metrics.get("B_train_rmse_mm", float("nan"))),
        "post_A_C_rmse_mm": float(result.metrics.get("A_C_rmse_mm", float("nan"))),
        "post_B_C_rmse_mm": float(result.metrics.get("B_C_rmse_mm", float("nan"))),
        "post_C_all_rmse_mm": float(result.metrics.get("C_all_rmse_mm", float("nan"))),
        "peeled_train_rmse_mm": float(peeled_metrics.get("train_rmse_mm", float("nan"))),
        "peeled_A_train_rmse_mm": float(peeled_metrics.get("A_train_rmse_mm", float("nan"))),
        "peeled_B_train_rmse_mm": float(peeled_metrics.get("B_train_rmse_mm", float("nan"))),
        "peeled_A_C_rmse_mm": float(peeled_metrics.get("A_C_rmse_mm", float("nan"))),
        "peeled_B_C_rmse_mm": float(peeled_metrics.get("B_C_rmse_mm", float("nan"))),
        "peeled_C_all_rmse_mm": float(peeled_metrics.get("C_all_rmse_mm", float("nan"))),
        "peeled_C_all_delta_vs_anchor_mm": peeled_delta,
        "allowed_peeled_delta_mm": allowed_delta,
        "theta_update_scaled_l2": theta_update,
        "controlled_geometry": bool(peeled_delta <= allowed_delta and theta_update <= 5.0),
        "beta_update_l2_mm": float(result.beta_update_l2_mm),
        "initial_beta_l2_mm": float(result.initial_beta_l2_mm),
        "beta_l2_mm": float(result.beta_l2_mm),
        "beta_prior_residual_scale": float(result.beta_prior_residual_scale),
        "data_residual_scale_m": float(result.data_residual_scale_m),
        "objective_data_initial": float(result.objective_terms_initial.get("data_term", float("nan"))),
        "objective_beta_prior_initial": float(result.objective_terms_initial.get("beta_prior_term", float("nan"))),
        "objective_data_final": float(result.objective_terms_final.get("data_term", float("nan"))),
        "objective_beta_prior_final": float(result.objective_terms_final.get("beta_prior_term", float("nan"))),
        "nfev": int(result.nfev),
        "success": bool(result.success),
    }


def peeled_geometry_comparison_rows(
    anchor_metrics: dict[str, float],
    peeled_metrics: dict[str, float],
) -> list[dict[str, Any]]:
    rows = []
    for split in ("train", "A_train", "B_train", "A_C", "B_C", "C_all"):
        anchor = float(anchor_metrics.get(f"{split}_rmse_mm", float("nan")))
        peeled = float(peeled_metrics.get(f"{split}_rmse_mm", float("nan")))
        rows.append(
            {
                "split": split,
                "anchor_geometry_rmse_mm": anchor,
                "fine_tuned_geometry_only_rmse_mm": peeled,
                "delta_mm_positive_is_worse": float(peeled - anchor),
                "improved": bool(peeled <= anchor),
            }
        )
    return rows


def render_bayesian_html_report(report: dict[str, Any]) -> str:
    """Render the focused Bayesian mainline report."""
    summary = report["final_summary"]
    split = report["settings"]["abc_split"]
    stage1 = report["stage1_geometry"]
    bayes = report["bayesian_residual"]
    basis = bayes["models"]["bayesian_basis"]
    final_model = bayes["final_statistical_model"]
    validation = report["validation"]
    target_class = "ok" if summary["bayesian_finetuned_C_all_rmse_mm"] <= 0.06 else "warn"
    peeled_class = "ok" if summary["peeled_geometry_improved"] else "bad"
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>Bayesian Calibration Mainline Report</title>
  <style>
    body {{ font-family: Arial, "Microsoft YaHei", sans-serif; margin: 28px; color: #1f2933; line-height: 1.55; }}
    h1, h2, h3 {{ color: #102a43; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; font-size: 13px; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 7px 9px; text-align: left; vertical-align: top; }}
    th {{ background: #f0f4f8; }}
    code {{ background: #f0f4f8; padding: 1px 4px; border-radius: 3px; }}
    .ok {{ color: #0b6b3a; font-weight: 700; }}
    .warn {{ color: #9a6700; font-weight: 700; }}
    .bad {{ color: #b42318; font-weight: 700; }}
    .note {{ background: #f8fafc; border-left: 4px solid #486581; padding: 10px 14px; margin: 12px 0 20px; }}
  </style>
</head>
<body>
  <h1>璐濆彾鏂潪鍑犱綍鏍囧畾涓荤嚎鎶ュ憡</h1>
  <div class="note">
    涓荤嚎鍥哄畾涓?<b>鍑犱綍姝ｅ垯鍖栭敋鐐?+ Bayesian basis 闈炲嚑浣曡鲸璇?+ 33 鍑犱綍鍙傛暟鍏ㄩ噺 MAP 寰皟</b>銆?    C 闆嗘槸 held-out validation锛屽苟鍙備笌 fine-tune 鎺ュ彈鍒ゆ柇锛涘畠涓嶆槸鏈€缁堝畬鍏ㄧ嫭绔?test銆?  </div>

  <h2>鏍稿績缁撹</h2>
  <table>
    <tr><th>椤圭洰</th><th>缁撴灉</th></tr>
    <tr><td>A/B/C split</td><td>A_full={split['A_full_samples']}, B_full={split['B_full_samples']}, A_train={split['A_train_samples']}, B_train={split['B_train_samples']}, A_C={split['A_C_samples']}, B_C={split['B_C_samples']}</td></tr>
    <tr><td>Stage 1 鍑犱綍閿氱偣</td><td>{escape(summary['stage1_anchor_method'])} / {escape(summary['stage1_anchor_label'])}</td></tr>
    <tr><td>鍑犱綍閿氱偣 C_all RMSE</td><td>{summary['stage1_anchor_C_all_rmse_mm']:.6f} mm</td></tr>
    <tr><td>Bayesian 琛ュ伩鍚?C_all RMSE</td><td>{summary['bayesian_before_finetune_C_all_rmse_mm']:.6f} mm锛屾敼鍠?{summary['bayesian_gain_vs_anchor_C_all_mm']:.6f} mm</td></tr>
    <tr><td>Bayesian + 鍏?3寰皟 C_all RMSE</td><td><span class="{target_class}">{summary['bayesian_finetuned_C_all_rmse_mm']:.6f} mm</span>锛岀浉瀵瑰井璋冨墠鏀瑰杽 {summary['finetune_gain_vs_bayesian_C_all_mm']:.6f} mm</td></tr>
    <tr><td>鏈€缁堟帴鍙楁ā鍨?/td><td>{escape(summary.get('final_accepted_method', 'unknown'))}; C_all RMSE={summary.get('final_accepted_C_all_rmse_mm', float('nan')):.6f} mm</td></tr>
    <tr><td>鍓ョ琛ュ伩鍚庣殑鍑犱綍鏈綋</td><td><span class="{peeled_class}">{summary['peeled_geometry_C_all_rmse_mm']:.6f} mm</span>锛岀浉瀵归敋鐐?delta={summary['peeled_geometry_delta_vs_anchor_C_all_mm']:.6f} mm</td></tr>
    <tr><td>寰皟鎺ュ彈鐘舵€?/td><td>accepted_by_C_all={escape(str(final_model['accepted_by_C_all']))}; objective_decreased={escape(str(final_model['objective_decreased']))}</td></tr>
    <tr><td>鏈€缁堥€変腑 basis groups</td><td>{escape(', '.join(summary['selected_basis_groups']) or 'none')}</td></tr>
  </table>

  <h2>鏁板涓荤嚎</h2>
  <p>瑙傛祴妯″瀷鍐欎负 <code>y_k = p(q_k; theta) + Phi(x_k) beta + eps_k</code>锛?  鍏朵腑 <code>p(q_k; theta)</code> 鏄?33 缁村嚑浣曞€欓€夊弬鏁扮殑姝ｈ繍鍔ㄥ锛?  <code>Phi(x_k) beta</code> 鏄彈鎺?Bayesian basis 闈炲嚑浣曟畫宸紝
  <code>eps_k ~ N(0, sigma^2 I)</code> 涓?<code>sigma=0.06 mm</code>銆?  Stage 1 鍏堝緱鍒板嚑浣曢敋鐐?<code>theta_anchor</code>锛汼tage 3 鍥哄畾閿氱偣鎷熷悎
  <code>beta</code>锛汼tage 4 鍚屾椂寰皟鍏ㄩ儴 33 涓嚑浣曞弬鏁板拰 <code>beta</code>锛?/p>
  <p><code>min ||y - p(q;theta) - Phi beta||^2 + lambda_theta ||D^-1(theta-theta_anchor)||^2 + beta^T Lambda beta</code></p>

  <h2>Stage 1: 鍑犱綍閿氱偣娑堣瀺</h2>
  <p>涓嬭〃缁欏嚭 M0/M6/W3/S1/D1 鍦?A_train銆丅_train銆丄_C銆丅_C銆丆_all 涓婄殑鍑犱綍-only 璇樊銆?/p>
  {html_table(stage1['split_metrics'], ['method', 'label', 'lambda', 'active_count', 'A_train_rmse_mm', 'B_train_rmse_mm', 'A_C_rmse_mm', 'B_C_rmse_mm', 'C_all_rmse_mm'])}

  <h2>Stage 2: Bayesian 鍊欓€夊嚱鏁板簱</h2>
  <p>杩欎簺鍑芥暟鏄彈鎺х粺璁℃畫宸ā鍨嬬殑鍊欓€夌粍銆傚畠浠瘮鏃犻檺鍒?GPR 鏇村急锛屽洜涓烘瘡缁勫嚱鏁板彲瑙ｉ噴銆佸垪鏁版湁闄愶紝骞跺彈 Gaussian prior 绾︽潫銆?/p>
  {html_table(bayes.get('candidate_library', []), None)}

  <h2>Stage 3: Bayesian basis 鍖归厤涓庨€夋嫨</h2>
  <p>閫夋嫨閫昏緫鏄?blocked CV 浼樺厛锛岃姹傚€欓€夌粍甯︽潵姝?CV gain 涓?effective DOF 涓嶈秴杩囬槇鍊硷紱
  C_all 鍙敤浜庨獙璇佸拰鎶ュ憡锛屼笉鐢ㄤ簬鐩存帴鎷熷悎绯绘暟銆?/p>
  <h3>鍊欓€夋悳绱?/h3>
  {html_table(basis.get('search_rows', []), ['step', 'group', 'formula', 'columns', 'prior_std_mm', 'cv_rmse_mm', 'cv_gain_mm', 'train_rmse_mm', 'C_all_rmse_mm', 'train_C_gap_mm', 'effective_dof', 'accepted', 'decision'])}
  <h3>鏈€缁堥€変腑鍑芥暟</h3>
  {html_table(basis.get('selected_groups', []), ['name', 'label', 'formula', 'columns', 'prior_std_mm'])}
  <h3>Top coefficients</h3>
  {html_table(basis.get('top_coefficients', []), ['feature', 'coef_x_mm', 'coef_y_mm', 'coef_z_mm', 'coef_l2_mm', 'posterior_std_l2_mm'])}

  <h2>Stage 4: A/B/C 绮惧害瀵规瘮</h2>
  <p>鍥涜鍒嗗埆鏄嚑浣曢敋鐐广€丅ayesian 琛ュ伩銆丅ayesian+鍏?3寰皟銆佷互鍙婂墺绂昏ˉ鍋垮悗鐨勫嚑浣曟湰浣撱€?/p>
  {html_table(report['bayesian_stage_metrics'], None)}

  <h2>Stage 5: 寰皟鍚庣殑鍑犱綍鏈綋澶嶈瘎</h2>
  <p>鏈〃涓嶅姞 <code>Phi beta</code>锛屽彧璇勪及 <code>p(q; theta_finetuned)</code>銆?  鑻?delta 涓烘锛岃鏄庡嚑浣曟湰浣撶浉瀵?Stage 1 閿氱偣鍙樺樊锛涜嫢涓鸿礋锛岃鏄庡嚑浣曟湰浣撴敼鍠勩€?/p>
  {html_table(report['fine_tuned_geometry_only']['comparison_rows'], None)}

  <h2>楠岃瘉妫€鏌?/h2>
  <h3>鏁版嵁瀛楁</h3>
  {html_table(validation.get('data_checks', []), None)}
  <h3>鍚堟垚妫€鏌?/h3>
  {html_table([validation.get('synthetic_statistical_check', {})], None)}
  <h3>鐧藉寲涓庢畫鐣欑粨鏋?/h3>
  {html_table(whiteness_rows(bayes.get('whiteness', {})), None)}

  <h2>缁撹杈圭晫</h2>
  <ul>
    <li>鎶ュ憡涓殑 C 鏄?held-out validation锛岀敤浜庡垽鏂綋鍓嶇湡瀹炴暟鎹?split 涓嬬殑娉涘寲锛屼笉搴斿啓鎴愭渶缁?untouched test銆?/li>
    <li>Bayesian basis 鐨勬彁鍗囪〃绀哄彈鎺х粺璁℃畫宸兘瑙ｉ噴閿氱偣鍚庣殑缁撴瀯锛涘畠涓嶆槸鏂扮殑鐗╃悊杩炴潌鍙傛暟銆?/li>
    <li>鍓ョ琛ュ伩澶嶈瘎鐢ㄤ簬鍒ゆ柇寰皟鍚庣殑鍑犱綍鍚戦噺鑷韩鏄惁鏇村ソ锛岄伩鍏嶆妸闈炲嚑浣曡ˉ鍋挎敹鐩婅瑙ｆ垚鍑犱綍鍙傛暟鏀剁泭銆?/li>
  </ul>
</body>
</html>"""


def html_table(rows: list[dict[str, Any]], keys: list[str] | None) -> str:
    if not rows:
        return "<p><em>鏃犳暟鎹?/em></p>"
    if keys is None:
        keys = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
    header = "".join(f"<th>{escape(str(key))}</th>" for key in keys)
    body = []
    for row in rows:
        cells = "".join(f"<td>{escape(format_cell(row.get(key)))}</td>" for key in keys)
        body.append(f"<tr>{cells}</tr>")
    return f"<table><tr>{header}</tr>{''.join(body)}</table>"


def format_cell(value: Any) -> str:
    if isinstance(value, float):
        if not np.isfinite(value):
            return "nan"
        return f"{value:.6g}"
    if isinstance(value, (list, tuple)):
        return ", ".join(format_cell(item) for item in value)
    if isinstance(value, dict):
        return "; ".join(f"{key}={format_cell(val)}" for key, val in value.items())
    return str(value)


def whiteness_rows(whiteness: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for name, payload in whiteness.items():
        if not isinstance(payload, dict):
            continue
        ljung = payload.get("ljung_box", {})
        spectral = payload.get("spectral", {})
        partial = payload.get("partial_correlation", {})
        if not isinstance(ljung, dict):
            ljung = {}
        if not isinstance(spectral, dict):
            spectral = {}
        if not isinstance(partial, dict):
            partial = {}
        rows.append(
            {
                "scope": name,
                "passed": payload.get("passed"),
                "ljung_box_passed": ljung.get("passed"),
                "spectral_passed": spectral.get("passed"),
                "partial_correlation_passed": partial.get("passed"),
            }
        )
    return rows


def render_bayesian_html_report(report: dict[str, Any]) -> str:
    """Render the current Bayesian mainline report.

    This later definition intentionally supersedes the older exploratory
    renderer above so generated reports stay readable after the mainline
    uses an unscaled geometry objective with lambda-ratio controlled fine-tune.
    """
    summary = report["final_summary"]
    split = report["settings"]["abc_split"]
    stage1 = report["stage1_geometry"]
    bayes = report["bayesian_residual"]
    basis = bayes["models"]["bayesian_basis"]
    final_model = bayes["final_statistical_model"]
    validation = report["validation"]
    search_rows = bayes.get("fine_tune", {}).get("search_rows", [])
    target_class = "ok" if summary["bayesian_finetuned_C_all_rmse_mm"] <= 0.06 else "warn"
    peeled_class = "ok" if summary["peeled_geometry_improved"] else "bad"
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>Bayesian Calibration Mainline Report</title>
  <style>
    body {{ font-family: Arial, "Microsoft YaHei", sans-serif; margin: 28px; color: #1f2933; line-height: 1.55; }}
    h1, h2, h3 {{ color: #102a43; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; font-size: 13px; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 7px 9px; text-align: left; vertical-align: top; }}
    th {{ background: #f0f4f8; }}
    code {{ background: #f0f4f8; padding: 1px 4px; border-radius: 3px; }}
    .ok {{ color: #0b6b3a; font-weight: 700; }}
    .warn {{ color: #9a6700; font-weight: 700; }}
    .bad {{ color: #b42318; font-weight: 700; }}
    .note {{ background: #f8fafc; border-left: 4px solid #486581; padding: 10px 14px; margin: 12px 0 20px; }}
  </style>
</head>
<body>
  <h1>璐濆彾鏂潪鍑犱綍鏍囧畾涓荤嚎鎶ュ憡</h1>
  <div class="note">
    涓荤嚎涓?<b>鏈櫎 sigma 鐨勫嚑浣曟鍒欏寲閿氱偣 + Bayesian basis 闈炲嚑浣曡鲸璇?+ 33 鍑犱綍鍙傛暟鍙楁帶 MAP 寰皟</b>銆?    Stage 1 鍜?Stage 4 閮戒笉鎶婁綅缃畫宸櫎浠?sigma锛汣 闆嗘槸 held-out validation锛屼笉鏄渶缁?untouched test銆?  </div>

  <h2>鏍稿績缁撹</h2>
  <table>
    <tr><th>椤圭洰</th><th>缁撴灉</th></tr>
    <tr><td>A/B/C split</td><td>A_full={split['A_full_samples']}, B_full={split['B_full_samples']}, A_train={split['A_train_samples']}, B_train={split['B_train_samples']}, A_C={split['A_C_samples']}, B_C={split['B_C_samples']}</td></tr>
    <tr><td>Stage 1 鍑犱綍閿氱偣</td><td>{escape(summary['stage1_anchor_method'])} / {escape(summary['stage1_anchor_label'])}</td></tr>
    <tr><td>Stage 1 閿氱偣 lambda</td><td>{summary['stage1_anchor_lambda']:.6g}</td></tr>
    <tr><td>Stage 4 鍥哄畾姣斾緥</td><td>ratio={summary['fine_tune_lambda_ratio']:.6g}; lambda_theta={summary['fine_tune_lambda_theta']:.6g}</td></tr>
    <tr><td>鍑犱綍閿氱偣 C_all RMSE</td><td>{summary['stage1_anchor_C_all_rmse_mm']:.6f} mm</td></tr>
    <tr><td>Bayesian 琛ュ伩鍚?C_all RMSE</td><td>{summary['bayesian_before_finetune_C_all_rmse_mm']:.6f} mm锛屾敼鍠?{summary['bayesian_gain_vs_anchor_C_all_mm']:.6f} mm</td></tr>
    <tr><td>Bayesian + 鍏?3寰皟 C_all RMSE</td><td><span class="{target_class}">{summary['bayesian_finetuned_C_all_rmse_mm']:.6f} mm</span>锛岀浉瀵瑰井璋冨墠鏀瑰杽 {summary['finetune_gain_vs_bayesian_C_all_mm']:.6f} mm</td></tr>
    <tr><td>鏈€缁堟帴鍙楁ā鍨?/td><td>{escape(summary.get('final_accepted_method', 'unknown'))}; C_all RMSE={summary.get('final_accepted_C_all_rmse_mm', float('nan')):.6f} mm</td></tr>
    <tr><td>鍓ョ琛ュ伩鍚庣殑鍑犱綍鏈綋</td><td><span class="{peeled_class}">{summary['peeled_geometry_C_all_rmse_mm']:.6f} mm</span>锛岀浉瀵归敋鐐?delta={summary['peeled_geometry_delta_vs_anchor_C_all_mm']:.6f} mm</td></tr>
    <tr><td>寰皟鎺ュ彈鐘舵€?/td><td>accepted_by_C_all={escape(str(final_model['accepted_by_C_all']))}; objective_decreased={escape(str(final_model['objective_decreased']))}</td></tr>
    <tr><td>鏈€缁堥€変腑 basis groups</td><td>{escape(', '.join(summary['selected_basis_groups']) or 'none')}</td></tr>
  </table>

  <h2>鏁板涓庝紭鍖栫洰鏍?/h2>
  <p>娴嬮噺妯″瀷涓?<code>y_k = p(q_k; theta) + Phi(x_k) beta + eps_k</code>锛屽叾涓?<code>eps_k ~ N(0, sigma^2 I)</code>锛岄粯璁?<code>sigma=0.06 mm</code>銆?  Stage 1 鍑犱綍閿氱偣涓?Stage 4 鑱斿悎寰皟閮戒娇鐢ㄦ湭闄?sigma 鐨勪綅缃畫宸紱sigma 鍙綔涓虹粺璁℃畫宸拰鍙楁帶闃堝€肩殑灏哄害鍙傝€冿細</p>
  <p><code>min ||y - p(q;theta) - Phi beta||^2 + lambda_theta ||D^-1(theta-theta_anchor)||^2 + beta^T Lambda beta</code></p>
  <p>Stage 4 浣跨敤 <code>lambda_theta = lambda_stage1 * ratio</code>銆傛湰杞悳绱?ratio 鍚庯紝灏?ratio 鍥哄寲涓哄悗缁粯璁ゆ瘮渚嬶紝浣夸笅涓€杞彲闅?Stage 1 鐨?lambda 鍔ㄦ€佽皟鏁淬€?/p>

  <h2>Stage 1: 鍑犱綍閿氱偣娑堣瀺</h2>
  <p>M0/M6/W3/S1/D1 鍧囧湪鏈櫎 sigma 鐨勪綅缃畫宸洰鏍囦笅杩愯锛孲tage 1 閿氱偣鍥哄畾涓?S1銆?/p>
  {html_table(stage1['split_metrics'], ['method', 'label', 'lambda', 'active_count', 'A_train_rmse_mm', 'B_train_rmse_mm', 'A_C_rmse_mm', 'B_C_rmse_mm', 'C_all_rmse_mm'])}

  <h2>Stage 2: Bayesian 鍊欓€夊嚱鏁板簱</h2>
  {html_table(bayes.get('candidate_library', []), None)}

  <h2>Stage 3: Bayesian basis 鍖归厤涓庨€夋嫨</h2>
  <p>閫夋嫨閫昏緫鏄?blocked CV 浼樺厛锛岃姹傚€欓€夌粍甯︽潵姝?CV gain 涓?effective DOF 涓嶈秴杩囬槇鍊硷紱C_all 鍙敤浜庨獙璇佸拰鎶ュ憡銆?/p>
  <h3>鍊欓€夋悳绱?/h3>
  {html_table(basis.get('search_rows', []), ['step', 'group', 'formula', 'columns', 'prior_std_mm', 'cv_rmse_mm', 'cv_gain_mm', 'train_rmse_mm', 'C_all_rmse_mm', 'train_C_gap_mm', 'effective_dof', 'accepted', 'decision'])}
  <h3>鏈€缁堥€変腑鍑芥暟</h3>
  {html_table(basis.get('selected_groups', []), ['name', 'label', 'formula', 'columns', 'prior_std_mm'])}
  <h3>Top coefficients</h3>
  {html_table(basis.get('top_coefficients', []), ['feature', 'coef_x_mm', 'coef_y_mm', 'coef_z_mm', 'coef_l2_mm', 'posterior_std_l2_mm'])}

  <h2>Stage 4: 鍑犱綍寰皟姝ｅ垯姣斾緥鎼滅储</h2>
  <p>鍙楁帶閫夋嫨瑙勫垯锛氫紭鍏堥€夋嫨 <code>peeled C_all delta <= sqrt(3)*sigma</code> 涓?<code>theta_update_scaled_l2 <= 5</code> 鐨勫€欓€夛紱鍦ㄥ彈鎺ч泦鍚堝唴閫?joint model C_all 鏈€浣庛€傝嫢鏃犲€欓€夋弧瓒冲彈鎺ф潯浠讹紝鍒欓€夊墺绂诲嚑浣曟伓鍖栨渶灏忕殑鍊欓€夈€?/p>
  {html_table(search_rows, ['ratio_to_stage1_lambda', 'lambda_theta', 'post_A_train_rmse_mm', 'post_B_train_rmse_mm', 'post_A_C_rmse_mm', 'post_B_C_rmse_mm', 'post_C_all_rmse_mm', 'peeled_C_all_rmse_mm', 'peeled_C_all_delta_vs_anchor_mm', 'allowed_peeled_delta_mm', 'theta_update_scaled_l2', 'controlled_geometry'])}

  <h2>Stage 5: A/B/C 绮惧害瀵规瘮</h2>
  {html_table(report['bayesian_stage_metrics'], None)}

  <h2>Stage 6: 寰皟鍚庣殑鍑犱綍鏈綋澶嶈瘎</h2>
  {html_table(report['fine_tuned_geometry_only']['comparison_rows'], None)}

  <h2>楠岃瘉妫€鏌?/h2>
  <h3>鏁版嵁瀛楁</h3>
  {html_table(validation.get('data_checks', []), None)}
  <h3>鍚堟垚妫€鏌?/h3>
  {html_table([validation.get('synthetic_statistical_check', {})], None)}
  <h3>鐧藉寲涓庢畫鐣欑粨鏋?/h3>
  {html_table(whiteness_rows(bayes.get('whiteness', {})), None)}

  <h2>缁撹杈圭晫</h2>
  <ul>
    <li>C 鏄?held-out validation锛屽苟鍙備笌鍙楁帶绛栫暐閫夋嫨锛屼笉搴斿啓鎴愭渶缁?untouched test銆?/li>
    <li>Bayesian basis 鐨勬敹鐩婅〃绀哄彈鎺х粺璁℃畫宸В閲婁簡鍑犱綍閿氱偣鍚庣殑缁撴瀯锛屼笉鏄柊鐨勭墿鐞嗚繛鏉嗗弬鏁般€?/li>
    <li>鍓ョ琛ュ伩澶嶈瘎鐢ㄤ簬鍒ゆ柇鍑犱綍鏈綋鏄惁浠嶅彈鎺э紱涓嶈兘鎶婇潪鍑犱綍琛ュ伩鏀剁泭璇В鎴愬嚑浣曞弬鏁版敹鐩娿€?/li>
  </ul>
</body>
</html>"""


__all__ = [
    "json_ready",
    "render_bayesian_html_report",
    "run_bayesian_calibration_analysis",
]

