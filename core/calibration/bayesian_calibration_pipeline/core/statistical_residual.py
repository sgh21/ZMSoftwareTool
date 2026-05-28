"""Controlled statistical residual models for non-geometric calibration.

The models in this module are deliberately less expressive than an
unrestricted exact GPR.  They expose explicit basis groups, fixed noise floors,
effective degrees of freedom, and train/C gaps so the report can explain what
structure was fitted and whether it is likely to be a local overfit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.cluster import KMeans


@dataclass(frozen=True)
class StatisticalResidualConfig:
    """Settings for the Bayesian basis residual model."""

    noise_std_m: float = 6.0e-5
    cv_folds: int = 4
    seed: int = 20260524
    harmonics: tuple[int, ...] = (1, 2, 3, 4)
    prior_std_mm_grid: tuple[float, ...] = (0.05, 0.10, 0.20, 0.50)
    max_basis_groups: int = 4
    max_effective_dof_fraction: float = 0.55
    rbf_center_count: int = 8
    local_cluster_count: int = 4
    min_cv_gain_mm: float = 0.001

    def quickened(self) -> "StatisticalResidualConfig":
        return StatisticalResidualConfig(
            noise_std_m=self.noise_std_m,
            cv_folds=min(self.cv_folds, 3),
            seed=self.seed,
            harmonics=tuple(h for h in self.harmonics if h <= 2),
            prior_std_mm_grid=(0.10, 0.30),
            max_basis_groups=min(self.max_basis_groups, 2),
            max_effective_dof_fraction=self.max_effective_dof_fraction,
            rbf_center_count=min(self.rbf_center_count, 4),
            local_cluster_count=min(self.local_cluster_count, 3),
            min_cv_gain_mm=self.min_cv_gain_mm,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "noise_std_m": float(self.noise_std_m),
            "noise_std_mm": float(self.noise_std_m * 1000.0),
            "cv_folds": int(self.cv_folds),
            "seed": int(self.seed),
            "harmonics": list(self.harmonics),
            "prior_std_mm_grid": list(self.prior_std_mm_grid),
            "max_basis_groups": int(self.max_basis_groups),
            "max_effective_dof_fraction": float(self.max_effective_dof_fraction),
            "rbf_center_count": int(self.rbf_center_count),
            "local_cluster_count": int(self.local_cluster_count),
            "min_cv_gain_mm": float(self.min_cv_gain_mm),
        }


@dataclass
class DatasetView:
    """Features and residual target for one split."""

    name: str
    dataset: dict[str, np.ndarray]
    geometry_positions: np.ndarray
    measured_positions: np.ndarray
    residuals: np.ndarray
    workspace: str


@dataclass
class BasisGroup:
    """One interpretable basis-function group."""

    name: str
    label: str
    formula: str
    prior_note: str
    train_matrix: np.ndarray
    eval_matrices: dict[str, np.ndarray]
    column_names: list[str]


@dataclass
class BayesianLinearFit:
    """MAP fit for a fixed design and diagonal Gaussian prior."""

    coefficients: np.ndarray
    posterior_std: np.ndarray
    prior_std_by_column: np.ndarray
    effective_dof: float
    evidence_score: float
    train_prediction: np.ndarray
    train_rmse_mm: float
    train_residual_rmse_mm: float


@dataclass
class StatisticalModelResult:
    """Report-ready controlled statistical model result."""

    method: str
    label: str
    selected: bool
    accepted: bool
    reason: str
    train_prediction: np.ndarray
    eval_predictions: dict[str, np.ndarray]
    metrics: dict[str, float]
    search_rows: list[dict[str, Any]]
    selected_groups: list[dict[str, Any]]
    top_coefficients: list[dict[str, Any]]
    model_info: dict[str, Any]
    design_train: np.ndarray
    design_eval: dict[str, np.ndarray]
    coefficients: np.ndarray
    prior_std_by_column: np.ndarray
    feature_names: list[str]




def run_bayesian_basis_residual_model(
    train_view: DatasetView,
    eval_views: dict[str, DatasetView],
    config: StatisticalResidualConfig,
    *,
    quick: bool = False,
    selection_view_name: str | None = None,
) -> dict[str, Any]:
    """Fit only the interpretable Bayesian basis residual model."""
    cfg = config.quickened() if quick else config
    folds = blocked_folds(train_view.dataset, cfg.cv_folds, cfg.seed)
    groups = build_basis_groups(train_view, eval_views, cfg)
    if selection_view_name is not None:
        raise ValueError("selection_view_name is no longer supported in the Bayesian mainline.")
    basis = fit_bayesian_basis_model(train_view, eval_views, groups, folds, cfg)
    cv_protocol = {
        "folds": int(len(folds)),
        "fold_sizes": [int(np.sum(mask)) for mask in folds],
        "protocol": (
            f"blocked folds inside {train_view.name}; C splits are not used to fit coefficients"
        ),
    }
    selection_rule = "Bayesian basis is the fixed mainline residual model"
    synthetic = run_synthetic_statistical_checks(cfg)
    return {
        "settings": cfg.as_dict(),
        "feature_inputs": [
            "q",
            "sin/cos(h*q)",
            "directions",
            "joint_torques",
            "geometry predicted position p_geo(q)",
            "workspace-standardized xyz z",
        ],
        "noise_prior": {
            "tracker_noise_std_m": float(cfg.noise_std_m),
            "tracker_noise_std_mm": float(cfg.noise_std_m * 1000.0),
            "interpretation": "minimum measurement-noise floor used in the Bayesian MAP objective",
        },
        "candidate_library": candidate_library_rows(cfg),
        "cv_protocol": cv_protocol,
        "synthetic_validation": synthetic,
        "models": {
            "bayesian_basis": model_to_report(basis),
        },
        "model_comparison": [method_summary_row(basis)],
        "final_statistical_model": {
            "method": basis.method,
            "label": basis.label,
            "C_all_rmse_mm": float(basis.metrics.get("C_all_rmse_mm", float("nan"))),
            "train_rmse_mm": float(basis.metrics.get("train_rmse_mm", float("nan"))),
            "selection_rule": selection_rule,
        },
        "_raw_model": basis,
    }


def fit_bayesian_basis_model(
    train_view: DatasetView,
    eval_views: dict[str, DatasetView],
    groups: list[BasisGroup],
    folds: list[np.ndarray],
    config: StatisticalResidualConfig,
) -> StatisticalModelResult:
    """Forward-select interpretable basis groups with blocked CV."""
    y = train_view.residuals
    baseline_cv = cv_rmse_for_design(np.zeros((len(y), 0)), y, np.zeros(0), folds, config)
    selected: list[tuple[BasisGroup, float]] = []
    selected_names: set[str] = set()
    search_rows: list[dict[str, Any]] = []
    current_cv = baseline_cv
    max_dof = max(1.0, float(config.max_effective_dof_fraction) * len(y))
    for step in range(int(config.max_basis_groups)):
        candidate_payloads: list[tuple[float, BasisGroup, float, BayesianLinearFit, dict[str, Any]]] = []
        for group in groups:
            if group.name in selected_names:
                continue
            for prior_std_mm in config.prior_std_mm_grid:
                trial_groups = selected + [(group, float(prior_std_mm) * 1.0e-3)]
                design, priors, names = assemble_basis_design(trial_groups)
                cv_rmse = cv_rmse_for_design(design, y, priors, folds, config)
                fit = fit_bayesian_linear(design, y, priors, config.noise_std_m)
                metrics = evaluate_prediction_metrics(
                    y,
                    fit.train_prediction,
                    prefix="train",
                )
                c_metrics = evaluate_model_on_views(
                    fit.coefficients,
                    trial_groups,
                    eval_views,
                    config,
                )
                cv_gain = current_cv - cv_rmse
                row = {
                    "step": int(step + 1),
                    "group": group.name,
                    "label": group.label,
                    "formula": group.formula,
                    "columns": int(group.train_matrix.shape[1]),
                    "prior_std_mm": float(prior_std_mm),
                    "cv_rmse_mm": float(cv_rmse),
                    "cv_gain_mm": float(cv_gain),
                    "train_rmse_mm": float(metrics["train_rmse_mm"]),
                    "C_all_rmse_mm": float(c_metrics.get("C_all_rmse_mm", float("nan"))),
                    "train_C_gap_mm": float(
                        c_metrics.get("C_all_rmse_mm", float("nan"))
                        - metrics["train_rmse_mm"]
                    ),
                    "effective_dof": float(fit.effective_dof),
                    "evidence_score": float(fit.evidence_score),
                    "accepted": False,
                    "decision": "candidate",
                }
                if cv_gain > float(config.min_cv_gain_mm) and fit.effective_dof <= max_dof:
                    candidate_payloads.append((cv_gain, group, float(prior_std_mm) * 1.0e-3, fit, row))
                else:
                    row["decision"] = (
                        "reject_cv_gain_or_effective_dof"
                        if fit.effective_dof <= max_dof
                        else "reject_effective_dof"
                    )
                search_rows.append(row)
        if not candidate_payloads:
            break
        candidate_payloads.sort(key=lambda item: (-item[0], item[1].name, item[2]))
        cv_gain, group, prior_std_m, _, row = candidate_payloads[0]
        row["accepted"] = True
        row["decision"] = "accept"
        selected.append((group, prior_std_m))
        selected_names.add(group.name)
        current_cv = current_cv - cv_gain
    if selected:
        design, priors, feature_names = assemble_basis_design(selected)
        fit = fit_bayesian_linear(design, y, priors, config.noise_std_m)
        eval_predictions = {
            name: assemble_eval_design(selected, view_name) @ fit.coefficients
            for name, view_name in ((key, key) for key in eval_views)
        }
        reason = "selected_by_blocked_cv"
        accepted = True
    else:
        design = np.zeros((len(y), 0), dtype=float)
        priors = np.zeros(0, dtype=float)
        feature_names = []
        fit = fit_bayesian_linear(design, y, priors, config.noise_std_m)
        eval_predictions = {name: np.zeros_like(view.residuals) for name, view in eval_views.items()}
        reason = "no_basis_group_passed_blocked_cv"
        accepted = False
    metrics = model_metrics(train_view, eval_views, fit.train_prediction, eval_predictions)
    selected_rows = [
        {
            "name": group.name,
            "label": group.label,
            "formula": group.formula,
            "columns": int(group.train_matrix.shape[1]),
            "prior_std_mm": float(prior_std_m * 1000.0),
        }
        for group, prior_std_m in selected
    ]
    return StatisticalModelResult(
        method="bayesian_basis",
        label="Bayesian interpretable basis residual model",
        selected=bool(selected),
        accepted=accepted,
        reason=reason,
        train_prediction=fit.train_prediction,
        eval_predictions=eval_predictions,
        metrics=metrics,
        search_rows=search_rows,
        selected_groups=selected_rows,
        top_coefficients=top_coefficients(feature_names, fit.coefficients, fit.posterior_std),
        model_info={
            "baseline_cv_rmse_mm": float(baseline_cv),
            "final_cv_rmse_mm": float(current_cv),
            "effective_dof": float(fit.effective_dof),
            "noise_std_mm": float(config.noise_std_m * 1000.0),
            "basis_count": int(len(feature_names)),
        },
        design_train=design,
        design_eval={name: assemble_eval_design(selected, name) for name in eval_views},
        coefficients=fit.coefficients,
        prior_std_by_column=priors,
        feature_names=feature_names,
    )






def fit_bayesian_linear(
    design: np.ndarray,
    residuals: np.ndarray,
    prior_std_by_column: np.ndarray,
    noise_std: float,
) -> BayesianLinearFit:
    """Fit multi-output Bayesian ridge with fixed diagonal priors."""
    x = np.asarray(design, dtype=float)
    y = np.asarray(residuals, dtype=float).reshape(-1, 3)
    if x.shape[0] != y.shape[0]:
        raise ValueError("design and residual sample counts must match.")
    if x.shape[1] == 0:
        pred = np.zeros_like(y)
        residual = y - pred
        return BayesianLinearFit(
            coefficients=np.zeros((0, 3), dtype=float),
            posterior_std=np.zeros((0, 3), dtype=float),
            prior_std_by_column=np.zeros(0, dtype=float),
            effective_dof=0.0,
            evidence_score=-0.5 * float(np.sum(residual * residual) / max(noise_std * noise_std, 1.0e-30)),
            train_prediction=pred,
            train_rmse_mm=euclidean_rmse_mm(y, pred),
            train_residual_rmse_mm=euclidean_rmse_mm(y, pred),
        )
    priors = np.asarray(prior_std_by_column, dtype=float).reshape(-1)
    if len(priors) != x.shape[1]:
        raise ValueError("prior length must match design columns.")
    sigma2 = max(float(noise_std) ** 2, 1.0e-30)
    precision = 1.0 / np.maximum(priors * priors, 1.0e-30)
    normal = (x.T @ x) / sigma2 + np.diag(precision)
    rhs = (x.T @ y) / sigma2
    coef = np.linalg.solve(normal, rhs)
    pred = x @ coef
    residual = y - pred
    inv_normal = np.linalg.pinv(normal)
    posterior_std = np.sqrt(np.maximum(np.diag(inv_normal), 0.0)).reshape(-1, 1)
    posterior_std = np.repeat(posterior_std, 3, axis=1)
    smoother = x @ np.linalg.solve(normal, x.T / sigma2)
    effective_dof = float(np.trace(smoother))
    penalty = float(np.sum((coef / np.maximum(priors.reshape(-1, 1), 1.0e-30)) ** 2))
    logdet = float(np.linalg.slogdet(normal)[1])
    evidence = -0.5 * (float(np.sum(residual * residual) / sigma2) + penalty + 3.0 * logdet)
    return BayesianLinearFit(
        coefficients=coef,
        posterior_std=posterior_std,
        prior_std_by_column=priors,
        effective_dof=effective_dof,
        evidence_score=evidence,
        train_prediction=pred,
        train_rmse_mm=euclidean_rmse_mm(y, pred),
        train_residual_rmse_mm=euclidean_rmse_mm(y, pred),
    )


def cv_rmse_for_design(
    design: np.ndarray,
    residuals: np.ndarray,
    prior_std_by_column: np.ndarray,
    folds: list[np.ndarray],
    config: StatisticalResidualConfig,
) -> float:
    """Return blocked-CV Euclidean residual RMSE in mm."""
    x = np.asarray(design, dtype=float)
    y = np.asarray(residuals, dtype=float).reshape(-1, 3)
    errors = []
    for holdout in folds:
        holdout = np.asarray(holdout, dtype=bool).reshape(-1)
        fit_mask = ~holdout
        if not np.any(holdout) or not np.any(fit_mask):
            continue
        fit = fit_bayesian_linear(
            x[fit_mask],
            y[fit_mask],
            np.asarray(prior_std_by_column, dtype=float),
            config.noise_std_m,
        )
        pred = x[holdout] @ fit.coefficients if x.shape[1] else np.zeros((int(np.sum(holdout)), 3))
        errors.append(y[holdout] - pred)
    if not errors:
        return euclidean_rmse_mm(y, np.zeros_like(y))
    error = np.vstack(errors)
    return float(np.sqrt(np.mean(np.sum(error * error, axis=1))) * 1000.0)


def build_basis_groups(
    train_view: DatasetView,
    eval_views: dict[str, DatasetView],
    config: StatisticalResidualConfig,
) -> list[BasisGroup]:
    """Create train/eval matrices for the interpretable basis library."""
    context = BasisContext.fit(train_view, config)
    groups: list[BasisGroup] = []
    for harmonic in config.harmonics:
        groups.append(context.group_harmonic(train_view, eval_views, int(harmonic), "joint"))
        groups.append(context.group_harmonic(train_view, eval_views, int(harmonic), "torque"))
        groups.append(context.group_harmonic(train_view, eval_views, int(harmonic), "direction"))
    groups.append(context.group_polynomial_linear(train_view, eval_views))
    groups.append(context.group_polynomial_pairwise(train_view, eval_views))
    groups.append(context.group_workspace_rbf(train_view, eval_views, length_scale=1.0))
    groups.append(context.group_workspace_rbf(train_view, eval_views, length_scale=2.0))
    groups.append(context.group_local_affine(train_view, eval_views))
    return [group for group in groups if group.train_matrix.shape[1] > 0]


@dataclass
class BasisContext:
    """Fit-time transforms needed to build comparable basis matrices."""

    q_mean: np.ndarray
    q_std: np.ndarray
    tau_mean: np.ndarray
    tau_std: np.ndarray
    z_mean: np.ndarray
    z_std: np.ndarray
    rbf_centers: np.ndarray
    cluster_centers: np.ndarray

    @classmethod
    def fit(cls, view: DatasetView, config: StatisticalResidualConfig) -> "BasisContext":
        q = np.asarray(view.dataset["joints"], dtype=float).reshape(-1, 6)
        tau = _matrix_or_zeros(view.dataset, "joint_torques", 6)
        z = np.asarray(view.geometry_positions, dtype=float).reshape(-1, 3)
        z_scaled, z_mean, z_std = standardize_fit_transform(z)
        center_count = max(1, min(int(config.rbf_center_count), len(z_scaled)))
        cluster_count = max(1, min(int(config.local_cluster_count), len(z_scaled)))
        rbf_centers = kmeans_centers(z_scaled, center_count, config.seed + 31)
        cluster_centers = kmeans_centers(z_scaled, cluster_count, config.seed + 47)
        _, q_mean, q_std = standardize_fit_transform(q)
        _, tau_mean, tau_std = standardize_fit_transform(tau)
        return cls(
            q_mean=q_mean,
            q_std=q_std,
            tau_mean=tau_mean,
            tau_std=tau_std,
            z_mean=z_mean,
            z_std=z_std,
            rbf_centers=rbf_centers,
            cluster_centers=cluster_centers,
        )

    def scaled_q(self, view: DatasetView) -> np.ndarray:
        q = np.asarray(view.dataset["joints"], dtype=float).reshape(-1, 6)
        return (q - self.q_mean.reshape(1, -1)) / self.q_std.reshape(1, -1)

    def scaled_tau(self, view: DatasetView) -> np.ndarray:
        tau = _matrix_or_zeros(view.dataset, "joint_torques", 6)
        return (tau - self.tau_mean.reshape(1, -1)) / self.tau_std.reshape(1, -1)

    def scaled_z(self, view: DatasetView) -> np.ndarray:
        z = np.asarray(view.geometry_positions, dtype=float).reshape(-1, 3)
        return (z - self.z_mean.reshape(1, -1)) / self.z_std.reshape(1, -1)

    def group_harmonic(
        self,
        train_view: DatasetView,
        eval_views: dict[str, DatasetView],
        harmonic: int,
        kind: str,
    ) -> BasisGroup:
        if kind == "joint":
            label = "joint harmonic"
            formula = f"sin({harmonic} q_i), cos({harmonic} q_i)"
        elif kind == "torque":
            label = "torque harmonic"
            formula = f"tau_i sin({harmonic} q_i), tau_i cos({harmonic} q_i)"
        elif kind == "direction":
            label = "direction harmonic"
            formula = f"s_i sin({harmonic} q_i), s_i cos({harmonic} q_i)"
        else:
            raise ValueError(kind)

        def raw(view: DatasetView) -> tuple[np.ndarray, list[str]]:
            q = np.asarray(view.dataset["joints"], dtype=float).reshape(-1, 6)
            sin_q = np.sin(float(harmonic) * q)
            cos_q = np.cos(float(harmonic) * q)
            if kind == "joint":
                weights = np.ones_like(q)
                prefix = f"joint_h{harmonic}"
            elif kind == "torque":
                weights = self.scaled_tau(view)
                prefix = f"torque_h{harmonic}"
            elif kind == "direction":
                weights = _directions(view.dataset)
                prefix = f"direction_h{harmonic}"
            else:
                raise ValueError(kind)
            matrix = np.hstack([weights * sin_q, weights * cos_q])
            names = [f"{prefix}_sin_j{i + 1}" for i in range(6)] + [
                f"{prefix}_cos_j{i + 1}" for i in range(6)
            ]
            return matrix, names

        train_raw, names = raw(train_view)
        train_matrix, eval_matrices, kept_names = standardize_group(train_raw, names, {
            key: raw(view)[0] for key, view in eval_views.items()
        })
        return BasisGroup(
            name=f"{kind}_harmonic_h{harmonic}",
            label=f"{label} h={harmonic}",
            formula=formula,
            prior_note="Gaussian coefficient prior selected from prior_std_mm_grid",
            train_matrix=train_matrix,
            eval_matrices=eval_matrices,
            column_names=kept_names,
        )

    def group_polynomial_linear(self, train_view: DatasetView, eval_views: dict[str, DatasetView]) -> BasisGroup:
        names = [f"q_linear_j{i + 1}" for i in range(6)]
        train_raw = self.scaled_q(train_view)
        eval_raw = {key: self.scaled_q(view) for key, view in eval_views.items()}
        train_matrix, eval_matrices, kept = standardize_group(train_raw, names, eval_raw)
        return BasisGroup(
            name="polynomial_linear_q",
            label="low-order polynomial linear q",
            formula="centered q_i",
            prior_note="Gaussian coefficient prior selected from prior_std_mm_grid",
            train_matrix=train_matrix,
            eval_matrices=eval_matrices,
            column_names=kept,
        )

    def group_polynomial_pairwise(self, train_view: DatasetView, eval_views: dict[str, DatasetView]) -> BasisGroup:
        pairs = [(i, j) for i in range(6) for j in range(i, 6)]
        names = [f"q_pair_j{i + 1}_j{j + 1}" for i, j in pairs]

        def pairwise(view: DatasetView) -> np.ndarray:
            q = self.scaled_q(view)
            return np.column_stack([q[:, i] * q[:, j] for i, j in pairs])

        train_matrix, eval_matrices, kept = standardize_group(
            pairwise(train_view),
            names,
            {key: pairwise(view) for key, view in eval_views.items()},
        )
        return BasisGroup(
            name="polynomial_pairwise_q",
            label="low-order polynomial pairwise q",
            formula="centered q_i q_j, i<=j",
            prior_note="Gaussian coefficient prior selected from prior_std_mm_grid",
            train_matrix=train_matrix,
            eval_matrices=eval_matrices,
            column_names=kept,
        )

    def group_workspace_rbf(
        self,
        train_view: DatasetView,
        eval_views: dict[str, DatasetView],
        *,
        length_scale: float,
    ) -> BasisGroup:
        names = [f"workspace_rbf_l{length_scale:g}_c{i + 1}" for i in range(len(self.rbf_centers))]

        def rbf(view: DatasetView) -> np.ndarray:
            z = self.scaled_z(view)
            d2 = pairwise_squared(z, self.rbf_centers)
            return np.exp(-d2 / (2.0 * float(length_scale) ** 2))

        train_matrix, eval_matrices, kept = standardize_group(
            rbf(train_view),
            names,
            {key: rbf(view) for key, view in eval_views.items()},
        )
        return BasisGroup(
            name=f"workspace_rbf_l{length_scale:g}",
            label=f"workspace RBF length={length_scale:g}",
            formula=f"exp(-||z-c_m||^2/(2*{length_scale:g}^2))",
            prior_note="Gaussian coefficient prior selected from prior_std_mm_grid",
            train_matrix=train_matrix,
            eval_matrices=eval_matrices,
            column_names=kept,
        )

    def group_local_affine(self, train_view: DatasetView, eval_views: dict[str, DatasetView]) -> BasisGroup:
        k = len(self.cluster_centers)
        names = []
        for cluster in range(k):
            names.extend([f"cluster{cluster + 1}_bias", f"cluster{cluster + 1}_zx", f"cluster{cluster + 1}_zy", f"cluster{cluster + 1}_zz"])

        def affine(view: DatasetView) -> np.ndarray:
            z = self.scaled_z(view)
            labels = np.argmin(pairwise_squared(z, self.cluster_centers), axis=1)
            blocks = []
            base = np.column_stack([np.ones(len(z), dtype=float), z])
            for cluster in range(k):
                blocks.append(base * (labels == cluster).reshape(-1, 1))
            return np.hstack(blocks)

        train_matrix, eval_matrices, kept = standardize_group(
            affine(train_view),
            names,
            {key: affine(view) for key, view in eval_views.items()},
        )
        return BasisGroup(
            name="local_cluster_affine",
            label="local cluster affine workspace residual",
            formula="1[c(k)=m] * [1, z_x, z_y, z_z]",
            prior_note="Gaussian coefficient prior selected from prior_std_mm_grid",
            train_matrix=train_matrix,
            eval_matrices=eval_matrices,
            column_names=kept,
        )


def assemble_basis_design(selected: list[tuple[BasisGroup, float]]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if not selected:
        return np.zeros((0, 0), dtype=float), np.zeros(0, dtype=float), []
    matrices = [group.train_matrix for group, _ in selected]
    design = np.column_stack(matrices)
    priors = np.concatenate([
        np.full(group.train_matrix.shape[1], float(prior), dtype=float)
        for group, prior in selected
    ])
    names: list[str] = []
    for group, _ in selected:
        names.extend(group.column_names)
    return design, priors, names


def assemble_eval_design(selected: list[tuple[BasisGroup, float]], eval_name: str) -> np.ndarray:
    if not selected:
        return np.zeros((0, 0), dtype=float)
    matrices = [group.eval_matrices[eval_name] for group, _ in selected]
    return np.column_stack(matrices)

def model_metrics(
    train_view: DatasetView,
    eval_views: dict[str, DatasetView],
    train_prediction: np.ndarray,
    eval_predictions: dict[str, np.ndarray],
) -> dict[str, float]:
    metrics = evaluate_prediction_metrics(train_view.residuals, train_prediction, prefix="train")
    c_errors = []
    for name, view in eval_views.items():
        prediction = eval_predictions[name]
        metrics.update(evaluate_prediction_metrics(view.residuals, prediction, prefix=name))
        if name.endswith("_C"):
            c_errors.append(view.residuals - prediction)
    if c_errors:
        c_error = np.vstack(c_errors)
        metrics["C_all_rmse_mm"] = float(np.sqrt(np.mean(np.sum(c_error * c_error, axis=1))) * 1000.0)
    return metrics


def evaluate_model_on_views(
    coefficients: np.ndarray,
    selected: list[tuple[BasisGroup, float]],
    eval_views: dict[str, DatasetView],
    config: StatisticalResidualConfig,
) -> dict[str, float]:
    predictions = {}
    for name, view in eval_views.items():
        design = assemble_eval_design(selected, name)
        predictions[name] = design @ coefficients if design.shape[1] else np.zeros_like(view.residuals)
    dummy_train = DatasetView(
        name="dummy",
        dataset={},
        geometry_positions=np.zeros((1, 3)),
        measured_positions=np.zeros((1, 3)),
        residuals=np.zeros((1, 3)),
        workspace="dummy",
    )
    return model_metrics(dummy_train, eval_views, np.zeros((1, 3)), predictions)


def evaluate_prediction_metrics(target: np.ndarray, prediction: np.ndarray, *, prefix: str) -> dict[str, float]:
    y = np.asarray(target, dtype=float).reshape(-1, 3)
    pred = np.asarray(prediction, dtype=float).reshape(-1, 3)
    err = y - pred
    return {
        f"{prefix}_rmse_mm": float(np.sqrt(np.mean(np.sum(err * err, axis=1))) * 1000.0),
        f"{prefix}_component_rmse_mm": float(np.sqrt(np.mean(err * err)) * 1000.0),
        f"{prefix}_max_mm": float(np.max(np.linalg.norm(err, axis=1)) * 1000.0),
    }


def method_summary_row(result: StatisticalModelResult) -> dict[str, Any]:
    train_rmse = result.metrics.get("train_rmse_mm")
    c_rmse = result.metrics.get("C_all_rmse_mm")
    train_c_gap = (
        float(c_rmse) - float(train_rmse)
        if train_rmse is not None and c_rmse is not None
        else None
    )
    return {
        "method": result.method,
        "label": result.label,
        "accepted": bool(result.accepted),
        "reason": result.reason,
        "train_rmse_mm": result.metrics.get("train_rmse_mm"),
        "A_train_rmse_mm": result.metrics.get("A_train_rmse_mm"),
        "B_train_rmse_mm": result.metrics.get("B_train_rmse_mm"),
        "A_C_rmse_mm": result.metrics.get("A_C_rmse_mm"),
        "B_C_rmse_mm": result.metrics.get("B_C_rmse_mm"),
        "C_all_rmse_mm": result.metrics.get("C_all_rmse_mm"),
        "train_C_gap_mm": train_c_gap,
        "effective_dof": result.model_info.get("effective_dof"),
        "basis_or_feature_count": result.model_info.get("basis_count", result.model_info.get("feature_count")),
    }


def model_to_report(result: StatisticalModelResult) -> dict[str, Any]:
    return {
        "method": result.method,
        "label": result.label,
        "accepted": bool(result.accepted),
        "reason": result.reason,
        "metrics": result.metrics,
        "model_info": result.model_info,
        "selected_groups": result.selected_groups,
        "top_coefficients": result.top_coefficients,
        "search_rows": result.search_rows,
    }


def candidate_library_rows(
    config: StatisticalResidualConfig,
) -> list[dict[str, Any]]:
    rows = [
        {
            "family": "joint harmonic",
            "formula": "sin(h q_i), cos(h q_i)",
            "purpose": "periodic joint-angle residual structure",
            "h": list(config.harmonics),
        },
        {
            "family": "torque harmonic",
            "formula": "tau_i sin(h q_i), tau_i cos(h q_i)",
            "purpose": "load/gravity-modulated periodic residual",
            "h": list(config.harmonics),
        },
        {
            "family": "direction harmonic",
            "formula": "s_i sin(h q_i), s_i cos(h q_i)",
            "purpose": "approach-direction-modulated periodic residual",
            "h": list(config.harmonics),
        },
        {
            "family": "low-order polynomial",
            "formula": "centered q_i and centered q_i q_j",
            "purpose": "smooth low-frequency joint-space residual",
        },
        {
            "family": "workspace RBF",
            "formula": "exp(-||z-c_m||^2/(2 l^2))",
            "purpose": "controlled local workspace residual basis",
            "centers": int(config.rbf_center_count),
        },
        {
            "family": "local cluster affine",
            "formula": "1[c(k)=m] * [1, z_x, z_y, z_z]",
            "purpose": "piecewise low-complexity residual map",
            "clusters": int(config.local_cluster_count),
        },
    ]
    return rows


def top_coefficients(
    feature_names: list[str],
    coefficients: np.ndarray,
    posterior_std: np.ndarray,
    *,
    limit: int = 20,
) -> list[dict[str, Any]]:
    if len(feature_names) == 0:
        return []
    coef = np.asarray(coefficients, dtype=float).reshape(len(feature_names), 3)
    std = np.asarray(posterior_std, dtype=float).reshape(len(feature_names), 3)
    scores = np.linalg.norm(coef, axis=1)
    order = np.argsort(scores)[::-1][: int(limit)]
    rows = []
    for index in order:
        rows.append(
            {
                "feature": feature_names[int(index)],
                "coef_x_mm": float(coef[index, 0] * 1000.0),
                "coef_y_mm": float(coef[index, 1] * 1000.0),
                "coef_z_mm": float(coef[index, 2] * 1000.0),
                "coef_l2_mm": float(scores[index] * 1000.0),
                "posterior_std_l2_mm": float(np.linalg.norm(std[index]) * 1000.0),
            }
        )
    return rows


def blocked_folds(dataset: dict[str, np.ndarray], fold_count: int, seed: int) -> list[np.ndarray]:
    count = len(dataset["joints"])
    labels = np.asarray(dataset.get("workspace_labels", np.array(["all"] * count)), dtype=object).reshape(-1)
    folds = [np.zeros(count, dtype=bool) for _ in range(max(2, int(fold_count)))]
    for label in sorted(set(str(value) for value in labels)):
        indices = np.flatnonzero(labels == label)
        chunks = np.array_split(indices, len(folds))
        for fold, chunk in zip(folds, chunks):
            fold[chunk] = True
    return [fold for fold in folds if np.any(fold) and np.any(~fold)]


def combine_views(name: str, views: list[DatasetView]) -> DatasetView:
    dataset = {
        "joints": np.concatenate([view.dataset["joints"] for view in views], axis=0),
        "measured_positions": np.concatenate([view.measured_positions for view in views], axis=0),
        "payloads": np.concatenate([
            np.asarray(view.dataset.get("payloads", np.zeros(len(view.dataset["joints"]))), dtype=float).reshape(-1)
            for view in views
        ]),
        "workspace_labels": np.concatenate([
            np.asarray([view.workspace] * len(view.dataset["joints"]), dtype=object)
            for view in views
        ]),
    }
    for key in ("directions", "joint_torques", "self_weight_joint_torques", "payload_joint_torques"):
        if all(key in view.dataset for view in views):
            dataset[key] = np.concatenate([np.asarray(view.dataset[key], dtype=float).reshape(-1, 6) for view in views], axis=0)
    return DatasetView(
        name=name,
        dataset=dataset,
        geometry_positions=np.concatenate([view.geometry_positions for view in views], axis=0),
        measured_positions=dataset["measured_positions"],
        residuals=np.concatenate([view.residuals for view in views], axis=0),
        workspace="combined",
    )


def make_dataset_view(
    name: str,
    dataset: dict[str, np.ndarray],
    geometry_positions: np.ndarray,
    workspace: str,
) -> DatasetView:
    measured = np.asarray(dataset["measured_positions"], dtype=float).reshape(-1, 3)
    geometry = np.asarray(geometry_positions, dtype=float).reshape(-1, 3)
    return DatasetView(
        name=name,
        dataset=dataset,
        geometry_positions=geometry,
        measured_positions=measured,
        residuals=measured - geometry,
        workspace=workspace,
    )


def run_synthetic_statistical_checks(config: StatisticalResidualConfig) -> dict[str, Any]:
    rng = np.random.default_rng(int(config.seed) + 909)
    q = rng.uniform(-2.0, 2.0, size=(160, 6))
    residual = np.zeros((160, 3), dtype=float)
    residual[:, 0] = 0.20e-3 * np.sin(2.0 * q[:, 0])
    residual += rng.normal(0.0, config.noise_std_m, size=residual.shape)
    dataset = {
        "joints": q,
        "measured_positions": residual,
        "directions": np.sign(q),
        "joint_torques": np.ones_like(q),
        "payloads": np.zeros(len(q)),
    }
    dataset["directions"][dataset["directions"] == 0.0] = 1.0
    geometry = np.zeros_like(residual)
    view = make_dataset_view("synthetic_train", dataset, geometry, "synthetic")
    train = make_dataset_view("synthetic_eval", dataset, geometry, "synthetic")
    groups = build_basis_groups(view, {"synthetic_eval": train}, config.quickened())
    folds = blocked_folds({"joints": q, "workspace_labels": np.array(["synthetic"] * len(q))}, 4, config.seed)
    result = fit_bayesian_basis_model(view, {"synthetic_eval": train}, groups, folds, config.quickened())
    selected_names = [row["name"] for row in result.selected_groups]
    noise_only = rng.normal(0.0, config.noise_std_m, size=residual.shape)
    noise_rmse = float(np.sqrt(np.mean(np.sum(noise_only * noise_only, axis=1))) * 1000.0)
    noise_dataset = {
        "joints": q,
        "measured_positions": noise_only,
        "directions": dataset["directions"],
        "joint_torques": dataset["joint_torques"],
        "payloads": dataset["payloads"],
    }
    noise_view = make_dataset_view("noise_only_train", noise_dataset, geometry, "synthetic")
    noise_eval = make_dataset_view("noise_only_eval", noise_dataset, geometry, "synthetic")
    noise_groups = build_basis_groups(noise_view, {"noise_only_eval": noise_eval}, config.quickened())
    noise_result = fit_bayesian_basis_model(
        noise_view,
        {"noise_only_eval": noise_eval},
        noise_groups,
        folds,
        config.quickened(),
    )
    noise_model_rmse = float(noise_result.metrics["train_rmse_mm"])
    return {
        "harmonic_recovery_passed": bool(any("joint_harmonic_h2" == name for name in selected_names)),
        "selected_groups": selected_names,
        "synthetic_train_rmse_mm": float(result.metrics["train_rmse_mm"]),
        "noise_floor_reference_rmse_mm": noise_rmse,
        "noise_only_model_train_rmse_mm": noise_model_rmse,
        "noise_only_not_interpolated_passed": bool(noise_model_rmse > 0.025),
        "pure_noise_not_interpolated_reference": "noise-only pass means train RMSE stayed above 0.025 mm rather than collapsing toward zero",
    }


def standardize_group(
    train_raw: np.ndarray,
    names: list[str],
    eval_raw: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray], list[str]]:
    train = np.asarray(train_raw, dtype=float)
    _, mean, std = standardize_fit_transform(train)
    centered = train - mean.reshape(1, -1)
    norms = np.linalg.norm(centered, axis=0)
    threshold = max(float(np.max(norms)) if norms.size else 0.0, 1.0) * 1.0e-12
    keep = norms > threshold
    if not np.any(keep):
        return np.zeros((len(train), 0), dtype=float), {
            key: np.zeros((len(value), 0), dtype=float) for key, value in eval_raw.items()
        }, []
    train_std = centered[:, keep] / std.reshape(1, -1)[:, keep]
    eval_std = {
        key: (np.asarray(value, dtype=float)[:, keep] - mean.reshape(1, -1)[:, keep]) / std.reshape(1, -1)[:, keep]
        for key, value in eval_raw.items()
    }
    kept_names = [name for name, flag in zip(names, keep) if bool(flag)]
    return train_std, eval_std, kept_names


def standardize_fit_transform(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(values, dtype=float)
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std[std <= 1.0e-12] = 1.0
    return (x - mean.reshape(1, -1)) / std.reshape(1, -1), mean, std


def kmeans_centers(values: np.ndarray, count: int, seed: int) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if int(count) <= 1 or len(x) <= 1:
        return np.mean(x, axis=0, keepdims=True)
    kmeans = KMeans(n_clusters=int(count), n_init=10, random_state=int(seed))
    kmeans.fit(x)
    return np.asarray(kmeans.cluster_centers_, dtype=float)


def pairwise_squared(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa = np.sum(np.asarray(a, dtype=float) ** 2, axis=1, keepdims=True)
    bb = np.sum(np.asarray(b, dtype=float) ** 2, axis=1, keepdims=True).T
    return np.maximum(aa + bb - 2.0 * np.asarray(a, dtype=float) @ np.asarray(b, dtype=float).T, 0.0)


def euclidean_rmse_mm(target: np.ndarray, prediction: np.ndarray) -> float:
    err = np.asarray(target, dtype=float).reshape(-1, 3) - np.asarray(prediction, dtype=float).reshape(-1, 3)
    return float(np.sqrt(np.mean(np.sum(err * err, axis=1))) * 1000.0)


def _matrix_or_zeros(dataset: dict[str, np.ndarray], key: str, width: int) -> np.ndarray:
    count = len(dataset["joints"])
    if key not in dataset:
        return np.zeros((count, int(width)), dtype=float)
    return np.asarray(dataset[key], dtype=float).reshape(count, int(width))


def _directions(dataset: dict[str, np.ndarray]) -> np.ndarray:
    if "directions" not in dataset:
        q = np.asarray(dataset["joints"], dtype=float).reshape(-1, 6)
        values = np.sign(q)
    else:
        values = np.sign(np.asarray(dataset["directions"], dtype=float).reshape(-1, 6))
    values[values == 0.0] = 1.0
    return values

