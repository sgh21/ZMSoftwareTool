"""Service layer for robot parameter identification and live accuracy state.

The UI intentionally talks to this module instead of directly coupling itself
to the Bayesian calibration package.  The default identification path is S1:
cross-validated L2 regularization, identifiability-based local weights, and
sequential fitting over pose subspaces.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

from core.accuracy_evaluator import evaluate_position_errors


def _prepare_imports() -> None:
    """Work around six / yourdfpy compatibility issue with Python 3.12."""
    import six  # noqa: F401

    for importer in sys.meta_path:
        if type(importer).__name__ == "_SixMetaPathImporter" and not hasattr(importer, "_path"):
            importer._path = None
    import dateutil.parser  # noqa: F401


_prepare_imports()

from core.calibration.bayesian_calibration_pipeline.core.data_io import load_dataset
from core.calibration.bayesian_calibration_pipeline.core.dynamic_identifiability import (
    SubspaceIdentifiabilityPartition,
    build_identifiability_subspace_partition,
    compute_pose_identifiability_metrics,
    fit_subspace_sequential_l2,
)
from core.calibration.bayesian_calibration_pipeline.core.geometric import (
    select_geometric_parameters,
)
from core.calibration.bayesian_calibration_pipeline.core.parameters import (
    ErrorParameter,
    build_error_parameters,
    parameter_scales,
    vector_to_named_dict,
    zero_error_vector,
)
from core.calibration.bayesian_calibration_pipeline.core.redundancy import output_jacobian
from core.calibration.bayesian_calibration_pipeline.core.regularization import (
    RegularizedLMResult,
    fit_l2_lm,
    make_lambda_grid,
    random_folds,
    select_independent_parameters,
)
from core.calibration.bayesian_calibration_pipeline.core.robot_model import (
    MultiSourceRobotModel,
    load_nominal_robot,
)


@dataclass(frozen=True)
class IdentificationOptions:
    """Runtime controls for S1 identification.

    Defaults are deliberately smaller than the research ablation script so the
    desktop tool remains usable interactively.  Tests and experiments can pass
    explicit values for deterministic smoke runs.
    """

    method: str = "S1"
    max_nfev: int = 60
    cv_folds: int = 3
    lambda_min_power: float = -10.0
    lambda_max_power: float = 0.0
    lambda_count: int = 5
    redundancy_tolerance: float = 1.0e-7
    redundancy_max_combinations: int = 200_000
    rho_threshold: float = 0.5
    kappa_threshold: float = 0.05
    risk_beta: float = 0.0
    risk_power: float = 1.0
    min_weight: float = 1.0
    max_weight: float = 100.0
    strong_weight: float = 10_000.0
    dynamic_risk_quantile: float = 0.25
    subspace_min_cluster_size: int = 4
    subspace_k_candidates: tuple[int, ...] = (2, 3, 4)
    seed: int = 20260524
    jacobian_method: str = "analytic"
    lambda_grid: tuple[float, ...] | None = None


@dataclass
class CalibrationResult:
    """Result of a robot error-parameter identification run."""

    success: bool
    message: str
    method: str = "S1"
    nominal_positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=float))
    predicted_positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=float))
    calibrated_positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=float))
    measured_positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=float))
    positioning_errors: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=float))
    fit_errors: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=float))
    error_vector: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    error_parameters: list[ErrorParameter] = field(default_factory=list)
    parameter_names: list[str] = field(default_factory=list)
    parameter_values: dict[str, float] = field(default_factory=dict)
    # Backward-compatible names: these are fit residuals to measured data.
    rmse_mm: float = 0.0
    max_error_mm: float = 0.0
    per_sample_errors_mm: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    # User-facing positioning error: identified prediction minus nominal FK.
    position_error_rmse_mm: float = 0.0
    position_error_max_mm: float = 0.0
    per_sample_position_errors_mm: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=float)
    )
    nominal_to_measured_rmse_mm: float = 0.0
    nominal_to_measured_max_mm: float = 0.0
    joint_count: int = 0
    nfev: int = 0
    confidence: float = 100.0
    selected_lambda: float = 0.0
    cv_scores: list[dict[str, Any]] = field(default_factory=list)
    active_indices: list[int] = field(default_factory=list)
    subspace_summary: dict[str, Any] = field(default_factory=dict)
    dataset_paths: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CurrentAccuracyState:
    """Live state for one current robot configuration."""

    nominal_position: np.ndarray
    predicted_position: np.ndarray
    positioning_error: np.ndarray
    error_norm_mm: float
    rms_mm: float
    max_error_mm: float
    over_tolerance_rate: float
    confidence: float
    health_score: float
    health_level: str
    health_message: str


class CalibrationService:
    """Robot kinematic identification service used by the desktop UI."""

    def __init__(
        self,
        project_root: str | Path | None = None,
        nominal_config_path: str | Path | None = None,
    ) -> None:
        self.project_root = Path(project_root or Path.cwd()).resolve()
        self.nominal_config_path = self._resolve_nominal_config_path(nominal_config_path)
        nominal = load_nominal_robot(_load_nominal_config(self.nominal_config_path))
        self._model = MultiSourceRobotModel(nominal)
        self._full_parameters = build_error_parameters()
        self._geometric_parameters = select_geometric_parameters(self._full_parameters)
        self._scales = parameter_scales(self._geometric_parameters)
        self._active_error_vector = zero_error_vector(self._geometric_parameters)
        self._active_parameter_values = vector_to_named_dict(
            self._active_error_vector, self._geometric_parameters
        )
        self._active_confidence = 0.0

    @property
    def model(self) -> MultiSourceRobotModel:
        return self._model

    @property
    def geometric_parameters(self) -> list[ErrorParameter]:
        return list(self._geometric_parameters)

    @property
    def active_parameter_values(self) -> dict[str, float]:
        return dict(self._active_parameter_values)

    @property
    def active_confidence(self) -> float:
        return float(self._active_confidence)

    def compute_nominal_position(
        self,
        joint_angles: list[float] | np.ndarray,
        *,
        joint_unit: str = "auto",
    ) -> np.ndarray:
        """Compute nominal TCP position from configurable nominal MD-H parameters."""
        angles = normalize_joint_configs(np.asarray(joint_angles, dtype=float).reshape(1, 6), joint_unit)
        zero = zero_error_vector(self._geometric_parameters)
        return self._model.position(angles[0], zero, self._geometric_parameters)

    def compute_nominal_positions(
        self,
        joint_configs: np.ndarray,
        *,
        joint_unit: str = "auto",
    ) -> np.ndarray:
        """Compute nominal TCP positions for multiple joint configurations."""
        joints = normalize_joint_configs(joint_configs, joint_unit)
        zero = zero_error_vector(self._geometric_parameters)
        return self._model.batch_positions(joints, zero, self._geometric_parameters)

    def compute_predicted_position(
        self,
        joint_angles: list[float] | np.ndarray,
        parameter_values: dict[str, float] | None = None,
        *,
        joint_unit: str = "auto",
        confidence: float | None = None,
    ) -> CurrentAccuracyState:
        """Predict actual TCP position and current positioning error.

        The positioning error is defined as:
        ``identified_model_position - nominal_model_position``.
        """
        vector = (
            self.error_vector_from_values(parameter_values)
            if parameter_values is not None
            else self._active_error_vector
        )
        conf = self._active_confidence if confidence is None else float(confidence)
        joints = normalize_joint_configs(np.asarray(joint_angles, dtype=float).reshape(1, 6), joint_unit)
        zero = zero_error_vector(self._geometric_parameters)
        nominal = self._model.position(joints[0], zero, self._geometric_parameters)
        predicted = self._model.position(joints[0], vector, self._geometric_parameters)
        error = predicted - nominal
        thresholds = self._load_thresholds()
        tolerance_m = thresholds["position_rms_limit_mm"] / 1000.0
        metrics = evaluate_position_errors(error.reshape(1, 3), tolerance_m)
        norm_mm = float(np.linalg.norm(error) * 1000.0)
        health_score = _score_from_error(norm_mm, thresholds["position_rms_limit_mm"])
        health_level = (
            "good" if health_score >= 80.0 else "warning" if health_score >= 60.0 else "critical"
        )
        return CurrentAccuracyState(
            nominal_position=nominal,
            predicted_position=predicted,
            positioning_error=error,
            error_norm_mm=norm_mm,
            rms_mm=float(metrics.rms * 1000.0),
            max_error_mm=float(metrics.max_error * 1000.0),
            over_tolerance_rate=float(metrics.over_tolerance_rate),
            confidence=float(conf),
            health_score=float(health_score),
            health_level=health_level,
            health_message=f"health={health_level}, score={health_score:.1f}",
        )

    def load_calibration_data(self, path: str | Path) -> dict[str, np.ndarray]:
        """Load one pkl identification dataset."""
        return load_dataset(path)

    def load_identification_data(self, paths: str | Path | Iterable[str | Path]) -> dict[str, Any]:
        """Load and concatenate one or more pkl files into a stable schema."""
        path_list = _coerce_paths(paths)
        if not path_list:
            raise ValueError("At least one identification data file is required.")
        datasets = [load_dataset(path) for path in path_list]
        merged: dict[str, Any] = {
            "joints": np.concatenate([np.asarray(data["joints"], dtype=float) for data in datasets]),
            "measured_positions": np.concatenate(
                [np.asarray(data["measured_positions"], dtype=float) for data in datasets]
            ),
            "dataset_paths": [str(path.resolve()) for path in path_list],
            "sample_counts": [int(len(data["joints"])) for data in datasets],
        }
        for key in ("payloads", "directions", "joint_torques"):
            if all(key in data for data in datasets):
                merged[key] = np.concatenate([np.asarray(data[key], dtype=float) for data in datasets])
        return merged

    def run_calibration(
        self,
        joint_configs: np.ndarray,
        measured_positions: np.ndarray,
        *,
        max_nfev: int | None = None,
        tolerance: float = 1e-7,
        **kwargs: Any,
    ) -> CalibrationResult:
        """Backward-compatible alias for S1 parameter identification."""
        options = kwargs.pop("options", None)
        if options is None:
            defaults = IdentificationOptions()
            options = IdentificationOptions(
                max_nfev=max_nfev or defaults.max_nfev,
                redundancy_tolerance=tolerance,
            )
        return self.run_identification(
            joint_configs,
            measured_positions,
            options=options,
            **kwargs,
        )

    def run_identification(
        self,
        joint_configs: np.ndarray,
        measured_positions: np.ndarray,
        *,
        payloads: np.ndarray | float | None = None,
        directions: np.ndarray | None = None,
        dataset_paths: Iterable[str | Path] | None = None,
        options: IdentificationOptions | None = None,
        joint_unit: str = "auto",
        activate: bool = False,
    ) -> CalibrationResult:
        """Run S1 identification and return model, fit, and positioning metrics."""
        opts = options or IdentificationOptions()
        if opts.method.upper() != "S1":
            return CalibrationResult(False, f"Unsupported identification method: {opts.method}")

        try:
            joints = normalize_joint_configs(joint_configs, joint_unit)
            measured = np.asarray(measured_positions, dtype=float).reshape(-1, 3)
            _validate_dataset(joints, measured)
            payload_array = _subset_or_none(payloads, np.arange(len(joints)), len(joints))
            direction_array = _subset_or_none(directions, np.arange(len(joints)), len(joints))
            result = self._run_s1(
                joints,
                measured,
                payloads=payload_array,
                directions=direction_array,
                dataset_paths=[str(Path(path)) for path in dataset_paths or []],
                options=opts,
            )
            if result.success and activate:
                self.set_active_parameters(result.parameter_values, confidence=result.confidence)
            return result
        except Exception as exc:  # noqa: BLE001
            return CalibrationResult(False, f"S1 identification failed: {exc}", method="S1")

    def identify_from_files(
        self,
        paths: str | Path | Iterable[str | Path],
        *,
        options: IdentificationOptions | None = None,
    ) -> CalibrationResult:
        """Convenience wrapper: load pkl files, concatenate, and run S1."""
        data = self.load_identification_data(paths)
        return self.run_identification(
            data["joints"],
            data["measured_positions"],
            payloads=data.get("payloads"),
            directions=data.get("directions"),
            dataset_paths=data.get("dataset_paths", []),
            options=options,
            joint_unit="auto",
            activate=True,
        )

    def set_active_parameters(
        self,
        parameter_values: dict[str, float],
        *,
        confidence: float = 100.0,
    ) -> None:
        """Set the active identified parameter model used for live prediction."""
        self._active_error_vector = self.error_vector_from_values(parameter_values)
        self._active_parameter_values = vector_to_named_dict(
            self._active_error_vector, self._geometric_parameters
        )
        self._active_confidence = float(confidence)

    def load_active_parameters(self, path: str | Path) -> dict[str, Any]:
        """Load an identified parameter YAML file and activate it."""
        from core.calibration_persistence import load_identification_result

        loaded = load_identification_result(path)
        self.set_active_parameters(
            loaded["error_parameters"],
            confidence=float(loaded.get("confidence", 100.0)),
        )
        return loaded

    def error_vector_from_values(self, values: dict[str, float] | None) -> np.ndarray:
        """Convert a name/value parameter mapping into the geometry33 vector."""
        vector = np.zeros(len(self._geometric_parameters), dtype=float)
        source = values or {}
        for index, parameter in enumerate(self._geometric_parameters):
            vector[index] = float(source.get(parameter.name, 0.0))
        return vector

    def compute_error_metrics(
        self,
        predicted_positions: np.ndarray,
        reference_positions: np.ndarray,
    ) -> dict[str, float]:
        """Compute Euclidean position-error metrics in millimeters."""
        errors_m = np.linalg.norm(
            np.asarray(predicted_positions, dtype=float).reshape(-1, 3)
            - np.asarray(reference_positions, dtype=float).reshape(-1, 3),
            axis=1,
        )
        return {
            "rmse_mm": float(np.sqrt(np.mean(errors_m**2)) * 1000.0),
            "max_error_mm": float(np.max(errors_m) * 1000.0),
            "mean_error_mm": float(np.mean(errors_m) * 1000.0),
            "std_error_mm": float(np.std(errors_m) * 1000.0),
        }

    def _run_s1(
        self,
        joints: np.ndarray,
        measured: np.ndarray,
        *,
        payloads: np.ndarray | float | None,
        directions: np.ndarray | None,
        dataset_paths: list[str],
        options: IdentificationOptions,
    ) -> CalibrationResult:
        parameters = self._geometric_parameters
        zero = zero_error_vector(parameters)
        nominal = self._model.batch_positions(joints, zero, parameters, payloads, directions)
        active = list(range(len(parameters)))

        independent_indices: list[int] = []
        try:
            redundancy = select_independent_parameters(
                self._model,
                joints,
                parameters,
                payloads=payloads,
                directions=directions,
                tolerance=options.redundancy_tolerance,
                max_combinations=options.redundancy_max_combinations,
                jacobian_method=options.jacobian_method,
            )
            independent_indices = [int(index) for index in redundancy.independent_indices]
        except Exception:
            independent_indices = []

        partition = self._build_s1_partition(joints, zero, payloads, directions, options)
        lambdas = _lambda_grid(options)
        cv_scores = self._evaluate_s1_cv(joints, measured, payloads, directions, lambdas, options)
        selected = min(
            cv_scores,
            key=lambda row: (
                float(row["max_rmse_mm"]),
                float(row["mean_rmse_mm"]),
                float(row["lambda"]),
            ),
        )
        selected_lambda = float(selected["lambda"])
        final_fit = fit_subspace_sequential_l2(
            self._model,
            joints,
            measured,
            parameters,
            partition,
            lambda_value=selected_lambda,
            active_indices=active,
            payloads=payloads,
            directions=directions,
            max_nfev=options.max_nfev,
        )
        predicted = self._model.batch_positions(
            joints, final_fit.result.vector, parameters, payloads, directions
        )
        return self._build_result(
            final_fit.result,
            nominal,
            predicted,
            measured,
            selected_lambda,
            cv_scores,
            partition,
            independent_indices,
            dataset_paths,
            options,
        )

    def _build_s1_partition(
        self,
        joints: np.ndarray,
        vector: np.ndarray,
        payloads: np.ndarray | float | None,
        directions: np.ndarray | None,
        options: IdentificationOptions,
    ) -> SubspaceIdentifiabilityPartition:
        parameters = self._geometric_parameters
        pose_metrics = compute_pose_identifiability_metrics(
            self._model,
            joints,
            parameters,
            vector,
            payloads=payloads,
            directions=directions,
            tolerance=options.redundancy_tolerance,
            rho_threshold=options.rho_threshold,
            kappa_threshold=options.kappa_threshold,
            risk_beta=options.risk_beta,
            risk_power=options.risk_power,
            min_weight=options.min_weight,
            max_weight=options.max_weight,
            risk_quantile=options.dynamic_risk_quantile,
            jacobian_method=options.jacobian_method,
        )
        jacobian = output_jacobian(
            self._model,
            joints,
            vector,
            parameters,
            payloads=payloads,
            directions=directions,
            method=options.jacobian_method,
        )
        return build_identifiability_subspace_partition(
            pose_metrics,
            jacobian,
            parameters,
            k_candidates=_valid_k_candidates(len(joints), options.subspace_k_candidates),
            min_cluster_size=max(1, min(options.subspace_min_cluster_size, len(joints))),
            seed=options.seed + 509,
            tolerance=options.redundancy_tolerance,
            rho_threshold=options.rho_threshold,
            kappa_threshold=options.kappa_threshold,
            risk_beta=options.risk_beta,
            risk_power=options.risk_power,
            min_weight=options.min_weight,
            max_weight=options.max_weight,
            strong_weight=options.strong_weight,
        )

    def _evaluate_s1_cv(
        self,
        joints: np.ndarray,
        measured: np.ndarray,
        payloads: np.ndarray | float | None,
        directions: np.ndarray | None,
        lambdas: np.ndarray,
        options: IdentificationOptions,
    ) -> list[dict[str, Any]]:
        folds = random_folds(len(joints), options.cv_folds, seed=options.seed + 11)
        zero = zero_error_vector(self._geometric_parameters)
        active = list(range(len(self._geometric_parameters)))
        fold_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, SubspaceIdentifiabilityPartition]] = {}

        scores: list[dict[str, Any]] = []
        for lambda_value in lambdas:
            fold_rmse: list[float] = []
            for fold_index, validation_indices in enumerate(folds):
                validation_indices = np.asarray(validation_indices, dtype=int).reshape(-1)
                if validation_indices.size == 0 or validation_indices.size >= len(joints):
                    continue
                if fold_index not in fold_cache:
                    mask = np.ones(len(joints), dtype=bool)
                    mask[validation_indices] = False
                    train_indices = np.flatnonzero(mask)
                    train_payloads = _subset_or_none(payloads, train_indices, len(joints))
                    train_directions = _subset_or_none(directions, train_indices, len(joints))
                    partition = self._build_s1_partition(
                        joints[train_indices],
                        zero,
                        train_payloads,
                        train_directions,
                        options,
                    )
                    fold_cache[fold_index] = (
                        train_indices,
                        validation_indices,
                        np.asarray(train_payloads) if train_payloads is not None else np.asarray([]),
                        partition,
                    )
                train_indices, validation_indices, _, partition = fold_cache[fold_index]
                train_payloads = _subset_or_none(payloads, train_indices, len(joints))
                train_directions = _subset_or_none(directions, train_indices, len(joints))
                validation_payloads = _subset_or_none(payloads, validation_indices, len(joints))
                validation_directions = _subset_or_none(directions, validation_indices, len(joints))
                fit = fit_subspace_sequential_l2(
                    self._model,
                    joints[train_indices],
                    measured[train_indices],
                    self._geometric_parameters,
                    partition,
                    lambda_value=float(lambda_value),
                    active_indices=active,
                    payloads=train_payloads,
                    directions=train_directions,
                    max_nfev=options.max_nfev,
                )
                predicted = self._model.batch_positions(
                    joints[validation_indices],
                    fit.result.vector,
                    self._geometric_parameters,
                    validation_payloads,
                    validation_directions,
                )
                fold_rmse.append(_euclidean_rmse_m(measured[validation_indices], predicted))
            values = np.asarray(fold_rmse, dtype=float)
            scores.append(
                {
                    "lambda": float(lambda_value),
                    "fold_rmse_mm": [float(value * 1000.0) for value in values],
                    "mean_rmse_mm": float(np.mean(values) * 1000.0) if values.size else float("inf"),
                    "max_rmse_mm": float(np.max(values) * 1000.0) if values.size else float("inf"),
                    "std_rmse_mm": float(np.std(values) * 1000.0) if values.size else float("inf"),
                }
            )
        return scores

    def _build_result(
        self,
        fit: RegularizedLMResult,
        nominal: np.ndarray,
        predicted: np.ndarray,
        measured: np.ndarray,
        selected_lambda: float,
        cv_scores: list[dict[str, Any]],
        partition: SubspaceIdentifiabilityPartition,
        independent_indices: list[int],
        dataset_paths: list[str],
        options: IdentificationOptions,
    ) -> CalibrationResult:
        fit_vectors = predicted - measured
        fit_errors_mm = np.linalg.norm(fit_vectors, axis=1) * 1000.0
        positioning_vectors = predicted - nominal
        positioning_errors_mm = np.linalg.norm(positioning_vectors, axis=1) * 1000.0
        nominal_errors_mm = np.linalg.norm(nominal - measured, axis=1) * 1000.0
        parameter_values = vector_to_named_dict(fit.vector, self._geometric_parameters)
        best_cv = min(cv_scores, key=lambda row: (row["max_rmse_mm"], row["mean_rmse_mm"], row["lambda"]))
        confidence = _confidence_from_cv(
            float(best_cv["max_rmse_mm"]),
            self._load_thresholds()["position_rms_limit_mm"],
        )
        subspace_summary = {
            "K": int(partition.K),
            "cluster_sizes": [int(size) for size in partition.cluster_sizes],
            "order": [int(item) for item in partition.order],
            "summaries": partition.subspace_summaries,
        }
        return CalibrationResult(
            success=bool(fit.success),
            message=str(fit.message),
            method="S1",
            nominal_positions=nominal,
            predicted_positions=predicted,
            calibrated_positions=predicted,
            measured_positions=measured,
            positioning_errors=positioning_vectors,
            fit_errors=fit_vectors,
            error_vector=np.asarray(fit.vector, dtype=float).copy(),
            error_parameters=list(self._geometric_parameters),
            parameter_names=[parameter.name for parameter in self._geometric_parameters],
            parameter_values=parameter_values,
            rmse_mm=float(np.sqrt(np.mean(fit_errors_mm**2))),
            max_error_mm=float(np.max(fit_errors_mm)),
            per_sample_errors_mm=fit_errors_mm,
            position_error_rmse_mm=float(np.sqrt(np.mean(positioning_errors_mm**2))),
            position_error_max_mm=float(np.max(positioning_errors_mm)),
            per_sample_position_errors_mm=positioning_errors_mm,
            nominal_to_measured_rmse_mm=float(np.sqrt(np.mean(nominal_errors_mm**2))),
            nominal_to_measured_max_mm=float(np.max(nominal_errors_mm)),
            joint_count=len(measured),
            nfev=int(fit.nfev),
            confidence=confidence,
            selected_lambda=float(selected_lambda),
            cv_scores=cv_scores,
            active_indices=independent_indices,
            subspace_summary=subspace_summary,
            dataset_paths=dataset_paths,
            metadata={
                "options": {
                    "max_nfev": int(options.max_nfev),
                    "cv_folds": int(options.cv_folds),
                    "lambda_count": int(len(_lambda_grid(options))),
                    "jacobian_method": options.jacobian_method,
                },
                "parameter_count": len(self._geometric_parameters),
                "independent_count": len(independent_indices),
            },
        )

    def _resolve_nominal_config_path(self, path: str | Path | None) -> Path | None:
        if path is not None:
            return Path(path).resolve()
        default_path = self.project_root / "config" / "nominal_robot.yaml"
        return default_path if default_path.exists() else None

    def _load_thresholds(self) -> dict[str, float]:
        defaults = {
            "position_rms_limit_mm": 0.5,
            "max_error_limit_mm": 1.0,
            "over_tolerance_rate_limit": 0.05,
            "min_model_confidence": 0.8,
        }
        path = self.project_root / "config" / "thresholds.yaml"
        if not path.exists():
            return defaults
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            return defaults
        accuracy = data.get("accuracy", {}) if isinstance(data, dict) else {}
        if not isinstance(accuracy, dict):
            return defaults
        for key in defaults:
            if key in accuracy:
                defaults[key] = float(accuracy[key])
        return defaults


def normalize_joint_configs(joint_configs: np.ndarray, unit: str = "auto") -> np.ndarray:
    """Return N x 6 joint angles in radians.

    pkl datasets in this project are radians; UI spin boxes are degrees.  The
    auto mode converts only when values clearly exceed a radian-scale range.
    """
    joints = np.asarray(joint_configs, dtype=float).reshape(-1, 6)
    unit_key = (unit or "auto").lower()
    if unit_key in {"rad", "radian", "radians"}:
        return joints
    if unit_key in {"deg", "degree", "degrees"}:
        return np.deg2rad(joints)
    if unit_key != "auto":
        raise ValueError("joint_unit must be 'auto', 'radians', or 'degrees'.")
    max_abs = float(np.max(np.abs(joints))) if joints.size else 0.0
    return np.deg2rad(joints) if max_abs > (2.0 * math.pi + 1.0e-3) else joints


def _load_nominal_config(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Nominal robot config must be a mapping: {path}")
    return data.get("nominal_robot", data)


def _coerce_paths(paths: str | Path | Iterable[str | Path]) -> list[Path]:
    if isinstance(paths, (str, Path)):
        return [Path(paths).resolve()]
    return [Path(path).resolve() for path in paths]


def _validate_dataset(joints: np.ndarray, measured: np.ndarray) -> None:
    if joints.ndim != 2 or joints.shape[1] != 6:
        raise ValueError(f"joint_configs must be Nx6, got {joints.shape}")
    if measured.ndim != 2 or measured.shape[1] != 3:
        raise ValueError(f"measured_positions must be Nx3, got {measured.shape}")
    if len(joints) != len(measured):
        raise ValueError("joint and position sample counts must match")
    if len(joints) < 6:
        raise ValueError(f"Need at least 6 samples for S1 identification, got {len(joints)}")


def _lambda_grid(options: IdentificationOptions) -> np.ndarray:
    if options.lambda_grid is not None:
        values = np.asarray(options.lambda_grid, dtype=float)
        return values[values >= 0.0]
    return make_lambda_grid(
        options.lambda_min_power,
        options.lambda_max_power,
        options.lambda_count,
    )


def _valid_k_candidates(sample_count: int, candidates: tuple[int, ...]) -> tuple[int, ...]:
    values = tuple(sorted({int(value) for value in candidates if 2 <= int(value) <= sample_count}))
    if values:
        return values
    return (2,) if sample_count >= 2 else (1,)


def _subset_or_none(
    values: np.ndarray | float | None,
    indices: np.ndarray,
    sample_count: int,
) -> np.ndarray | float | None:
    if values is None:
        return None
    array = np.asarray(values, dtype=float)
    if array.ndim == 0 or array.size == 1:
        return float(array.reshape(-1)[0])
    if array.shape[0] != int(sample_count):
        return values
    return array[np.asarray(indices, dtype=int)]


def _euclidean_rmse_m(measured: np.ndarray, predicted: np.ndarray) -> float:
    errors = np.linalg.norm(
        np.asarray(predicted, dtype=float).reshape(-1, 3)
        - np.asarray(measured, dtype=float).reshape(-1, 3),
        axis=1,
    )
    return float(np.sqrt(np.mean(errors**2)))


def _confidence_from_cv(cv_max_rmse_mm: float, limit_mm: float) -> float:
    """Return an empirical reliability score from cross-validation error.

    This is not a Bayesian posterior probability.  It maps the worst-fold
    cross-validation RMSE to 0-100 using the configured RMS limit as scale:
    confidence = 100 / (1 + cv_max_rmse_mm / (10 * limit_mm)).
    """
    if not np.isfinite(cv_max_rmse_mm):
        return 0.0
    scale = max(float(limit_mm), 1.0e-6)
    confidence = 100.0 / (1.0 + max(cv_max_rmse_mm, 0.0) / (10.0 * scale))
    return float(max(0.0, min(100.0, confidence)))


def _score_from_error(error_mm: float, limit_mm: float) -> float:
    scale = max(float(limit_mm), 1.0e-6)
    ratio = max(float(error_mm), 0.0) / scale
    score = 100.0 / (1.0 + ratio / 3.0)
    return float(max(0.0, min(100.0, score)))
