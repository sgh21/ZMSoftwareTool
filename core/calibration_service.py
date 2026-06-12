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
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from core.accuracy_evaluator import evaluate_position_errors
from core.calibration_dataset_packager import (
    default_packed_dataset_path,
    pack_raw_calibration_pair,
    split_calibration_input_paths,
)
from core.health_evaluator import evaluate_positioning_health


def _prepare_imports() -> None:
    """Work around six / yourdfpy compatibility issue with Python 3.12."""
    import six  # noqa: F401

    for importer in sys.meta_path:
        if type(importer).__name__ == "_SixMetaPathImporter" and not hasattr(importer, "_path"):
            importer._path = None
    import dateutil.parser  # noqa: F401


_prepare_imports()

from core.calibration.bayesian_calibration_pipeline.core.data_io import load_dataset  # noqa: E402
from core.calibration.bayesian_calibration_pipeline.core.dynamic_identifiability import (  # noqa: E402
    SubspaceIdentifiabilityPartition,
    build_identifiability_subspace_partition,
    compute_pose_identifiability_metrics,
    fit_subspace_sequential_l2,
)
from core.calibration.bayesian_calibration_pipeline.core.geometric import (  # noqa: E402
    select_geometric_parameters,
)
from core.calibration.bayesian_calibration_pipeline.core.parameters import (  # noqa: E402
    ErrorParameter,
    build_error_parameters,
    parameter_scales,
    vector_to_named_dict,
    zero_error_vector,
)
from core.calibration.bayesian_calibration_pipeline.core.redundancy import output_jacobian  # noqa: E402
from core.calibration.bayesian_calibration_pipeline.core.regularization import (  # noqa: E402
    RegularizedLMResult,
    fit_l2_lm,
    make_lambda_grid,
    random_folds,
    select_independent_parameters,
)
from core.calibration.bayesian_calibration_pipeline.core.robot_model import (  # noqa: E402
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
    debug_fast: bool = False
    fixed_lambda: float = 1.0e-10


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
    nominal_robot: dict[str, Any] = field(default_factory=dict)
    identified_robot: dict[str, Any] = field(default_factory=dict)
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
        self._nominal_config_signature = _file_signature(self.nominal_config_path)
        self._model = MultiSourceRobotModel(nominal)
        self._full_parameters = build_error_parameters()
        self._geometric_parameters = select_geometric_parameters(self._full_parameters)
        self._scales = parameter_scales(self._geometric_parameters)
        self._active_error_vector = zero_error_vector(self._geometric_parameters)
        self._active_parameter_values = vector_to_named_dict(
            self._active_error_vector, self._geometric_parameters
        )
        self._active_identified_robot: dict[str, Any] | None = None
        self._active_identified_model: MultiSourceRobotModel | None = None
        self._active_parameter_source_path: Path | None = None
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

    def reload_nominal_parameters(self, nominal_config_path: str | Path | None = None) -> None:
        """Reload the nominal MD-H/tool parameters used by live calculations."""
        if nominal_config_path is not None:
            self.nominal_config_path = Path(nominal_config_path).resolve()
        elif self.nominal_config_path is None:
            self.nominal_config_path = self._resolve_nominal_config_path(None)
        nominal = load_nominal_robot(_load_nominal_config(self.nominal_config_path))
        self._model = MultiSourceRobotModel(nominal)
        self._nominal_config_signature = _file_signature(self.nominal_config_path)

    def reload_nominal_parameters_if_changed(self) -> bool:
        """Reload nominal parameters when the configured YAML file changed."""
        if self.nominal_config_path is None:
            self.nominal_config_path = self._resolve_nominal_config_path(None)
        current_signature = _file_signature(self.nominal_config_path)
        if current_signature == self._nominal_config_signature:
            return False
        self.reload_nominal_parameters()
        return True

    def compute_nominal_position(
        self,
        joint_angles: list[float] | np.ndarray,
        *,
        joint_unit: str = "auto",
    ) -> np.ndarray:
        """Compute nominal TCP position from configurable nominal MD-H parameters."""
        self.reload_nominal_parameters_if_changed()
        angles = normalize_joint_configs(
            np.asarray(joint_angles, dtype=float).reshape(1, 6),
            joint_unit,
        )
        zero = zero_error_vector(self._geometric_parameters)
        return self._model.position(angles[0], zero, self._geometric_parameters)

    def compute_nominal_pose(
        self,
        joint_angles: list[float] | np.ndarray,
        *,
        joint_unit: str = "auto",
    ) -> np.ndarray:
        """Compute nominal TCP pose ``x, y, z, rx, ry, rz`` from joint angles.

        Position is returned in meters and orientation in intrinsic xyz Euler
        radians, matching the nominal model transform helper.
        """
        self.reload_nominal_parameters_if_changed()
        angles = normalize_joint_configs(
            np.asarray(joint_angles, dtype=float).reshape(1, 6),
            joint_unit,
        )
        zero = zero_error_vector(self._geometric_parameters)
        transform = self._model.transform(angles[0], zero, self._geometric_parameters)
        rpy = Rotation.from_matrix(transform[:3, :3]).as_euler("xyz")
        return np.concatenate([transform[:3, 3], rpy])

    def compute_nominal_positions(
        self,
        joint_configs: np.ndarray,
        *,
        joint_unit: str = "auto",
    ) -> np.ndarray:
        """Compute nominal TCP positions for multiple joint configurations."""
        self.reload_nominal_parameters_if_changed()
        joints = normalize_joint_configs(joint_configs, joint_unit)
        zero = zero_error_vector(self._geometric_parameters)
        return self._model.batch_positions(joints, zero, self._geometric_parameters)

    def solve_nominal_joint_angles_for_pose(
        self,
        pose_xyzrpy: list[float] | np.ndarray,
        *,
        initial_joint_angles: list[float] | np.ndarray | None = None,
        initial_unit: str = "auto",
        output_unit: str = "degrees",
        max_nfev: int = 80,
    ) -> np.ndarray:
        """Estimate one nominal joint configuration for a target TCP pose.

        This is a local least-squares helper for workstation pose previews.  It
        is deliberately small and is not used by the S1 identification path.
        """
        self.reload_nominal_parameters_if_changed()
        target = np.asarray(pose_xyzrpy, dtype=float).reshape(6)
        if initial_joint_angles is None:
            initial_joint_angles = np.zeros(6, dtype=float)
            initial_unit = "radians"
        initial = normalize_joint_configs(
            np.asarray(initial_joint_angles, dtype=float).reshape(1, 6),
            initial_unit,
        )[0]
        target_rotation = Rotation.from_euler("xyz", target[3:]).as_matrix()
        zero = zero_error_vector(self._geometric_parameters)

        def residual(joint_values: np.ndarray) -> np.ndarray:
            transform = self._model.transform(joint_values, zero, self._geometric_parameters)
            position_error = transform[:3, 3] - target[:3]
            rotation_error = (
                Rotation.from_matrix(target_rotation.T @ transform[:3, :3]).as_rotvec()
            )
            return np.concatenate([position_error, 0.25 * rotation_error])

        result = least_squares(
            residual,
            initial,
            method="trf",
            max_nfev=max(10, int(max_nfev)),
            xtol=1.0e-9,
            ftol=1.0e-9,
            gtol=1.0e-9,
        )
        output = np.asarray(result.x, dtype=float).reshape(6)
        if (output_unit or "radians").lower() in {"deg", "degree", "degrees"}:
            return np.rad2deg(output)
        return output

    def compute_predicted_position(
        self,
        joint_angles: list[float] | np.ndarray,
        parameter_values: dict[str, float] | None = None,
        *,
        joint_unit: str = "auto",
        confidence: float | None = None,
        tolerance_mm: float | None = None,
    ) -> CurrentAccuracyState:
        """Predict actual TCP position and current positioning error.

        The positioning error is defined as:
        ``identified_model_position - nominal_model_position``.
        """
        if parameter_values is None:
            self.reload_nominal_parameters_if_changed()
        elif parameter_values is not None:
            self.reload_nominal_parameters_if_changed()
        conf = self._active_confidence if confidence is None else float(confidence)
        joints = normalize_joint_configs(
            np.asarray(joint_angles, dtype=float).reshape(1, 6),
            joint_unit,
        )
        zero = zero_error_vector(self._geometric_parameters)
        nominal = self._model.position(joints[0], zero, self._geometric_parameters)
        if parameter_values is not None:
            identified_model = self._model_from_error_parameters(parameter_values)
        else:
            identified_model = self._active_identified_model or self._model
        predicted = identified_model.position(joints[0], zero, self._geometric_parameters)
        error = predicted - nominal
        thresholds = self._load_thresholds()
        limit_mm = (
            thresholds["position_rms_limit_mm"]
            if tolerance_mm is None
            else max(float(tolerance_mm), 0.0)
        )
        tolerance_m = limit_mm / 1000.0
        metrics = evaluate_position_errors(error.reshape(1, 3), tolerance_m)
        norm_mm = float(np.linalg.norm(error) * 1000.0)
        health = evaluate_positioning_health(norm_mm, limit_mm)
        return CurrentAccuracyState(
            nominal_position=nominal,
            predicted_position=predicted,
            positioning_error=error,
            error_norm_mm=norm_mm,
            rms_mm=float(metrics.rms * 1000.0),
            max_error_mm=float(metrics.max_error * 1000.0),
            over_tolerance_rate=float(metrics.over_tolerance_rate),
            confidence=float(conf),
            health_score=float(health.score),
            health_level=health.level,
            health_message=health.message,
        )

    def load_calibration_data(self, path: str | Path) -> dict[str, np.ndarray]:
        """Load one processed pkl identification dataset."""
        return load_dataset(path)

    def load_identification_data(self, paths: str | Path | Iterable[str | Path]) -> dict[str, Any]:
        """Load processed pkl data, or pack one raw CSV/TXT pair before loading."""
        path_list = _coerce_paths(paths)
        if not path_list:
            raise ValueError("At least one identification data file is required.")
        dataset_paths = self._prepare_identification_dataset_paths(path_list)
        datasets = [load_dataset(path) for path in dataset_paths]
        merged: dict[str, Any] = {
            "joints": np.concatenate([np.asarray(data["joints"], dtype=float) for data in datasets]),
            "measured_positions": np.concatenate(
                [np.asarray(data["measured_positions"], dtype=float) for data in datasets]
            ),
            "dataset_paths": [str(path.resolve()) for path in dataset_paths],
            "sample_counts": [int(len(data["joints"])) for data in datasets],
        }
        if dataset_paths != path_list:
            merged["raw_dataset_paths"] = [str(path.resolve()) for path in path_list]
        for key in ("payloads", "directions", "joint_torques"):
            if all(key in data for data in datasets):
                merged[key] = np.concatenate([np.asarray(data[key], dtype=float) for data in datasets])
        return merged

    def _prepare_identification_dataset_paths(self, paths: list[Path]) -> list[Path]:
        pkl_paths, csv_paths, txt_paths = split_calibration_input_paths(paths)
        if pkl_paths and not csv_paths and not txt_paths:
            return pkl_paths
        if not pkl_paths and len(csv_paths) == 1 and len(txt_paths) == 1:
            output_path = default_packed_dataset_path(
                self.project_root,
                csv_paths[0],
                txt_paths[0],
            )
            return [pack_raw_calibration_pair(csv_paths[0], txt_paths[0], output_path)]
        raise ValueError(
            "Identification data must be either one or more .pkl/.pickle files, "
            "or exactly one raw .csv file plus one raw .txt file."
        )

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
        initial_parameter_path: str | Path | None = None,
    ) -> CalibrationResult:
        """Run S1 identification and return model, fit, and positioning metrics."""
        opts = options or IdentificationOptions()
        if _method_key(opts) not in {"S1", "S1_DEBUG", "DEBUG", "DEBUG_FAST"}:
            return CalibrationResult(False, f"Unsupported identification method: {opts.method}")

        try:
            joints = normalize_joint_configs(joint_configs, joint_unit)
            measured = np.asarray(measured_positions, dtype=float).reshape(-1, 3)
            _validate_dataset(joints, measured)
            payload_array = _subset_or_none(payloads, np.arange(len(joints)), len(joints))
            direction_array = _subset_or_none(directions, np.arange(len(joints)), len(joints))
            initial_path = (
                Path(initial_parameter_path).resolve()
                if initial_parameter_path is not None
                else None
            )
            initial_nominal_robot = (
                self.load_initial_nominal_robot(initial_path)
                if initial_path is not None
                else None
            )
            run_model = (
                self._model_from_nominal_dict(initial_nominal_robot)
                if initial_nominal_robot is not None
                else self._model
            )
            result = self._run_s1(
                joints,
                measured,
                payloads=payload_array,
                directions=direction_array,
                dataset_paths=[str(Path(path)) for path in dataset_paths or []],
                options=opts,
                model=run_model,
                nominal_robot=initial_nominal_robot,
                initial_vector=None,
                initial_parameter_path=initial_path,
            )
            if result.success and activate:
                self.set_active_parameters(
                    result.parameter_values,
                    confidence=result.confidence,
                    identified_robot=result.identified_robot,
                )
            return result
        except Exception as exc:  # noqa: BLE001
            return CalibrationResult(
                False,
                f"{_result_method(opts)} identification failed: {exc}",
                method=_result_method(opts),
            )

    def identify_from_files(
        self,
        paths: str | Path | Iterable[str | Path],
        *,
        options: IdentificationOptions | None = None,
        initial_parameter_path: str | Path | None = None,
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
            initial_parameter_path=initial_parameter_path,
        )

    def set_active_parameters(
        self,
        parameter_values: dict[str, float],
        *,
        confidence: float = 100.0,
        identified_robot: dict[str, Any] | None = None,
    ) -> None:
        """Set the active complete identified model used for live prediction."""
        self._active_parameter_source_path = None
        self._active_error_vector = self.error_vector_from_values(parameter_values)
        self._active_parameter_values = vector_to_named_dict(
            self._active_error_vector, self._geometric_parameters
        )
        robot = (
            identified_robot
            if identified_robot is not None
            else self._identified_robot_from_error_parameters(parameter_values)
        )
        self._active_identified_robot = _plain_dict(robot)
        if identified_robot is not None:
            self._active_identified_model = self._model_from_identified_robot_for_fk(
                identified_robot
            )
        else:
            self._active_identified_model = self._model_from_error_parameters(
                self._active_parameter_values
            )
        self._active_confidence = float(confidence)

    def load_active_parameters(self, path: str | Path) -> dict[str, Any]:
        """Load an identified parameter YAML file and activate it."""
        from core.calibration_persistence import load_identification_result

        parameter_path = Path(path).resolve()
        self.reload_nominal_parameters_if_changed()
        loaded = load_identification_result(parameter_path)
        self.set_active_parameters(
            loaded["error_parameters"],
            confidence=float(loaded.get("confidence", 100.0)),
            identified_robot=loaded["identified_robot"],
        )
        self._active_parameter_source_path = parameter_path
        return loaded

    def _current_nominal_robot_config(self) -> dict[str, Any]:
        from core.nominal_parameter_service import NominalParameterService

        service = NominalParameterService(
            self.project_root,
            nominal_path=self.nominal_config_path,
        )
        return _plain_dict(service.load_nominal())

    def _identified_robot_from_error_parameters(
        self,
        parameter_values: dict[str, Any],
        nominal_robot: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        from core.nominal_parameter_service import nominal_after_applying_error_parameters

        return nominal_after_applying_error_parameters(
            nominal_robot if nominal_robot is not None else self._current_nominal_robot_config(),
            parameter_values,
        )

    def _model_from_error_parameters(
        self,
        parameter_values: dict[str, Any],
    ) -> MultiSourceRobotModel:
        return self._model_from_nominal_dict(
            self._identified_robot_from_error_parameters(parameter_values)
        )

    def _model_from_identified_robot_for_fk(
        self,
        identified_robot: dict[str, Any],
    ) -> MultiSourceRobotModel:
        current = self._current_nominal_robot_config()
        source = _plain_dict(identified_robot)
        mdh = source.get("mdh")
        if not isinstance(mdh, dict):
            raise KeyError("identified_robot.mdh is required for robot FK.")
        robot = _plain_dict(current)
        robot["mdh"] = _plain_dict(mdh)
        return self._model_from_nominal_dict(robot)

    def _model_from_nominal_dict(self, nominal_robot: dict[str, Any]) -> MultiSourceRobotModel:
        return MultiSourceRobotModel(load_nominal_robot(nominal_robot))

    def error_vector_from_values(self, values: dict[str, float] | None) -> np.ndarray:
        """Convert a name/value parameter mapping into the geometry33 vector."""
        vector = np.zeros(len(self._geometric_parameters), dtype=float)
        source = values or {}
        for index, parameter in enumerate(self._geometric_parameters):
            vector[index] = float(source.get(parameter.name, 0.0))
        return vector

    def load_initial_parameter_vector(self, path: str | Path) -> np.ndarray:
        """Load initial calibration parameters as a geometry error vector.

        The YAML may contain ``initial_parameters``/``initial_robot`` for a
        standalone initial model, ``identified_robot`` from a previous result,
        ``nominal_robot`` for direct model values, or flat ``error_parameters``.
        Absolute MD-H, base, and hand-eye/tool values are converted into deltas
        relative to the current nominal robot.
        """
        parameter_path = Path(path).resolve()
        data = yaml.safe_load(parameter_path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise TypeError(f"Initial parameter YAML must be a mapping: {parameter_path}")
        values = self._initial_parameter_values_from_document(data)
        return self.error_vector_from_values(values)

    def load_initial_nominal_robot(self, path: str | Path) -> dict[str, Any]:
        """Load the nominal robot model used as the baseline for identification.

        ``initial_parameter_path`` is a nominal-parameter file, not an initial
        error vector.  The S1 optimizer estimates errors relative to this
        model, so the identified model saved after fitting is:
        ``initial_nominal_robot + identified_error_parameters``.

        For backward compatibility, a document containing only flat
        ``error_parameters`` is treated as a previous delta from the current
        nominal model and converted into a nominal baseline first.
        """
        parameter_path = Path(path).resolve()
        data = yaml.safe_load(parameter_path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise TypeError(f"Initial nominal YAML must be a mapping: {parameter_path}")
        return self._initial_nominal_robot_from_document(data)

    def _initial_parameter_values_from_document(self, data: dict[str, Any]) -> dict[str, float]:
        values: dict[str, float] = {}
        flat = _extract_initial_flat_error_parameters(data)
        if flat:
            values.update({str(key): float(value or 0.0) for key, value in flat.items()})

        robot = _extract_initial_robot(data)
        if robot is not None:
            values.update(self._initial_robot_to_error_values(robot))
        if not values:
            raise KeyError(
                "Initial parameter YAML must contain initial_parameters/initial_robot/"
                "identified_robot/nominal_robot or error_parameters."
            )
        return values

    def _initial_nominal_robot_from_document(self, data: dict[str, Any]) -> dict[str, Any]:
        robot = _extract_initial_robot(data)
        if robot is not None:
            return self._merge_initial_nominal_robot(robot)

        flat = _extract_initial_flat_error_parameters(data)
        if flat:
            values = {str(key): float(value or 0.0) for key, value in flat.items()}
            return self._identified_robot_from_error_parameters(values)

        raise KeyError(
            "Initial nominal YAML must contain initial_parameters/initial_robot/"
            "identified_robot/nominal_robot, or previous error_parameters."
        )

    def _merge_initial_nominal_robot(self, robot: dict[str, Any]) -> dict[str, Any]:
        current = _plain_dict(self._current_nominal_robot_config())
        current.setdefault("base_xyz", [0.0, 0.0, 0.0])
        current.setdefault("base_rpy", [0.0, 0.0, 0.0])
        source = _plain_dict(robot)
        merged = _plain_dict(current)

        for source_key, target_key in (
            ("base_xyz", "base_xyz"),
            ("base_rpy", "base_rpy"),
            ("tool_xyz", "tool_xyz"),
            ("tool_rpy", "tool_rpy"),
        ):
            value = _first_initial_vector3(source, source_key)
            if value is not None:
                merged[target_key] = _float_array(value, 3, source_key).tolist()

        mdh = source.get("mdh")
        if isinstance(mdh, dict):
            target_mdh = _plain_dict(merged.get("mdh", {}))
            for key in ("alpha", "a", "d", "theta_offset"):
                if key in mdh:
                    target_mdh[key] = _float_array(mdh[key], 6, f"mdh.{key}").tolist()
            merged["mdh"] = target_mdh

        if "joint_limits" in source:
            merged["joint_limits"] = _plain_value(source["joint_limits"])
        return merged

    def _initial_robot_to_error_values(self, robot: dict[str, Any]) -> dict[str, float]:
        current = self._current_nominal_robot_config()
        values: dict[str, float] = {}
        mdh = robot.get("mdh")
        if isinstance(mdh, dict):
            for key, prefix in (
                ("alpha", "delta_alpha"),
                ("a", "delta_a"),
                ("d", "delta_d"),
                ("theta_offset", "delta_theta"),
            ):
                if key in mdh:
                    incoming = _float_array(mdh[key], 6, f"mdh.{key}")
                    baseline = _float_array(current["mdh"][key], 6, f"current.mdh.{key}")
                    for index, value in enumerate(incoming - baseline, start=1):
                        values[f"{prefix}_{index}"] = float(value)

        for source_key, prefix in (
            ("base_xyz", "delta_Bt"),
            ("base_rpy", "delta_Bu"),
            ("tool_xyz", "delta_Tt"),
            ("tool_rpy", "delta_Tu"),
        ):
            source_value = _first_initial_vector3(robot, source_key)
            if source_value is None:
                continue
            incoming = _float_array(source_value, 3, source_key)
            baseline = (
                np.zeros(3, dtype=float)
                if source_key.startswith("base_")
                else _float_array(current[source_key], 3, f"current.{source_key}")
            )
            for axis, value in zip(("x", "y", "z"), incoming - baseline, strict=True):
                values[f"{prefix}{axis}"] = float(value)
        return values

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
        model: MultiSourceRobotModel,
        nominal_robot: dict[str, Any] | None,
        initial_vector: np.ndarray | None,
        initial_parameter_path: Path | None,
    ) -> CalibrationResult:
        parameters = self._geometric_parameters
        zero = zero_error_vector(parameters)
        nominal = model.batch_positions(joints, zero, parameters, payloads, directions)
        active = list(range(len(parameters)))
        start_vector = _initial_vector_or_zero(initial_vector, len(parameters))

        if _debug_fast_enabled(options):
            return self._run_s1_debug_fast(
                joints,
                measured,
                nominal,
                payloads=payloads,
                directions=directions,
                dataset_paths=dataset_paths,
                options=options,
                model=model,
                nominal_robot=nominal_robot,
                initial_vector=start_vector,
                initial_parameter_path=initial_parameter_path,
            )

        independent_indices: list[int] = []
        try:
            redundancy = select_independent_parameters(
                model,
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

        partition = self._build_s1_partition(
            joints,
            start_vector,
            payloads,
            directions,
            options,
            model=model,
        )
        lambdas = _lambda_grid(options)
        cv_scores = self._evaluate_s1_cv(
            joints,
            measured,
            payloads,
            directions,
            lambdas,
            options,
            start_vector,
            model=model,
        )
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
            model,
            joints,
            measured,
            parameters,
            partition,
            lambda_value=selected_lambda,
            active_indices=active,
            payloads=payloads,
            directions=directions,
            initial_vector=start_vector,
            max_nfev=options.max_nfev,
        )
        predicted = model.batch_positions(
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
            initial_parameter_path,
            nominal_robot,
        )

    def _run_s1_debug_fast(
        self,
        joints: np.ndarray,
        measured: np.ndarray,
        nominal: np.ndarray,
        *,
        payloads: np.ndarray | float | None,
        directions: np.ndarray | None,
        dataset_paths: list[str],
        options: IdentificationOptions,
        model: MultiSourceRobotModel,
        nominal_robot: dict[str, Any] | None,
        initial_vector: np.ndarray,
        initial_parameter_path: Path | None,
    ) -> CalibrationResult:
        parameters = self._geometric_parameters
        selected_lambda = max(float(options.fixed_lambda), 0.0)
        fit = fit_l2_lm(
            model,
            joints,
            measured,
            parameters,
            list(range(len(parameters))),
            lambda_value=selected_lambda,
            payloads=payloads,
            directions=directions,
            initial_vector=initial_vector,
            max_nfev=options.max_nfev,
            norm="l2",
        )
        predicted = model.batch_positions(
            joints,
            fit.vector,
            parameters,
            payloads,
            directions,
        )
        fit_errors_mm = np.linalg.norm(predicted - measured, axis=1) * 1000.0
        cv_scores = [
            {
                "lambda": selected_lambda,
                "fold_rmse_mm": [],
                "mean_rmse_mm": float(np.sqrt(np.mean(fit_errors_mm**2))),
                "max_rmse_mm": float(np.max(fit_errors_mm)),
                "std_rmse_mm": 0.0,
                "mode": "debug_fast_no_cv",
            }
        ]
        return self._build_result(
            fit,
            nominal,
            predicted,
            measured,
            selected_lambda,
            cv_scores,
            None,
            [],
            dataset_paths,
            options,
            initial_parameter_path,
            nominal_robot,
        )

    def _build_s1_partition(
        self,
        joints: np.ndarray,
        vector: np.ndarray,
        payloads: np.ndarray | float | None,
        directions: np.ndarray | None,
        options: IdentificationOptions,
        *,
        model: MultiSourceRobotModel | None = None,
    ) -> SubspaceIdentifiabilityPartition:
        parameters = self._geometric_parameters
        run_model = model or self._model
        pose_metrics = compute_pose_identifiability_metrics(
            run_model,
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
            run_model,
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
        initial_vector: np.ndarray | None,
        *,
        model: MultiSourceRobotModel | None = None,
    ) -> list[dict[str, Any]]:
        folds = random_folds(len(joints), options.cv_folds, seed=options.seed + 11)
        run_model = model or self._model
        start_vector = _initial_vector_or_zero(initial_vector, len(self._geometric_parameters))
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
                        start_vector,
                        train_payloads,
                        train_directions,
                        options,
                        model=run_model,
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
                    run_model,
                    joints[train_indices],
                    measured[train_indices],
                    self._geometric_parameters,
                    partition,
                    lambda_value=float(lambda_value),
                    active_indices=active,
                    payloads=train_payloads,
                    directions=train_directions,
                    initial_vector=start_vector,
                    max_nfev=options.max_nfev,
                )
                predicted = run_model.batch_positions(
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
        partition: SubspaceIdentifiabilityPartition | None,
        independent_indices: list[int],
        dataset_paths: list[str],
        options: IdentificationOptions,
        initial_parameter_path: Path | None,
        nominal_robot_config: dict[str, Any] | None = None,
    ) -> CalibrationResult:
        fit_vectors = predicted - measured
        fit_errors_mm = np.linalg.norm(fit_vectors, axis=1) * 1000.0
        positioning_vectors = predicted - nominal
        positioning_errors_mm = np.linalg.norm(positioning_vectors, axis=1) * 1000.0
        nominal_errors_mm = np.linalg.norm(nominal - measured, axis=1) * 1000.0
        parameter_values = vector_to_named_dict(fit.vector, self._geometric_parameters)
        nominal_robot = (
            _plain_dict(nominal_robot_config)
            if nominal_robot_config is not None
            else self._current_nominal_robot_config()
        )
        identified_robot = self._identified_robot_from_error_parameters(
            parameter_values,
            nominal_robot=nominal_robot,
        )
        best_cv = min(cv_scores, key=lambda row: (row["max_rmse_mm"], row["mean_rmse_mm"], row["lambda"]))
        confidence = _confidence_from_cv(
            float(best_cv["max_rmse_mm"]),
            self._load_thresholds()["position_rms_limit_mm"],
        )
        if partition is None:
            subspace_summary = {
                "K": 1,
                "cluster_sizes": [int(len(measured))],
                "order": [0],
                "summaries": [
                    {
                        "subspace": 0,
                        "sample_count": int(len(measured)),
                        "mode": "debug_fast_no_subspace",
                    }
                ],
            }
        else:
            subspace_summary = {
                "K": int(partition.K),
                "cluster_sizes": [int(size) for size in partition.cluster_sizes],
                "order": [int(item) for item in partition.order],
                "summaries": partition.subspace_summaries,
            }
        return CalibrationResult(
            success=bool(fit.success),
            message=str(fit.message),
            method=_result_method(options),
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
            nominal_robot=nominal_robot,
            identified_robot=identified_robot,
            metadata={
                "options": {
                    "max_nfev": int(options.max_nfev),
                    "cv_folds": int(options.cv_folds),
                    "lambda_count": 0
                    if _debug_fast_enabled(options)
                    else int(len(_lambda_grid(options))),
                    "fixed_lambda": float(options.fixed_lambda),
                    "debug_fast": bool(_debug_fast_enabled(options)),
                    "jacobian_method": options.jacobian_method,
                },
                "initial_parameter_path": str(initial_parameter_path)
                if initial_parameter_path is not None
                else "",
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
    from core.nominal_parameter_service import DEFAULT_NOMINAL_ROBOT, NominalParameterService

    if path is None:
        return _plain_dict(DEFAULT_NOMINAL_ROBOT)
    service = NominalParameterService(
        path.parent.parent,
        nominal_path=path,
    )
    return service.load_nominal()


def _plain_dict(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("Robot parameters must be a mapping.")
    return {str(key): _plain_value(item) for key, item in value.items()}


def _plain_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_plain_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _file_signature(path: Path | None) -> tuple[int, int] | None:
    if path is None or not path.exists():
        return None
    stat = path.stat()
    return (int(stat.st_mtime_ns), int(stat.st_size))


def _coerce_paths(paths: str | Path | Iterable[str | Path]) -> list[Path]:
    if isinstance(paths, (str, Path)):
        return [Path(paths).resolve()]
    return [Path(path).resolve() for path in paths]


def _method_key(options: IdentificationOptions) -> str:
    return (options.method or "S1").upper()


def _debug_fast_enabled(options: IdentificationOptions) -> bool:
    return bool(options.debug_fast) or _method_key(options) in {"S1_DEBUG", "DEBUG", "DEBUG_FAST"}


def _result_method(options: IdentificationOptions) -> str:
    return "S1_DEBUG" if _debug_fast_enabled(options) else "S1"


def _initial_vector_or_zero(vector: np.ndarray | None, parameter_count: int) -> np.ndarray:
    if vector is None:
        return np.zeros(parameter_count, dtype=float)
    return np.asarray(vector, dtype=float).reshape(parameter_count).copy()


def _extract_initial_flat_error_parameters(data: dict[str, Any]) -> dict[str, Any] | None:
    identification = data.get("identification")
    if isinstance(identification, dict) and isinstance(
        identification.get("error_parameters"), dict
    ):
        return identification["error_parameters"]
    calibration = data.get("calibration")
    if isinstance(calibration, dict) and isinstance(calibration.get("error_parameters"), dict):
        return calibration["error_parameters"]
    if isinstance(data.get("error_parameters"), dict):
        return data["error_parameters"]
    return None


def _extract_initial_robot(data: dict[str, Any]) -> dict[str, Any] | None:
    for section_name in ("initial_parameters", "initial_robot", "identified_robot", "nominal_robot"):
        section = data.get(section_name)
        if isinstance(section, dict):
            return section
    identification = data.get("identification")
    if isinstance(identification, dict):
        identified = identification.get("identified_robot")
        if isinstance(identified, dict):
            return identified
        nominal = identification.get("nominal_robot")
        if isinstance(nominal, dict):
            return nominal
    if isinstance(data.get("mdh"), dict):
        return data
    return None


def _first_initial_vector3(mapping: dict[str, Any], key: str) -> Any | None:
    aliases = {
        "tool_xyz": ("tool_xyz", "hand_eye_xyz", "target_xyz", "target_offset_xyz"),
        "tool_rpy": ("tool_rpy", "hand_eye_rpy", "target_rpy", "target_offset_rpy"),
    }.get(key, (key,))
    for alias in aliases:
        if alias in mapping:
            return mapping[alias]
    return None


def _float_array(value: Any, count: int, name: str) -> np.ndarray:
    values = np.asarray(value, dtype=float).reshape(-1)
    if values.size != count:
        raise ValueError(f"{name} must contain {count} numeric values, got {values.size}.")
    return values


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
