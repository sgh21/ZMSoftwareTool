"""Long-term robot model degradation monitoring from fixed-board PnP observations."""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from core.accuracy_evaluator import confidence_from_uncertainty
from core.parameter_repository import ParameterFileRepository


DEFAULT_T_TOOL_CAMERA = np.array(
    [
        [0.999934, 0.003458, -0.01094, 0.000049],
        [-0.003407, 0.999983, 0.004657, -0.087182],
        [0.010956, -0.00462, 0.999929, 0.072556],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
DEFAULT_CAMERA_K = np.array(
    [
        [3675.2707, 0.0, 1231.6813],
        [0.0, 3674.6618, 1073.9725],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
DEFAULT_CAMERA_D = np.array(
    [-0.11570, 0.32350, 0.00100, -0.00060, -2.9623],
    dtype=np.float64,
)
DEFAULT_BOARD_GRID = (39, 34)
DEFAULT_SQUARE_SIZE_MM = 0.5
DEFAULT_ORIENTATION_WEIGHT_MM_PER_RAD = 100.0
IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}
DEFAULT_MONITORING_CONFIG_RELATIVE = Path("config") / "model_monitoring.yaml"
ProgressReporter = Callable[[str, str, int], None]


@dataclass(frozen=True)
class MonitoringProgress:
    stage: str
    current: int
    total: int
    message: str


@dataclass(frozen=True)
class CameraConfig:
    K: np.ndarray
    D: np.ndarray
    board_grid: tuple[int, int] = DEFAULT_BOARD_GRID
    square_size_mm: float = DEFAULT_SQUARE_SIZE_MM
    r_diag_preference: str = "none"


@dataclass(frozen=True)
class MonitoringConfig:
    T_tool_camera: np.ndarray
    camera: CameraConfig
    orientation_weight_mm_per_rad: float = DEFAULT_ORIENTATION_WEIGHT_MM_PER_RAD
    position_rms_limit_mm: float = 0.5


@dataclass(frozen=True)
class ObservationSet:
    source_path: Path
    names: list[str]
    transforms: np.ndarray
    reproj_mean_px: np.ndarray | None = None

    def __post_init__(self) -> None:
        transforms = np.asarray(self.transforms, dtype=np.float64)
        if transforms.ndim != 3 or transforms.shape[1:] != (4, 4):
            raise ValueError(f"T_cam_board must be Nx4x4, got {transforms.shape}")
        if len(self.names) != transforms.shape[0]:
            raise ValueError("Observation name count must match transform count.")


@dataclass(frozen=True)
class PoseDrift:
    name: str
    delta_transform: np.ndarray
    log_vector: np.ndarray
    position_drift_mm: float
    orientation_drift_deg: float
    reference_reproj_mean_px: float | None = None
    current_reproj_mean_px: float | None = None


@dataclass(frozen=True)
class DegradationEvaluationResult:
    reference_source: Path
    current_source: Path
    sample_count: int
    pose_drifts: list[PoseDrift]
    stacked_residual: np.ndarray
    position_drift_rms_mm: float
    orientation_drift_rms_deg: float
    combined_drift_mm: float
    mahalanobis_d2: float | None
    confidence_before: float
    confidence_after: float
    position_uncertainty_before_mm: float
    position_uncertainty_after_mm: float
    active_model_path: Path | None = None
    backup_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def format_log(self) -> str:
        lines = [
            "模型退化评估完成",
            f"参考观测: {self.reference_source}",
            f"当前观测: {self.current_source}",
            f"有效匹配位姿: {self.sample_count}",
            f"整体位置漂移 RMS: {self.position_drift_rms_mm:.6f} mm",
            f"整体姿态漂移 RMS: {self.orientation_drift_rms_deg:.6f} deg",
            f"综合等效漂移: {self.combined_drift_mm:.6f} mm",
        ]
        if self.mahalanobis_d2 is not None:
            lines.append(f"Mahalanobis D^2: {self.mahalanobis_d2:.6f}")
        else:
            lines.append("Mahalanobis D^2: 未配置健康基线，已跳过")
        lines.extend(
            [
                f"模型置信度: {self.confidence_before:.3f}% -> {self.confidence_after:.3f}%",
                "定位不确定性 RMS: "
                f"{self.position_uncertainty_before_mm:.6f} mm -> "
                f"{self.position_uncertainty_after_mm:.6f} mm",
                "",
                "逐位姿漂移:",
            ]
        )
        for drift in self.pose_drifts:
            reproj = ""
            if drift.reference_reproj_mean_px is not None or drift.current_reproj_mean_px is not None:
                ref = "--" if drift.reference_reproj_mean_px is None else f"{drift.reference_reproj_mean_px:.4f}"
                cur = "--" if drift.current_reproj_mean_px is None else f"{drift.current_reproj_mean_px:.4f}"
                reproj = f", reproj_px(ref/current)={ref}/{cur}"
            lines.append(
                f"- {drift.name}: position={drift.position_drift_mm:.6f} mm, "
                f"orientation={drift.orientation_drift_deg:.6f} deg{reproj}"
            )
        return "\n".join(lines)


class ModelDegradationMonitoringService:
    """Evaluate fixed-board PnP time-series drift and update model state."""

    def __init__(self, project_root: str | Path) -> None:
        self.project_root = Path(project_root).resolve()
        self._parameter_repository = ParameterFileRepository(self.project_root)

    def load_monitoring_config(
        self,
        calibration_result_path: str | Path | None = None,
        monitoring_config_path: str | Path | None = None,
    ) -> MonitoringConfig:
        document = self._load_monitoring_document(monitoring_config_path)
        monitoring = _monitoring_section(document)
        if not monitoring:
            legacy_document = self._load_document(calibration_result_path)
            legacy_section = _identification_section(legacy_document)
            monitoring = (
                legacy_section.get("monitoring", {})
                if isinstance(legacy_section, dict)
                else {}
            )
        if not isinstance(monitoring, dict):
            monitoring = {}
        hand_eye = (
            monitoring.get("hand_eye", {})
            if isinstance(monitoring.get("hand_eye"), dict)
            else {}
        )
        camera = monitoring.get("camera", {}) if isinstance(monitoring.get("camera"), dict) else {}
        evaluation = (
            monitoring.get("evaluation", {})
            if isinstance(monitoring.get("evaluation"), dict)
            else {}
        )
        T_tool_camera = _matrix4(hand_eye.get("T_tool_camera", DEFAULT_T_TOOL_CAMERA), "T_tool_camera")
        K = _matrix3(camera.get("K", DEFAULT_CAMERA_K), "camera.K")
        D = np.asarray(camera.get("D", DEFAULT_CAMERA_D), dtype=np.float64).reshape(-1)
        board_grid = tuple(int(value) for value in camera.get("board_grid", DEFAULT_BOARD_GRID))
        if len(board_grid) != 2:
            raise ValueError("camera.board_grid must contain [cols, rows].")
        thresholds = self._load_thresholds()
        return MonitoringConfig(
            T_tool_camera=T_tool_camera,
            camera=CameraConfig(
                K=K,
                D=D,
                board_grid=(int(board_grid[0]), int(board_grid[1])),
                square_size_mm=float(camera.get("square_size_mm", DEFAULT_SQUARE_SIZE_MM)),
                r_diag_preference=str(camera.get("r_diag_preference", "none")),
            ),
            orientation_weight_mm_per_rad=float(
                evaluation.get(
                    "orientation_weight_mm_per_rad",
                    DEFAULT_ORIENTATION_WEIGHT_MM_PER_RAD,
                )
            ),
            position_rms_limit_mm=float(thresholds.get("position_rms_limit_mm", 0.5)),
        )

    def load_observations(
        self,
        path: str | Path,
        config: MonitoringConfig | None = None,
        *,
        source_label: str = "观测",
        progress_reporter: ProgressReporter | None = None,
    ) -> ObservationSet:
        source = Path(path).resolve()
        if not source.exists():
            raise FileNotFoundError(source)
        if source.is_dir():
            return self._load_observations_from_image_dir(
                source,
                config or self.load_monitoring_config(),
                source_label=source_label,
                progress_reporter=progress_reporter,
            )
        suffix = source.suffix.lower()
        if suffix == ".npz":
            observations = self._load_observations_from_npz(source)
            if progress_reporter is not None:
                progress_reporter(
                    f"加载{source_label}",
                    f"已加载{source_label} PnP 结果: {source.name}",
                    1,
                )
            return observations
        if suffix in {".yaml", ".yml", ".json"}:
            observations = self._load_observations_from_mapping(source)
            if progress_reporter is not None:
                progress_reporter(
                    f"加载{source_label}",
                    f"已加载{source_label} PnP 结果: {source.name}",
                    1,
                )
            return observations
        raise ValueError(f"Unsupported observation source: {source}")

    def evaluate(
        self,
        reference_path: str | Path,
        current_path: str | Path,
        *,
        calibration_result_path: str | Path | None = None,
        progress_callback: Callable[[MonitoringProgress], None] | None = None,
    ) -> DegradationEvaluationResult:
        total = self._progress_total(reference_path, current_path)
        progress_current = 0

        def report(stage: str, message: str, increment: int = 0) -> None:
            nonlocal progress_current
            progress_current = min(total, progress_current + max(0, int(increment)))
            if progress_callback is not None:
                progress_callback(
                    MonitoringProgress(
                        stage=stage,
                        current=progress_current,
                        total=total,
                        message=message,
                    )
                )

        report("准备评估", "正在读取模型评估配置...")
        config = self.load_monitoring_config(calibration_result_path)
        reference = self.load_observations(
            reference_path,
            config,
            source_label="参考观测",
            progress_reporter=report,
        )
        current = self.load_observations(
            current_path,
            config,
            source_label="当前观测",
            progress_reporter=report,
        )
        ref_indices, cur_indices, names = _matched_indices(reference.names, current.names)
        if not names:
            raise ValueError("Reference and current observations have no matched poses.")
        report("计算漂移", f"正在计算 {len(names)} 个匹配位姿的末端漂移...")

        T_tool_camera = config.T_tool_camera
        T_camera_tool = rigid_inverse(T_tool_camera)
        pose_drifts: list[PoseDrift] = []
        residuals: list[np.ndarray] = []
        for ref_idx, cur_idx, name in zip(ref_indices, cur_indices, names, strict=True):
            T_ref = reference.transforms[ref_idx]
            T_cur = current.transforms[cur_idx]
            delta = T_tool_camera @ T_ref @ rigid_inverse(T_cur) @ T_camera_tool
            log_vector = se3_log(delta)
            position_mm = float(np.linalg.norm(log_vector[:3]) * 1000.0)
            orientation_deg = float(np.linalg.norm(log_vector[3:]) * 180.0 / math.pi)
            residuals.append(log_vector)
            pose_drifts.append(
                PoseDrift(
                    name=name,
                    delta_transform=delta,
                    log_vector=log_vector,
                    position_drift_mm=position_mm,
                    orientation_drift_deg=orientation_deg,
                    reference_reproj_mean_px=_optional_index(reference.reproj_mean_px, ref_idx),
                    current_reproj_mean_px=_optional_index(current.reproj_mean_px, cur_idx),
                )
            )

        stacked = np.concatenate(residuals).astype(np.float64)
        position_values = np.asarray([item.position_drift_mm for item in pose_drifts], dtype=np.float64)
        orientation_values = np.asarray([item.orientation_drift_deg for item in pose_drifts], dtype=np.float64)
        position_rms = float(np.sqrt(np.mean(position_values**2)))
        orientation_rms_deg = float(np.sqrt(np.mean(orientation_values**2)))
        orientation_rms_rad = orientation_rms_deg * math.pi / 180.0
        combined = float(
            math.sqrt(
                position_rms**2
                + (config.orientation_weight_mm_per_rad * orientation_rms_rad) ** 2
            )
        )
        document = self._load_document(calibration_result_path)
        before_confidence = _current_confidence(document)
        before_uncertainty = _current_position_uncertainty(document)
        after_uncertainty = float(math.sqrt(before_uncertainty**2 + position_rms**2))
        after_confidence = recommended_confidence(
            before_confidence,
            after_uncertainty,
            config.position_rms_limit_mm,
        )
        mahalanobis_d2 = self._mahalanobis_d2(stacked, document)
        report("完成评估", "模型退化评估计算完成", 1)

        return DegradationEvaluationResult(
            reference_source=reference.source_path,
            current_source=current.source_path,
            sample_count=len(pose_drifts),
            pose_drifts=pose_drifts,
            stacked_residual=stacked,
            position_drift_rms_mm=position_rms,
            orientation_drift_rms_deg=orientation_rms_deg,
            combined_drift_mm=combined,
            mahalanobis_d2=mahalanobis_d2,
            confidence_before=before_confidence,
            confidence_after=after_confidence,
            position_uncertainty_before_mm=before_uncertainty,
            position_uncertainty_after_mm=after_uncertainty,
            active_model_path=self._resolve_model_path(calibration_result_path),
            metadata={
                "orientation_weight_mm_per_rad": config.orientation_weight_mm_per_rad,
                "position_rms_limit_mm": config.position_rms_limit_mm,
                "confidence_formula": "100 * limit_mm^2 / (limit_mm^2 + uncertainty_rmse_mm^2)",
                "uncertainty_formula": "sqrt(previous_uncertainty_rmse_mm^2 + position_drift_rms_mm^2)",
            },
        )

    def apply_recommended_update(
        self,
        calibration_result_path: str | Path,
        result: DegradationEvaluationResult,
    ) -> Path | None:
        path = Path(calibration_result_path).resolve()
        document = self._load_document(path)
        if document.get("kind") == "identified_model":
            record = result_to_yaml_record(result)
            self._parameter_repository.append_confidence_history(
                path,
                result.confidence_after,
                source="model_degradation_monitoring",
                reason=(
                    "combined_drift_mm="
                    f"{result.combined_drift_mm:.6f}; "
                    f"position_rms_mm={result.position_drift_rms_mm:.6f}"
                ),
                position_uncertainty_rmse_mm=result.position_uncertainty_after_mm,
                evaluation_record=record,
            )
            return None
        section = _identification_section(document, create=True)
        backup_path = self._backup_model(path)

        metrics = section.setdefault("metrics", {})
        if not isinstance(metrics, dict):
            raise TypeError("identification.metrics must be a mapping.")
        section["confidence"] = float(result.confidence_after)
        section["confidence_current"] = float(result.confidence_after)
        history = section.setdefault("confidence_history", [])
        if isinstance(history, list):
            history.append(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "value": float(result.confidence_after),
                    "source": "model_degradation_monitoring",
                    "reason": (
                        "legacy file update; "
                        f"combined_drift_mm={result.combined_drift_mm:.6f}"
                    ),
                }
            )
        metrics["position_uncertainty_rmse_mm"] = float(result.position_uncertainty_after_mm)
        monitoring = section.setdefault("monitoring", {})
        if not isinstance(monitoring, dict):
            raise TypeError("identification.monitoring must be a mapping.")
        monitoring["last_degradation_evaluation"] = result_to_yaml_record(
            result,
            backup_path=backup_path,
        )
        path.write_text(
            yaml.safe_dump(document, allow_unicode=True, default_flow_style=False, sort_keys=False),
            encoding="utf-8",
        )
        return backup_path

    def ensure_monitoring_defaults(
        self,
        monitoring_config_path: str | Path | None = None,
    ) -> Path:
        path = self._resolve_monitoring_config_path(monitoring_config_path)
        document = self._load_monitoring_document(path)
        if document.get("kind") == "camera_monitoring":
            payload = document.setdefault("payload", {})
            if not isinstance(payload, dict):
                raise TypeError("camera_monitoring payload must be a mapping.")
            monitoring = payload.setdefault("model_monitoring", {})
        else:
            monitoring = document.setdefault("model_monitoring", {})
        if not isinstance(monitoring, dict):
            raise TypeError("model_monitoring must be a mapping.")
        changed = self._ensure_monitoring_defaults(monitoring)
        if changed or not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                yaml.safe_dump(document, allow_unicode=True, default_flow_style=False, sort_keys=False),
                encoding="utf-8",
            )
        return path

    def _load_document(self, calibration_result_path: str | Path | None) -> dict[str, Any]:
        path = self._resolve_model_path(calibration_result_path)
        if not path.exists():
            return {}
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise TypeError(f"Calibration result YAML must be a mapping: {path}")
        return data

    def _load_monitoring_document(
        self,
        monitoring_config_path: str | Path | None,
    ) -> dict[str, Any]:
        path = self._resolve_monitoring_config_path(monitoring_config_path)
        if not path.exists():
            return {}
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise TypeError(f"Monitoring config YAML must be a mapping: {path}")
        return data

    def _resolve_monitoring_config_path(self, path: str | Path | None) -> Path:
        if path is None:
            self._parameter_repository.ensure_initial_versions()
            active = self._parameter_repository.active_path_for("camera_monitoring")
            if active is not None:
                return active
            return (self.project_root / DEFAULT_MONITORING_CONFIG_RELATIVE).resolve()
        return Path(path).resolve()

    def _resolve_model_path(self, path: str | Path | None) -> Path:
        if path is not None:
            return Path(path).resolve()
        self._parameter_repository.ensure_initial_versions()
        active = self._parameter_repository.active_path_for("identified_model")
        if active is not None:
            return active
        return (self.project_root / "config" / "calibration_result.yaml").resolve()

    def _load_observations_from_npz(self, path: Path) -> ObservationSet:
        data = np.load(path, allow_pickle=True)
        if "T_cam_board" not in data.files:
            raise ValueError(f"{path} does not contain T_cam_board.")
        transforms = np.asarray(data["T_cam_board"], dtype=np.float64)
        filenames = data["filenames"].astype(str).tolist() if "filenames" in data.files else []
        if not filenames:
            filenames = [f"pose_{index:04d}" for index in range(transforms.shape[0])]
        reproj = np.asarray(data["reproj_mean_px"], dtype=np.float64) if "reproj_mean_px" in data.files else None
        return ObservationSet(path, filenames, transforms, reproj)

    def _load_observations_from_mapping(self, path: Path) -> ObservationSet:
        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
        else:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise TypeError(f"Observation mapping must be a dict: {path}")
        if "observations" in data:
            return _observation_set_from_rows(path, data["observations"])
        if "T_cam_board" in data:
            transforms = np.asarray(data["T_cam_board"], dtype=np.float64)
            names = [str(item) for item in data.get("filenames", [])]
            if not names:
                names = [f"pose_{index:04d}" for index in range(transforms.shape[0])]
            reproj = np.asarray(data["reproj_mean_px"], dtype=np.float64) if "reproj_mean_px" in data else None
            return ObservationSet(path, names, transforms, reproj)
        sibling = _find_pose_npz_sibling(path)
        if sibling is not None:
            return self._load_observations_from_npz(sibling)
        raise ValueError(f"{path} does not contain PnP transforms.")

    def _load_observations_from_image_dir(
        self,
        path: Path,
        config: MonitoringConfig,
        *,
        source_label: str,
        progress_reporter: ProgressReporter | None = None,
    ) -> ObservationSet:
        try:
            import cv2
        except ModuleNotFoundError as exc:
            raise ImportError("图片目录 PnP 解算需要安装 opencv-python。") from exc
        from core.vision.pnp import estimate_pose_from_image

        image_paths = sorted(item for item in path.iterdir() if item.suffix.lower() in IMAGE_SUFFIXES)
        if not image_paths:
            raise ValueError(f"No images found in {path}.")
        transforms: list[np.ndarray] = []
        names: list[str] = []
        reproj: list[float] = []
        for index, image_path in enumerate(image_paths, start=1):
            image = cv2.imread(str(image_path))
            if image is None:
                if progress_reporter is not None:
                    progress_reporter(
                        f"{source_label} PnP",
                        f"{source_label} PnP {index}/{len(image_paths)}: {image_path.name} 读取失败",
                        1,
                    )
                continue
            pose = estimate_pose_from_image(
                image,
                config.camera.board_grid,
                config.camera.square_size_mm,
                config.camera.K,
                config.camera.D,
                r_diag_preference=config.camera.r_diag_preference,
            )
            if not pose.success or pose.T_cam_board is None:
                if progress_reporter is not None:
                    progress_reporter(
                        f"{source_label} PnP",
                        f"{source_label} PnP {index}/{len(image_paths)}: {image_path.name} 未检测到完整棋盘格",
                        1,
                    )
                continue
            transforms.append(pose.T_cam_board)
            names.append(image_path.name)
            reproj.append(float(pose.reproj_mean_px or 0.0))
            if progress_reporter is not None:
                progress_reporter(
                    f"{source_label} PnP",
                    f"{source_label} PnP {index}/{len(image_paths)}: {image_path.name}, "
                    f"重投影={float(pose.reproj_mean_px or 0.0):.4f}px",
                    1,
                )
        if not transforms:
            raise ValueError(f"No valid checkerboard PnP observations found in {path}.")
        return ObservationSet(
            path,
            names,
            np.stack(transforms, axis=0),
            np.asarray(reproj, dtype=np.float64),
        )

    def _backup_model(self, path: Path) -> Path:
        backup_dir = self.project_root / "storage" / "model_versions"
        backup_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"{path.stem}_before_degradation_{timestamp}{path.suffix}"
        shutil.copy2(path, backup_path)
        return backup_path

    def _ensure_monitoring_defaults(self, monitoring: dict[str, Any]) -> bool:
        changed = False
        hand_eye = monitoring.setdefault("hand_eye", {})
        if isinstance(hand_eye, dict):
            if "convention" not in hand_eye:
                hand_eye["convention"] = "E_T_C"
                changed = True
            if "T_tool_camera" not in hand_eye:
                hand_eye["T_tool_camera"] = DEFAULT_T_TOOL_CAMERA.tolist()
                changed = True
        camera = monitoring.setdefault("camera", {})
        if isinstance(camera, dict):
            defaults = {
                "K": DEFAULT_CAMERA_K.tolist(),
                "D": DEFAULT_CAMERA_D.tolist(),
                "board_grid": list(DEFAULT_BOARD_GRID),
                "square_size_mm": DEFAULT_SQUARE_SIZE_MM,
                "r_diag_preference": "none",
            }
            for key, value in defaults.items():
                if key not in camera:
                    camera[key] = value
                    changed = True
        evaluation = monitoring.setdefault("evaluation", {})
        if isinstance(evaluation, dict) and "orientation_weight_mm_per_rad" not in evaluation:
            evaluation["orientation_weight_mm_per_rad"] = DEFAULT_ORIENTATION_WEIGHT_MM_PER_RAD
            changed = True
        return changed

    def _mahalanobis_d2(self, residual: np.ndarray, document: dict[str, Any]) -> float | None:
        section = _identification_section(document)
        monitoring = section.get("monitoring", {}) if isinstance(section, dict) else {}
        baseline = monitoring.get("baseline", {}) if isinstance(monitoring, dict) else {}
        if not isinstance(baseline, dict):
            return None
        mean = baseline.get("mean")
        covariance = baseline.get("covariance")
        if mean is None or covariance is None:
            return None
        mean_arr = np.asarray(mean, dtype=np.float64).reshape(-1)
        cov_arr = np.asarray(covariance, dtype=np.float64)
        if mean_arr.shape != residual.shape or cov_arr.shape != (residual.size, residual.size):
            return None
        diff = residual - mean_arr
        inv_cov = np.linalg.pinv(cov_arr)
        return float(diff.T @ inv_cov @ diff)

    def _load_thresholds(self) -> dict[str, float]:
        defaults = {"position_rms_limit_mm": 0.5}
        path = self.project_root / "config" / "thresholds.yaml"
        if not path.exists():
            return defaults
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            return defaults
        accuracy = data.get("accuracy", {}) if isinstance(data, dict) else {}
        if isinstance(accuracy, dict) and "position_rms_limit_mm" in accuracy:
            defaults["position_rms_limit_mm"] = float(accuracy["position_rms_limit_mm"])
        return defaults

    def _progress_total(self, reference_path: str | Path, current_path: str | Path) -> int:
        return (
            self._observation_progress_units(reference_path)
            + self._observation_progress_units(current_path)
            + 1
        )

    def _observation_progress_units(self, path: str | Path) -> int:
        source = Path(path)
        if source.is_dir():
            return max(1, len([item for item in source.iterdir() if item.suffix.lower() in IMAGE_SUFFIXES]))
        return 1


def rigid_inverse(T: np.ndarray) -> np.ndarray:
    matrix = _matrix4(T, "T")
    output = np.eye(4, dtype=np.float64)
    R = matrix[:3, :3]
    p = matrix[:3, 3]
    output[:3, :3] = R.T
    output[:3, 3] = -R.T @ p
    return output


def se3_log(T: np.ndarray) -> np.ndarray:
    matrix = _matrix4(T, "T")
    R = matrix[:3, :3]
    p = matrix[:3, 3]
    phi = Rotation.from_matrix(R).as_rotvec()
    theta = float(np.linalg.norm(phi))
    omega = _skew(phi)
    if theta < 1.0e-10:
        v_inv = np.eye(3) - 0.5 * omega + (omega @ omega) / 12.0
    else:
        theta2 = theta * theta
        coeff = (1.0 / theta2) - ((1.0 + math.cos(theta)) / (2.0 * theta * math.sin(theta)))
        v_inv = np.eye(3) - 0.5 * omega + coeff * (omega @ omega)
    rho = v_inv @ p
    return np.concatenate([rho, phi]).astype(np.float64)


def recommended_confidence(
    current_confidence: float,
    uncertainty_rmse_mm: float,
    limit_mm: float,
) -> float:
    del current_confidence
    return confidence_from_uncertainty(uncertainty_rmse_mm, limit_mm)


def result_to_yaml_record(
    result: DegradationEvaluationResult,
    *,
    backup_path: Path | None = None,
) -> dict[str, Any]:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reference_source": str(result.reference_source),
        "current_source": str(result.current_source),
        "backup_path": str(backup_path) if backup_path is not None else "",
        "sample_count": int(result.sample_count),
        "position_drift_rms_mm": float(result.position_drift_rms_mm),
        "orientation_drift_rms_deg": float(result.orientation_drift_rms_deg),
        "combined_drift_mm": float(result.combined_drift_mm),
        "mahalanobis_d2": result.mahalanobis_d2,
        "confidence_before": float(result.confidence_before),
        "confidence_after": float(result.confidence_after),
        "position_uncertainty_before_mm": float(result.position_uncertainty_before_mm),
        "position_uncertainty_after_mm": float(result.position_uncertainty_after_mm),
        "metadata": _plain_value(result.metadata),
        "per_pose": [
            {
                "name": drift.name,
                "position_drift_mm": float(drift.position_drift_mm),
                "orientation_drift_deg": float(drift.orientation_drift_deg),
                "reference_reproj_mean_px": drift.reference_reproj_mean_px,
                "current_reproj_mean_px": drift.current_reproj_mean_px,
            }
            for drift in result.pose_drifts
        ],
    }


def _matched_indices(reference_names: list[str], current_names: list[str]) -> tuple[list[int], list[int], list[str]]:
    ref_map = {name: index for index, name in enumerate(reference_names)}
    cur_map = {name: index for index, name in enumerate(current_names)}
    common = [name for name in reference_names if name in cur_map]
    if common:
        return [ref_map[name] for name in common], [cur_map[name] for name in common], common
    count = min(len(reference_names), len(current_names))
    return list(range(count)), list(range(count)), [f"pose_{index:04d}" for index in range(count)]


def _observation_set_from_rows(path: Path, rows: Any) -> ObservationSet:
    if not isinstance(rows, list) or not rows:
        raise ValueError("observations must be a non-empty list.")
    names: list[str] = []
    transforms: list[np.ndarray] = []
    reproj: list[float] = []
    has_reproj = False
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise TypeError("Each observation row must be a mapping.")
        names.append(str(row.get("name", f"pose_{index:04d}")))
        transforms.append(_matrix4(row.get("T_cam_board"), "T_cam_board"))
        if "reproj_mean_px" in row:
            has_reproj = True
            reproj.append(float(row["reproj_mean_px"]))
        else:
            reproj.append(float("nan"))
    reproj_array = np.asarray(reproj, dtype=np.float64) if has_reproj else None
    return ObservationSet(path, names, np.stack(transforms, axis=0), reproj_array)


def _find_pose_npz_sibling(path: Path) -> Path | None:
    candidates = [
        path.with_suffix(".npz"),
        path.with_name(f"{path.stem}_none.npz"),
        path.parent / "camera_board_poses.npz",
        path.parent / "camera_board_poses_none.npz",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _identification_section(document: dict[str, Any], *, create: bool = False) -> dict[str, Any]:
    if document.get("kind") == "identified_model":
        payload = document.get("payload")
        if not isinstance(payload, dict):
            raise TypeError("identified_model payload must be a mapping.")
        if "identification" not in payload and create:
            payload["identification"] = {}
        section = payload.get("identification") or payload
        if not isinstance(section, dict):
            raise TypeError("Identification section must be a mapping.")
        return section
    if "identification" not in document and create:
        document["identification"] = {}
    section = document.get("identification") or document.get("calibration") or document
    if not isinstance(section, dict):
        raise TypeError("Identification section must be a mapping.")
    return section


def _monitoring_section(document: dict[str, Any]) -> dict[str, Any]:
    if document.get("kind") == "camera_monitoring":
        payload = document.get("payload")
        if not isinstance(payload, dict):
            raise TypeError("camera_monitoring payload must be a mapping.")
        section = payload.get("model_monitoring") or payload.get("monitoring") or payload
        if not isinstance(section, dict):
            raise TypeError("Monitoring section must be a mapping.")
        return section
    section = document.get("model_monitoring") or document.get("monitoring") or {}
    if not isinstance(section, dict):
        raise TypeError("Monitoring section must be a mapping.")
    return section


def _current_confidence(document: dict[str, Any]) -> float:
    section = _identification_section(document)
    if "confidence_current" in section:
        return float(section["confidence_current"])
    history = section.get("confidence_history")
    if isinstance(history, list) and history:
        latest = history[-1]
        if isinstance(latest, dict) and "value" in latest:
            return float(latest["value"])
    return float(section.get("confidence", 100.0))


def _current_position_uncertainty(document: dict[str, Any]) -> float:
    section = _identification_section(document)
    metrics = section.get("metrics", {})
    if isinstance(metrics, dict):
        return float(
            metrics.get(
                "position_uncertainty_rmse_mm",
                metrics.get("fit_rmse_mm", metrics.get("rmse_mm", 0.0)),
            )
        )
    return 0.0


def _optional_index(values: np.ndarray | None, index: int) -> float | None:
    if values is None or index >= len(values):
        return None
    value = float(values[index])
    return None if math.isnan(value) else value


def _matrix4(value: Any, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"{name} must be 4x4, got {matrix.shape}.")
    return matrix


def _matrix3(value: Any, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"{name} must be 3x3, got {matrix.shape}.")
    return matrix


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=np.float64).reshape(3)
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float64,
    )


def _plain_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain_value(item) for item in value]
    if isinstance(value, tuple):
        return [_plain_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value
