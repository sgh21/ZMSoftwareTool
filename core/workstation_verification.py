from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from core.calibration_service import CalibrationService
from core.config_manager import load_yaml, save_yaml

POSE_KEYS = ("x", "y", "z", "rx", "ry", "rz")
JOINT_KEYS = ("q1", "q2", "q3", "q4", "q5", "q6")


@dataclass(frozen=True)
class WorkstationConfig:
    workstation_id: str
    input_type: str
    pose: tuple[float, float, float, float, float, float] | None = None
    joint_degrees: tuple[float, float, float, float, float, float] | None = None


@dataclass(frozen=True)
class WorkstationAccuracyPreview:
    workstation: WorkstationConfig
    pose: tuple[float, float, float, float, float, float]
    joint_degrees: tuple[float, float, float, float, float, float]
    error_mm: float
    threshold_mm: float
    over_limit: bool
    health_score: float
    health_level: str

    @property
    def workstation_id(self) -> str:
        return self.workstation.workstation_id


class WorkstationVerificationService:
    """Persist machining workstations and compute lightweight accuracy previews."""

    def __init__(
        self,
        project_root: str | Path,
        *,
        calibration_service: CalibrationService | None = None,
        config_path: str | Path | None = None,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.config_path = Path(config_path or self.project_root / "config" / "workstations.yaml")
        self._calibration_service = calibration_service or CalibrationService(self.project_root)

    def load_workstations(self) -> list[WorkstationConfig]:
        data = load_yaml(self.config_path)
        rows = data.get("workstations", [])
        if not isinstance(rows, list):
            raise TypeError("workstations must be a list.")
        return [
            _normalize_workstation(row, index + 1)
            for index, row in enumerate(rows)
            if isinstance(row, dict)
        ]

    def save_workstations(self, workstations: Iterable[WorkstationConfig]) -> Path:
        rows = [self.complete_workstation(workstation) for workstation in workstations]
        document = {
            "workstations": [_serialize_workstation(workstation) for workstation in rows],
        }
        return save_yaml(self.config_path, document)

    def complete_workstation(self, workstation: WorkstationConfig) -> WorkstationConfig:
        input_type = _normalize_input_type(workstation.input_type)
        if input_type == "joint":
            joints = _require_six(workstation.joint_degrees, "joint_degrees")
            pose = tuple(
                float(value)
                for value in self._calibration_service.compute_nominal_pose(
                    joints,
                    joint_unit="degrees",
                )
            )
            return WorkstationConfig(
                workstation.workstation_id,
                input_type,
                pose=_six_tuple(pose),
                joint_degrees=_six_tuple(joints),
            )
        pose = _require_six(workstation.pose, "pose")
        joints = (
            _six_tuple(workstation.joint_degrees)
            if workstation.joint_degrees is not None
            else None
        )
        return WorkstationConfig(
            workstation.workstation_id,
            input_type,
            pose=_six_tuple(pose),
            joint_degrees=joints,
        )

    def preview_workstations(
        self,
        workstations: Iterable[WorkstationConfig],
        *,
        threshold_mm: float,
        initial_joint_degrees: list[float] | tuple[float, ...] | np.ndarray | None = None,
    ) -> list[WorkstationAccuracyPreview]:
        threshold = max(float(threshold_mm), 0.0)
        previews: list[WorkstationAccuracyPreview] = []
        for workstation in workstations:
            completed = self.complete_workstation(workstation)
            if completed.input_type == "joint":
                joints = _require_six(completed.joint_degrees, "joint_degrees")
                pose = _require_six(completed.pose, "pose")
            else:
                pose = _require_six(completed.pose, "pose")
                if completed.joint_degrees is None:
                    seed = initial_joint_degrees
                    if seed is None:
                        seed = np.zeros(6, dtype=float)
                        seed_unit = "radians"
                    else:
                        seed_unit = "degrees"
                    joints = tuple(
                        float(value)
                        for value in self._calibration_service.solve_nominal_joint_angles_for_pose(
                            pose,
                            initial_joint_angles=seed,
                            initial_unit=seed_unit,
                            output_unit="degrees",
                        )
                    )
                    completed = WorkstationConfig(
                        completed.workstation_id,
                        completed.input_type,
                        pose=pose,
                        joint_degrees=_six_tuple(joints),
                    )
                else:
                    joints = _require_six(completed.joint_degrees, "joint_degrees")

            state = self._calibration_service.compute_predicted_position(
                joints,
                joint_unit="degrees",
                tolerance_mm=threshold,
            )
            previews.append(
                WorkstationAccuracyPreview(
                    workstation=completed,
                    pose=_six_tuple(pose),
                    joint_degrees=_six_tuple(joints),
                    error_mm=float(state.error_norm_mm),
                    threshold_mm=threshold,
                    over_limit=bool(state.error_norm_mm > threshold),
                    health_score=float(state.health_score),
                    health_level=state.health_level,
                )
            )
        return previews


def _normalize_workstation(raw: dict[str, Any], fallback_index: int) -> WorkstationConfig:
    workstation_id = str(raw.get("id") or raw.get("workstation_id") or f"WS-{fallback_index:03d}")
    input_type = _normalize_input_type(raw.get("input_type", raw.get("mode", "pose")))
    pose = _coerce_six(raw.get("pose"), POSE_KEYS)
    joints = _coerce_six(
        raw.get("joint_degrees", raw.get("joint_angles", raw.get("joints"))),
        JOINT_KEYS,
    )
    if input_type == "pose" and pose is None:
        pose = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    if input_type == "joint" and joints is None:
        joints = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return WorkstationConfig(
        workstation_id=workstation_id,
        input_type=input_type,
        pose=pose,
        joint_degrees=joints,
    )


def _serialize_workstation(workstation: WorkstationConfig) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": workstation.workstation_id,
        "input_type": _normalize_input_type(workstation.input_type),
    }
    if workstation.pose is not None:
        row["pose"] = {
            key: float(value)
            for key, value in zip(POSE_KEYS, _six_tuple(workstation.pose), strict=True)
        }
    if workstation.joint_degrees is not None:
        row["joint_degrees"] = [
            float(value)
            for value in _six_tuple(workstation.joint_degrees)
        ]
    return row


def _normalize_input_type(value: Any) -> str:
    text = str(value or "pose").strip().lower()
    if text in {"joint", "joints", "q", "关节角"}:
        return "joint"
    return "pose"


def _coerce_six(
    value: Any,
    keys: tuple[str, str, str, str, str, str],
) -> tuple[float, float, float, float, float, float] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return _six_tuple([float(value.get(key, 0.0) or 0.0) for key in keys])
    if isinstance(value, (list, tuple)):
        if len(value) != 6:
            raise ValueError("Expected 6 values.")
        return _six_tuple(value)
    raise TypeError("Expected a mapping or a 6-value list.")


def _require_six(
    value: tuple[float, float, float, float, float, float] | None,
    name: str,
) -> tuple[float, float, float, float, float, float]:
    if value is None:
        raise ValueError(f"{name} is required.")
    return _six_tuple(value)


def _six_tuple(values: Any) -> tuple[float, float, float, float, float, float]:
    array = np.asarray(values, dtype=float).reshape(6)
    if not np.all(np.isfinite(array)):
        raise ValueError("Workstation values must be finite.")
    return tuple(float(value) for value in array)  # type: ignore[return-value]
