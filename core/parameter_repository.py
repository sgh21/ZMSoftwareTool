"""Timestamped parameter file repository.

The repository owns versioned parameter files under ``storage/parameters`` and
the lightweight active-combination pointer in ``config/active_parameters.yaml``.
It deliberately does not interpret robot kinematics; it only validates the
parameter kind envelope and keeps file writes traceable.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml


PARAMETER_KINDS = ("controller_model", "identified_model", "camera_monitoring")
ACTIVE_PARAMETERS_RELATIVE = Path("config") / "active_parameters.yaml"
NOMINAL_ROBOT_RELATIVE = Path("config") / "nominal_robot.yaml"
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
_TIMESTAMP_RE = re.compile(r"_(\d{8}_\d{6})(?:_(\d{2}))?\.ya?ml$")


@dataclass(frozen=True)
class ParameterVersion:
    kind: str
    path: Path
    created_at: str
    updated_at: str
    timestamp_key: str


class ParameterFileRepository:
    """Create, list, load, and activate timestamped parameter versions."""

    def __init__(self, project_root: str | Path) -> None:
        self.project_root = Path(project_root).resolve()
        self.storage_root = self.project_root / "storage" / "parameters"
        self.active_path = self.project_root / ACTIVE_PARAMETERS_RELATIVE

    def kind_dir(self, kind: str) -> Path:
        _validate_kind(kind)
        return self.storage_root / kind

    def resolve_path(self, path: str | Path) -> Path:
        return self._resolve_path(path)

    def create_version(
        self,
        kind: str,
        payload: dict[str, Any],
        *,
        metadata: dict[str, Any] | None = None,
        timestamp: str | None = None,
    ) -> Path:
        _validate_kind(kind)
        if not isinstance(payload, dict):
            raise TypeError("parameter payload must be a mapping.")
        now = _now_local_iso()
        stamp = timestamp or datetime.now().strftime(TIMESTAMP_FORMAT)
        directory = self.kind_dir(kind)
        directory.mkdir(parents=True, exist_ok=True)
        path = self._unique_path(directory, kind, stamp)
        document: dict[str, Any] = {
            "schema_version": 1,
            "kind": kind,
            "created_at": now,
            "updated_at": now,
            "payload": _plain_value(payload),
        }
        if metadata:
            document["metadata"] = _plain_value(metadata)
        self._write_yaml(path, document)
        return path

    def load_document(
        self,
        path: str | Path,
        *,
        expected_kind: str | None = None,
    ) -> dict[str, Any]:
        file_path = self._resolve_path(path)
        data = self._read_mapping(file_path)
        kind = data.get("kind")
        if expected_kind is not None:
            _validate_kind(expected_kind)
            if kind != expected_kind:
                raise ValueError(
                    f"Parameter kind mismatch: expected {expected_kind}, got {kind!r}."
                )
        if kind is not None:
            _validate_kind(str(kind))
        if kind in PARAMETER_KINDS and not isinstance(data.get("payload"), dict):
            raise TypeError(f"Parameter payload must be a mapping: {file_path}")
        return data

    def load_payload(self, path: str | Path, *, expected_kind: str) -> dict[str, Any]:
        document = self.load_document(path, expected_kind=expected_kind)
        payload = document.get("payload")
        if not isinstance(payload, dict):
            raise TypeError(f"Parameter payload must be a mapping: {path}")
        return payload

    def list_versions(self, kind: str) -> list[ParameterVersion]:
        _validate_kind(kind)
        directory = self.kind_dir(kind)
        if not directory.exists():
            return []
        versions: list[ParameterVersion] = []
        for path in sorted(directory.glob(f"{kind}_*.y*ml")):
            try:
                document = self.load_document(path, expected_kind=kind)
            except Exception:
                continue
            versions.append(
                ParameterVersion(
                    kind=kind,
                    path=path.resolve(),
                    created_at=str(document.get("created_at", "")),
                    updated_at=str(document.get("updated_at", "")),
                    timestamp_key=_timestamp_sort_key(path),
                )
            )
        versions.sort(key=lambda item: item.timestamp_key)
        return versions

    def latest_version(self, kind: str) -> ParameterVersion | None:
        versions = self.list_versions(kind)
        return versions[-1] if versions else None

    def import_parameter_file(self, kind: str, source_path: str | Path) -> Path:
        _validate_kind(kind)
        source = Path(source_path).resolve()
        data = self._read_mapping(source)
        if data.get("kind") in PARAMETER_KINDS:
            if data["kind"] != kind:
                raise ValueError(
                    f"Parameter kind mismatch: expected {kind}, got {data['kind']!r}."
                )
            payload = data.get("payload")
            if not isinstance(payload, dict):
                raise TypeError("parameter payload must be a mapping.")
        else:
            payload = self._legacy_payload(kind, data)
        return self.create_version(
            kind,
            payload,
            metadata={"source_path": str(source), "imported_from_legacy": True},
        )

    def ensure_initial_versions(self) -> dict[str, Path]:
        """Migrate legacy files once when active pointers are missing."""
        active = self.load_active()
        changed = False
        created: dict[str, Path] = {}

        nominal = active.get("nominal_robot")
        if not nominal:
            active["nominal_robot"] = str(NOMINAL_ROBOT_RELATIVE)
            changed = True

        identified = self._active_path_if_valid(active, "identified_model")
        if identified is None:
            legacy = self.project_root / "config" / "calibration_result.yaml"
            if legacy.exists():
                identified = self.import_parameter_file("identified_model", legacy)
                active["identified_model"] = self._to_relative_string(identified)
                created["identified_model"] = identified
                changed = True

        camera = self._active_path_if_valid(active, "camera_monitoring")
        if camera is None:
            legacy = self.project_root / "config" / "model_monitoring.yaml"
            if legacy.exists():
                camera = self.import_parameter_file("camera_monitoring", legacy)
                active["camera_monitoring"] = self._to_relative_string(camera)
                created["camera_monitoring"] = camera
                changed = True

        if changed:
            self.save_active(active)
        return created

    def load_active(self) -> dict[str, str]:
        if not self.active_path.exists():
            return {}
        data = self._read_mapping(self.active_path)
        active = data.get("active_parameters", data)
        if not isinstance(active, dict):
            raise TypeError("active_parameters.yaml must contain a mapping.")
        return {
            str(key): str(value)
            for key, value in active.items()
            if isinstance(value, (str, Path)) and str(value)
        }

    def save_active(self, active: dict[str, str | Path | None]) -> Path:
        normalized: dict[str, str] = {}
        for key, value in active.items():
            if value is None or str(value) == "":
                continue
            if key != "nominal_robot":
                _validate_kind(key)
            normalized[str(key)] = self._to_relative_string(value)
        document = {
            "schema_version": 1,
            "updated_at": _now_local_iso(),
            "active_parameters": normalized,
        }
        self.active_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_yaml(self.active_path, document)
        return self.active_path

    def active_path_for(self, kind: str) -> Path | None:
        if kind == "nominal_robot":
            raw = self.load_active().get("nominal_robot", str(NOMINAL_ROBOT_RELATIVE))
            return self._resolve_path(raw)
        _validate_kind(kind)
        raw = self.load_active().get(kind)
        if not raw:
            return None
        path = self._resolve_path(raw)
        if not path.exists():
            return None
        self.load_document(path, expected_kind=kind)
        return path

    def activate_version(self, kind: str, path: str | Path | None) -> Path | None:
        if path is None:
            active = self.load_active()
            active.pop(kind, None)
            self.save_active(active)
            return None
        _validate_kind(kind)
        version_path = self._resolve_path(path)
        self.load_document(version_path, expected_kind=kind)
        active = self.load_active()
        active.setdefault("nominal_robot", str(NOMINAL_ROBOT_RELATIVE))
        active[kind] = self._to_relative_string(version_path)
        self.save_active(active)
        return version_path

    def select_latest_versions(
        self,
        kinds: tuple[str, ...] = PARAMETER_KINDS,
    ) -> dict[str, Path]:
        selected: dict[str, Path] = {}
        for kind in kinds:
            latest = self.latest_version(kind)
            if latest is not None:
                selected[kind] = latest.path
        return selected

    def ensure_identified_model_version(self, source_path: str | Path) -> Path:
        source = Path(source_path).resolve()
        data = self._read_mapping(source)
        if data.get("kind") == "identified_model":
            self.load_document(source, expected_kind="identified_model")
            return source
        if data.get("kind") in PARAMETER_KINDS:
            raise ValueError(
                f"Parameter kind mismatch: expected identified_model, got {data['kind']!r}."
            )
        return self.import_parameter_file("identified_model", source)

    def create_controller_model_from_identified(
        self,
        identified_model_path: str | Path | None = None,
    ) -> Path:
        """Create a controller-model version by copying MD-H/tool FK fields.

        This is intended for the first version of controller parameters when a
        real controller export is not available yet.
        """
        if identified_model_path is None:
            latest = self.latest_version("identified_model")
            if latest is None:
                raise FileNotFoundError("No identified_model version is available.")
            identified_path = latest.path
        else:
            identified_path = self._resolve_path(identified_model_path)
        document = self.load_document(identified_path, expected_kind="identified_model")
        payload = document["payload"]
        section = payload.get("identification")
        if not isinstance(section, dict):
            raise TypeError("identified_model payload must contain identification.")
        identified_robot = section.get("identified_robot")
        if not isinstance(identified_robot, dict):
            raise TypeError("identification.identified_robot must be a mapping.")
        controller_model: dict[str, Any] = {}
        mdh = identified_robot.get("mdh")
        if not isinstance(mdh, dict):
            raise TypeError("identified_robot.mdh must be a mapping.")
        controller_model["mdh"] = _plain_value(mdh)
        for key in ("tool_xyz", "tool_rpy"):
            if key in identified_robot:
                controller_model[key] = _plain_value(identified_robot[key])
        return self.create_version(
            "controller_model",
            {
                "controller_model": controller_model,
                "source_identified_model": self._to_relative_string(identified_path),
                "source_identification_timestamp": str(section.get("timestamp", "")),
            },
            metadata={"source": "copied_from_identified_model"},
        )

    def append_confidence_history(
        self,
        identified_model_path: str | Path,
        value: float,
        *,
        source: str,
        reason: str,
        position_uncertainty_rmse_mm: float | None = None,
        evaluation_record: dict[str, Any] | None = None,
    ) -> Path:
        path = self._resolve_path(identified_model_path)
        document = self.load_document(path, expected_kind="identified_model")
        payload = document["payload"]
        section = payload.get("identification")
        if section is None:
            section = payload
        if not isinstance(section, dict):
            raise TypeError("identified_model payload must contain a mapping.")
        now = _now_local_iso()
        history = section.setdefault("confidence_history", [])
        if not isinstance(history, list):
            raise TypeError("confidence_history must be a list.")
        history.append(
            {
                "timestamp": now,
                "value": float(value),
                "source": str(source),
                "reason": str(reason),
            }
        )
        section["confidence_current"] = float(value)
        section["confidence"] = float(value)
        section["updated_at"] = now
        if position_uncertainty_rmse_mm is not None:
            metrics = section.setdefault("metrics", {})
            if not isinstance(metrics, dict):
                raise TypeError("identification.metrics must be a mapping.")
            metrics["position_uncertainty_rmse_mm"] = float(position_uncertainty_rmse_mm)
        if evaluation_record is not None:
            monitoring = section.setdefault("monitoring", {})
            if not isinstance(monitoring, dict):
                raise TypeError("identification.monitoring must be a mapping.")
            monitoring["last_degradation_evaluation"] = _plain_value(evaluation_record)
        document["updated_at"] = now
        self._write_yaml(path, document)
        return path

    def _active_path_if_valid(self, active: dict[str, str], kind: str) -> Path | None:
        raw = active.get(kind)
        if not raw:
            return None
        path = self._resolve_path(raw)
        if not path.exists():
            return None
        try:
            self.load_document(path, expected_kind=kind)
        except Exception:
            return None
        return path

    def _unique_path(self, directory: Path, kind: str, stamp: str) -> Path:
        path = directory / f"{kind}_{stamp}.yaml"
        if not path.exists():
            return path
        for index in range(2, 100):
            candidate = directory / f"{kind}_{stamp}_{index:02d}.yaml"
            if not candidate.exists():
                return candidate
        raise FileExistsError(f"Too many timestamp collisions for {kind}_{stamp}.")

    def _legacy_payload(self, kind: str, data: dict[str, Any]) -> dict[str, Any]:
        if kind == "identified_model":
            if isinstance(data.get("identification"), dict):
                return {"identification": data["identification"]}
            if isinstance(data.get("calibration"), dict):
                return {"identification": data["calibration"]}
            if isinstance(data.get("error_parameters"), dict):
                return {"identification": data}
            raise ValueError("identified_model import requires an identification payload.")
        if kind == "camera_monitoring":
            if isinstance(data.get("model_monitoring"), dict):
                return {"model_monitoring": data["model_monitoring"]}
            if isinstance(data.get("monitoring"), dict):
                return {"model_monitoring": data["monitoring"]}
            if any(key in data for key in ("hand_eye", "camera", "evaluation")):
                return {"model_monitoring": data}
            raise ValueError("camera_monitoring import requires monitoring parameters.")
        if kind == "controller_model":
            if isinstance(data.get("controller_model"), dict):
                return {"controller_model": data["controller_model"]}
            if any(key in data for key in ("mdh", "tool_xyz", "tool_rpy", "tool_frame")):
                return {"controller_model": data}
            raise ValueError("controller_model import requires MD-H/tool parameters.")
        raise ValueError(f"Unsupported parameter kind: {kind}")

    def _read_mapping(self, path: Path) -> dict[str, Any]:
        file_path = self._resolve_path(path)
        if not file_path.exists():
            raise FileNotFoundError(file_path)
        text = file_path.read_text(encoding="utf-8")
        if file_path.suffix.lower() == ".json":
            data = json.loads(text)
        else:
            data = yaml.safe_load(text) or {}
        if not isinstance(data, dict):
            raise TypeError(f"Parameter file must contain a mapping: {file_path}")
        return data

    def _write_yaml(self, path: Path, document: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(
                _plain_value(document),
                allow_unicode=True,
                default_flow_style=False,
                sort_keys=False,
            ),
            encoding="utf-8",
        )

    def _resolve_path(self, path: str | Path) -> Path:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.project_root / candidate
        return candidate.resolve()

    def _to_relative_string(self, path: str | Path) -> str:
        resolved = self._resolve_path(path)
        try:
            return resolved.relative_to(self.project_root).as_posix()
        except ValueError:
            return str(resolved)


def _validate_kind(kind: str) -> None:
    if kind not in PARAMETER_KINDS:
        raise ValueError(
            f"Unsupported parameter kind: {kind!r}. "
            f"Expected one of {', '.join(PARAMETER_KINDS)}."
        )


def _timestamp_sort_key(path: Path) -> str:
    match = _TIMESTAMP_RE.search(path.name)
    if not match:
        return path.name
    suffix = match.group(2) or "00"
    return f"{match.group(1)}_{suffix}"


def _now_local_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _plain_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain_value(item) for item in value]
    if isinstance(value, tuple):
        return [_plain_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value
