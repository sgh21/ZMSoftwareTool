from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from core.calibration_persistence import load_identification_result


VECTOR3_KEYS = ("base_xyz", "base_rpy", "tool_xyz", "tool_rpy")
MDH_KEYS = ("alpha", "a", "d", "theta_offset")
AXES = ("x", "y", "z")
DEFAULT_NOMINAL_ROBOT = {
    "base_xyz": [3.335740524, 1.591246404, -0.475598057],
    "base_rpy": [0.0189015, -0.0114514, 1.6069778],
    "tool_xyz": [0.0, 0.0, 0.039],
    "tool_rpy": [0.0, 0.0, 0.0],
    "mdh": {
        "alpha": [0.0, np.pi / 2.0, 0.0, 0.0, np.pi / 2.0, -np.pi / 2.0],
        "a": [0.0, 0.0, -0.612, -0.5723, 0.0, 0.0],
        "d": [0.1273, 0.0, 0.0, 0.163941, 0.1157, 0.0922],
        "theta_offset": [0.0] * 6,
    },
    "joint_limits": [
        [-np.pi, np.pi],
        [-2.4, -0.4],
        [-2.6, 0.2],
        [-np.pi, np.pi],
        [-np.pi, np.pi],
        [-np.pi, np.pi],
    ],
}


@dataclass(frozen=True)
class NominalParameterUpdateResult:
    nominal_path: Path
    backup_path: Path
    mode: str
    applied_error_parameter_hash: str | None = None


class NominalParameterService:
    """Load, update, persist, and roll back the nominal robot parameter file."""

    def __init__(
        self,
        project_root: str | Path | None = None,
        *,
        nominal_path: str | Path | None = None,
        backup_path: str | Path | None = None,
    ) -> None:
        self.project_root = Path(project_root or Path.cwd()).resolve()
        self.nominal_path = Path(nominal_path or self.project_root / "config" / "nominal_robot.yaml")
        self.backup_path = Path(
            backup_path
            or self.project_root / "storage" / "model_versions" / "nominal_robot_previous.yaml"
        )

    def has_backup(self) -> bool:
        return self.backup_path.exists()

    def load_nominal(self) -> dict[str, Any]:
        """Return the current nominal parameters, falling back to built-in defaults."""
        return self.load_document()["nominal_robot"]

    def load_document(self) -> dict[str, Any]:
        """Return the persisted nominal document with normalized numeric parameters."""
        if not self.nominal_path.exists():
            return {"nominal_robot": _normalize_nominal(_plain(DEFAULT_NOMINAL_ROBOT))}
        data = yaml.safe_load(self.nominal_path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise TypeError(f"Nominal parameter file must contain a mapping: {self.nominal_path}")
        return _normalize_nominal_document(data)

    def dump_current_yaml(self) -> str:
        return yaml.safe_dump(
            {"nominal_robot": self.load_nominal()},
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        )

    def value_template_yaml(self) -> str:
        return yaml.safe_dump(
            {
                "nominal_values": self.load_nominal()
            },
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        )

    def update_direct_yaml(self, yaml_text: str) -> NominalParameterUpdateResult:
        return self.update_direct(_load_yaml_text(yaml_text), mode="direct")

    def update_direct(
        self,
        nominal_data: dict[str, Any],
        *,
        mode: str = "direct",
    ) -> NominalParameterUpdateResult:
        updated = _normalize_nominal(_unwrap_nominal(nominal_data))
        return self._persist(updated, mode=mode)

    def update_values_yaml(self, yaml_text: str) -> NominalParameterUpdateResult:
        return self.update_values(_load_yaml_text(yaml_text), mode="values")

    def update_values(
        self,
        values_data: dict[str, Any],
        *,
        mode: str = "values",
    ) -> NominalParameterUpdateResult:
        current = self.load_nominal()
        updated = _merge_nominal_values(current, values_data)
        return self._persist(updated, mode=mode)

    def update_from_identification_file(
        self,
        path: str | Path,
    ) -> NominalParameterUpdateResult:
        source_path = Path(path).resolve()
        loaded = load_identification_result(path)
        fingerprint = error_parameter_fingerprint(loaded["error_parameters"])
        update_metadata = {
            "mode": "identification",
            "source_path": str(source_path),
            "source_timestamp": loaded.get("timestamp", ""),
            "applied_error_parameter_hash": fingerprint,
        }
        updated = _normalize_nominal(_unwrap_nominal(loaded["identified_robot"]))
        return self._persist(
            updated,
            mode="identification",
            update_metadata=update_metadata,
            applied_error_parameter_hash=fingerprint,
        )

    def rollback(self) -> Path:
        """Restore the one retained previous nominal file and consume the backup."""
        if not self.backup_path.exists():
            raise FileNotFoundError(f"No nominal parameter backup found: {self.backup_path}")
        data = yaml.safe_load(self.backup_path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise TypeError(f"Backup file must contain a mapping: {self.backup_path}")
        document = _normalize_nominal_document(data)
        self.nominal_path.parent.mkdir(parents=True, exist_ok=True)
        self.nominal_path.write_text(
            yaml.safe_dump(
                document,
                allow_unicode=True,
                sort_keys=False,
                default_flow_style=False,
            ),
            encoding="utf-8",
        )
        self.backup_path.unlink()
        return self.nominal_path

    def _persist(
        self,
        nominal: dict[str, Any],
        *,
        mode: str,
        update_metadata: dict[str, Any] | None = None,
        applied_error_parameter_hash: str | None = None,
    ) -> NominalParameterUpdateResult:
        current_document = self.load_document()
        self.backup_path.parent.mkdir(parents=True, exist_ok=True)
        self.backup_path.write_text(
            yaml.safe_dump(
                current_document,
                allow_unicode=True,
                sort_keys=False,
                default_flow_style=False,
            ),
            encoding="utf-8",
        )

        self.nominal_path.parent.mkdir(parents=True, exist_ok=True)
        document: dict[str, Any] = {"nominal_robot": nominal}
        if update_metadata:
            document["nominal_update"] = _sanitize_update_metadata(update_metadata)
        self.nominal_path.write_text(
            yaml.safe_dump(
                document,
                allow_unicode=True,
                sort_keys=False,
                default_flow_style=False,
            ),
            encoding="utf-8",
        )
        return NominalParameterUpdateResult(
            nominal_path=self.nominal_path,
            backup_path=self.backup_path,
            mode=mode,
            applied_error_parameter_hash=applied_error_parameter_hash,
        )


def error_parameter_fingerprint(parameter_values: dict[str, Any]) -> str:
    """Return a stable fingerprint for an identified error-parameter mapping."""
    normalized = {
        str(key): float(value or 0.0)
        for key, value in sorted(parameter_values.items(), key=lambda item: str(item[0]))
    }
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def nominal_after_applying_error_parameters(
    nominal_data: dict[str, Any],
    parameter_values: dict[str, Any],
) -> dict[str, Any]:
    """Return nominal parameters after absorbing identified error parameters."""
    nominal = _normalize_nominal(_unwrap_nominal(nominal_data))
    delta = _normalize_delta({"error_parameters": parameter_values})
    return _add_delta(nominal, delta)


def nominal_parameter_sets_close(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    atol: float = 1.0e-10,
) -> bool:
    left_nominal = _normalize_nominal(_unwrap_nominal(left))
    right_nominal = _normalize_nominal(_unwrap_nominal(right))
    for key in VECTOR3_KEYS:
        if not np.allclose(left_nominal[key], right_nominal[key], atol=atol, rtol=0.0):
            return False
    for key in MDH_KEYS:
        if not np.allclose(
            left_nominal["mdh"][key],
            right_nominal["mdh"][key],
            atol=atol,
            rtol=0.0,
        ):
            return False
    return True


def _load_yaml_text(text: str) -> dict[str, Any]:
    data = yaml.safe_load(text or "") or {}
    if not isinstance(data, dict):
        raise TypeError("YAML content must be a mapping.")
    return data


def _unwrap_nominal(data: dict[str, Any]) -> dict[str, Any]:
    section = data.get("nominal_robot", data)
    if not isinstance(section, dict):
        raise TypeError("nominal_robot must be a mapping.")
    return section


def _normalize_nominal(data: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise TypeError("Nominal parameters must be a mapping.")

    mdh = data.get("mdh")
    if not isinstance(mdh, dict):
        raise KeyError("nominal_robot.mdh is required.")

    nominal: dict[str, Any] = {
        key: _float_list(data, key, 3, required=True)
        for key in VECTOR3_KEYS
    }
    nominal["mdh"] = {
        key: _float_list(mdh, key, 6, required=True)
        for key in MDH_KEYS
    }

    if "joint_limits" in data and data["joint_limits"] is not None:
        nominal["joint_limits"] = _joint_limits(data["joint_limits"])
    return nominal


def _normalize_nominal_document(data: dict[str, Any]) -> dict[str, Any]:
    document: dict[str, Any] = {"nominal_robot": _normalize_nominal(_unwrap_nominal(data))}
    metadata = data.get("nominal_update")
    if isinstance(metadata, dict):
        document["nominal_update"] = _sanitize_update_metadata(metadata)
    return document


def _merge_nominal_values(current: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    source = _extract_nominal_values_source(data)
    updated = _plain(_normalize_nominal(_unwrap_nominal(current)))
    has_value = False

    for key in VECTOR3_KEYS:
        if key in source:
            updated[key] = _float_list(source, key, 3, required=True)
            has_value = True

    mdh = source.get("mdh")
    if mdh is not None:
        if not isinstance(mdh, dict):
            raise TypeError("nominal_values.mdh must be a mapping.")
        for key in MDH_KEYS:
            if key in mdh:
                updated["mdh"][key] = _float_list(mdh, key, 6, required=True)
                has_value = True

    if "joint_limits" in source:
        updated["joint_limits"] = _joint_limits(source["joint_limits"])
        has_value = True

    if not has_value:
        raise KeyError(
            "nominal_values must contain at least one nominal field: "
            "base_xyz/base_rpy/tool_xyz/tool_rpy/mdh/joint_limits."
        )
    return _normalize_nominal(updated)


def _extract_nominal_values_source(data: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise TypeError("Nominal values must be a mapping.")
    if "nominal_delta" in data:
        raise KeyError("nominal_delta is no longer supported. Use nominal_values.")
    for section_name in ("nominal_values", "nominal_robot"):
        section = data.get(section_name)
        if isinstance(section, dict):
            return section
    return data


def _sanitize_update_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    allowed = (
        "mode",
        "source_path",
        "source_timestamp",
        "applied_error_parameter_hash",
    )
    return {
        key: str(metadata[key])
        for key in allowed
        if key in metadata and metadata[key] is not None
    }


def _normalize_delta(data: dict[str, Any]) -> dict[str, Any]:
    source = _extract_delta_source(data)
    delta: dict[str, Any] = {
        "base_xyz": [0.0, 0.0, 0.0],
        "base_rpy": [0.0, 0.0, 0.0],
        "tool_xyz": [0.0, 0.0, 0.0],
        "tool_rpy": [0.0, 0.0, 0.0],
        "mdh": {
            "alpha": [0.0] * 6,
            "a": [0.0] * 6,
            "d": [0.0] * 6,
            "theta_offset": [0.0] * 6,
        },
    }

    for key in VECTOR3_KEYS:
        if key in source:
            delta[key] = _float_list(source, key, 3, required=False)

    mdh = source.get("mdh", {})
    if mdh is not None and not isinstance(mdh, dict):
        raise TypeError("Delta mdh must be a mapping.")
    for key in MDH_KEYS:
        if isinstance(mdh, dict) and key in mdh:
            delta["mdh"][key] = _float_list(mdh, key, 6, required=False)

    flat = source.get("error_parameters", source)
    if isinstance(flat, dict):
        _apply_flat_error_parameters(delta, flat)
    return delta


def _extract_delta_source(data: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise TypeError("Delta parameters must be a mapping.")

    for section_name in ("identification", "calibration"):
        section = data.get(section_name)
        if isinstance(section, dict) and isinstance(section.get("error_parameters"), dict):
            return {"error_parameters": section["error_parameters"]}

    section = data.get("error_parameters")
    if isinstance(section, dict):
        return {"error_parameters": section}
    return data


def _apply_flat_error_parameters(delta: dict[str, Any], flat: dict[str, Any]) -> None:
    for joint in range(1, 7):
        index = joint - 1
        _assign_if_present(delta["mdh"]["alpha"], index, flat, f"delta_alpha_{joint}")
        _assign_if_present(delta["mdh"]["a"], index, flat, f"delta_a_{joint}")
        _assign_if_present(delta["mdh"]["d"], index, flat, f"delta_d_{joint}")
        _assign_if_present(delta["mdh"]["theta_offset"], index, flat, f"delta_theta_{joint}")

    for index, axis in enumerate(AXES):
        _assign_if_present(delta["base_xyz"], index, flat, f"delta_Bt{axis}")
        _assign_if_present(delta["base_rpy"], index, flat, f"delta_Bu{axis}")
        _assign_if_present(delta["tool_xyz"], index, flat, f"delta_Tt{axis}")
        _assign_if_present(delta["tool_rpy"], index, flat, f"delta_Tu{axis}")


def _assign_if_present(target: list[float], index: int, source: dict[str, Any], key: str) -> None:
    if key in source:
        target[index] = _float_or_zero(source[key])


def _add_delta(current: dict[str, Any], delta: dict[str, Any]) -> dict[str, Any]:
    updated: dict[str, Any] = {
        key: [
            float(current[key][index]) + float(delta[key][index])
            for index in range(3)
        ]
        for key in VECTOR3_KEYS
    }
    updated["mdh"] = {
        key: [
            float(current["mdh"][key][index]) + float(delta["mdh"][key][index])
            for index in range(6)
        ]
        for key in MDH_KEYS
    }
    if "joint_limits" in current:
        updated["joint_limits"] = current["joint_limits"]
    return _normalize_nominal(updated)


def _float_list(
    data: dict[str, Any],
    key: str,
    length: int,
    *,
    required: bool,
) -> list[float]:
    if key not in data or data[key] is None:
        if required:
            raise KeyError(f"{key} is required.")
        return [0.0] * length
    values = data[key]
    if not isinstance(values, (list, tuple)):
        raise TypeError(f"{key} must be a list of {length} numbers.")
    if len(values) != length:
        raise ValueError(f"{key} must contain {length} numbers.")
    return [_float_or_zero(value) for value in values]


def _joint_limits(values: Any) -> list[list[float]]:
    if not isinstance(values, (list, tuple)) or len(values) != 6:
        raise ValueError("joint_limits must contain 6 [min, max] pairs.")
    limits: list[list[float]] = []
    for item in values:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("Each joint limit must be a [min, max] pair.")
        limits.append([_float_or_zero(item[0]), _float_or_zero(item[1])])
    return limits


def _float_or_zero(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    number = float(value)
    if not np.isfinite(number):
        raise ValueError("Nominal parameters must be finite numbers.")
    return number


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value
