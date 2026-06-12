from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

TOOL_VECTOR3_KEYS = ("tool_xyz", "tool_rpy")
CALIBRATION_FRAME_VECTOR3_KEYS = ("base_xyz", "base_rpy", "tool_xyz", "tool_rpy")
MDH_KEYS = ("alpha", "a", "d", "theta_offset")
AXES = ("x", "y", "z")
DEFAULT_NOMINAL_ROBOT = {
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


class NominalParameterService:
    """Read the nominal robot parameter file.

    Nominal parameters are the design baseline.  Application code must not
    update or roll them back through this service; new calibrated parameters
    belong in timestamped identified-model files.
    """

    def __init__(
        self,
        project_root: str | Path | None = None,
        *,
        nominal_path: str | Path | None = None,
    ) -> None:
        self.project_root = Path(project_root or Path.cwd()).resolve()
        self.nominal_path = Path(nominal_path or self.project_root / "config" / "nominal_robot.yaml")

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



def nominal_after_applying_error_parameters(
    nominal_data: dict[str, Any],
    parameter_values: dict[str, Any],
) -> dict[str, Any]:
    """Return nominal parameters after absorbing only identified MD-H errors.

    Base-frame and target-ball offset errors remain calibration-layer data.
    They are intentionally not folded into the robot nominal FK model.
    """
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
    for key in TOOL_VECTOR3_KEYS:
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
        for key in TOOL_VECTOR3_KEYS
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

    for key in CALIBRATION_FRAME_VECTOR3_KEYS:
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
        key: list(current[key])
        for key in TOOL_VECTOR3_KEYS
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
