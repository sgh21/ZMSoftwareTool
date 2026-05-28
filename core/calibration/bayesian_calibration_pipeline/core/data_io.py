"""Load and save calibration datasets with a stable schema."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np


def load_dataset(path: str | Path) -> dict[str, Any]:
    """Load real or simulated ``.pkl`` data into the baseline schema.

    Required output keys are ``joints`` and ``measured_positions``. Optional
    keys include ``payloads``, ``directions``, ``joint_torques``, and
    simulation truth metadata.
    """
    dataset_path = Path(path)
    with dataset_path.open("rb") as file:
        raw = pickle.load(file)

    if isinstance(raw, dict):
        joints = _first_present(raw, ("joints", "joint_configs", "q"))
        positions = _first_present(
            raw, ("measured_positions", "laser_points", "positions", "points")
        )
        payloads = raw.get("payloads", raw.get("payload", None))
        directions = _first_optional(raw, ("directions", "approach_directions", "h"))
        if directions is None:
            directions = _first_optional(raw, ("approach_directions_raw", "directions_raw"))
        joint_torques = _first_optional(raw, ("joint_torques", "gravity_torques", "tau_g", "torques"))
        self_weight_joint_torques = _first_optional(
            raw, ("self_weight_joint_torques", "self_gravity_torques")
        )
        payload_joint_torques = _first_optional(
            raw, ("payload_joint_torques", "payload_gravity_torques")
        )
        extra = {k: v for k, v in raw.items() if k not in {"joints", "joint_configs", "q"}}
    elif isinstance(raw, list):
        joints = [item["joints"] for item in raw]
        positions = [_first_present(item, ("measured_positions", "laser_points", "positions")) for item in raw]
        payloads = [float(item.get("payload", 0.0)) for item in raw]
        directions = _list_optional(raw, ("directions", "direction", "approach_directions", "approach_direction", "h"))
        if directions is None:
            directions = _list_optional(raw, ("approach_directions_raw", "directions_raw"))
        joint_torques = _list_optional(raw, ("joint_torques", "gravity_torques", "tau_g", "torques"))
        self_weight_joint_torques = _list_optional(
            raw, ("self_weight_joint_torques", "self_gravity_torques")
        )
        payload_joint_torques = _list_optional(
            raw, ("payload_joint_torques", "payload_gravity_torques")
        )
        extra = {}
    else:
        raise TypeError(f"Unsupported dataset type: {type(raw)!r}")

    joint_array = np.asarray(joints, dtype=float).reshape(-1, 6)
    position_array = np.asarray(positions, dtype=float).reshape(-1, 3)
    if len(joint_array) != len(position_array):
        raise ValueError("joint and position sample counts must match.")

    output: dict[str, Any] = {
        "joints": joint_array,
        "measured_positions": position_array,
        "payloads": _payload_array(payloads, len(joint_array)),
    }
    if directions is not None:
        output["directions"] = np.asarray(directions, dtype=float).reshape(-1, 6)
    if joint_torques is not None:
        output["joint_torques"] = np.asarray(joint_torques, dtype=float).reshape(-1, 6)
    if self_weight_joint_torques is not None:
        output["self_weight_joint_torques"] = np.asarray(self_weight_joint_torques, dtype=float).reshape(-1, 6)
    if payload_joint_torques is not None:
        output["payload_joint_torques"] = np.asarray(payload_joint_torques, dtype=float).reshape(-1, 6)
    if "joint_torques" not in output and {
        "self_weight_joint_torques",
        "payload_joint_torques",
    }.issubset(output):
        output["joint_torques"] = output["self_weight_joint_torques"] + output["payload_joint_torques"]
    for key, value in extra.items():
        if key not in output:
            output[key] = value
    return output


def save_dataset(path: str | Path, dataset: dict[str, Any]) -> Path:
    """Write a dataset pkl without changing the schema."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as file:
        pickle.dump(dataset, file)
    return output_path


def _payload_array(payloads: Any, count: int) -> np.ndarray:
    if payloads is None:
        return np.zeros(count, dtype=float)
    values = np.asarray(payloads, dtype=float).reshape(-1)
    if values.size == 1:
        return np.full(count, float(values[0]), dtype=float)
    if values.size != count:
        raise ValueError(f"Expected {count} payload values, got {values.size}.")
    return values


def _first_present(mapping: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    raise KeyError(f"None of these keys were found: {keys}")


def _first_optional(mapping: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _list_optional(items: list[dict[str, Any]], keys: tuple[str, ...]) -> list[Any] | None:
    values: list[Any] = []
    found = False
    for item in items:
        value = _first_optional(item, keys)
        values.append(value)
        found = found or value is not None
    if not found:
        return None
    if any(value is None for value in values):
        raise ValueError(f"Optional per-sample keys are incomplete: {keys}")
    return values

