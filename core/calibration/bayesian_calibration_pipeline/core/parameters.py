"""Paper-style 54-dimensional multisource error vector."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ErrorParameter:
    """One scalar calibration parameter."""

    name: str
    group: str
    unit: str
    scale: float
    lower: float
    upper: float


def build_error_parameters() -> list[ErrorParameter]:
    """Return the 54 parameters from the paper's unified error model.

    Composition:
        24 MD-H kinematic, 12 base/tool frame, 6 reduction ratio,
        6 joint backlash, and 6 joint flexibility parameters.
    """
    params: list[ErrorParameter] = []
    for prefix, unit, scale in (
        ("delta_alpha", "rad", 7.0e-4),
        ("delta_a", "m", 14.0e-4),
        ("delta_d", "m", 14.0e-4),
        ("delta_theta", "rad", 7.0e-4),
    ):
        for joint in range(1, 7):
            params.append(_param(f"{prefix}_{joint}", "kinematic", unit, scale))

    for prefix, unit, scale in (
        ("delta_Bt", "m", 21.0e-4),
        ("delta_Bu", "rad", 7.0e-4),
        ("delta_Tt", "m", 7.0e-4),
        ("delta_Tu", "rad", 3.5e-4),
    ):
        for axis in ("x", "y", "z"):
            params.append(_param(f"{prefix}{axis}", "frame", unit, scale))

    for joint in range(1, 7):
        params.append(_param(f"delta_rrd_{joint}", "reduction_ratio", "ratio", 5.0e-5))
    for joint in range(1, 7):
        params.append(_param(f"delta_backlash_{joint}", "backlash", "rad", 3.0e-4))
    for joint in range(1, 7):
        params.append(_param(f"delta_flex_{joint}", "flexibility", "rad/Nm", 5.0e-7))
    return params


def _param(name: str, group: str, unit: str, scale: float) -> ErrorParameter:
    return ErrorParameter(
        name=name,
        group=group,
        unit=unit,
        scale=scale,
        lower=-4.0 * scale,
        upper=4.0 * scale,
    )


def zero_error_vector(parameters: list[ErrorParameter] | None = None) -> np.ndarray:
    """Return a zero vector in the active parameter order."""
    params = parameters if parameters is not None else build_error_parameters()
    return np.zeros(len(params), dtype=float)


def parameter_scales(parameters: list[ErrorParameter] | None = None) -> np.ndarray:
    """Return finite-difference and optimizer scales."""
    params = parameters if parameters is not None else build_error_parameters()
    return np.asarray([max(p.scale, 1.0e-12) for p in params], dtype=float)


def parameter_bounds(
    parameters: list[ErrorParameter] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return prior bounds for sensitivity sampling."""
    params = parameters if parameters is not None else build_error_parameters()
    return (
        np.asarray([p.lower for p in params], dtype=float),
        np.asarray([p.upper for p in params], dtype=float),
    )


def sample_truth_vector(
    rng: np.random.Generator,
    parameters: list[ErrorParameter] | None = None,
    sigma_scale: float = 1.0,
) -> np.ndarray:
    """Sample a simulated physical robot error vector."""
    params = parameters if parameters is not None else build_error_parameters()
    return rng.normal(0.0, parameter_scales(params) * float(sigma_scale))


def vector_to_components(
    vector: np.ndarray,
    parameters: list[ErrorParameter] | None = None,
) -> dict[str, np.ndarray]:
    """Unpack a flat error vector into arrays used by the forward model."""
    params = parameters if parameters is not None else build_error_parameters()
    values = np.asarray(vector, dtype=float).reshape(-1)
    if values.size != len(params):
        raise ValueError(f"Expected {len(params)} parameters, got {values.size}.")
    by_name = {param.name: values[index] for index, param in enumerate(params)}

    comp = {
        "delta_alpha": np.zeros(6),
        "delta_a": np.zeros(6),
        "delta_d": np.zeros(6),
        "delta_theta": np.zeros(6),
        "base_xyz": np.zeros(3),
        "base_rpy": np.zeros(3),
        "tool_xyz": np.zeros(3),
        "tool_rpy": np.zeros(3),
        "rrd": np.zeros(6),
        "backlash": np.zeros(6),
        "flex": np.zeros(6),
    }
    for joint in range(1, 7):
        j = joint - 1
        comp["delta_alpha"][j] = by_name.get(f"delta_alpha_{joint}", 0.0)
        comp["delta_a"][j] = by_name.get(f"delta_a_{joint}", 0.0)
        comp["delta_d"][j] = by_name.get(f"delta_d_{joint}", 0.0)
        comp["delta_theta"][j] = by_name.get(f"delta_theta_{joint}", 0.0)
        comp["rrd"][j] = by_name.get(f"delta_rrd_{joint}", 0.0)
        comp["backlash"][j] = by_name.get(f"delta_backlash_{joint}", 0.0)
        comp["flex"][j] = by_name.get(f"delta_flex_{joint}", 0.0)

    for index, axis in enumerate(("x", "y", "z")):
        comp["base_xyz"][index] = by_name.get(f"delta_Bt{axis}", 0.0)
        comp["base_rpy"][index] = by_name.get(f"delta_Bu{axis}", 0.0)
        comp["tool_xyz"][index] = by_name.get(f"delta_Tt{axis}", 0.0)
        comp["tool_rpy"][index] = by_name.get(f"delta_Tu{axis}", 0.0)
    return comp


def vector_to_named_dict(
    vector: np.ndarray,
    parameters: list[ErrorParameter] | None = None,
) -> dict[str, float]:
    """Serialize a parameter vector with stable human-readable names."""
    params = parameters if parameters is not None else build_error_parameters()
    values = np.asarray(vector, dtype=float).reshape(len(params))
    return {param.name: float(values[index]) for index, param in enumerate(params)}

