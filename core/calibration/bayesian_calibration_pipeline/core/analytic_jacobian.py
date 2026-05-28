"""Analytic output Jacobians for supported calibration parameter subsets."""

from __future__ import annotations

from typing import Any

import numpy as np

from core.calibration.bayesian_calibration_pipeline.core.parameters import ErrorParameter, vector_to_components
from core.calibration.bayesian_calibration_pipeline.core.transforms import make_transform, modified_dh_transform


SUPPORTED_GEOMETRY_PREFIXES = (
    "delta_alpha_",
    "delta_a_",
    "delta_d_",
    "delta_theta_",
    "delta_Bt",
    "delta_Bu",
    "delta_Tt",
)


def supports_geometry33_jacobian(parameters: list[ErrorParameter]) -> bool:
    """Return True when every parameter has an implemented geometry derivative."""
    if not parameters:
        return True
    return all(_parameter_kind(parameter.name)[0] != "unsupported" for parameter in parameters)


def geometry33_output_jacobian(
    model: Any,
    joint_configs: np.ndarray,
    error_vector: np.ndarray,
    parameters: list[ErrorParameter],
    payloads: np.ndarray | float | None = None,
    directions: np.ndarray | None = None,
) -> np.ndarray:
    """Analytic stacked xyz Jacobian for DH/base/tool-translation geometry parameters.

    The transform convention is exactly the current project convention:
    ``T = T_base @ prod_i(Rx(alpha_i) Tx(a_i) Rz(theta_i) Tz(d_i)) @ T_tool``.
    Payloads and directions are accepted for API compatibility; geometry33
    derivatives do not depend on them because rrd/backlash/flex are excluded.
    """
    _ = payloads, directions
    if not supports_geometry33_jacobian(parameters):
        raise ValueError("geometry33_output_jacobian only supports geometry33 parameter names.")

    joints = np.asarray(joint_configs, dtype=float)
    if joints.ndim != 2 or joints.shape[1] != 6:
        raise ValueError("joint_configs must be an N x 6 array.")
    vector = np.asarray(error_vector, dtype=float).reshape(len(parameters))
    comp = vector_to_components(vector, parameters)
    jacobian = np.zeros((joints.shape[0] * 3, len(parameters)), dtype=float)
    for sample_index, q in enumerate(joints):
        sample_jacobian = _single_geometry_jacobian(model, q, comp, parameters)
        jacobian[3 * sample_index : 3 * sample_index + 3, :] = sample_jacobian
    return jacobian


def _single_geometry_jacobian(
    model: Any,
    q: np.ndarray,
    comp: dict[str, np.ndarray],
    parameters: list[ErrorParameter],
) -> np.ndarray:
    nominal = model.nominal
    q_eff = np.asarray(q, dtype=float).reshape(6) + nominal.theta_offset + comp["delta_theta"]
    alpha = nominal.alpha + comp["delta_alpha"]
    a = nominal.a + comp["delta_a"]
    d = nominal.d + comp["delta_d"]

    t_base = nominal.base_xyz + comp["base_xyz"]
    rpy_base = nominal.base_rpy + comp["base_rpy"]
    t_tool = nominal.tool_xyz + comp["tool_xyz"]
    rpy_tool = nominal.tool_rpy + comp["tool_rpy"]

    t_base_matrix = make_transform(t_base, rpy_base)
    t_tool_matrix = make_transform(t_tool, rpy_tool)
    link_matrices = [
        modified_dh_transform(alpha[i], a[i], q_eff[i], d[i])
        for i in range(6)
    ]

    prefixes = [t_base_matrix]
    for link in link_matrices:
        prefixes.append(prefixes[-1] @ link)

    suffixes: list[np.ndarray] = [np.eye(4, dtype=float) for _ in range(6)]
    running = t_tool_matrix
    for i in range(5, -1, -1):
        suffixes[i] = running
        running = link_matrices[i] @ running
    chain_after_base = running
    before_tool = prefixes[6]

    base_derivatives = _transform_derivatives(t_base, rpy_base)
    dh_derivatives = [
        _modified_dh_derivatives(alpha[i], a[i], q_eff[i], d[i])
        for i in range(6)
    ]

    jacobian = np.zeros((3, len(parameters)), dtype=float)
    for column, parameter in enumerate(parameters):
        kind, index = _parameter_kind(parameter.name)
        if kind == "delta_alpha":
            derivative = prefixes[index] @ dh_derivatives[index]["alpha"] @ suffixes[index]
        elif kind == "delta_a":
            derivative = prefixes[index] @ dh_derivatives[index]["a"] @ suffixes[index]
        elif kind == "delta_d":
            derivative = prefixes[index] @ dh_derivatives[index]["d"] @ suffixes[index]
        elif kind == "delta_theta":
            derivative = prefixes[index] @ dh_derivatives[index]["theta"] @ suffixes[index]
        elif kind == "base_translation":
            derivative = base_derivatives[f"t{index}"] @ chain_after_base
        elif kind == "base_rotation":
            derivative = base_derivatives[f"r{index}"] @ chain_after_base
        elif kind == "tool_translation":
            derivative = before_tool @ _translation_derivative(index)
        else:  # pragma: no cover - guarded by supports_geometry33_jacobian.
            raise ValueError(f"Unsupported geometry parameter: {parameter.name}")
        jacobian[:, column] = derivative[:3, 3]
    return jacobian


def _parameter_kind(name: str) -> tuple[str, int]:
    for prefix, kind in (
        ("delta_alpha_", "delta_alpha"),
        ("delta_a_", "delta_a"),
        ("delta_d_", "delta_d"),
        ("delta_theta_", "delta_theta"),
    ):
        if name.startswith(prefix):
            return kind, int(name[len(prefix) :]) - 1
    for prefix, kind in (
        ("delta_Bt", "base_translation"),
        ("delta_Bu", "base_rotation"),
        ("delta_Tt", "tool_translation"),
    ):
        if name.startswith(prefix):
            return kind, _axis_index(name[len(prefix) :])
    return "unsupported", -1


def _axis_index(axis: str) -> int:
    mapping = {"x": 0, "y": 1, "z": 2}
    if axis not in mapping:
        raise ValueError(f"Unknown axis suffix: {axis!r}")
    return mapping[axis]


def _modified_dh_derivatives(
    alpha: float,
    a: float,
    theta: float,
    d: float,
) -> dict[str, np.ndarray]:
    _ = a
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    zeros = np.zeros((4, 4), dtype=float)

    d_alpha = zeros.copy()
    d_alpha[1, 0] = -st * sa
    d_alpha[1, 1] = -ct * sa
    d_alpha[1, 2] = -ca
    d_alpha[1, 3] = -ca * d
    d_alpha[2, 0] = st * ca
    d_alpha[2, 1] = ct * ca
    d_alpha[2, 2] = -sa
    d_alpha[2, 3] = -sa * d

    d_a = zeros.copy()
    d_a[0, 3] = 1.0

    d_d = zeros.copy()
    d_d[1, 3] = -sa
    d_d[2, 3] = ca

    d_theta = zeros.copy()
    d_theta[0, 0] = -st
    d_theta[0, 1] = -ct
    d_theta[1, 0] = ct * ca
    d_theta[1, 1] = -st * ca
    d_theta[2, 0] = ct * sa
    d_theta[2, 1] = -st * sa

    return {"alpha": d_alpha, "a": d_a, "d": d_d, "theta": d_theta}


def _translation_derivative(axis: int) -> np.ndarray:
    derivative = np.zeros((4, 4), dtype=float)
    derivative[axis, 3] = 1.0
    return derivative


def _transform_derivatives(
    xyz: np.ndarray,
    rpy: np.ndarray,
) -> dict[str, np.ndarray]:
    xyz_array = np.asarray(xyz, dtype=float).reshape(3)
    rx, ry, rz = np.asarray(rpy, dtype=float).reshape(3)

    rz_matrix = _rz(rz)
    ry_matrix = _ry(ry)
    rx_matrix = _rx(rx)
    d_rx = rz_matrix @ ry_matrix @ _d_rx(rx)
    d_ry = rz_matrix @ _d_ry(ry) @ rx_matrix
    d_rz = _d_rz(rz) @ ry_matrix @ rx_matrix

    output = {
        "t0": _translation_derivative(0),
        "t1": _translation_derivative(1),
        "t2": _translation_derivative(2),
    }
    for index, d_rotation in enumerate((d_rx, d_ry, d_rz)):
        derivative = np.zeros((4, 4), dtype=float)
        derivative[:3, :3] = d_rotation
        derivative[:3, 3] = 0.0 * xyz_array
        output[f"r{index}"] = derivative
    return output


def _rx(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=float)


def _ry(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)


def _rz(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def _d_rx(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray([[0.0, 0.0, 0.0], [0.0, -s, -c], [0.0, c, -s]], dtype=float)


def _d_ry(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray([[-s, 0.0, c], [0.0, 0.0, 0.0], [-c, 0.0, -s]], dtype=float)


def _d_rz(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray([[-s, -c, 0.0], [c, -s, 0.0], [0.0, 0.0, 0.0]], dtype=float)


