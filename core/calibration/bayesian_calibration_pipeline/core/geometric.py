"""Geometry33 parameter selection utilities.

This module is intentionally small and self-contained.  The mature pipeline
identifies only the 33 geometric candidates: MDH link errors, base-frame
translation/rotation, and tool/target translation.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from core.calibration.bayesian_calibration_pipeline.core.parameters import ErrorParameter, build_error_parameters


GEOMETRIC_PREFIXES = (
    "delta_alpha_",
    "delta_a_",
    "delta_d_",
    "delta_theta_",
    "delta_Bt",
    "delta_Bu",
    "delta_Tt",
)


def select_geometric_parameters(
    parameters: Sequence[ErrorParameter] | None = None,
) -> list[ErrorParameter]:
    """Return the 33 geometric candidates used by the standalone pipeline."""
    source = list(build_error_parameters() if parameters is None else parameters)
    selected = [
        parameter
        for parameter in source
        if parameter.name.startswith(GEOMETRIC_PREFIXES)
    ]
    if len(selected) != 33:
        raise ValueError(f"Expected 33 geometric parameters, got {len(selected)}.")
    return selected


def project_vector_to_parameters(
    source_vector: np.ndarray,
    source_parameters: Sequence[ErrorParameter],
    target_parameters: Sequence[ErrorParameter],
) -> np.ndarray:
    """Project a vector between parameter lists by stable parameter name."""
    values = np.asarray(source_vector, dtype=float).reshape(len(source_parameters))
    by_name = {
        parameter.name: values[index]
        for index, parameter in enumerate(source_parameters)
    }
    missing = [parameter.name for parameter in target_parameters if parameter.name not in by_name]
    if missing:
        raise KeyError(f"Missing source parameters: {missing}")
    return np.asarray([by_name[parameter.name] for parameter in target_parameters], dtype=float)


