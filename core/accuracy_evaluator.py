from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AccuracyMetrics:
    rms: float
    max_error: float
    over_tolerance_rate: float


def evaluate_position_errors(errors: np.ndarray, tolerance: float) -> AccuracyMetrics:
    values = np.linalg.norm(np.asarray(errors, dtype=float).reshape(-1, 3), axis=1)
    if values.size == 0:
        return AccuracyMetrics(rms=0.0, max_error=0.0, over_tolerance_rate=0.0)
    rms = float(np.sqrt(np.mean(values**2)))
    max_error = float(np.max(values))
    over_tolerance_rate = float(np.mean(values > float(tolerance)))
    return AccuracyMetrics(rms=rms, max_error=max_error, over_tolerance_rate=over_tolerance_rate)


def confidence_from_uncertainty(
    uncertainty_rmse_mm: float,
    limit_mm: float,
) -> float:
    """Map model uncertainty relative to the accuracy limit to 0-100 confidence."""
    limit = max(float(limit_mm), 1.0e-6)
    uncertainty = max(float(uncertainty_rmse_mm), 0.0)
    confidence = 100.0 * (limit * limit) / (limit * limit + uncertainty * uncertainty)
    return float(max(0.0, min(100.0, confidence)))
