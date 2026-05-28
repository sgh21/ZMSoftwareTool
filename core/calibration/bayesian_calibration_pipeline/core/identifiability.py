"""SVD-based per-parameter identifiability metrics.

The metrics are computed in the same scale-normalized parameter coordinates
used by the regularization experiments: ``z_i = theta_i / scale_i``.  If
``theta = scale * z``, then the stacked output Jacobian with respect to ``z`` is
``J_scaled = J_theta @ diag(scale)``.  This keeps ``rho``, ``kappa`` and the
derived regularization weights comparable across length and angle parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from core.calibration.bayesian_calibration_pipeline.core.parameters import ErrorParameter, parameter_scales


@dataclass
class IdentifiabilityMetrics:
    """Per-parameter metrics from SVD of the scale-normalized Jacobian."""

    parameter_names: list[str]
    scales: np.ndarray
    rank: int
    tolerance: float
    singular_values: np.ndarray
    rho: np.ndarray
    kappa: np.ndarray
    eta: np.ndarray
    raw_risk: np.ndarray
    risk: np.ndarray
    base_weights: np.ndarray
    unidentifiable_mask: np.ndarray
    rho_threshold: float
    kappa_threshold: float
    risk_beta: float
    risk_power: float
    min_weight: float
    max_weight: float
    scaled_jacobian: bool = True


def compute_identifiability_metrics(
    jacobian: np.ndarray,
    parameters: list[ErrorParameter],
    *,
    tolerance: float = 1.0e-7,
    rank: int | None = None,
    rho_threshold: float = 0.5,
    kappa_threshold: float = 0.05,
    risk_beta: float = 0.5,
    risk_power: float = 1.0,
    min_weight: float = 1.0,
    max_weight: float = 100.0,
    scaled_jacobian: bool = True,
) -> IdentifiabilityMetrics:
    """Compute ``rho``, ``kappa`` and L2 weight coefficients for each parameter.

    ``rho_j`` is the energy of parameter axis ``e_j`` inside the identifiable
    right-singular subspace. ``kappa_j`` additionally discounts projections
    onto weak singular directions by ``(sigma_i / sigma_1)^2``. ``raw_risk_j``
    mixes nullspace risk and weak-direction risk. ``risk_j`` applies a power-law
    transform to ``raw_risk_j`` before mapping it to the base L2 weight.
    """
    raw_jacobian = np.asarray(jacobian, dtype=float)
    if raw_jacobian.ndim != 2:
        raise ValueError("jacobian must be a 2-D array")
    parameter_count = len(parameters)
    if raw_jacobian.shape[1] != parameter_count:
        raise ValueError(
            f"jacobian has {raw_jacobian.shape[1]} columns, expected {parameter_count}"
        )

    scales = parameter_scales(parameters)
    analysis_jacobian = raw_jacobian * scales.reshape(1, -1) if scaled_jacobian else raw_jacobian
    _, singular_values, vh = np.linalg.svd(analysis_jacobian, full_matrices=True)
    if singular_values.size == 0 or singular_values[0] <= 1.0e-30:
        computed_rank = 0
    else:
        computed_rank = int(np.sum(singular_values > float(tolerance) * singular_values[0]))
    used_rank = computed_rank if rank is None else max(0, min(int(rank), parameter_count))

    v = _right_singular_vectors(vh, parameter_count)
    identifiable = v[:, :used_rank] if used_rank > 0 else np.zeros((parameter_count, 0))
    rho = np.sum(identifiable * identifiable, axis=1) if used_rank > 0 else np.zeros(parameter_count)

    kappa = np.zeros(parameter_count, dtype=float)
    if used_rank > 0 and singular_values.size > 0 and singular_values[0] > 1.0e-30:
        relative_strength = np.square(singular_values[:used_rank] / singular_values[0])
        kappa = np.sum(np.square(identifiable) * relative_strength.reshape(1, -1), axis=1)

    rho = np.clip(rho, 0.0, 1.0)
    kappa = np.clip(kappa, 0.0, rho)
    eta = np.clip(1.0 - rho, 0.0, 1.0)
    beta = float(np.clip(risk_beta, 0.0, 1.0))
    power = float(risk_power)
    if not np.isfinite(power) or power < 0.0:
        raise ValueError("risk_power must be finite and non-negative")
    raw_risk = np.clip(beta * eta + (1.0 - beta) * (1.0 - kappa), 0.0, 1.0)
    risk = np.power(raw_risk, power)
    low = float(min_weight)
    high = max(float(max_weight), low)
    base_weights = low + (high - low) * risk
    unidentifiable = (rho < float(rho_threshold)) | (kappa < float(kappa_threshold))

    return IdentifiabilityMetrics(
        parameter_names=[parameter.name for parameter in parameters],
        scales=scales,
        rank=used_rank,
        tolerance=float(tolerance),
        singular_values=np.asarray(singular_values, dtype=float),
        rho=rho.astype(float),
        kappa=kappa.astype(float),
        eta=eta.astype(float),
        raw_risk=raw_risk.astype(float),
        risk=risk.astype(float),
        base_weights=base_weights.astype(float),
        unidentifiable_mask=unidentifiable.astype(bool),
        rho_threshold=float(rho_threshold),
        kappa_threshold=float(kappa_threshold),
        risk_beta=beta,
        risk_power=power,
        min_weight=low,
        max_weight=high,
        scaled_jacobian=bool(scaled_jacobian),
    )


def strategy_weights(
    metrics: IdentifiabilityMetrics,
    *,
    active_indices: Iterable[int],
    strong: bool = False,
    strong_weight: float = 10_000.0,
) -> np.ndarray:
    """Return full-length L2 weights for a strategy.

    Parameters outside ``active_indices`` are still assigned their base weights
    for reporting, but only active entries affect the optimizer.  With
    ``strong=True``, entries marked unidentifiable within the active set are
    raised to at least ``strong_weight``.
    """
    weights = np.asarray(metrics.base_weights, dtype=float).copy()
    if strong:
        active = np.asarray(list(active_indices), dtype=int)
        if active.size:
            mask = np.asarray(metrics.unidentifiable_mask, dtype=bool)[active]
            weights[active[mask]] = np.maximum(weights[active[mask]], float(strong_weight))
    return weights


def metrics_table(
    metrics: IdentifiabilityMetrics,
    *,
    active_indices: Iterable[int],
    weights: np.ndarray | None = None,
    pruned_indices: Iterable[int] = (),
) -> list[dict[str, object]]:
    """Return a JSON/HTML friendly per-parameter metrics table."""
    active_set = {int(index) for index in active_indices}
    pruned_set = {int(index) for index in pruned_indices}
    used_weights = metrics.base_weights if weights is None else np.asarray(weights, dtype=float)
    rows: list[dict[str, object]] = []
    for index, name in enumerate(metrics.parameter_names):
        rows.append(
            {
                "index": int(index),
                "parameter": name,
                "scale": float(metrics.scales[index]),
                "svd_active": bool(index in active_set),
                "rho": float(metrics.rho[index]),
                "kappa": float(metrics.kappa[index]),
                "eta": float(metrics.eta[index]),
                "raw_risk": float(metrics.raw_risk[index]),
                "risk": float(metrics.risk[index]),
                "weight": float(used_weights[index]),
                "unidentifiable": bool(metrics.unidentifiable_mask[index]),
                "pruned": bool(index in pruned_set),
            }
        )
    return rows


def _right_singular_vectors(vh: np.ndarray, parameter_count: int) -> np.ndarray:
    """Return a p x p right-singular-vector matrix, padding only if necessary."""
    if vh.shape == (parameter_count, parameter_count):
        return vh.T
    v = np.zeros((parameter_count, parameter_count), dtype=float)
    rows = min(vh.shape[0], parameter_count)
    v[:, :rows] = vh[:rows, :parameter_count].T
    if rows < parameter_count:
        v[:, rows:] = np.eye(parameter_count)[:, rows:]
    return v


