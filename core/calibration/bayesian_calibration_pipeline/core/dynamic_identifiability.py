"""Dynamic and subspace-local identifiability weighting for geometry LM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Any

import numpy as np

from core.calibration.bayesian_calibration_pipeline.core.identifiability import (
    IdentifiabilityMetrics,
    compute_identifiability_metrics,
)
from core.calibration.bayesian_calibration_pipeline.core.parameters import ErrorParameter, parameter_scales
from core.calibration.bayesian_calibration_pipeline.core.redundancy import output_jacobian
from core.calibration.bayesian_calibration_pipeline.core.regularization import RegularizedLMResult, fit_l2_lm
from core.calibration.bayesian_calibration_pipeline.core.robot_model import MultiSourceRobotModel


@dataclass
class PoseIdentifiabilityMetrics:
    """Per-pose identifiability metrics for a fixed parameter vector."""

    parameter_names: list[str]
    scales: np.ndarray
    rho: np.ndarray
    kappa: np.ndarray
    eta: np.ndarray
    raw_risk: np.ndarray
    risk: np.ndarray
    base_weights: np.ndarray
    ranks: np.ndarray
    feature_matrix: np.ndarray
    risk_quantile: float
    aggregated_risk: np.ndarray
    aggregated_weights: np.ndarray


@dataclass
class DynamicIdentifiabilityFit:
    """Result of dynamic identifiability-weighted L2 fitting."""

    result: RegularizedLMResult
    final_weights: np.ndarray
    final_pose_metrics: PoseIdentifiabilityMetrics
    iterations: list[dict[str, float]]


@dataclass
class SubspaceIdentifiabilityPartition:
    """Identifiability-feature subspace partition and local diagnostics."""

    labels: np.ndarray
    K: int
    cluster_sizes: list[int]
    candidate_scores: list[dict[str, float | int | bool]]
    features: np.ndarray
    centers: np.ndarray
    order: list[int]
    subspace_metrics: list[IdentifiabilityMetrics]
    subspace_weights: list[np.ndarray]
    subspace_summaries: list[dict[str, float | int]]


@dataclass
class SubspaceSequentialFit:
    """Result of sequential subspace identifiability-weighted L2 fitting."""

    result: RegularizedLMResult
    stage_records: list[dict[str, float | int]]


def compute_pose_identifiability_metrics(
    model: MultiSourceRobotModel,
    joint_configs: np.ndarray,
    parameters: list[ErrorParameter],
    error_vector: np.ndarray,
    payloads: np.ndarray | float | None = None,
    directions: np.ndarray | None = None,
    *,
    tolerance: float = 1.0e-7,
    rho_threshold: float = 0.5,
    kappa_threshold: float = 0.05,
    risk_beta: float = 0.0,
    risk_power: float = 1.0,
    min_weight: float = 1.0,
    max_weight: float = 100.0,
    risk_quantile: float = 0.25,
    jacobian_method: str = "auto",
) -> PoseIdentifiabilityMetrics:
    """Compute local ``rho/kappa/risk/weight`` for each sampled pose.

    The Jacobian is evaluated once for the full pose set, then split into
    ``N`` blocks of shape ``3 x p``.  Each block gives the local observable
    parameter directions around that single robot configuration.
    """
    joints = np.asarray(joint_configs, dtype=float).reshape(-1, 6)
    vector = np.asarray(error_vector, dtype=float).reshape(len(parameters))
    jacobian = output_jacobian(
        model,
        joints,
        vector,
        parameters,
        payloads=payloads,
        directions=directions,
        method=jacobian_method,
    )
    return pose_identifiability_from_jacobian(
        jacobian,
        len(joints),
        parameters,
        tolerance=tolerance,
        rho_threshold=rho_threshold,
        kappa_threshold=kappa_threshold,
        risk_beta=risk_beta,
        risk_power=risk_power,
        min_weight=min_weight,
        max_weight=max_weight,
        risk_quantile=risk_quantile,
    )


def pose_identifiability_from_jacobian(
    jacobian: np.ndarray,
    pose_count: int,
    parameters: list[ErrorParameter],
    *,
    tolerance: float = 1.0e-7,
    rho_threshold: float = 0.5,
    kappa_threshold: float = 0.05,
    risk_beta: float = 0.0,
    risk_power: float = 1.0,
    min_weight: float = 1.0,
    max_weight: float = 100.0,
    risk_quantile: float = 0.25,
) -> PoseIdentifiabilityMetrics:
    """Build pose-local metrics from a stacked xyz Jacobian."""
    p = len(parameters)
    blocks = np.asarray(jacobian, dtype=float).reshape(int(pose_count), 3, p)
    rho = np.zeros((pose_count, p), dtype=float)
    kappa = np.zeros_like(rho)
    eta = np.zeros_like(rho)
    raw_risk = np.zeros_like(rho)
    risk = np.zeros_like(rho)
    weights = np.zeros_like(rho)
    ranks = np.zeros(pose_count, dtype=int)
    for pose_index in range(int(pose_count)):
        metrics = compute_identifiability_metrics(
            blocks[pose_index],
            parameters,
            tolerance=tolerance,
            rank=None,
            rho_threshold=rho_threshold,
            kappa_threshold=kappa_threshold,
            risk_beta=risk_beta,
            risk_power=risk_power,
            min_weight=min_weight,
            max_weight=max_weight,
            scaled_jacobian=True,
        )
        rho[pose_index] = metrics.rho
        kappa[pose_index] = metrics.kappa
        eta[pose_index] = metrics.eta
        raw_risk[pose_index] = metrics.raw_risk
        risk[pose_index] = metrics.risk
        weights[pose_index] = metrics.base_weights
        ranks[pose_index] = int(metrics.rank)

    quantile = float(np.clip(risk_quantile, 0.0, 1.0))
    aggregated_risk = np.quantile(risk, quantile, axis=0)
    low = float(min_weight)
    high = max(float(max_weight), low)
    aggregated_weights = low + (high - low) * np.clip(aggregated_risk, 0.0, 1.0)
    features = _standardize_features(
        np.concatenate([rho, kappa, np.log10(np.maximum(weights, 1.0e-30))], axis=1)
    )
    return PoseIdentifiabilityMetrics(
        parameter_names=[parameter.name for parameter in parameters],
        scales=parameter_scales(parameters),
        rho=rho,
        kappa=kappa,
        eta=eta,
        raw_risk=raw_risk,
        risk=risk,
        base_weights=weights,
        ranks=ranks,
        feature_matrix=features,
        risk_quantile=quantile,
        aggregated_risk=aggregated_risk.astype(float),
        aggregated_weights=aggregated_weights.astype(float),
    )


def fit_dynamic_identifiability_l2(
    model: MultiSourceRobotModel,
    joint_configs: np.ndarray,
    measured_positions: np.ndarray,
    parameters: list[ErrorParameter],
    active_indices: Iterable[int],
    *,
    lambda_value: float,
    payloads: np.ndarray | float | None = None,
    directions: np.ndarray | None = None,
    initial_vector: np.ndarray | None = None,
    max_nfev: int = 80,
    outer_iterations: int = 3,
    convergence_tol: float = 1.0e-3,
    tolerance: float = 1.0e-7,
    rho_threshold: float = 0.5,
    kappa_threshold: float = 0.05,
    risk_beta: float = 0.0,
    risk_power: float = 1.0,
    min_weight: float = 1.0,
    max_weight: float = 100.0,
    risk_quantile: float = 0.25,
    jacobian_method: str = "auto",
    position_noise_std_m: float | None = None,
) -> DynamicIdentifiabilityFit:
    """Run stable outer-loop dynamic identifiability-weighted L2 LM."""
    parameter_count = len(parameters)
    current = (
        np.zeros(parameter_count, dtype=float)
        if initial_vector is None
        else np.asarray(initial_vector, dtype=float).reshape(parameter_count).copy()
    )
    active = [int(index) for index in active_indices]
    last_weights: np.ndarray | None = None
    iterations: list[dict[str, float]] = []
    final_pose_metrics: PoseIdentifiabilityMetrics | None = None
    final_result: RegularizedLMResult | None = None
    for outer in range(max(1, int(outer_iterations))):
        pose_metrics = compute_pose_identifiability_metrics(
            model,
            joint_configs,
            parameters,
            current,
            payloads=payloads,
            directions=directions,
            tolerance=tolerance,
            rho_threshold=rho_threshold,
            kappa_threshold=kappa_threshold,
            risk_beta=risk_beta,
            risk_power=risk_power,
            min_weight=min_weight,
            max_weight=max_weight,
            risk_quantile=risk_quantile,
            jacobian_method=jacobian_method,
        )
        weights = np.asarray(pose_metrics.aggregated_weights, dtype=float)
        previous = current.copy()
        result = fit_l2_lm(
            model,
            joint_configs,
            measured_positions,
            parameters,
            active,
            lambda_value=float(lambda_value),
            payloads=payloads,
            directions=directions,
            initial_vector=current,
            max_nfev=max_nfev,
            norm="l2",
            regularization_weights=weights,
            position_noise_std_m=position_noise_std_m,
        )
        current = result.vector.copy()
        if last_weights is None:
            relative_weight_change = float("inf")
        else:
            relative_weight_change = float(
                np.linalg.norm(weights - last_weights)
                / max(np.linalg.norm(last_weights), 1.0e-30)
            )
        theta_delta = float(np.linalg.norm(current - previous))
        iterations.append(
            {
                "iteration": float(outer + 1),
                "weight_min": float(np.min(weights)),
                "weight_max": float(np.max(weights)),
                "weight_mean": float(np.mean(weights)),
                "relative_weight_change": relative_weight_change,
                "theta_delta_l2": theta_delta,
                "nfev": float(result.nfev),
                "success": float(bool(result.success)),
            }
        )
        final_pose_metrics = pose_metrics
        final_result = result
        if last_weights is not None and relative_weight_change <= float(convergence_tol):
            break
        last_weights = weights
    if final_pose_metrics is None or final_result is None:
        raise RuntimeError("Dynamic identifiability fitting produced no iteration.")
    return DynamicIdentifiabilityFit(
        result=final_result,
        final_weights=np.asarray(final_pose_metrics.aggregated_weights, dtype=float),
        final_pose_metrics=final_pose_metrics,
        iterations=iterations,
    )


def build_identifiability_subspace_partition(
    pose_metrics: PoseIdentifiabilityMetrics,
    stacked_jacobian: np.ndarray,
    parameters: list[ErrorParameter],
    *,
    k_candidates: tuple[int, ...] = (2, 3, 4),
    min_cluster_size: int = 10,
    seed: int = 123,
    tolerance: float = 1.0e-7,
    rho_threshold: float = 0.5,
    kappa_threshold: float = 0.05,
    risk_beta: float = 0.0,
    risk_power: float = 1.0,
    min_weight: float = 1.0,
    max_weight: float = 100.0,
    strong_weight: float = 10_000.0,
) -> SubspaceIdentifiabilityPartition:
    """Cluster poses by local identifiability features and compute local weights."""
    features = np.asarray(pose_metrics.feature_matrix, dtype=float)
    labels, centers, candidate_scores = _select_kmeans_partition(
        features,
        k_candidates=k_candidates,
        min_cluster_size=int(min_cluster_size),
        seed=int(seed),
    )
    labels = _order_labels_by_cluster_risk(labels, pose_metrics.risk)
    K = int(np.max(labels)) + 1
    centers = np.asarray([np.mean(features[labels == k], axis=0) for k in range(K)], dtype=float)
    cluster_sizes = [int(np.sum(labels == k)) for k in range(K)]

    p = len(parameters)
    blocks = np.asarray(stacked_jacobian, dtype=float).reshape(len(labels), 3, p)
    subspace_metrics: list[IdentifiabilityMetrics] = []
    subspace_weights: list[np.ndarray] = []
    summaries: list[dict[str, float | int]] = []
    for subspace in range(K):
        mask = labels == subspace
        local_jacobian = blocks[mask].reshape(-1, p)
        metrics = compute_identifiability_metrics(
            local_jacobian,
            parameters,
            tolerance=tolerance,
            rank=None,
            rho_threshold=rho_threshold,
            kappa_threshold=kappa_threshold,
            risk_beta=risk_beta,
            risk_power=risk_power,
            min_weight=min_weight,
            max_weight=max_weight,
            scaled_jacobian=True,
        )
        weights = np.asarray(metrics.base_weights, dtype=float).copy()
        weights[metrics.unidentifiable_mask] = np.maximum(
            weights[metrics.unidentifiable_mask], float(strong_weight)
        )
        singular_values = np.linalg.svd(local_jacobian, compute_uv=False)
        positive = singular_values[singular_values > 1.0e-15]
        condition = float(positive[0] / positive[-1]) if positive.size else float("inf")
        subspace_metrics.append(metrics)
        subspace_weights.append(weights)
        summaries.append(
            {
                "subspace": int(subspace),
                "sample_count": int(np.sum(mask)),
                "rank": int(metrics.rank),
                "condition_number": condition,
                "mean_risk": float(np.mean(metrics.risk)),
                "weight_min": float(np.min(weights)),
                "weight_max": float(np.max(weights)),
                "unidentifiable_count": int(np.sum(metrics.unidentifiable_mask)),
            }
        )
    order = [int(row["subspace"]) for row in sorted(summaries, key=lambda row: (row["mean_risk"], row["subspace"]))]
    return SubspaceIdentifiabilityPartition(
        labels=labels.astype(int),
        K=K,
        cluster_sizes=cluster_sizes,
        candidate_scores=candidate_scores,
        features=features,
        centers=centers,
        order=order,
        subspace_metrics=subspace_metrics,
        subspace_weights=subspace_weights,
        subspace_summaries=summaries,
    )


def fit_subspace_sequential_l2(
    model: MultiSourceRobotModel,
    joint_configs: np.ndarray,
    measured_positions: np.ndarray,
    parameters: list[ErrorParameter],
    partition: SubspaceIdentifiabilityPartition,
    *,
    lambda_value: float,
    active_indices: Iterable[int],
    payloads: np.ndarray | float | None = None,
    directions: np.ndarray | None = None,
    initial_vector: np.ndarray | None = None,
    max_nfev: int = 80,
    position_noise_std_m: float | None = None,
) -> SubspaceSequentialFit:
    """Run sequential LM over identifiability subspaces."""
    joints = np.asarray(joint_configs, dtype=float).reshape(-1, 6)
    measured = np.asarray(measured_positions, dtype=float).reshape(-1, 3)
    current = (
        np.zeros(len(parameters), dtype=float)
        if initial_vector is None
        else np.asarray(initial_vector, dtype=float).reshape(len(parameters)).copy()
    )
    active = [int(index) for index in active_indices]
    labels = np.asarray(partition.labels, dtype=int)
    stage_records: list[dict[str, float | int]] = []
    last_result: RegularizedLMResult | None = None
    for stage_index, subspace in enumerate(partition.order):
        indices = np.flatnonzero(labels == int(subspace))
        result = fit_l2_lm(
            model,
            joints[indices],
            measured[indices],
            parameters,
            active,
            lambda_value=float(lambda_value),
            payloads=_subset_optional(payloads, indices, len(joints)),
            directions=_subset_optional(directions, indices, len(joints)),
            initial_vector=current,
            max_nfev=max_nfev,
            norm="l2",
            regularization_weights=partition.subspace_weights[int(subspace)],
            position_noise_std_m=position_noise_std_m,
        )
        current = result.vector.copy()
        last_result = result
        stage_records.append(
            {
                "stage": int(stage_index + 1),
                "subspace": int(subspace),
                "sample_count": int(indices.size),
                "lambda": float(lambda_value),
                "rmse_component_m": float(result.component_rmse),
                "nfev": int(result.nfev),
                "weight_min": float(np.min(partition.subspace_weights[int(subspace)])),
                "weight_max": float(np.max(partition.subspace_weights[int(subspace)])),
            }
        )
    if last_result is None:
        last_result = fit_l2_lm(
            model,
            joints,
            measured,
            parameters,
            active,
            lambda_value=0.0,
            payloads=payloads,
            directions=directions,
            initial_vector=current,
            max_nfev=1,
            position_noise_std_m=position_noise_std_m,
        )
    return SubspaceSequentialFit(result=last_result, stage_records=stage_records)


def _select_kmeans_partition(
    features: np.ndarray,
    *,
    k_candidates: tuple[int, ...],
    min_cluster_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float | int | bool]]]:
    candidates = sorted({int(k) for k in k_candidates if 2 <= int(k) <= len(features)})
    if not candidates:
        raise ValueError("At least one valid K candidate is required.")
    best: tuple[np.ndarray, np.ndarray, float] | None = None
    scores: list[dict[str, float | int | bool]] = []
    for offset, K in enumerate(candidates):
        labels, centers, inertia = _kmeans(features, K, seed + 1009 * offset)
        sizes = [int(np.sum(labels == k)) for k in range(K)]
        silhouette = _silhouette_score(features, labels)
        balance = _balance_score(sizes)
        valid = min(sizes) >= int(min_cluster_size)
        score = 0.75 * silhouette + 0.25 * balance - (0.75 if not valid else 0.0)
        scores.append(
            {
                "K": int(K),
                "score": float(score),
                "silhouette": float(silhouette),
                "balance": float(balance),
                "min_cluster_size": int(min(sizes)),
                "inertia": float(inertia),
                "valid": bool(valid),
            }
        )
        if best is None or score > best[2]:
            best = (labels, centers, float(score))
    if best is None:
        raise RuntimeError("K-means partition did not produce a valid result.")
    return best[0], best[1], scores


def _kmeans(
    features: np.ndarray,
    K: int,
    seed: int,
    *,
    n_init: int = 8,
    max_iter: int = 100,
) -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    best_labels: np.ndarray | None = None
    best_centers: np.ndarray | None = None
    best_inertia = float("inf")
    for _ in range(n_init):
        centers = _kmeans_plus_plus(features, K, rng)
        labels = np.zeros(len(features), dtype=int)
        for _iteration in range(max_iter):
            distances = _squared_distances(features, centers)
            next_labels = np.argmin(distances, axis=1).astype(int)
            centers = _repair_empty_clusters(features, next_labels, centers, rng)
            next_centers = np.asarray(
                [np.mean(features[next_labels == k], axis=0) for k in range(K)], dtype=float
            )
            if np.array_equal(labels, next_labels):
                centers = next_centers
                break
            labels = next_labels
            centers = next_centers
        inertia = float(np.sum(np.min(_squared_distances(features, centers), axis=1)))
        if inertia < best_inertia:
            best_labels = labels.copy()
            best_centers = centers.copy()
            best_inertia = inertia
    if best_labels is None or best_centers is None:
        raise RuntimeError("K-means failed.")
    return best_labels, best_centers, best_inertia


def _kmeans_plus_plus(
    features: np.ndarray,
    K: int,
    rng: np.random.Generator,
) -> np.ndarray:
    centers = [features[int(rng.integers(0, len(features)))]]
    distances = np.full(len(features), np.inf, dtype=float)
    for _ in range(1, K):
        distances = np.minimum(distances, np.sum(np.square(features - centers[-1]), axis=1))
        total = float(np.sum(distances))
        if total <= 1.0e-30:
            centers.append(features[int(rng.integers(0, len(features)))])
        else:
            centers.append(features[int(rng.choice(len(features), p=distances / total))])
    return np.asarray(centers, dtype=float)


def _repair_empty_clusters(
    features: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    repaired = centers.copy()
    counts = np.bincount(labels, minlength=len(centers))
    if np.all(counts > 0):
        return repaired
    distances = np.min(_squared_distances(features, centers), axis=1)
    for cluster in np.where(counts == 0)[0]:
        index = int(np.argmax(distances)) if float(np.max(distances)) > 0.0 else int(rng.integers(0, len(features)))
        repaired[int(cluster)] = features[index]
    return repaired


def _squared_distances(features: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return np.sum(np.square(features[:, None, :] - centers[None, :, :]), axis=2)


def _silhouette_score(features: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int)
    unique = np.unique(labels)
    if unique.size < 2 or len(features) <= unique.size:
        return 0.0
    distances = np.sqrt(np.maximum(_pairwise_squared_distances(features), 0.0))
    scores = []
    for index in range(len(features)):
        same = labels == labels[index]
        same[index] = False
        a = float(np.mean(distances[index, same])) if np.any(same) else 0.0
        b = float("inf")
        for cluster in unique:
            if cluster == labels[index]:
                continue
            mask = labels == cluster
            if np.any(mask):
                b = min(b, float(np.mean(distances[index, mask])))
        denom = max(a, b, 1.0e-30)
        scores.append((b - a) / denom)
    return float(np.mean(scores))


def _pairwise_squared_distances(features: np.ndarray) -> np.ndarray:
    gram = features @ features.T
    sq = np.sum(features * features, axis=1)
    return np.maximum(sq[:, None] + sq[None, :] - 2.0 * gram, 0.0)


def _balance_score(sizes: list[int]) -> float:
    values = np.asarray(sizes, dtype=float)
    if values.size == 0 or float(np.max(values)) <= 0.0:
        return 0.0
    return float(np.min(values) / np.max(values))


def _order_labels_by_cluster_risk(labels: np.ndarray, risk: np.ndarray) -> np.ndarray:
    means = []
    for label in sorted(np.unique(labels)):
        means.append((float(np.mean(risk[labels == label])), int(label)))
    mapping = {old: new for new, (_, old) in enumerate(sorted(means))}
    return np.asarray([mapping[int(label)] for label in labels], dtype=int)


def _standardize_features(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=float)
    center = np.mean(values, axis=0)
    scale = np.std(values, axis=0)
    return (values - center) / np.maximum(scale, 1.0e-12)


def _subset_optional(
    values: np.ndarray | float | None,
    indices: np.ndarray,
    sample_count: int,
) -> np.ndarray | float | None:
    if values is None:
        return None
    array = np.asarray(values, dtype=float)
    if array.ndim == 0 or array.size == 1:
        return float(array.reshape(-1)[0])
    if array.shape[0] != int(sample_count):
        return values
    return array[np.asarray(indices, dtype=int)]


