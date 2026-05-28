"""Dataset utilities for real A/B/C Bayesian Calibration Pipeline experiments."""

from __future__ import annotations

from typing import Any

import numpy as np

from core.calibration.bayesian_calibration_pipeline.core.data_io import load_dataset

_PER_SAMPLE_KEYS = (
    "payloads",
    "directions",
    "joint_torques",
    "self_weight_joint_torques",
    "payload_joint_torques",
)

_METADATA_KEYS = (
    "source_dataset",
)


def load_real_pair(real_a: str, real_b: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Load and canonicalize the two real calibration workspaces."""
    return canonical_dataset(load_dataset(real_a)), canonical_dataset(load_dataset(real_b))


def canonical_dataset(dataset: dict[str, Any]) -> dict[str, np.ndarray]:
    """Normalize supported dataset schemas to the fields used by this pipeline."""
    joints = np.asarray(dataset["joints"], dtype=float).reshape(-1, 6)
    measured = np.asarray(dataset["measured_positions"], dtype=float).reshape(-1, 3)
    output: dict[str, np.ndarray] = {
        "joints": joints,
        "measured_positions": measured,
    }
    payloads = dataset.get("payloads", dataset.get("payload", None))
    if payloads is None:
        output["payloads"] = np.zeros(len(joints), dtype=float)
    else:
        values = np.asarray(payloads, dtype=float).reshape(-1)
        output["payloads"] = (
            np.full(len(joints), float(values[0]), dtype=float)
            if values.size == 1
            else values.reshape(len(joints))
        )
    if dataset.get("directions", None) is not None:
        output["directions"] = np.asarray(dataset["directions"], dtype=float).reshape(-1, 6)
    for key in _PER_SAMPLE_KEYS:
        if key in output or key not in dataset:
            continue
        output[key] = np.asarray(dataset[key], dtype=float).reshape(-1, 6)
    for key in _METADATA_KEYS:
        if key in dataset:
            output[key] = dataset[key]
    return output


def split_dataset_for_c(
    dataset: dict[str, np.ndarray],
    fraction: float,
    seed: int,
    label: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Split by the run008/run010-aligned permutation rule."""
    count = len(dataset["joints"])
    if not 0.0 < float(fraction) < 1.0:
        raise ValueError("real_c_fraction must be between 0 and 1")
    c_count = max(1, min(count - 1, int(round(count * float(fraction)))))
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(count)
    c_indices = np.sort(order[:c_count])
    train_indices = np.sort(order[c_count:])
    return (
        subset_dataset(dataset, train_indices, f"{label}_train"),
        subset_dataset(dataset, c_indices, f"{label}_heldout_c"),
    )


def subset_dataset(
    dataset: dict[str, np.ndarray],
    indices: np.ndarray,
    label: str,
) -> dict[str, np.ndarray]:
    """Return a row subset while preserving optional per-sample fields."""
    index_array = np.asarray(indices, dtype=int).reshape(-1)
    output: dict[str, np.ndarray] = {
        "name": np.asarray(label),
        "joints": dataset["joints"][index_array],
        "measured_positions": dataset["measured_positions"][index_array],
    }
    for key in _PER_SAMPLE_KEYS:
        if key not in dataset:
            continue
        values = np.asarray(dataset[key])
        output[key] = values[index_array] if values.shape[:1] == (len(dataset["joints"]),) else values
    for key in _METADATA_KEYS:
        if key in dataset:
            output[key] = dataset[key]
    return output


def head_dataset(dataset: dict[str, np.ndarray], count: int) -> dict[str, np.ndarray]:
    """Return the first rows for smoke runs."""
    return subset_dataset(dataset, np.arange(min(int(count), len(dataset["joints"]))), "quick")


def concat_c_dataset(
    datasets: list[dict[str, np.ndarray]],
    label: str = "real_c_validation",
) -> dict[str, np.ndarray]:
    """Concatenate held-out parts from both real workspaces into C."""
    output: dict[str, np.ndarray] = {
        "name": np.asarray(label),
        "joints": np.concatenate([dataset["joints"] for dataset in datasets], axis=0),
        "measured_positions": np.concatenate(
            [dataset["measured_positions"] for dataset in datasets], axis=0
        ),
    }
    for key in _PER_SAMPLE_KEYS:
        if not all(key in dataset for dataset in datasets):
            continue
        output[key] = np.concatenate(
            [_normalize_per_sample(dataset[key], len(dataset["joints"]), key) for dataset in datasets],
            axis=0,
        )
    for key in _METADATA_KEYS:
        if all(key in dataset for dataset in datasets):
            output[key] = [dataset[key] for dataset in datasets]
    return output


def _normalize_per_sample(values: Any, count: int, key: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if key == "payloads":
        flat = array.reshape(-1)
        return (
            np.full(count, float(flat[0]), dtype=float)
            if flat.size == 1
            else flat.reshape(count)
        )
    return array.reshape(count, 6)


