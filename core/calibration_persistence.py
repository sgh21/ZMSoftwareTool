"""Persistence helpers for identified robot error-parameter models."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def save_identification_result(
    output_path: str | Path,
    parameter_values: dict[str, float],
    *,
    nominal_robot: dict[str, Any],
    identified_robot: dict[str, Any],
    fit_rmse_mm: float,
    fit_max_error_mm: float,
    position_error_rmse_mm: float,
    position_error_max_mm: float,
    sample_count: int,
    confidence: float = 100.0,
    method: str = "S1",
    selected_lambda: float = 0.0,
    dataset_paths: list[str] | None = None,
    cv_scores: list[dict[str, Any]] | None = None,
    subspace_summary: dict[str, Any] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    """Save identified error parameters, full models, and run metadata as YAML."""
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).isoformat()
    identification: dict[str, Any] = {
        "timestamp": timestamp,
        "method": str(method),
        "confidence": float(confidence),
        "sample_count": int(sample_count),
        "selected_lambda": float(selected_lambda),
        "dataset_paths": list(dataset_paths or []),
        "metrics": {
            "fit_rmse_mm": float(fit_rmse_mm),
            "fit_max_error_mm": float(fit_max_error_mm),
            "position_error_rmse_mm": float(position_error_rmse_mm),
            "position_error_max_mm": float(position_error_max_mm),
            # Backward-compatible metric aliases.
            "rmse_mm": float(fit_rmse_mm),
            "max_error_mm": float(fit_max_error_mm),
        },
        "error_parameters": _serialize_parameters(parameter_values),
    }
    identification["nominal_robot"] = _sanitize_metadata(nominal_robot)
    identification["identified_robot"] = _sanitize_metadata(identified_robot)
    if cv_scores:
        identification["cv_scores"] = _sanitize_metadata({"rows": cv_scores})["rows"]
    if subspace_summary:
        identification["subspace_summary"] = _sanitize_metadata(subspace_summary)
    if extra_metadata:
        identification["metadata"] = _sanitize_metadata(extra_metadata)

    document = {"identification": identification}
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(document, fh, allow_unicode=True, default_flow_style=False, sort_keys=False)
    return path


def load_identification_result(path: str | Path) -> dict[str, Any]:
    """Load a previously saved identified parameter YAML file."""
    file_path = Path(path).resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"Identification file not found: {file_path}")

    data = yaml.safe_load(file_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Identification file must contain a mapping: {file_path}")

    section = data.get("identification") or data.get("calibration") or data
    if not isinstance(section, dict):
        raise TypeError(f"Identification section must be a mapping: {file_path}")
    metrics = section.get("metrics", {}) if isinstance(section.get("metrics", {}), dict) else {}
    nominal_robot = _required_mapping(section.get("nominal_robot"), "nominal_robot", file_path)
    identified_robot = _required_mapping(
        section.get("identified_robot"),
        "identified_robot",
        file_path,
    )

    return {
        "timestamp": str(section.get("timestamp", "")),
        "method": str(section.get("method", "S1")),
        "confidence": float(section.get("confidence", 100.0)),
        "sample_count": int(section.get("sample_count", 0)),
        "selected_lambda": float(section.get("selected_lambda", 0.0)),
        "dataset_paths": list(section.get("dataset_paths", [])),
        "fit_rmse_mm": float(metrics.get("fit_rmse_mm", metrics.get("rmse_mm", 0.0))),
        "fit_max_error_mm": float(
            metrics.get("fit_max_error_mm", metrics.get("max_error_mm", 0.0))
        ),
        "position_error_rmse_mm": float(metrics.get("position_error_rmse_mm", 0.0)),
        "position_error_max_mm": float(metrics.get("position_error_max_mm", 0.0)),
        "rmse_mm": float(metrics.get("rmse_mm", metrics.get("fit_rmse_mm", 0.0))),
        "max_error_mm": float(metrics.get("max_error_mm", metrics.get("fit_max_error_mm", 0.0))),
        "error_parameters": dict(section.get("error_parameters", {})),
        "nominal_robot": nominal_robot,
        "identified_robot": identified_robot,
        "cv_scores": list(section.get("cv_scores", [])),
        "subspace_summary": dict(section.get("subspace_summary", {})),
        "metadata": dict(section.get("metadata", {})),
    }


def record_identification_history(
    db_path: str | Path,
    *,
    result_yaml_path: str | Path,
    method: str,
    success: bool,
    message: str,
    sample_count: int,
    fit_rmse_mm: float,
    fit_max_error_mm: float,
    position_error_rmse_mm: float,
    position_error_max_mm: float,
    selected_lambda: float,
    confidence: float,
    dataset_paths: list[str] | None = None,
) -> int:
    """Append one identification run to the SQLite history database."""
    path = Path(db_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        _ensure_history_schema(conn)
        cursor = conn.execute(
            """
            INSERT INTO identification_runs (
                timestamp, method, success, message, result_yaml_path,
                dataset_paths_json, sample_count, fit_rmse_mm, fit_max_error_mm,
                position_error_rmse_mm, position_error_max_mm, selected_lambda, confidence
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                str(method),
                int(bool(success)),
                str(message),
                str(Path(result_yaml_path).resolve()),
                json.dumps(list(dataset_paths or []), ensure_ascii=False),
                int(sample_count),
                float(fit_rmse_mm),
                float(fit_max_error_mm),
                float(position_error_rmse_mm),
                float(position_error_max_mm),
                float(selected_lambda),
                float(confidence),
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)
    finally:
        conn.close()


def list_identification_history(db_path: str | Path, limit: int = 20) -> list[dict[str, Any]]:
    """Return recent identification records from SQLite."""
    path = Path(db_path).resolve()
    if not path.exists():
        return []
    conn = sqlite3.connect(path)
    try:
        conn.row_factory = sqlite3.Row
        _ensure_history_schema(conn)
        rows = conn.execute(
            """
            SELECT *
            FROM identification_runs
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
        output: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["dataset_paths"] = json.loads(item.pop("dataset_paths_json") or "[]")
            item["success"] = bool(item["success"])
            output.append(item)
        return output
    finally:
        conn.close()


def save_calibration_result(
    output_path: str | Path,
    parameter_values: dict[str, float],
    rmse_mm: float,
    max_error_mm: float,
    joint_count: int,
    confidence: float = 100.0,
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    """Backward-compatible wrapper for older calibration call sites."""
    return save_identification_result(
        output_path,
        parameter_values,
        fit_rmse_mm=rmse_mm,
        fit_max_error_mm=max_error_mm,
        position_error_rmse_mm=float(extra_metadata.get("position_error_rmse_mm", 0.0))
        if extra_metadata
        else 0.0,
        position_error_max_mm=float(extra_metadata.get("position_error_max_mm", 0.0))
        if extra_metadata
        else 0.0,
        sample_count=joint_count,
        confidence=confidence,
        method=str(extra_metadata.get("method", "S1")) if extra_metadata else "S1",
        nominal_robot=_required_extra_model(extra_metadata, "nominal_robot"),
        identified_robot=_required_extra_model(extra_metadata, "identified_robot"),
        extra_metadata=extra_metadata,
    )


def load_calibration_result(path: str | Path) -> dict[str, Any]:
    """Backward-compatible wrapper for older calibration call sites."""
    return load_identification_result(path)


def _ensure_history_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS identification_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            method TEXT NOT NULL,
            success INTEGER NOT NULL,
            message TEXT NOT NULL,
            result_yaml_path TEXT NOT NULL,
            dataset_paths_json TEXT NOT NULL,
            sample_count INTEGER NOT NULL,
            fit_rmse_mm REAL NOT NULL,
            fit_max_error_mm REAL NOT NULL,
            position_error_rmse_mm REAL NOT NULL,
            position_error_max_mm REAL NOT NULL,
            selected_lambda REAL NOT NULL,
            confidence REAL NOT NULL
        )
        """
    )


def _serialize_parameters(values: dict[str, float]) -> dict[str, float]:
    return {str(key): float(_sanitize_value(val)) for key, val in values.items()}


def _required_mapping(value: Any, key: str, file_path: Path) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(
            f"Identification file must contain a complete {key} mapping: {file_path}"
        )
    return dict(value)


def _required_extra_model(extra_metadata: dict[str, Any] | None, key: str) -> dict[str, Any]:
    if not extra_metadata or not isinstance(extra_metadata.get(key), dict):
        raise ValueError(f"extra_metadata must contain {key} for calibration result saving.")
    return dict(extra_metadata[key])


def _sanitize_value(value: Any) -> float:
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return float(value)
    return float(value)


def _sanitize_metadata(metadata: Any) -> Any:
    if isinstance(metadata, dict):
        return {str(key): _sanitize_metadata(value) for key, value in metadata.items()}
    if isinstance(metadata, list):
        return [_sanitize_metadata(value) for value in metadata]
    if isinstance(metadata, tuple):
        return [_sanitize_metadata(value) for value in metadata]
    if isinstance(metadata, np.ndarray):
        return metadata.tolist()
    if isinstance(metadata, np.floating):
        return float(metadata)
    if isinstance(metadata, np.integer):
        return int(metadata)
    if isinstance(metadata, (str, int, float, bool, type(None))):
        return metadata
    return str(metadata)
